import numpy as np
import time
import matplotlib.pyplot as plt
import json

class neural_net:
    def __init__(self,input_neuron_count=3,output_neuron_count=5,hidden_neuron_count=[2],learning_rate=0.1,bias_rate=0.1,init_neuron_bias=0.01):
        self.input_neuron_count = input_neuron_count
        self.output_neuron_count = output_neuron_count
        self.hidden_layer_count = len(hidden_neuron_count)
        self.hidden_neuron_count = hidden_neuron_count
        self.eta = learning_rate
        self.eta_b = bias_rate
        self.init_neuron_bias = init_neuron_bias
        self.NN = []
        self.dw = []
        self.bias = []
    
        #input to first hidden
        m = self.get_rnd_matrix(self.input_neuron_count,hidden_neuron_count[0])
        self.NN.append(m)

        self.dw.append(
                            np.zeros((hidden_neuron_count[0],self.input_neuron_count))
        )

        self.bias.append(
                            np.full(self.input_neuron_count,init_neuron_bias)
        )

        #all hidden layers
        for hidden_layer in range(0,self.hidden_layer_count-1):
            hright = hidden_layer+1
            hleft = hidden_layer
            m = self.get_rnd_matrix(self.hidden_neuron_count[hleft],self.hidden_neuron_count[hright])
            self.NN.append(m)

            self.dw.append(
                            np.zeros((hidden_neuron_count[hright],self.hidden_neuron_count[hleft]))
            )

        #add bias vector for each hidden layer
        for hidden_layer in range(0,self.hidden_layer_count):
            self.bias.append(
                            np.full(self.hidden_neuron_count[hidden_layer],init_neuron_bias)
            )

        #last hidden to output
        last_hidden_layer = self.hidden_layer_count-1
        last_hidden_layer_size = self.hidden_neuron_count[last_hidden_layer]
        m = self.get_rnd_matrix(last_hidden_layer_size,self.output_neuron_count)
        self.NN.append(m)

        self.dw.append(
                            np.zeros((self.output_neuron_count,last_hidden_layer_size))
        )

        self.bias.append(
                            np.full(self.output_neuron_count,init_neuron_bias)
        )

    def clear_dw(self):
        self.zs = []
        self.dbs = []
        for dw in self.dw:
            dw.fill(0)

    def get_rnd_matrix(self,left,right):
        return np.random.normal(0,0.5,size=(right,left))


    """
    #https://stackoverflow.com/questions/32109319/how-to-implement-the-relu-function-in-numpy
    def activation(self,x):
        return x * (x > 0)

    def activationp(self,x):
        return 1. * (x > 0)
    """

    def activation(self,x):
        return 1/(1+np.exp(-x))

    def activationp(self,x):
        return self.activation(x)*(1-self.activation(x))

 
    def forward(self,input):
        self.zs = []
        layer_count = len(self.NN)

        #find z and a for input neurons
        z = np.array(input + self.bias[0])
        a = self.activation(z)
        self.zs.append(z)
      
        for layer in range(0,layer_count):
            #inputs to forward layer
            z = np.dot(self.NN[layer],a) + self.bias[layer+1]  #vector z coming out of later layer-1 into layer
            a = self.activation(z)
            self.zs.append(z)
        return a

    def backward(self,output,target):
        #last vector in zs array is for the output layer
        output_layer = len(self.zs)-1

        #the first goal here is to build up a column of delta values for each layer
        #seed delta list with output layer
        deltas = [
                        self.activationp(self.zs[output_layer]) * 
                        (output - target)
        ]
        #ok, now put it on the zs of the first hidden layer
        hidden_layer = output_layer - 1
        
        #go backward through the network
        for layer in range(hidden_layer,-1,-1):
            neuron_count = len(self.zs[layer])

            #need this because delta vectors are compiled in reverse order compared to zs
            delta_layer = hidden_layer - layer
            
            #new vector for deltas of these neurons
            delta_vector = np.array(
                            [
                                self.activationp(self.zs[layer][neuron]) *  # f'(z)
                                np.dot(                                     # dot product is the sum in the defitions of the deltas
                                        deltas[delta_layer],                # the deltas for the forward layer
                                        self.NN[layer][:,neuron]            # the weight "fan-in" weight column the desintation neuron 
                                    )      
                                for neuron in range(neuron_count)
                            ]
            )

            #add the vector to the ongoing on for the network
            deltas.append(delta_vector)

        deltas.reverse()
    
        """
        #educational way with nested for loops
        for w in range(len(self.NN)):
            ahead = w + 1
            behind = w
            for row in range(0,len(deltas[ahead])):
                for col in range(0,len(deltas[behind])):
                    self.dw[w][row][col] = deltas[ahead][row] * self.activation(zs[behind][col])
            print(f"w[{w}]={self.dw[w]}")
        """

        for w in range(len(self.NN)):
            ahead = w + 1
            behind = w

            as_vector = self.activation(self.zs[behind])
            
            #note see Karpathy, makemore Part 4 around 25:50 the c=a*b
            #these 4 lines (https://www.youtube.com/watch?v=q8SA3rM6ckI&t=1638s)
            #mat_activations = np.broadcast_to(as_vector,(len(deltas[ahead]),len(as_vector)))
            ds_vector = deltas[ahead].reshape((len(deltas[ahead]),1))
            mat_deltas = np.broadcast_to(ds_vector,(len(deltas[ahead]),len(as_vector)))
            #self.dw[w] += np.multiply(mat_activations,mat_deltas)

            #are the same as (this saves the mat_activations = no.broadcast_to(...) step
            self.dw[w] += mat_deltas * as_vector

        self.dbs += deltas


    def adjust_weights(self,data_count):
        for w in range(len(self.NN)):
            self.NN[w] -= self.eta * self.dw[w]/data_count

    def adjust_biases(self,data_count):
        for (bias,db) in zip(self.bias,self.dbs):
            bias -= self.eta_b*db/data_count


    def set_weights(self,layer,weights):
        self.NN[layer] = weights

    def get_weights(self,layer):
        return self.NN[layer]

"""
nn = neural_net(    input_neuron_count=4,
                    output_neuron_count = 3,
                    hidden_neuron_count=[3,4],
                    learning_rate=3
                )

pairs = [
    
            [[0.10,0.50,0.10,0.25],[0.25,0.75,0.5]],
            [[1.00,0.00,0.20,0.33],[1,0.35,0.1]],
            [[1.00,0.50,0.35,0.10],[0.7,0.85,0.8]],
            [[0.30,0.20,0.85,0.95],[0.5,0.6,0.55]],
            [[0.70,0.60,0.50,0.85],[0.2,0.9,0.40]],
            [[0.88,0.20,0.25,0.65],[0.1,0.4,0.1]],
            [[0.60,0.25,0.15,0.75],[0.5,0.1,0.9]],
]
"""


nn = neural_net(    input_neuron_count=784,
                    output_neuron_count = 6,
                    hidden_neuron_count=[40,20],
                    learning_rate=3
                )


# 4,000 epochs, loss=2.15
# 16,200 epochs, loss=0.011
# 21,600 epochs, loss=0.0071
# 22,700 epochs, loss=0.00672
with open("training_pairs.json","r") as f:
    pairs = json.load(f)

test_pair = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0]]
 

"""
nn.set_weights(0,np.array(
                    [
                        [0.1,0.1,0.1],
                        [0.2,0.2,0.2],
                    ]
))

nn.set_weights(1,np.array(
                    [
                        [0.1,0.1],
                        [0.2,0.2],
                        [0.3,0.3]
                    ]
))
"""

start_time = time.time()

loss_track = []
EPS = 1e-3
epoch = 0
while True:
    nn.clear_dw()
    L = 0
    Acc = 0

    #send in input/output pairs and track needed weight changes
    for (input,target) in pairs:
        nn_output = nn.forward(input)
        diff = target - nn_output
        L = L + 0.5 * np.dot(diff,diff).sum()
        nn.backward(np.array(nn_output),target)
    
    #now, actually adjust the weights and biases
    nn.adjust_weights(len(pairs))
    nn.adjust_biases(len(pairs))

    #information output
    if epoch == 0 or epoch % 100 == 0:
        cur_time = time.time()
        dt = cur_time - start_time
        print(f"t={dt:.2f}, epoch={epoch:,}, loss={L}")
        nn_output = nn.forward(test_pair[0])

        diff = test_pair[1] - nn_output
        Acc = Acc + 0.5 * np.dot(diff,diff).sum()
        print(f"Test: Output={nn_output}, Target={test_pair[1]}")
        
        loss_track.append([dt,epoch,L,Acc])

        with open("loss.csv","w") as f:
            f.write("time,epoch,loss,accuracy\n")
            for [dt,epoch,loss,acc] in loss_track:
                f.write(f"{dt},{epoch},{loss},{acc}\n")

    if L < EPS:
        break;
    
    epoch = epoch + 1

end_time = time.time()
print(f"Took {end_time - start_time} seconds, {(end_time - start_time)/60} minutes.")

plt.plot(loss['epoch'],loss['loss'],loss['acc'])
plt.savefig("loss.png")