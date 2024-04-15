import random
import json
import math
import numpy as np

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
        
        self.build_nn()

    def activate(self,x):
        return 1/(1+math.exp(-x))

    def activatep(self,x):
        return self.activate(x) * (1.0-self.activate(x))


    def clear_dw(self):
        for layer in range(len(self.NN)-1):
            for neuron in range(len(self.NN[layer])):
                self.NN[layer][neuron]['dw'] = [0] * len(self.NN[layer+1])
                self.NN[layer][neuron]['db'] = 0.0

    def forward(self,x):
        assert len(x) == len(self.NN[0])
        
        #load up input layer
        input_layer = 0
        for neuron in range(len(x)):
            self.NN[input_layer][neuron]['z'] = x[neuron]
            self.NN[input_layer][neuron]['a'] = self.activate(x[neuron])

        #propagate from input layer through the entire network
        for layer in range(len(self.NN)-1):
            forward_layer = layer + 1
            for forward_neuron in range(len(self.NN[forward_layer])):
                z = 0.0
                for neuron in range(len(self.NN[layer])):
                    z = z + self.NN[layer][neuron]['w'][forward_neuron] * self.NN[layer][neuron]['a'] + self.NN[layer][neuron]['b']
                self.NN[forward_layer][forward_neuron]['z'] = z
                self.NN[forward_layer][forward_neuron]['a'] = self.activate(z)

        return([x['a'] for x in self.NN[forward_layer]])

    def backward(self,y):
        #compute all of the deltas for
        #1) output layer
        output_layer = len(self.NN)-1
        for neuron in range(len(self.NN[output_layer])):
            z = self.NN[output_layer][neuron]['z']
            a = self.NN[output_layer][neuron]['a']
            dL = a - y[neuron]
            self.NN[output_layer][neuron]['delta'] = dL * self.activatep(z)
            self.NN[output_layer][neuron]['db'] += self.eta_b * self.NN[output_layer][neuron]['delta']
    
        #2) hidden layers
        for hidden_layer in range(output_layer-1,-1,-1):
            forward_layer = hidden_layer + 1
            for neuron in range(len(self.NN[hidden_layer])):
                dsum = 0.0
                for forward_neuron in range(len(self.NN[forward_layer])):
                    w = self.NN[hidden_layer][neuron]['w'][forward_neuron]
                    delta_forward_layer = self.NN[forward_layer][forward_neuron]['delta']
                    dsum = dsum + delta_forward_layer * w
                z = self.NN[hidden_layer][neuron]['z']
                self.NN[hidden_layer][neuron]['delta'] = self.activatep(z) * dsum

        #adjust all weights
        for layer in range(output_layer-1,-1,-1):
            for neuron in range(len(self.NN[layer])):
                self.NN[layer][neuron]['db'] += self.eta_b *  self.NN[layer][neuron]['delta']
                a = self.NN[layer][neuron]['a']
                for forward_neuron in range(len(self.NN[layer+1])):
                    delta = self.NN[layer+1][forward_neuron]['delta']
                    #NN[L][x]['w'][y] = NN[L][x]['w'][y] - ETA * delta * a
                    self.NN[layer][neuron]['dw'][forward_neuron] += self.eta * delta * a

    def adjust_weights(self,data_count):
        #adjust all weights
        for layer in range(len(self.NN)-2,-1,-1):
            for x in range(len(self.NN[layer])):
                for y in range(len(self.NN[layer+1])):
                    # -= due to "gradient descent." 
                    self.NN[layer][x]['w'][y] -= self.NN[layer][x]['dw'][y] / data_count
    
    def adjust_biases(self,data_count):
        #adjust all biases
        for layer in range(len(self.NN)):
            for neuron in range(len(self.NN[layer])):
                # -= due to "gradient descent." 
                self.NN[layer][neuron]['b'] -= self.NN[layer][neuron]['db'] / data_count

    def adjust_network(self,data_count):
        self.adjust_weights(data_count)
        self.adjust_biases(data_count)

    def input_neuron(self,forward_layer_weight_count,desc):
        #wlist = [random.uniform(-0.5,0.5) for i in range(forward_layer_weight_count)]
        wlist = np.random.normal(0.0,1,size=forward_layer_weight_count).tolist()
        wacc_list = [0.0] * forward_layer_weight_count
        #wlist = [0.1*(i+1) for i in range(forward_layer_weight_count)]
        return {
                "desc": desc,
                "z":0.0,        #input
                "a": 0.0,       #activation
                "b": self.init_neuron_bias,       #bias
                "w": wlist,     #weights
                "delta": 0.0,
                "dw": wacc_list,
                "db": 0.0
                }

    def hidden_neuron(self,forward_layer_weight_count,desc):
        #wlist = [random.uniform(-0.5,0.5) for i in range(forward_layer_weight_count)]
        #wlist = [0.1*(i+1) for i in range(forward_layer_weight_count)]
        wlist = np.random.normal(0.0,1,size=forward_layer_weight_count).tolist()
        wacc_list = [0.0] * forward_layer_weight_count
        return {
                "desc": desc,
                "z":0,          #input
                "a": 0.0,       #activation
                "b": self.init_neuron_bias,       #bias
                "w": wlist,     #weights
                "delta": 0.0,
                "dw": wacc_list,
                "db": 0.0
                }

    def output_neuron(self,forward_layer_weight_count,desc):
        return {
                "desc": desc,
                "z": 0.0,
                "a": 0.0,
                "b": self.init_neuron_bias,
                "delta": 0.0,
                "dw": 0.0,
                "db": 0.0
                }

    #input_neuron_count=3,output_neuron_count=5,hidden_layer_count=1,hidden_neuron_count=[2],learning_rate=0.1,bias_rate=0.1):

    def build_nn(self):
        #input layer
        layer = []
        for i in range(self.input_neuron_count):
            layer.append(self.input_neuron(self.hidden_neuron_count[0],f"I{i}") )
        self.NN.append(layer)

        #inner layers
        for i in range(self.hidden_layer_count):
            layer = []

            if i < self.hidden_layer_count-1:
                neuron_count = self.hidden_neuron_count[i+1]
            else:
                neuron_count = self.output_neuron_count

            for j in range(self.hidden_neuron_count[i]):
                layer.append(self.hidden_neuron(neuron_count,f"H{i}{j}") )
            self.NN.append(layer)
            
        #output layer
        layer = []
        for i in range(self.output_neuron_count):
            layer.append(self.output_neuron(self.output_neuron_count,f"O{i}") )
        self.NN.append(layer)

    def json(self,file):
        with open(file,"w") as f:
            json.dump(self.NN,f,indent=2)

    def get_weights(self,layer,neuron):
        return self.NN[layer][neuron]['w']

    def get_nn(self):
        return self.NN

    def set_weights(self,layer,weights):
        for neuron in self.NN[layer]:
            neuron['w'] = weights

