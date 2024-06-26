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
        #return 1/(1+math.exp(-x))
        return math.tanh(x)

    def activatep(self,x):
        #return self.activate(x) * (1.0-self.activate(x))
        return (1.0/math.cosh(x))**2

    def activatepp(self,x):
        # a = 2 * math.exp(-2 * x)
        # b = 1 + math.exp(-x)
        # c = b**3
        # d = math.exp(-x)
        # e = b**2
        # return a/c-d/e
        return -2.0 * self.activatep(x) * self.activate(x)


    def clear_dw(self):
        for layer in range(len(self.NN)-1):
            for neuron in range(len(self.NN[layer])):
                self.NN[layer][neuron]['dw'] = [0] * len(self.NN[layer+1])
                self.NN[layer][neuron]['db'] = 0.0

    #get a weight
    def W(self,layer,neuron,forward_neuron):
        return self.NN[layer][neuron]['w'][forward_neuron]

    #decrement a weight by the amount delta
    def decW(self,layer,neuron,forward_neuron,delta):
        self.NN[layer][neuron]['w'][forward_neuron] -= delta
        
    def forward(self,x):
        assert len(x) == len(self.NN[0])
        
        #load up input layer
        input_layer = 0
        for neuron in range(len(x)):
            self.NN[input_layer][neuron]['z'] = x[neuron]
            self.NN[input_layer][neuron]['a'] = self.activate(x[neuron])
            self.NN[input_layer][neuron]['ap'] = self.activatep(x[neuron])

        #propagate from input layer through the entire network
        output_layer = len(self.NN)-1
        for layer in range(len(self.NN)-1):
            forward_layer = layer + 1

            for forward_neuron in range(len(self.NN[forward_layer])):
                z = 0.0
                deriv = 0.0
                for neuron in range(len(self.NN[layer])):
                    #z = z + self.NN[layer][neuron]['w'][forward_neuron] * self.NN[layer][neuron]['a'] + self.NN[layer][neuron]['b']
                    z = z + self.W(layer,neuron,forward_neuron) * self.NN[layer][neuron]['a'] + self.NN[layer][neuron]['b']
                    deriv += self.W(layer,neuron,forward_neuron) * self.activatep(self.NN[layer][neuron]['z'])

                self.NN[forward_layer][forward_neuron]['z'] = z
                self.NN[forward_layer][forward_neuron]['a'] = self.activate(z)
                self.NN[forward_layer][forward_neuron]['ap'] = self.activatep(z) * deriv            

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
                    #w = self.NN[hidden_layer][neuron]['w'][forward_neuron]
                    w = self.W(hidden_layer,neuron,forward_neuron)
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
            for neuron in range(len(self.NN[layer])):
                for forward_neuron in range(len(self.NN[layer+1])):
                    # -= due to "gradient descent." 
                    # self.NN[layer][x]['w'][y] -= self.NN[layer][x]['dw'][y] / data_count
                    self.decW(layer,neuron,forward_neuron,self.NN[layer][neuron]['dw'][forward_neuron] / data_count)
    
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
        wlist = np.random.normal(0.0,0.1,size=forward_layer_weight_count).tolist()
        wacc_list = [0.0] * forward_layer_weight_count
        #wlist = [0.1*(i+1) for i in range(forward_layer_weight_count)]
        return {
                "desc": desc,
                "z": 0.0,        #input
                "a": 0.0,       #activation
                "b": self.init_neuron_bias,       #bias
                "w": wlist,     #weights
                "delta": 0.0,
                "dw": wacc_list,
                "db": 0.0,
                "ap": 0.0,
                "app": 0.0
                }

    def hidden_neuron(self,forward_layer_weight_count,desc):
        #wlist = [random.uniform(-0.5,0.5) for i in range(forward_layer_weight_count)]
        #wlist = [0.1*(i+1) for i in range(forward_layer_weight_count)]
        wlist = np.random.normal(0.0,0.1,size=forward_layer_weight_count).tolist()
        wacc_list = [0.0] * forward_layer_weight_count
        return {
                "desc": desc,
                "z":0,          #input
                "a": 0.0,       #activation
                "b": self.init_neuron_bias,       #bias
                "w": wlist,     #weights
                "delta": 0.0,
                "dw": wacc_list,
                "db": 0.0,
                "ap": 0.0,
                "app": 0.0
                }

    def output_neuron(self,forward_layer_weight_count,desc):
        return {
                "desc": desc,
                "z": 0.0,
                "a": 0.0,
                "b": self.init_neuron_bias,
                "delta": 0.0,
                "dw": 0.0,
                "db": 0.0,
                "ap": 0.0,
                "app": 0.0
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

    def dz_forward_by_dz_backward(self,forward_layer,forward_layer_neuron):
        back_layer = forward_layer - 1
        sum = 0.0
        for back_neuron in range(len(self.NN[back_layer])):
            sum += self.W(back_layer,back_neuron,forward_layer_neuron) * self.activatep(self.NN[back_layer][back_neuron]['z'])
        return sum

    def get_neuron_count(self,layer):
        return len(self.NN[layer])

    def get_z(self,layer,neuron):
        return self.NN[layer][neuron]['z']

    def set_all_weights(self,val):
        c = 0
        for layer in range(len(self.NN)-1):
            forward_layer = layer + 1
            for forward_neuron in range(len(self.NN[forward_layer])):
                for neuron in range(len(self.NN[layer])):
                    self.NN[layer][neuron]['w'][forward_neuron] = val
                    c += 1
        print(f"{c} weights set to {val}")

    def set_all_weights_rnd(self,val):
        c = 0
        for layer in range(len(self.NN)-1):
            forward_layer = layer + 1
            for forward_neuron in range(len(self.NN[forward_layer])):
                for neuron in range(len(self.NN[layer])):
                    self.NN[layer][neuron]['w'][forward_neuron] = random.uniform(-val,val)
                    c += 1
        print(f"{c} weights set to [-{val},{val}]")

    def get_deriv1(self):
        output_layer = len(self.NN)-1
        l_layer = output_layer - 0
        k_layer = output_layer - 1
        j_layer = output_layer - 2
        i_layer = output_layer - 3

        #single input neuron
        i = 0
        zi = self.get_z(i_layer,i)
        #single output neuron
        l = 0
        sumk = 0
        for k in range(self.get_neuron_count(k_layer)):
            sumj = 0
            for j in range(self.get_neuron_count(j_layer)):
                #def W(self,layer,neuron,forward_neuron):
                wjk = self.W(j_layer,j,k)
                wij = self.W(i_layer,i,j)
                zj = self.get_z(j_layer,j)
                sumj +=  wjk * self.activatep(zj) * wij * self.activatep(zi)
            zk = self.get_z(k_layer,k)
            wkl = self.W(k_layer,k,l)
            sumk += wkl * self.activatep(zk) * sumj

        zl = self.get_z(l_layer,l)
        #return sumk
        return self.activatep(zl) * sumk
    
    def neuron_range(self,layer):
        return range(len(self.NN[layer]))

    def get_deriv(self):
        for back_layer in range(len(self.NN)-1):
            forward_layer = back_layer + 1
            for forward_neuron in self.neuron_range(forward_layer): #range(len(self.NN[forward_layer])):
                for back_neuron in self.neuron_range(back_layer):   #range(self.get_neuron_count(back_layer)):
                    #input to first hidden layer has no sum, so handle it separately
                    if back_layer == 0:
                        zi = self.get_z(back_layer,back_neuron)
                        zj = self.get_z(forward_layer,forward_neuron)
                        wij = self.W(back_layer,back_neuron,forward_neuron)
                        self.NN[forward_layer][forward_neuron]['ap'] = self.activatep(zj) * wij * self.activatep(zi)
                        self.NN[forward_layer][forward_neuron]['app'] = self.activatepp(zj) * (wij * self.activatep(zi))**2 + wij * self.activatep(zj) * self.activatepp(zi)
                    else:
                        zk = self.get_z(forward_layer,forward_neuron)
                        sumj = 0
                        sum2j = 0
                        for j in self.neuron_range(back_layer):      #range(self.get_neuron_count(back_layer)):
                            wjk = self.W(back_layer,j,forward_neuron)
                            daj_dzi = self.NN[back_layer][j]['ap']
                            d2aj_dzi2 = self.NN[back_layer][j]['app']
                            sumj += wjk * daj_dzi
                            sum2j += wjk * d2aj_dzi2
                           
                        self.NN[forward_layer][forward_neuron]['ap'] = self.activatep(zk) * sumj
                        self.NN[forward_layer][forward_neuron]['app'] = self.activatepp(zk) * sumj*sumj + self.activatep(zk) * sum2j 
        
        output_layer = len(self.NN)-1
        return {
                    "ap": [self.NN[output_layer][i]['ap'] for i in self.neuron_range(output_layer)],
                    "app": [self.NN[output_layer][i]['app'] for i in self.neuron_range(output_layer)]
        }
