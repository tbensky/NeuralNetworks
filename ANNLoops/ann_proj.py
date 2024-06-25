import nn_draw_graphviz
import matplotlib.pyplot as plt
import neural_net
import json
import time
import math



nn = neural_net.neural_net(
            input_neuron_count=1,
            output_neuron_count=4,
            hidden_neuron_count=[10,5],
            learning_rate=0.5,
            init_neuron_bias=0.01
            )

#t,x,y,vx,vy

v0=30
dmax=25
pairs = [
    [[0.15],[1.505320908/dmax,4.024230274/dmax,9.81936043/v0,25.49973351/v0]],
    [[0.3],[2.948769254/dmax,7.661012594/dmax,9.43427925/v0,23.01909416/v0]],
    [[0.5],[4.790417472/dmax,11.95657049/dmax,8.994238571/v0,19.97982246/v0]],
    [[1.0],[9.068584899/dmax,20.24294006/dmax,8.176363485/v0,13.35860991/v0]]
]


x_train = []
y_train = []
for (input,output) in pairs:
    x_train.append(input[0])
    y_train.append(output[0])

#path to dump network state for plotting/animations
EPS = 1e-3
loss_track = {"epoch":[], "L":[]}
start_time = time.time()
update_start = time.time()
epoch = 0
while True:
    nn.clear_dw()
    L = 0

    #send in input/output pairs and track needed weight changes
    for (input,output) in pairs:
        out = nn.forward(input)
        #format of out: out[0]=x, out[1]=y, out[2]=vx, out[3]=vy
        L = L + 0.5 * sum([(out[i] - output[i])**2 for i in range(len(output))])
        v = math.sqrt(out[2]**2+out[3]**2)
        deriv = nn.get_deriv()
        xpp = deriv['app'][0]
        ypp = deriv['app'][1]
        C = nn.W(0,0,0)
        g = 9.8
        L = L + (xpp - C*v*out[2])**2
        L = L + (ypp - (-g - C*v*out[3]))**2
        nn.backward(output)
    
    #now, actually adjust the weights and biases
    nn.adjust_network(len(pairs))

    # if L > 0.1:
    #     nn.json(f'{INFO_PATH}/nn_{epoch:010d}.json')
    
    loss_track['epoch'].append(epoch)
    loss_track['L'].append(L)

    #information output
    if epoch == 0 or epoch % 1000 == 0:
        cur_time = time.time()
        print(f"epoch={epoch:,}, loss={L}, dt={cur_time-update_start} sec")
        update_start = cur_time

    if L < EPS:
        break;
    
    epoch = epoch + 1
    
end_time = time.time()
print(f"Run time={(end_time-start_time)}s or {(end_time-start_time)/60} min.")

#nn_draw_graphviz.draw(NN)


nn.json("nn.json")


