import nn_draw_graphviz
import matplotlib.pyplot as plt
import neural_net
import json
import time
import math



nn = neural_net.neural_net(
            input_neuron_count=1,
            output_neuron_count=4,
            hidden_neuron_count=[5,25,10],
            learning_rate=0.1,
            init_neuron_bias=0.01
            )

#t,x,y,vx,vy

pairs = [
    [[0.15],[1.505320908,4.024230274,9.81936043,25.49973351]],
    [[0.3],[2.948769254,7.661012594,9.43427925,23.01909416]],
    [[0.5],[4.790417472,11.95657049,8.994238571,19.97982246]],
    [[1.0],[9.068584899,20.24294006,8.176363485,13.35860991]]
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


