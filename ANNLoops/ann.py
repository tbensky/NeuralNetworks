import nn_draw_graphviz
import matplotlib.pyplot as plt
import neural_net
import json
import time
import math

#for testing matrix version


"""
#testing with common non-random weights
#used for checking a's, z's and delta's between two
#implementations

input = [0.25,0.75,0.10]
output = [1,0,0.5]

nn.set_weights(0,[0.1,0.2])
nn.set_weights(1,[0.1,0.2,0.3])

out = nn.forward(input)
nn.backward(output)
#nn.adjust_network(1)
nn_draw_graphviz.draw(nn.get_nn())
quit()
"""


"""
nn = neural_net.neural_net(
            input_neuron_count=784,
            output_neuron_count=6,
            hidden_neuron_count=[30,20],
            learning_rate=1,
            init_neuron_bias=0.01
            )
"""


def run_test():
    #this outputs agrees with mma exact results pretty well with hidden_neuron_count=[1]
    nn = neural_net.neural_net( 
            input_neuron_count=1,
            output_neuron_count=1,
            #hidden_neuron_count=[40,35,25,20,15],
            hidden_neuron_count=[2,2],
            learning_rate=0.1,
            init_neuron_bias=0.01
            )

    nn.set_all_weights_rnd(1)
    x = [x/10 for x in range(-50,50,1)]
    y = [nn.forward([xval]) for xval in x]
    y = []
    dyp = []
    dypp = []
    for xval in x:
        y.append(nn.forward([xval]))
        dyp.append(nn.get_deriv()['ap'][0])
        dypp.append(nn.get_deriv()['app'][0])
    plt.plot(x,y,label='func')
    plt.plot(x,dyp,label='d1')
    plt.plot(x,dypp,label='d2')
    plt.legend()
    plt.show()

#run_test()
#exit()

nn = neural_net.neural_net(
            input_neuron_count=1,
            output_neuron_count=1,
            hidden_neuron_count=[40,35,25,20,15],
            learning_rate=20,
            init_neuron_bias=0.01
            )


#x,sin(x) pairs
# freq = 0.5
# x_train = [x/20.0 for x in range(20)]
# y_train = list(map(lambda x: math.sin(freq * x),x_train))
# pairs = []
# for p in zip(x_train,y_train):
#     pairs.append([[p[0]],[p[1]]])

# x = [x/10.0 for x in range(-50,50,1)]
# plt.plot(x,[nn.activatepp(xx) for xx in x])
# plt.show()
#exit()

#this is y(x)=sin(2x) data
#y'(x)=2cos(2x)
#y''(x)=-4sin(2x)
pairs = [
        [[0.01],[0.0099833]],
        [[0.05],[0.0998334]],
        [[0.1],[0.198669]],
        [[0.15],[0.29552]],
        [[0.2],[0.389418]],
        [[0.25],[0.479426]],
        [[0.3],[0.564642]],
        [[0.35],[0.644218]],
        [[0.4],[0.717356]],
        [[0.45],[0.783327]],
        [[0.5],[0.841471]],
        [[0.55],[0.891207]],
        [[0.6],[0.932039]],
        [[0.65],[0.963558]],
        [[0.7],[0.98545]],
        [[0.75],[0.997495]],
        [[0.8],[0.999574]],
        [[0.85],[0.991665]],
        [[0.9],[0.973848]],
        [[0.95],[0.9463]]
]

[0.12, 0.27, 0.43, 0.63, 0.73, 0.83, 0.93]

x_train = []
y_train = []
for (input,output) in pairs:
    x_train.append(input[0])
    y_train.append(output[0])

#hardcoded 7 input,output training pairs for testing
# pairs = [
    
#             [[0.10,0.50,0.10,0.25],[0.25,0.75,0.5]],
#             [[1.00,0.00,0.20,0.33],[1,0.35,0.1]],
#             [[1.00,0.50,0.35,0.10],[0.7,0.85,0.8]],
#             [[0.30,0.20,0.85,0.95],[0.5,0.6,0.55]],
#             [[0.70,0.60,0.50,0.85],[0.2,0.9,0.40]],
#             [[0.88,0.20,0.25,0.65],[0.1,0.4,0.1]],
#             [[0.60,0.25,0.15,0.75],[0.5,0.1,0.9]],
# ]

#load training pairs from training_pairs.json file
"""
with open("training_pairs.json","r") as f:
    pairs = json.load(f)
"""

#path to dump network state for plotting/animations
INFO_PATH = "/Users/tom/Desktop/JSON"
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
        L = L + 0.5 * sum([(out[i] - output[i])**2 for i in range(len(output))])
        nn.backward(output)
    
    #now, actually adjust the weights and biases
    nn.adjust_network(len(pairs))

    if L > 0.1:
        nn.json(f'{INFO_PATH}/nn_{epoch:010d}.json')
    
    loss_track['epoch'].append(epoch)
    loss_track['L'].append(L)

    #information output
    if epoch == 0 or epoch % 1000 == 0:
        cur_time = time.time()
        print(f"epoch={epoch:,}, loss={L}, dt={cur_time-update_start} sec")
        update_start = cur_time
    """
        with open(f'{INFO_PATH}/loss_{epoch:010d}.csv', 'w') as f:
            f.write("epoch,loss\n")
            for i in range(len(loss_track['epoch'])):
                f.write(f"{loss_track['epoch'][i]},{loss_track['L'][i]}\n")
    """

    if L < EPS:
        break;
    
    epoch = epoch + 1
    
end_time = time.time()
print(f"Run time={(end_time-start_time)}s or {(end_time-start_time)/60} min.")

#nn_draw_graphviz.draw(NN)


nn.json("nn.json")


# plt.plot(loss_track['epoch'],loss_track['L'])
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.savefig("loss.png",dpi=300)
# plt.show()

y_nn = list(map(lambda x: nn.forward([x]),x_train))
y_cos = list(map(lambda x: 2.0*math.cos(2*x),x_train))
y_dsin =  list(map(lambda x: -4.0*math.sin(2*x),x_train))
plt.plot(x_train,y_train)
plt.plot(x_train,y_nn)
plt.plot(x_train,y_cos)
plt.plot(x_train,y_dsin)

dz = []
dz2 = []
timestr = time.strftime("%Y%m%d-%H%M")
with open(f"results_{timestr}.csv", 'w') as f:
    f.write("f,fp,fpp\n")
    x_train = [0.02, 0.12, 0.27, 0.385, 0.64, 0.735, 0.825, 0.93]
    for x in x_train:
        nn.forward([x])
        deriv = nn.get_deriv()
        f.write(f"{x},{deriv['ap'][0]},{deriv['app'][0]}\n")
        dz.append(deriv['ap'][0])
        dz2.append(deriv['app'][0])
        """
        for i in range(len(loss_track['epoch'])):
            f.write(f"{loss_track['epoch'][i]},{loss_track['L'][i]}\n")
        """

plt.plot(x_train,dz,label='d1')
plt.plot(x_train,dz2,label='d2')
plt.show()

print("NN output on trained (x,y) pairs")
print("{")
for (x,y) in zip(x_train,y_nn):
    print("{" + f"{x},{y[0]}" + "},")
print("}")