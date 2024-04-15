import nn_draw_graphviz
import matplotlib.pyplot as plt
import neural_net
import json
import time

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

nn = neural_net.neural_net(
            input_neuron_count=4,
            output_neuron_count=3,
            hidden_neuron_count=[3,4],
            learning_rate=2,
            init_neuron_bias=0.01
            )


#hardcoded 7 input,output training pairs for testing
pairs = [
    
            [[0.10,0.50,0.10,0.25],[0.25,0.75,0.5]],
            [[1.00,0.00,0.20,0.33],[1,0.35,0.1]],
            [[1.00,0.50,0.35,0.10],[0.7,0.85,0.8]],
            [[0.30,0.20,0.85,0.95],[0.5,0.6,0.55]],
            [[0.70,0.60,0.50,0.85],[0.2,0.9,0.40]],
            [[0.88,0.20,0.25,0.65],[0.1,0.4,0.1]],
            [[0.60,0.25,0.15,0.75],[0.5,0.1,0.9]],
]

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
        print(f"epoch={epoch:,}, loss={L}")
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


plt.plot(loss_track['epoch'],loss_track['L'])
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig("loss.png",dpi=300)
