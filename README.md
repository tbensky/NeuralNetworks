# Introduction

This repo contains work the author did in an effort to learn about artififical neural networks (ANNs) during the early months of 2024. I didn't want to just jump in and start with PyTorch, as I wanted to understand the inner workings and structure of neural networks, including how the backpropagation algorithm works.

Acryonyms:

* `ANN` = artificial neural network
* `BP` = backpropagation


Note: This readme is really only a short summary of all of the details I captured in [bp.pdf](https://github.com/tbensky/NeuralNetworks/blob/main/Backprop_derive/bp.pdf).

## Repo contents

Here's what you'll find in this repo:

* `Backprop_derive` The first thing I did was to go through all of the mathematics behind backpropagation.  I started with very first principles using only the structure of a fully interconnected ANN and the chain-rule from Calculus.  Look for a file called [bp.pdf](https://github.com/tbensky/NeuralNetworks/blob/main/Backprop_derive/bp.pdf) as the latest LaTex build of what I found. 

To the mathematically inclined, there are a lot of interesting patterns and logic in deriving all of the backpropagation formulas.  It kind of "makes sense" in the end and was a fun mathematical exercise.

* `ANN` This is a kind of messy collection of files developed as I was getting going. Ignore this folder.

# ANNLoops

This is a dense-layer, neural network written with for-loops. This means a clumsy and slow model, but very instructive. For example, each neuron is treated as an entity (here a Python dictionary) as follows:

```python
{
    "desc": desc,
    "z":0.0,                        #input
    "a": 0.0,                       #activation
    "b": self.init_neuron_bias,     #bias
    "w": wlist,                     #weights
    "delta": 0.0,                   #delta (neuron's contributio to the overall error)
    "dw": wacc_list,                #$dw$, or how the neuron's weights into the next layer should change
    "db": 0.0                       #$db$, or how the neuron's bias should change.
}
```

Here, each neuron has the following internals:

* `desc` a description. This is for convenience and is a text string like `I0` meaning the first neuron in the input layer.

* `z` is the neuron's input, which is the weighted sum of all activations from the previous layer, into this neuron.

* `a` is the neuron's activation, of $a=f(z)$, where $f$ is the activation function (sigmoid, ReLU, etc).

* `b` is the neuron's bias.

* `w` is a list of weights. Each weight in the list "connects" this neuron to each neuron in the forward layer. So if this neuron is in in layer $N$, and layer $N+1$ has $m$ neurons in it, then this list will have $m$ entries.

* `delta` is the neuron's contribution to the overall loss of the network. It is found by summing all of the deltas in the forward layer, backward through all of the interconnecting weights into this neuron.

* `dw` is a list the same size as the `w` list. The BP algorithm will fill this `dw` list with how all of the weights in the `w` list should change, in an effort to lower the loss.

* `db` is how the neuron's bias should change, as per the BP algorithm.

So this is the core of the code in `ANNLoops.` When using such discrete structures for the neurons, a lot of for-loops will be needed to run the forward and backward passes. Thus the name `ANNLoops.`

The ANN in this folder may be run by doing a 

```
python3 ann.py
```

It is programmed to train a series of 7 training pairs held in a list of lists as follows:

```python
pairs = [
    
            [[0.10,0.50,0.10,0.25],[0.25,0.75,0.5]],
            [[1.00,0.00,0.20,0.33],[1,0.35,0.1]],
            [[1.00,0.50,0.35,0.10],[0.7,0.85,0.8]],
            [[0.30,0.20,0.85,0.95],[0.5,0.6,0.55]],
            [[0.70,0.60,0.50,0.85],[0.2,0.9,0.40]],
            [[0.88,0.20,0.25,0.65],[0.1,0.4,0.1]],
            [[0.60,0.25,0.15,0.75],[0.5,0.1,0.9]],
]
```

The format is 

```
pairs = [
            [[input1],[output1]],
            [[input2],[output2]],
            ...
            [[inputN],[outputN]]
]
```

You can also uncommment the block of code that loads in training pairs from a file called `training_pairs.json`.  This is a valid ``json`` structure of a list of list of `[input],[output]` training pairs.  The current `training_pairs.json` file contains 100 random digits from the MNIST training set.

## API
You can see that this  model is run using a simple API (in `neural_net.py`') which you can see is

```python
nn = neural_net.neural_net(
            input_neuron_count=4,
            output_neuron_count=3,
            hidden_neuron_count=[3,4],
            learning_rate=2,
            init_neuron_bias=0.01
            )
```

It allows you to configure any arbitray depths and length ANN that you desire.  The `hidden_neuron_count` is a list of the number of neurons you want in each hidden layer.  So `[3,4]` means you want 2 hidden layers, the first with 3 neurons and thee next with 4 neurons.

## Running the ANN

Training the ANN follows some logic, as shown by this code

```python
while True:
    #clear the gradients (dw lists)
    nn.clear_dw()
    L = 0

    #send in input/output pairs and track needed weight changes
    for (input,output) in pairs:
        out = nn.forward(input)
        L = L + 0.5 * sum([(out[i] - output[i])**2 for i in range(len(output))])
        nn.backward(output)
    
    #now, actually adjust the weights and biases
    nn.adjust_network(len(pairs))

    #information output
    if epoch == 0 or epoch % 1000 == 0:
        print(f"epoch={epoch:,}, loss={L}")

    if L < EPS:
        break;
    
    epoch = epoch + 1
```

# ANNMatrices

In this code the ANN is now thought of not in terms of discrete neurons, but in terms of transforming the input vector into an output vector using matrix multiplications and `NumPy`. Thus all for-loops were removes (as much as possible), in favor of matrix multiplications.

This was also a fun mathematical exercise in matrices and vector reshaping, with details all captured in [bp.pdf](https://github.com/tbensky/NeuralNetworks/blob/main/Backprop_derive/bp.pdf). 

The code and API is identical to the `ANNLoops` code.