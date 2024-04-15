# Introduction

This repo contains work the author did in an effort to learn about artififical neural networks (ANNs) during the early months of 2024. I didn't want to just jump in and start with PyTorch, as I wanted to understand the inner workings and structure of neural networks, including the backpropagation algorithm.

Here's what's here:


* `Backprop_derive` The first thing I did was to go through all of the mathematics behind backpropagation.  I started with very first principles using only the structure of a fully interconnected ANN and the chain-rule from Calculus.  

* `ANNLoops`  A dense-layer, neural network written with for-loops. This means a clumsy and slow model, but very instructive. For example, each neuron is treated as an entity (here a Python dictionary) as follows:

```python
{
    "desc": desc,
"z":0.0,                            #input
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

* `a` is the neuron's activation, of $a=f(z)$, where $f$ is the activation function (sigmoid, ReLU, etc).

* `b` is the neuron's bias.

* `w` is a list of weights. Each weight in the list "connects" this neuron to each neuron in the forward layer. So if this neuron is in in layer $N$, and layer $N+1$ has $m$ neurons in it, then this list will have $m$ entries.

