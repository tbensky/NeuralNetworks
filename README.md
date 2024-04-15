# Introduction

This repo contains work the author did in an effort to learn about artififical neural networks (ANNs) during the early months of 2024. I didn't want to just jump in and start with PyTorch, as I wanted to understand the inner workings and structure of neural networks, including the backpropagation algorithm.

Here's what's here:

* ANNLoops: A dense-layer, neural network written with for-loops. This means a clumsy and slow model, but very instructive. For example, each neuron is treated as an entity (here a Python dictionary) as follows:

```python
{
    "desc": desc,
    "z":0.0,        #input
    "a": 0.0,       #activation
    "b": self.init_neuron_bias,       #bias
    "w": wlist,     #weights
    "delta": 0.0,
    "dw": wacc_list,
    "db": 0.0
}
```

Here, each neuron has the following internals:

* `desc' a description. This is for convenience and is a text string like `I0` meaning the first neuron in the input laer.