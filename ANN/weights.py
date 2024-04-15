import matplotlib.pyplot as plt
import json
import glob
import re
import pandas as pd
import random
import statistics

#no trailing / on path


path = "/Users/tom/Desktop/JSON"
files = sorted(glob.glob(f"{path}/*.json"))

c = 0 
for file in files:
    f = open(file,"r")
    n = json.load(f)
    f.close()

    c = c + 1
    if c == 100:
        break

    layer = 0
    neuron = 0
    plt.plot([x for x in range(len(n[layer][neuron]['w']))],n[layer][neuron]['w'])
    plt.title(f"{file}, {neuron}")
    plt.ylim([-0.1,0.1])
    plt.draw()
    plt.pause(0.001)
    print(statistics.mean(n[layer][neuron]['w']),statistics.stdev(n[layer][neuron]['w']))
