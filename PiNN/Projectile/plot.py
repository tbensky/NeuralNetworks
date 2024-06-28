import csv
import matplotlib.pyplot as plt
import pandas as pd
import math

while True:
    df = pd.read_csv("results.csv")
    plt.plot(df['x'],df['y'])
    plt.show()
