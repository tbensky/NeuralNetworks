import csv
import matplotlib.pyplot as plt
import pandas as pd
import math


df = pd.read_csv("results01.csv")
print (df)
plt.plot(df['t'],df['yp'],label='yp')
plt.plot(df['t'],df['ypp'],label='ypp')
plt.plot(df['t'],list(map(lambda x: math.sin(2*x),df['t'])),label='sin(2X)')
plt.plot(df['t'],list(map(lambda x: 2.0*math.cos(2*x),df['t'])),label='2cos(2x)')
plt.plot(df['t'],list(map(lambda x: -4.0*math.sin(2*x),df['t'])),label='-4sin(2x)')
plt.legend()
plt.show()
