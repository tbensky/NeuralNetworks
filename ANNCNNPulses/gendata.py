import math
import random

HEIGHT = 100
WIDTH = 100
SIZE = (HEIGHT*WIDTH)

screen = ['0'] * SIZE

def cls():
    for row in range(HEIGHT):
        for col in range(WIDTH):
            screen[row*WIDTH+col] = '0'


def plot(x,y,c):
    offset = (HEIGHT/2-y)*WIDTH + (WIDTH/2+x)
    if offset > 0 and offset < SIZE:
        screen[int(offset)] = c

def plotbig(x,y,c):
    #plot(x,y,c)
    
    for dx in range(-1,1):
        for dy in range(-1,1):
            plot(x+dx,y+dy,c)

def dump():
    for row in range(HEIGHT):
        for col in range(WIDTH):
            offset = row*WIDTH + col
            print(screen[offset],end='')
        print()

def dump_pair(output):
    print("[[",end="")
    for i in range(SIZE-1):
        print(f"{screen[i]},",end="")
    print(f"{screen[i]}",end="")
    print(f"],[{output}]",end="")
    print("]",end="")



def gauss(A,x0,sigma):
    x = -HEIGHT/2
    while x < HEIGHT/2:
        y = int(A*math.exp(-((x-x0)**2)/(2*math.pi*sigma**2)))
        plotbig(x,y,'1')
        x += 0.1

def square(A,x0,width,offset):
    x = -WIDTH/2
    while x < WIDTH/2:
        if x <= x0 or x >= (x0 + width):
            y = int(offset)
        else:
            y = int(offset + A)
        plotbig(x,y,'1')
        x += 1

    for y in range(int(offset),int(A + offset)+1):
        plotbig(x0,y,'1')
        plotbig(x0+width,y,'1')
    
    

A = 20 #random.uniform(5,20)
x0 = 0 #random.uniform(-30,30)
sd = random.uniform(1,20)
width = random.uniform(2,25)
offset = 0 # random.uniform(-30,30)

#square(A,x0,width,offset)


print("[") # open json

N = 100
for i in range(0,N):
    cls()
    A = 20 #random.uniform(5,20)
    x0 = 0 #random.uniform(-30,30)
    sd = random.uniform(1,20)
    width = random.uniform(2,25)
    offset = 0 # random.uniform(-30,30)

    n = random.randint(0,1)

    if n == 0:
        gauss(A,x0,sd)
        output = "0,0,1"
    if n == 1:
        square(A,x0,width,offset)
        output = "0,1,0"

    dump_pair(output)
    if i < N-1:
        print(",")


print("\n]") # close json

