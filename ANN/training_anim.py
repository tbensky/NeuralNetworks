import matplotlib.pyplot as plt
import json
import glob
import re
import pandas as pd
import random
import statistics

#no trailing / on path
class training_anim:
    def __init__(self,path):
        #self.ax.set_aspect('equal', adjustable='datalim')
        self.xmax = 1.5
        self.xmin = -1.5
        self.ymax = 2
        self.ymin = -2
        self.epoch = 0
        self.path = path
        self.iter = 0;

        #https://stackoverflow.com/questions/3207219/how-do-i-list-all-files-of-a-directory

        files = sorted(glob.glob(f"{path}/*.json"))

        """
        #get max and min weights
        for file in files:
           
            f = open(f"{file}","r")
            self.NN = json.load(f)
            f.close()

            if self.iter == 0:
                self.wmax = self.wmin = self.NN[0][0]['w'][0]
                self.iter += 1;

            for layer in range(len(self.NN)-1):
                for neuron1 in range(len(self.NN[layer])):
                    for neuron2 in range(len(self.NN[layer+1])):
                        w =  self.NN[layer][neuron1]['w'][neuron2]
                        if w > self.wmax:
                            self.wmax = w
                        if w < self.wmin:
                            self.wmin = w
        print(f"wmin={self.wmin}, wmax={self.wmax}")
        """

        for file in files:
            f = open(f"{file}","r")
            self.NN = json.load(f)
            f.close()

            sepoch = self.get_epoch(file)
            #self.df = pd.read_csv(f'{path}/loss_{sepoch}.csv')

            self.draw_nn()
            self.epoch = self.epoch + 1
            print(f"{file}")

    def get_epoch(self,file):
        return re.search("\d{10}",file).group()
        
    def circle(self,x,y,r):
        cir = plt.Circle((x,y), r, color='r',fill=False)
        self.ax.add_patch(cir)
        
    def line(self,x1,y1,x2,y2,w):
        #self.wmin = -0.05
        #self.wmax = 0.05
        c = (w-self.wmin)/(self.wmax - self.wmin)
        if c >= 0 and c <= 1:
            plt.plot([x1, x2],[y1,y2],color=(c,c,c),linewidth=1)
        #plt.axline((0, 0), (1, 1), linewidth=4, color='r')

    def wmaxmin(self):
        wlist = []
        for layer in range(len(self.NN)-1):
            for neuron1 in range(len(self.NN[layer])):
                for neuron2 in range(len(self.NN[layer+1])):
                    w =  self.NN[layer][neuron1]['w'][neuron2]
                    wlist.append(w)

        self.wstd = statistics.stdev(wlist)
        self.wavg = statistics.mean(wlist)
        self.wmax = max(wlist)/self.wavg
        self.wmin = min(wlist)/self.wavg
        
    def draw_nn(self):
        fig, self.ax = plt.subplots()
        self.ax.set_xlim([-2,2])
        self.ax.set_ylim([-2,2])
        
        layers = len(self.NN)
        dx = (self.xmax - self.xmin) / layers
        x = self.xmin
        for layer in range(len(self.NN)):
            y = self.ymin
            dy = (self.ymax - self.ymin)/len(self.NN[layer])
            radius = dy / 10
            for neuron in range(len(self.NN[layer])):
                self.circle(x,y,radius)
                self.NN[layer][neuron]['pos'] = {"x":x, "y":y}
                y += dy
            x += dx

        self.wmaxmin()

        for layer in range(len(self.NN)-1):
            for neuron1 in range(len(self.NN[layer])):
                for neuron2 in range(len(self.NN[layer+1])):
                    self.line(
                            self.NN[layer][neuron1]['pos']['x'],self.NN[layer][neuron1]['pos']['y'],
                            self.NN[layer+1][neuron2]['pos']['x'],self.NN[layer+1][neuron2]['pos']['y'],
                            self.NN[layer][neuron1]['w'][neuron2]/self.wavg
                          )
        # a = plt.axes([.65, .6, .21, .21])
        # plt.plot(self.df['epoch'],self.df['loss'])
        # plt.xlabel("Epoch")
        # plt.ylabel("Loss")
        plt.savefig(f"{self.path}/w_{self.epoch:05d}.png",dpi=100)
        plt.close()
        
        
        

training_anim("/Users/tom/Desktop/JSON")