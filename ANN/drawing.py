import matplotlib.pyplot as plt

class drawing:
    def __init__(self):
        fig, self.ax = plt.subplots()
        self.ax.set_xlim([-2,2])
        self.ax.set_ylim([-2,2])
        #self.ax.set_aspect('equal', adjustable='datalim')
        self.xmax = 1.5
        self.xmin = -1.5
        self.ymax = 1
        self.ymin = -1
       
        
    def circle(self,x,y,r):
        cir = plt.Circle((x,y), r, color='r',fill=False)
        self.ax.add_patch(cir)
        
    def line(self,x1,y1,x2,y2,w):
        plt.plot([x1, x2],[y1,y2],color=(w,0,0))
        #plt.axline((0, 0), (1, 1), linewidth=4, color='r')
        
    def draw_nn(self,NN):
        layers = len(NN)
        dx = (self.xmax - self.xmin) / layers
        x = self.xmin
        for layer in range(len(NN)):
            y = self.ymin
            dy = (self.ymax - self.ymin)/len(NN[layer])
            for neuron in range(len(NN[layer])):
                self.circle(x,y,0.1)
                NN[layer][neuron]['pos'] = {"x":x, "y":y}
                y += dy
            x += dx
            
        for layer in range(len(NN)-1):
            for neuron1 in range(len(NN[layer])):
                for neuron2 in range(len(NN[layer+1])):
                    self.line(
                            NN[layer][neuron1]['pos']['x'],NN[layer][neuron1]['pos']['y'],
                            NN[layer+1][neuron2]['pos']['x'],NN[layer+1][neuron2]['pos']['y'],
                            NN[layer][neuron1]['w'][neuron2]
                          )
        plt.show()
        
        
        

NN = [
      [{"w": [1,1,1,1]},{"w":[1,1,1,1]},{"w":[1,1,1,1]}],
      [{"w":[1,1,1]},{"w":[1,1,1]},{"w":[1,1,1]},{"w":[1,1,1]}],
      [{"w":[1,1]},{"w":[1,1]},{"w":[1,1]}],
      [{},{}]
      ]

d = drawing()
d.draw_nn(NN)
