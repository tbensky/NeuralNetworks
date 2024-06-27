import torch
from torch import optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision.transforms import ToTensor
import json
import matplotlib.pyplot as plt
import math

class neural_net(nn.Module):
    def __init__(self,input_neuron_count=1,hidden_neuron_count=10,output_neuron_count=1,learning_rate=0.1,bias_rate=0.1,init_neuron_bias=0.01):
        super(neural_net,self).__init__()

        #tanh works best for this
        self.activation = torch.nn.Tanh() 
        
        #6 layers seems about right
        self.layer1 = torch.nn.Linear(input_neuron_count, hidden_neuron_count)
        self.layer2 = torch.nn.Linear(hidden_neuron_count, hidden_neuron_count)
        self.layer3 = torch.nn.Linear(hidden_neuron_count, hidden_neuron_count)
        self.layer4 = torch.nn.Linear(hidden_neuron_count, hidden_neuron_count)
        self.layer5 = torch.nn.Linear(hidden_neuron_count, hidden_neuron_count)
        self.layer6 = torch.nn.Linear(hidden_neuron_count, output_neuron_count)


    def forward(self,x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        x = self.activation(x)
        x = self.layer3(x)
        x = self.activation(x)
        x = self.layer4(x)
        x = self.activation(x)
        x = self.layer5(x)
        x = self.activation(x)
        x = self.layer6(x)

        return x

#fewer hidden neurons make ypp oscillate about exact derivative
ann = neural_net(input_neuron_count=1,hidden_neuron_count=35,output_neuron_count=1)
optimizer = optim.SGD(ann.parameters(),lr=0.05)
loss_fn = nn.MSELoss()


pairs = [
        [[0.01],[0.0099833]],
        [[0.05],[0.0998334]],
        [[0.1],[0.198669]],
        [[0.15],[0.29552]],
        [[0.2],[0.389418]],
        [[0.25],[0.479426]],
        [[0.3],[0.564642]],
        [[0.35],[0.644218]],
        [[0.4],[0.717356]],
        [[0.45],[0.783327]],
        [[0.5],[0.841471]],
        [[0.55],[0.891207]],
        [[0.6],[0.932039]],
        [[0.65],[0.963558]],
        [[0.7],[0.98545]],
        [[0.75],[0.997495]],
        [[0.8],[0.999574]],
        [[0.85],[0.991665]],
        [[0.9],[0.973848]],
        [[0.95],[0.9463]]
]

#https://stackoverflow.com/questions/41924453/pytorch-how-to-use-dataloaders-for-custom-datasets
inputs = []
targets = []
for (input,target) in pairs:
    inputs.append(input)
    targets.append(target)

inputs = torch.tensor(inputs,dtype=torch.float32)
target = torch.tensor(targets,dtype=torch.float32)

train = TensorDataset(inputs, target)
train_loader = DataLoader(train, batch_size=len(pairs), shuffle=True)

epoch = 0
while True:
    loss_total = 0.0
    for (data,target) in train_loader:
        out = ann(data)
        loss = loss_fn(out,target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss_total += loss.item()

    if epoch % 1000 == 0:
        print(f"epoch={epoch},loss={loss_total}")
    epoch += 1

    #need to train down to 1e-5 for ypp to work best
    if loss_total < 1e-5:
        break



x_train = []
y_nn = []
dy_nn = []
dyy_nn = []
for x_raw in inputs:
    x = torch.tensor([x_raw],requires_grad = True)
    y = ann.forward(x)
    x_train.append(x.item())
    y_nn.append(y.item())
    yp = torch.autograd.grad(y,x,create_graph=True,grad_outputs = torch.ones_like(y),allow_unused = True,retain_graph = True)
    ypp = torch.autograd.grad(yp,x,create_graph=True,grad_outputs = torch.ones_like(y),allow_unused = True,retain_graph = True)

    dy_nn.append(yp[0].item())
    dyy_nn.append(ypp[0].item())
  

y_train = []
for y in targets:
    y_train.append(y)


y_cos = list(map(lambda x: 2.0*math.cos(2*x),x_train))
y_dsin =  list(map(lambda x: -4.0*math.sin(2*x),x_train))
plt.plot(x_train,y_train,'o',label='$y_{train}$')
plt.plot(x_train,y_nn,'x',label='$y_{nn}$')
plt.plot(x_train,dy_nn,'o',label="$y_{nn}'$")
plt.plot(x_train,dyy_nn,'o',label="$y_{nn}''$")
plt.plot(x_train,y_cos,label='$2\cos(2x)$')
plt.plot(x_train,y_dsin,label='$-4\sin(2x)$')
plt.xlabel("t")
plt.ylabel("y, y' or y''")
plt.legend()
plt.show()
