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
        self.layer2 = torch.nn.Linear(hidden_neuron_count, output_neuron_count)#hidden_neuron_count)
        #self.layer3 = torch.nn.Linear(hidden_neuron_count, output_neuron_count) #hidden_neuron_count)
        #self.layer4 = torch.nn.Linear(hidden_neuron_count, hidden_neuron_count)
        #self.layer5 = torch.nn.Linear(hidden_neuron_count, hidden_neuron_count)
        #self.layer6 = torch.nn.Linear(hidden_neuron_count, output_neuron_count)


    def forward(self,x):

        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        #x = self.activation(x)
        #x = self.layer3(x)
        # x = self.activation(x)
        # x = self.layer4(x)
        # x = self.activation(x)
        # x = self.layer5(x)
        # x = self.activation(x)
        # x = self.layer6(x)

        return x
    
    def get_weight(self):
        return self.layer1.weight[0].item()

    def compute_ux(self,x_in):
        return torch.autograd.functional.jacobian(self, x_in, create_graph=True)

    def L(self,data,outputs,targets):
        data_loss = torch.mean((outputs-targets)**2)
        #data_loss = torch.sqrt(torch.sum((outputs-targets)**2))

        phys_loss = 0.0
        g = 9.8

        #https://stackoverflow.com/questions/64988010/getting-the-outputs-grad-with-respect-to-the-input
        #https://discuss.pytorch.org/t/first-and-second-derivates-of-the-output-with-respect-to-the-input-inside-a-loss-function/99757
        #torch.tensor([t_raw],requires_grad = True)
        for x_in in [torch.tensor([x],requires_grad=True) for x in [0.25,2.0,6.0,8.0,10.0]]:
            y_out = self.forward(x_in)

            #u_x = torch.autograd.grad(y_out, x_in, grad_outputs=torch.ones_like(y_out), create_graph=True, retain_graph=True)
            #print(u_x)
            #u_xx = torch.autograd.grad(u_x, x_in, grad_outputs=torch.ones_like(u_x[0]), create_graph=True, retain_graph=True)
            #print(u_xx)
            
            
            u_x = self.compute_ux(x_in) #torch.autograd.functional.jacobian(self, x_in, create_graph=True) 
            u_xx = torch.autograd.functional.jacobian(self.compute_ux, x_in,create_graph=True)
        
            vx = y_out[2]
            vy = y_out[3]
            v = torch.sqrt(vx*vx+vy*vy)
         
            C = 0.01 #self.get_weight()

            dx = C * v * vx
            dy = C * v * vy
            phys_loss += (u_xx[0] + dx)**2 + (u_xx[1] + g + dy)**2
      
        phys_loss = torch.sqrt(phys_loss)
        #return data_loss
        #return phys_loss
        #return torch.add(data_loss,phys_loss)
        return data_loss + phys_loss

def dump_results():
    ts = [x/10. for x in range(0,200,1)]
    x_nn = []
    y_nn = []

    with open("results.csv","w") as f:
        f.write("x,y\n")
        for t_raw in ts:
            t = torch.tensor([t_raw],requires_grad = True)
            y = ann.forward(t)
            f.write(f"{y[0].item()},{y[1].item()}\n")

#fewer hidden neurons make ypp oscillate about exact derivative
ann = neural_net(input_neuron_count=1,hidden_neuron_count=50,output_neuron_count=4)
optimizer = optim.SGD(ann.parameters(),lr=0.001)
#loss_fn = nn.MSELoss()


#projecile data with drag
#t,x,y,vx,vy data
pairs = [
    [[0.15],[1.505320908,4.024230274,9.81936043,25.49973351]],
    [[0.3],[2.948769254,7.661012594,9.43427925,23.01909416]],
    [[0.5],[4.790417472,11.95657049,8.994238571,19.97982246]],
    [[1.0],[9.068584899,20.24294006,8.176363485,13.35860991]],
    [[7.0],[34.0088771,0.827816308,2.864402605,-28.55554438]]
]
   


#https://stackoverflow.com/questions/41924453/pytorch-how-to-use-dataloaders-for-custom-datasets
inputs = []
targets = []
for (input,target) in pairs:
    inputs.append(input)
    targets.append(target)

inputs = torch.tensor(inputs,dtype=torch.float32,requires_grad=True)
target = torch.tensor(targets,dtype=torch.float32)

train = TensorDataset(inputs, target)
train_loader = DataLoader(train, batch_size=len(pairs), shuffle=False)

epoch = 0
loss_fn = ann.L
while True:
    loss_total = 0.0
    for (data,target) in train_loader:
        out = ann(data)
        loss = loss_fn(data,out,target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss_total += loss.item()

    if epoch % 1000 == 0:
        print(f"epoch={epoch},loss={loss_total}")
        dump_results()
        
    epoch += 1

    #need to train down to 1e-5 for ypp to work best
    if loss_total < 1e-3:
        break


x_train = []
y_train = []

for out in target:
    x_train.append(out[0])
    y_train.append(out[1])

ts = [x/10. for x in range(0,100,1)]
x_nn = []
y_nn = []
for t_raw in ts:
    t = torch.tensor([t_raw],requires_grad = True)
    y = ann.forward(t)
    x_nn.append(y[0].item())
    y_nn.append(y[1].item())

print(x_nn)
print(y_nn)
plt.plot(x_nn,y_nn,'.')
plt.plot(x_train,y_train,'o')
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.show()
exit()


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
