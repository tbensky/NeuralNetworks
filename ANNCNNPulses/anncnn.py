import torch
from torch import optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision.transforms import ToTensor
import json
import numpy as np
from matplotlib import pyplot as plt

class neural_net(nn.Module):
    def __init__(self):
        super(neural_net,self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1,32,5,padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2),
            )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32,64,5,padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2)
            )

        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(7*7*64,1024),
            nn.Dropout(0.5),
            nn.Linear(1024,3)
            )

    
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        return self.fc1(x)


#https://discuss.pytorch.org/t/error-while-running-cnn-for-grayscale-image/598/2
class channel1(nn.Module):
    def __init__(self):
        super(channel1, self).__init__()
        self.conv_layer_count = 2 #32
        self.conv1 = nn.Conv2d(1, self.conv_layer_count, kernel_size=5, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        #self.conv2 = nn.Conv2d(32, 3, kernel_size=5, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=self.conv_layer_count, out_channels=1, kernel_size=5, bias=False)
        self.relu2 = nn.ReLU()

        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8836,1024),
            #nn.Dropout(0.5),
            nn.Linear(1024,3)
            )

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.fc1(x)
        return x

    def get_conv_layer_count(self):
        return self.conv_layer_count

    def run_conv1(self,x):
        return self.conv1(x)

    def get_conv1(self):
        return self.conv1.weight


ann = channel1() #neural_net()
optimizer = optim.SGD(ann.parameters(),lr=0.1)
loss_fn = nn.MSELoss()

with open("pairs.json","r") as f:
    pairs = json.load(f)

#https://stackoverflow.com/questions/41924453/pytorch-how-to-use-dataloaders-for-custom-datasets
#https://stackoverflow.com/questions/44429199/how-to-load-a-list-of-numpy-arrays-to-pytorch-dataset-loader
size = 100
input_list = []
target_list = []
for (the_input,the_target) in pairs:
    #resize sequential bits of input images into sizexsize squares
    temp = np.array(the_input)
    temp = np.resize(temp,(size,size))
    input_list.append(temp)
        
    #no resizing needed for short output labels (3 bit in this case)
    temp = np.array(the_target)
    target_list.append(temp)

#make into np.arrays to avoid torch warning about slowness
inputs_np = np.array(input_list)
targets_np = np.array(target_list)

#get into tensor form. 
#inputs.shape is [N,size,size] (N=3 of samples)
inputs = torch.tensor(inputs_np,dtype=torch.float)

#target.sape is [N,1,3]
targets = torch.tensor(targets_np,dtype=torch.float)

#train = MyDataset(inputs,targets)
train = TensorDataset(inputs,targets)
train_loader = DataLoader(train) 


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
        plt.tight_layout()
        print(f"epoch={epoch},loss={loss_total}")
        for i in range(ann.get_conv_layer_count()):
            plt.subplot(2,1,i+1)
            plt.axis('off')
            w = ann.get_conv1()[i][0]
            w = w.detach().numpy()
            plt.imshow(w, interpolation='nearest',cmap='gray')
        plt.savefig(f"plots/conv_{epoch}.png")
        plt.close()
    
        

    epoch += 1

    if loss_total < 1e-3:
        break

for i in range(5):
    plt.subplot(8,4,i+1)
    plt.axis('off')
    w = ann.get_conv1()[i][0]
    w = w.detach().numpy()
    plt.imshow(w, interpolation='nearest',cmap='gray')
plt.savefig(f"plots/conv_{epoch}.png")
plt.close()
