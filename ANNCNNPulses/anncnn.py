#
#old version
#doesn't use pytorch dataloaders
#

import torch
from torch import optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision.transforms import ToTensor
import json
import numpy as np
from matplotlib import pyplot as plt
from torchvision import utils
import os
import torch.nn.functional as F
import time

#https://discuss.pytorch.org/t/error-while-running-cnn-for-grayscale-image/598/2
class neural_net(nn.Module):
    def __init__(self):
        super(neural_net, self).__init__()
        
        self.conv_layer_count = 50
        self.K = 20

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=self.conv_layer_count, kernel_size=self.K, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        self.mp1 =  nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(in_channels=self.conv_layer_count, out_channels=1, kernel_size=self.K)
        self.relu2 = nn.ReLU()
        self.act = nn.Tanh()
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(10*10, 1024)
        self.fc2 = nn.Linear(1024, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)

        x = self.mp1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.mp1(x)
    
        #K=10, size=18x18
        #K=25, size=7x7
        #K=20, 10x10
        #print(x.size()) 
        #exit()
       
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)

        #x = F.normalize(x)
        return x

    def get_conv_layer_count(self):
        return self.conv_layer_count

    def run_conv1(self,x):
        return self.conv1(x)

    def get_conv1(self):
        return self.conv1.weight

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if device == torch.device("cpu") and torch.backends.mps.is_available():
    device = torch.device("mps")


ann = neural_net()
ann.to(device)



########################
## learning rate here ##
########################

#lr=2 or 1.5 reveals interesting features, but loss=nan
#plan: try lr between 1.5 and 15

#Seq01: first working one: lr=0.005, momentum=1.0, dropout=0.25, normalize output, K=20, conv_layer=50, CrossEntropyLoss

optimizer = optim.SGD(ann.parameters(),lr=0.01,momentum=1.0)


#CrossEntropyLoss reveals curved sections
loss_fn = nn.CrossEntropyLoss() 

#MSELoss reveals straight sections
#loss_fn = nn.MSELoss()
#loss_fn = nn.BCEWithLogitsLoss()

with open("pairs.json","r") as f:
    pairs = json.load(f)

#https://stackoverflow.com/questions/41924453/pytorch-how-to-use-dataloaders-for-custom-datasets
#https://stackoverflow.com/questions/44429199/how-to-load-a-list-of-numpy-arrays-to-pytorch-dataset-loader
#https://stackoverflow.com/questions/72808402/pytorch-identifying-batch-size-as-number-of-channels-in-conv2d-layer

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

#view inputs if needed
# for i in range(len(inputs_np)):
#     plt.imshow(inputs_np[i])
#     plt.show()
# exit()

#get into tensor form. 
#inputs.shape is [N,size,size] (N=3 of samples)
inputs = torch.tensor(inputs_np,dtype=torch.float)
print(inputs.shape)

#target.shape is [N,1,3]
targets = torch.tensor(targets_np,dtype=torch.float)
targets = F.normalize(targets)
print(len(targets))
print(targets.shape)


inputs = inputs.to(device)
targets = targets.to(device)

#train = MyDataset(inputs,targets)
train = TensorDataset(inputs,targets)

train_loader = DataLoader(train,shuffle=True,batch_size=10)


os.system("rm plots/*.png")
os.system("rm loss.csv")

epoch = 0
img_count = 0
loss_track = []
es = time.time()
while True:
    loss_total = 0.0

    correct = 0
    #ann.to(device)
    for data,target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        out = ann(data)
        loss = loss_fn(out,target)
        loss.backward()
        optimizer.step()
       
        loss_total += loss.item()

        count_correct = (torch.abs(out - target) < 0.01).float().sum()
        if count_correct == 3:
            correct += 1
    
    loss_track.append({"epoch": epoch,"loss": loss_total})
    if epoch % 5 == 0:
        ee = time.time()
        print(f"epoch={epoch},loss={loss_total}, correct={correct}, time={ee-es} sec")
        es = ee
    
        plt.tight_layout()
    
        #https://stackoverflow.com/questions/55594969/how-to-visualise-filters-in-a-cnn-with-pytorch
        kernels = ann.conv1.cpu().weight.detach().clone()      
        # normalize to (0,1) range so that matplotlib
        # can plot them
        kernels = kernels - kernels.min()
        kernels = kernels / kernels.max()
        filter_img =utils.make_grid(kernels, nrow = 10)
        # change ordering since matplotlib requires images to 
        # be (H, W, C)
        plt.imshow(filter_img.permute(1, 2, 0))
        plt.savefig(f"plots/conv_{img_count:05d}.png",dpi=300)
        img_count += 1
        plt.close()

        with open("loss.csv","a") as f:
            for pair in loss_track:
                f.write(f"{pair['epoch']},{pair['loss']}\n")

        # exit()
        # for i in range(ann.get_conv_layer_count()):
        #     plt.subplot(8,4,i+1)
        #     plt.axis('off')
        #     w = ann.get_conv1()[i][0]
        #     w = w.detach().numpy()
        #     plt.imshow(w)
        # plt.savefig(f"plots/conv_{epoch}.png")
        # plt.close()




    
        

    epoch += 1

    #if correct > 0.95*len(targets):
    if loss_total < 1e-10:
        break

print(f"epoch={epoch},loss={loss_total}")
kernels = ann.conv1.weight.detach().clone()
       
# normalize to (0,1) range so that matplotlib
# can plot them
kernels = kernels - kernels.min()
kernels = kernels / kernels.max()
filter_img =utils.make_grid(kernels, nrow = 10)
# change ordering since matplotlib requires images to 
# be (H, W, C)
plt.imshow(filter_img.permute(1, 2, 0))
plt.savefig(f"plots/conv_{epoch}.png",dpi=300)
plt.close()

