import torch
from torch import optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision.transforms import ToTensor
import json

class neural_net(nn.Module):
    def __init__(self,input_neuron_count=4,hidden_neuron_count=5,output_neuron_count=3,learning_rate=0.1,bias_rate=0.1,init_neuron_bias=0.01):
        super(neural_net,self).__init__()

        self.layer1 = torch.nn.Linear(input_neuron_count, hidden_neuron_count)
        self.relu = torch.nn.Sigmoid()
        self.layer2 = torch.nn.Linear(hidden_neuron_count, output_neuron_count)

    def forward(self,x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

    def info(self):
        print(self.classifier)


ann = neural_net(input_neuron_count=784,hidden_neuron_count=40,output_neuron_count=6)
optimizer = optim.SGD(ann.parameters(),lr=0.1)
loss_fn = nn.MSELoss()


"""
pairs = [
    
            [[0.10,0.50,0.10,0.25],[0.25,0.75,0.5]],
            [[1.00,0.00,0.20,0.33],[1,0.35,0.1]],
            [[1.00,0.50,0.35,0.10],[0.7,0.85,0.8]],
            [[0.30,0.20,0.85,0.95],[0.5,0.6,0.55]],
            [[0.70,0.60,0.50,0.85],[0.2,0.9,0.40]],
            [[0.88,0.20,0.25,0.65],[0.1,0.4,0.1]],
            [[0.60,0.25,0.15,0.75],[0.5,0.1,0.9]],
]
"""


with open("training_pairs.json","r") as f:
    pairs = json.load(f)

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

test_pair_raw = [
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ],
    [
        [0, 0, 1, 0, 0, 1]
    ]
]

test_pair1 = TensorDataset(
                        torch.tensor(test_pair_raw[0],dtype=torch.float32),
                        torch.tensor(test_pair_raw[1],dtype=torch.float32))
test_pair = DataLoader(test_pair1,batch_size=1)

#not sure what this line is for
#ann.train()

epoch = 0
while True:
    loss_total = 0.0
    acc_total = 0.0
    for (data,target) in train_loader:
        out = ann(data)
        loss = loss_fn(out,target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss_total += loss.item()

        tin, tout = next(iter(test_pair))
        out = ann(tin)
        acc_total += loss_fn(out,tout)

    if epoch % 1000 == 0:
        print(f"epoch={epoch},loss={loss_total}, acc={acc_total}")
    epoch += 1

    if loss_total < 1e-3:
        break


"""
input,output = next(iter(train_loader))
print(input,output)
out = ann(input)
print(f"input={input}, output={out}")
"""