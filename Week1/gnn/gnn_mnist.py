import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import sklearn.metrics as metrics

#download and prepare the MNIST dataset
BATCH_SIZE = 32

transform = transforms.Compose([transforms.ToTensor()])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

print(f"Number of training samples: {len(trainset)}")
print(testset[10])

class GNN(nn.Module):
    def __init__(self):
        super(GNN, self).__init__()
        self.fc1 = nn.Linear(26*26*32, 128) # 26*26*32 is the flattened size of the MNIST images after convolution
        self.fc2 = nn.Linear(128, 10) 

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x