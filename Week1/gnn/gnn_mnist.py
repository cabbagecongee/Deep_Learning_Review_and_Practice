#reference: https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/gnn/gnn-basic-edge-1.ipynb

import time
import numpy as np
from scipy.spatial.distance import cdist
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
import matplotlib.pyplot as plt

def main():
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    RANDOM_SEED = 1
    LEARNING_RATE = 0.01
    BATCH_SIZE = 128
    IMG_SIZE = 28
    NUM_EPOCHS = 10
    NUM_CLASSES = 10

    #dataset
    train_indices = torch.arange(0, 59000) 
    test_indices = torch.arange(59000, 60000) 

    custom_transform = transforms.Compose([transforms.ToTensor()])
    train_and_valid = datasets.MNIST(root='./data', train=True, download=True, transform=custom_transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=custom_transform)
    train_dataset = Subset(train_and_valid, train_indices)
    valid_dataset = Subset(train_and_valid, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=4, shuffle=False)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, num_workers=4, shuffle=False)

    for images, labels in train_loader:
        print("Image batch dimensions:", images.shape)
        print("Label batch dimensions:", labels.shape)
        break

    #make a 2D coordinate array for images (MNIST: 784 x 2)
    def make_coordinate_array(img_size, out_size=4): #out_size is the size of the output grid, default is 4, this means we want a 4x4 grid of coordinates
        n_rows = img_size * img_size 
        col, row = np.meshgrid(np.arange(img_size), np.arange(img_size)) #create a meshgrid of coordinates
        coord = np.stack((col, row), axis=2).reshape(-1, 2) #stack the coordinates and reshape them to a 2D array
        coord = (coord - np.mean(coord, axis=0)) / (np.std(coord, axis=0) + 1e-5)  # normalize the coordinates
        coord = torch.from_numpy(coord).float()  # convert to a tensor
        
        #reshape to [N, N, out_size]
        coord = torch.cat((coord.unsqueeze(0).repeat(n_rows, 1, int(out_size / 2-1)), coord.unsqueeze(1).repeat(1, n_rows, 1)), dim=2)
        return coord

    class GraphNet(nn.Module):
        def __init__(self, img_size=IMG_SIZE, coord_features=4, num_classes=10):
            super(GraphNet, self).__init__()

            n_rows = img_size**2
            self.fc = nn.Linear(n_rows, num_classes, bias=False) #bias=False means we don't use bias in the linear layer, this is because we will use the adjacency matrix to compute the output
            coord = make_coordinate_array(img_size, coord_features)
            self.register_buffer('coord', coord) #register_buffer means we register the coordinate tensor as a buffer, it will not be updated during training, but will be saved with the model

            #edge predictor
            self.pred_edge_fc = nn.Sequential(nn.Linear(coord_features, 64), # coord -> hidden layer with 64 units
                                            nn.ReLU(),
                                            nn.Linear(64, 1), #hidden -> output layer with 1 unit (edge weight)
                                            nn.Tanh())
            
        def forward(self, x):
            B = x.size(0)

            #predict edges
            A = self.pred_edge_fc(self.coord).squeeze()  #predict edges using the coordinate tensor, A is the adjacency 
            #added normalization
            A = (A + A.transpose(0, 1)) / 2
            A = torch.softmax(A, dim=-1)
            self.A = A


            A_tensor = self.A.unsqueeze(0).expand(B, -1, -1)  #expand the adjacency matrix to match the batch size
            
            x_reshape = x.view(B, -1, 1)  #reshape the input tensor to [B, N, 1]
            avg_neighbor_features = torch.bmm(A_tensor, x_reshape).view(B, -1)
            logits = self.fc(avg_neighbor_features)  #compute the logits using the linear layer
            probas = F.softmax(logits, dim=1)  #apply softmax to the logits to get probabilities
            return logits, probas

    torch.manual_seed(RANDOM_SEED)
    model = GraphNet(img_size=IMG_SIZE, num_classes=NUM_CLASSES)

    model = model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)  

    def compute_acc(model, data_loader, device):
        correct_pred, num_examples = 0, 0
        for features, targets in data_loader:
            features = features.to(device)
            targets = targets.to(device)
            logits, probas = model(features)
            _, predicted_labels = torch.max(probas, 1)
            num_examples += targets.size(0)
            correct_pred += (predicted_labels == targets).sum()
        return correct_pred.float()/num_examples * 100
        

    start_time = time.time()

    cost_list = []
    train_acc_list, valid_acc_list = [], []


    for epoch in range(NUM_EPOCHS):
        epoch_costs = []
        model.train()
        for batch_idx, (features, targets) in enumerate(train_loader):
            
            features = features.to(DEVICE)
            targets = targets.to(DEVICE)
                
            logits, probas = model(features)
            cost = F.cross_entropy(logits, targets)
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
            
            epoch_costs.append(cost.item())
            if not batch_idx % 150:
                print (f'Epoch: {epoch+1:03d}/{NUM_EPOCHS:03d} | '
                    f'Batch {batch_idx:03d}/{len(train_loader):03d} |' 
                    f' Cost: {cost:.4f}')
        cost_list.append(np.mean(epoch_costs))
            

        model.eval()
        with torch.set_grad_enabled(False): # save memory during inference
            
            train_acc = compute_acc(model, train_loader, device=DEVICE)
            valid_acc = compute_acc(model, valid_loader, device=DEVICE)
            
            print(f'Epoch: {epoch+1:03d}/{NUM_EPOCHS:03d}\n'
                f'Train Accuracy: {train_acc:.2f} | Validation Accuracy: {valid_acc:.2f}')
            
            train_acc_list.append(train_acc.cpu().item())
            valid_acc_list.append(valid_acc.cpu().item())
            
        elapsed = (time.time() - start_time)/60
        print(f'Time elapsed: {elapsed:.2f} min')
    
    elapsed = (time.time() - start_time)/60
    print(f'Total Training Time: {elapsed:.2f} min')

    train_acc_list = [x.item() if torch.is_tensor(x) else x for x in train_acc_list]
    valid_acc_list = [x.item() if torch.is_tensor(x) else x for x in valid_acc_list]
    cost_list = [x.item() if torch.is_tensor(x) else x for x in cost_list]

    plt.plot(cost_list)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss over {NUM_EPOCHS} Epochs")
    plt.savefig("/opt/repo/output/gnn_loss.png")
    plt.show()

    plt.plot(train_acc_list)
    plt.xlabel("Epoch")
    plt.ylabel("Training Accuracy")
    plt.title(f"Training Accuracy over {NUM_EPOCHS} Epochs")
    plt.savefig("/opt/repo/output/gnn_train_acc.png")
    plt.show()

    plt.plot(valid_acc_list)
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.title(f"Validation Accuracy over {NUM_EPOCHS} Epochs")
    plt.savefig("/opt/repo/output/gnn_valid_acc.png")
    plt.show()

if __name__ == "__main__":  
    main()