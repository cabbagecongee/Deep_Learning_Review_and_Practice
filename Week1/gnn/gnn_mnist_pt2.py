# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils

import time
from datetime import datetime

import networkx as nx
import numpy as np
import torch
import torch.optim as optim

from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import MNISTSuperpixels

import torch_geometric.transforms as T
 
from tensorboardX import SummaryWriter #interface between PyTorch and TensorBoard
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import json
import urllib.request


# %%
class GNNStack(nn.Module): #a custom stack that defines a flexible Graph Neural Network (GNN) architecture
    def __init__(self, input_dim, hidden_dim, output_dim, task='node', conv_type='SAGE'):
        super(GNNStack, self).__init__() #initializes the GNNStack class, which inherits from nn.Module
        self.task = task #task can be 'node' for node classification or 'graph' for graph classification
        self.conv_type = conv_type 
        self.convs = nn.ModuleList()
        self.convs.append(self.build_conv_model(input_dim, hidden_dim)) #first GNN convolution layer
        self.lns = nn.ModuleList() #a list to hold layer normalization layers for stabilization
        self.lns.append(nn.BatchNorm1d(hidden_dim)) #used after first conv layer
        self.lns.append(nn.BatchNorm1d(hidden_dim)) #used after second conv layer
        self.convs.append(self.build_conv_model(hidden_dim, hidden_dim)) #adds two more GNN convolution layers to the stack
 
        # post-message-passing
        self.post_mp = nn.Sequential( 
            nn.Linear(hidden_dim, hidden_dim), nn.Dropout(0.25), 
            nn.Linear(hidden_dim, output_dim)) 
        #a feedfoward multilayer perceptron (MLP) that takes the output of the last GNN layer and applies two linear transformations with a dropout in between.
        if not (self.task == 'node' or self.task == 'graph'):
            raise RuntimeError('Unknown task.')

        self.dropout = 0.25
        self.num_layers = len(self.convs) #number of GNN layers in the stack

    def build_conv_model(self, input_dim, hidden_dim):
        if self.conv_type == 'GCN':
            return pyg_nn.GCNConv(input_dim, hidden_dim)
        elif self.conv_type == 'SAGE':
            return pyg_nn.SAGEConv(input_dim, hidden_dim)
        elif self.conv_type == 'GraphConv':
            return pyg_nn.GraphConv(input_dim, hidden_dim)
        elif self.conv_type == 'GIN':
            return pyg_nn.GINConv(nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ))
        else:
            raise ValueError(f"Unknown conv_type: {self.conv_type}")


    def forward(self, data): #excpects a torch_geometric.data.Data object as input, which contains the graph structure and node features
        x, edge_index, batch = data.x, data.edge_index, data.batch #x is node features, edge_index is the connectivity of the graph, and batch is batch vector indicating which node belongs to which graph
        if data.num_node_features == 0:
          x = torch.ones(data.num_nodes, 1) #if no node features, initilize nodes with ones

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)  #three layers of convolution, edge_index is the connectivity of the graph
            emb = x
            x = F.relu(x) 
            x = F.dropout(x, p=self.dropout, training=self.training) #makes the model robust to overfitting by randomly setting a fraction of input units to 0 during training
            if not i == self.num_layers - 1: #means we are not at the last layer
                x = self.lns[i](x) #layer normalization to stabilize the learning process by normalizing the inputs to a layer for each mini-batch

        if self.task == 'graph':
            x = pyg_nn.global_mean_pool(x, batch) #global pooling operation to aggregate node features into a single graph-level representation

        x = self.post_mp(x) # self.post_mp is a sequential model that applies two linear transformations with a dropout in between so that the output of the last layer is transformed into the desired output dimension.

        return emb, F.log_softmax(x, dim=1) #emb is the intermediate node embeddings, and the second part is the output of the model after applying a log softmax function to the final layer's output, which is useful for multi-class classification tasks.

    def loss(self, pred, label):
        return F.nll_loss(pred, label) #calculates the negative log likelihood loss between the predicted and true labels, which is commonly used for classification tasks in PyTorch.

# %%
class CustomConv(pyg_nn.MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(CustomConv, self).__init__(aggr='add') # "Add" aggregation.
        self.lin = nn.Linear(in_channels, out_channels) # Linear transformation for node features.
        self.lin_self = nn.Linear (in_channels, out_channels) # Linear transformation for self-loops.
    
    def forward(self, x, edge_index):
        # x has shape [N, in_channels], this means that x is a matrix where each row corresponds to a node and each column corresponds to a feature.
        # edge_index has shape [2, E], where E is the number of edges.

        #add self loops to the adjacency matrix
        edge_index, _ = pyg_utils.remove_self_loops(edge_index) #removes self-loops from the edge index, the _ is a placeholder for the removed self-loops
        
        #transform node feature matrix
        self_x = self.lin_self(x) #applies the linear transformation to the node features
        x = self.lin(x) #applies the linear transformation to the node features

        return self_x + self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x) #propagates messages through the graph and aggregates them using the 'add' aggregation method, then adds the self-loop features to the aggregated features.

    def message(self, x_i, x_j, edge_index, size):
        # x_i has shape [E, in_channels], where E is the number of edges.
        # x_j has shape [E, in_channels], where E is the number of edges.
        # edge_index has shape [2, E], where E is the number of edges.

        row, col = edge_index
        # row contains the source nodes and col contains the target nodes of the edges.
        deg = pyg_utils.degree(row, size=size[0], dtype=x_j.dtype) #this calculates the degree of each node in the graph, which is the number of edges connected to each node. The size parameter specifies the number of nodes in the graph, and dtype ensures that the degree tensor has the same data type as x_j.
        deg_inv_sqrt = deg.pow(-0.5) #calculates the inverse square root of the degree of each node, which is used to normalize the messages sent along the edges
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col] #calculates the normalization factor for each edge based on the degrees of the source and target nodes, which is used to normalize the messages sent along the edges.  
        return x_j
    
    def update(self, aggr_out):
        #aggr_out has shape [N, out_channels], where N is the number of nodes and out_channels is the number of output channels.
        return aggr_out #returns the aggregated output, which is the result of the message passing

# %%
def train(train_loader, test_loader, input_dim, output_dim, writer):
    #build model
    model = GNNStack(input_dim, 32, output_dim, task='graph') #initializes the GNNStack model with input dimension, hidden dimension, output dimension, and task type
    opt = optim.Adam(model.parameters(), lr= 0.001, weight_decay=5e-4) #uses the Adam optimizer with a learning rate of 0.001

    #train
    for epoch in range(100):
        total_loss = 0
        model.train() #sets the model to training mode
        for batch in train_loader:
            opt.zero_grad() #zeroes the gradients of the optimizer
            emb, pred = model(batch) #forward pass through the model
            loss = model.loss(pred, batch.y)
            loss.backward() #backward pass to compute gradients
            opt.step() #updates the model parameters using the optimizer
            total_loss += loss.item() * batch.num_graphs #accumulates the total loss for the epoch * batch.num_graphs is used to scale the loss by the number of graphs in the batch, which is useful for graph classification tasks.
        total_loss /= len(train_loader.dataset) #scales the total loss by the number of samples in the dataset
        writer.add_scalar('train/loss', total_loss, epoch) #logs the training loss to TensorBoard
        writer.add_scalar('test/loss', loss.item(), epoch)

        if epoch % 10 == 0:
            test_acc = test(test_loader, model)
            print(f'Epoch: {epoch:03d} | Loss: {total_loss:.4f} | Test Accuracy: {test_acc:.4f}')
            writer.add_scalar('test/acc', test_acc, epoch)
        
    return model


# %%
def test(loader, model, is_validation=False):
    model.eval()

    correct = 0
    for data in loader:
        with torch.no_grad(): # disables gradient calculation, which is useful for inference and testing to save memory and computation time
            emb, pred = model(data)
            pred = pred.argmax(dim=1) #gets the predicted class labels by taking the index of the maximum value along the specified dimension (dim=1)
            label = data.y

        if model.task == 'node':
            mask = data.val_mask if is_validation else data.test_mask
            #node classification: only eval on nodes in test set
            pred = pred[mask]
            label = label[mask]
        correct += pred.eq(label).sum().item() #counts the number of correct predictions by comparing the predicted labels with the true labels and summing the number of matches

    if model.task == 'graph':
        total = len(loader.dataset) #for graph classification, the total number of graphs in the dataset
    else:
        total = 0
        for data in loader.dataset:
            total += torch.sum(data.train_mask).item() #for node classification, the total number of nodes in the training set
    return correct / total if total > 0 else 0 #returns the accuracy as the ratio of correct predictions to the total number of samples, or 0 if there are no samples
    

# %%
dataset = MNISTSuperpixels(root='./data', transform=T.Cartesian(), pre_transform=None)
train_dataset = dataset[:60000]
test_dataset  = dataset[60000:]

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=64)

writer = SummaryWriter(log_dir="./log")
model = train(train_loader, test_loader, input_dim=1, output_dim=10, writer=writer)
writer.flush()
writer.close()


