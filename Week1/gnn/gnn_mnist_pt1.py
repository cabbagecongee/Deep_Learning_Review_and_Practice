import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import sklearn.metrics as metrics
import matplotlib.pyplot as plt

#download and prepare the MNIST dataset
BATCH_SIZE = 32

transform = transforms.Compose([transforms.ToTensor()])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# print(f"Number of training samples: {len(trainset)}")
# print(testset[10])

class GNN(nn.Module):
    def __init__(self):
        super(GNN, self).__init__() # Initialize the GNN model
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3) # Convolutional layer with 32 filters of size 3x3, just here for image classfication
        self.d1= nn.Linear(26*26*32, 128) # 26*26*32 is the flattened size of the MNIST images after convolution
        self.d2 = nn.Linear(128, 10) 

    def forward(self, x):
        x = self.conv1(x) # Apply the first convolutional layer
        x = F.relu(x) # Apply ReLU activation, F is the functional module in PyTorch
        x = x.flatten(start_dim=1) # Flatten the tensor to prepare for the fully connected layers
        # x = x.view(x.size(0), -1) #essentially does the same thing as flatten, but more explicit, x.size(0) is the batch size, -1 means infer the size of the second dimension automatically
        x = self.d1(x) # Apply the first fully connected layer
        x = F.relu(x) # Apply ReLU activation

        #logits =>32x10 
        logits = self.d2(x) # Apply the second fully connected layer, basically the ten classes of MNIST (0-9)
        return logits 

n_epochs = 5
learning_rate = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Use GPU if available, otherwise use CPU
model = GNN().to(device) # Instantiate the GNN model and move it to the device
criterion = nn.CrossEntropyLoss() # Loss function for multi-class classification, criterion is the loss function
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) # Optimizer for training the model, Adam optimizer is used here

def train(epoch):
    model.train() # Set the model to training mode
    train_running_loss = 0.0  # Initialize the running loss and training accuracy
    train_acc = 0.0
    ##training step
    for i, (images, labels) in enumerate(trainloader, 1): # Iterate over the training data
        images, labels = images.to(device), labels.to(device) # Move data to the device

        logits = model(images) # Forward pass through the model, modle(images) returns the logits (raw scores) for each class
        loss = criterion(logits, labels) # Compute the loss, criterion is the loss function, logits are the raw scores from the model, labels are the true labels
        optimizer.zero_grad() # Zero the gradients
        loss.backward() # Backward pass
        optimizer.step() # Update the weights

        train_running_loss += loss.detach().item()  # Accumulate the loss, detach() is used to remove the tensor from the computation graph, item() returns the value as a Python number
        train_acc += (logits.argmax(dim=1).flatten() == labels).float().mean().item()
        if i % 100 == 0: # Print the loss every 100 batches
            print('Epoch %d, | Loss: %.4f, | Training Accuracy: %.4f' \
                % (epoch, train_running_loss / i, train_acc / i)) # Print the average loss and accuracy for the epoch
        
def test():
    test_acc = 0.0  # Initialize the test accuracy
    model.eval()
    for i, (images, labels) in enumerate(testloader, 0): # Iterate over the test data, 0 means the index starts from 0
        images, labels = images.to(device), labels.to(device) # Move data to the device
        outputs = model(images) # Forward pass through the model
        test_acc += (torch.argmax(outputs, 1).flatten() == labels).type(torch.float).mean().item() # Compute the test accuracy
        preds = torch.argmax(outputs, 1).flatten().cpu().numpy() # Get the predicted labels and move them to CPU for further processing
        #convert back to numpy for sklearn metrics
        labels = labels.cpu().numpy() # Move the true labels to CPU and convert to numpy
        # p = metrics.precision_score(labels, preds, average='macro', zero_division=0) # Compute precision score
        # print('precision: ', p)
    
    print('Test Accuracy: %.4f' % (test_acc / i)) # Print the average test accuracy


if __name__ == "__main__":
    test()
    for epoch in range(n_epochs): # Loop over the number of epochs
        train(epoch) # Train the model for the current epoch
        test() # Test the model after each epoch
