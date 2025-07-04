import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt


n_epochs = 10
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5 
log_interval = 10

random_seeds = 1
torch.backends.cudnn.enabled = False #since cuDNN uses nondeterministic algorithms, we disable it for reproducibility
torch.manual_seed(random_seeds)

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(
        './data/', train=True, download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,)) # Normalizing the dataset, these values are the mean and std of the MNIST dataset
        ])),
    batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(
        './data/', train=False, download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])), 
    batch_size=batch_size_test, shuffle=True) #same normalization as above but for the test set

examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)
print(example_data.shape)  # should be [1000, 1, 28, 28] for the test set this is the tensor shape of the images (1000 images, 1 channel (grayscale), 28x28 pixels)

#building the network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5) #conv1 means convolutional layer 1, 1 is the number of input channels (grayscale image), 10 is the number of output channels (filters), kernel_size=5 means a 5x5 filter
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5) #conv2 is the second convolutional layer, 10 is the number of input channels (output from conv1), 20 is the number of output channels (filters)
        self.conv2_drop = nn.Dropout2d() # Dropout layer to prevent overfitting
        self.fc1 = nn.Linear(320, 50) #fc1 means fully connected layer 1, 320 is the flattened size of the output from the second convolution layer
        self.fc2 = nn.Linear(50, 10) #fc2 is the final fully connected layer, 10 is the number of output classes (digits 0-9)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2)) # Apply ReLU activation and max pooling after the first convolution layer
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2)) # Apply dropout after the second convolution layer
        x = x.view(-1, 320)  # Flatten the tensor
        x = F.relu(self.fc1(x)) # Fully connected layer with ReLU activation
        x = F.dropout(x, training=self.training)  # Apply dropout during training
        x = self.fc2(x) # Final output layer
        return F.log_softmax(x, dim = 1) # Log softmax for multi-class classification
    
network = Net()  # Instantiate the network
optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)  # Stochastic Gradient Descent optimizer

train_losses = []
train_counter = []
test_losses = []
test_counter = [i * len(train_loader.dataset) for i in range(n_epochs + 1)]
test_accuracies = []

def train(epoch):
    network.train()
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    avg_loss = running_loss / len(train_loader)
    train_losses.append(avg_loss)
    train_counter.append(epoch * len(train_loader.dataset))

    torch.save(network.state_dict(), 'Week1/results/mnist_cnn.pth')
    torch.save(optimizer.state_dict(), 'Week1/results/mnist_cnn_optimizer.pth')

def test():
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = network(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True) # Get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset) # Average loss over the test set
    test_losses.append(test_loss)

    accuracy = 100. * correct / len(test_loader.dataset)
    test_accuracies.append(accuracy)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n')

test()
for epoch in range(1, n_epochs + 1):
    train(epoch)
    test()

fig = plt.figure()
plt.plot(train_losses, color='blue')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.savefig('Week1/results/mnist_cnn_loss.png')
plt.show()  # Show the loss plot 

epochs = list(range(len(test_accuracies)))  # usually [0, 1, 2, ... n_epochs]

fig = plt.figure()
plt.plot(epochs, test_accuracies, marker='o', color='green')
plt.title('Test Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.grid(True)
plt.savefig('Week1/results/mnist_cnn_accuracy.png')
plt.show()
