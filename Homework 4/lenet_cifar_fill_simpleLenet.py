from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision 
import torchvision.transforms as transforms
import time
from collections import OrderedDict

# Preparing for Data
print('==> Preparing data..')

# Training Data augmentation
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
# Testing Data preparation
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

#classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        ############################
        #### Put your code here ####
        ############################
        self.convnet = nn.Sequential(OrderedDict([
            ('c1',nn.Conv2d(3,6,kernel_size=(5,5))),                # First Convolutional Layer
            ('relu1',nn.ReLU()),                                    # Relu layer as activation function
            ('s2',nn.MaxPool2d(kernel_size=(2,2), stride=2)),       # Second Maxpool layer for subsampling
            ('c3',nn.Conv2d(6,16,kernel_size=(5,5))),               # Third convolutional layer    
            ('relu3',nn.ReLU()),                                    # Relu layer as activation function
            ('s4',nn.MaxPool2d(kernel_size=(2,2), stride=2)),       # Fourth Maxpool layer for subsampling
            ('c5',nn.Conv2d(16,120,kernel_size=(5,5))),             # Fifth convolutional layer
            ('relu5',nn.ReLU())                                     # Relu layer as activation function
        ]))
        
        self.fc = nn.Sequential(OrderedDict([                       
            ('f6',nn.Linear(120,84)),                               # Sixth Linear Layer - 120 input 84 output
            ('relu6',nn.ReLU()),                                    # Relu layer as activation function
            ('f7',nn.Linear(84,10)),                                # Seventh Linear Layer - 84 input 10 output (10 output classes)
            ('sig7',nn.LogSoftmax(dim=-1))                          # Log Softmax Classifier for classifying 10 output classes
        ]))
        ###########################
        #### End of your codes ####
        ###########################

    def forward(self, x):
        ############################
        #### Put your code here ####
        ############################

        out = self.convnet(x)                           # Passing batch images through convolutional layers
        out = out.view(out.size(0), -1)                 # Flattening the data to fed it to FC. Linear Layers
        out = self.fc(out)                              # Passing flattened data to fully connected linear layers
        
        ###########################
        #### End of your codes ####
        ###########################
    
        return out



def train(model, device, train_loader, optimizer, epoch):
    model.train()
    count = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        ############################
        #### Put your code here ####
        ############################
        
        optimizer.zero_grad()                           # defining gradient in each epoch as 0
        output  = model(data)                           # modeling for each image batch    
        criterion = nn.CrossEntropyLoss()               
        loss = criterion(output,target)                 # sum up batch loss    
        loss.backward()                                 # This is where the model learns by backpropagating
        optimizer.step()                                # And optimizes its weights here

        ###########################
        #### End of your codes ####
        ###########################
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test( model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():
    time0 = time.time()
    # Training settings
    batch_size = 128
    epochs = 50
    lr = 0.05
    no_cuda = True
    save_model = False
    use_cuda = not no_cuda and torch.cuda.is_available()
    torch.manual_seed(100)
    device = torch.device("cuda" if use_cuda else "cpu")
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)

    model = LeNet().to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    for epoch in range(1, epochs + 1):
        train( model, device, train_loader, optimizer, epoch)
        test( model, device, test_loader)

    if (save_model):
        torch.save(model.state_dict(),"cifar_lenet.pt")
    time1 = time.time() 
    print ('Traning and Testing total excution time is: %s seconds ' % (time1-time0))   
if __name__ == '__main__':
    main()
