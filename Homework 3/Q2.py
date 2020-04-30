import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim

def init():
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,)),])     # Transform the dataset images to numbers
    trainset = datasets.MNIST(r'..\input\MNIST', download=True, train=True, transform=transform)      # Download the training dataset with labels: 60000 images
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)                 # Load the training dataset in iterator with batch size of 128
      
    testset = datasets.MNIST(r'..\input\MNIST', download=True, train=False, transform=transform)      # Download the testing dataset with labels: 10000 images
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=True)                   # Load the testing dataset in iterator with batch size of 128
      
    model=nn.Sequential(nn.Linear(784,200), # 1 layer:- 784 input 128 output
                    nn.ReLU(),          # Defining Regular linear unit as activation function
                    nn.Linear(200,50),  # 2 Layer:- 128 Input and 64 output
                    nn.ReLU(),          # Defining Regular linear unit as activation function
                    nn.Linear(50,10),   # 3 Layer:- 64 Input and 10 O/P as (0-9)
                    # nn.Softmax()      # Not required because we are using Cross Entropy Loss layer which will calculate Softmax as well as Loss together
                ) 
                  
    # Loss function: Cross Entropy loss as it implements Softmax classifier followed by NLL internally
    loss_function = nn.CrossEntropyLoss() 

    # defining the optimiser with stochastic gradient descent and default parameters
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    return model,trainloader,testloader, loss_function, optimizer


# Model Training Function
def train(model,trainloader, loss_function, optimizer):
    start_time = time()
    epochs = 10 # total number of iteration for training
    for epoch in range(epochs):
        total_loss = 0
        for images, labels in trainloader:
            # Flatenning MNIST images with size [128,784]
            images = images.view(images.shape[0], -1) 
        
            # defining gradient in each epoch as 0
            optimizer.zero_grad()
            
            # modeling for each image batch
            output = model(images)
            
            # calculating the loss
            loss = loss_function(output, labels)
            
            # This is where the model learns by backpropagating
            loss.backward()
            
            # And optimizes its weights here
            optimizer.step()
            
            # calculating the loss
            total_loss += loss.item()
        else:
            print("Epoch {} - Training loss: {}".format(epoch, total_loss/len(trainloader)))        # Printing loss at each epoch
    print("\nTraining Time (in minutes) =",(time()-start_time)/60)                                  # Total training time output

# Model testing function
def test(model,testloader):
    correct_prediction, total_images = 0, 0                    # Initialize variables for model accuracy
    for images,labels in testloader:
        for i in range(len(labels)):
            img = images[i].view(1, 784)                        # Flatten each image to 1,784 size

            with torch.no_grad():   
                model_output = model(img)                              # Using no gradients, find probabilities of test image after model testing

            probability_dist = torch.exp(model_output)                                  # calculating probabilities
            probability_array = np.array(probability_dist.numpy()[0])                   # Converting probabilities to list
            probability_normalized = probability_array/np.sum(probability_array)        # Normalizing the probabilities calculated        

            pred_label = np.argmax(probability_normalized)                              # Find the label of the predicted image with index as location having maximum probability
            true_label = labels.numpy()[i]                                              # Find true label of the test image

            if(true_label == pred_label):
                correct_prediction += 1                         # Increment if predicted and test image have same label
            total_images += 1

    print("\nNumber Of Images Tested = ", total_images)
    print("Model Accuracy = ", (correct_prediction/total_images)*100,"%")      # Printing model accuracy
      
# main Function where init, train and test functions will be called
def main():                         
    model,trainloader,testloader, loss_function, optimizer = init()             # Initializing the model and hyperparameters
    train(model,trainloader, loss_function, optimizer)                          # training the model with training data
    test(model,testloader)                                                      # testing the model with testing data

if __name__ == "__main__":
    main()              # Main Function