import math
import numpy as np  
from download_mnist import load
import operator  
import time
# classify using kNN  
# x_train = np.load('../x_train.npy')
# y_train = np.load('../y_train.npy')
# x_test = np.load('../x_test.npy')
# y_test = np.load('../y_test.npy')
x_train, y_train, x_test, y_test = load()
x_train = x_train.reshape(60000,28,28)
x_test  = x_test.reshape(10000,28,28)
x_train = x_train.astype(float)
x_test = x_test.astype(float)

def kNNClassify(newInput, dataSet, labels, k): 
    result=[]
    ########################
    # Input your code here #
    ########################
    test_len = len(newInput)                                        # Getting the length of test images
    train_len= len(dataSet)                                         # Getting the length of training images

    l2_distance=np.zeros((test_len,train_len))
    for i in range(test_len):                                       # Finding L2 distance of each test image
        for j in range(train_len):
            distance = np.linalg.norm(dataSet[j]-newInput[i])       # Calculating the L2 Distance
            l2_distance[i,j] = distance                             # Storing the L2 distance for each test image
    

    for i in range(test_len):                                       # Finding the true label for each test image
        label_count = np.zeros(10)                                  # Creating array for counting the label values for numbers - 0-9
        knn_indices = np.argsort(l2_distance[i])[:k]                # Sorting in ascending order and finding indices of "K" nearest neighbours
        for j in range(len(knn_indices)):                           # Finding label of each "K" nearest neighbour
            label = labels[knn_indices[j]]                          # Getting the label of each "K" nearest neighbour and incrementing the count
            label_count[label]+=1           
        result.append(np.argmax(label_count))                       # Appending the label (index) having max count to the result list 
    
    
    ####################
    # End of your code #
    ####################
    return result

start_time = time.time()
outputlabels=kNNClassify(x_test[0:5],x_train,y_train,5)
print("Output labels for 5 test images are: ",outputlabels)
result = y_test[0:5] - outputlabels
result = (1 - np.count_nonzero(result)/len(outputlabels))
print ("---classification accuracy for knn on mnist: %s ---" %result)
print ("---execution time: %s seconds ---" % (time.time() - start_time))