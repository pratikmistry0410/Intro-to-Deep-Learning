import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


# load mini training data and labels
mini_train = np.load('knn_minitrain.npy')
mini_train_label = np.load('knn_minitrain_label.npy')

# randomly generate test data
mini_test = np.random.randint(20, size=20)
mini_test = mini_test.reshape(10,2)


# Define knn classifier
def kNNClassify(newInput, dataSet, labels, k):
    result=[]
    ########################
    # Input your code here #
    ########################
    l2_distance=np.zeros((10,40))                               # Array for storing L2 distance for all test data w.r.t training data
    for i in range(len(newInput)):                              # Finding L2 distance of each random test data
        for j in range(len(dataSet)):
            distance = np.linalg.norm(dataSet[j]-newInput[i])   # Calculating the L2 Distance 
            l2_distance[i,j] = distance                         # Storing the L2 distance for each test data
    
    
    for i in range(len(newInput)):                                # Finding the true label for each random test data
        label_count = np.zeros(4)                                 # Creating array for counting the label values for KNN - 0,1,2,3
        knn_indices = np.argsort(l2_distance[i])[:k]              # Sorting in ascending order and finding indices of "K" nearest neighbours
        for j in range(len(knn_indices)):                         # Finding label of each "K" nearest neighbour
            label = labels[knn_indices[j]]                        # Getting the label of each "K" nearest neighbour and incrementing the count
            label_count[label]+=1                                 
        result.append(np.argmax(label_count))                     # Appending the label (index) having max count to the result list 

    ####################
    # End of your code #
    ####################
    return result

outputlabels=kNNClassify(mini_test,mini_train,mini_train_label,10)


print ('random test points are:', mini_test)
print ('knn classfied labels for test:', outputlabels)

# plot train data and classfied test data
train_x = mini_train[:,0]
train_y = mini_train[:,1]
fig = plt.figure()
plt.scatter(train_x[np.where(mini_train_label==0)], train_y[np.where(mini_train_label==0)], color='red')
plt.scatter(train_x[np.where(mini_train_label==1)], train_y[np.where(mini_train_label==1)], color='blue')
plt.scatter(train_x[np.where(mini_train_label==2)], train_y[np.where(mini_train_label==2)], color='yellow')
plt.scatter(train_x[np.where(mini_train_label==3)], train_y[np.where(mini_train_label==3)], color='black')

test_x = mini_test[:,0]
test_y = mini_test[:,1]
outputlabels = np.array(outputlabels)
plt.scatter(test_x[np.where(outputlabels==0)], test_y[np.where(outputlabels==0)], marker='^', color='red')
plt.scatter(test_x[np.where(outputlabels==1)], test_y[np.where(outputlabels==1)], marker='^', color='blue')
plt.scatter(test_x[np.where(outputlabels==2)], test_y[np.where(outputlabels==2)], marker='^', color='yellow')
plt.scatter(test_x[np.where(outputlabels==3)], test_y[np.where(outputlabels==3)], marker='^', color='black')

#save diagram as png file
plt.savefig("miniknn.png")