
import numpy as np
import math

# L2 Loss Node i.e. ||q^2||
class L2Function():
    inputArr=np.zeros((3,1))                            
    def __init__(self,arr):                             # Constructor of the node for variable initialization
        self.inputArr=np.copy(arr)
    def output(self):
        return np.sum(np.square(self.inputArr))         # Output of node: q1^2 + q2^2 + q3^2 + ..... + qN^2
    def localGradient(self):
        return np.multiply(2,self.inputArr)             # Local Gradient: 2q
    def downstream(self,upstreamGradient):
        return self.localGradient()*upstreamGradient    # Downstream gradient of current node = Upstream gradient of previous node = upstreamGradient*localGradient


# Sigmoid Node i.e. f(x) = 1 / (1+e^-x)
class SigmoidFunction():
    inputArr =np.zeros((3,1))
    def __init__(self,arr):                             # Constructor of the node for variable initialization
        self.inputArr=np.copy(arr)                     
    def sigmoid(self):
        s = 1/(1+np.exp(-self.inputArr))                  # Sigmoid of each input value of x
        return s
    def localGradient(self):
        return np.multiply((1-self.sigmoid()),self.sigmoid())       # Local gradient: (1-p(x))p(x)
        # return ((1-(1/(1+np.exp(-self.inputArr))))*(1/(1+np.exp(-self.inputArr))))
    def downstream(self,upstreamGradient):
        return np.multiply(self.localGradient(),upstreamGradient)   # Downstream gradient of current node = Upstream gradient of previous node = upstreamGradient*localGradient


# Multiply/Switcher Node i.e. x*y
class NodeMultiply():
    inputArr1=np.zeros((3,3))
    inputArr2=np.zeros((3,1))
    def __init__(self,arr1,arr2):                       # Constructor of the node for variable initialization
        self.inputArr1=np.copy(arr1)
        self.inputArr2=np.copy(arr2)
    def output(self):
        return np.dot(self.inputArr1,self.inputArr2)    # Multiplying W.x
    def localGradient1(self):
        return np.transpose(self.inputArr2)             # Local gradient for calculating dW is x^T
    def localGradient2(self):
        return np.transpose(self.inputArr1)
    def downstream1(self,upstreamGradient):             # Local gradient for calculating dx is W^T
        return np.dot(upstreamGradient,self.localGradient1())           # Downstream gradient of current node = Upstream gradient of previous node = upstreamGradient*localGradient
    def downstream2(self,upstreamGradient):
        return np.dot(self.localGradient2(),upstreamGradient)           # Downstream gradient of current node = Upstream gradient of previous node = upstreamGradient*localGradient



# Computational Graph Class as an INTERFACE to calculate the Forward and Backward Propogation values 
class ComputationalGraphFunction():
    W=np.zeros((3,3))
    X=np.zeros((3,1))

    def __init__(self,arr1,arr2):                       # Constructor of the implementation node for variable initialization
        self.W = np.copy(arr1)
        self.X = np.copy(arr2)

    def forward(self):
        N1 = NodeMultiply(self.W,self.X)                # Creating object of Multiply Node and passing W and X as input variables
        N2 = SigmoidFunction(N1.output())               # Creating object of Sigmoid Node and passing W.x as input variable
        N3 = L2Function(N2.sigmoid())                   # Creating object of L2 Node and passing sigmoid of (W.x) as input variables
        
        print("\nThe output of multiplication function of W and X is: \n", N1.output())
        print("The output of sigmoid function is: \n", N2.sigmoid())
        print("The output of L2 loss function is: ", N3.output())

        return N3.output()                              # Return feed forward output

    def backward(self):
        # Calculating the feed forward for each node which will help in back propogation since downstreamGradient = localGradient*upstreamGradient
        N1 = NodeMultiply(self.W,self.X)
        N2 = SigmoidFunction(N1.output())
        N3 = L2Function(N2.sigmoid())
        
        # Calculating the gradients of W and x using back propogation and chain rule
        dW = N1.downstream1(N2.downstream(N3.downstream(1)))              
        dX = N1.downstream2(N2.downstream(N3.downstream(1)))
        
        print("\nThe local gradient at L2 Loss function is: \n", N3.localGradient())
        print("The local gradient at sigmoid function is: \n", N2.localGradient())
        return dW,dX                                    # Return gradients of W and x as output


# Driver Function
if __name__ == "__main__":
    computationalGraph = ComputationalGraphFunction([[1,2,3],[3,1,2],[2,3,1]],[[1],[1],[1]])              # Creating object of ComputationalGraphFunction Node and passing W = 3x3 and x = 3x1 array
    forward_feed_output = computationalGraph.forward()              # Calculating output forward propogration
    print("\nThe output of given computational function by forward propogation is: ", forward_feed_output)
    print("------------------------------------------------------------------------------------------------------------------")
    
    dW, dx = computationalGraph.backward()                          # Calculating gradients of W and X using backward propogation
    print("------------------------------------------------------------------------------------------------------------------")
    print("\nThe local gradient of W i.e. dW by back propogation is: \n", dW)
    print("\nThe local gradient of X i.e. dx by back propogation is: \n", dx)