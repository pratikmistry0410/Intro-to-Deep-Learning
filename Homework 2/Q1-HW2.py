import math
import numpy as np

# Inverse Node i.e 1/x
class InverseFunction():
    input = 1                                         
    def __init__(self,num):                           # Constructor of the node for variable initialization
        self.input=num
    def output(self):
        result = 1/self.input                         # Output of node: 1/x
        return result
    def localGradient(self):
        return -1*math.pow(self.input,-2)             # Local gradient: -1/(x^2)
    def downstream(self,upstreamGradient):
        return self.localGradient()*upstreamGradient        # Downstream gradient of current node = Upstream gradient of previous node = upstreamGradient*localGradient


# Linear Node i.e. x+2
class LinearFunction():
    input = 1
    b=0
    def __init__(self,num,b):                        # Constructor of the node for variable initialization           
        self.input=num
        self.b = b
    def output(self):
        result = self.input+self.b                   # Output of node: x + c
        return result
    def localGradient(self):                         # Local gradient: 1
        return 1
    def downstream(self,upstreamGradient):
        return self.localGradient()*upstreamGradient        # Downstream gradient of current node = Upstream gradient of previous node = upstreamGradient*localGradient

# Add/Distributor Node i.e. x+y
class AddFunction():
    input1 = 0
    input2 = 0
    def __init__(self,num1,num2):                   # Constructor of the node for variable initialization
        self.input1=num1
        self.input2=num2
    def output(self):
        result = self.input1+self.input2            # Output of node: x + y
        return result
    def localGradient(self):                        # Local gradient: 1
        return 1
    def downstream(self,upstreamGradient):
        return self.localGradient()*upstreamGradient        # Downstream gradient of current node = Upstream gradient of previous node = upstreamGradient*localGradient


# Sine Node i.e. Sin(x)
class SineFunction():
    input = 1
    def __init__(self,num):                         # Constructor of the node for variable initialization
        self.input=num 
    def output(self):
        result = math.sin(self.input)               # Output of node: Sin(x)
        return result
    def localGradient(self):                        # Local gradient: Cos(x)
        return math.cos(self.input)
    def downstream(self,upstreamGradient):
        return self.localGradient()*upstreamGradient        # Downstream gradient of current node = Upstream gradient of previous node = upstreamGradient*localGradient


# Cosine Node i.e. Cos(x)
class CosineFunction():
    input = 1
    def __init__(self,num):                         # Constructor of the node for variable initialization
        self.input=num
    def output(self):
        result = math.cos(self.input)               # Output of node: Cos(x)
        return result
    def localGradient(self):
        return -math.sin(self.input)                # Local gradient: -Sin(x)
    def downstream(self,upstreamGradient):
        return self.localGradient()*upstreamGradient        # Downstream gradient of current node = Upstream gradient of previous node = upstreamGradient*localGradient


# Multiply/Switcher Node i.e. x*y
class MultiplyFunction():
    input1 = 1
    input2 = 1
    def __init__(self,num1,num2):                   # Constructor of the node for variable initialization
        self.input1=num1
        self.input2=num2
    def output(self):
        result = self.input1*self.input2            # Output of node: x * y
        return result
    def localGradient1(self):                       # Local gradient for calculating dW is x
        return self.input2
    def localGradient2(self):                       # Local gradient for calculating dX is W
        return self.input1
    def downstream1(self,upstreamGradient):
        return self.localGradient1()*upstreamGradient               # Downstream gradient of current node = Upstream gradient of previous node = upstreamGradient*localGradient
    def downstream2(self,upstreamGradient):
        return self.localGradient2()*upstreamGradient               # Downstream gradient of current node = Upstream gradient of previous node = upstreamGradient*localGradient

# Square Function/Node i.e. X^2
class SquareFunction():
    input = 1
    def __init__(self,num):                         # Constructor of the node for variable initialization
        self.input=num
    def output(self):
        result = math.pow(self.input,2)             # Output of node: x^2
        return result
    def localGradient(self):                        # Local gradient: 2x
        return 2*self.input
    def downstream(self,upstream):                  # Downstream gradient of current node = Upstream gradient of previous node = upstreamGradient*localGradient
        return self.localGradient()*upstream


# Computational Graph Class as an INTERFACE to calculate the Forward and Backward Propogation values 
class ComputationalGraphFunction():
    W1=0
    W2=0
    X1=0
    X2=0
    def __init__(self,n1,n2,n3,n4):                         # Constructor of the node for variable initialization
        self.W1 = n1
        self.X1 = n2
        self.W2 = n3
        self.X2 = n4

    def forward (self):
        N1=MultiplyFunction(self.W1,self.X1)                    # Creating first object of Multiply node and passing W1 and X1 as input variables
        N2=SineFunction(N1.output())                            # Creating object of Sine Node and passing W1.x1 as input variable
        N3=SquareFunction(N2.output())                          # Creating object of Squaring Function/Node and calculating Sin^2(W1.X1)
        
        N4 = MultiplyFunction(self.W2,self.X2)                  # Creating object of Multiply node and passing W2 and X2 as input variables
        N5 = CosineFunction(N4.output())                        # Creating object of Cosine Node and passing W2.x2 as input variable
        
        N6 = AddFunction(N3.output(),N5.output())               # Creating object of Add Node and passing Sin^2(W1.X1) and Cos(W2.X2) as input variables
        N7 = LinearFunction(N6.output(),2)                      # Creating object of Linear Node and passing output of add node as input = x+2
        N8 = InverseFunction(N7.output())                       # Creating object of Inverse Node and to calculate 1/x i.e. output of given function
        
        
        print("\nThe output of W1*X1 function is: ", N1.output())
        print("The output of Sin(W1*X1) function is: ", N2.output())
        print("The output of Sin^2(W1*X1) function is: ", N3.output())
        print("\nThe output of W2*X2 function is: ", N4.output())
        print("The output of Cos(W2*X2) function is: ", N5.output())
        print("\nThe output of Sin^2(W1.X1) + Cos(W2.X2) function is: ", N6.output())
        print("The output of (Sin^2(W1.X1) + Cos(W2.X2) + 2) function is: ", N7.output())
        print("The output of inverse of (Sin^2(W1.X1) + Cos(W2.X2) + 2) function is: ", N8.output())
        
        result= N8.output()                                     
        return result                                           # Return feed forward output
        
    def backward (self):
        # Calculating the feed forward for each node which will help in back propogation since downstreamGradient = localGradient*upstreamGradient
        N1 = MultiplyFunction(self.W1,self.X1)
        N2 = SineFunction(N1.output())
        N3 = SquareFunction(N2.output())
        N4 = MultiplyFunction(self.W2,self.X2)
        N5 = CosineFunction(N4.output())
        N6 = AddFunction(N3.output(),N5.output())
        N7 = LinearFunction(N6.output(),2)
        N8 = InverseFunction(N7.output())

        # Calculating the gradients of W and x using back propogation and chain rule
        dW1= N1.downstream1(N2.downstream(N3.downstream(N6.downstream(N7.downstream(N8.downstream(1))))))
        dX1= N1.downstream2(N2.downstream(N3.downstream(N6.downstream(N7.downstream(N8.downstream(1))))))
        dW2= N4.downstream1(N5.downstream(N6.downstream(N7.downstream(N8.downstream(1)))))
        dX2= N4.downstream2(N5.downstream(N6.downstream(N7.downstream(N8.downstream(1)))))

        print("\nThe local gradient at inverse function i.e. 1/(Sin^2(W1.X1) + Cos(W2.X2) + 2) is: ", N8.localGradient())
        print("The local gradient at linear function i.e. (Sin^2(W1.X1) + Cos(W2.X2) + 2) is: ", N7.localGradient())
        print("The local gradient at distributor function i.e. Sin^2(W1.X1) + Cos(W2.X2) is: " , N6.localGradient())
        print("The local gradient at Square Function i.e. Sin^2(W1.X1) is: ", N3.localGradient())
        print("The local gradient at Sine Function i.e. Sin(W1.X1) is: ", N2.localGradient())

        print("\nThe local gradient at Cosine Function i.e. Cos(W2.X2) is: ", N5.localGradient())
        
        return dW1,dX1,dW2,dX2
    

# Driver Function
if __name__ == "__main__":
    computationalGraph = ComputationalGraphFunction(-3,-2,2,-1)
    forward_feed_output = computationalGraph.forward()
    print("\nThe output of given computational function by forward propogation is: ", forward_feed_output)
    print("------------------------------------------------------------------------------------------------------------------")

    dW1, dx1, dW2, dx2 = computationalGraph.backward()
    print("------------------------------------------------------------------------------------------------------------------")
    print("\nThe local gradient of W i.e. dW1 and dW2 by back propogation is: ", dW1, "and" , dW2)
    print("\nThe local gradient of X i.e. dx1 and dx2 by back propogation is: ", dx1, "and", dx2)