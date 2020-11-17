'''
Design of a Neural Network from scratch

*************<IMP>*************
Mention hyperparameters used and describe functionality in detail in this space
- carries 1 mark

1. Number of layers - 3(input, 1 hidden, output)
2. Number of neurons per layer:
    Input layer - 9
    Hidden layer - 5
    Output layer - 1
3. Dimensions of Weight Matrices:
    Weight Matrix dimensions are (number of neruons in current layer) X (number of neurons in previous layer)
    Z is WX + b
    here X is taken with features along rows and samples along columns.
    W1 -> 5X9
    W2 -> 1X5
4. Dimensions of Bias Matrices:
    Bias Matrix will be a column vector with the number of rows equal to the number of neurons in the current layer
    b1 -> 5X1
    b2 -> 1X1
5. Activation functions :
    hidden layer -> tanh
    output layer -> sigmoid
6. Loss Function : Cross Entropy
7. Learning Rate : 0.05
8. Epochs : 2500
9. Train Test Split Used : 70/30
10. Training Accuracy obtained: 88.0579 %
11. Testing Accuracy obtained: 86.2069 %
'''
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split #Allowed for splitting dataset
parameters=dict() #Global dictionary to store weight and bias matrices
def clean(x): # Function that preprocesses the input
    #We round mean values to appropriate number of decimal places
    #Filling missing values in age column with mean of the column
    x['Age']=x['Age'].fillna(round(x['Age'].mean()))
    #Filling missing values in weight column with mean of the column
    x['Weight']=x['Weight'].fillna(round(x['Weight'].mean()))
    #Filling missing values in delivery column with the mode of the column as it has categorical values
    x['Delivery phase']=x['Delivery phase'].fillna(x['Delivery phase'].mode()[0])
    #Filling HB values with mean of the column values
    x['HB']=x['HB'].fillna(round(x['HB'].mean(),1))
    #Filling IFA values with mode of the column as it is has categorical values
    x['IFA']=x['IFA'].fillna(x['IFA'].mode()[0])
    #Missing BP values are willing with the mean of the column. Rounded to 3 decial places
    x['BP']=x['BP'].fillna(round(x['BP'].mean(),3))
    #Education and residence are filled with the mode of the column as they are a categorical column
    x['Education']=x['Education'].fillna(x['Education'].mode()[0])
    x['Residence']=x['Residence'].fillna(x['Residence'].mode()[0])
    #All non categorical columns are scaled from a range of 1 to 10
    x['Age']=(x['Age']-x['Age'].min())/(x['Age'].max()-x['Age'].min())*10
    x['Weight']=(x['Weight']-x['Weight'].min())/(x['Weight'].max()-x['Weight'].min())*10
    x['HB']=(x['HB']-x['HB'].min())/(x['HB'].max()-x['HB'].min())*10
    x['BP']=(x['BP']-x['BP'].min())/(x['BP'].max()-x['BP'].min())*10
    return x

#The below functions can be used if it is desired to change the neural network archiecture in the future
def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivative(y):
    return y * (1 - y)

# np.tanh(x) will suffice for tanh activation function
def tanh_derivative(y):
    return 1-(y*y)

def relu(x):
	if x<0:
		return 0
	else:
		return x

def relu_derivative(y):
	if y<0:
		return 0
	else:
		return 1

class NN:
    #init function for the NN class
    #The hidden layer size is set
    #Weight matrices and bias matrices are initialized
    #X and Y are initialized
    ''' X and Y are dataframes '''
    def __init__(self,X,Y):
        np.random.seed(1) #random seed 1 (pesudo-random)
        #Network consists of input layer, 1 hidden layer and then output layer
        #Number of neurons in hidden layer is set to 5
        self.h_size=5
        #Input is intialized with the features dataframe
        self.input = X
        #Learning rate is set as 0.05. Learning rate defines at what rate we progress down the slope during gradient descent
        self.learning_rate=0.05 
        '''
        Weight Matrix dimensions are (number of neruons in current layer) X (number of neurons in previous layer)
        Accordingly, weights1 size is (number of neurons in hidden layer) X (number of neurons in input layer)
                     weights2 size is (number of neurons in output layer) X (number of neurons in hidden layer)
                     Weight matrices multiplies by 0.01 to avoid exploding gradient problem
        '''
        self.weights1 = np.random.randn(self.h_size,self.input.shape[1])*0.01#We have 5 nodes in first hidden layer, 5X9 
        self.weights2 = np.random.randn(Y.shape[1],self.h_size)*0.01#Second hidden layer also has 5 nodes
        '''
        Bias is column vector with length corresponding to number of neurons in that particular layer
        Both the bias vectors are initialized to zero vectors in the beginning.
        '''
        self.bias1=np.zeros((self.h_size,1))
        self.bias2=np.zeros((Y.shape[1],1))
        self.y = Y
        #Output is initialized to a zero vector
        self.output = np.zeros(Y.shape)
        #Weights and bias are stored in a global dictionary called parameters for easier access and updation later
        self.parameters = {"W1": self.weights1,"b1": self.bias1,"W2": self.weights2,"b2": self.bias2}

       
    def get_cost(A2,Y):
        m=Y.shape[0]
        logprobs=np.multiply(np.log(A2),Y) + np.multiply(np.log(1-A2),(1-Y))
        cost= -np.sum(logprobs)
        cost=float(np.squeeze(cost))/m
        return cost

    #Forward Propogation
    def feedforward(self):
        '''
        The weight matrices and bias matrices that are updated during every epoch 
        are accessed from the parameters dictionary
        '''
        W1 = self.parameters["W1"]
        b1 = self.parameters["b1"]
        W2 = self.parameters["W2"]
        b2 = self.parameters["b2"]
        '''
        Z1=W1.X(T)+b1
        '''
        Z1 = np.dot(W1,self.input.T)+b1
        '''
        Hidden layer uses the tanh activation function.
        '''
        A1 = np.tanh(Z1)
        '''
        Z2=W2.A1+b2
        '''
        Z2 = np.dot(W2,A1)+b2
        '''
        Output layer uses sigmoid activation function (which has been defined earlier)
        '''
        A2 = sigmoid(Z2)
        '''
        Z1,Z2,A1,A2 are stored in a dictionary called cache 
        They are later to be used for back propogation
        '''
        self.cache={"Z1":Z1,"A1":A1,"Z2":Z2,"A2":A2}
        return A2
    
    #Backward Propogation
    def backprop(self):
	# application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        #m is the number of entries
        m=self.input.shape[0]
        #Getting the stored weight matrices
        W1=self.parameters["W1"]
        W2=self.parameters["W2"]
        #Getting the stored bias matrices
        b1=self.parameters["b1"]
        b2=self.parameters["b2"]
        #Getting the cached valeus
        A1=self.cache["A1"]
        A2=self.cache["A2"]
        '''
        Using chain rule for differentiation
        We find dW1, dW2, dB1, dB2
        Then update W1, W2, b1, b2 using the derivative values and the learning rate
        '''
        dZ2=A2-self.y.T
        dW2=(1/m)*(np.dot(dZ2,A1.T))
        db2 = (1/m)*np.sum(dZ2,axis=1,keepdims=True)
        dZ1 = np.dot(W2.T,dZ2)*(1-np.power(A1,2))
        dW1 = (1/m)*(np.dot(dZ1,self.input))
        db1 = (1/m)*np.sum(dZ1,axis=1,keepdims=True)
        W1 = W1-(self.learning_rate*dW1)
        b1 = b1-(self.learning_rate*db1)
        W2 = W2-(self.learning_rate*dW2)
        b2 = b2-(self.learning_rate*db2)
        '''
        Store the updated weight and bias matrices for further forward and back prop cycles
        '''
        self.parameters={"W1":W1,"W2":W2,"b1":b1,"b2":b2}

    def train(self,X,Y):
        '''
        Function that trains the neural network by taking x_train and y_train samples as input
        '''
        self.output=self.feedforward()
        self.backprop()
    def fit(self,X,Y):
        '''
        fit runs forward and back propogation as specified by number of epochs
        In each epoch the train function is called.
        Here 2500 epochs are used
        '''
        for i in range(2500):
            self.train(X,Y)
            if (i%100==0):
                print("Epoch ",i)
                        
	
    def predict(self,X):
        """
        The predict function performs a simple feed forward of weights
        and outputs yhat values 
        yhat is a list of the predicted value for X
        """
        
        '''
        The final W1, W2, b1 and b2 values are stored after all the epochs 
        in the parameters dictionary. These final weight and bias matrices
        used for making predictions given another piece of data
        '''
        W1 = self.parameters["W1"]#5X9
        b1 = self.parameters["b1"]#5X1
        W2 = self.parameters["W2"]#1X5
        b2 = self.parameters["b2"]#1X1
        Z1 = np.dot(W1,X.T)+b1#5X67
        A1 = np.tanh(Z1)#5X67
        Z2 = np.dot(W2,A1)+b2#1X67
        A2 = sigmoid(Z2)#1X67
        yhat=A2.T
        return yhat

    def CM(y_test,y_test_obs):
        '''
        Prints confusion matrix,precision,recall and F! Score 
        y_test is list of y values in the test dataset
        y_test_obs is list of y values predicted by the model
        '''

        for i in range(len(y_test_obs)):
            if(y_test_obs[i]>0.6):
                y_test_obs[i]=1
            else:
                y_test_obs[i]=0
		
        cm=[[0,0],[0,0]]
        fp=0
        fn=0
        tp=0
        tn=0
		
        for i in range(len(y_test)):
            if(y_test[i]==1 and y_test_obs[i]==1):
                tp=tp+1
            if(y_test[i]==0 and y_test_obs[i]==0):
                tn=tn+1
            if(y_test[i]==1 and y_test_obs[i]==0):
                fp=fp+1
            if(y_test[i]==0 and y_test_obs[i]==1):
                fn=fn+1
            cm[0][0]=tn
            cm[0][1]=fp
            cm[1][0]=fn
            cm[1][1]=tp

        p= tp/(tp+fp)
        r=tp/(tp+fn)
        f1=(2*p*r)/(p+r)
        print("Confusion Matrix : ")
        print(cm)
        print("\n")
        print(f"Precision : {p}")
        print(f"Recall : {r}")
        print(f"F1 SCORE : {f1}")
        
    def accuracy(y_test,y_test_obs):
        '''
        Accuracy is calculated as the (number of correct predictions)/ (total no. of predictions)
        '''
        incorrect=0
        for i in range(len(y_test)):
            if y_test[i]!=y_test_obs[i]:
                incorrect+=1
        return ((len(y_test)-incorrect)/len(y_test)*100)

#data is loaded from LBW_Dataset.csv file
data=pd.read_csv("LBW_Dataset.csv") 
'''
We pass the dataframe to function called clean. 
'''
data=clean(data).iloc[:,:] #We can use pandas for cleaning
'''
All columns except the last one is included in X
'''
X=data.iloc[:,:-1] #All columns except target column i.e predicted value Y
'''
The last column is Y
'''
Y=data.iloc[:,-1] 
'''
Train test split using sklearn.
70% train set
30% test set
'''
X_train, X_test, Y_train, Y_test= train_test_split(X, Y, test_size=0.3, random_state=1)
Y_train=Y_train.values.reshape(Y_train.shape[0],1)
Y_test=Y_test.values.reshape(Y_test.shape[0],1)

'''
Initialise NN class
'''
Neural_Network=NN(X_train,Y_train)
'''
Mapping X to Y
'''
Neural_Network.fit(X_train,Y_train)
#Getting the predictions of the training set. To find train set accuracy
out_train=Neural_Network.predict(X_train)
#Getting the predictions of the test set. To find the test set accuracy
out=Neural_Network.predict(X_test)
#Calculating performance on training set
print("\nTraining Set Statistics:")
NN.CM(Y_train,out_train)
train_acc=NN.accuracy(Y_train,out_train)
print("Training Set Accuracy= ",train_acc)
#Calculating performance on test set
print("\nTest Set Statistics:")
NN.CM(Y_test,out)
test_acc=NN.accuracy(Y_test,out)
print("Testing Set Accuracy= ",test_acc)

'''
Training Accuracy obtained: 88.0579 %
Testing Accuracy obtained: 86.2069 %
'''