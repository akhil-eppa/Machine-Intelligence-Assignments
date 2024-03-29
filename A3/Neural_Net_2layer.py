'''
Design of a Neural Network from scratch

*************<IMP>*************
Mention hyperparameters used and describe functionality in detail in this space
- carries 1 mark
'''
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split #Allowed for splitting dataset
def clean(x): 
    #We round mean values to appropriate number of decimal places
    x['Age']=x['Age'].fillna(round(x['Age'].mean()))
    x['Weight']=x['Weight'].fillna(round(x['Weight'].mean()))
    x['Delivery phase']=x['Delivery phase'].fillna(x['Delivery phase'].mode()[0])
    x['HB']=x['HB'].fillna(round(x['HB'].mean(),1))
    x['IFA']=x['IFA'].fillna(x['IFA'].mode()[0])
    x['BP']=x['BP'].fillna(round(x['BP'].mean(),3))
    x['Education']=x['Education'].fillna(x['Education'].mode()[0])
    x['Residence']=x['Residence'].fillna(x['Residence'].mode()[0])
    return x
    
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
    def __init__(self,X,Y):
        self.input = X
        self.weights1 = np.random.rand(self.input.shape[1],4) #Since we have 4 nodes in first hidden layer it will return a 9*4(i.e 3d) array with random weights
        self.weights2 = np.random.rand(4,1) #Second hidden layer also has 4 nodes and it is the last hidden layer in this case
        #self.weights3 = np.random.rand(4,1)#From second hidden layer to output layer
        self.y = Y
        self.output = np.zeros(Y.shape) #Initialisation
    ''' X and Y are dataframes '''

       
    #Forward Propogation
    def feedforward(self):
        self.layer1=sigmoid(np.dot(self.input, self.weights1)) #Returns the dot product of the two arrays
        self.layer2=sigmoid(np.dot(self.layer1, self.weights2))
        #self.layer3=sigmoid(np.dot(self.layer2, self.weights3))
        return self.layer2
    
    #Backward Propogation
    def backprop(self):
	# application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        d_weights2 = np.dot(self.layer1.T, (self.y -self.output)*sigmoid_derivative(self.output))
        d_weights1 = np.dot(self.input.T, np.dot((self.y -self.output)*sigmoid_derivative(self.output), self.weights2.T)*sigmoid_derivative(self.layer1))
    
        self.weights1 += d_weights1
        self.weights2 += d_weights2
        '''
        d_weights3 = np.dot(self.layer2.T, (2*(self.y - self.output) * sigmoid_derivative(self.output)))
        d_weights2 = np.dot(self.layer1.T,  (np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.weights3.T) * sigmoid_derivative(self.layer2)))
        d_weights1 = np.dot(self.input.T, (np.dot(np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.weights3.T) * sigmoid_derivative(self.layer2)),self.weights2.T))
        self.weights1+=d_weights1
        self.weights2+=d_weights2
        self.weights3+=d_weights3
        '''
    def train(self,X,Y):
        self.output=self.feedforward()
        self.backprop()
    def fit(self,X,Y):
        for i in range(1000): #Why 1000? and not just len?
            self.train(X,Y)
        '''
        Function that trains the neural network by taking x_train and y_train samples as input
        '''
	
    def predict(self,X):
        self.input=X
        yhat=self.feedforward()
        
        """
        The predict function performs a simple feed forward of weights
        and outputs yhat values 
        yhat is a list of the predicted value for df X
        """
        return yhat
    """def __init__(self,X,Y):
        self.input = X
        self.weights1 = np.random.rand(self.input.shape[1],4)#We have 4 nodes in first hidden layer
        self.weights2 = np.random.rand(4,4)#Second hidden layer also has 4 nodes
        self.weights3 = np.random.rand(4,1)#From second hidden layer to output layer #Defined at the beginning of the class
        self.y = Y
        self.output = np.zeros(Y.shape) """

    def CM(self,y_test,y_test_obs):
        '''
        Prints confusion matrix 
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
data=pd.read_csv("LBW_Dataset.csv") 
data=clean(data) #We can use pandas for cleaning
data.to_csv(r'LBW_Dataset_Cleaned.csv', index=False) #We are using this only for testing purpose. Need to remove when submitting final version
data=pd.read_csv("LBW_Dataset_Cleaned.csv") # Subsequently need to remove this as well
X=data.iloc[:,:-1] # All columns except target column which will be Y i.e predicted
                   # The first : takes all rows, the second : after the , takes all columns upto(excluding) column -1 which is the last column
Y=data.iloc[:,-1]  # All rows and only the last column
X_train, X_test, Y_train, Y_test= train_test_split(X, Y, test_size=0.3, random_state=0) #Method allowed as it is for splitting dataset0
#print(X_train.shape)
Y_train=Y_train.values.reshape(Y_train.shape[0],1) #Reshape it as a column vector
Y_test=Y_test.values.reshape(Y_test.shape[0],1) #Reshape it as a column vector

Neural_Network=NN(X_train,Y_train) #Initialisation Step
Neural_Network.fit(X_train,Y_train)
temp_out=Neural_Network.feedforward()
out=Neural_Network.predict(X_test)
NN.CM(Y_test,out)