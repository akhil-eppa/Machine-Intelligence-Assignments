'''
Design of a Neural Network from scratch

*************<IMP>*************
Mention hyperparameters used and describe functionality in detail in this space
- carries 1 mark
'''
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split #Allowed for splitting dataset
parameters=dict()
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
    #x=np.array([x],dtype=np.float128)
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
        #n_x=9
        #n_h=4
        #n_y=1
        np.random.seed(2) #Pesudo-random
        self.input = X
        self.learning_rate=0.05
        self.weights1 = np.random.randn(4,self.input.shape[1])*0.01#We have 4 nodes in first hidden layer, 4X9 Multiplying by 0.01 to convert into decimals
        # As a thumb rule, the weight matrices must have the follwoing dimensions:
        # The number of rows must be equal to the number of neurons in the current layer.The number of columns must be equal to the number of neurons in previous layer
        # As a thumb rule, the bias matrices must have the follwoing dimensions:
        # The number of rows must be equal to the number of neurons in the current layer and it must be a column vector
        # The same rule applies for back propagation as well for dW and dB
        self.weights2 = np.random.randn(Y.shape[1],4)*0.01#Second hidden layer also has 4 nodes
        #self.weights3 = np.random.rand(4,1)#From second hidden layer to output layer
        self.bias1=np.zeros((4,1))
        self.bias2=np.zeros((Y.shape[1],1))
        self.y = Y
        self.output = np.zeros(Y.shape)
        self.parameters = {"W1": self.weights1,
                  "b1": self.bias1,
                  "W2": self.weights2,
                  "b2": self.bias2}
    ''' X and Y are dataframes '''

       
    #Forward Propogation
    def feedforward(self):
        W1 = self.parameters["W1"]#4X9
        b1 = self.parameters["b1"]#4X1
        W2 = self.parameters["W2"]#1X4
        b2 = self.parameters["b2"]#1X1
        Z1 = np.dot(W1,self.input.T)+b1#4X67
        A1 = np.tanh(Z1)#4X67
        Z2 = np.dot(W2,A1)+b2#1X67
        A2 = sigmoid(Z2)#1X67
        self.cache={"Z1":Z1,"A1":A1,"Z2":Z2,"A2":A2}
        #self.layer1=sigmoid(np.dot(self.input, self.weights1))
        #self.layer2=sigmoid(np.dot(self.layer1, self.weights2))
        #self.layer3=sigmoid(np.dot(self.layer2, self.weights3))
        return A2
    
    #Backward Propogation
    def backprop(self):
	# application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        m=self.input.shape[0] # Number of rows i.e 67
        W1=self.parameters["W1"]
        W2=self.parameters["W2"]
        b1=self.parameters["b1"]
        b2=self.parameters["b2"]
        A1=self.cache["A1"]
        A2=self.cache["A2"]
        dZ2=A2-self.y.T#1X67
        dW2=(1/m)*(np.dot(dZ2,A1.T))#1X4
        db2 = (1/m)*np.sum(dZ2,axis=1,keepdims=True)
        dZ1 = np.dot(W2.T,dZ2)*(1-np.power(A1,2))#4X67
        dW1 = (1/m)*(np.dot(dZ1,self.input))
        db1 = (1/m)*np.sum(dZ1,axis=1,keepdims=True)
        W1 = W1-(self.learning_rate*dW1)
        b1 = b1-(self.learning_rate*db1)
        W2 = W2-(self.learning_rate*dW2)
        b2 = b2-(self.learning_rate*db2)
        self.parameters={"W1":W1,"W2":W2,"b1":b1,"b2":b2}
        #d_weights2 = np.dot(self.layer1.T, (self.y -self.output)*sigmoid_derivative(self.output))
        #d_weights1 = np.dot(self.input.T, np.dot((self.y -self.output)*sigmoid_derivative(self.output), self.weights2.T)*sigmoid_derivative(self.layer1))
    
        #self.weights1 += d_weights1
        #self.weights2 += d_weights2
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
        for i in range(100): # 100 times forward and backward propagation will take place
            self.train(X,Y)
        '''
        Function that trains the neural network by taking x_train and y_train samples as input
        '''
	
    def predict(self,X):
        W1 = self.parameters["W1"]#4X9
        b1 = self.parameters["b1"]#4X1
        W2 = self.parameters["W2"]#1X4
        b2 = self.parameters["b2"]#1X1
        Z1 = np.dot(W1,X.T)+b1#4X67
        A1 = np.tanh(Z1)#4X67
        Z2 = np.dot(W2,A1)+b2#1X67
        A2 = sigmoid(Z2)#1X67
        yhat=A2.T
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
#data.to_csv(r'LBW_Dataset_Cleaned.csv', index=False) #We are using this only for testing purpose. Need to remove when submitting final version
#data=pd.read_csv("LBW_Dataset_Cleaned.csv")
X=data.iloc[:,:-1] #All columns except target column which will be Y i.e predicted
Y=data.iloc[:,-1] 
X_train, X_test, Y_train, Y_test= train_test_split(X, Y, test_size=0.3, random_state=0)
#print(X_train.shape)
Y_train=Y_train.values.reshape(Y_train.shape[0],1)
Y_test=Y_test.values.reshape(Y_test.shape[0],1)

Neural_Network=NN(X_train,Y_train)
Neural_Network.fit(X_train,Y_train)
#temp_out=Neural_Network.feedforward()
out=Neural_Network.predict(X_test)
#for i in temp_out:
#print(i)
#out=Neural_Network.predict(X_test)
NN.CM(Y_test,out)
