'''
Design of a Neural Network from scratch

*************<IMP>*************
Mention hyperparameters used and describe functionality in detail in this space
- carries 1 mark
'''
import numpy as np
import pandas as pd
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

# np.tanh(x) will suffice for tanh activation
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

	''' X and Y are dataframes '''
	
	def fit(self,X,Y):
		'''
		Function that trains the neural network by taking x_train and y_train samples as input
		'''
	
	def predict(self,X):

		"""
		The predict function performs a simple feed forward of weights
		and outputs yhat values 

		yhat is a list of the predicted value for df X
		"""
		
		return yhat

	def CM(y_test,y_test_obs):
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
data=clean(data)
data.to_csv(r'LBW_Dataset_Cleaned.csv', index=False)
