'''
Assume df is a pandas dataframe object of the dataset given
'''
import numpy as np
import pandas as pd
import random

'''Calculate the entropy of the enitre dataset'''
	#input:pandas_dataframe
	#output:int/float/double/large

def get_entropy_of_dataset(df):
    categories=df.keys()[-1]
    entropy = 0
    classes=df[categories].unique()
    for i in classes:
        prob=df[categories].value_counts()[i]/(len(df[categories]))
        entropy=entropy + (-prob*np.log2(prob))
    return entropy



'''Return entropy of the attribute provided as parameter'''
	#input:pandas_dataframe,str   {i.e the column name ,ex: Temperature in the Play tennis dataset}
	#output:int/float/double/large
def get_entropy_of_attribute(df,attribute):
    categories=df.keys()[-1]
    #targets gets the unique predictions like yes or no
    targets=df[categories].unique() 
    #values gets the unique variables in an attribute. Like for example weather can have sunny, cold or rainy
    values=df[attribute].unique()
    entropy_of_attribute = 0
    for val in values:
        #outer loop iterates through various values of an attribute
        #temporary variable to store entropy
        entropy=0
        for i in targets:
            #inner loop iterates through various target variables
            n=len(df[attribute][df[attribute]==val][df[categories]==i])
            total=len(df[attribute][df[attribute]==val])
            prob=n/total
            #if condition is used to avoid cases of log(0)
            if prob==0:
                entropy+=0
            else:
                entropy= entropy + (-prob*np.log2(prob))
        prob2=total/len(df[categories])
        entropy_of_attribute= entropy_of_attribute + (-prob2*entropy)
    return abs(entropy_of_attribute)



'''Return Information Gain of the attribute provided as parameter'''
	#input:int/float/double/large,int/float/double/large
	#output:int/float/double/large
def get_information_gain(df,attribute):
    #Information Gain(T,X)=Entropy(T) - Entropy(T,X)
	information_gain = get_entropy_of_dataset(df)-get_entropy_of_attribute(df,attribute)
	return information_gain



''' Returns Attribute with highest info gain'''  
	#input: pandas_dataframe
	#output: ({dict},'str')     
def get_selected_attribute(df):
    attributes=df.keys()
    #we need to remove target column from list of attributes
    attributes=attributes[:-1]
    information_gains={}
    for i in attributes:
        #attributes and their information gains are stored as key value pairs in a dictionary
        information_gains[i]=get_information_gain(df,i)
    #get the key whose value is the max. 
    selected_column=max(information_gains, key=information_gains.get)

    '''
    Return a tuple with the first element as a dictionary which has IG of all columns 
    and the second element as a string with the name of the column selected_column
    example : ({'A':0.123,'B':0.768,'C':1.23} , 'C')
    '''
    return (information_gains,selected_column)



'''
------- TEST CASES --------
How to run sample test cases ?

Simply run the file DT_SampleTestCase.py
Follow convention and do not change any file / function names

'''