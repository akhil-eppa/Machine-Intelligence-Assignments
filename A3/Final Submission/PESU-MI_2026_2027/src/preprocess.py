
import pandas as pd
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

#Read the uncleaned CSV file. LBW_Dataset has a lot of missing values
data=pd.read_csv("LBW_Dataset.csv")
'''
The unclean dataframe is passed to a function called clean that
fills up all the missing values with mode or mean accordinly.
Also the columns with continuous values are scaled in the range of 0 to 10.
'''
data=clean(data).iloc[:,:]
'''
Cleaned dataframe is stored as a CSV file named LBW_Dataset_Cleaned.csv
'''
data.to_csv(r'LBW_Dataset_Cleaned.csv', index=False) 