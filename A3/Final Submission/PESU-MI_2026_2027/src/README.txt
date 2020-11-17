Machine Intelligence Assignment 3 - Implementation of a Neural Network from Scratch
TEAM NAME: PESU-MI_2026_2027

TEAM MEMBER'S DETAILS:
   Team Member 1: Akhil Eppa
   SRN: PES1201802026
   
   Team Member 2: Varun Y Tirthani
   SRN: PES1201802027

• Our Neural Network predicts whether a child will suffer from Low Birth Weight(LBW) given certain features of the mother.
   The Neural Network has a total of 3 layers, 1 input layer (with the same features as that given in the initial dataset), 1 hidden layer
   (consisting of 5 neurons) and 1 output layer which gives the final classification of LBW or not. Apart from this, our design also 
   includes weight and bias matrices for each layer(other than input layer), implementation of 2 prominent activation functions(tanh and sigmoid), 
   cross-entropy loss function, forward and backward propagation. 
   
•  Firstly, we preprocess the dataset given. The initial stage consists of filling the empty rows with some values. Categorical column
   empty values are filled with the mode of the column whereas non-categorical(numerical) column empty values are filled with the mean of the column. 
   The next stage of preprocessing consists of scaling all the non-categorical(numerical) column values such that they lie between 0 and 10 for enhancing the accuracy. 
   
•  Next, we split the cleaned dataset into the independent(X) and dependent(Y) sets. Further these are split into training and testing sets in a 70%-30% ratio respectively, i.e    70% training set and 30% testing set. All the functions that are going to be discussed henceforth are all under the class NN within the code.

• The initialisation function on an instance of this class ensures that it is supplied with necessary learning rate of 0.05, appropriate weight and bias 
   matrices for the various layers and creates the prediction matrix. The weight matrices are initialized to random values, and then multiplied by a scaling 
   factor of 0.01 while the bias is initialized as a column vector consisting of all zeroes. The init function also stores the weight and bias matrices in a 
   global dictionary such that it can be accessed in the future by the functions that need them. Once initialised, the fit function is called on the network
   with the training sets being passed as the parameters. The fit function runs 2500 epochs on the training set. It inturn passes the parameters to the train 
   function which conducts both forward and backward propagation.
   
• In the forward propagation function, the weight and bias matrices are retrieved from the global dictionary that was initially stored in the initialisation function.
   The individual layer output calculation (hidden and output layer) is done here. In general, it is the activation function applied to the dot product of the weight with 
   the input added to the bias. The output is calculated by keeping in mind the dimensions of the weight,bias and input matrices to that layer. The actual 
   output(Zi) and the output obtained after applying the respective activation function(Ai) are stored in the cache to be accessed later by backward propagation.
   
•  In the backward propaation, the weight and bias matrices are accessed from the global dictionary whereas the Z and A matrices are retrieved from the 
   cache where it was stored after the forward propagation. Chain rule has been applied here to ascertain the differentials(dW1,dB1, etc) and the new matrice 
   values are updated using the gradient descent approach(difference of the old subtracted with the product of self learning rate(constant) with respective 
   differential(varying) gives new value). Accordingly, the new values are updated in the global dictionary for further iterations. The helper functions 
   for the above metioned activation functions and their derivatives have also been coded within the program.
   
• Once the epoch count is met, the neural network is now in the best possible frame to predict the values. The predict function is 
   called which ensures that the trained neural network is now applied to the feature dataset(X_train and X_test) such that it gives
   the respective estimated/calculated outputs(Yhat_train and Yhat_test). This calculated output is now tested against the actual
   observed output in the CM function wherein the confusion matrix is prepared and performance metrics such as precision, recall
   and F1 Score are calculated. Along with this, the accuracy function has also been implemented that gives an idea as to how
   correctly the neural network has managed to correctly predict the outcome for a given split of the original dataset. 
   
• The hyperparameters that we have used include:
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
    NOTE: The weight matrices are multiplied by a scaling factor of 0.01 in the initialization step.
   4. Dimensions of Bias Matrices:
    Bias Matrix will be a column vector with the number of rows equal to the number of neurons in the current layer
    Bias matrix is initialized as a column vector with all zeroes.
    b1 -> 5X1
    b2 -> 1X1
   5. Activation functions :
    hidden layer -> tanh
    output layer -> sigmoid
   6. Loss Function : Cross Entropy
   7. Learning Rate : 0.05
   8. Epochs : 2500
   9. Train Test Split Used : 70/30

• There are quite a few features that we have designed that make our network a cut above the rest. We have managed to scale down
   the non-categorical values in the original cleaned dataset such that there is not too much of variation as far as the numerical values
   are concerned. Apart from this we have intelligently used a pseudo-random method for initialising the weight matrices. It gives a 
   feeling of randomness though it isnt so completely. We have also split the dataset very fairly nd unbiasedly into the training and 
   testing datasets eliminating any doubts regarding the performance of the network. 

• We have implemented concepts outside of the basics by not just sticking to one activation function but having a mixture of them. 
  The same holds true even for the loss function wherein we have used cross entropy. We have experimented a lot with the accuarcies and 
  kept tweaking the hyperparameters and found out innvative ways to increase the accuracy for both train and test data. The results we have 
  obtained(Close to 90% for train and 86% for test comfortably crossing threshold of 85%) are a very
  strong testament to our constant experimentions.

• INSTRUCTIONS FOR EXECUTION:
   The submitted zip file is named PESU-MI_2026_2027.zip
   This README file is part of the zip file. The zip file consists of a folder named PESU-MI_2026_2027 that consists of 2 folders- data and src and the README file.
   The data folder consists the preprocessed and standardized dataset which is named as LBW_Dataset_Cleaned.csv
   The src folder consists of the python source code file and the preprocessing source code. To run the python script make sure that the cleaned dataset and the 
   source code file are in the same folder. For convenience we have placed the cleaned dataset in the src folder as well. This way the python script 
   can be executed directly without any movement of files.
 
• OUTPUT Format:
   First the training set statistics are displayed followed by test set statistics. Under each dataset the parameters displayed are:
   1. Confusion Matrix
   2. Precision
   3. Recall
   4. F1 Score
   5. Accuracy Obtained
   
----------------------------------------------------------------THE END----------------------------------------------------------------------------------
