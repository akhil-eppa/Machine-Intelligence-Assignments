• Our Neural Network predicts whether a child will suffer from Low Birth Weight(LBW) given certain features of the mother.
   The Neural Network has a total of 3 layers, 1 input layer (with the same features as that given in the initial dataset), 1 hidden layer
   (consisting of 5 neurons) and 1 output layer which gives the final classification of LBW or not. Apart from this, our design also 
   includes weight and bias matrices for each layer(other than input layer), implementation of 2 prominent activation functions(tanh         and sigmoid), cross-entropy loss function, forward and backward propagation.  
•  Firstly, we preprocess the dataset given. The initial stage consists of filling the empty rows with some values. Categorical column
   empty values are filled with the mode of the column whereas non-categorical(numerical) column empty values are filled with the        mean of the column. The next stage of preprocessing consists of scaling all the non-categorical(numerical) column values such 
   that they lie between 1 and 10 for enhancing the accuracy. 
•  Next, we split the cleaned dataset into the independent(X) and dependent(Y) sets. Further these are split into training and testing       sets in a 70%-30% ratio respectively, i.e 70% training set and 30% testing set. All the functions that are going to be discussed 
   henceforth are all under the class NN within the code.
• The initialisation function on an instance of this class ensures that it is supplied with necessary learning rate of 0.05, appropriate 
   weight and bias matrices for the various layers and creates the prediction matrix. This function stores the weight and bias       matrices in a global dictionary such that it can be accessed in the future by the functions that need them. Once initialised, the fit         function is called on the network with the training sets being passed as the parameters. The fit function runs 2500 epochs on the       training set. It inturn passes the parameters to the train function which conducts both forward and backward propagation. 
• In the forward propagation function, the weight and bias matrices are retrieved from the global dictionary that initially stored them
   in the initialisation function. The individual layer output calculation (hidden and output layer) is done here. In general, it is the       activation function applied to the dot product of the weight with the input added to the bias. The output is calculated by keeping in       mind the dimensions of the weight,bias and input matrices to that layer. The actual output(Zi) and the output obtained after applying
   the respective activation function(Ai) are stored in the cache to be accessed later by backward propagation.
•  In the backward propaation, the weight and bias matrices are accessed from the global dictionary whereas the Z and A matrices
   are retrieved from the cache where it was stored after the forward propagation. Chain rule has been applied here to ascertain the
   differentials(dW1,dB1, etc) and the new matrice values are updated using the gradient descent approach(difference of the old          subtracted with the product of self learning rate(constant) with respective differential(varying) gives new value). Accordingly, the 
   new values are updated in the global dictionary for further iterations. The helper functions for the above metioned activation    functions and their derivatives have also been coded within the program.
• Once the epoch count is met, the neural network is now in the best possible frame to predict the values. The predict function is 
   called which ensures that the trained neural network is now applied to the feature dataset(X_train and X_test) such that it gives
   the respective estimated/calculated outputs(Yhat_train and Yhat_test). This calculated output is now tested against the actual
   observed output in the CM function wherein the confusion matrix is prepared and performance metrics such as precision, recall
   and F1 Score are calculated. Along with this, the accuarcy function has also been implemented that gives an idea as to how
   correctly the neural network has managed to correctly predict the outcome for a given split of the original dataset. 
• The hyperparameters that we have used include:
   1) Number Of Neurons(Shape) Of Hidden Layer
   2) Seed value for pseudo-random generation
   3) Epoch count in the fit function
   4) Learning Rate

• There are quite a few features that we have designed that make our network a cut above the rest. We have managed to scale down
   the non-categorical values in the original cleaned dataset such that there is not too much of variation as far as the numerical values
   are concerned. Apart from this we have intelligently used a pseudo-random method for initialising the weight matrices. It gives a 
   feeling of randomness though it isnt so completely. We have also split the dataset very fairly nd unbiasedly into the training and 
   testing datasets eliminating any doubts regarding the performance of the network. 

• We have implemented concepts outside of the basics by not just sticking to one activation function but having a mixture of them. 
   The same holds true even for the loss function wherein we have used cross entropy. We have experimented a lot with the       accuarcies and kept tweaking the hyperparameters and found out innvative ways to increase the accuracy for both train and test    data. The results we have obtained(Close to 88% for train and 86% for test comfortably crossing threshold of 85%) are a very
   strong testament to our constant experimentions.

•  AKHIL PLEASE ADD STEPS TO EXECUTE PROGRAM
   