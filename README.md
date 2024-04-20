# Multinomial_Logistic_Regression-and-MultiLayer_Perceptron_Classification_and_Regression
# 1 Multinomial Logistic Regression
Implement a Multinomial logistic regression model from scratch using numpy
and pandas. You have to train this model on Wine Quality Dataset to classify
a wine’s quality based on the values of its various contents.
## 1.1 Dataset Analysis and Preprocessing 
1. Describe the dataset using mean, standard deviation, min, and max values
for all attributes.
2. Draw a graph that shows the distribution of the various labels across the
entire dataset. You are allowed to use standard libraries like Matplotlib.
3. Partition the dataset into train, validation, and test sets. You can use
sklearn for this.
4. Normalise and standarize the data. Make sure to handle the missing or
inconsistent data values if necessary. You can use sklearn for this.

## 1.2 Model Building from Scratch 
1. Create a Multinomial Logistic Regression model from scratch and Use
cross entropy loss as loss function and Gradient descent as the optimization
algorithm (write seperate methods for these).
2. Train the model, use sklearn classification report and print metrics on the
validation set while training. Also, report loss and accuracy on train set.
1.3 Hyperparameter Tuning and Evaluation [15 marks]
1. Use your validation set and W&B logging to fine-tune the hyperparameters
( learning rate , epochs) for optimal results.
2. Evaluate your model on test dataset and print sklearn classification report.
   
# 2 Multi Layer Perceptron Classification
In this part, you are required to implement MLP classification from scratch
using numpy, pandas and experiment with various activation functions and optimization techniques, evaluate the model’s performance, and draw comparisons
with the previously implemented multinomial logistic regression.
Use the same dataset as Task 1
## 2.1 Model Building from Scratch 
Build an MLP classifier class with the following specifications:
1. Create a class where you can modify and access the learning rate, activation function, optimisers, number of hidden layers and neurons.
2. Implement methods for forward propagation, backpropagation, and training.
3. Different activation functions introduce non-linearity to the model and
affect the learning process. Implement the Sigmoid, Tanh, and ReLU
activation functions and make them easily interchangeable within your
MLP framework.
4. Optimization techniques dictate how the neural network updates its weights
during training. Implement methods for the Stochastic Gradient Descent
(SGD), Batch Gradient Descent, and Mini-Batch Gradient Descent algorithms from scratch, ensuring that they can be employed within your MLP
architecture.

## 2.2 Model Training & Hyperparameter Tuning using W&B

Effective tuning can vastly improve a model’s performance. Integrate Weights
& Biases (W&B) to log and track your model’s metrics. Using W&B and your
validation set, experiment with hyperparameters such as learning rate, epochs,
hidden layer neurons, activation functions, and optimization techniques. You
have to use W&B for loss tracking during training and to log effects of different
activation functions and optimizers on model performance.
1. Log your scores - loss and accuracy on validation set and train set using
W&B.
2. Report metrics: accuracy, f-1 score, precision, and recall. You are allowed
to use sklearn metrics for this part.
3. You have to report the scores(ordered) for all the combinations of :
• Activation functions : sigmoid, tanh and ReLU (implemented from
scratch)
• Optimizers : SGD, batch gradient descent, and mini-batch gradient
descent (implemented from scratch).
4. Tune your model on various hyperparameters, such as learning rate, epochs,
and hidden layer neurons.
• Plot the trend of accuracy scores with change in these hyperparameters.
• Report the parameters for the best model that you get (for the various
values you trained the model on).
• Report the scores mentioned in 2.2.2 for all values of hyperparameters
in a table.
## 2.3 Evaluating Model 
1. Test and print the classification report on the test set. (use sklearn)
2. Compare the results with the results of the logistic regression model.
## 2.4 Multi-Label Classification 
For this part, you will be training and testing your model on Multilabel dataset:
”advertisement.csv” as provided in Assignment 1.
1. Modify your model accordingly to classify multilabel data.
2. (a) Log your scores - loss and accuracy on validation set and train set
using W&B.

(b) Report metrics: accuracy, f-1 score, precision, and recall.
(c) You have to report the scores(ordered) for all the combinations of :
• Activation functions : sigmoid, tanh and ReLU (implemented
from scratch)
• Optimizers : SGD, batch gradient descent and mini-batch gradient descent (implemented from scratch).
(d) Tune your model on various hyperparameters, such as learning rate,
epochs, and hidden layer neurons.
• Plot the trend of accuracy scores with change in these hyperparameters.
• Report the parameters for the best model that you get (for the
various values you trained the model on).
• Report the scores mentioned in Point b for all values of hyperparameters in a table.
3. Evaluate your model on the test set and report accuracy, f1 score, precision, and recall.

# 3 Multilayer Perceptron Regression
In this task, you will implement a Multi-layer Perceptron (MLP) for regression
from scratch, and integrate Weights & Biases (W&B) for tracking and tuning. Using the Boston Housing dataset, you have to predict housing prices
while following standard machine learning practices. In this dataset, the column
MEDV gives the median value of owner-occupied homes in $1000’s.
## 3.1 Data Preprocessing 
1. Describe the dataset using mean, standard deviation, min, and max values
for all attributes.
2. Draw a graph that shows the distribution of the various labels across the
entire dataset. You are allowed to use standard libraries like Matplotlib.
3. Partition the dataset into train, validation, and test sets.
4. Normalise and standarize the data. Make sure to handle the missing or
inconsistent data values if necessary.
## 3.2 MLP Regression Implementation from Scratch
In this part, you are required to implement MLP regression from scratch using
numpy, pandas and experiment with various activation functions and optimization techniques, and evaluate the model’s performance.
5
1. Create a class where you can modify and access the learning rate, activation function, optimisers, number of hidden layers and neurons.
2. Implement methods for forward propagation, backpropagation, and training.
3. Implement the Sigmoid, Tanh, and ReLU activation functions and make
them easily interchangeable within your MLP framework.
4. Implement methods for the Stochastic Gradient Descent (SGD), Batch
Gradient Descent, and Mini-Batch Gradient Descent algorithms from scratch,
ensuring that they can be employed within your MLP architecture.
## 3.3 Model Training & Hyperparameter Tuning using W&B
1. Log your scores - loss (Mean Squared Error) on the validation set using
W&B.
2. Report metrics: MSE, RMSE, R-squared.
3. You have to report the scores(ordered) for all the combinations of :
• Activation functions : sigmoid, tanh and ReLU (implemented from
scratch)
• Optimizers : SGD, batch gradient descent and mini-batch gradient
descent (implemented from scratch).
4. Tune your model on various hyperparameters, such as learning rate, epochs,
and hidden layer neurons.
• Report the parameters for the best model that you get (for the various
values you trained the model on).
• Report the scores mentioned in 3.3.2 for all values of hyperparameters
in a table.
## 3.4 Evaluating Model 
1. Test your model on the test set and report loss score (MSE, RMSE, R-squared)
