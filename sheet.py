import random
import numpy as np
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

'The random module provides functions to generate random numbers, shuffle data, or make random selections.'

'numpy is a Python library for numerical computations. '

'PyTorch is an open-source deep learning framework. It supports tensor computations and GPU acceleration. -->'
'Used to define, train, and evaluate neural networks. It provides tensors (multi-dimensional arrays) and operations '
'for deep learning tasks.'

'pandas is a data manipulation and analysis library in Python. '

'The torch.nn module in PyTorch provides tools to define neural network architectures.--->'
'Contains pre-defined layers like Linear, Conv2d, etc.'
'Includes activation functions (ReLU, Sigmoid) and loss functions (CrossEntropyLoss, etc.).'


'The torch.optim module provides optimization algorithms like SGD and Adam.-->'
'Optimizers adjust the model weights to minimize the loss during training.'


'This is a tool from scikit-learn for splitting data into training and testing sets while maintaining the same class distribution.--->'
'Ensures that each subset has an equal proportion of classes, reducing bias in classification problems.'


'A preprocessor from scikit-learn used to standardize features by removing the mean and scaling to unit variance.--->'
'Normalizes input data to improve model convergence during training.'


'classification_report: Provides precision, recall, F1-score, and support for classification models.'
'confusion_matrix: Summarizes the performance of a classification model by showing the true positives, true negatives, false positives, and false negatives.'
'Used for evaluating the performance of a trained model on classification tasks.'


# Set random seeds for reproducibility
random.seed(120)
np.random.seed(120)
torch.manual_seed(120)
torch.backends.cudnn.deterministic = True  
torch.backends.cudnn.benchmark = False  


# Load the dataset
file_path = "C:\\Users\\Egen AI\\Documents\\ps\\combined_music_dataset.csv"
music_dataset = pd.read_csv(file_path)

# Step 1: Separate features and labels
X = music_dataset.drop(columns=['filename', 'genre', 'duration(in sec)'])  # Features (MFCCs)
y = music_dataset['genre']  # Labels (genres)

'.drop(): Removes specific columns from the dataset.'
'Purpose: X now contains only the input features (numerical data from MFCCs, spectral features, etc.) '
'used to train the neural network.'
'Purpose: y contains the categories or classes that the model will predict.'


# Step 2: Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X) 

'''StandardScaler: This is a class from scikit-learn used to normalize or standardize data.

Purpose of Standardization:

Adjusts the features in X to have a mean of 0 and a standard deviation of 1.
Ensures that all features contribute equally to the model and prevents features
with largernumerical ranges from dominating.
Why it’s important: Neural networks perform better when the data is standardized,
especially when using activation functions like ReLU.

scaler: An instance of StandardScaler is created. This object will handle the normalization process.


scaler.fit_transform(X):
fit(): Calculates the mean and standard deviation of each feature in X.
transform(): Applies the normalization formula to each feature
X_scaled = (X -μ)/σ (sigma)
X = Original feature value.
μ = Mean of the feature 
σ = Standard Deviation of the feature

X_scaled: A NumPy array containing the standardized features.

Each column (feature) now has a mean of 0 and a standard deviation of 1.

scaler: StandardScaler object created for normalization.
X_scaled: Standardized version of the feature matrix X, ready for training.
'''


# Step 3: Split the data into training and test sets (70% training, 30% testing) with stratified sampling
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
for train_index, test_index in sss.split(X_scaled, y):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]


'''
Why Choose StratifiedShuffleSplit?

You have an imbalanced dataset.
You need to maintain the same class proportions in training and testing sets.
You require multiple splits for cross-validation or other repeated experiments.


Why Choose train_test_split?

Your dataset is balanced, so class distribution is not a concern.
You only need a simple split for training and testing.
You enable stratification by setting stratify=labels to mimic the behavior of StratifiedShuffleSplit.
'''

'''
1-line
StratifiedShuffleSplit: A splitting strategy from scikit-learn designed for classification tasks.
Parameters:

n_splits=1: Only one split will be created (one training and one test set).
test_size=0.3: 30% of the data will be reserved for testing, and 70% will be used for training.
random_state=42: Ensures reproducibility so the split will be the same every time the code is run.
Purpose: The sss object will generate indices for training and
test sets while preserving the class distribution.

2-line
sss.split(X_scaled, y):

X_scaled: The normalized features.
y: The target labels (genres).
The method returns two sets of indices:
train_index: Indices for the training set.
test_index: Indices for the testing set.
for train_index, test_index: The loop extracts the indices
for training and test sets (since n_splits=1, it will run only once).

3-line
X_scaled[train_index]: Selects the rows from X_scaled corresponding to the training indices.
X_scaled[test_index]: Selects the rows from X_scaled corresponding to the testing indices.
X_train: Features for the training set (70% of the data).
X_test: Features for the test set (30% of the data).

4-line
y.iloc[train_index]: Selects the target labels corresponding to the training indices.
y.iloc[test_index]: Selects the target labels corresponding to the testing indices.
y_train: Labels for the training set.
y_test: Labels for the testing set.

'''

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train)
X_test_tensor = torch.FloatTensor(X_test)
y_train_tensor = torch.LongTensor(y_train.factorize()[0])  # Convert categories to integers
y_test_tensor = torch.LongTensor(y_test.factorize()[0])

'''
1-line
torch.FloatTensor(): Converts the X_train NumPy array into a PyTorch FloatTensor.
Why FloatTensor?
Neural networks expect input data to be in float32 format for computation.
X_train_tensor: A PyTorch tensor containing the training features, 
now ready to be fed into the neural network.

2-line
Similar to the previous line, this converts X_test (testing features)
into a PyTorch FloatTensor.
X_test_tensor: A PyTorch tensor containing the testing features.
o
3-line
y_train.factorize(): This method converts categorical labels (e.g., 'Classical', 'Rock') into integers.
Returns a tuple:
[0]: An array of integer-encoded labels (e.g., [0, 1, 2, 3]).
[1]: A list of unique class labels (e.g., ['Classical', 'Rock', 'Jazz', 'Pop']).
torch.LongTensor(): Converts the array of integer labels into a PyTorch LongTensor.
Why LongTensor?
CrossEntropyLoss (used later) requires the target labels to be in integer (LongTensor) format,
not float or one-hot encoded.
y_train_tensor: A PyTorch tensor containing the integer-encoded training labels.

4-line
Similar to the previous line, this converts y_test into a PyTorch 
LongTensor of integer-encoded labels.
y_test_tensor: A PyTorch tensor containing the integer-encoded testing labels.

X_train_tensor: Training features as FloatTensor.
X_test_tensor: Testing features as FloatTensor.
y_train_tensor: Training labels as LongTensor (integer-encoded).
y_test_tensor: Testing labels as LongTensor (integer-encoded).

'''

'''
1-line
class MusicGenreClassifier: Defines a custom neural network class named MusicGenreClassifier.
nn.Module: The base class for all neural network models in PyTorch.
Provides utilities for building and training neural networks.
Allows seamless integration with PyTorch's autograd and optimization libraries.

2-line
__init__: Constructor method, called when an object of the class is created.
input_size: The number of features (columns) in the input dataset (e.g., MFCCs).
num_classes: The number of output classes (unique music genres).
These parameters make the model dynamic and adaptable to datasets with different input sizes or class counts.

3-line
super(): Calls the constructor of the parent class (nn.Module).
Ensures that the PyTorch module initialization happens correctly.
Required when subclassing nn.Module

4-line
nn.Linear(input_size, 1024):
A fully connected (dense) layer with:
Input size: input_size (number of features from the dataset).
Output size: 1024 neurons.
Why 1024 neurons?
A large number of neurons help the network learn complex relationships in the data.
The first layer processes raw input features and passes them to the next layer.

5-line 
A fully connected layer with:
Input size: 1024 (output from the previous layer).
Output size: 512 neurons.
Reduces the number of neurons while retaining important features.

6-line
A fully connected layer with:
Input size: 512 neurons.
Output size: 256 neurons.
Further refines the feature representation.

7-line
A fully connected layer with:
Input size: 256 neurons.
Output size: 128 neurons.
This layer continues to condense the learned features.

8-line 
Output Layer: A fully connected layer with:
Input size: 128 neurons.
Output size: num_classes (number of music genres).
This layer produces the final predictions for each class.
No activation function is applied here because 
CrossEntropyLoss expects raw logits as inputs.

9-line
nn.ReLU(): Applies the ReLU (Rectified Linear Unit) activation function.
Formula: 
f(x)=max(0,x)
Why ReLU?
Reduces the likelihood of the vanishing gradient problem.
Computationally efficient.
Applied after each fully connected layer (except the final output layer).


'''

# Step 4: Define the neural network model
class MusicGenreClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MusicGenreClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 1024)  # Increased neurons in the first layer
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, num_classes)  # Output layer
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)  # Added dropout for regularization
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.relu(self.fc4(x))
        x = self.fc5(x)  # No activation here, softmax in loss function
        return x

'''
What It Does:
The forward method defines how the input data flows through the layers of the neural network during the forward pass.
In this method, the input tensor x passes through 5 fully connected layers (fc1 to fc5),
with activation functions and dropout layers applied in between.

Fully Connected Layer: Transforms the input into a new representation with 1024 features.
ReLU Activation: Applies the ReLU (Rectified Linear Unit) function to introduce non-linearity.
Dropout: Randomly disables neurons to prevent overfitting.

Fully Connected Layer: Further reduces dimensions to 512.
ReLU Activation: Non-linearity applied again.
Dropout: Prevents overfitting.

Fully Connected Layer: Reduces dimensions to 256.
ReLU Activation: Applies non-linearity.
Dropout: Further reduces overfitting risk.

Fully Connected Layer: Reduces dimensions to 128.
ReLU Activation: Non-linearity applied.
No Dropout: No dropout is applied here.

Fully Connected Layer: Maps the final output to the number of classes (genres).
No Activation Function: Outputs raw logits. The CrossEntropyLoss function expects raw logits, so no softmax is applied here.

ReLU Activation: Prevents vanishing gradient issues, ensuring neurons remain active.
Dropout Layers: Combat overfitting by randomly turning off neurons during training.
No Activation in Last Layer: Because nn.CrossEntropyLoss combines softmax + cross-entropy, applying softmax here would be redundant.
'''

'''
forward: This method defines the forward pass of the neural network.
x: Represents the input tensor (a batch of feature data).

self.fc1(x):
  
The input tensor x passes through the first fully connected layer (fc1).
The layer transforms the input using a linear transformation.
self.relu(...):

The ReLU activation function is applied to the output of fc1.
Ensures non-linearity in the data, enabling the model to learn complex patterns.

self.dropout(x):
Applies dropout regularization to the output of the first layer.
Randomly sets 50% of neurons to zero during training to prevent overfitting.


Dropout is like randomly turning off some neurons in a layer during training.
This helps the model not depend too much on specific neurons and reduces overfitting.

Many experts avoid using Dropout in the last hidden layer 
because it can reduce the accuracy of the final predictions.
'''

# Step 5: Initialize the model, loss function, and optimizer
input_size = X_train.shape[1]  # Number of features
num_classes = len(y.unique())   # Number of unique genres
model = MusicGenreClassifier(input_size, num_classes)

'''
X_train.shape[1]

X_train is your training data (a 2D matrix with samples and features).
shape[1] gives the number of features per sample (e.g., if you're using 13 MFCC features, input_size will be 13).
len(y.unique())

y contains the labels (genres) for your data samples.
unique() finds distinct labels (e.g., Jazz, Rock, Pop, Classical).
len() counts how many unique genres there are (e.g., 4).
MusicGenreClassifier(input_size, num_classes)

This creates an instance of your neural network class, setting the input size and output size.
The model knows how many features to expect in the input and how many output classes to predict.
Why important?
The model architecture must match the data dimensions; otherwise, it won’t train properly.

'''

criterion = nn.CrossEntropyLoss()  # Loss function for multi-class classification
optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Lower learning rate

'''
1. nn.CrossEntropyLoss() – Loss Function
Purpose: Measures how well the model's predicted probability distribution (from Softmax) matches the true labels.
Type: Used specifically for multi-class classification tasks.
Input:
Logits (raw outputs) from the final layer (not softmax probabilities).
Target labels (as integers, e.g., 0, 1, 2, representing class indices).
Working:
Applies nn.LogSoftmax to logits.
Computes the Negative Log-Likelihood Loss (NLLLoss).

Why CrossEntropyLoss?

It combines Softmax + Negative Log-Likelihood Loss (NLLLoss) in one step.
Handles multi-class classification effectively.

----------------------------------------------------------------------------------------------

2. optim.Adam – Optimizer
Purpose: Adjusts the model's parameters (weights and biases) to minimize the loss function.
Type: Adaptive learning rate optimizer (Adaptive Moment Estimation).
Key Parameters:

model.parameters(): Passes all learnable parameters of the model (weights & biases) to the optimizer.
lr=0.0001: Learning rate (controls the step size for updating parameters). A lower value ensures small, stable updates, preventing overshooting optimal values.
Why Adam?

Combines momentum (faster convergence) and adaptive learning rates for each parameter.
Works well for complex models and large datasets.

'''

# Step 6: Train the model
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:  # Print every 10 epochs
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

''''
1. num_epochs = 50
Definition: The total number of times the model will see the entire training dataset.
Each epoch represents one complete pass over the dataset.

2. for epoch in range(num_epochs):
A loop over epochs to repeatedly train the model.

3. model.train()
Purpose: Puts the model into training mode.
Enables behaviors like:
Dropout layers are active.
Batch Normalization layers update their statistics.
Ensures the model behaves correctly during training.

4. optimizer.zero_grad()
Purpose: Clears previously accumulated gradients from the last iteration.
Why? Gradients accumulate by default in PyTorch (loss.backward() adds gradients).
Prevents gradients from being added repeatedly across batches.

5. outputs = model(X_train_tensor)
Purpose: Perform a forward pass through the model using X_train_tensor.
Returns the raw outputs (logits) from the final layer.

6. loss = criterion(outputs, y_train_tensor)
Purpose: Calculate the loss using the predicted outputs (outputs) and actual labels (y_train_tensor).
criterion is CrossEntropyLoss.
The loss measures how far the predicted outputs are from the true labels.

7. loss.backward()
Purpose: Perform backward propagation to calculate the gradients of the loss w.r.t. model parameters.
Each parameter’s gradient is stored in param.grad.

8. optimizer.step()
Purpose: Update the model's parameters using the computed gradients.
Controlled by the optimizer (Adam) and learning rate (lr=0.0001).

9. Epoch Logging:
Every 10 epochs, the code prints the current epoch number and loss value.
loss.item() extracts the numerical loss value from the PyTorch tensor.

'''

# Step 7: Evaluate the model
model.eval()  # Set the model to evaluation mode (turns off dropout)
with torch.no_grad():
    y_pred = model(X_test_tensor)
    _, predicted = torch.max(y_pred, 1)
    
'''
1. model.eval()
Purpose: Switches the model to evaluation mode.
Why? Certain layers like Dropout and Batch Normalization behave differently during training and evaluation:
Dropout: Turns off (all neurons remain active).
Batch Normalization: Uses learned statistics (mean and variance) instead of batch statistics.
Ensures consistent behavior when making predictions.

2. with torch.no_grad():
Purpose: Disables gradient computation during evaluation.
Why?
Saves memory and computation since gradients aren’t needed during evaluation.
Prevents PyTorch from tracking operations on tensors, improving speed.

3. y_pred = model(X_test_tensor)
Purpose: Pass the test data (X_test_tensor) through the model to get predictions.
What happens here?
The model performs a forward pass on the test data.
Returns raw scores (logits) from the final layer.

_, predicted = torch.max(y_pred, 1)
Purpose: Extracts the class predictions from the model's output.
torch.max(y_pred, 1) returns:
First value (_): The maximum value (logit) along dim=1 (for each row).
Second value (predicted): The index of the maximum value, which corresponds to the predicted class label.


'''

# Step 8: Print evaluation metrics
print("Confusion Matrix:")
print(confusion_matrix(y_test_tensor, predicted))

print("\nClassification Report:")
print(classification_report(y_test_tensor, predicted, target_names=y.unique()))