import pandas as pd
import torch 
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# Load the dataset
file_path = "C:\\Users\\Egen AI\\Documents\\ps\\combined_music_dataset.csv"
music_dataset = pd.read_csv(file_path)

# Step 1: Separate features and labels
X = music_dataset.drop(columns=['filename', 'genre', 'duration(in sec)'])  # Features (MFCCs)
y = music_dataset['genre']  # Labels (genres)

# Step 2: Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Split the data into training and test sets (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train)
X_test_tensor = torch.FloatTensor(X_test)
y_train_tensor = torch.LongTensor(y_train.factorize()[0])  # Convert categories to integers
y_test_tensor = torch.LongTensor(y_test.factorize()[0])

# Step 4: Define the neural network model
class MusicGenreClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MusicGenreClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)  # First hidden layer
        self.fc2 = nn.Linear(128, 64)           # Second hidden layer
        self.fc3 = nn.Linear(64, num_classes)   # Output layer
        self.relu = nn.ReLU()                    # Activation function

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x=self.dropout(x)
        x = self.relu(self.fc2(x))
        x=self.dropout(x)
        x = self.relu(self.fc3(x))
        x=self.dropout(x)
        x = self.relu(self.fc4(x))
        x = self.fc5(x)
        return x

# Step 5: Initialize the model, loss function, and optimizer
input_size = X_train.shape[1]  # Number of features
num_classes = len(y.unique())   # Number of unique genres
model = MusicGenreClassifier(input_size, num_classes)

criterion = nn.CrossEntropyLoss()  # Loss function for multi-class classification
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Optimizer

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

# Step 7: Evaluate the model
model.eval()
with torch.no_grad():
    y_pred = model(X_test_tensor)
    _, predicted = torch.max(y_pred, 1)

# Step 8: Print evaluation metrics
print(confusion_matrix(y_test_tensor, predicted))
print(classification_report(y_test_tensor, predicted, target_names=y.unique()))
    