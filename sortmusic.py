import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import librosa
import numpy as np

# Dataset class for loading music files and their labels
class MusicDataset(Dataset):
    def __init__(self, audio_files, labels, transform=None):
        """
        Initialize the dataset with audio file paths and labels.
        """
        self.audio_files = audio_files  # List of audio file paths
        self.labels = labels  # List of corresponding labels (genres)
        self.transform = transform  # Optional data transformation

    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        return len(self.audio_files)

    def __getitem__(self, idx):
        """
        Load an audio file, extract features, and return features and labels.
        """
        audio_path = self.audio_files[idx]  # Get the audio file path
        label = self.labels[idx]  # Get the corresponding label

        # Load audio file using librosa (y = audio signal, sr = sampling rate)
        y, sr = librosa.load(audio_path, sr=22050)  # Load audio with 22.05 kHz sampling rate

        # Extract MFCC (Mel-frequency cepstral coefficients) as features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)  # 13 MFCC coefficients
        mfcc = np.mean(mfcc, axis=1)  # Average MFCC features along the time axis
        
        mfcc = torch.tensor(mfcc, dtype=torch.float32)  # Convert MFCC to tensor

        if self.transform:
            mfcc = self.transform(mfcc)

        return mfcc, label  # Return the features (MFCC) and the label

# Neural network model for music genre classification
class MusicClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        """
        Define the architecture of the neural network (ANN).
        """
        super(MusicClassifier, self).__init__()
        
        # Fully connected layers
        self.fc1 = nn.Linear(input_size, hidden_size)  # Input to hidden layer
        self.relu = nn.ReLU()  # ReLU activation function
        self.fc2 = nn.Linear(hidden_size, num_classes)  # Hidden to output layer
        self.softmax = nn.Softmax(dim=1)  # Softmax for multi-class classification
    
    def forward(self, x):
        """
        Define the forward pass through the network.
        """
        x = self.fc1(x)  # Pass input through the first fully connected layer
        x = self.relu(x)  # Apply ReLU activation
        x = self.fc2(x)  # Pass through the second fully connected layer
        x = self.softmax(x)  # Apply Softmax for multi-class output
        return x

# Function to train the model
def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    """
    Train the model using the provided data.
    """
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Loop through the data
        for inputs, labels in train_loader:
            inputs = inputs.view(inputs.size(0), -1)  # Flatten the input tensor (MFCC features)
            labels = labels.long()  # Convert labels to long integers for CrossEntropyLoss
            
            optimizer.zero_grad()  # Zero out the gradients from previous steps
            outputs = model(inputs)  # Get predictions from the model
            
            loss = criterion(outputs, labels)  # Calculate the loss
            loss.backward()  # Backpropagate the loss
            optimizer.step()  # Update the model weights
            
            running_loss += loss.item()  # Add current loss to the running total
            
            # Calculate accuracy for this batch
            _, predicted = torch.max(outputs, 1)  # Get the predicted label (highest probability)
            total += labels.size(0)  # Count total samples
            correct += (predicted == labels).sum().item()  # Count correct predictions
        
        epoch_accuracy = 100 * correct / total  # Calculate accuracy for this epoch
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}, Accuracy: {epoch_accuracy}%")

# Function to evaluate the model on a test set
def evaluate_model(model, test_loader):
    """
    Evaluate the model on the test data.
    """
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    
    with torch.no_grad():  # No need to compute gradients during evaluation
        for inputs, labels in test_loader:
            inputs = inputs.view(inputs.size(0), -1)  # Flatten the input tensor (MFCC features)
            outputs = model(inputs)  # Get predictions from the model
            _, predicted = torch.max(outputs, 1)  # Get the predicted label (highest probability)
            total += labels.size(0)  # Count total samples
            correct += (predicted == labels).sum().item()  # Count correct predictions
    
    accuracy = 100 * correct / total  # Calculate accuracy
    print(f"Test Accuracy: {accuracy}%")

# Set the paths to the audio files and their corresponding labels
audio_files = ["jazz1.wav", "pop1.wav", "classical1.wav", "rock1.wav"]  # Replace with actual file paths
labels = [0, 1, 2, 3]  # Replace with actual genre labels (0 = jazz, 1 = pop, 2 = classical, 3 = rock)

# Hyperparameters
input_size = 13  # Number of MFCC coefficients (input features)
hidden_size = 64  # Number of hidden units in the network
num_classes = 4  # Number of genres (adjust based on your dataset)
learning_rate = 0.001  # Learning rate for the optimizer
num_epochs = 10  # Number of training epochs

# Create a custom dataset and DataLoader
train_dataset = MusicDataset(audio_files, labels)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Initialize the model
model = MusicClassifier(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()  # Cross-entropy loss for multi-class classification
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
train_model(model, train_loader, criterion, optimizer, num_epochs)

# Evaluate the model on the test data
evaluate_model(model, test_loader)
