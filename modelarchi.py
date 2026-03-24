import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split


# 1. Load the audio sample
audio_file = "C:\\Users\\Egen AI\\Documents\\ps\\forest-lullaby-110624.wav"
y, sr = librosa.load(audio_file, sr=None)

# 2. Crop the audio to the desired size (e.g., from 2 seconds to 5 seconds)
start_time = 2  # start time in seconds
end_time = 5    # end time in seconds

# Convert start and end times to sample indices
start_sample = librosa.time_to_samples(start_time, sr=sr)
end_sample = librosa.time_to_samples(end_time, sr=sr)

# Crop the audio
cropped_audio = y[start_sample:end_sample]

# 3. Extract the Mel Spectrogram from the cropped audio
mel_spectrogram = librosa.feature.melspectrogram(y=cropped_audio, sr=sr)

# Display the Mel Spectrogram
librosa.display.specshow(librosa.power_to_db(mel_spectrogram, ref=np.max), y_axis='mel', x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel Spectrogram')
plt.show()

# 4. Flatten the Mel Spectrogram to feed into an ANN
# We flatten the spectrogram into a 1D vector to input it into the ANN.
mel_spectrogram_flat = mel_spectrogram.flatten()

# 5. Prepare the input tensor
input_tensor = torch.tensor(mel_spectrogram_flat).float()

# 6. Define the ANN Model
class AudioANN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(AudioANN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Define input size, hidden size, and output size
input_size = mel_spectrogram_flat.shape[0]  # This is the length of the flattened Mel Spectrogram
hidden_size = 128  # Number of neurons in the hidden layer
output_size = 10   # Assuming 10 output classes for classification, adjust as needed

# Instantiate the model
model = AudioANN(input_size, hidden_size, output_size)

# 7. Define Loss Function and Optimizer
loss_function = nn.CrossEntropyLoss()  # For multi-class classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 8. Training Loop (Example for a single input)
# In practice, you should batch your data and split into training and validation sets.
# Here, we show a single iteration for demonstration.

# Define a dummy target (replace this with your actual target labels)
target = torch.tensor([1])  # Assuming a target class index, e.g., '1' for class 1

# Zero the gradients
optimizer.zero_grad()

# Forward pass
output = model(input_tensor)

# Calculate loss
loss = loss_function(output, target)

# Backward pass (gradient computation)
loss.backward()

# Update weights
optimizer.step()

# Print the loss
print(f"Loss: {loss.item()}")

# 9. Evaluate the Model (For demonstration, we print the output)
print("Model output:", output)
print(pd.shape)
