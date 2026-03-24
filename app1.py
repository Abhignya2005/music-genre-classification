from flask import Flask, render_template, request, redirect, url_for
import torch
import torch.nn as nn
import torch.optim as optim
import os
from werkzeug.utils import secure_filename
import numpy as np
import librosa
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Path to save uploaded files
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed audio file extensions
ALLOWED_EXTENSIONS = {'wav', 'mp3'}

# Load the pre-trained model
class MusicGenreClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MusicGenreClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.relu(self.fc4(x))
        x = self.fc5(x)  # Output layer (no activation here, softmax is applied in loss)
        return x

model = MusicGenreClassifier(input_size=13, num_classes=4)  # Adjust according to your model
model.load_state_dict(torch.load("music_genre_classifier.pth"))
model.eval()

# Function to check file extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route for homepage
@app.route('/')
def index():
    return render_template('index1.html')

# Route to handle file upload and genre prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Pre-process the audio file (e.g., convert to mel spectrogram)
        audio_data, sr = librosa.load(filepath, sr=None)
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=40)
        mfccs = np.mean(mfccs.T, axis=0)

        # Normalize and convert to tensor
        scaler = StandardScaler()
        mfccs_scaled = scaler.fit_transform(mfccs.reshape(-1, 1)).flatten()
        mfccs_tensor = torch.FloatTensor(mfccs_scaled).unsqueeze(0)  # Add batch dimension

        # Predict the genre
        with torch.no_grad():
            output = model(mfccs_tensor)
            _, predicted = torch.max(output, 1)

        genre = 'Predicted Genre'  # Map the predicted value to your genres here, based on your model output
        genre_list = ['Pop', 'Rock', 'Jazz', 'Classical', 'Hip-Hop', 'Electronic', 'Reggae', 'Blues', 'Country', 'Metal']  # Example genres
        predicted_genre = genre_list[predicted.item()]

        return render_template('result.html', genre=predicted_genre)

    return "File type not allowed", 400

if __name__ == '__main__':
    app.run(debug=True)
