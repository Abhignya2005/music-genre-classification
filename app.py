from flask import Flask, render_template, request, redirect, url_for
import os
import random
import numpy as np
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from werkzeug.utils import secure_filename
import librosa

app = Flask(__name__)

# Ensure the 'uploads' directory exists
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ----------------------- Model Code -----------------------
random.seed(120)
np.random.seed(120)
torch.manual_seed(120)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

file_path = "combined_music_dataset.csv"
music_dataset = pd.read_csv(file_path)
X = music_dataset.drop(columns=['filename', 'genre', 'duration(in sec)'])
y = music_dataset['genre']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
for train_index, test_index in sss.split(X_scaled, y):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

X_train_tensor = torch.FloatTensor(X_train)
X_test_tensor = torch.FloatTensor(X_test)
y_train_tensor = torch.LongTensor(y_train.factorize()[0])
y_test_tensor = torch.LongTensor(y_test.factorize()[0])

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
        x = self.fc5(x)
        return x

input_size = X_train.shape[1]
num_classes = len(y.unique())
model = MusicGenreClassifier(input_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

model.eval()
with torch.no_grad():
    y_pred = model(X_test_tensor)
    _, predicted = torch.max(y_pred, 1)

conf_matrix = confusion_matrix(y_test_tensor, predicted)
class_report = classification_report(y_test_tensor, predicted, target_names=y.unique())

# ----------------------- Routes -----------------------
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/welcome')
def welcome():
    return render_template('welcome.html')

@app.route('/thankyou')
def thank_you():
    return render_template('thankyou.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    filename = secure_filename(file.filename)
    file_path = os.path.join('uploads', filename)
    file.save(file_path)

    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs = np.mean(mfccs, axis=1)
    mfccs_scaled = scaler.transform([mfccs])

    input_tensor = torch.FloatTensor(mfccs_scaled)
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
    _, predicted_class = torch.max(output, 1)

    genre = y_train.unique()[predicted_class.item()]

    genre_image_map = {
        'classical': 'classical1.jpg',
        'Jazz': 'jazz.jpg',
        'Pop': 'pop.jpg',
        'Rock': 'rock.jpg'
    }
    genre_image = genre_image_map.get(genre)

    return render_template('prediction_result.html', genre=genre, genre_image=genre_image)

if __name__ == "__main__":
    app.run(debug=True)
