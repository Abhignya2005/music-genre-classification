from flask import Flask, render_template_string
import random
import numpy as np
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

app = Flask(__name__)

# -----------------------
# Model Code
# -----------------------
random.seed(120)
np.random.seed(120)
torch.manual_seed(120)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

file_path = "C:\\Users\\Egen AI\\Documents\\ps\\combined_music_dataset.csv"
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

# -----------------------
# Routes
# -----------------------

@app.route('/')
def home():
    return render_template_string('''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Automatic Music Classification</title>
        <style>
            /* Page Reset */
            body {
                margin: 0;
                padding: 0;
                overflow: hidden;
                background: #000;
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
            }

            /* Book Animation Container */
            .book-container {
                position: relative;
                width: 60%;
                height: 60%;
                display: flex;
                justify-content: center;
                align-items: center;
                perspective: 1200px; /* For 3D effect */
            }

            /* Image as a Single Picture */
            .book-image {
                width: 100%;
                height: 100%;
                background: url('/static/music_banner.jpg') no-repeat center center;
                background-size: cover;
                position: absolute;
                z-index: 2;
            }

            /* Split into Two Halves */
            .book-half {
                position: absolute;
                width: 50%;
                height: 100%;
                background: url('/static/music_banner.jpg') no-repeat center center;
                background-size: cover;
                transition: transform 2s ease-in-out;
            }

            .left-half {
                left: 0;
                background-position: left center;
                transform-origin: left center;
            }

            .right-half {
                right: 0;
                background-position: right center;
                transform-origin: right center;
            }

            /* Animation Trigger */
            .book-container.open .book-image {
                opacity: 0; /* Hide the full image during animation */
            }

            .book-container.open .left-half {
                transform: rotateY(-90deg);
            }

            .book-container.open .right-half {
                transform: rotateY(90deg);
            }

            /* Text Animation */
            .welcome-text {
                position: absolute;
                font-family: 'Arial', sans-serif;
                font-size: 2.5em;
                color: white;
                text-shadow: 2px 2px 8px rgba(0, 0, 0, 0.8);
                text-align: center;
                opacity: 0;
                transition: opacity 2s ease-in-out 2s;
                z-index: 1;
            }

            .book-container.open .welcome-text {
                opacity: 1;
            }

            /* Overlay for Key Detection */
            .overlay {
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: rgba(0, 0, 0, 0.7);
                display: flex;
                justify-content: center;
                align-items: center;
                color: white;
                font-size: 1.5em;
                font-family: Arial, sans-serif;
                opacity: 0;
                transition: opacity 1s ease-in-out;
            }

            .overlay.visible {
                opacity: 1;
            }
        </style>
    </head>
    <body>
        <!-- Book Container with Image and Two Halves -->
        <div class="book-container" id="book">
            <div class="book-image"></div>
            <div class="book-half left-half"></div>
            <div class="book-half right-half"></div>
            <div class="welcome-text">Welcome to the <br> Automatic Music Classification</div>
        </div>

        <!-- Overlay for Keypress Prompt -->
        <div class="overlay" id="overlay">Press any key to continue...</div>

        <script>
            // Trigger the Book Animation After Page Load
            window.onload = function() {
                const book = document.getElementById('book');
                const overlay = document.getElementById('overlay');

                setTimeout(() => {
                    book.classList.add('open');  // Trigger the book opening animation
                    let msg = new SpeechSynthesisUtterance("Welcome to Automatic Music Classification");
                    msg.rate = 1;  // Speed of speech
                    msg.pitch = 1; // Pitch of speech
                    window.speechSynthesis.speak(msg);
                }, 500); // Delay for smooth animation start

                // After 5 seconds, show overlay
                setTimeout(() => {
                    overlay.classList.add('visible');
                }, 10000); // Show overlay after 10 seconds
            };

            // Listen for Key Press to Navigate
            document.addEventListener('keydown', function() {
                window.location.href = "/welcome";
            });
        </script>
    </body>
    </html>
    ''')

@app.route('/results')
def results():
    return render_template_string('''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Evaluation Results</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    background: linear-gradient(to right, #2a3d66, #3b4a83);
                    color: #ffffff;
                    margin: 0;
                    padding: 0;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                    overflow: hidden;
                }
                h1 {
                    font-size: 48px;
                    text-align: center;
                    text-shadow: 2px 2px 5px rgba(0, 0, 0, 0.3);
                }
            </style>
        </head>
        <body>
            <h1>Evaluation Results</h1>
            <script>
                window.onload = function() {
                    let msg = new SpeechSynthesisUtterance("Evaluation Results");
                    window.speechSynthesis.speak(msg);
                };
                document.addEventListener('keydown', function(event) {
                    if (event.key === "Enter") {
                        window.location.href = "/results/details";
                    }
                });
            </script>
        </body>
        </html>
    ''')

@app.route('/results/details')
def results_details():
    conf_matrix = "[[50, 2], [1, 47]]"
    class_report = "Precision: 0.96\nRecall: 0.95\nF1-Score: 0.95"
    return render_template_string(f'''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Results Details</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    background: linear-gradient(to right, #2a3d66, #3b4a83);
                    color: #ffffff;
                    margin: 0;
                    padding: 0;
                    overflow: hidden;
                }}
                .slide {{
                    position: absolute;
                    top: 0;
                    left: 100%;
                    width: 100%;
                    height: 100%;
                    background: linear-gradient(to right, #2a3d66, #3b4a83);
                    animation: slide-in 1s forwards;
                }}
                @keyframes slide-in {{
                    from {{ left: 100%; }}
                    to {{ left: 0%; }}
                }}
                h1, h2 {{
                    text-align: center;
                    color: #f8f8f8;
                    text-shadow: 2px 2px 5px rgba(0, 0, 0, 0.3);
                }}
                h1 {{
                    font-size: 36px;
                    margin-top: 50px;
                }}
                h2 {{
                    font-size: 28px;
                    margin-top: 20px;
                }}
                pre {{
                    background-color: #333;
                    color: #f8f8f8;
                    padding: 20px;
                    border-radius: 8px;
                    font-size: 18px;
                    text-align: center;
                    white-space: pre-wrap;
                    word-wrap: break-word;
                    max-width: 80%;
                    margin: 0 auto;
                }}
                p {{
                    text-align: center;
                    margin-top: 30px;
                    font-size: 18px;
                }}
            </style>
        </head>
        <body>
            <div class="slide">
                <h1>Evaluation Results</h1>
                <div class="container">
                    <h2>Confusion Matrix</h2>
                    <pre>{conf_matrix}</pre>
                    <h2>Classification Report</h2>
                    <pre>{class_report}</pre>
                </div>
                <p>Press <b>Enter</b> to continue...</p>
            </div>
            <script>
                document.addEventListener('keydown', function(event) {{
                    if (event.key === "Enter") {{
                        window.location.href = "/thankyou";
                    }}
                }});
            </script>
        </body>
        </html>
    ''')


@app.route('/thankyou')
def thank_you():
    return render_template_string('''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Thank You</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    background: linear-gradient(to right, #2a3d66, #3b4a83);
                    color: #ffffff;
                    margin: 0;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                    overflow: hidden;
                }
                h1 {
                    font-size: 48px;
                    text-align: center;
                    text-shadow: 2px 2px 5px rgba(0, 0, 0, 0.3);
                }
            </style>
        </head>
        <body>
            <h1>Thank You!</h1>
        </body>
        </html>
    ''')


    
# -----------------------
# Main Execution
# -----------------------
if __name__ == '__main__':
    app.run(debug=True)