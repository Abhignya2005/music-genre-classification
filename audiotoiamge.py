import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

audio_path = "C:\\Users\\Egen AI\\Documents\\ps\\forest-lullaby-110624.wav"
y, sr = librosa.load(audio_path)
spect = librosa.feature.melspectrogram(y=y, sr=sr)
spect_dB = librosa.power_to_db(spect, ref=np.max)
plt.figure(figsize=(1.28, 1.28), dpi=100)
librosa.display.specshow(spect_dB, sr=sr, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0fdB')
plt.title('Mel-frequency spectrogram')
plt.xlabel('Time')
plt.ylabel('Frequency (Hz)')
plt.tight_layout()  # Optional; you can remove it if it causes issues
plt.savefig('resized_spectrogram_128x128.jpeg', dpi=100)  # Corrected the filename and parameter separation
plt.show()
