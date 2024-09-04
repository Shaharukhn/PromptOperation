from flask import Flask, render_template, request, jsonify
from sklearn.preprocessing import StandardScaler
import numpy as np
import tensorflow as tf
import librosa
import os

app = Flask(__name__)

# Load the scaler parameters
scaler = StandardScaler()
scaler.mean_ = np.load('scaler_mean.npy')
scaler.scale_ = np.load('scaler_scale.npy')

# Load your pre-trained models
fnn_model = tf.keras.models.load_model('models/DL_Clasification_model_1.h5')
autoencoder_model = tf.keras.models.load_model('models/Encoder_Model3e.h5')

def process_audio(file_path):
    y, sr = librosa.load(file_path, sr=None)  # Load the audio file
    # Extract the same features used during training
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_scaled = np.mean(mfccs.T, axis=0)
    
    spectral_features = extract_spectral_features(y, sr)
    temporal_features = extract_temporal_features(y)
    
    # Combine features into a single array
    features = np.concatenate([mfccs_scaled, spectral_features, temporal_features])
    print("Extracted Features Shape:", features.shape)  # Debug statement
    return features

def extract_spectral_features(audio, sample_rate):
    spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate)[0]
    spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate)[0]
    return np.mean(spectral_centroids), np.mean(spectral_rolloff), np.mean(spectral_contrast)

def extract_temporal_features(audio):
    zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)[0]
    autocorrelation = librosa.autocorrelate(audio)
    return np.mean(zero_crossing_rate), np.mean(autocorrelation)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_fnn', methods=['POST'])
def predict_fnn():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    file_path = os.path.join("uploads", file.filename)
    file.save(file_path)
    
    features = process_audio(file_path).reshape(1, -1)  # Adjust shape as needed
    features_scaled = scaler.transform(features)  # Apply the same scaling
    print("Processed Scaled Features for FNN:", features_scaled)  # Debug statement
    prediction = fnn_model.predict(features_scaled)
    print("Model Prediction:", prediction)  # Debug statement
    result = "Abnormal" if prediction > 0.5 else "Normal"
    
    return jsonify({'prediction': result})

@app.route('/predict_autoencoder', methods=['POST'])
def predict_autoencoder():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    file_path = os.path.join("uploads", file.filename)
    file.save(file_path)
    
    features = process_audio(file_path).reshape(1, -1)  # Adjust shape as needed
    features_scaled = scaler.transform(features)  # Apply the same scaling
    print("Processed Scaled Features for Autoencoder:", features_scaled)  # Debug statement
    
    reconstruction = autoencoder_model.predict(features_scaled)
    print("Reconstruction:", reconstruction)  # Debug statement
    
    anomaly_score = np.mean(np.square(features_scaled - reconstruction))
    print("Anomaly Score:", anomaly_score)  # Debug statement
    
    # Experiment with different thresholds
    threshold = 0.2  # Adjust based on your validation data
    result = "Abnormal" if anomaly_score > threshold else "Normal"
    
    return jsonify({'anomaly_score': anomaly_score, 'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)
