import os
import numpy as np
import pyshark
import tensorflow as tf
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tensorflow import keras
from tensorflow.keras import layers
import joblib

# Load pre-trained model if available
AUTOENCODER_PATH = "autoencoder_model.h5"
ISO_FOREST_PATH = "iso_forest_model.pkl"

# Initialize scalers
scaler = MinMaxScaler()
label_encoder = LabelEncoder()
iso_forest = None
autoencoder = None
buffer_size = 10000  # Keep a rolling buffer of 10,000 packets
live_data = []  # Store packets

# Build Autoencoder Model
def build_autoencoder(input_dim):
    model = keras.Sequential([
        layers.Dense(64, activation="relu", input_shape=(input_dim,)),
        layers.Dense(32, activation="relu"),
        layers.Dense(16, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(input_dim, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

# Load or Train Models
def initialize_models():
    global iso_forest, autoencoder
    
    if os.path.exists(AUTOENCODER_PATH):
        print("🔄 Loading pre-trained autoencoder...")
        autoencoder = keras.models.load_model(AUTOENCODER_PATH)
    else:
        autoencoder = None  # Will be trained later

    if os.path.exists(ISO_FOREST_PATH):
        print("🔄 Loading pre-trained Isolation Forest model...")
        iso_forest = joblib.load(ISO_FOREST_PATH)
    else:
        print("🚀 Initializing new Isolation Forest model...")
        iso_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)

initialize_models()

# Real-time Anomaly Detection
def analyze_live_traffic():
    global autoencoder, iso_forest, live_data
    
    print("🚀 Starting real-time anomaly detection...")
    capture = pyshark.LiveCapture(interface="wlo1")

    for packet in capture.sniff_continuously():
        try:
            # Extract packet features
            packet_size = int(packet.length)
            protocol = int(packet.highest_layer, 16) if packet.highest_layer.isnumeric() else 0

            # Store in buffer
            live_data.append([packet_size, protocol])
            if len(live_data) > buffer_size:
                live_data.pop(0)  # Keep only last 10,000 packets

            if len(live_data) < 5000:
                continue  # Wait until buffer has enough data

            live_data_np = np.array(live_data)
            live_data_scaled = scaler.fit_transform(live_data_np)

            # Ensure Isolation Forest is trained
            if iso_forest is None or len(live_data) == 5000:  
                print("🚀 Training Isolation Forest...")
                iso_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
                iso_forest.fit(live_data_scaled)
                joblib.dump(iso_forest, ISO_FOREST_PATH)  # Save trained model

            # Ensure Autoencoder is trained
            if autoencoder is None:
                print("🚀 Training Autoencoder...")
                autoencoder = build_autoencoder(input_dim=live_data_scaled.shape[1])
                autoencoder.fit(live_data_scaled, live_data_scaled, epochs=50, batch_size=32, shuffle=True)
                autoencoder.save(AUTOENCODER_PATH)  # Save trained model

            # Process the current packet
            input_data = np.array([[packet_size, protocol]])
            input_scaled = scaler.transform(input_data)

            isolation_score = iso_forest.decision_function(input_scaled)[0]
            reconstruction = autoencoder.predict(input_scaled)
            mse = np.mean(np.abs(reconstruction - input_scaled))

            # Dynamic Thresholds
            iso_scores = iso_forest.decision_function(live_data_scaled)
            anomaly_threshold = np.percentile(iso_scores, 5)  # Bottom 5% are anomalies

            # Detect anomalies
            if isolation_score < anomaly_threshold or mse > 0.02:
                print(f"🚨 Anomaly detected! Packet size: {packet_size}, Protocol: {protocol}, MSE: {mse}, Isolation Score: {isolation_score}")

        except Exception as e:
            print(f"⚠️ Error processing packet: {e}")

# Run live monitoring
analyze_live_traffic()
