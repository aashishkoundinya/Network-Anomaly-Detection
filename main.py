import os
import pandas as pd
import numpy as np
import tensorflow as tf
import pyshark
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from tensorflow.keras import layers

# Load dataset
csv_file = "Jain_Logs_UTF8.csv"  # Change this to your actual file
df = pd.read_csv(csv_file)

# Fix column names: Remove spaces, lowercase all
df.columns = df.columns.str.strip().str.lower()

# Check available columns
print("Available columns:", df.columns)

# Ensure required columns exist
required_columns = ['length', 'protocol']
if not all(col in df.columns for col in required_columns):
    raise KeyError(f"Missing columns: {set(required_columns) - set(df.columns)}")

# Select only numerical features
df = df[required_columns]

# Normalize data
scaler = MinMaxScaler()
label_encoder = LabelEncoder()
df['protocol'] = label_encoder.fit_transform(df['protocol'])
df_scaled = scaler.fit_transform(df)

# Define Autoencoder model for anomaly detection
model = keras.Sequential([
    layers.Dense(32, activation="relu", input_shape=(df_scaled.shape[1],)),
    layers.Dense(16, activation="relu"),
    layers.Dense(8, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(32, activation="relu"),
    layers.Dense(df_scaled.shape[1], activation="sigmoid")
])

model.compile(optimizer="adam", loss="mse")
model.fit(df_scaled, df_scaled, epochs=20, batch_size=16, shuffle=True)

# Get reconstruction loss threshold
reconstructions = model.predict(df_scaled)
mse_loss = np.mean(np.abs(reconstructions - df_scaled), axis=1)
threshold = np.percentile(mse_loss, 95)  # Set threshold for anomaly detection

print(f"Anomaly detection threshold: {threshold}")

# Real-time anomaly detection function
def analyze_live_traffic():
    print("Starting real-time packet capture...")
    
    capture = pyshark.LiveCapture(interface="wlo1")  # Change interface if needed

    for packet in capture.sniff_continuously():
        try:
            # Extract packet features
            packet_size = int(packet.length)
            protocol = int(packet.highest_layer, 16) if packet.highest_layer.isnumeric() else 0
            
            # Normalize input
            input_data = scaler.transform([[packet_size, protocol]])

            # Predict reconstruction error
            reconstruction = model.predict(input_data)
            mse = np.mean(np.abs(reconstruction - input_data))

            # Log anomalies
            if mse > threshold:
                print(f"🚨 Anomaly detected! Packet size: {packet_size}, Protocol: {protocol}, MSE: {mse}")

        except Exception as e:
            print(f"Error processing packet: {e}")

# Run live monitoring
analyze_live_traffic()
