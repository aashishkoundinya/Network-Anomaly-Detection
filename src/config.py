import os
from collections import deque

# File paths
AUTOENCODER_PATH = "autoencoder_model.h5"
ISO_FOREST_PATH = "iso_forest_model.pkl"

# Buffer configurations
BUFFER_SIZE = 20000
DEQUE_SIZE = 6000
WINDOW_SIZE = 100
MIN_TRAINING_PACKETS = 20000

# Model configurations
ISOLATION_FOREST_ESTIMATORS = 100
ISOLATION_FOREST_CONTAMINATION = 0.1
AUTOENCODER_EPOCHS = 50
AUTOENCODER_BATCH_SIZE = 32

# Visualization configurations
VISUALIZATION_UPDATE_INTERVAL = 2  # seconds

# Network interface
NETWORK_INTERFACE = "wlo1"  # Change this to your network interface

# Anomaly detection thresholds
ANOMALY_CONFIDENCE_THRESHOLD = 0.6
ZSCORE_THRESHOLD = 3
COMBINED_ZSCORE_THRESHOLD = 5
