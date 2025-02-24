import os
import numpy as np
import pyshark
import tensorflow as tf
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from tensorflow import keras
from tensorflow.keras import layers
import joblib
import time
from collections import deque
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import threading
import pandas as pd

# Constants and file paths
AUTOENCODER_PATH = "autoencoder_model.h5"
ISO_FOREST_PATH = "iso_forest_model.pkl"

# Enhanced buffers for better statistical analysis
timing_buffer = deque(maxlen=6000)
packet_sizes = deque(maxlen=6000)
intervals = deque(maxlen=6000)
anomaly_scores = deque(maxlen=6000)
timestamps = deque(maxlen=6000)
is_anomaly_buffer = deque(maxlen=6000)  # Added to track anomaly status

# Visualization data storage
visualization_data = {
    'timestamps': [],
    'intervals': [],
    'packet_sizes': [],
    'anomaly_scores': [],
    'is_anomaly': []
}

# Global variables
scaler = RobustScaler()
buffer_size = 20000
live_data = []

interval_stats = {
    'mean': 0,
    'std': 0,
    'max': 0,
    'min': float('inf'),
    'median': 0,
    'q1': 0,
    'q3': 0
}

iso_forest = None
autoencoder = None

class MovingStats:
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.values = deque(maxlen=window_size)
        
    def add(self, value):
        self.values.append(value)
        
    def get_stats(self):
        if not self.values:
            return 0, 0, 0
        arr = np.array(self.values)
        return np.mean(arr), np.std(arr), np.median(arr)

# Initialize moving statistics trackers
interval_tracker = MovingStats(window_size=100)
packet_size_tracker = MovingStats(window_size=100)

def build_autoencoder(input_dim):
    """Build and compile the autoencoder model"""
    model = keras.Sequential([
        layers.Dense(128, activation="relu", input_shape=(input_dim,)),
        layers.Dropout(0.1),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.1),
        layers.Dense(32, activation="relu"),
        layers.Dense(16, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(128, activation="relu"),
        layers.Dense(input_dim, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

def initialize_models():
    """Initialize or load pre-trained models"""
    global iso_forest, autoencoder, scaler
    
    try:
        if os.path.exists(AUTOENCODER_PATH) and os.path.exists(ISO_FOREST_PATH):
            print("🔄 Loading pre-trained models...")
            autoencoder = keras.models.load_model(AUTOENCODER_PATH)
            iso_forest = joblib.load(ISO_FOREST_PATH)
            
            scaler.with_centering = True
            scaler.with_scaling = True
            
            return True
        return False
        
    except Exception as e:
        print(f"⚠️ Error loading models: {e}")
        return False

def extract_features(packet, prev_timestamp):
    """Extract features from a packet with enhanced error handling"""
    try:
        # Basic packet information
        packet_size = int(packet.length)
        protocol = hash(packet.highest_layer) % 1000
        
        # Handle IP information
        if hasattr(packet, 'ip'):
            src_ip = hash(packet.ip.src) % 1000
            dst_ip = hash(packet.ip.dst) % 1000
        else:
            # Use default values for non-IP packets
            src_ip = 0
            dst_ip = 0
        
        # Handle transport layer information
        if hasattr(packet, 'transport_layer') and packet.transport_layer:
            transport_layer = packet.transport_layer.lower()
            src_port = int(getattr(packet[transport_layer], 'srcport', 0))
            dst_port = int(getattr(packet[transport_layer], 'dstport', 0))
        else:
            src_port = 0
            dst_port = 0
            
        timestamp = float(packet.sniff_timestamp)
        interval = timestamp - prev_timestamp if prev_timestamp else 0
        
        return [src_ip, dst_ip, packet_size, protocol, src_port, dst_port, interval], timestamp
        
    except Exception as e:
        print(f"⚠️ Error extracting features: {e}")
        return None, prev_timestamp

def update_visualization():
    """Update the visualization in real-time with improved error handling"""
    plt.ion()
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    while True:
        try:
            # Clear all axes
            for ax in [ax1, ax2, ax3, ax4]:
                ax.clear()
            
            # Ensure we have enough data points
            min_points = min(len(timestamps), len(intervals), len(packet_sizes), len(anomaly_scores))
            if min_points == 0:
                plt.pause(2)
                continue
                
            # Get the last min_points for visualization
            recent_timestamps = list(timestamps)[-min_points:]
            recent_intervals = list(intervals)[-min_points:]
            recent_packet_sizes = list(packet_sizes)[-min_points:]
            recent_anomaly_scores = list(anomaly_scores)[-min_points:]
            
            # Convert timestamps to datetime
            plot_timestamps = [datetime.fromtimestamp(ts) for ts in recent_timestamps]
            
            # Plot 1: Packet Intervals
            ax1.plot(plot_timestamps, recent_intervals, 'b-', alpha=0.6)
            ax1.set_title('Packet Intervals Over Time')
            ax1.set_xlabel('Time')
            ax1.set_ylabel('Interval (s)')
            ax1.tick_params(axis='x', rotation=45)
            
            # Plot 2: Packet Sizes
            ax2.plot(plot_timestamps, recent_packet_sizes, 'g-', alpha=0.6)
            ax2.set_title('Packet Sizes Over Time')
            ax2.set_xlabel('Time')
            ax2.set_ylabel('Size (bytes)')
            ax2.tick_params(axis='x', rotation=45)
            
            # Plot 3: Anomaly Scores Distribution
            if recent_anomaly_scores:
                sns.histplot(recent_anomaly_scores, bins=50, ax=ax3)
                ax3.set_title('Anomaly Score Distribution')
                ax3.set_xlabel('Anomaly Score')
                ax3.set_ylabel('Count')
            
            # Plot 4: Scatter plot of Packet Size vs Interval
            scatter = ax4.scatter(recent_intervals, recent_packet_sizes, 
                                c=recent_anomaly_scores, cmap='viridis', alpha=0.6)
            ax4.set_title('Packet Size vs Interval')
            ax4.set_xlabel('Interval (s)')
            ax4.set_ylabel('Packet Size (bytes)')
            plt.colorbar(scatter, ax=ax4, label='Anomaly Score')
            
            plt.tight_layout()
            plt.draw()
            plt.pause(2)
            
        except Exception as e:
            print(f"Visualization error: {e}")
            plt.pause(2)

def is_anomaly(features, recent_intervals, recent_packet_sizes, isolation_score, mse):
    """Enhanced anomaly detection using multiple criteria"""
    if len(recent_intervals) < 50:
        return False, 0.0
    
    interval = features[-1]
    packet_size = features[2]
    
    # Get current statistics
    int_mean, int_std, int_median = interval_tracker.get_stats()
    size_mean, size_std, size_median = packet_size_tracker.get_stats()
    
    # Calculate Z-scores
    if int_std > 0:
        interval_zscore = abs((interval - int_mean) / int_std)
    else:
        interval_zscore = 0
        
    if size_std > 0:
        size_zscore = abs((packet_size - size_mean) / size_std)
    else:
        size_zscore = 0
    
    # Multiple criteria for anomaly detection
    criteria = {
        'isolation_forest': isolation_score < np.percentile(list(anomaly_scores)[-100:] if anomaly_scores else [0], 5),
        'autoencoder': mse > np.percentile(list(anomaly_scores)[-100:] if anomaly_scores else [0], 95),
        'interval_spike': interval_zscore > 3,
        'size_spike': size_zscore > 3,
        'combined_spike': (interval_zscore + size_zscore) > 5
    }
    
    # Calculate confidence score (0 to 1)
    confidence = sum(criteria.values()) / len(criteria)
    
    # Require at least 3 criteria to be met for an anomaly
    is_anomalous = confidence >= 0.6
    
    return is_anomalous, confidence

def analyze_live_traffic():
    """Main function to analyze network traffic"""
    global autoencoder, live_data, iso_forest
    print("🚀 Starting real-time anomaly detection...")
    
    # Start visualization in a separate thread
    viz_thread = threading.Thread(target=update_visualization, daemon=True)
    viz_thread.start()
    
    models_loaded = initialize_models()
    capture = pyshark.LiveCapture(interface="wlo1")
    prev_timestamp = None
    initial_training_done = models_loaded
    min_training_packets = 20000
    
    packet_bar = tqdm(total=min_training_packets, desc="📡 Capturing Packets", unit="pkt")
    
    anomaly_counter = 0
    total_packets = 0
    
    try:
        for packet in capture.sniff_continuously():
            try:
                features, current_timestamp = extract_features(packet, prev_timestamp)
                if features is None:
                    continue
                    
                live_data.append(features)
                if current_timestamp:
                    timestamps.append(current_timestamp)
                
                packet_sizes.append(features[2])
                if prev_timestamp and current_timestamp:
                    interval = current_timestamp - prev_timestamp
                    intervals.append(interval)
                    interval_tracker.add(interval)
                
                packet_size_tracker.add(features[2])
                
                if len(live_data) > buffer_size:
                    live_data.pop(0)
                
                packet_bar.update(1)
                total_packets += 1
                
                if len(live_data) < min_training_packets and not initial_training_done:
                    prev_timestamp = current_timestamp
                    continue
                    
                if not initial_training_done:
                    packet_bar.close()
                    print("\n📊 Training models with initial data...")
                    
                    live_data_np = np.array(live_data)
                    live_data_scaled = scaler.fit_transform(live_data_np)
                    
                    print("🔍 Training Isolation Forest...")
                    iso_forest = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
                    iso_forest.fit(live_data_scaled)
                    joblib.dump(iso_forest, ISO_FOREST_PATH)
                    
                    print("🔧 Training Autoencoder...")
                    autoencoder = build_autoencoder(input_dim=live_data_scaled.shape[1])
                    
                    for epoch in tqdm(range(50), desc="Training Autoencoder"):
                        autoencoder.fit(
                            live_data_scaled, live_data_scaled,
                            epochs=1, batch_size=32, shuffle=True, verbose=0
                        )
                    
                    autoencoder.save(AUTOENCODER_PATH)
                    initial_training_done = True
                    print("\n✅ Initial training complete!")
                    continue
                
                # Process packet for anomaly detection
                live_data_np = np.array([features])
                live_data_scaled = scaler.transform(live_data_np)
                
                isolation_score = iso_forest.decision_function(live_data_scaled)[0]
                reconstruction = autoencoder.predict(live_data_scaled)
                mse = np.mean(np.abs(reconstruction - live_data_scaled))
                
                anomaly_scores.append(mse)
                
                # Enhanced anomaly detection
                is_anomalous, confidence = is_anomaly(
                    features,
                    list(intervals),
                    list(packet_sizes),
                    isolation_score,
                    mse
                )
                
                is_anomaly_buffer.append(is_anomalous)
                
                if is_anomalous:
                    anomaly_counter += 1
                    anomaly_rate = (anomaly_counter / total_packets) * 100
                    
                    print(f"\n🚨 Anomaly detected!")
                    print(f"   Confidence: {confidence:.2f}")
                    print(f"   Anomaly Rate: {anomaly_rate:.2f}%")
                    print(f"   MSE: {mse:.4f}")
                    print(f"   Isolation Score: {isolation_score:.4f}")
                    print(f"   Packet Size: {features[2]} bytes")
                    print(f"   Interval: {features[-1]:.4f}s\n")
                
                # Periodic statistics update
                if total_packets % 1000 == 0:
                    print(f"\n📊 Statistics after {total_packets} packets:")
                    print(f"   - Anomaly Rate: {(anomaly_counter/total_packets)*100:.2f}%")
                    print(f"   - Average Interval: {np.mean(list(intervals)):.4f}s")
                    print(f"   - Average Packet Size: {np.mean(list(packet_sizes)):.0f} bytes\n")
                
                prev_timestamp = current_timestamp
                
            except Exception as e:
                print(f"⚠️ Error processing packet: {e}")
                continue
                
    except KeyboardInterrupt:
        print("\n👋 Stopping packet capture...")
        capture.close()
    finally:
        packet_bar.close()

if __name__ == "__main__":
    analyze_live_traffic()
    