import os
import numpy as np
import pyshark
import tensorflow as tf
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers
import joblib
import time
from collections import deque
from tqdm import tqdm  

AUTOENCODER_PATH = "autoencoder_model.h5"
ISO_FOREST_PATH = "iso_forest_model.pkl"

# Increased buffer sizes for better baseline establishment
timing_buffer = deque(maxlen=2000)  # Increased from 1000
scaler = MinMaxScaler()
buffer_size = 15000  # Increased from 10000
live_data = []  

# Add statistical tracking for adaptive thresholds
interval_stats = {
    'mean': 0,
    'std': 0,
    'max': 0,
    'min': float('inf')
}

iso_forest = None  
autoencoder = None  

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

def initialize_models():
    global iso_forest, autoencoder

    if os.path.exists(AUTOENCODER_PATH) and os.path.exists(ISO_FOREST_PATH):
        print("🔄 Loading pre-trained models...")
        autoencoder = keras.models.load_model(AUTOENCODER_PATH)
        iso_forest = joblib.load(ISO_FOREST_PATH)
        return True
    return False

def extract_features(packet, prev_timestamp):
    try:
        packet_size = int(packet.length)
        protocol = hash(packet.highest_layer) % 1000  
        src_ip = hash(packet.ip.src) % 1000
        dst_ip = hash(packet.ip.dst) % 1000
        src_port = int(packet[packet.transport_layer].srcport) if hasattr(packet, "transport_layer") else 0
        dst_port = int(packet[packet.transport_layer].dstport) if hasattr(packet, "transport_layer") else 0
        timestamp = float(packet.sniff_timestamp)
        interval = timestamp - prev_timestamp if prev_timestamp else 0
        return [src_ip, dst_ip, packet_size, protocol, src_port, dst_port, interval], timestamp
    except Exception:
        return None, prev_timestamp

def update_interval_statistics(intervals):
    """Update running statistics for network intervals"""
    global interval_stats
    
    interval_stats['mean'] = np.mean(intervals)
    interval_stats['std'] = np.std(intervals)
    interval_stats['max'] = max(intervals)
    interval_stats['min'] = min(intervals)

def is_traffic_spike(current_interval, recent_intervals):
    """
    Enhanced traffic spike detection using multiple criteria
    Returns (is_spike, confidence)
    """
    if len(recent_intervals) < 50:  # Need enough data for baseline
        return False, 0.0

    # Update running statistics
    update_interval_statistics(recent_intervals)
    
    # Calculate dynamic thresholds
    mean_interval = interval_stats['mean']
    std_interval = interval_stats['std']
    
    # Multiple criteria for spike detection
    criteria = {
        'std_deviation': current_interval < (mean_interval - 2 * std_interval),
        'ratio_to_mean': current_interval < (mean_interval * 0.3),  # Increased from 0.5
        'sudden_change': current_interval < (np.median(recent_intervals[-10:]) * 0.4)
    }
    
    # Calculate confidence score (0 to 1)
    confidence = sum(criteria.values()) / len(criteria)
    
    # Require at least 2 criteria to be met for a spike
    is_spike = confidence >= 0.66  # At least 2 out of 3 criteria
    
    return is_spike, confidence

def analyze_live_traffic():
    global autoencoder, live_data, timing_buffer, iso_forest
    print("\nStarting real-time anomaly detection...\n")
    
    models_loaded = initialize_models()
    
    capture = pyshark.LiveCapture(interface="wlo1")
    prev_timestamp = None
    initial_training_done = models_loaded

    # Increase initial training size
    min_training_packets = 7500  # Increased from 5000

    packet_bar = tqdm(total=min_training_packets, desc="Capturing Packets", unit="pkt", 
                     bar_format="{l_bar}{bar} {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
                     colour="green")

    # Add counters for monitoring
    spike_counter = 0
    total_packets = 0
    
    for packet in capture.sniff_continuously():
        try:
            features, prev_timestamp = extract_features(packet, prev_timestamp)
            if features is None:
                continue

            live_data.append(features)
            timing_buffer.append(prev_timestamp)
            if len(live_data) > buffer_size:
                live_data.pop(0)

            packet_bar.update(1)
            total_packets += 1

            if len(live_data) < min_training_packets and not initial_training_done:
                continue

            if not initial_training_done:
                packet_bar.close()
                live_data_np = np.array(live_data)
                live_data_scaled = scaler.fit_transform(live_data_np)

                print("\nTraining Isolation Forest...\n")
                iso_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
                iso_forest.fit(live_data_scaled)
                joblib.dump(iso_forest, ISO_FOREST_PATH)

                print("🔧 Training Autoencoder...")
                autoencoder = build_autoencoder(input_dim=live_data_scaled.shape[1])
                
                for epoch in tqdm(range(50), desc="Training Autoencoder", 
                                bar_format="{l_bar}{bar} {n_fmt}/{total_fmt} [{elapsed}<{remaining}]", 
                                colour="green"):
                    autoencoder.fit(live_data_scaled, live_data_scaled, 
                                  epochs=1, batch_size=32, shuffle=True, verbose=0)
                
                autoencoder.save(AUTOENCODER_PATH)
                initial_training_done = True
                continue

            live_data_np = np.array(live_data)
            live_data_scaled = scaler.transform(live_data_np)
            
            input_data = np.array([features])
            input_scaled = scaler.transform(input_data)
            
            isolation_score = iso_forest.decision_function(input_scaled)[0]
            reconstruction = autoencoder.predict(input_scaled)
            mse = np.mean(np.abs(reconstruction - input_scaled))

            iso_scores = iso_forest.decision_function(live_data_scaled)
            anomaly_threshold = np.percentile(iso_scores, 5)

            if len(timing_buffer) > 50:
                recent_intervals = np.diff(list(timing_buffer))[-50:]  # Get last 50 intervals
                is_spike, confidence = is_traffic_spike(features[-1], recent_intervals)
                
                if is_spike:
                    spike_counter += 1
                    spike_rate = (spike_counter / total_packets) * 100
                    
                    print(f"⚠️ Traffic spike detected! Interval: {features[-1]:.4f}s "
                          f"(Avg: {interval_stats['mean']:.4f}s, Confidence: {confidence:.2f}, "
                          f"Spike Rate: {spike_rate:.2f}%)")

            if isolation_score < anomaly_threshold or mse > 0.02:
                print(f"🚨 Anomaly detected! Packet details: {features}, MSE: {mse:.4f}, "
                      f"Isolation Score: {isolation_score:.4f}")

            # Periodically report statistics
            if total_packets % 1000 == 0:
                print(f"\n📊 Statistics after {total_packets} packets:")
                print(f"   - Spike Rate: {(spike_counter/total_packets)*100:.2f}%")
                print(f"   - Average Interval: {interval_stats['mean']:.4f}s")
                print(f"   - Interval Std Dev: {interval_stats['std']:.4f}s\n")

        except Exception as e:
            print(f"⚠️ Error processing packet: {e}")

if __name__ == "__main__":
    analyze_live_traffic()
    