import os
import numpy as np
import pyshark
import tensorflow as tf
from sklearn.preprocessing import RobustScaler
import joblib
import time
from collections import deque
from tqdm import tqdm
import threading
import pandas as pd
import logging
from datetime import datetime
import platform

from constants import *
from models import *
from utils import *

def analyze_live_traffic():
    """Main function to analyze network traffic with enhanced capabilities"""
    global autoencoder, live_data, iso_forest, scaler, dbscan, threshold, dos_threshold
    
    print("🚀 Starting enhanced real-time anomaly detection...")
    
    # Initialize logging once to prevent duplicates
    initialize_logging()
    
    # Initialize periodic summary tracking
    last_summary_time = time.time()
    summary_interval = 30  # 30 seconds
    last_packet_count = 0
    last_anomaly_count = 0
    
    # Configure colored terminal output if available
    try:
        from colorama import init, Fore, Back, Style
        init()  # Initialize colorama
        color_enabled = True
    except ImportError:
        color_enabled = False
        print("Note: Install 'colorama' package for colored terminal output")
    
    # Define color codes if available
    if color_enabled:
        COLORS = {
            'red': Fore.RED,
            'orange': Fore.YELLOW,
            'blue': Fore.BLUE,
            'green': Fore.GREEN,
            'reset': Style.RESET_ALL,
            'bold': Style.BRIGHT
        }
    else:
        # Fallback with no colors
        COLORS = {
            'red': '',
            'orange': '',
            'blue': '',
            'green': '',
            'reset': '',
            'bold': ''
        }
    
    # Start visualization in a separate thread
    viz_thread = threading.Thread(target=update_visualization, daemon=True)
    viz_thread.start()
    
    models_loaded = initialize_models()
    
    # Get the appropriate network interface
    # In Windows, it might be different from wlo1
    if platform.system() == "Windows":
        default_interface = "Ethernet"  # Change as needed
    else:
        default_interface = "wlo1"  # Linux default
    
    # Attempt to get list of interfaces
    try:
        import netifaces
        interfaces = netifaces.interfaces()
        if interfaces:
            print(f"Available interfaces: {', '.join(interfaces)}")
            if default_interface not in interfaces and interfaces:
                default_interface = interfaces[0]
    except ImportError:
        print("Package netifaces not installed. Using default interface.")
    
    print(f"Using interface: {default_interface}")
    
    try:
        capture = pyshark.LiveCapture(interface=default_interface)
    except Exception as e:
        print(f"Error initializing capture on {default_interface}: {e}")
        print("Attempting to use any available interface...")
        try:
            capture = pyshark.LiveCapture()
        except Exception as e2:
            print(f"Failed to initialize capture: {e2}")
            logging.error(f"Failed to initialize packet capture: {e2}")
            return
    
    prev_timestamp = None
    initial_training_done = models_loaded
    min_training_packets = 20000
    
    # Only show progress bar during initial training
    packet_bar = tqdm(total=min_training_packets, desc="📡 Capturing Packets", unit="pkt")
    
    anomaly_counter = 0
    total_packets = 0
    last_training_update = time.time()
    training_interval = 3600  # Retrain models every hour
    
    # Counters for severity tracking
    severity_counts = {
        3: 0,  # Critical
        2: 0,  # Warning
        1: 0   # Info
    }
    
    # Create arrays for feature history
    all_features = []
    is_anomalous_list = []
    
    # Print header message
    print("\n" + "="*80)
    print(" " + COLORS['bold'] + "ENHANCED NETWORK ANOMALY DETECTION SYSTEM" + COLORS['reset'] + " ")
    print("="*80)
    print(f"Logging anomalies to: {LOG_PATH}")
    print("Initial model training phase will begin after collecting enough data...")
    print("="*80 + "\n")
    
    # Set default threshold values
    network_type = detect_network_type()
    network_config = NETWORK_TYPES[network_type]
    threshold = network_config['threshold']
    dos_threshold = network_config['dos_threshold']
    print(f"🌍 Initial network environment: {network_type}")
    print(f"Initial detection parameters - Threshold: {threshold}, DoS Threshold: {dos_threshold}")
    
    try:
        for packet in capture.sniff_continuously():
            try:
                # Extract features with enhanced feature set
                features, current_timestamp, metadata = extract_features(packet, prev_timestamp)
                if features is None:
                    continue
                    
                # Store data
                live_data.append(features)
                all_features.append(features)
                
                if current_timestamp:
                    timestamps.append(current_timestamp)
                
                # Track statistics
                packet_sizes.append(features[2])  # Packet size
                port_tracker.add(features[5])     # Destination port
                protocol_tracker.add(features[3]) # Protocol
                
                if prev_timestamp and current_timestamp:
                    interval = current_timestamp - prev_timestamp
                    intervals.append(interval)
                    interval_tracker.add(interval)
                
                packet_size_tracker.add(features[2])
                
                if len(live_data) > buffer_size:
                    # Remove oldest data
                    live_data.pop(0)
                
                # Update progress bar during initial training phase
                if not initial_training_done:
                    packet_bar.update(1)
                
                total_packets += 1
                
                # Initial training phase
                if len(live_data) < min_training_packets and not initial_training_done:
                    prev_timestamp = current_timestamp
                    continue
                    
                if not initial_training_done:
                    packet_bar.close()
                    print("\n📊 Training models with initial data...")
                    
                    # Convert to numpy array for processing
                    live_data_np = np.array(live_data)
                    
                    # Fit scaler and transform data
                    live_data_scaled = scaler.fit_transform(live_data_np)
                    
                    print("🔍 Training Isolation Forest...")
                    iso_forest = IsolationForest(
                        n_estimators=150, 
                        contamination=0.05,  # Start with conservative contamination assumption
                        max_samples="auto",
                        random_state=42
                    )
                    iso_forest.fit(live_data_scaled)
                    joblib.dump(iso_forest, ISO_FOREST_PATH)
                    
                    print("🔧 Training Autoencoder...")
                    autoencoder = build_autoencoder(input_dim=live_data_scaled.shape[1])
                    
                    # Add callbacks for more effective training
                    callbacks = [
                        EarlyStopping(monitor='loss', patience=5, restore_best_weights=True),
                        ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3, min_lr=0.0001)
                    ]
                    
                    for epoch in tqdm(range(3), desc="Training Autoencoder"):  # Initial quick training
                        autoencoder.fit(
                            live_data_scaled, live_data_scaled,
                            epochs=10, batch_size=64, shuffle=True, 
                            verbose=0, callbacks=callbacks
                        )
                    
                    autoencoder.save(AUTOENCODER_PATH)
                    
                    # Train DBSCAN for anomaly clustering
                    print("🧩 Training DBSCAN for anomaly clustering...")
                    dbscan = DBSCAN(eps=0.5, min_samples=10)
                    dbscan.fit(live_data_scaled)
                    joblib.dump(dbscan, DBSCAN_PATH)
                    
                    # Save the scaler
                    joblib.dump(scaler, SCALER_PATH)
                    
                    initial_training_done = True
                    print("\n✅ Initial training complete!")
                    
                    # Start with clean slate after training
                    is_anomalous_list = [False] * len(live_data)
                    
                    # Do NOT reinitialize progress bar for monitoring after initial training
                    # We'll only show summary updates every 30 seconds
                    print("\n📡 Monitoring Network - Summary updates every 30 seconds...")
                    continue
                
                # Process packet for anomaly detection
                feature_vector = np.array([features])
                feature_buffer.append(features)  # Store for future analysis
                
                # Scale the features
                feature_scaled = scaler.transform(feature_vector)
                
                # Get anomaly scores
                isolation_score = iso_forest.decision_function(feature_scaled)[0]
                reconstruction = autoencoder.predict(feature_scaled, verbose=0)
                mse = float(np.mean(np.abs(reconstruction - feature_scaled)))
                
                anomaly_scores.append(mse)
                
                # Enhanced anomaly detection with classification
                is_anomalous, confidence, classification = is_anomaly(
                    features,
                    metadata,
                    isolation_score,
                    mse
                )
                
                # Store anomaly status
                is_anomaly_buffer.append(is_anomalous)
                is_anomalous_list.append(is_anomalous)
                
                # If anomaly detected, log it (but don't print to terminal)
                if is_anomalous:
                    anomaly_counter += 1
                    
                    # Only log significant anomalies to file
                    if confidence >= 0.6 and classification:
                        # Log to file
                        log_anomaly(features, metadata, confidence, classification, 
                                    isolation_score, mse)
                        
                        # Track severity
                        severity_level = classification['severity']
                        severity_counts[severity_level] += 1
                
                # Periodic summary update (every 30 seconds)
                current_time = time.time()
                if current_time - last_summary_time > summary_interval:
                    # Update network type only during summary updates, not for every packet
                    network_type = detect_network_type()
                    network_config = NETWORK_TYPES[network_type]
                    threshold = network_config['threshold']
                    dos_threshold = network_config['dos_threshold']
                    
                    # Calculate statistics for the last interval
                    packets_this_interval = total_packets - last_packet_count
                    anomalies_this_interval = anomaly_counter - last_anomaly_count
                    anomaly_rate = (anomaly_counter / total_packets) * 100
                    
                    # Create a comprehensive summary message
                    print("\n" + "="*80)
                    print(f" 📊 NETWORK MONITORING SUMMARY - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ")
                    print("="*80)
                    
                    # Network type information - ONLY in the summary
                    print(f"🌍 Network environment: {network_type}")
                    print(f"📊 Detection parameters - Threshold: {threshold}, DoS Threshold: {dos_threshold}")
                    
                    # Update statistics
                    print(f"📌 Packets analyzed (last {summary_interval}s): {packets_this_interval}")
                    print(f"🚨 Anomalies detected (last {summary_interval}s): {anomalies_this_interval}")
                    print(f"📈 Overall anomaly rate: {anomaly_rate:.2f}%")
                    
                    # Severity breakdown
                    print(f"\n🔍 Anomaly Severity Breakdown:")
                    print(f"   - {COLORS['red']}Critical: {severity_counts[3]}{COLORS['reset']}")
                    print(f"   - {COLORS['orange']}Warning: {severity_counts[2]}{COLORS['reset']}")
                    print(f"   - {COLORS['blue']}Info: {severity_counts[1]}{COLORS['reset']}")
                    
                    # Network metrics
                    if len(packet_sizes) > 0:
                        print(f"\n📍 Network Metrics:")
                        print(f"   - Average packet size: {np.mean(list(packet_sizes)):.0f} bytes")
                        print(f"   - Max packet size: {max(packet_sizes)} bytes")
                        
                    if len(intervals) > 0:
                        print(f"   - Average interval: {np.mean(list(intervals)):.4f}s")
                        print(f"   - Unique protocols: {len(set(protocol_tracker.values))}")
                    
                    # Top sources (last 1000 packets)
                    if len(all_features) > 0:
                        recent_src_ips = [f[0] for f in all_features[-1000:]]
                        if recent_src_ips:
                            from collections import Counter
                            top_sources = Counter(recent_src_ips).most_common(3)
                            print(f"\n📡 Top traffic sources (last 1000 packets):")
                            for src_hash, count in top_sources:
                                print(f"   - Source hash {src_hash}: {count} packets")
                    
                    print("\n" + "-"*80 + "\n")
                    
                    # Update for next interval
                    last_summary_time = current_time
                    last_packet_count = total_packets
                    last_anomaly_count = anomaly_counter
                
                # Periodic model retraining to adapt to network changes
                if initial_training_done and current_time - last_training_update > training_interval and len(all_features) > 10000:
                    # Convert to numpy arrays
                    features_np = np.array(all_features[-10000:])  # Use last 10000 packets
                    anomalous_np = np.array(is_anomalous_list[-10000:])
                    
                    # Update models with new data
                    print("\n🔄 Retraining models with updated network data...")
                    update_training(features_np, anomalous_np)
                    last_training_update = current_time
                    
                prev_timestamp = current_timestamp
                
            except Exception as e:
                logging.error(f"Packet processing error: {e}")
                continue
                
    except KeyboardInterrupt:
        print("\n👋 Stopping packet capture...")
        capture.close()
        if packet_bar and not packet_bar.disable:
            packet_bar.close()
    finally:
        print("\n📝 Final Summary:")
        print(f"Total packets analyzed: {total_packets}")
        print(f"Anomalies detected: {anomaly_counter} ({(anomaly_counter/max(1,total_packets))*100:.2f}%)")
        print(f"  - Critical: {severity_counts[3]}")
        print(f"  - Warning: {severity_counts[2]}")
        print(f"  - Info: {severity_counts[1]}")
        print(f"Anomaly details saved to: {LOG_PATH}")

if __name__ == "__main__":
    analyze_live_traffic()