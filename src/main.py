import threading
from tqdm import tqdm
import pyshark
import numpy as np
from sklearn.preprocessing import RobustScaler
from config import *
from models import *
from data_structure import *
from visualization import *
from feature_extraction import *
from anomaly_detection import *

def analyze_live_traffic():
    """Main function to analyze network traffic"""
    # Initialize components
    data_buffers = DataBuffers()
    interval_tracker = MovingStats(window_size=WINDOW_SIZE)
    packet_size_tracker = MovingStats(window_size=WINDOW_SIZE)
    scaler = RobustScaler()
    
    print("🚀 Starting real-time anomaly detection...")
    
    # Initialize visualization
    visualizer = NetworkVisualizer(data_buffers)
    viz_thread = threading.Thread(target=visualizer.update, daemon=True)
    viz_thread.start()
    
    # Initialize models
    models_loaded, autoencoder, iso_forest = initialize_models()
    initial_training_done = models_loaded
    
    # Initialize network capture
    capture = pyshark.LiveCapture(interface=NETWORK_INTERFACE)
    prev_timestamp = None
    
    packet_bar = tqdm(total=MIN_TRAINING_PACKETS, desc="📡 Capturing Packets", unit="pkt")
    
    anomaly_counter = 0
    total_packets = 0
    
    try:
        for packet in capture.sniff_continuously():
            try:
                features, current_timestamp = extract_features(packet, prev_timestamp)
                if features is None:
                    continue
                    
                data_buffers.live_data.append(features)
                if current_timestamp:
                    data_buffers.timestamps.append(current_timestamp)
                
                data_buffers.packet_sizes.append(features[2])
                if prev_timestamp and current_timestamp:
                    interval = current_timestamp - prev_timestamp
                    data_buffers.intervals.append(interval)
                    interval_tracker.add(interval)
                
                packet_size_tracker.add(features[2])
                
                if len(data_buffers.live_data) > BUFFER_SIZE:
                    data_buffers.live_data.pop(0)
                
                packet_bar.update(1)
                total_packets += 1
                
                if len(data_buffers.live_data) < MIN_TRAINING_PACKETS and not initial_training_done:
                    prev_timestamp = current_timestamp
                    continue
                    
                if not initial_training_done:
                    packet_bar.close()
                    print("\n📊 Training models with initial data...")
                    
                    live_data_np = np.array(data_buffers.live_data)
                    live_data_scaled = scaler.fit_transform(live_data_np)
                    
                    print("🔍 Training Isolation Forest...")
                    iso_forest = create_isolation_forest()
                    iso_forest.fit(live_data_scaled)
                    joblib.dump(iso_forest, ISO_FOREST_PATH)
                    
                    print("🔧 Training Autoencoder...")
                    autoencoder = build_autoencoder(input_dim=live_data_scaled.shape[1])
                    
                    for epoch in tqdm(range(AUTOENCODER_EPOCHS), desc="Training Autoencoder"):
                        autoencoder.fit(
                            live_data_scaled, live_data_scaled,
                            epochs=1, batch_size=AUTOENCODER_BATCH_SIZE, 
                            shuffle=True, verbose=0
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
                
                data_buffers.anomaly_scores.append(mse)
                
                # Enhanced anomaly detection
                is_anomalous, confidence = is_anomaly(
                    features,
                    data_buffers,
                    interval_tracker,
                    packet_size_tracker,
                    isolation_score,
                    mse
                )
                
                data_buffers.is_anomaly_buffer.append(is_anomalous)
                
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
                    print(f"   - Average Interval: {np.mean(list(data_buffers.intervals)):.4f}s")
                    print(f"   - Average Packet Size: {np.mean(list(data_buffers.packet_sizes)):.0f} bytes\n")
                
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
