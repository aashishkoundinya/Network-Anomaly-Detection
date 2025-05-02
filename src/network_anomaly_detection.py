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
import logging
from sklearn.cluster import DBSCAN
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Global logger variable
anomaly_logger = None

# Add this function to initialize logging once
def initialize_logging():
    """Initialize logging once to prevent duplicate handlers"""
    global anomaly_logger
    
    # Configure logging with handler once
    anomaly_logger = logging.getLogger('anomaly_logger')
    anomaly_logger.setLevel(logging.INFO)
    
    # Check if handler already exists
    if not anomaly_logger.handlers:
        # Create handler
        handler = logging.FileHandler(LOG_PATH, mode='a')
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] - %(message)s', 
                                     datefmt='%Y-%m-%d %H:%M:%S')
        handler.setFormatter(formatter)
        anomaly_logger.addHandler(handler)
    
    return anomaly_logger

def analyze_live_traffic():
    """Main function to analyze network traffic with enhanced capabilities"""
    global autoencoder, live_data, iso_forest, last_summary_time, threshold, dos_threshold
    
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
    import platform
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

# Constants and file paths
AUTOENCODER_PATH = "autoencoder_model.h5"
ISO_FOREST_PATH = "iso_forest_model.pkl"
DBSCAN_PATH = "dbscan_model.pkl"
SCALER_PATH = "scaler_model.pkl"
LOG_PATH = "anomalies.log"

# Enhanced buffers for better statistical analysis
timing_buffer = deque(maxlen=10000)
packet_sizes = deque(maxlen=10000)
intervals = deque(maxlen=10000)
anomaly_scores = deque(maxlen=10000)
timestamps = deque(maxlen=10000)
is_anomaly_buffer = deque(maxlen=10000)
feature_buffer = deque(maxlen=10000)  # Store all features for better analysis

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
buffer_size = 25000  # Increased buffer for better model learning
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

COMMON_MULTICAST_ADDRESSES = [
    "224.0.0.1",    # All hosts on the local network segment
    "224.0.0.251",  # mDNS (Multicast DNS)
    "224.0.0.252",  # LLMNR (Link-Local Multicast Name Resolution)
    "224.0.0.22",   # IGMP
    "239.255.255.250",  # SSDP (Simple Service Discovery Protocol)
    "255.255.255.255",  # Broadcast
    "0.0.0.0",  # Often used in mDNS or when IP's cant be parsed
]

# Common service ports that have legitimate bursts of traffic
COMMON_SERVICE_PORTS = {
    53: "DNS",
    5353: "mDNS",
    5355: "LLMNR",
    1900: "SSDP",
    67: "DHCP Server",
    68: "DHCP Client",
    137: "NetBIOS Name Service",
    138: "NetBIOS Datagram Service",
    139: "NetBIOS Session Service",
    445: "SMB",
}

iso_forest = None
autoencoder = None
dbscan = None  # Added for anomaly clustering/classification

# Anomaly classification dictionary with severity levels
ANOMALY_TYPES = {
    # High Severity (Level 3) - Critical threats requiring immediate attention
    'dos_attack': {
        'description': 'Potential DoS Attack - Excessive traffic pattern',
        'severity': 3,
        'severity_label': 'CRITICAL',
        'color': 'red',
        'requires_immediate_action': True
    },
    'data_exfiltration': {
        'description': 'Possible Data Exfiltration - Unusual data transfer',
        'severity': 3,
        'severity_label': 'CRITICAL',
        'color': 'red',
        'requires_immediate_action': True
    },
    'credential_stuffing': {
        'description': 'Credential Stuffing Attack - Multiple login attempts',
        'severity': 3,
        'severity_label': 'CRITICAL',
        'color': 'red',
        'requires_immediate_action': True
    },
    'intrusion_attempt': {
        'description': 'Potential Intrusion Attempt - Suspicious access pattern',
        'severity': 3,
        'severity_label': 'CRITICAL',
        'color': 'red',
        'requires_immediate_action': True
    },
    'mass_registration': {
        'description': 'Mass Device Registration - Multiple devices registering simultaneously',
        'severity': 3,
        'severity_label': 'CRITICAL',
        'color': 'red',
        'requires_immediate_action': True
    },
    
    # Medium Severity (Level 2) - Significant threats requiring investigation
    'scanning': {
        'description': 'Port/Network Scanning Activity',
        'severity': 2,
        'severity_label': 'WARNING',
        'color': 'orange',
        'requires_immediate_action': False
    },
    'protocol_violation': {
        'description': 'Protocol Violation or Malformed Packets',
        'severity': 2,
        'severity_label': 'WARNING',
        'color': 'orange',
        'requires_immediate_action': False
    },
    'excessive_traffic': {
        'description': 'Unusual Traffic Volume - Abnormal traffic pattern',
        'severity': 2,
        'severity_label': 'WARNING',
        'color': 'orange',
        'requires_immediate_action': False
    },
    'unusual_destination': {
        'description': 'Connection to Unusual Destination',
        'severity': 2,
        'severity_label': 'WARNING',
        'color': 'orange',
        'requires_immediate_action': False
    },
    
    # Low Severity (Level 1) - Minor anomalies worth monitoring
    'timing_anomaly': {
        'description': 'Timing-based Anomaly',
        'severity': 1,
        'severity_label': 'INFO',
        'color': 'blue',
        'requires_immediate_action': False
    },
    'size_anomaly': {
        'description': 'Packet Size Anomaly',
        'severity': 1,
        'severity_label': 'INFO',
        'color': 'blue',
        'requires_immediate_action': False
    },
    'behavioral': {
        'description': 'General Behavioral Anomaly',
        'severity': 1,
        'severity_label': 'INFO',
        'color': 'blue',
        'requires_immediate_action': False
    }
}

NETWORK_TYPES = {
    'ENTERPRISE': {
        'threshold': 0.35,  # More stringent for enterprise
        'dos_threshold': 100,  # Higher threshold for DoS detection
        'min_packets_for_training': 20000,
        'expected_patterns': ['high_traffic', 'uniform_protocols']
    },
    'HOME': {
        'threshold': 0.45,  # More relaxed for home networks
        'dos_threshold': 200,  # Much higher threshold for IoT devices
        'min_packets_for_training': 5000,  # Less data needed for pattern learning
        'expected_patterns': ['streaming', 'iot_bursts', 'cloud_sync']
    },
    'SMALL_OFFICE': {
        'threshold': 0.4,
        'dos_threshold': 150,
        'min_packets_for_training': 10000,
        'expected_patterns': ['cloud_services', 'email_traffic']
    }
}

# IoT device patterns that often trigger false positives
IOT_PATTERNS = {
    'ports': [554, 8080, 8008, 1900, 32400],  # RTSP, HTTP alt, SSDP, Plex
    'protocols': ['SSDP', 'MDNS', 'DHCP', 'LLMNR'],
    'multicast_ips': ['239.255.255.250', '224.0.0.251'],
    'common_sizes': [range(68, 80), range(400, 600), range(1400, 1500)]
}

def detect_network_type():
    """Automatically detect what type of network we're on"""
    # Analyze the first N packets to determine network type
    if len(live_data) < 1000:
        return 'HOME'  # Default to home for small samples
    
    # Count unique IPs in the last 5000 packets
    recent_packets = live_data[-5000:]
    unique_src_ips = set()
    for packet_data in recent_packets:
        if len(packet_data) > 0:
            unique_src_ips.add(packet_data[0])  # Source IP hash
    
    # Improved heuristics for network type detection
    unique_ip_count = len(unique_src_ips)
    port_variety = len(set(port_tracker.values))
    protocol_variety = len(set(protocol_tracker.values))
    
    # Enterprise networks typically have many unique IPs, ports and protocols
    if unique_ip_count > 50 or (unique_ip_count > 30 and protocol_variety > 10):
        return 'ENTERPRISE'
    # Home networks have fewer devices and usually fewer protocols
    elif unique_ip_count < 15:
        return 'HOME'
    else:
        return 'SMALL_OFFICE'

def is_iot_traffic(features, metadata):
    """Identify IoT device traffic patterns"""
    src_port = features[4]
    dst_port = features[5]
    packet_size = features[2]
    dst_ip = metadata.get('dst_ip', '')
    protocol = metadata.get('protocol_name', '')
    
    # Check against known IoT patterns
    if dst_port in IOT_PATTERNS['ports']:
        return True
    
    if dst_ip in IOT_PATTERNS['multicast_ips']:
        return True
    
    if protocol in IOT_PATTERNS['protocols']:
        return True
    
    # Check for common IoT packet sizes
    for size_range in IOT_PATTERNS['common_sizes']:
        if isinstance(size_range, range) and packet_size in size_range:
            return True
        elif isinstance(size_range, int) and packet_size == size_range:
            return True
    
    return False

def adjust_thresholds_for_network():
    """Dynamically adjust detection thresholds based on network type"""
    global threshold, dos_threshold
    
    network_type = detect_network_type()
    network_config = NETWORK_TYPES[network_type]
    
    threshold = network_config['threshold']
    dos_threshold = network_config['dos_threshold']
    
    print(f"🌐 Detected network type: {network_type}")
    print(f"📊 Adjusted threshold: {threshold}")
    print(f"🛡️ DoS threshold: {dos_threshold}")
    
    return network_type

class MovingStats:
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.values = deque(maxlen=window_size)
        
    def add(self, value):
        self.values.append(value)
        
    def get_stats(self):
        if not self.values:
            return 0, 0, 0, 0, 0
        arr = np.array(self.values)
        return np.mean(arr), np.std(arr), np.median(arr), np.percentile(arr, 25), np.percentile(arr, 75)

# Initialize moving statistics trackers with larger window for stability
interval_tracker = MovingStats(window_size=500)
packet_size_tracker = MovingStats(window_size=500)
port_tracker = MovingStats(window_size=500)
protocol_tracker = MovingStats(window_size=500)

def build_autoencoder(input_dim):
    """Build an improved autoencoder model with regularization"""
    # Input layer
    inputs = keras.Input(shape=(input_dim,))
    
    # Encoder
    x = layers.Dense(128, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Dense(64, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    
    # Bottleneck
    x = layers.Dense(32, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    encoded = layers.Dense(16, activation="relu", name="bottleneck")(x)
    
    # Decoder
    x = layers.Dense(32, activation="relu")(encoded)
    x = layers.BatchNormalization()(x)
    
    x = layers.Dense(64, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Dense(128, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    
    # Output layer
    outputs = layers.Dense(input_dim, activation="sigmoid")(x)
    
    # Create model
    model = keras.Model(inputs, outputs)
    
    # Compile with better loss function for outlier detection
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), 
                 loss=keras.losses.MeanSquaredError())
    
    return model

def initialize_models():
    """Initialize or load pre-trained models with enhanced error handling"""
    global iso_forest, autoencoder, scaler, dbscan
    
    try:
        if (os.path.exists(AUTOENCODER_PATH) and 
            os.path.exists(ISO_FOREST_PATH) and 
            os.path.exists(SCALER_PATH)):
            
            print("🔄 Loading pre-trained models...")
            autoencoder = keras.models.load_model(AUTOENCODER_PATH)
            iso_forest = joblib.load(ISO_FOREST_PATH)
            scaler = joblib.load(SCALER_PATH)
            
            # Load DBSCAN if available
            if os.path.exists(DBSCAN_PATH):
                dbscan = joblib.load(DBSCAN_PATH)
            else:
                dbscan = DBSCAN(eps=0.5, min_samples=5)
            
            return True
        return False
        
    except Exception as e:
        print(f"⚠️ Error loading models: {e}")
        logging.error(f"Failed to load models: {e}")
        return False

def extract_features(packet, prev_timestamp):
    """Extract enhanced features from a packet"""
    try:
        # Basic packet information
        packet_size = int(packet.length)
        
        # Extract protocol information more robustly
        try:
            protocol = int(hash(packet.highest_layer) % 1000)
            protocol_name = packet.highest_layer
        except:
            protocol = 0
            protocol_name = "unknown"
        
        # Enhanced IP handling
        src_ip = "0.0.0.0"
        dst_ip = "0.0.0.0"
        src_ip_hash = 0
        dst_ip_hash = 0
        
        if hasattr(packet, 'ip'):
            try:
                src_ip = packet.ip.src
                dst_ip = packet.ip.dst
                src_ip_hash = int(hash(src_ip) % 1000)
                dst_ip_hash = int(hash(dst_ip) % 1000)
            except:
                pass
        
        # Enhanced transport layer handling
        src_port = 0
        dst_port = 0
        tcp_flags = 0
        udp_length = 0
        transport_layer = "none"
        
        if hasattr(packet, 'transport_layer') and packet.transport_layer:
            transport_layer = packet.transport_layer.lower()
            
            try:
                src_port = int(getattr(packet[transport_layer], 'srcport', 0))
                dst_port = int(getattr(packet[transport_layer], 'dstport', 0))
                
                # TCP specific features
                if transport_layer == 'tcp' and hasattr(packet.tcp, 'flags'):
                    tcp_flags = int(packet.tcp.flags, 16)
                
                # UDP specific features
                if transport_layer == 'udp' and hasattr(packet.udp, 'length'):
                    udp_length = int(packet.udp.length)
            except:
                pass
        
        # Timestamp handling
        timestamp = float(packet.sniff_timestamp)
        interval = timestamp - prev_timestamp if prev_timestamp else 0
        
        # Calculate additional derived features
        payload_size = max(0, packet_size - 54)  # Approximate header size
        header_ratio = 0 if packet_size == 0 else (packet_size - payload_size) / packet_size
        
        # Create a more comprehensive feature vector
        features = [
            src_ip_hash, 
            dst_ip_hash, 
            packet_size, 
            protocol, 
            src_port, 
            dst_port, 
            interval,
            tcp_flags,                  # TCP flags for detecting scan patterns
            udp_length,                 # UDP length
            payload_size,               # Size of the packet payload
            header_ratio,               # Ratio of header to total size
            int(hash(transport_layer) % 100)  # Transport layer type
        ]
        
        # Store additional metadata for logging but not for model
        metadata = {
            'src_ip': src_ip,
            'dst_ip': dst_ip,
            'protocol_name': protocol_name,
            'transport_layer': transport_layer,
            'timestamp': timestamp
        }
        
        return features, timestamp, metadata
        
    except Exception as e:
        print(f"⚠️ Error extracting features: {e}")
        logging.error(f"Feature extraction error: {e}")
        return None, prev_timestamp, None

def update_visualization():
    """Enhanced visualization system focused on network traffic and anomaly severity"""
    plt.ion()
    
    # Create a larger figure with a more sophisticated layout
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3)
    
    # Create axes with specific layouts
    ax_traffic = fig.add_subplot(gs[0, :])  # Network traffic (full width, top row)
    ax_anomaly_timeline = fig.add_subplot(gs[1, :])  # Anomaly timeline (full width, middle row)
    ax_severity_pie = fig.add_subplot(gs[2, 0])  # Severity pie chart (bottom left)
    ax_packet_dist = fig.add_subplot(gs[2, 1])  # Packet size distribution (bottom middle)
    ax_top_anomalies = fig.add_subplot(gs[2, 2])  # Top anomaly types (bottom right)
    
    # Create figure title with real-time updates
    fig.suptitle('Network Anomaly Detection - Real-time Monitoring', fontsize=16)
    
    # Store data for severity tracking
    severity_counts = {'CRITICAL': 0, 'WARNING': 0, 'INFO': 0}
    anomaly_type_counts = {}
    
    # Window time (5 minutes) for rolling traffic rate calculation
    window_seconds = 300
    
    # Track number of anomalies detected in different time windows
    anomalies_last_minute = deque(maxlen=60)  # Store count of anomalies per second for last minute
    anomalies_last_hour = deque(maxlen=60)    # Store count of anomalies per minute for last hour
    
    # Initialize time trackers
    last_second = 0
    last_minute = 0
    current_second_anomalies = 0
    current_minute_anomalies = 0
    
    # For traffic rate calculation
    packets_per_window = deque(maxlen=window_seconds)
    
    while True:
        try:
            # Clear all axes
            for ax in [ax_traffic, ax_anomaly_timeline, ax_severity_pie, ax_packet_dist, ax_top_anomalies]:
                ax.clear()
            
            # Ensure we have enough data points
            if len(timestamps) < 10:
                # plt.pause(2)
                plt.pause(2)
                continue
            
            # Make copies of shared data to prevent mutation during iteration
            timestamps_copy = list(timestamps)
            intervals_copy = list(intervals)
            packet_sizes_copy = list(packet_sizes)
            anomaly_scores_copy = list(anomaly_scores)
            is_anomaly_buffer_copy = list(is_anomaly_buffer)
            
            # Convert timestamps to datetime for all data
            all_datetimes = [datetime.fromtimestamp(ts) for ts in timestamps_copy]
            
            # Get current time and calculate time windows
            now = datetime.now()
            window_start_time = now - pd.Timedelta(seconds=window_seconds)
            
            # =====================================================
            # 1. NETWORK TRAFFIC RATE PLOT WITH ANOMALY MARKERS
            # =====================================================
            
            # Calculate packets per second over the recent window
            packets_in_window = sum(1 for dt in all_datetimes if dt > window_start_time)
            packets_per_second = packets_in_window / min(window_seconds, len(timestamps_copy))
            packets_per_window.append(packets_per_second)
            
            # Create time series for traffic rate
            if len(timestamps_copy) >= 2:
                # Create a regular time series for the plot
                min_time = max(all_datetimes[0], now - pd.Timedelta(minutes=10))
                max_time = now
                
                # Create time bins (every 5 seconds) - Fixed the deprecated warning
                time_bins = pd.date_range(start=min_time, end=max_time, freq='5s')
                
                # Count packets in each bin
                packet_counts = []
                for i in range(len(time_bins)-1):
                    start, end = time_bins[i], time_bins[i+1]
                    count = sum(1 for dt in all_datetimes if start <= dt < end)
                    packet_counts.append(count / 5.0)  # packets per second
                
                # Plot traffic rate
                ax_traffic.plot(time_bins[:-1], packet_counts, 'b-', linewidth=2, alpha=0.7)
                ax_traffic.fill_between(time_bins[:-1], packet_counts, color='blue', alpha=0.2)
                
                # Get anomalies with timestamps
                anomaly_indices = [i for i, is_anom in enumerate(is_anomaly_buffer_copy) if is_anom]
                
                if anomaly_indices and len(feature_buffer) > max(anomaly_indices):
                    # Get metadata for these anomalies to determine severity
                    anomaly_times = [all_datetimes[i] for i in anomaly_indices if i < len(all_datetimes)]
                    
                    # Group anomalies by severity if we have classification data
                    critical_times = []
                    warning_times = []
                    info_times = []
                    
                    # Track current second and minute for anomaly counting
                    current_second = int(now.timestamp())
                    current_minute = current_second // 60
                    
                    # Reset counters if we've moved to a new second/minute
                    if current_second != last_second:
                        anomalies_last_minute.append(current_second_anomalies)
                        current_second_anomalies = 0
                        last_second = current_second
                        
                    if current_minute != last_minute:
                        anomalies_last_hour.append(current_minute_anomalies)
                        current_minute_anomalies = 0
                        last_minute = current_minute
                    
                    # Process each anomaly
                    for idx in anomaly_indices:
                        if idx >= len(feature_buffer):
                            continue
                            
                        # Get anomaly time
                        if idx < len(all_datetimes):
                            anom_time = all_datetimes[idx]
                            
                            # Count recent anomalies
                            if anom_time > now - pd.Timedelta(seconds=1):
                                current_second_anomalies += 1
                            if anom_time > now - pd.Timedelta(minutes=1):
                                current_minute_anomalies += 1
                            
                            # Determine severity based on anomaly type if available
                            # This requires that classify_anomaly() was called and returned data
                            severity = 1  # Default low severity
                            
                            # For demonstration, assign severities based on patterns
                            # In the actual code, this would use classification results
                            if idx < len(packet_sizes_copy):
                                packet_size = packet_sizes_copy[idx]
                                if packet_size > 1500:  # Large packet
                                    severity = 3  # Critical
                                    severity_counts['CRITICAL'] += 1
                                    critical_times.append(anom_time)
                                elif packet_size > 800:  # Medium packet
                                    severity = 2  # Warning
                                    severity_counts['WARNING'] += 1
                                    warning_times.append(anom_time)
                                else:
                                    severity = 1  # Info
                                    severity_counts['INFO'] += 1
                                    info_times.append(anom_time)
                        
                    # Plot anomalies with different colors based on severity - FIXED MARKERS
                    if critical_times:
                        # For each critical anomaly, draw a vertical line
                        for ctime in critical_times:
                            ax_traffic.axvline(x=ctime, color='red', linestyle='-', alpha=0.5, linewidth=1)
                        # Add markers at the top of the plot - Use standard markers instead of Unicode
                        y_max = ax_traffic.get_ylim()[1]
                        ax_traffic.scatter(critical_times, [y_max * 0.95] * len(critical_times), 
                                        color='red', marker='v', s=120, label='Critical')
                    
                    if warning_times:
                        for wtime in warning_times:
                            ax_traffic.axvline(x=wtime, color='orange', linestyle='-', alpha=0.3, linewidth=1)
                        y_max = ax_traffic.get_ylim()[1]
                        ax_traffic.scatter(warning_times, [y_max * 0.90] * len(warning_times), 
                                        color='orange', marker='s', s=80, label='Warning')
                    
                    if info_times:
                        y_max = ax_traffic.get_ylim()[1]
                        ax_traffic.scatter(info_times, [y_max * 0.85] * len(info_times), 
                                        color='blue', marker='o', s=50, label='Info')
                
                # Add legend if we have plotted anomalies
                if severity_counts['CRITICAL'] + severity_counts['WARNING'] + severity_counts['INFO'] > 0:
                    ax_traffic.legend(loc='upper left')
                
                # Format the traffic plot
                ax_traffic.set_title('Network Traffic Rate with Anomaly Markers')
                ax_traffic.set_xlabel('Time')
                ax_traffic.set_ylabel('Packets/sec')
                ax_traffic.tick_params(axis='x', rotation=30)
                ax_traffic.grid(True, alpha=0.3)
                
                # Add moving average line for trend
                if len(packet_counts) > 5:
                    window_size = min(len(packet_counts), 5)
                    moving_avg = pd.Series(packet_counts).rolling(window=window_size).mean().tolist()[window_size-1:]
                    moving_avg_times = time_bins[window_size-1:-1]
                    ax_traffic.plot(moving_avg_times, moving_avg, 'r-', linewidth=1.5, 
                                 label='Trend (5-point MA)')
                
                # Add alert count in the corner
                alert_text = (f"Alerts: {sum(severity_counts.values())} total\n"
                             f"Critical: {severity_counts['CRITICAL']}\n"
                             f"Warning: {severity_counts['WARNING']}\n"
                             f"Info: {severity_counts['INFO']}")
                ax_traffic.text(0.01, 0.99, alert_text, transform=ax_traffic.transAxes,
                             verticalalignment='top', horizontalalignment='left',
                             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # =====================================================
            # 2. ANOMALY TIMELINE - Heat map of anomaly frequency
            # =====================================================
            
            # Create a heat map of anomaly frequency over time
            if len(timestamps_copy) > 10:
                # Calculate time bins for the last hour (1-minute bins)
                hour_ago = now - pd.Timedelta(hours=1)
                time_bins_hour = pd.date_range(start=hour_ago, end=now, freq='1min')
                
                # Count anomalies in each bin
                anomaly_counts = []
                for i in range(len(time_bins_hour)-1):
                    start, end = time_bins_hour[i], time_bins_hour[i+1]
                    count = sum(1 for i, dt in enumerate(all_datetimes) 
                               if start <= dt < end and i < len(is_anomaly_buffer_copy) and is_anomaly_buffer_copy[i])
                    anomaly_counts.append(count)
                
                # Create heatmap-style timeline
                # Use different coloring based on count
                colors = []
                for count in anomaly_counts:
                    if count == 0:
                        colors.append('green')
                    elif count < 3:
                        colors.append('yellow')
                    elif count < 10:
                        colors.append('orange')
                    else:
                        colors.append('red')
                
                # Plot bars with height proportional to count
                bars = ax_anomaly_timeline.bar(time_bins_hour[:-1], anomaly_counts, 
                                          width=pd.Timedelta(minutes=1), 
                                          color=colors, alpha=0.7)
                
                # Add count labels to bars with significant counts
                for i, (count, bar) in enumerate(zip(anomaly_counts, bars)):
                    if count > 0:
                        height = bar.get_height()
                        ax_anomaly_timeline.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                                             str(count), ha='center', va='bottom', 
                                             fontsize=8)
                
                # Format the timeline
                ax_anomaly_timeline.set_title('Anomaly Frequency Timeline (Last Hour)')
                ax_anomaly_timeline.set_xlabel('Time')
                ax_anomaly_timeline.set_ylabel('Anomaly Count')
                ax_anomaly_timeline.tick_params(axis='x', rotation=30)
                ax_anomaly_timeline.grid(True, alpha=0.3, axis='y')
            
            # =====================================================
            # 3. SEVERITY PIE CHART
            # =====================================================
            
            # Create a pie chart of anomaly severity distribution
            if sum(severity_counts.values()) > 0:
                labels = []
                sizes = []
                colors = []
                
                for severity, count in severity_counts.items():
                    if count > 0:
                        labels.append(f"{severity} ({count})")
                        sizes.append(count)
                        if severity == 'CRITICAL':
                            colors.append('red')
                        elif severity == 'WARNING':
                            colors.append('orange')
                        else:
                            colors.append('blue')
                
                # Plot the pie chart
                wedges, texts, autotexts = ax_severity_pie.pie(
                    sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                    startangle=90, pctdistance=0.85)
                
                # Style the text
                for text in texts:
                    text.set_fontsize(9)
                for autotext in autotexts:
                    autotext.set_fontsize(9)
                    autotext.set_color('white')
                
                ax_severity_pie.set_title('Anomaly Severity Distribution')
                # Draw circle in the middle for a donut chart effect
                centre_circle = plt.Circle((0, 0), 0.5, fc='white')
                ax_severity_pie.add_artist(centre_circle)
                ax_severity_pie.axis('equal')
            else:
                ax_severity_pie.text(0.5, 0.5, "No anomalies\ndetected yet", 
                                 ha='center', va='center', fontsize=12)
                ax_severity_pie.axis('off')
            
            # =====================================================
            # 4. PACKET SIZE DISTRIBUTION
            # =====================================================
            
            # Create a histogram of packet sizes
            if packet_sizes_copy:
                # Get recent packet sizes for analysis
                recent_sizes = packet_sizes_copy[-1000:]
                
                # Create histogram
                n, bins, patches = ax_packet_dist.hist(recent_sizes, bins=30, 
                                                    alpha=0.7, color='skyblue', 
                                                    edgecolor='black', linewidth=0.5)
                
                # Add a line for the mean
                mean_size = np.mean(recent_sizes)
                ax_packet_dist.axvline(mean_size, color='red', linestyle='dashed', 
                                    linewidth=1, label=f'Mean: {mean_size:.0f}')
                
                # Add lines for standard deviations
                std_size = np.std(recent_sizes)
                ax_packet_dist.axvline(mean_size + 2*std_size, color='orange', 
                                    linestyle='dotted', linewidth=1, 
                                    label=f'+2σ: {mean_size + 2*std_size:.0f}')
                ax_packet_dist.axvline(mean_size - 2*std_size, color='orange', 
                                    linestyle='dotted', linewidth=1, 
                                    label=f'-2σ: {max(0, mean_size - 2*std_size):.0f}')
                
                # Format the histogram
                ax_packet_dist.set_title('Packet Size Distribution')
                ax_packet_dist.set_xlabel('Size (bytes)')
                ax_packet_dist.set_ylabel('Frequency')
                ax_packet_dist.legend(loc='upper right', fontsize=8)
                ax_packet_dist.grid(True, alpha=0.3)
            else:
                ax_packet_dist.text(0.5, 0.5, "Insufficient data\nfor distribution", 
                                ha='center', va='center', fontsize=12)
                ax_packet_dist.axis('off')
            
            # =====================================================
            # 5. TOP ANOMALY TYPES BAR CHART
            # =====================================================
            
            # Mock data for top anomaly types (in real implementation, this would come from classification)
            # This would be replaced with actual classification results
            if sum(severity_counts.values()) > 0:
                # For demonstration, create a simulated distribution of anomaly types
                anomaly_type_counts = {
                    'Size Anomaly': int(severity_counts['INFO'] * 0.8),
                    'Timing Anomaly': int(severity_counts['INFO'] * 0.2),
                    'Port Scanning': int(severity_counts['WARNING'] * 0.6),
                    'Protocol Violation': int(severity_counts['WARNING'] * 0.4),
                    'DoS Attack': int(severity_counts['CRITICAL'] * 0.7),
                    'Data Exfiltration': int(severity_counts['CRITICAL'] * 0.3)
                }
                
                # Filter out zero values
                anomaly_type_counts = {k: v for k, v in anomaly_type_counts.items() if v > 0}
                
                # Sort by count
                sorted_types = sorted(anomaly_type_counts.items(), key=lambda x: x[1], reverse=True)
                
                # Take top 5
                top_types = sorted_types[:5]
                
                # Colors based on severity
                colors = []
                for atype, _ in top_types:
                    if atype in ['DoS Attack', 'Data Exfiltration']:
                        colors.append('red')
                    elif atype in ['Port Scanning', 'Protocol Violation']:
                        colors.append('orange')
                    else:
                        colors.append('blue')
                
                # Create horizontal bar chart
                y_pos = range(len(top_types))
                ax_top_anomalies.barh(y_pos, [count for _, count in top_types], 
                                  color=colors, alpha=0.7)
                
                # Add labels
                ax_top_anomalies.set_yticks(y_pos)
                ax_top_anomalies.set_yticklabels([atype for atype, _ in top_types])
                
                # Add count labels
                for i, (_, count) in enumerate(top_types):
                    ax_top_anomalies.text(count + 0.1, i, str(count), va='center')
                
                # Format the chart
                ax_top_anomalies.set_title('Top Anomaly Types')
                ax_top_anomalies.set_xlabel('Count')
                ax_top_anomalies.invert_yaxis()  # Display with highest count at the top
                ax_top_anomalies.grid(True, alpha=0.3, axis='x')
            else:
                ax_top_anomalies.text(0.5, 0.5, "No anomalies\nclassified yet", 
                                   ha='center', va='center', fontsize=12)
                ax_top_anomalies.axis('off')
            
            # Add timestamp of last update
            fig.text(0.5, 0.01, f"Last Updated: {now.strftime('%Y-%m-%d %H:%M:%S')}", 
                    ha='center', fontsize=10)
            
            # Overall layout adjustments
            plt.tight_layout(rect=[0, 0.02, 1, 0.95])  # Leave space for title and timestamp
            plt.draw()
            plt.pause(2)
            
        except Exception as e:
            print(f"Visualization error: {e}")
            logging.error(f"Visualization error: {e}")
            plt.pause(2)

def is_anomaly(features, metadata, isolation_score, mse):
    """Enhanced anomaly detection with network type awareness"""
    global threshold, dos_threshold
    
    if len(intervals) < 50:
        return False, 0.0, None
    
    interval = features[6]
    packet_size = features[2]
    
    # Get current statistics
    int_mean, int_std, int_median, int_q1, int_q3 = interval_tracker.get_stats()
    size_mean, size_std, size_median, size_q1, size_q3 = packet_size_tracker.get_stats()
    
    # Calculate Z-scores with protection against division by zero
    interval_zscore = abs((interval - int_mean) / max(int_std, 0.001))
    size_zscore = abs((packet_size - size_mean) / max(size_std, 0.001))
    
    # Check if this is IoT traffic
    is_iot = is_iot_traffic(features, metadata)
    
    # Get percentiles of anomaly scores for adaptive thresholding
    if anomaly_scores:
        recent_scores = list(anomaly_scores)[-500:] if len(anomaly_scores) > 500 else list(anomaly_scores)
        ae_threshold = np.percentile(recent_scores, 90)
        iso_threshold = np.percentile(recent_scores, 10)
    else:
        ae_threshold = 0.3
        iso_threshold = -0.3
    
    # Multiple criteria for anomaly detection with weights based on severity
    criteria = {
        # High severity criteria (weight 2.0)
        'isolation_forest_extreme': (isolation_score < iso_threshold * 1.2, 2.0),
        'autoencoder_extreme': (mse > ae_threshold * 1.2, 2.0),
        
        # Adjusted DoS pattern detection based on network type
        'dos_pattern': (interval < 0.001 and len([i for i in list(intervals)[-100:] if i < 0.001]) > dos_threshold, 2.0),
        
        'extreme_size': (size_zscore > 6, 2.0),
        'extreme_timing': (interval_zscore > 6, 2.0),
        
        # Medium severity criteria (weight 1.5)
        'isolation_forest': (isolation_score < iso_threshold, 1.5),
        'autoencoder': (mse > ae_threshold, 1.5),
        'scanning_pattern': (features[7] == 2 and len(set(port_tracker.values[-100:])) > 15, 1.5),
        'payload_anomaly': (features[9] > size_q3 * 2.5, 1.5),
        
        # Lower severity criteria (weight 1.0)
        'interval_spike': (interval_zscore > 3, 1.0),
        'size_spike': (size_zscore > 3, 1.0),
        'combined_spike': ((interval_zscore + size_zscore) > 5, 1.0),
        'rare_protocol': (features[3] not in protocol_tracker.values and len(protocol_tracker.values) > 30, 1.0),
    }
    
    # Reduce weights for IoT traffic
    if is_iot:
        # IoT traffic often has burst patterns that look like anomalies
        for criterion, (condition, weight) in criteria.items():
            if criterion in ['dos_pattern', 'interval_spike', 'timing_anomaly']:
                criteria[criterion] = (condition, weight * 0.5)  # Reduce weight by half
    
    # Calculate weighted score
    score_sum = sum(weight for condition, weight in criteria.values() if condition)
    max_possible_score = sum(weight for _, weight in criteria.values())
    confidence = score_sum / max_possible_score
    
    # REMOVED DEBUG PRINTING - No more "Triggered criteria" messages
    
    # Determine if this is an anomaly
    is_anomalous = confidence >= threshold
    
    # If anomalous, classify it
    anomaly_classification = None
    if is_anomalous:
        anomaly_classification = classify_anomaly(features, metadata, confidence, isolation_score)
    
    return is_anomalous, confidence, anomaly_classification

def classify_anomaly(features, metadata, confidence, isolation_score):
    """Classify the type of anomaly based on features and scores with severity levels"""
    
    # Extract relevant features for classification
    interval = features[6]  # Interval
    packet_size = features[2]  # Packet size
    src_port = features[4]  # Source port
    dst_port = features[5]  # Destination port
    tcp_flags = features[7]  # TCP flags
    payload_size = features[9]  # Payload size
    
    # Get moving statistics
    int_mean, int_std, int_median, int_q1, int_q3 = interval_tracker.get_stats()
    size_mean, size_std, size_median, size_q1, size_q3 = packet_size_tracker.get_stats()
    
    # Calculate Z-scores
    int_zscore = abs((interval - int_mean) / max(int_std, 0.001))
    size_zscore = abs((packet_size - size_mean) / max(size_std, 0.001))
    
    # Classification rules with severity-based approach
    anomaly_types = []
    anomaly_details = {}
    
    # Collect recent network activity metrics for context
    if len(intervals) > 200:
        recent_intervals = list(intervals)[-200:]
        recent_packets = list(packet_sizes)[-200:]
        recent_ports = list(port_tracker.values)[-200:]
        unique_ports_contacted = len(set(recent_ports))
        
        # Check for multicast/broadcast addresses - don't flag these as unusual
        is_multicast = False
        if 'dst_ip' in metadata:
            dst_ip = metadata['dst_ip']
            if dst_ip in COMMON_MULTICAST_ADDRESSES:
                is_multicast = True
        
        # Check for common service ports that have legitimate bursts
        is_common_service = dst_port in COMMON_SERVICE_PORTS
        service_name = COMMON_SERVICE_PORTS.get(dst_port, "")
        
        # 1. DoS attack patterns (CRITICAL severity) - with improved detection
        rapid_packet_count = sum(1 for i in recent_intervals if i < 0.001)
        
        # Don't flag mDNS or other common services as DoS unless extremely unusual
        if (int_zscore < 0.5 and interval < 0.001 and rapid_packet_count > 200 and 
            not (is_multicast and is_common_service)):
            anomaly_types.append('dos_attack')
            anomaly_details['dos_evidence'] = f"Rapid packet sequence: {rapid_packet_count} packets with interval < 0.001s"
        
        # 2. Mass registration detection (CRITICAL severity)
        # Don't flag multicast service discovery as mass registration
        if len(port_tracker.values) > 100 and not is_multicast:
            # If we see multiple similar-sized registration-like packets in quick succession
            if (dst_port == 80 or dst_port == 443) and payload_size > 200:
                similar_sized_packets = sum(1 for p in recent_packets if abs(p - packet_size) < 20)
                if similar_sized_packets > 20 and interval < 0.05:
                    anomaly_types.append('mass_registration')
                    anomaly_details['registration_evidence'] = f"Potential mass registration: {similar_sized_packets} similar packets in rapid succession"
        
        # 3. Data exfiltration (CRITICAL severity) - improved detection
        # Don't flag large multicast packets as exfiltration
        if packet_size > size_q3 * 4 and payload_size > 2000 and not is_multicast:
            anomaly_types.append('data_exfiltration')
            anomaly_details['exfil_evidence'] = f"Very large payload ({payload_size} bytes) - {size_zscore:.1f}σ above normal"
        
        # 4. Intrusion attempts (CRITICAL severity) - complex pattern detection
        if tcp_flags in [2, 18] and unique_ports_contacted > 5:  # SYN or SYN-ACK flags with many ports
            suspicious_port_activity = False
            
            # Check for common suspicious ports (SSH, RDP, DB ports, etc.)
            suspicious_ports = [22, 23, 3389, 1433, 3306, 5432, 27017]
            targeted_port = 0
            
            for suspicious_port in suspicious_ports:
                if suspicious_port in recent_ports:
                    suspicious_port_activity = True
                    targeted_port = suspicious_port
                    break
            
            if suspicious_port_activity:
                anomaly_types.append('intrusion_attempt')
                anomaly_details['intrusion_evidence'] = f"Suspicious activity targeting port {targeted_port}"
        
        # 5. Credential stuffing (CRITICAL severity)
        if dst_port in [80, 443, 8080, 8443]:  # Web ports
            # Look for patterns of repeated similar-sized login attempts
            if 100 < payload_size < 500:  # Typical login payload size
                similar_sized_packets = sum(1 for p in recent_packets if abs(p - packet_size) < 30)
                if similar_sized_packets > 40:  # Increased threshold
                    anomaly_types.append('credential_stuffing')
                    anomaly_details['credential_evidence'] = f"Potential credential stuffing: {similar_sized_packets} similar-sized login attempts"
        
        # 6. Port scanning detection (WARNING severity)
        unique_ports_ratio = unique_ports_contacted / len(recent_ports) if recent_ports else 0
        if unique_ports_ratio > 0.4 and len(recent_ports) > 40 and tcp_flags == 2:  # More unique ports with SYN flag
            anomaly_types.append('scanning')
            anomaly_details['scan_evidence'] = f"Multiple ports accessed ({unique_ports_contacted} unique ports in last {len(recent_ports)} packets)"
        
        # 7. Protocol violations (WARNING severity)
        protocol = features[3]
        protocol_counts = {}
        for p in protocol_tracker.values:
            protocol_counts[p] = protocol_counts.get(p, 0) + 1
        
        if protocol not in protocol_counts or protocol_counts[protocol] < 3:
            # Don't flag common services as protocol violations
            if not is_common_service:
                anomaly_types.append('protocol_violation')
                anomaly_details['protocol_evidence'] = f"Unusual protocol: {metadata['protocol_name']}"
        
        # 8. Unusual destination (WARNING severity)
        if 'dst_ip' in metadata and not is_multicast:
            dst_ip = metadata['dst_ip']
            # Check if this IP is rarely seen
            if not (dst_ip.startswith('10.') or dst_ip.startswith('192.168.') or dst_ip.startswith('172.')):
                # External IP - could be more suspicious
                anomaly_types.append('unusual_destination')
                anomaly_details['destination_evidence'] = f"Connection to unusual external IP: {dst_ip}"
        
        # 9. Excessive traffic (WARNING severity) - don't flag multicast
        if len(packet_sizes) > 1000 and not is_multicast:
            recent_1000 = list(packet_sizes)[-1000:]
            avg_size = np.mean(recent_1000)
            if packet_size > avg_size * 5 and packet_size > 1500:
                anomaly_types.append('excessive_traffic')
                anomaly_details['traffic_evidence'] = f"Packet size ({packet_size} bytes) much larger than average ({avg_size:.0f} bytes)"
    
    # 10. Timing-based anomalies (INFO severity - lower priority)
    # For multicast, only flag extreme timing anomalies
    if (is_multicast and int_zscore > 10) or (not is_multicast and int_zscore > 5):
        anomaly_types.append('timing_anomaly')
        anomaly_details['timing_evidence'] = f"Interval ({interval:.4f}s) is {int_zscore:.1f}σ from mean"
    
    # 11. Size-based anomalies (INFO severity - lower priority)
    # For multicast, only flag extreme size anomalies
    if (size_zscore > 6 and not is_multicast) or (size_zscore > 12 and is_multicast):
        if 'excessive_traffic' not in anomaly_types:
            anomaly_types.append('size_anomaly')
            anomaly_details['size_evidence'] = f"Packet size ({packet_size} bytes) is {size_zscore:.1f}σ from mean"
    
    # 12. General behavioral anomaly (INFO severity - catch-all)
    if not anomaly_types and (confidence > 0.7 or isolation_score < -0.7):
        # Don't flag common multicast services for general behavior
        if not (is_multicast and is_common_service):
            anomaly_types.append('behavioral')
            anomaly_details['behavioral_evidence'] = f"General behavioral anomaly: score={confidence:.2f}, isolation={isolation_score:.2f}"
    
    # If we have detected anomaly types
    if anomaly_types:
        # Primary anomaly type - prioritize by severity
        # First check if any critical severity types are present
        critical_types = [t for t in anomaly_types if t in ['dos_attack', 'data_exfiltration', 'credential_stuffing', 'intrusion_attempt', 'mass_registration']]
        warning_types = [t for t in anomaly_types if t in ['scanning', 'protocol_violation', 'excessive_traffic', 'unusual_destination']]
        info_types = [t for t in anomaly_types if t in ['timing_anomaly', 'size_anomaly', 'behavioral']]
        
        if critical_types:
            primary_type = critical_types[0]  # Use the first critical type as primary
        elif warning_types:
            primary_type = warning_types[0]  # Use the first warning type as primary
        else:
            primary_type = info_types[0]  # Use the first info type as primary
        
        # Get type info
        type_info = ANOMALY_TYPES.get(primary_type, {
            'description': 'Unknown Anomaly',
            'severity': 1,
            'severity_label': 'INFO',
            'color': 'blue',
            'requires_immediate_action': False
        })
        
        # Add context for multicast traffic
        if is_multicast:
            if service_name:
                anomaly_details['context'] = f"This is {service_name} multicast traffic which can often have bursts of activity"
            else:
                anomaly_details['context'] = f"This is multicast traffic which can often have bursts of activity"
        
        # Full list of types for detailed logging
        all_types = ", ".join(anomaly_types)
        
        # Get severity info
        severity_level = type_info['severity']
        severity_label = type_info['severity_label']
        
        return {
            'primary_type': primary_type,
            'all_types': all_types,
            'description': type_info['description'],
            'severity': severity_level,
            'severity_label': severity_label,
            'color': type_info['color'],
            'requires_immediate_action': type_info['requires_immediate_action'],
            'details': anomaly_details
        }
    
    return None

def log_anomaly(features, metadata, confidence, classification, isolation_score, mse):
    """Log detailed anomaly information to file with severity levels"""
    global anomaly_logger
    
    timestamp = datetime.fromtimestamp(metadata['timestamp']).strftime('%Y-%m-%d %H:%M:%S.%f')
    
    # Default severity if classification is None or missing severity info
    severity_level = 1  # Default to INFO level
    severity_label = "INFO"
    severity_color = "blue"
    
    # Check if we have a valid classification
    if classification is None:
        # Create a basic classification if none exists
        classification = {
            'primary_type': 'behavioral',
            'all_types': 'behavioral',
            'description': 'General Behavioral Anomaly',
            'severity': 1,
            'severity_label': 'INFO',
            'color': 'blue',
            'requires_immediate_action': False,
            'details': {'behavioral_evidence': f"Anomaly detected with confidence {confidence:.2f}"}
        }
    
    # Extract severity information
    severity_level = classification.get('severity', 1)
    severity_label = classification.get('severity_label', 'INFO')
    severity_color = classification.get('color', 'blue')
    
    # Create log prefix based on severity
    if severity_level == 3:
        log_prefix = "[CRITICAL]"
    elif severity_level == 2:
        log_prefix = "[WARNING]"
    else:
        log_prefix = "[INFO]"
    
    # Base log message with severity information
    log_message = (
        f"{log_prefix} ANOMALY DETECTED [Confidence: {confidence:.2f}]\n"
        f"  Time: {timestamp}\n"
        f"  Type: {classification.get('primary_type', 'unknown')} - {classification.get('description', 'Unknown Anomaly')}\n"
        f"  Severity: {severity_label} (Level {severity_level})\n"
        f"  Additional Types: {classification.get('all_types', 'unknown')}\n"
        f"  Source: {metadata['src_ip']}:{features[4]} → Destination: {metadata['dst_ip']}:{features[5]}\n"
        f"  Protocol: {metadata['protocol_name']} ({metadata['transport_layer']})\n"
        f"  Packet Size: {features[2]} bytes (Payload: {features[9]} bytes)\n"
        f"  Interval: {features[6]:.6f}s\n"
        f"  MSE Score: {mse:.4f}, Isolation Score: {isolation_score:.4f}\n"
    )
    
    # Add evidence details from classification
    log_message += "  Evidence:\n"
    for key, value in classification.get('details', {'evidence': 'No detailed evidence available'}).items():
        log_message += f"    - {value}\n"
    
    # Add action recommendations based on severity
    if classification.get('requires_immediate_action', False):
        log_message += "\n  ACTION REQUIRED: This anomaly requires immediate investigation!\n"
    elif severity_level == 2:  # Warning
        log_message += "\n  RECOMMENDATION: Monitor this activity closely.\n"
    
    # Add separator for readability
    log_message += "-" * 80 + "\n"
    
    # Use the pre-configured logger instead of creating a new one
    # This is what was causing duplicate logs - we were creating multiple handlers
    if severity_level == 3:
        anomaly_logger.critical(log_message)
    elif severity_level == 2:
        anomaly_logger.warning(log_message)
    else:
        anomaly_logger.info(log_message)
    
    # Flush to ensure it's written
    for handler in anomaly_logger.handlers:
        handler.flush()
    
    # Return formatted message for terminal display with color information
    return {
        'message': log_message,
        'severity': severity_level,
        'color': severity_color
    }

def update_training(live_data_np, is_anomalous_list):
    """Periodically update the models to adapt to evolving traffic patterns"""
    global autoencoder, iso_forest, scaler, dbscan
    
    print("\n🔄 Updating models with new data...")
    
    # Prepare data, excluding confirmed anomalies to prevent model bias
    normal_indices = [i for i, is_anom in enumerate(is_anomalous_list) if not is_anom]
    
    if len(normal_indices) < 1000:
        print("Not enough normal data for retraining")
        return
    
    normal_data = live_data_np[normal_indices]
    
    # Update scaler with new normal data
    scaler = RobustScaler().fit(normal_data)
    normal_data_scaled = scaler.transform(normal_data)
    
    # Update Isolation Forest
    iso_forest = IsolationForest(
        n_estimators=150,  # Increased for better accuracy
        contamination=0.05,  # Lower contamination assumption
        max_samples=min(1000, len(normal_data)),
        random_state=42
    )
    iso_forest.fit(normal_data_scaled)
    
    # Update autoencoder
    input_dim = normal_data_scaled.shape[1]
    autoencoder = build_autoencoder(input_dim)
    
    # Add early stopping and learning rate reduction
    callbacks = [
        EarlyStopping(monitor='loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3, min_lr=0.0001)
    ]
    
    # Train with validation split for better generalization
    autoencoder.fit(
        normal_data_scaled, normal_data_scaled,
        epochs=30, 
        batch_size=64, 
        shuffle=True, 
        verbose=1,
        validation_split=0.2,
        callbacks=callbacks
    )
    
    # Update DBSCAN for clustering
    dbscan = DBSCAN(eps=0.5, min_samples=10)
    dbscan.fit(normal_data_scaled)
    
    # Save updated models
    autoencoder.save(AUTOENCODER_PATH)
    joblib.dump(iso_forest, ISO_FOREST_PATH)
    joblib.dump(dbscan, DBSCAN_PATH)
    joblib.dump(scaler, SCALER_PATH)
    
    print("✅ Model updates completed and saved!")

if __name__ == "__main__":
    analyze_live_traffic()