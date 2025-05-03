# Network Anomaly Detection

## Overview

This advanced network anomaly detection system combines multiple machine learning algorithms with statistical analysis to detect and classify unusual network traffic patterns in real-time. Built for cybersecurity professionals and network administrators, the system utilizes a multi-algorithmic approach with Autoencoder Neural Networks, Isolation Forest, and DBSCAN clustering to provide robust, adaptive anomaly detection with minimal false positives.

The system is designed to operate continuously on live network traffic, automatically adjusting to different network environments (enterprise, home, small office), and providing detailed classification of detected anomalies with supporting evidence.

## Tech Stack

Last run on these versions

- Python (v3.10.13)
- tensorflow (v2.18.0)
- scikit-learn (v1.6.1)
- pyshark (v0.6)
- numpy (v2.0.2)
- pandas (v2.2.3)
- matplotlib (v3.10.0)
- seaborn (v0.13.2)
- colorama (v0.4.6) (Optional)
- tqdm (v4.67.1)
- netifaces (v0.11.0) (Optional)

## Features

- Real-time detection of network anomalies with adaptive thresholding
- Multi-algorithm approach combining deep learning, statistical methods, and clustering
- Advanced feature extraction from network packets (12+ features per packet)
- Anomaly classification into 12+ specific attack types with severity levels
- Adaptive learning that adjusts to different network environments automatically
- Special handling for IoT traffic to reduce false positives
- Real-time visualization dashboard with multiple views:
    - Traffic rate with anomaly markers
    - Anomaly frequency timeline
    - Severity distribution
    - Packet size distribution
    - Top anomaly types
- Detailed logging with evidence and response recommendations
- Periodic model retraining to adapt to evolving network conditions
- Configurable detection sensitivity based on network type

## How it Works

### Architecture

1. constants.py: Configuration settings, thresholds, and data structures
2. models.py: Machine learning model definitions and training functions
3. utils.py: Utility functions for feature extraction, analysis, and visualization
4. main.py: Main program flow and packet processing logic

### Detection Algorithms

1. Autoencoder Neural Network

    - Deep learning model that learns normal network traffic patterns
    - Architecture: Input layer → 128 → 64 → 32 → 16 (bottleneck) → 32 → 64 → 128 → Output
    - Uses regularization (L2, dropout, batch normalization) to prevent overfitting
    - Anomaly score based on reconstruction error (Mean Absolute Error)

2. Isolation Forest

    - Ensemble of 150 isolation trees for outlier detection
    - Contamination parameter: 0.05 (assumes 5% anomalous traffic)
    - Randomly selects features and split points to isolate observations
    - Anomaly score based on average path length (shorter paths = more anomalous)

3. DBSCAN Clustering

    - Density-based spatial clustering algorithm
    - Parameters: eps=0.5, min_samples=10
    - Groups similar anomalies to identify attack patterns and campaigns
    - Helps distinguish between coordinated attacks and isolated incidents

## Weighted Scoring System

```python
# Example criteria with weights
criteria = {
    # High severity criteria (weight 2.0)
    'isolation_forest_extreme': (isolation_score < iso_threshold * 1.2, 2.0),
    'autoencoder_extreme': (mse > ae_threshold * 1.2, 2.0),
    'dos_pattern': (interval < 0.001 and len([i for i in list(intervals)[-100:] if i < 0.001]) > dos_threshold, 2.0),
    
    # Medium severity criteria (weight 1.5)
    'isolation_forest': (isolation_score < iso_threshold, 1.5),
    'autoencoder': (mse > ae_threshold, 1.5),
    'scanning_pattern': (features[7] == 2 and len(set(port_tracker.values[-100:])) > 15, 1.5),
    
    # Low severity criteria (weight 1.0)
    'interval_spike': (interval_zscore > 3, 1.0),
    'rare_protocol': (features[3] not in protocol_tracker.values and len(protocol_tracker.values) > 30, 1.0),
}
```

## Feature Extraction

`src_ip_hash` - Hash of source IP  
`dst_ip_hash` - Hash of destination IP  
`packet_size` - Total size in bytes  
`protocol` - Numeric hash of protocol  
`src_port` - Source port number  
`dst_port` - Destination port number  
`interval` - Time since last packet  
`tcp_flags` - TCP flags (SYN, ACK, etc.)  
`udp_length` - Size of UDP payload  
`payload_size` - Size of actual data  
`header_ratio` - Header size to packet size  
`transport_layer_hash` - Hash of transport layer type  

## Anomaly Classification

1. Critical Severity (Level 3)

    - DoS Attack: Excessive traffic patterns
    - Data Exfiltration: Unusual data transfer
    - Credential Stuffing: Multiple login attempts
    - Intrusion Attempt: Suspicious access pattern
    - Mass Registration: Multiple device registrations


2. Warning Severity (Level 2)

    - Scanning: Port/network scanning activity
    - Protocol Violation: Malformed packets
    - Excessive Traffic: Abnormal traffic volume
    - Unusual Destination: Connection to suspicious IPs


3. Info Severity (Level 1)

    - Timing Anomaly: Unusual packet timing
    - Size Anomaly: Unusual packet size
    - Behavioral: General behavioral anomaly


## Workflow

1. Initialization:

    - Load pre-trained models or prepare for training
    - Set up network capture interface
    - Initialize visualization thread

2. Initial Training (if no models exist):

    - Collect minimum 20,000 packets
    - Train all three models on baseline traffic
    - Save models for future use

3. Continuous Monitoring:

    - For each packet:
        - Extract features
        - Apply statistical analysis and ML models
        - Calculate weighted score
        - Determine if anomalous
        - Classify anomaly type if detected
        - Log, update statistics, update visualization

4. Periodic Updates:

    - Provide summary updates every 30 seconds
    - Retrain models hourly on recent data
    - Adjust thresholds based on network type

## Run

1. Clone the repository

    ```bash
    https://github.com/aashishkoundinya/Network-Anomaly-Detection

2. Install requirments

    ```bash
    pip install -r requirments.txt

3. Run the system

    ```bash
    python main.py
    ```

    or

    ```bash
    python network_anomaly_detection.py
