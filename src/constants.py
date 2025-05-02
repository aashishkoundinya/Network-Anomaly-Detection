import os
from collections import deque

# File paths
AUTOENCODER_PATH = "autoencoder_model.h5"
ISO_FOREST_PATH = "iso_forest_model.pkl"
DBSCAN_PATH = "dbscan_model.pkl"
SCALER_PATH = "scaler_model.pkl"
LOG_PATH = "anomalies.log"

# Buffer sizes and data storage
buffer_size = 25000  # Increased buffer for better model learning
live_data = []

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

# Interval statistics default values
interval_stats = {
    'mean': 0,
    'std': 0,
    'max': 0,
    'min': float('inf'),
    'median': 0,
    'q1': 0,
    'q3': 0
}

# Common network addresses and configurations
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

# Network type configurations
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

# Global threshold variables (will be updated dynamically)
threshold = 0.45  # Default threshold
dos_threshold = 150  # Default DoS threshold

# Global model variables
scaler = None
iso_forest = None 
autoencoder = None
dbscan = None
anomaly_logger = None