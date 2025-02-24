import numpy as np
from config import *

def is_anomaly(features, data_buffers, interval_tracker, packet_size_tracker, isolation_score, mse):
    """Enhanced anomaly detection using multiple criteria"""
    if len(data_buffers.intervals) < 50:
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
        'isolation_forest': isolation_score < np.percentile(list(data_buffers.anomaly_scores)[-100:] if data_buffers.anomaly_scores else [0], 5),
        'autoencoder': mse > np.percentile(list(data_buffers.anomaly_scores)[-100:] if data_buffers.anomaly_scores else [0], 95),
        'interval_spike': interval_zscore > ZSCORE_THRESHOLD,
        'size_spike': size_zscore > ZSCORE_THRESHOLD,
        'combined_spike': (interval_zscore + size_zscore) > COMBINED_ZSCORE_THRESHOLD
    }
    
    # Calculate confidence score (0 to 1)
    confidence = sum(criteria.values()) / len(criteria)
    
    # Require at least 3 criteria to be met for an anomaly
    is_anomalous = confidence >= ANOMALY_CONFIDENCE_THRESHOLD
    
    return is_anomalous, confidence
