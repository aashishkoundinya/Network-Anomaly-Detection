from collections import deque
import numpy as np
from config import *

class MovingStats:
    def __init__(self, window_size=WINDOW_SIZE):
        self.window_size = window_size
        self.values = deque(maxlen=window_size)
        
    def add(self, value):
        self.values.append(value)
        
    def get_stats(self):
        if not self.values:
            return 0, 0, 0
        arr = np.array(self.values)
        return np.mean(arr), np.std(arr), np.median(arr)

class DataBuffers:
    def __init__(self):
        self.timing_buffer = deque(maxlen=DEQUE_SIZE)
        self.packet_sizes = deque(maxlen=DEQUE_SIZE)
        self.intervals = deque(maxlen=DEQUE_SIZE)
        self.anomaly_scores = deque(maxlen=DEQUE_SIZE)
        self.timestamps = deque(maxlen=DEQUE_SIZE)
        self.is_anomaly_buffer = deque(maxlen=DEQUE_SIZE)
        self.live_data = []
        
        # Visualization data storage
        self.visualization_data = {
            'timestamps': [],
            'intervals': [],
            'packet_sizes': [],
            'anomaly_scores': [],
            'is_anomaly': []
        }
        
        # Statistics
        self.interval_stats = {
            'mean': 0,
            'std': 0,
            'max': 0,
            'min': float('inf'),
            'median': 0,
            'q1': 0,
            'q3': 0
        }
        