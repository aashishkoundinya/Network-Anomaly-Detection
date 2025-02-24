import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from config import VISUALIZATION_UPDATE_INTERVAL

class NetworkVisualizer:
    def __init__(self, data_buffers):
        self.data_buffers = data_buffers
        plt.ion()
        self.fig, ((self.ax1, self.ax2), (self.ax3, self.ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    def update(self):
        while True:
            try:
                # Clear all axes
                for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
                    ax.clear()
                
                # Ensure we have enough data points
                min_points = min(
                    len(self.data_buffers.timestamps),
                    len(self.data_buffers.intervals),
                    len(self.data_buffers.packet_sizes),
                    len(self.data_buffers.anomaly_scores)
                )
                
                if min_points == 0:
                    plt.pause(VISUALIZATION_UPDATE_INTERVAL)
                    continue
                    
                # Get the last min_points for visualization
                recent_timestamps = list(self.data_buffers.timestamps)[-min_points:]
                recent_intervals = list(self.data_buffers.intervals)[-min_points:]
                recent_packet_sizes = list(self.data_buffers.packet_sizes)[-min_points:]
                recent_anomaly_scores = list(self.data_buffers.anomaly_scores)[-min_points:]
                
                # Convert timestamps to datetime
                plot_timestamps = [datetime.fromtimestamp(ts) for ts in recent_timestamps]
                
                # Plot 1: Packet Intervals
                self.ax1.plot(plot_timestamps, recent_intervals, 'b-', alpha=0.6)
                self.ax1.set_title('Packet Intervals Over Time')
                self.ax1.set_xlabel('Time')
                self.ax1.set_ylabel('Interval (s)')
                self.ax1.tick_params(axis='x', rotation=45)
                
                # Plot 2: Packet Sizes
                self.ax2.plot(plot_timestamps, recent_packet_sizes, 'g-', alpha=0.6)
                self.ax2.set_title('Packet Sizes Over Time')
                self.ax2.set_xlabel('Time')
                self.ax2.set_ylabel('Size (bytes)')
                self.ax2.tick_params(axis='x', rotation=45)
                
                # Plot 3: Anomaly Scores Distribution
                if recent_anomaly_scores:
                    sns.histplot(recent_anomaly_scores, bins=50, ax=self.ax3)
                    self.ax3.set_title('Anomaly Score Distribution')
                    self.ax3.set_xlabel('Anomaly Score')
                    self.ax3.set_ylabel('Count')
                
                # Plot 4: Scatter plot of Packet Size vs Interval
                scatter = self.ax4.scatter(recent_intervals, recent_packet_sizes, 
                                    c=recent_anomaly_scores, cmap='viridis', alpha=0.6)
                self.ax4.set_title('Packet Size vs Interval')
                self.ax4.set_xlabel('Interval (s)')
                self.ax4.set_ylabel('Packet Size (bytes)')
                plt.colorbar(scatter, ax=self.ax4, label='Anomaly Score')
                
                plt.tight_layout()
                plt.draw()
                plt.pause(VISUALIZATION_UPDATE_INTERVAL)
                
            except Exception as e:
                print(f"Visualization error: {e}")
                plt.pause(VISUALIZATION_UPDATE_INTERVAL)
                