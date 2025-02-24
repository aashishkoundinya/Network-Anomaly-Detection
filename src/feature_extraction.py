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
    