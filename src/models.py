import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import DBSCAN

from constants import *

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
        else:
            # Initialize default models
            scaler = RobustScaler()
            return False
        
    except Exception as e:
        print(f"⚠️ Error loading models: {e}")
        if anomaly_logger:
            anomaly_logger.error(f"Failed to load models: {e}")
        return False

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