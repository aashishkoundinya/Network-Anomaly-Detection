from tensorflow import keras
from tensorflow.keras import layers
from sklearn.ensemble import IsolationForest
import joblib
import os
from config import *

def build_autoencoder(input_dim):
    """Build and compile the autoencoder model"""
    model = keras.Sequential([
        layers.Dense(128, activation="relu", input_shape=(input_dim,)),
        layers.Dropout(0.1),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.1),
        layers.Dense(32, activation="relu"),
        layers.Dense(16, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(128, activation="relu"),
        layers.Dense(input_dim, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

def create_isolation_forest():
    return IsolationForest(
        n_estimators=ISOLATION_FOREST_ESTIMATORS, 
        contamination=ISOLATION_FOREST_CONTAMINATION, 
        random_state=42
    )

def initialize_models():
    """Initialize or load pre-trained models"""
    try:
        if os.path.exists(AUTOENCODER_PATH) and os.path.exists(ISO_FOREST_PATH):
            print("🔄 Loading pre-trained models...")
            autoencoder = keras.models.load_model(AUTOENCODER_PATH)
            iso_forest = joblib.load(ISO_FOREST_PATH)
            return True, autoencoder, iso_forest
        return False, None, None
        
    except Exception as e:
        print(f"⚠️ Error loading models: {e}")
        return False, None, None
    