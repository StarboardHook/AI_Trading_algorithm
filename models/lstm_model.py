import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import joblib

# File paths
DATA_PATH = "data/historical_stock_data.csv"
MODEL_PATH = "models/lstm_model.h5"
SCALER_PATH = "models/scaler.pkl"

def load_data(filename, sequence_length=50):
    """Load and preprocess stock data."""
    df = pd.read_csv(filename, index_col=0, parse_dates=True)
    df = df[['close']]
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_scaled = scaler.fit_transform(df)
    
    X, y = [], []
    for i in range(len(df_scaled) - sequence_length):
        X.append(df_scaled[i:i+sequence_length])
        y.append(df_scaled[i+sequence_length])
    
    joblib.dump(scaler, SCALER_PATH)
    return np.array(X), np.array(y), scaler

def build_lstm_model(input_shape):
    """Build an LSTM model for stock price prediction."""
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model():
    """Train the LSTM model and save it."""
    X, y, scaler = load_data(DATA_PATH)
    model = build_lstm_model((X.shape[1], X.shape[2]))
    model.fit(X, y, epochs=50, batch_size=32, validation_split=0.2)
    model.save(MODEL_PATH)
    print("Model trained and saved successfully!")

def predict_next_price(new_data):
    """Predict the next day's closing price using live data."""
    scaler = joblib.load(SCALER_PATH)
    model = tf.keras.models.load_model(MODEL_PATH)
    
    new_data_scaled = scaler.transform(new_data)
    new_data_scaled = np.array([new_data_scaled])
    
    prediction = model.predict(new_data_scaled)
    return scaler.inverse_transform(prediction)[0][0]

if __name__ == "__main__":
    train_model()
