from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

# ---------------------------------------------------------
# 1. INITIAL SETUP & LOADING
# ---------------------------------------------------------
app = FastAPI(title="Apple Stock Prediction API")

# Load the Model (We use .keras format now)
# Ensure these files are in the same folder as main.py
try:
    model = tf.keras.models.load_model('apple_lstm_model.keras')
    scaler_input = joblib.load('scaler_input.pkl')
    scaler_target = joblib.load('scaler_target.pkl')
    print("Model and Scalers loaded successfully.")
except Exception as e:
    print(f"CRITICAL ERROR: Could not load model/scalers. {e}")
    # In production, we might want to crash here if files are missing

# ---------------------------------------------------------
# 2. DATA STRUCTURES
# ---------------------------------------------------------
class StockInput(BaseModel):
    # We expect a list of lists: [[Open, High, Low, Close, Volume], ...]
    # We need AT LEAST 61 data points to generate 60 returns.
    # It is safe to send more (e.g., 100 days); we will just take the latest.
    recent_data: list[list[float]]

# ---------------------------------------------------------
# 3. PREDICTION ENDPOINT
# ---------------------------------------------------------
@app.post("/predict")
def predict_stock(payload: StockInput):
    try:
        # A. CONVERT INPUT TO DATAFRAME
        # We assume the user sends data in chronological order (oldest -> newest)
        data = np.array(payload.recent_data)
        
        # Validation: We need 5 columns
        if data.shape[1] != 5:
            raise HTTPException(status_code=400, detail=f"Expected 5 columns (Open, High, Low, Close, Vol), got {data.shape[1]}")

        df = pd.DataFrame(data, columns=['Open', 'High', 'Low', 'Close', 'Volume'])

        # B. PREPROCESSING (Exact same logic as training)
        # Calculate Percentage Returns
        df_returns = df.pct_change()
        
        # Clean Infinities (if Volume was 0) and NaNs
        df_returns.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_returns.dropna(inplace=True)

        # C. CHECK DATA LENGTH
        # The model expects input shape (1, 60, 5) -> 60 time steps
        if len(df_returns) < 60:
            raise HTTPException(
                status_code=400, 
                detail=f"Not enough valid data. Need 61+ raw rows to create 60 returns. You sent {len(data)} rows, resulting in {len(df_returns)} returns."
            )

        # D. PREPARE INPUT FOR MODEL
        # Take exactly the last 60 returns
        last_60_returns = df_returns.tail(60).values
        
        # Scale the inputs (using the loaded input scaler)
        scaled_input = scaler_input.transform(last_60_returns)
        
        # Reshape for LSTM: (1 sample, 60 time steps, 5 features)
        model_input = scaled_input.reshape(1, 60, 5)

        # E. PREDICT
        # The model outputs a SCALED return
        prediction_scaled = model.predict(model_input)
        
        # Inverse transform to get the actual Predicted Return % (e.g., 0.015 for 1.5%)
        # Note: scaler_target expects 2D array, so we pass the result directly
        predicted_return = scaler_target.inverse_transform(prediction_scaled)[0][0]

        # F. CALCULATE IMPLIED PRICE
        # Price_Tomorrow = Price_Today * (1 + Predicted_Return)
        last_close_price = df.iloc[-1]['Close'] # The most recent known price
        predicted_price = last_close_price * (1 + predicted_return)

        # G. GENERATE SIGNAL
        # Thresholds: Buy if > 1.0%, Sell if < -1.0%
        signal = "NEUTRAL"
        confidence = "LOW"
        
        if predicted_return > 0.01: 
            signal = "BUY"
            confidence = "HIGH"
        elif predicted_return < -0.01:
            signal = "SELL"
            confidence = "HIGH"

        # H. RETURN JSON
        return {
            "status": "success",
            "last_actual_price": float(last_close_price),
            "predicted_price": float(predicted_price),
            "predicted_return_percentage": float(predicted_return * 100),
            "signal": signal,
            "confidence": confidence
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}

# ---------------------------------------------------------
# 4. HEALTH CHECK (Optional but good for Render)
# ---------------------------------------------------------
@app.get("/")
def home():
    return {"message": "Stock Prediction API is Running"}