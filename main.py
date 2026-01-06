from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # <--- NEW IMPORT
from pydantic import BaseModel
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

app = FastAPI(title="Apple Stock Prediction API")

# --- THE FIX: ALLOW CROSS-ORIGIN REQUESTS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (POST, GET, etc.)
    allow_headers=["*"],
)
# --------------------------------------------

# Load artifacts
try:
    model = tf.keras.models.load_model('apple_lstm_model.keras')
    scaler_input = joblib.load('scaler_input.pkl')
    scaler_target = joblib.load('scaler_target.pkl')
    print("Model and Scalers loaded successfully.")
except Exception as e:
    print(f"CRITICAL ERROR: Could not load model/scalers. {e}")

class StockInput(BaseModel):
    recent_data: list[list[float]]

@app.post("/predict")
def predict_stock(payload: StockInput):
    try:
        data = np.array(payload.recent_data)
        
        if data.shape[1] != 5:
            raise HTTPException(status_code=400, detail="Expected 5 columns")

        df = pd.DataFrame(data, columns=['Open', 'High', 'Low', 'Close', 'Volume'])
        df_returns = df.pct_change()
        df_returns.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_returns.dropna(inplace=True)

        if len(df_returns) < 60:
            raise HTTPException(status_code=400, detail="Not enough data points.")

        last_60_returns = df_returns.tail(60).values
        scaled_input = scaler_input.transform(last_60_returns)
        model_input = scaled_input.reshape(1, 60, 5)

        prediction_scaled = model.predict(model_input)
        predicted_return = scaler_target.inverse_transform(prediction_scaled)[0][0]

        last_close_price = df.iloc[-1]['Close']
        predicted_price = last_close_price * (1 + predicted_return)

        signal = "NEUTRAL"
        confidence = "LOW"
        if predicted_return > 0.01: 
            signal = "BUY"
            confidence = "HIGH"
        elif predicted_return < -0.01:
            signal = "SELL"
            confidence = "HIGH"

        return {
            "status": "success",
            "predicted_price": float(predicted_price),
            "predicted_return_percentage": float(predicted_return * 100),
            "signal": signal,
            "confidence": confidence
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/")
def home():
    return {"message": "Stock Prediction API is Running"}

