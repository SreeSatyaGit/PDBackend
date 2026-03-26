import torch
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from chronos import Chronos2Pipeline
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Chronos DataFrame API",
    description="API for time series forecasting with flexible horizons and frequencies",
    version="1.2.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
MODEL_PATH = "/Users/bharadwajanandivada/Downloads/chronos2_model"
pipeline = None

@app.on_event("startup")
async def load_model():
    global pipeline
    try:
        print(f"Loading Chronos2Pipeline from {MODEL_PATH}...")
        pipeline = Chronos2Pipeline.from_pretrained(
            MODEL_PATH,
            device_map="auto",
            dtype=torch.bfloat16, # Fixed deprecation warning
        )
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")

class ForecastRequest(BaseModel):
    context_data: List[Dict[str, Any]]
    future_data: Optional[List[Dict[str, Any]]] = None
    target_column: str = "Energia"
    id_column: str = "id"
    timestamp_column: str = "timestamp"
    prediction_length: int = 30
    freq: str = "1min" # New: dynamic frequency (D, H, ME, etc.)

@app.post("/predict")
async def predict(request: ForecastRequest):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet")
    
    try:
        # 1. Convert input context to DataFrame
        context_df = pd.DataFrame(request.context_data)
        if context_df.empty:
            raise HTTPException(status_code=400, detail="Context data is empty")

        # 2. Process Context DataFrame
        context_df[request.timestamp_column] = pd.to_datetime(context_df[request.timestamp_column])
        context_df = context_df.sort_values(request.timestamp_column).set_index(request.timestamp_column)
        
        # Resample based on the requested frequency (e.g. 1min, D, H)
        context_resampled = context_df.resample(request.freq).agg('last')
        
        # Forward fill numerical values
        context_resampled = context_resampled.ffill()
        
        # Forward and backward fill the ID column if it exists
        if request.id_column in context_resampled.columns:
            context_resampled[request.id_column] = context_resampled[request.id_column].ffill().bfill()
        else:
            context_resampled[request.id_column] = 'single_series'
            
        context_resampled = context_resampled.reset_index()

        # 3. Process Optional Future DataFrame
        future_resampled = None
        current_pred_len = request.prediction_length # Initialize with default or user-provided prediction_length
        
        if request.future_data and len(request.future_data) > 0:
            future_df = pd.DataFrame(request.future_data)
            future_df[request.timestamp_column] = pd.to_datetime(future_df[request.timestamp_column])
            future_df = future_df.set_index(request.timestamp_column)
            
            # Resample to same frequency
            future_resampled = future_df.resample(request.freq).agg('last').ffill().reset_index()
            
            # Align horizons
            if len(future_resampled) > current_pred_len:
                future_resampled = future_resampled.iloc[:current_pred_len]
            elif len(future_resampled) < current_pred_len:
                current_pred_len = len(future_resampled)
        
        # Final fallback for length
        if current_pred_len is None:
            current_pred_len = 30

        # 4. Perform Prediction using predict_df
        pred_df = pipeline.predict_df(
            context_resampled,
            future_df=future_resampled,
            prediction_length=current_pred_len,
            quantile_levels=[0.1, 0.5, 0.9],
            id_column=request.id_column,
            timestamp_column=request.timestamp_column,
            target=request.target_column
        )

        # 5. Format the result for JSON response
        pred_df[request.timestamp_column] = pred_df[request.timestamp_column].dt.strftime("%Y-%m-%d %H:%M:%S")
        
        return pred_df.to_dict(orient="records")

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": pipeline is not None}
