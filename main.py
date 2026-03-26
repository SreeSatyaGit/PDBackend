import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from chronos import BaseChronosPipeline
import numpy as np
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Chronos Forecasting API",
    description="API for time series forecasting using local Chronos2Model",
    version="1.1.1"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permits all origins
    allow_credentials=True,
    allow_methods=["*"],  # Permits all methods
    allow_headers=["*"],  # Permits all headers
)

# Configuration
MODEL_PATH = "/Users/bharadwajanandivada/Downloads/chronos2_model"

# Global variable for the pipeline
pipeline = None


@app.on_event("startup")
async def load_model():
    global pipeline
    try:
        print(f"Loading BaseChronosPipeline from {MODEL_PATH}...")
        pipeline = BaseChronosPipeline.from_pretrained(
            MODEL_PATH,
            device_map="auto",
            dtype=torch.bfloat16,
        )
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise RuntimeError(f"Could not load model from {MODEL_PATH}")


class ForecastRequest(BaseModel):
    context: List[float]
    timestamps: List[str]  # ["2024-03-20 10:00:00", "2024-03-20 10:15:00", ...]
    prediction_length: int

class ForecastResponse(BaseModel):
    median: List[float]
    lower_80: List[float]
    upper_80: List[float]
    forecast_timestamps: List[str]


@app.post("/predict", response_model=ForecastResponse)
async def predict(request: ForecastRequest):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet")
    if len(request.context) == 0:
        raise HTTPException(status_code=400, detail="Context time series cannot be empty")
    if len(request.context) != len(request.timestamps):
        raise HTTPException(status_code=400, detail="Context and timestamps must have the same length")
    if request.prediction_length <= 0:
        raise HTTPException(status_code=400, detail="Prediction length must be greater than 0")

    try:
        # Convert timestamps to pandas DatetimeIndex to infer frequency
        ts_index = pd.to_datetime(request.timestamps)
        
        # If there's more than one point, we can infer the frequency
        if len(ts_index) >= 2:
            inferred_freq = pd.infer_freq(ts_index)
            # If freq can't be inferred directly (e.g. irregular), take the last diff
            if inferred_freq is None:
                inferred_freq = ts_index[-1] - ts_index[-2]
        else:
            raise HTTPException(status_code=400, detail="At least 2 timestamps are needed to infer frequency")

        # Generate future timestamps
        future_index = pd.date_range(
            start=ts_index[-1],
            periods=request.prediction_length + 1,
            freq=inferred_freq
        )[1:] # Exclude the last historical point
        
        forecast_timestamps = [t.strftime("%Y-%m-%d %H:%M:%S") for t in future_index]

        # Convert context to tensor with shape (batch_size, n_variates, seq_len)
        context_tensor = torch.tensor(request.context, dtype=torch.float32).reshape(1, 1, -1)
        
        # Chronos2 uses predict_quantiles for direct quantile forecasting
        # quantile_levels corresponds to [0.1, 0.5, 0.9] which maps to [lower_80, median, upper_80]
        quantiles, _ = pipeline.predict_quantiles(
            context_tensor,
            prediction_length=request.prediction_length,
            quantile_levels=[0.1, 0.5, 0.9],
        )
        
        # quantiles is a list of tensors [ [1, prediction_length, 3], ... ]
        # We extract the first series in the batch
        series_quantiles = quantiles[0][0] # shape (prediction_length, 3)
        
        lower_80 = series_quantiles[:, 0].tolist()
        median = series_quantiles[:, 1].tolist()
        upper_80 = series_quantiles[:, 2].tolist()

        return ForecastResponse(
            median=median,
            lower_80=lower_80,
            upper_80=upper_80,
            forecast_timestamps=forecast_timestamps
        )
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    if pipeline is not None:
        return {"status": "healthy", "model_loaded": True}
    else:
        return {"status": "unhealthy", "model_loaded": False}
