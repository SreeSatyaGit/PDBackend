import requests
import json

URL = "http://localhost:8000/predict"

# Dummy historical data (e.g., a simple sine wave)
import math
historical_data = [math.sin(i * 0.1) for i in range(100)]

import datetime
now = datetime.datetime.now()
historical_timestamps = [(now - datetime.timedelta(minutes=15*i)).strftime("%Y-%m-%d %H:%M:%S") for i in range(100)]
historical_timestamps.reverse()

payload = {
    "context": historical_data,
    "timestamps": historical_timestamps,
    "prediction_length": 12
}

response = requests.post(URL, json=payload)

print(f"Status Code: {response.status_code}")
if response.status_code == 200:
    data = response.json()
    print("Forecast Median:")
    print(data["median"])
    print("\nForecast 80% Confidence Interval Lower:")
    print(data["lower_80"])
    print("\nForecast 80% Confidence Interval Upper:")
    print(data["upper_80"])
    print("\nForecast Timestamps:")
    print(data["forecast_timestamps"])
else:
    print(f"Error: {response.text}")
