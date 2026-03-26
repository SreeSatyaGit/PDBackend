import requests
import json
import datetime
import math

URL = "http://localhost:8001/predict"

# Generate dummy historical data (100 minutes of data)
now = datetime.datetime.now()
context_data = []
for i in range(100):
    ts = (now - datetime.timedelta(minutes=100-i)).strftime("%Y-%m-%d %H:%M:%S")
    context_data.append({
        "timestamp": ts,
        "Energia": 10.0 + math.sin(i * 0.1) + (i * 0.01),
        "Tot_ict": 5.0 + math.cos(i * 0.1),
        "id": "series_1"
    })

# Generate dummy future data (30 minutes of data for exogenous variable)
future_data = []
for i in range(30):
    ts = (now + datetime.timedelta(minutes=i)).strftime("%Y-%m-%d %H:%M:%S")
    future_data.append({
        "timestamp": ts,
        "Tot_ict": 5.0 + math.cos((100+i) * 0.1),
        "id": "series_1"
    })

payload = {
    "context_data": context_data,
    "future_data": future_data,
    "target_column": "Energia",
    "prediction_length": 30,
    "id_column": "id"
}

print("Sending request to API...")
try:
    response = requests.post(URL, json=payload)
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Received {len(data)} prediction rows.")
        print("\nFirst 5 predictions:")
        for row in data[:5]:
            print(row)
    else:
        print(f"Error: {response.text}")
except Exception as e:
    print(f"Request failed: {e}")
