import pandas as pd
import joblib

print("SMART TRAFFIC CONGESTION PREDICTION")
print("-----------------------------------")

# Load trained model
model = joblib.load("traffic_ai_model.pkl")
encoder = joblib.load("congestion_encoder.pkl")
features = joblib.load("traffic_features.pkl")

# Get user input
hour = int(input("Enter Hour (0-23): "))
day = int(input("Enter Day of Week (0-6): "))
vehicles = float(input("Enter Vehicle Count: "))
speed = float(input("Enter Average Speed: "))
weather = input("Enter Weather: ")
incident = input("Enter Incident (Yes/No): ")

# Prepare input
data = {
    "hour": hour,
    "dayofweek": day,
    "vehiclecount": vehicles,
    "averagespeed": speed
}

for col in features:
    if col.startswith("weather_"):
        data[col] = int(
            col == "weather_" + weather.lower()
        )

    elif col.startswith("incident_"):
        data[col] = int(
            col == "incident_" + incident.lower()
        )

# Create DataFrame
input_df = pd.DataFrame([data])
input_df = input_df.reindex(columns=features, fill_value=0)

# Predict
prediction = model.predict(input_df)

result = encoder.inverse_transform(prediction)

print("\n-----------------------------------")
print("PREDICTED TRAFFIC CONGESTION:", result[0])
print("-----------------------------------")