import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

print("SMART TRAFFIC AI PREDICTION SYSTEM")
print("-----------------------------------")

# Step 1: Load Dataset
df = pd.read_csv("traffic_data.csv")

# Step 2: Clean Column Names
df.columns = (
    df.columns.str.strip()
    .str.lower()
    .str.replace(r'[^a-z0-9]', '', regex=True)
)

# Step 3: Select Features
features = [
    "hour",
    "dayofweek",
    "vehiclecount",
    "averagespeed",
    "weather",
    "incident"
]

target = "congestionlevel"

# Step 4: Prepare Data
data = df[features + [target]].copy()
data = data.dropna()

# Step 5: Encode Categorical Values
data = pd.get_dummies(
    data,
    columns=["weather", "incident"],
    dtype=int
)

X = data.drop(columns=[target])
y = data[target].astype(str)

# Step 6: Encode Target
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Step 7: Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Step 8: Train AI Model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Step 9: Make Predictions
predictions = model.predict(X_test)

# Step 10: Evaluate Model
accuracy = accuracy_score(y_test, predictions)

print("\nModel Training Completed!")
print("Model Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        target_names=encoder.classes_
    )
)

# Step 11: Save Model
joblib.dump(model, "traffic_ai_model.pkl")
joblib.dump(encoder, "congestion_encoder.pkl")
joblib.dump(list(X.columns), "traffic_features.pkl")

print("\nAI Model Saved Successfully!")
print("TRAFFIC PREDICTION COMPLETED!")