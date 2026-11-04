import pandas as pd
import matplotlib.pyplot as plt
import re

print("SMART TRAFFIC INTELLIGENCE SYSTEM")

# Load dataset
df = pd.read_csv("traffic_data.csv")

# Clean column names
df.columns = [
    re.sub(r'[^a-z0-9]', '', str(c).lower())
    for c in df.columns
]

# Display dataset
print("\nDataset:")
print(df.head())

# Vehicle Count Analysis
print("\nAverage Vehicle Count:")
print(df["vehiclecount"].mean())

# Average Speed Analysis
print("\nAverage Speed:")
print(df["averagespeed"].mean())

# Congestion Analysis
print("\nCongestion Levels:")
print(df["congestionlevel"].value_counts())

# Weather Analysis
print("\nWeather Conditions:")
print(df["weather"].value_counts())

# Incident Analysis
print("\nTraffic Incidents:")
print(df["incident"].value_counts())

# Graph 1: Congestion Analysis
plt.figure(figsize=(8, 5))
df["congestionlevel"].value_counts().plot(kind="bar")
plt.title("Traffic Congestion Analysis")
plt.xlabel("Congestion Level")
plt.ylabel("Number of Records")
plt.tight_layout()
plt.show()

# Graph 2: Vehicle Count
plt.figure(figsize=(8, 5))
df["vehiclecount"].plot(kind="hist", bins=10)
plt.title("Vehicle Count Analysis")
plt.xlabel("Vehicle Count")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Graph 3: Average Speed
plt.figure(figsize=(8, 5))
df["averagespeed"].plot(kind="hist", bins=10)
plt.title("Average Speed Analysis")
plt.xlabel("Average Speed")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

print("\nTRAFFIC ANALYSIS COMPLETED SUCCESSFULLY!")