import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import re

st.set_page_config(
    page_title="Smart Traffic Intelligence",
    page_icon="🚦",
    layout="wide"
)

# Load dataset
df = pd.read_csv("traffic_data.csv")

# Clean column names
df.columns = [
    re.sub(r'[^a-z0-9]', '', str(c).lower())
    for c in df.columns
]

st.title("🚦 Smart Traffic Incident and Congestion Intelligence System")
st.markdown("### AI-Based Traffic Monitoring and Prediction Dashboard")

st.divider()

# Dashboard metrics
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Traffic Records", len(df))

with col2:
    st.metric(
        "Average Vehicle Count",
        round(df["vehiclecount"].mean(), 2)
    )

with col3:
    st.metric(
        "Average Speed",
        round(df["averagespeed"].mean(), 2)
    )

st.divider()

# Congestion Analysis
st.subheader("Traffic Congestion Analysis")

fig, ax = plt.subplots()

df["congestionlevel"].value_counts().plot(
    kind="bar", ax=ax
)

ax.set_xlabel("Congestion Level")
ax.set_ylabel("Number of Records")

st.pyplot(fig)
plt.close(fig)

# Weather Analysis
st.subheader("Weather Analysis")

fig, ax = plt.subplots()

df["weather"].value_counts().plot(
    kind="bar", ax=ax
)

ax.set_xlabel("Weather")
ax.set_ylabel("Records")

st.pyplot(fig)
plt.close(fig)

# Incident Analysis
st.subheader("Traffic Incident Analysis")

fig, ax = plt.subplots()

df["incident"].value_counts().plot(
    kind="bar", ax=ax
)

ax.set_xlabel("Incident")
ax.set_ylabel("Records")

st.pyplot(fig)
plt.close(fig)

st.divider()

# AI Prediction
st.header("🤖 AI Traffic Congestion Prediction")

model = joblib.load("traffic_ai_model.pkl")
encoder = joblib.load("congestion_encoder.pkl")
features = joblib.load("traffic_features.pkl")

st.write("Enter traffic details to predict congestion.")

hour = st.slider("Hour", 0, 23, 18)
day = st.selectbox("Day of Week", list(range(7)))

vehicles = st.number_input(
    "Vehicle Count", min_value=0, value=500
)

speed = st.number_input(
    "Average Speed", min_value=0.0, value=35.0
)

weather_options = [
    c.replace("weather_", "")
    for c in features if c.startswith("weather_")
]

incident_options = [
    c.replace("incident_", "")
    for c in features if c.startswith("incident_")
]

weather = st.selectbox("Weather", weather_options)

incident = st.selectbox("Incident", incident_options)

if st.button("Predict Traffic Congestion"):

    data = {
        "hour": hour,
        "dayofweek": day,
        "vehiclecount": vehicles,
        "averagespeed": speed
    }

    for col in features:
        if col.startswith("weather_"):
            data[col] = int(
                col == "weather_" + weather
            )

        elif col.startswith("incident_"):
            data[col] = int(
                col == "incident_" + incident
            )

    input_df = pd.DataFrame([data])
    input_df = input_df.reindex(
        columns=features, fill_value=0
    )

    prediction = model.predict(input_df)
    result = encoder.inverse_transform(prediction)[0]

    st.success(
        f"Predicted Traffic Congestion: {result.upper()}"
    )

st.divider()

st.caption(
    "Smart Traffic Intelligence System | "
    "Data Warehousing | Data Mining | Artificial Intelligence"
)