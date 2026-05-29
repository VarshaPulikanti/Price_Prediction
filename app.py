import json

import streamlit as st

from src.config import METRICS_PATH, MODEL_REGISTRY_PATH
from src.predict import predict_price
from src.train import train_model

st.set_page_config(page_title="House Price Predictor", layout="centered")
st.title("California Housing Price Prediction")
st.write("Portfolio-ready ML system with model benchmarking, evaluation, and inference.")

with st.sidebar:
    st.header("Model")
    if st.button("Train / Retrain Model"):
        metrics = train_model()
        st.success("Model trained and saved in models/")
        st.json(metrics)

    if MODEL_REGISTRY_PATH.exists():
        st.subheader("Model Benchmark (CV RMSE)")
        registry_data = json.loads(MODEL_REGISTRY_PATH.read_text(encoding="utf-8"))
        st.json(registry_data)

    if METRICS_PATH.exists():
        st.subheader("Latest Metrics")
        metrics_data = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        st.json(metrics_data)

st.subheader("Prediction Inputs")
med_inc = st.number_input("Median Income", min_value=0.1, max_value=20.0, value=3.8)
house_age = st.number_input("House Age", min_value=1.0, max_value=60.0, value=28.0)
ave_rooms = st.number_input("Average Rooms", min_value=1.0, max_value=15.0, value=5.4)
ave_bedrooms = st.number_input("Average Bedrooms", min_value=0.3, max_value=5.0, value=1.1)
population = st.number_input("Population", min_value=3.0, max_value=50000.0, value=1400.0)
ave_occup = st.number_input("Average Occupancy", min_value=0.5, max_value=20.0, value=3.2)
latitude = st.number_input("Latitude", min_value=32.0, max_value=42.0, value=34.2)
longitude = st.number_input("Longitude", min_value=-125.0, max_value=-113.0, value=-118.2)

if st.button("Predict House Price"):
    features = {
        "MedInc": med_inc,
        "HouseAge": house_age,
        "AveRooms": ave_rooms,
        "AveBedrms": ave_bedrooms,
        "Population": population,
        "AveOccup": ave_occup,
        "Latitude": latitude,
        "Longitude": longitude,
    }
    try:
        prediction = predict_price(features)
        st.success(f"Predicted median house value: ${prediction * 100000:,.0f}")
    except FileNotFoundError:
        st.error("Model is missing. Train the model from the sidebar first.")
