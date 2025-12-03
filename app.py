import streamlit as st
import pandas as pd
import joblib
import json
import numpy as np

CLASSIFIER_PATH = "models/best_classifier.pkl"
REGRESSOR_PATH = "models/best_regressor.pkl"
BRAND_MODEL_MAP_PATH = "models/brand_model_map.json"

classifier = joblib.load(CLASSIFIER_PATH)
regressor = joblib.load(REGRESSOR_PATH)

with open(BRAND_MODEL_MAP_PATH, "r") as f:
    brand_model_map = json.load(f)

brand_model_map = {k.lower(): [m.lower() for m in v] for k, v in brand_model_map.items()}


st.set_page_config(page_title="Car Price Predictor", layout="wide")
st.title("Used Car Price & Category Prediction App")
st.write("Enter the car details below to predict price or classify it as High/Low price.")

st.sidebar.header("Input Features")

brand = st.sidebar.selectbox("Brand", sorted(brand_model_map.keys()))

model = st.sidebar.selectbox("Model", sorted(brand_model_map[brand]))

seller_type = st.sidebar.selectbox("Seller Type", ["dealer", "individual", "trustmark_dealer", "other"])
fuel_type = st.sidebar.selectbox("Fuel Type", ["petrol", "diesel", "cng", "other"])
transmission = st.sidebar.selectbox("Transmission Type", ["manual", "automatic"])

vehicle_age = st.sidebar.number_input("Vehicle Age (years)", min_value=0, max_value=25, value=5)
km_driven = st.sidebar.number_input("KM Driven", min_value=0, max_value=500000, value=50000)
mileage = st.sidebar.number_input("Mileage (kmpl)", min_value=5.0, max_value=40.0, value=18.0)
engine = st.sidebar.number_input("Engine (CC)", min_value=500, max_value=6000, value=1197)
max_power = st.sidebar.number_input("Max Power (bhp)", min_value=20, max_value=400, value=85)
seats = st.sidebar.selectbox("Seats", [2, 4, 5, 6, 7, 8])

input_data = pd.DataFrame([{
    "brand": brand,
    "model": model,
    "seller_type": seller_type.lower(),
    "fuel_type": fuel_type.lower(),
    "transmission_type": transmission.lower(),
    "vehicle_age": vehicle_age,
    "km_driven": km_driven,
    "mileage": mileage,
    "engine": engine,
    "max_power": max_power,
    "seats": seats
}])

#st.subheader("Predictions")

col1, col2 = st.columns(2)

with col1:
    st.write("### Price Prediction")
    if st.button("Predict Price"):
        price_pred = regressor.predict(input_data)[0]
        st.success(f"Estimated Price: **₹ {int(price_pred):,}**")

with col2:
    st.write("### High/Low Price Classification")
    if st.button("Classify Price Category"):
        class_pred = classifier.predict(input_data)[0]
        label = "High Price" if class_pred == 1 else "Low Price"
        color = "🟢" if class_pred == 1 else "🔵"
        st.success(f"{color} This car is predicted to be: **{label}**")

st.markdown("### Input Summary")
st.dataframe(input_data)
