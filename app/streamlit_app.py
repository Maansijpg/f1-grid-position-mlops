import streamlit as st
from src.predict import predict_grid

st.set_page_config(page_title="F1 Grid Predictor")
st.title("F1 Grid Position Predictor")

st.write(
    "This demo uses a trained model to predict a driver's grid position "
    "from lap time, tyre compound, and air temperature."
)

lap_time = st.number_input(
    "Representative lap time (seconds)",
    min_value=60.0,
    max_value=120.0,
    value=90.0,
    step=0.1,
)

compound = st.selectbox("Tyre compound", ["HARD", "MEDIUM", "SOFT"])

air_temp = st.number_input(
    "Air temperature (°C)",
    min_value=10.0,
    max_value=50.0,
    value=30.0,
    step=0.5,
)

if st.button("Predict grid position"):
    grid = predict_grid(lap_time, compound, air_temp)
    st.success(f"Predicted grid position: {grid}")

