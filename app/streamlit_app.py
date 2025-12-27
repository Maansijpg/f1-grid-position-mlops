import streamlit as st

st.set_page_config(page_title="F1 Grid Predictor")
st.title("F1 Grid Position Predictor")
st.write(
    "This demo uses a trained model to predict a driver's grid position "
    "from lap time, tyre compound, and air temperature."
)

