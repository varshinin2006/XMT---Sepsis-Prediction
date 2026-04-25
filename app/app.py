import streamlit as st
import numpy as np
from utils import load_model

st.title("Sepsis + Pneumonia Prediction System")

icu_model = load_model("models/icu_model.pkl")

st.header("ICU Prediction")

if st.button("Run ICU Prediction"):
    sample = np.zeros((1, icu_model.n_features_in_))
    pred = icu_model.predict_proba(sample)[0][1]
    st.write("Sepsis Risk:", pred)

st.header("System Ready 🚀")