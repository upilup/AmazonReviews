# app.py — Amazon Reviews Sentiment (3-class)
import streamlit as st
import importlib

st.set_page_config(
    page_title="Amazon Reviews — Sentiment",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dukung run dari root ataupun dari folder deployment
try:
    home = importlib.import_module("src.home")
    eda  = importlib.import_module("src.eda")
    pred = importlib.import_module("src.prediction")
except Exception:
    import home, eda
    import prediction as pred

with st.sidebar:
    st.write("# Navigation")
    page = st.radio("Page", ["Home", "EDA", "Predict Sentiment"])

if page == "Home":
    home.home()
elif page == "EDA":
    eda.eda()
else:  # Predict Sentiment
    pred.run()