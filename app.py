import streamlit as st
import numpy as np
import joblib
import datetime

st.set_page_config(page_title="SkyCast", page_icon="🌧️", layout="centered")
st.title("🌧️ SkyCast –  Rain Predictor")

# Load model
try:
    model = joblib.load('final_model.pkl')
except Exception:
    st.error("Model could not be loaded.")
    st.stop()

def date_to_day_of_year(date_obj):
    return date_obj.timetuple().tm_yday

def map_cloud_choice(choice):
    mapping = {
        "Clear": 0,
        "Partly Cloudy": 50,
        "Cloudy": 100
    }
    return mapping.get(choice, 50)

def map_wind_choice(choice):
    mapping = {
        "Low": 5,
        "Moderate": 15,
        "High": 30
    }
    return mapping.get(choice, 10)

def preprocess(day, maxtemp, mintemp, humidity, cloud, windspeed):
    max_day = 366
    day_sin = np.sin(2 * np.pi * day / max_day)
    day_cos = np.cos(2 * np.pi * day / max_day)
    month = int(((day - 1) // 30) + 1)
    if month > 12: month = 12
    temp_range = maxtemp - mintemp
    humid_cloud = humidity * cloud
    # Use placeholder/default values for missing technical features
    pressure = 1015.0
    temparature = (maxtemp + mintemp) / 2
    dewpoint = mintemp  # Approximate
    winddirection = 180  # Assume average
    sunshine = 5  # Assume average sunshine
    
    return np.array([[
        day, pressure, maxtemp, temparature, mintemp, dewpoint,
        humidity, cloud, sunshine, winddirection, windspeed,
        temp_range, humid_cloud, day_sin, day_cos, month
    ]])

with st.form("weather_form"):
    input_date = st.date_input("Date", datetime.date.today())
    day = date_to_day_of_year(input_date)

    maxtemp = st.number_input("Max Temperature (°C)", -50.0, 60.0, 30.0)
    mintemp = st.number_input("Min Temperature (°C)", -50.0, 60.0, 20.0)
    humidity = st.slider("Humidity (%)", 0, 100, 75)
    cloud_choice = st.selectbox("Cloud Cover", ["Clear", "Partly Cloudy", "Cloudy"])
    cloud = map_cloud_choice(cloud_choice)
    wind_choice = st.selectbox("Wind Speed", ["Low", "Moderate", "High"])
    windspeed = map_wind_choice(wind_choice)

    submit = st.form_submit_button("🔮 Predict Rain")

if submit:
    if mintemp > maxtemp:
        st.error("Min temperature cannot be higher than max temperature.")
    else:
        X = preprocess(day, maxtemp, mintemp, humidity, cloud, windspeed)
        proba = model.predict_proba(X)[0, 1]
        st.metric("🌧️ Rain Probability", f"{proba:.0%}")
        if proba > 0.7:
            st.success("Very likely to rain. Bring an umbrella!")
        elif proba > 0.4:
            st.info("Possible rain. Stay prepared.")
        else:
            st.success("Rain unlikely. Enjoy your day!")
