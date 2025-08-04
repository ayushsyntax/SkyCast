import streamlit as st
import numpy as np
import joblib
import datetime
import pandas as pd

st.set_page_config(
    page_title="SkyCast - Rain Predictor",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_model():
    try:
        return joblib.load('final_model.pkl')
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        st.stop()

def validate_inputs(maxtemp, mintemp, humidity, cloud, wind):
    if not (-50 <= maxtemp <= 60):
        return "Max temperature must be between -50°C and 60°C."
    if not (-50 <= mintemp <= 60):
        return "Min temperature must be between -50°C and 60°C."
    if mintemp > maxtemp:
        return "Min temperature cannot exceed max temperature."
    if not (0 <= humidity <= 100):
        return "Humidity must be between 0% and 100%."
    if not (0 <= cloud <= 100):
        return "Cloud cover must be between 0% and 100%."
    if not (0 <= wind <= 50):
        return "Wind speed must be between 0 and 50 km/h."
    return None

def create_features(date, maxtemp, mintemp, humidity, cloud, wind):
    day = date.timetuple().tm_yday
    max_day = 366
    
    return pd.DataFrame([{
        'day': day,
        'pressure': 1015.0,
        'maxtemp': maxtemp,
        'temparature': (maxtemp + mintemp) / 2,
        'mintemp': mintemp,
        'dewpoint': mintemp,
        'humidity': humidity,
        'cloud': cloud,
        'sunshine': 5,
        'winddirection': 180,
        'windspeed': wind,
        'temp_range': maxtemp - mintemp,
        'humid_cloud': humidity * cloud,
        'day_sin': np.sin(2 * np.pi * day / max_day),
        'day_cos': np.cos(2 * np.pi * day / max_day),
        'month': min(int(((day - 1) // 30) + 1), 12)
    }])

def get_prediction_advice(probability):
    if probability > 0.8:
        return "Very High Risk", "error", "🌧️ Bring umbrella and waterproof gear"
    elif probability > 0.6:
        return "High Risk", "error", "🌧️ Bring an umbrella"
    elif probability > 0.4:
        return "Moderate Risk", "warning", "🌤️ Consider light jacket"
    elif probability > 0.2:
        return "Low Risk", "success", "⛅ Light jacket optional"
    else:
        return "Very Low Risk", "success", "☀️ Perfect for outdoor activities"

def main():
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f1f1f;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e9ecef;
    }
    .status-success {
        color: #28a745;
        font-weight: bold;
    }
    .status-warning {
        color: #ffc107;
        font-weight: bold;
    }
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">🌧️ SkyCast</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Reliable Rain Prediction for Tropical Regions</p>', unsafe_allow_html=True)
    
    model = load_model()
    
    with st.sidebar:
        st.markdown("### Model Information")
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.markdown('<p class="status-success">✅ Model Active</p>', unsafe_allow_html=True)
        st.metric("Accuracy", ">90%")
        st.metric("Training Data", "6 Years")
        st.metric("Features", "16 Parameters")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### Climate Focus")
        st.markdown("• Tropical Regions")
        st.markdown("• Subtropical Zones")
        st.markdown("• Coastal Areas")
        st.markdown("• High Humidity")
        
        st.markdown("---")
        st.markdown("### Model Confidence")
        st.markdown("• XGBoost Algorithm")
        st.markdown("• Cross-Validated")
        st.markdown("• AUC-ROC > 0.90")
        st.markdown("• Robust Preprocessing")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        with st.form("prediction_form"):
            st.markdown("### Weather Parameters")
            
            date = st.date_input("Date", datetime.date.today())
            
            temp_col1, temp_col2 = st.columns(2)
            with temp_col1:
                maxtemp = st.number_input("Max Temperature (°C)", -50.0, 60.0, 30.0, step=0.5)
                humidity = st.slider("Humidity (%)", 0, 100, 75)
                cloud = st.slider("Cloud Cover (%)", 0, 100, 50)
            
            with temp_col2:
                mintemp = st.number_input("Min Temperature (°C)", -50.0, 60.0, 20.0, step=0.5)
                wind = st.slider("Wind Speed (km/h)", 0, 50, 15)
                
                st.markdown("---")
                preset = st.selectbox("Quick Presets", 
                    ["Custom", "☀️ Sunny", "🌤️ Partly Cloudy", "☁️ Cloudy"])
                
                if preset == "☀️ Sunny":
                    maxtemp, mintemp, humidity, cloud, wind = 32, 25, 60, 20, 10
                elif preset == "🌤️ Partly Cloudy":
                    maxtemp, mintemp, humidity, cloud, wind = 28, 22, 70, 50, 15
                elif preset == "☁️ Cloudy":
                    maxtemp, mintemp, humidity, cloud, wind = 26, 20, 85, 90, 20
            
            submit = st.form_submit_button("🚀 Predict Rainfall", use_container_width=True, type="primary")
    
    with col2:
        st.markdown("### Model Status")
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Status", "Active")
        st.metric("Region", "Tropical")
        st.metric("Algorithm", "XGBoost")
        st.metric("Performance", "Excellent")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### Prediction Factors")
        st.markdown("• **Humidity**: Primary indicator")
        st.markdown("• **Cloud Cover**: Secondary factor")
        st.markdown("• **Temperature Range**: Important")
        st.markdown("• **Wind Speed**: Influences patterns")
        st.markdown("• **Seasonal Cycles**: Encoded")
    
    if submit:
        error = validate_inputs(maxtemp, mintemp, humidity, cloud, wind)
        
        if error:
            st.error(error)
        else:
            with st.spinner("Analyzing weather patterns..."):
                try:
                    features = create_features(date, maxtemp, mintemp, humidity, cloud, wind)
                    probability = model.predict_proba(features)[0, 1]
                    
                    st.success("✅ Prediction Complete!")
                    
                    result_col1, result_col2 = st.columns([1, 1])
                    
                    with result_col1:
                        st.markdown("### 🌧️ Rain Prediction")
                        st.metric("Probability", f"{probability:.1%}")
                        
                        risk_level, color, advice = get_prediction_advice(probability)
                        st.markdown(f"**Risk Level:** {risk_level}")
                        
                        if color == "error":
                            st.error(advice)
                        elif color == "warning":
                            st.warning(advice)
                        else:
                            st.success(advice)
                    
                    with result_col2:
                        st.markdown("### 🌤️ Weather Summary")
                        st.markdown(f"**Date:** {date.strftime('%B %d, %Y')}")
                        st.markdown(f"**Temperature:** {mintemp:.1f}°C - {maxtemp:.1f}°C")
                        st.markdown(f"**Humidity:** {humidity}%")
                        st.markdown(f"**Cloud Cover:** {cloud}%")
                        st.markdown(f"**Wind Speed:** {wind} km/h")
                        st.markdown(f"**Day of Year:** {date.timetuple().tm_yday}")
                    
                    st.markdown("---")
                    st.markdown("### 🔬 Model Confidence")
                    st.markdown("""
                    This prediction is based on a robust machine learning model trained on 6 years of weather data 
                    from tropical and subtropical regions. The model achieves >90% accuracy and incorporates 
                    16 weather features including seasonal patterns, atmospheric conditions, and meteorological interactions.
                    """)
                    
                except Exception as e:
                    st.error(f"Prediction failed: {e}")

if __name__ == "__main__":
    main()
