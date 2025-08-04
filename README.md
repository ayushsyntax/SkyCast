# 🌧️ SkyCast - Rainfall Prediction Application

SkyCast is a robust and user-friendly application designed to forecast rainfall probability using advanced machine learning techniques. It is optimized for climates typical of tropical and subtropical coastal regions and adheres to the best practices in data science and modern web deployment.

## 🚀 Features

- **Intuitive, Minimalist Interface**: Effortless, streamlined design for all users
- **Reliable Machine Learning Predictions**: Trained on high-quality, realistic weather data
- **Quick, Actionable Output**: Returns rainfall probability, risk level, and clear recommendations
- **Comprehensive EDA Support**: Ready for data exploration with popular Python tools
- **Strong Input Validation**: Prevents errors and ensures data consistency

## 🛠️ Installation

Clone the repository:
```bash
git clone 
cd SkyCast
```

Install the required packages:
```bash
pip install -r requirements.txt
```

Launch the web application:
```bash
streamlit run app.py
```

Access SkyCast in your browser at `http://localhost:8501`

## 📊 How to Use

1. **Input Required Data:**  
   - **Date:** Choose from the calendar.
   - **Max Temperature (°C)**
   - **Min Temperature (°C)**
   - **Humidity (%):** Use a slider.
   - **Cloud Cover:** Select from “Clear”, “Partly Cloudy”, or “Cloudy”.
   - **Wind Speed:** Choose from “Low”, “Moderate”, or “High”.

2. **Press “Predict Rainfall”** to generate your forecast.

3. **Interpret Results:**  
   - **Rain Probability:** Shown in %  
   - **Risk Assessment:** Low, Medium, or High  
   - **Actionable Advice:** For example, “Bring an umbrella!”

## 🔍 How It Works

SkyCast merges user-friendly input with rigorous machine learning:
- **Feature Engineering:** Converts date to day-of-year, derives values like temperature range, humidity × cloud, and encodes seasonality with sine/cosine transforms. Technical fields not provided by the user are automatically imputed with realistic, climate-informed defaults.
- **Model Pipeline:** Prepared features are passed to an XGBoost classifier trained on the [Kaggle Playground Series - S5E3 dataset](https://www.kaggle.com/competitions/playground-series-s5e3), which outputs the rainfall probability for the selected day.
- **Output Handling:** The probability is mapped to a risk level and a tailored recommendation for the user. All inputs are validated for logical consistency, with clear error messages for illogical combinations (e.g., min temp higher than max temp).

## 📊 Exploratory Data Analysis (EDA)

SkyCast supports efficient data exploration with:
- **numpy:** High-performance numerical operations
- **pandas:** Flexible data manipulation and cleaning
- **matplotlib:** Industry-standard plotting
- **seaborn:** Beautiful and informative statistical graphics

These packages help you understand distributions, correlations, and time-based patterns of the weather data.

## 🔧 Technical Overview

- **Model:** XGBoost classifier (AUC-ROC: **>0.90**)
- **Baseline:** Logistic Regression (AUC-ROC: **~0.88**)
- **Features:** 16 (raw and engineered)
- **Framework:** Streamlit for deployment
- **Language:** Python 3.x

## 📁 Project Structure

```
SkyCast/
├── app.py              # Main app (Streamlit)
├── projectrain.py      # Full EDA and model training pipeline
├── final_model.pkl     # Trained pipeline/model
├── requirements.txt    # All library dependencies
└── README.md           # Documentation
```

## 🏆 Data and Model Performance

- **Dataset:** [Kaggle Playground Series - S5E3](https://www.kaggle.com/competitions/playground-series-s5e3)
- **Data Coverage:** Six years, tropical/subtropical coastal weather, numeric meteorological features
- **Imbalance:** 75% rain, 25% no rain (handled in modeling)
- **Model Results:**  
  - **XGBoost:** Cross-validated AUC-ROC > 0.90  
  - **Logistic Regression:** Cross-validated AUC-ROC ≈ 0.88

## 🌍 Climate Applicability

- **Best for:** Tropical and subtropical regions: Southeast Asia, southern India, coastal Australia, and similar zones
- **Key weather:** High humidity, frequent clouds, moderate seasonal wind, seasonal temperatures

## 📋 EDA and Modeling Insights

- Data is complete, clean, and suited for binary classification
- Strongest predictors: humidity, cloud cover, dew point, with seasonality effects well-captured
- Feature engineering encodes temporal cycles and atmospheric interactions
- Imbalance addressed with class weighting and stratified validation
- All findings and visualization code included in `projectrain.py`

