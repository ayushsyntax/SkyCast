
---

# 🌧️ SkyCast - Rainfall Prediction Application

**SkyCast** is a lightweight, intelligent, and user-friendly web application that predicts rainfall probability using state-of-the-art machine learning. Designed for tropical and subtropical coastal climates, it combines clean UI with accurate forecasts grounded in real weather data.

---

## 🚀 Key Features

* **🧭 Minimalist Interface**
  Clean, intuitive design suitable for users of all backgrounds

* **🔍 Smart Predictions**
  Powered by a well-tuned XGBoost model trained on realistic weather data

* **⚡ Instant Results**
  Predicts rainfall probability, risk category, and offers useful, actionable advice

* **📊 Built-In EDA Support**
  Includes tools for quick analysis of weather trends and patterns

* **✅ Robust Input Validation**
  Catches inconsistencies (e.g., minimum > maximum temperature) before processing

---

## 🛠️ Installation

**1. Clone the repository**

```bash
git clone https://github.com/your-username/SkyCast.git
cd SkyCast
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Run the app**

```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501` to access the app.

---

## 📋 How to Use

1. **Fill in the input fields:**

   * `Date`: Select via calendar
   * `Max Temperature (°C)`
   * `Min Temperature (°C)`
   * `Humidity (%)`: Use slider
   * `Cloud Cover`: Choose from *Clear*, *Partly Cloudy*, *Cloudy*
   * `Wind Speed`: Choose from *Low*, *Moderate*, *High*

2. **Click "Predict Rainfall"**
   The model will analyze your inputs and display:

   * 🌧️ **Rain Probability** (in %)
   * ⚠️ **Risk Level**: *Low*, *Medium*, or *High*
   * 💡 **Advice**: E.g., “Bring an umbrella!”

---

## 🔍 How It Works

SkyCast blends user input with engineered features for precise prediction:

* **Feature Engineering**

  * Converts date into cyclical time features (day of year → sine/cosine)
  * Calculates temperature range and combines features like humidity × cloud
  * Automatically handles missing or ambiguous values with realistic defaults

* **Machine Learning Model**

  * Uses an **XGBoost classifier**, trained on the [Kaggle Playground Series - S5E3](https://www.kaggle.com/competitions/playground-series-s5e3) dataset
  * Produces a probability score for rainfall based on the current inputs

* **Post-Prediction Mapping**

  * Translates probability into **risk categories**
  * Displays clear, contextual **recommendations** for users

---

## 📊 Exploratory Data Analysis (EDA)

SkyCast includes tools for deep insights into weather data:

| Tool         | Use Case                              |
| ------------ | ------------------------------------- |
| `numpy`      | High-performance calculations         |
| `pandas`     | Data manipulation and cleaning        |
| `matplotlib` | Visualization of trends and anomalies |
| `seaborn`    | Correlation heatmaps, distributions   |

All EDA logic and plots are available in `projectrain.py`.

---

## ⚙️ Technical Overview

| Component     | Description                          |
| ------------- | ------------------------------------ |
| **Model**     | XGBoost Classifier                   |
| **Baseline**  | Logistic Regression (for comparison) |
| **AUC-ROC**   | > 0.90 (XGBoost) / \~0.88 (LogReg)   |
| **Languages** | Python 3.x                           |
| **Framework** | Streamlit                            |
| **Features**  | 16 (including engineered ones)       |

---

## 📁 Project Structure

```
SkyCast/
├── app.py              # Streamlit app logic
├── projectrain.py      # EDA, preprocessing, training
├── final_model.pkl     # Trained model pipeline
├── requirements.txt    # All Python dependencies
└── README.md           # This documentation
```

---

## 🏆 Dataset and Model Performance

* **Source:** [Kaggle Playground Series - S5E3](https://www.kaggle.com/competitions/playground-series-s5e3)
* **Coverage:** Six years of meteorological data, coastal climates
* **Label Balance:** 75% rain, 25% no rain (imbalance handled via weighting)
* **Performance Metrics:**

  * **XGBoost AUC-ROC:** > **0.90**
  * **Logistic Regression AUC-ROC:** \~ **0.88**

---

## 🌍 Climate Focus

Designed for:

* **Tropical/Subtropical Zones**: e.g., Southeast Asia, South India, coastal Australia
* **Frequent Conditions**: High humidity, cyclical cloud cover, moderate wind patterns

---

## 🧠 Modeling and EDA Insights

* Clean, complete dataset suitable for binary classification
* Top predictors: humidity, cloud cover, dew point, temperature range
* Engineered temporal features enhance seasonal awareness
* Class imbalance managed with weighting and stratified validation
* Reproducible pipeline available in `projectrain.py`

---

## 📬 Feedback & Contributions

Contributions, bug reports, and feature requests are welcome!
Please open an [issue](https://github.com/your-username/SkyCast/issues) or submit a pull request.

---

