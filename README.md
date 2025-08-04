
---

# **SkyCast – Rainfall Prediction Web App**

A lightweight application for predicting rainfall probability using machine learning. Built with simplicity in mind, **SkyCast** delivers real-time forecasts optimized for tropical and subtropical coastal climates. It pairs clean interface design with data-driven accuracy.

---

## **Features**

* **Minimal UI**
  Straightforward layout. No distractions. Built for usability.

* **ML-Driven Forecasts**
  Core predictions powered by an XGBoost classifier trained on real-world weather data.

* **Instant Output**
  Generates rainfall probability, assigns a risk level, and suggests practical advice.

* **EDA Utilities**
  Built-in tools for quick visual and statistical exploration of weather patterns.

* **Input Sanity Checks**
  Prevents errors like inverted temperature ranges or empty fields.

---

## **Installation**

```bash
git clone https://github.com/your-username/SkyCast.git
cd SkyCast
pip install -r requirements.txt
streamlit run app.py
```

App will launch at `http://localhost:8501`.

---

## **Usage**

1. Provide the following inputs:

   * `Date`
   * `Max Temp (°C)` / `Min Temp (°C)`
   * `Humidity (%)` (slider)
   * `Cloud Cover`: *Clear*, *Partly Cloudy*, *Cloudy*
   * `Wind Speed`: *Low*, *Moderate*, *High*

2. Click **“Predict Rainfall”**
   Outputs:

   * **Probability (% chance)**
   * **Risk Level**: Low / Medium / High
   * **Advice** (e.g., “Carry an umbrella.”)

---

## **Behind the Model**

### **Feature Engineering**

* Date → cyclical features (day of year via sine/cosine)
* Derived metrics: temp range, humidity × cloud cover
* Fallback defaults for missing values

### **Model Architecture**

* Core: **XGBoost Classifier**
* Dataset: [Kaggle Playground S5E3](https://www.kaggle.com/competitions/playground-series-s5e3)
* Output: Probability score + risk class

### **Postprocessing**

* Risk thresholds based on calibrated cutoffs
* Recommendations mapped to risk outcome

---

## **EDA Toolkit**

Available in `projectrain.py`:

| Library      | Use                         |
| ------------ | --------------------------- |
| `numpy`      | Vectorized calculations     |
| `pandas`     | Data manipulation           |
| `matplotlib` | Line plots, trends          |
| `seaborn`    | Correlation + distributions |

---

## **System Overview**

| Component      | Detail                       |
| -------------- | ---------------------------- |
| Model          | XGBoost Classifier           |
| Baseline       | Logistic Regression          |
| AUC-ROC        | XGB: > 0.90 · LogReg: \~0.88 |
| Framework      | Streamlit                    |
| Python         | 3.x                          |
| Total Features | 16 (with engineered inputs)  |

---

## **Directory Layout**

```
SkyCast/
├── app.py              # Streamlit app logic
├── projectrain.py      # EDA + preprocessing pipeline
├── final_model.pkl     # Serialized model object
├── requirements.txt    # Dependencies list
└── README.md           # Documentation
```

---

## **Performance Snapshot**

* Dataset: 6 years of coastal climate data
* Class Ratio: 75% rain / 25% no rain
* Imbalance: Handled via class weighting
* Metrics:

  * **XGBoost AUC-ROC**: > 0.90
  * **LogReg AUC-ROC**: \~0.88

---

## **Target Climate**

Designed for:

* **Regions**: Southeast Asia, South India, coastal Australia
* **Patterns**: High humidity · Cloud cycles · Moderate wind

---

## **Modeling Notes**

* Clean binary classification task
* Top features: humidity, cloud cover, dew point, temp range
* Seasonal signal captured via cyclical time
* Stratified validation + class weights for balance
* Fully reproducible in `projectrain.py`

---

## **Feedback**

Suggestions and issues welcome.
Open a GitHub [issue](https://github.com/your-username/SkyCast/issues) or submit a pull request.

---


