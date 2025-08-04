# **SkyCast – Rainfall Prediction Web App**

**SkyCast** is a cutting-edge web application designed to predict rainfall probabilities using advanced machine learning techniques. Tailored for tropical and subtropical coastal climates.

---

## **Key Features**

* **Intuitive User Experience**  
  A thoughtfully designed interface that prioritizes user engagement and ease of navigation, making weather predictions accessible to everyone.

* **Advanced Machine Learning**  
  Utilizes an **XGBoost** classifier, trained on extensive real-world weather data, to deliver precise rainfall forecasts.

* **Real-Time Predictions**  
  Instantly generates rainfall probabilities, categorizes risk levels, and provides actionable advice tailored to the forecast.

* **Exploratory Data Analysis Tools**  
  Integrated utilities for visualizing and analyzing weather patterns, empowering users to understand trends and anomalies.

* **Robust Input Validation**  
  Ensures data integrity with checks that prevent common input errors, enhancing the reliability of predictions.

---

## **Installation**

To set up **SkyCast**, follow these steps:

```bash
git clone https://github.com/ayushsyntax/SkyCast.git
cd SkyCast
pip install -r requirements.txt
streamlit run app.py
```

The application will launch at `http://localhost:8501`.

---

## **How to Use**

1. Enter the following details:
   * `Date`
   * `Max Temp (°C)` / `Min Temp (°C)`
   * `Humidity (%)` (using a slider)
   * `Cloud Cover`: *Clear*, *Partly Cloudy*, *Cloudy*
   * `Wind Speed`: *Low*, *Moderate*, *High*

2. Click **“Predict Rainfall”**  
   The output will include:
   * **Probability (% chance)**
   * **Risk Level**: Low / Medium / High
   * **Advice** (e.g., “Carry an umbrella.”)

---

## **Model Insights**

### **Feature Engineering**

* Transforming the date into cyclical features to capture seasonal trends.
* Creating derived metrics such as temperature range and humidity combined with cloud cover.
* Implementing fallback defaults for missing data to maintain prediction accuracy.

### **Model Architecture**

* **Core Model**: **XGBoost Classifier**
* **Dataset**: Sourced from [Kaggle Playground S5E3](https://www.kaggle.com/competitions/playground-series-s5e3)
* **Output**: Provides a probability score along with a risk classification.

### **Postprocessing Techniques**

* Establishing risk thresholds based on calibrated cutoffs.
* Mapping recommendations to specific risk outcomes for user guidance.

---

## **Exploratory Data Analysis Toolkit**

Available in `projectrain.py`:

| Library      | Purpose                     |
| ------------ | --------------------------- |
| `numpy`      | Efficient numerical operations |
| `pandas`     | Data manipulation and analysis |
| `matplotlib` | Visualization of trends and patterns |
| `seaborn`    | Advanced statistical graphics |

---

## **System Overview**

| Component      | Description                  |
| -------------- | ---------------------------- |
| Model          | XGBoost Classifier           |
| Baseline       | Logistic Regression          |
| AUC-ROC        | XGB: > 0.90 · LogReg: ~0.88 |
| Framework      | Streamlit                    |
| Python Version | 3.x                          |
| Total Features | 16 (including engineered inputs) |

---

## **Directory Structure**

```
SkyCast/
├── app.py              # Main application logic
├── projectrain.py      # EDA and preprocessing scripts
├── final_model.pkl     # Serialized machine learning model
├── requirements.txt    # List of dependencies
└── README.md           # Project documentation
```

---

## **Performance Overview**

* **Dataset**: Comprised of 6 years of coastal climate data.
* **Class Distribution**: 75% rain / 25% no rain.
* **Imbalance Handling**: Addressed through class weighting.
* **Performance Metrics**:
  * **XGBoost AUC-ROC**: > 0.90
  * **Logistic Regression AUC-ROC**: ~0.88

---

## **Target Climate**

**SkyCast** is specifically designed for:
* **Regions**: Southeast Asia, South India, coastal Australia.
* **Weather Patterns**: High humidity, cloud cycles, and moderate wind conditions.

---

## **Modeling Considerations**

* Focused on a clean binary classification task.
* Key features influencing predictions include humidity, cloud cover, dew point, and temperature range.
* Seasonal signals are captured through cyclical time transformations.
* Utilizes stratified validation and class weights to ensure balanced predictions.
* Fully reproducible analysis available in `projectrain.py`.

---

## **Feedback and Contributions**

We welcome your suggestions, feedback, and contributions to enhance **SkyCast**. If you encounter any issues or have ideas for new features, please feel free to reach out. You can open a GitHub [issue](https://github.com/ayushsyntax/SkyCast/issues) or submit a pull request to contribute to the project.

---

