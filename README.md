# ⚡ GridSense — Electricity Load Forecasting & Anomaly Detection

GridSense is a machine learning-based system designed to analyze electricity demand, forecast electricity load, and detect unusual consumption patterns.

The project combines data preprocessing, feature engineering, machine learning, anomaly detection, and an interactive Streamlit dashboard into a single workflow.

## 🚀 Features

* Electricity demand analysis
* Data preprocessing and cleaning
* Time-series feature engineering
* Electricity load forecasting
* Anomaly detection using machine learning
* Feature importance analysis
* Model evaluation using regression metrics
* Interactive Streamlit dashboard
* Visualization of electricity demand and detected anomalies

## 🛠️ Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **Joblib**
* **Plotly**
* **Streamlit**
* **Jupyter Notebook**

## 📂 Project Structure

```text
GridSense/
│
├── app.py
├── GridSense.ipynb
├── requirements.txt
│
├── models/
│   └── Trained model files
│
├── outputs/
│   └── Processed data and model outputs
│
├── README.md
├── LICENSE
└── .gitignore
```

## 🔄 Project Workflow

```text
Raw Electricity Data
        ↓
Data Preprocessing
        ↓
Exploratory Data Analysis
        ↓
Feature Engineering
        ↓
Forecasting Model
        ↓
Anomaly Detection
        ↓
Model Evaluation
        ↓
Saved Models
        ↓
Streamlit Dashboard
```

## 📊 Machine Learning

### Electricity Load Forecasting

The forecasting pipeline uses engineered demand-related features to predict electricity demand.

Evaluation includes:

* MAE
* RMSE
* MAPE
* R² Score

### Anomaly Detection

Anomaly detection is used to identify unusual electricity demand observations based on relevant demand and engineered features.

The system provides:

* Anomaly identification
* Anomaly scores
* Visualization of unusual observations
* Anomaly data table

## 📈 Streamlit Dashboard

The Streamlit application provides an interactive interface for exploring the GridSense results.

The dashboard includes:

* Dataset overview
* Electricity demand statistics
* Demand visualization
* Anomaly visualization
* Detected anomaly records
* Forecasting feature importance
* Processed dataset preview

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/GridSense-Electricity-Load-Forecasting.git
```

Navigate to the project directory:

```bash
cd GridSense-Electricity-Load-Forecasting
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

### Windows

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## ▶️ Run the Streamlit Application

Run:

```bash
streamlit run app.py
```

The application will open in your browser.

## 📓 Notebook

`GridSense.ipynb` contains the development workflow, including:

* Data loading
* Data inspection
* Preprocessing
* Exploratory data analysis
* Feature engineering
* Model training
* Model evaluation
* Anomaly detection
* Model saving

## 🔮 Future Improvements

* LSTM-based deep learning forecasting
* Comparison between traditional ML and LSTM models
* Improved multi-step forecasting
* Interactive future-demand prediction
* Additional anomaly detection techniques
* Real-time electricity data integration
* Cloud deployment

## 📌 Project Status

**Current Status:** Machine Learning Pipeline + Streamlit Dashboard

The project is actively being developed with additional forecasting and deployment features planned.

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
