# ⚡ Electric Power Consumption Predictor

A professional-grade AI application to predict daily electricity consumption using Machine Learning. This project compares **Linear Regression** and **Random Forest** models to provide the most accurate forecasts for residential energy usage.

## 🌟 Features

- **Realistic Data Simulation**: Includes weekend peaks, seasonal trends, and stochastic noise.
- **Dual Model Comparison**: Automatically trains, evaluates (MAE, R²), and selects the best model.
- **Enhanced Streamlit UI**: 
  - Sidebar with model performance metrics.
  - Interactive Plotly charts for forecasting.
  - Multi-day prediction range support.
- **Desktop Application**: A clean Tkinter interface for offline predictions.
- **Modular Architecture**: Clean separation of data, logic, and presentation layers.

## 📁 Project Structure

```
power_prediction_app/
│
├── data/
│   └── consumption.csv      # Historical dataset
│
├── model/
│   ├── train_model.py       # Model training & comparison script
│   ├── predict.py           # Prediction wrapper class
│   └── saved_model.pkl      # The best-performing saved model
│
├── app/
│   ├── streamlit_app.py     # Modern Web Dashboard
│   └── desktop_app.py       # Desktop (Tkinter) Application
│
├── utils/
│   └── preprocess.py        # Feature engineering & utility functions
│
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation (this file)
```

## 🚀 Getting Started

### 1. Prerequisites

- Python 3.8 or higher.
- A virtual environment is recommended.

### 2. Installation

Clone the repository and install the required dependencies:

```bash
cd power_prediction_app
pip install -r requirements.txt
```

### 3. Model Training

Before running the applications, you must train the model using your historical data:

```bash
python model/train_model.py
```
*This script will generate a detailed evaluation and save the best model to `model/saved_model.pkl`.*

### 4. Running the Applications

#### Launch Streamlit Dashboard (Web UI):
```bash
streamlit run app/streamlit_app.py
```

#### Launch Desktop Application (Tkinter):
```bash
python app/desktop_app.py
```

## 🧪 Model Evaluation

The training script evaluates models based on:
- **MAE (Mean Absolute Error)**: Average absolute difference between predicted and actual values.
- **R² Score (Coefficient of Determination)**: Proportion of variance explained by the model.

## 🤝 Contributing

This project is designed for academic presentation and can be easily extended with more features like real-time sensor data integration or cloud deployment.

---
Developed with ❤️ by Antigravity (Powered by Google DeepMind)
