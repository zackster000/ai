import streamlit as st
import pandas as pd
import plotly.express as px
import datetime
import os
import sys

# Add the project root to the path so we can import models and utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Correct the sys.path to allow imports from other project folders
base_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(base_path, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Now these imports will work regardless of where streamlit is run from
try:
    from model.predict import Predictor
except ImportError:
    # If the above fails, try adding the current parent folder specifically
    sys.path.append(os.path.join(project_root, 'power_prediction_app'))
    from model.predict import Predictor

# Page Config
st.set_page_config(page_title="Electric Consumption Predictor", layout="wide")

# Sidebar
st.sidebar.title("⚡ Power Prediction App")
st.sidebar.write("This application uses Machine Learning to predict daily electricity consumption based on historical patterns and date-based features.")

# Try to load the model
try:
    # We pass None or let it default to the robust internal logic in Predictor
    predictor = Predictor()
    st.sidebar.success(f"Model Loaded: {predictor.model_name}")
    st.sidebar.write(f"**Model Metrics:**")
    st.sidebar.write(f"- MAE: {predictor.metrics['mae']:.4f}")
    st.sidebar.write(f"- R² Score: {predictor.metrics['r2']:.4f}")
except Exception as e:
    st.sidebar.warning("Model not found! Please run `train_model.py` to generate the saved_model.pkl.")
    st.sidebar.error(str(e))
    predictor = None

# Main Content
st.title("Electric Consumption Predictor")
st.write("Enter the date and previous day's consumption to get a prediction for future usage.")

# Tabs for Single and Multi-day Predictions
tab1, tab2 = st.tabs(["Single Prediction", "Multi-day Forecast"])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Input Parameters")
        target_date = st.date_input("Prediction Date", datetime.date.today() + datetime.timedelta(days=1))
        last_consumption = st.number_input("Last Consumption (kWh)", min_value=1.0, max_value=100.0, value=25.0, help="The electricity consumption from the previous day.")
        
        predict_button = st.button("🚀 Predict Consumption", key="single_predict")
        
    with col2:
        if predict_button:
            if predictor is None:
                st.error("Cannot predict without a trained model.")
            else:
                with st.spinner("Analyzing patterns and calculating..."):
                    prediction = predictor.predict_single(target_date.strftime('%Y-%m-%d'), last_consumption)
                    
                st.subheader("Prediction Result")
                st.markdown(f"""
                <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 5px solid #ff4b4b;">
                    <h1 style="color: #1f1f1f; margin: 0;">Predicted: {prediction} kWh</h1>
                    <p style="color: #666; margin-top: 5px;">For date: {target_date.strftime('%B %d, %Y')}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Feedback/Info
                variation = ((prediction - last_consumption) / last_consumption) * 100
                direction = "INCREASE" if variation > 0 else "DECREASE"
                st.info(f"This is a {abs(variation):.1f}% {direction} compared to the previous day.")

with tab2:
    st.subheader("Forecast Range")
    col3, col4 = st.columns(2)
    
    with col3:
        start_date = st.date_input("Start Date", datetime.date.today())
        end_date = st.date_input("End Date", datetime.date.today() + datetime.timedelta(days=7))
        initial_consumption = st.number_input("Initial Last Consumption (kWh)", min_value=1.0, max_value=100.0, value=25.0, key="multi_initial")
        
        multi_predict_button = st.button("📈 Generate Forecast", key="multi_predict")
        
    if multi_predict_button:
        if predictor is None:
            st.error("Cannot predict without a trained model.")
        elif start_date > end_date:
            st.error("End date must be after start date.")
        else:
            with st.spinner("Generating time-series forecast..."):
                forecast_df = predictor.predict_range(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), initial_consumption)
                
            st.success(f"Forecast generated for {len(forecast_df)} days.")
            
            # Visualization
            fig = px.line(forecast_df, x='date', y='predicted_consumption', 
                        title=f"Predicted Consumption Forecast ({predictor.model_name})",
                        labels={'predicted_consumption': 'Consumption (kWh)', 'date': 'Date'},
                        markers=True)
            fig.update_layout(template="plotly_white")
            st.plotly_chart(fig, use_container_with_width=True)
            
            # Data Table
            st.subheader("Forecast Data")
            st.dataframe(forecast_df, use_container_with_width=True)

# Footer
st.markdown("---")
st.markdown("Developed with ❤️ using Python and Streamlit.")
