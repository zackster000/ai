import joblib
import pandas as pd
import datetime
import os
import sys

# Add the project root to the path so we can import utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.preprocess import get_features_for_date

class Predictor:
    def __init__(self, model_path=None):
        if model_path is None:
            # Default to model/saved_model.pkl relative to this script
            base_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(base_dir, 'saved_model.pkl')
            
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} not found. Please run train_model.py first.")
        
        # Load the model and its metadata
        self.model_data = joblib.load(model_path)
        self.model = self.model_data['model']
        self.model_name = self.model_data['model_name']
        self.metrics = self.model_data['metrics']
        self.feature_names = self.model_data['feature_names']

    def predict_single(self, date_str, last_consumption):
        """
        Predict for a single date given the last consumption.
        """
        features = get_features_for_date(date_str, last_consumption)
        prediction = self.model.predict(features)
        return round(float(prediction[0]), 2)

    def predict_range(self, start_date, end_date, initial_last_consumption):
        """
        Predict for a range of dates.
        The last consumption for each day is the prediction from the previous day.
        """
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        dates = pd.date_range(start, end)
        
        predictions = []
        current_last = initial_last_consumption
        
        for date in dates:
            pred = self.predict_single(date.strftime('%Y-%m-%d'), current_last)
            predictions.append({'date': date.strftime('%Y-%m-%d'), 'predicted_consumption': pred})
            current_last = pred # Use prediction as the lag for the next day
            
        return pd.DataFrame(predictions)

if __name__ == "__main__":
    # Test predictor
    try:
        predictor = Predictor()
        print(f"Loaded {predictor.model_name}")
        print(f"Prediction for tomorrow: {predictor.predict_single('2024-04-05', 25.0)}")
    except Exception as e:
        print(f"Prediction test failed: {e}")
