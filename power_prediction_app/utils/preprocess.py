import pandas as pd
import numpy as np

def preprocess_data(df):
    """
    Function to preprocess the raw consumption data.
    - Converts date to datetime
    - Extracts features: day_of_week, month, is_weekend
    - Adds lag feature (lag_1)
    - Handles missing values
    """
    # Ensure date is in datetime format
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # Feature extraction
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    
    # Lag feature (previous day's consumption)
    df['lag_1'] = df['consumption'].shift(1)
    
    # Handle missing values (the first row will have a NaN lag_1)
    # We can fill it with the mean or drop it. Dropping is safer for training.
    df = df.dropna()
    
    return df

def get_features_for_date(date_str, last_consumption):
    """
    Utility function to prepare features for a single prediction.
    """
    date = pd.to_datetime(date_str)
    day_of_week = date.dayofweek
    month = date.month
    is_weekend = 1 if day_of_week >= 5 else 0
    
    return pd.DataFrame({
        'day_of_week': [day_of_week],
        'month': [month],
        'is_weekend': [is_weekend],
        'lag_1': [last_consumption]
    })
