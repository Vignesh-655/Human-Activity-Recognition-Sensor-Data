"""
Data Preprocessing Module for Human Activity Recognition.
Handles signal filtering, noise reduction, missing value checks, and normalization.
"""

import numpy as np
import pandas as pd
from scipy import signal
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib
from pathlib import Path

def butter_lowpass_filter(data, cutoff=20.0, fs=50.0, order=3):
    """
    Applies a digital low-pass Butterworth filter to sensor signals.
    - cutoff: Cutoff frequency in Hz (20Hz standard for human motion)
    - fs: Sampling rate in Hz (50Hz on smartphone)
    - order: Filter order (3rd order filter)
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    filtered = signal.filtfilt(b, a, data, axis=-1)
    return filtered

def median_filter(data, kernel_size=3):
    """
    Applies median filtering to suppress sudden spikes or sensor glitches.
    """
    return signal.medfilt(data, kernel_size=kernel_size)

def clean_and_check_data(df):
    """
    Checks for missing values, infinite values, and data integrity.
    Returns: cleaned DataFrame and diagnostic dict.
    """
    null_counts = df.isnull().sum().sum()
    inf_counts = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
    
    diagnostic = {
        "total_rows": len(df),
        "total_columns": df.shape[1],
        "null_values": int(null_counts),
        "infinite_values": int(inf_counts)
    }
    
    # If any nulls exist, forward fill then backward fill
    if null_counts > 0:
        df = df.ffill().bfill()
        
    return df, diagnostic

def fit_and_save_scaler(X_train, scaler_type="standard", save_path="models/scaler.pkl"):
    """
    Fits feature scaler on training data and serializes to disk.
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    if scaler_type == "standard":
        scaler = StandardScaler()
    elif scaler_type == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown scaler type: {scaler_type}")
        
    scaler.fit(X_train)
    joblib.dump(scaler, save_path)
    print(f"Fitted {scaler_type} scaler and saved to {save_path}")
    return scaler

def load_scaler(save_path="models/scaler.pkl"):
    """Load previously saved scaler."""
    return joblib.load(save_path)
