"""
Feature Extraction and Sliding Window Segmentation Module for HAR.
Converts raw tri-axial accelerometer & gyroscope time-series streams into feature vectors.
"""

import numpy as np
import pandas as pd
from scipy import stats, fft

def calculate_sma(x, y, z):
    """Signal Magnitude Area (SMA)."""
    return np.mean(np.abs(x) + np.abs(y) + np.abs(z))

def calculate_vector_magnitude(x, y, z):
    """Euclidean norm / Vector Magnitude: sqrt(x^2 + y^2 + z^2)."""
    return np.sqrt(x**2 + y**2 + z**2)

def calculate_signal_entropy(signal, num_bins=10):
    """Estimate Shannon entropy of signal distribution."""
    counts, _ = np.histogram(signal, bins=num_bins, density=True)
    counts = counts[counts > 0]
    if len(counts) == 0:
        return 0.0
    probs = counts / np.sum(counts)
    return -np.sum(probs * np.log2(probs + 1e-12))

def extract_axis_features(signal_1d, prefix=""):
    """
    Extracts time and frequency domain features for a single 1D sensor axis.
    """
    features = {}
    
    # Time domain
    features[f"{prefix}_mean"] = float(np.mean(signal_1d))
    features[f"{prefix}_std"] = float(np.std(signal_1d))
    features[f"{prefix}_min"] = float(np.min(signal_1d))
    features[f"{prefix}_max"] = float(np.max(signal_1d))
    features[f"{prefix}_median"] = float(np.median(signal_1d))
    features[f"{prefix}_mad"] = float(np.median(np.abs(signal_1d - np.median(signal_1d))))
    features[f"{prefix}_iqr"] = float(stats.iqr(signal_1d))
    features[f"{prefix}_energy"] = float(np.mean(signal_1d ** 2))
    features[f"{prefix}_entropy"] = float(calculate_signal_entropy(signal_1d))
    features[f"{prefix}_skew"] = float(stats.skew(signal_1d))
    features[f"{prefix}_kurtosis"] = float(stats.kurtosis(signal_1d))
    
    # Frequency domain (FFT)
    n = len(signal_1d)
    fft_vals = np.abs(fft.fft(signal_1d))[:n // 2]
    features[f"{prefix}_fft_mean"] = float(np.mean(fft_vals))
    features[f"{prefix}_fft_std"] = float(np.std(fft_vals))
    features[f"{prefix}_fft_energy"] = float(np.mean(fft_vals ** 2))
    features[f"{prefix}_fft_peak_freq"] = float(np.argmax(fft_vals))
    
    return features

def extract_window_features(window_df):
    """
    Extracts comprehensive feature vector from a time-series window (e.g. 128 samples).
    Expected columns: body_acc_x, body_acc_y, body_acc_z, body_gyro_x, body_gyro_y, body_gyro_z
    """
    feats = {}
    
    # Accelerometer axes
    ax = window_df["body_acc_x"].values
    ay = window_df["body_acc_y"].values
    az = window_df["body_acc_z"].values
    
    # Gyroscope axes
    gx = window_df["body_gyro_x"].values
    gy = window_df["body_gyro_y"].values
    gz = window_df["body_gyro_z"].values
    
    for name, arr in [
        ("tBodyAcc_X", ax), ("tBodyAcc_Y", ay), ("tBodyAcc_Z", az),
        ("tBodyGyro_X", gx), ("tBodyGyro_Y", gy), ("tBodyGyro_Z", gz)
    ]:
        feats.update(extract_axis_features(arr, prefix=name))
        
    # Magnitudes
    acc_mag = calculate_vector_magnitude(ax, ay, az)
    gyro_mag = calculate_vector_magnitude(gx, gy, gz)
    
    feats.update(extract_axis_features(acc_mag, prefix="tBodyAccMag"))
    feats.update(extract_axis_features(gyro_mag, prefix="tBodyGyroMag"))
    
    # Signal Magnitude Areas (SMA)
    feats["tBodyAcc_SMA"] = float(calculate_sma(ax, ay, az))
    feats["tBodyGyro_SMA"] = float(calculate_sma(gx, gy, gz))
    
    # Cross-axis correlations
    feats["corr_acc_xy"] = float(np.corrcoef(ax, ay)[0, 1]) if np.std(ax) > 1e-6 and np.std(ay) > 1e-6 else 0.0
    feats["corr_acc_xz"] = float(np.corrcoef(ax, az)[0, 1]) if np.std(ax) > 1e-6 and np.std(az) > 1e-6 else 0.0
    feats["corr_acc_yz"] = float(np.corrcoef(ay, az)[0, 1]) if np.std(ay) > 1e-6 and np.std(az) > 1e-6 else 0.0
    feats["corr_gyro_xy"] = float(np.corrcoef(gx, gy)[0, 1]) if np.std(gx) > 1e-6 and np.std(gy) > 1e-6 else 0.0
    feats["corr_gyro_xz"] = float(np.corrcoef(gx, gz)[0, 1]) if np.std(gx) > 1e-6 and np.std(gz) > 1e-6 else 0.0
    feats["corr_gyro_yz"] = float(np.corrcoef(gy, gz)[0, 1]) if np.std(gy) > 1e-6 and np.std(gz) > 1e-6 else 0.0
    
    return feats

def sliding_window_segmentation(df, window_size=128, step_size=64):
    """
    Segments continuous sensor stream into overlapping windows and extracts features for each.
    Returns: DataFrame of extracted feature rows.
    """
    total_len = len(df)
    features_list = []
    
    for start in range(0, total_len - window_size + 1, step_size):
        end = start + window_size
        window_df = df.iloc[start:end]
        feats = extract_window_features(window_df)
        
        # If activity column present, take mode as label
        if "activity" in window_df.columns:
            feats["activity"] = window_df["activity"].mode()[0]
        if "activity_id" in window_df.columns:
            feats["activity_id"] = int(window_df["activity_id"].mode()[0])
            
        features_list.append(feats)
        
    return pd.DataFrame(features_list)
