"""
Data Loader Module for Human Activity Recognition (HAR) using Smartphone Sensors.
Loads raw signals and engineered features from UCI HAR Dataset.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw" / "UCI HAR Dataset"
PROCESSED_DIR = DATA_DIR / "processed"

ACTIVITY_MAP = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING"
}

def load_feature_names():
    """Load and clean feature names from features.txt."""
    features_file = RAW_DIR / "features.txt"
    if not features_file.exists():
        raise FileNotFoundError(f"features.txt not found at {features_file}")
    
    feature_names = []
    seen = {}
    with open(features_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                name = parts[1]
                # Handle duplicate feature names by appending index
                if name in seen:
                    seen[name] += 1
                    name = f"{name}_{seen[name]}"
                else:
                    seen[name] = 1
                feature_names.append(name)
    return feature_names

def load_har_data():
    """
    Load train and test data splits with features, subjects, and activity labels.
    Returns:
        X_train, y_train, sub_train, X_test, y_test, sub_test, feature_names
    """
    feature_names = load_feature_names()
    
    print("Loading train features...")
    X_train = pd.read_csv(RAW_DIR / "train" / "X_train.txt", sep=r"\s+", header=None, names=feature_names)
    y_train = pd.read_csv(RAW_DIR / "train" / "y_train.txt", sep=r"\s+", header=None, names=["activity_id"])["activity_id"]
    sub_train = pd.read_csv(RAW_DIR / "train" / "subject_train.txt", sep=r"\s+", header=None, names=["subject_id"])["subject_id"]
    
    print("Loading test features...")
    X_test = pd.read_csv(RAW_DIR / "test" / "X_test.txt", sep=r"\s+", header=None, names=feature_names)
    y_test = pd.read_csv(RAW_DIR / "test" / "y_test.txt", sep=r"\s+", header=None, names=["activity_id"])["activity_id"]
    sub_test = pd.read_csv(RAW_DIR / "test" / "subject_test.txt", sep=r"\s+", header=None, names=["subject_id"])["subject_id"]
    
    return X_train, y_train, sub_train, X_test, y_test, sub_test, feature_names

def load_raw_inertial_signals(split="train"):
    """
    Loads raw tri-axial accelerometer and gyroscope time series signals (128 readings per window).
    Signals:
      - body_acc_x, body_acc_y, body_acc_z
      - body_gyro_x, body_gyro_y, body_gyro_z
      - total_acc_x, total_acc_y, total_acc_z
    Returns:
        signals: dict of numpy arrays, each shape (N, 128)
    """
    inertial_dir = RAW_DIR / split / "Inertial Signals"
    signal_names = [
        "body_acc_x", "body_acc_y", "body_acc_z",
        "body_gyro_x", "body_gyro_y", "body_gyro_z",
        "total_acc_x", "total_acc_y", "total_acc_z"
    ]
    signals = {}
    for sig in signal_names:
        file_path = inertial_dir / f"{sig}_{split}.txt"
        if file_path.exists():
            arr = np.loadtxt(file_path)
            signals[sig] = arr
    return signals

def export_processed_datasets():
    """
    Creates processed datasets and sample stream data for demo and dashboard.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    X_train, y_train, sub_train, X_test, y_test, sub_test, feature_names = load_har_data()
    
    # Save a merged sample dataframe for quick EDA and inspection
    train_df = X_train.copy()
    train_df["subject"] = sub_train
    train_df["activity_id"] = y_train
    train_df["activity"] = y_train.map(ACTIVITY_MAP)
    
    test_df = X_test.copy()
    test_df["subject"] = sub_test
    test_df["activity_id"] = y_test
    test_df["activity"] = y_test.map(ACTIVITY_MAP)
    
    print(f"Train dataset shape: {train_df.shape}")
    print(f"Test dataset shape: {test_df.shape}")
    
    # Save sample window signals for live dashboard simulation (10 windows per activity)
    raw_test = load_raw_inertial_signals("test")
    y_test_arr = y_test.values
    
    samples_list = []
    for act_id, act_name in ACTIVITY_MAP.items():
        indices = np.where(y_test_arr == act_id)[0]
        # pick first 3 windows of this activity
        for win_idx in indices[:3]:
            for t in range(128):
                samples_list.append({
                    "sample_window": int(win_idx),
                    "time_step": t,
                    "timestamp_sec": round(t * (1.0 / 50.0), 4), # 50Hz sampling
                    "activity_id": act_id,
                    "activity": act_name,
                    "body_acc_x": raw_test["body_acc_x"][win_idx, t],
                    "body_acc_y": raw_test["body_acc_y"][win_idx, t],
                    "body_acc_z": raw_test["body_acc_z"][win_idx, t],
                    "body_gyro_x": raw_test["body_gyro_x"][win_idx, t],
                    "body_gyro_y": raw_test["body_gyro_y"][win_idx, t],
                    "body_gyro_z": raw_test["body_gyro_z"][win_idx, t],
                    "total_acc_x": raw_test["total_acc_x"][win_idx, t],
                    "total_acc_y": raw_test["total_acc_y"][win_idx, t],
                    "total_acc_z": raw_test["total_acc_z"][win_idx, t],
                })
    
    sample_stream_df = pd.DataFrame(samples_list)
    sample_stream_file = PROCESSED_DIR / "sample_sensor_stream.csv"
    sample_stream_df.to_csv(sample_stream_file, index=False)
    print(f"Saved sample stream dataset to {sample_stream_file}")

    # Also save a compact test subset for instant web test prediction upload
    test_subset = test_df.sample(n=200, random_state=42)
    test_subset.to_csv(PROCESSED_DIR / "sample_test_features.csv", index=False)
    print(f"Saved sample test features to {PROCESSED_DIR / 'sample_test_features.csv'}")

if __name__ == "__main__":
    export_processed_datasets()
