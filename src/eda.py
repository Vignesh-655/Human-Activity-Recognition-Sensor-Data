"""
Exploratory Data Analysis (EDA) Module for Human Activity Recognition.
Generates comprehensive visual plots for signal patterns, activity distributions, and feature relationships.
"""

import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.data_loader import load_har_data, load_raw_inertial_signals, ACTIVITY_MAP

FIG_DIR = Path(__file__).resolve().parent.parent / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Set clean aesthetic style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.autolayout'] = True

ACTIVITY_COLORS = {
    "WALKING": "#2ecc71",            # Green
    "WALKING_UPSTAIRS": "#3498db",   # Blue
    "WALKING_DOWNSTAIRS": "#9b59b6", # Purple
    "SITTING": "#f1c40f",            # Yellow
    "STANDING": "#e67e22",           # Orange
    "LAYING": "#e74c3c"              # Red
}

def plot_activity_distribution(y_train, y_test):
    """Plot distribution of activities across train and test sets."""
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    
    train_counts = y_train.map(ACTIVITY_MAP).value_counts().reset_index()
    train_counts.columns = ["Activity", "Train"]
    
    test_counts = y_test.map(ACTIVITY_MAP).value_counts().reset_index()
    test_counts.columns = ["Activity", "Test"]
    
    merged = pd.merge(train_counts, test_counts, on="Activity")
    df_melt = pd.melt(merged, id_vars=["Activity"], var_name="Split", value_name="Count")
    
    sns.barplot(data=df_melt, x="Activity", y="Count", hue="Split", palette=["#1f77b4", "#ff7f0e"], ax=ax)
    ax.set_title("Distribution of Human Activities in Train vs Test Splits", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Activity Type", fontsize=12, fontweight="bold")
    ax.set_ylabel("Number of Window Samples", fontsize=12, fontweight="bold")
    plt.xticks(rotation=25, ha='right')
    
    # Add count labels on top of bars
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f"{int(height)}",
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=9, xytext=(0, 3),
                        textcoords='offset points')
            
    fig_path = FIG_DIR / "activity_distribution.png"
    plt.savefig(fig_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved: {fig_path}")

def plot_raw_sensor_signals(signals, y_train):
    """
    Plots representative raw tri-axial accelerometer & gyroscope waveforms for all 6 activities.
    """
    y_arr = y_train.values
    fig, axes = plt.subplots(6, 2, figsize=(15, 14), dpi=300, sharex=True)
    time_steps = np.arange(128) * (1.0 / 50.0) # 2.56 seconds window at 50 Hz
    
    for row_idx, (act_id, act_name) in enumerate(ACTIVITY_MAP.items()):
        # Find first sample index matching activity
        sample_idx = np.where(y_arr == act_id)[0][0]
        
        # Accelerometer
        ax_acc = axes[row_idx, 0]
        ax_acc.plot(time_steps, signals["total_acc_x"][sample_idx], label="Acc X", color="#e74c3c", lw=1.5)
        ax_acc.plot(time_steps, signals["total_acc_y"][sample_idx], label="Acc Y", color="#2ecc71", lw=1.5)
        ax_acc.plot(time_steps, signals["total_acc_z"][sample_idx], label="Acc Z", color="#3498db", lw=1.5)
        ax_acc.set_ylabel(f"{act_name}\nAcc (g)", fontsize=10, fontweight="bold")
        ax_acc.grid(True, linestyle="--", alpha=0.6)
        if row_idx == 0:
            ax_acc.set_title("Total Accelerometer Signals (3-Axis)", fontsize=12, fontweight="bold")
            ax_acc.legend(loc="upper right", frameon=True, fontsize=8)
            
        # Gyroscope
        ax_gyro = axes[row_idx, 1]
        ax_gyro.plot(time_steps, signals["body_gyro_x"][sample_idx], label="Gyro X", color="#e74c3c", lw=1.5)
        ax_gyro.plot(time_steps, signals["body_gyro_y"][sample_idx], label="Gyro Y", color="#2ecc71", lw=1.5)
        ax_gyro.plot(time_steps, signals["body_gyro_z"][sample_idx], label="Gyro Z", color="#3498db", lw=1.5)
        ax_gyro.set_ylabel("Gyro (rad/s)", fontsize=10, fontweight="bold")
        ax_gyro.grid(True, linestyle="--", alpha=0.6)
        if row_idx == 0:
            ax_gyro.set_title("Body Gyroscope Signals (3-Axis)", fontsize=12, fontweight="bold")
            ax_gyro.legend(loc="upper right", frameon=True, fontsize=8)
            
    axes[5, 0].set_xlabel("Time (seconds)", fontsize=11, fontweight="bold")
    axes[5, 1].set_xlabel("Time (seconds)", fontsize=11, fontweight="bold")
    
    fig.suptitle("Raw Accelerometer & Gyroscope Waveforms (2.56s Window @ 50Hz)", fontsize=15, fontweight="bold", y=1.005)
    fig_path = FIG_DIR / "raw_signals_by_activity.png"
    plt.savefig(fig_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved: {fig_path}")

def plot_magnitude_comparison(signals, y_train):
    """
    Compares the Total Acceleration Vector Magnitude: sqrt(ax^2 + ay^2 + az^2)
    between Dynamic Activities (Walking, Stairs) and Static Postures (Sitting, Standing, Laying).
    """
    y_arr = y_train.values
    time_steps = np.arange(128) * (1.0 / 50.0)
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), dpi=300, sharex=True)
    
    # Dynamic activities: Walking, Walking Upstairs, Walking Downstairs
    ax_dyn = axes[0]
    for act_id in [1, 2, 3]:
        act_name = ACTIVITY_MAP[act_id]
        idx = np.where(y_arr == act_id)[0][1]
        ax = signals["total_acc_x"][idx]
        ay = signals["total_acc_y"][idx]
        az = signals["total_acc_z"][idx]
        mag = np.sqrt(ax**2 + ay**2 + az**2)
        ax_dyn.plot(time_steps, mag, label=act_name, color=ACTIVITY_COLORS[act_name], lw=2)
    ax_dyn.set_title("Dynamic Activities - Acceleration Vector Magnitude (High Variance)", fontsize=12, fontweight="bold")
    ax_dyn.set_ylabel("Magnitude (g)", fontsize=11, fontweight="bold")
    ax_dyn.legend(loc="upper right", frameon=True)
    ax_dyn.grid(True, linestyle="--", alpha=0.7)
    
    # Static activities: Sitting, Standing, Laying
    ax_stat = axes[1]
    for act_id in [4, 5, 6]:
        act_name = ACTIVITY_MAP[act_id]
        idx = np.where(y_arr == act_id)[0][1]
        ax = signals["total_acc_x"][idx]
        ay = signals["total_acc_y"][idx]
        az = signals["total_acc_z"][idx]
        mag = np.sqrt(ax**2 + ay**2 + az**2)
        ax_stat.plot(time_steps, mag, label=act_name, color=ACTIVITY_COLORS[act_name], lw=2)
    ax_stat.set_title("Static Postures - Acceleration Vector Magnitude (~1.0g Gravity Constant)", fontsize=12, fontweight="bold")
    ax_stat.set_xlabel("Time (seconds)", fontsize=11, fontweight="bold")
    ax_stat.set_ylabel("Magnitude (g)", fontsize=11, fontweight="bold")
    ax_stat.legend(loc="upper right", frameon=True)
    ax_stat.grid(True, linestyle="--", alpha=0.7)
    
    fig_path = FIG_DIR / "signal_magnitude_comparison.png"
    plt.savefig(fig_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved: {fig_path}")

def plot_pca_clusters(X_train, y_train):
    """2D PCA projection demonstrating class separability."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    pca_df = pd.DataFrame({
        "PC1": X_pca[:, 0],
        "PC2": X_pca[:, 1],
        "Activity": y_train.map(ACTIVITY_MAP)
    })
    
    plt.figure(figsize=(10, 7), dpi=300)
    palette = [ACTIVITY_COLORS[act] for act in ACTIVITY_MAP.values()]
    sns.scatterplot(
        data=pca_df, x="PC1", y="PC2", hue="Activity",
        palette=palette, alpha=0.6, s=25
    )
    plt.title(f"2D PCA Projection of HAR Feature Space\n(Explained Variance: PC1={pca.explained_variance_ratio_[0]:.1%}, PC2={pca.explained_variance_ratio_[1]:.1%})",
              fontsize=13, fontweight="bold", pad=12)
    plt.xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]:.1%})", fontsize=11, fontweight="bold")
    plt.ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]:.1%})", fontsize=11, fontweight="bold")
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)
    
    fig_path = FIG_DIR / "pca_activity_separation.png"
    plt.savefig(fig_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved: {fig_path}")

def plot_correlation_heatmap(X_train):
    """Correlation heatmap of top features across acceleration and gyroscope."""
    # Select key representative features from the 561 features
    selected_cols = [c for c in X_train.columns if any(k in c for k in [
        "tBodyAcc-mean()", "tBodyAcc-std()", "tBodyAccMag-mean()", "tBodyAccMag-std()",
        "tBodyGyro-mean()", "tBodyGyro-std()", "tBodyGyroMag-mean()", "fBodyAcc-mean()",
        "fBodyAcc-max()"
    ])][:12]
    
    corr = X_train[selected_cols].corr()
    
    plt.figure(figsize=(10, 8), dpi=300)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title("Correlation Heatmap of Key Sensor Signal Features", fontsize=13, fontweight="bold", pad=12)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(fontsize=9)
    
    fig_path = FIG_DIR / "correlation_heatmap.png"
    plt.savefig(fig_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved: {fig_path}")

def run_all_eda():
    print("Running complete EDA suite...")
    X_train, y_train, sub_train, X_test, y_test, sub_test, feature_names = load_har_data()
    raw_train = load_raw_inertial_signals("train")
    
    plot_activity_distribution(y_train, y_test)
    plot_raw_sensor_signals(raw_train, y_train)
    plot_magnitude_comparison(raw_train, y_train)
    plot_pca_clusters(X_train, y_train)
    plot_correlation_heatmap(X_train)
    print("All EDA graphs generated successfully!")

if __name__ == "__main__":
    run_all_eda()
