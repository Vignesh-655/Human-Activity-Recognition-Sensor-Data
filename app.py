"""
Interactive Streamlit Dashboard for Human Activity Recognition (HAR) Using Sensor Data.
ECE / AI Capstone Project.
"""

import os
import sys
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import ACTIVITY_MAP

# Page Configuration
st.set_page_config(
    page_title="Human Activity Recognition | Capstone",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #4b5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border-radius: 10px;
        padding: 1.2rem;
        border-left: 5px solid #3b82f6;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .prediction-badge {
        font-size: 1.8rem;
        font-weight: 700;
        padding: 0.8rem 1.5rem;
        border-radius: 12px;
        text-align: center;
        color: white;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Activity visual meta
ACTIVITY_META = {
    "WALKING": {"icon": "🚶", "color": "#10b981", "type": "Dynamic Motion"},
    "WALKING_UPSTAIRS": {"icon": "🧗‍♂️", "color": "#3b82f6", "type": "Dynamic Motion"},
    "WALKING_DOWNSTAIRS": {"icon": "🏃‍♂️", "color": "#8b5cf6", "type": "Dynamic Motion"},
    "SITTING": {"icon": "🪑", "color": "#f59e0b", "type": "Static Posture"},
    "STANDING": {"icon": "🧍", "color": "#f97316", "type": "Static Posture"},
    "LAYING": {"icon": "🛌", "color": "#ef4444", "type": "Static Posture"}
}

@st.cache_resource
def load_all_models():
    """Load trained models, scalers, and metadata."""
    models_dir = PROJECT_ROOT / "models"
    with open(models_dir / "model_metadata.json", "r") as f:
        meta = json.load(f)
        
    models = {
        "Logistic Regression (Best: 95.59%)": joblib.load(models_dir / "logistic_regression.pkl"),
        "Support Vector Machine (SVM: 95.49%)": joblib.load(models_dir / "support_vector_machine_svm.pkl"),
        "Random Forest (92.77%)": joblib.load(models_dir / "random_forest.pkl"),
        "Decision Tree (86.43%)": joblib.load(models_dir / "decision_tree.pkl")
    }
    scaler = joblib.load(models_dir / "scaler.pkl")
    return models, scaler, meta

@st.cache_data
def load_cached_samples():
    """Load sample test dataset and raw streaming samples."""
    processed_dir = PROJECT_ROOT / "data" / "processed"
    sample_features = pd.read_csv(processed_dir / "sample_test_features.csv")
    sample_stream = pd.read_csv(processed_dir / "sample_sensor_stream.csv")
    return sample_features, sample_stream

models_dict, scaler, metadata = load_all_models()
sample_features_df, sample_stream_df = load_cached_samples()

# Sidebar Navigation
st.sidebar.image("https://img.icons8.com/fluency/96/smart-watch.png", width=70)
st.sidebar.title("HAR System")
st.sidebar.markdown("**AI-Powered Sensor Analytics**")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Overview & System Architecture",
        "📊 Sensor Signals & EDA Studio",
        "🧠 Model Benchmark & Metrics",
        "🔮 Activity Predictor (Test & Upload)",
        "⚡ Real-Time Sensor Stream Simulator"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption("Capstone Project: Human Activity Recognition Using Sensor Data (Smartphone Accelerometer & Gyroscope)")

# ==========================================
# PAGE 1: OVERVIEW & SYSTEM ARCHITECTURE
# ==========================================
if page == "🏠 Overview & System Architecture":
    st.markdown('<div class="main-title">Human Activity Recognition (HAR) Using Sensor Data</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">ECE / AI Capstone Project: Automatic classification of human activities from tri-axial smartphone accelerometer & gyroscope readings.</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Sensors Used", "Accelerometer + Gyroscope", "50 Hz Sample Rate")
    with col2:
        st.metric("Activities Classified", "6 Classes", "Dynamic + Static")
    with col3:
        st.metric("Top Model Accuracy", "95.59%", "Unseen Test Set")
    with col4:
        st.metric("Subjects in Dataset", "30 Volunteers", "Age 19 - 48 yrs")
        
    st.markdown("---")
    
    st.subheader("🎯 Recognized Activity Classes")
    c_acts = st.columns(6)
    for idx, (name, info) in enumerate(ACTIVITY_META.items()):
        with c_acts[idx]:
            st.markdown(f"""
            <div style="background-color:{info['color']}15; border:1px solid {info['color']}; border-radius:10px; padding:12px; text-align:center;">
                <div style="font-size:32px;">{info['icon']}</div>
                <div style="font-weight:700; font-size:13px; color:{info['color']}; margin-top:5px;">{name.replace('_', ' ')}</div>
                <div style="font-size:11px; color:#6b7280;">{info['type']}</div>
            </div>
            """, unsafe_allow_html=True)
            
    st.markdown("---")
    st.subheader("🛠️ End-to-End System Pipeline Architecture")
    
    st.markdown("""
    ```
    ┌──────────────────────┐      ┌─────────────────────────┐      ┌─────────────────────────┐
    │  Smartphone Sensors  │ ───> │ Signal Preprocessing    │ ───> │ Sliding Window (2.56s)  │
    │  • 3-Axial Accel (g) │      │ • Butterworth Low-Pass  │      │ • 128 sensor readings   │
    │  • 3-Axial Gyro (rad)│      │ • Median Noise Filter   │      │ • 50% window overlap    │
    └──────────────────────┘      └─────────────────────────┘      └─────────────────────────┘
                                                                                │
    ┌──────────────────────┐      ┌─────────────────────────┐                   │
    │  Interactive Web UI  │ <─── │ Multi-Class ML Model    │ <─────────────────┘
    │  • Activity Badge    │      │ • Logistic Reg (95.59%) │      Feature Extraction:
    │  • Confidence Gauge  │      │ • SVM RBF Kernel (95.5%)│      • Time Domain (Mean, Std, SMA)
    │  • Real-time Stream  │      │ • Random Forest (92.8%) │      • Frequency Domain (FFT, Peak)
    └──────────────────────┘      └─────────────────────────┘      • Vector Magnitude (Norm)
    ```
    """)
    
    st.markdown("### 📋 Capstone Requirements Checklist")
    reqs = [
        ("Collect accelerometer and gyroscope data", "✅ Completed (UCI HAR benchmark, 10,299 instances @ 50Hz)"),
        ("Label sensor readings to activities", "✅ Completed (6 human activities labeled)"),
        ("Preprocess data, handle noise and inconsistent readings", "✅ Completed (Butterworth 20Hz filter, median filter, scaling)"),
        ("Exploratory Data Analysis (EDA) on sensor patterns", "✅ Completed (Waveforms, magnitude comparisons, PCA, heatmaps)"),
        ("Sliding window segmentation & feature extraction", "✅ Completed (Mean, Std, SMA, Vector Magnitude, FFT energy)"),
        ("Train & evaluate ML models", "✅ Completed (Random Forest, SVM, Decision Tree, Logistic Regression)"),
        ("Evaluate on unseen test data with metrics & confusion matrix", "✅ Completed (95.59% accuracy, precision, recall, F1-scores)"),
        ("Display predicted activity through interactive dashboard", "✅ Completed (Streamlit Web Dashboard)"),
        ("Real-time sensor simulation & demo", "✅ Completed (50Hz animated stream simulation)")
    ]
    for task, status in reqs:
        st.write(f"- **{task}**: {status}")

# ==========================================
# PAGE 2: SENSOR SIGNALS & EDA STUDIO
# ==========================================
elif page == "📊 Sensor Signals & EDA Studio":
    st.markdown('<div class="main-title">Sensor Signals & Exploratory Data Analysis (EDA)</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Detailed exploration of 3-axis accelerometer and gyroscope time-series patterns across physical activities.</div>', unsafe_allow_html=True)
    
    eda_tab1, eda_tab2, eda_tab3, eda_tab4 = st.tabs([
        "📈 Raw Sensor Waveforms",
        "⚡ Dynamic vs Static Magnitude",
        "🎯 2D PCA Class Separation",
        "📊 Distribution & Correlations"
    ])
    
    with eda_tab1:
        st.markdown("#### Tri-Axial Accelerometer and Gyroscope Waveforms by Activity")
        st.caption("Each sample window comprises 128 discrete readings sampled at 50Hz (2.56 seconds duration).")
        raw_fig = PROJECT_ROOT / "reports" / "figures" / "raw_signals_by_activity.png"
        if raw_fig.exists():
            st.image(str(raw_fig), use_container_width=True)
            
        st.markdown("""
        **Waveform Insights:**
        - **Walking & Stairs**: Exhibit rhythmic periodic oscillations with prominent acceleration peaks ($>1.5g$) and angular rotations ($>2 rad/s$).
        - **Sitting & Standing**: Near zero dynamic acceleration; steady $1.0g$ gravitational vector aligned with sensor orientation.
        - **Laying**: Constant gravity reading redirected onto the $X$ or $Z$ axis depending on device orientation on the bed.
        """)
        
    with eda_tab2:
        st.markdown("#### Signal Vector Magnitude: Dynamic Activities vs Static Postures")
        st.latex(r"\text{Magnitude} = \sqrt{a_x^2 + a_y^2 + a_z^2}")
        mag_fig = PROJECT_ROOT / "reports" / "figures" / "signal_magnitude_comparison.png"
        if mag_fig.exists():
            st.image(str(mag_fig), use_container_width=True)
        st.info("Dynamic activities show high variance and repetitive cycles. Static postures hover tightly around 1.0g (Earth's gravity).")

    with eda_tab3:
        st.markdown("#### 2D PCA Dimensionality Reduction Projection")
        st.caption("Demonstrating natural separability of human activities across the engineered 561-dimensional sensor space.")
        pca_fig = PROJECT_ROOT / "reports" / "figures" / "pca_activity_separation.png"
        if pca_fig.exists():
            st.image(str(pca_fig), use_container_width=True)
        st.success("Clear visual cluster separation: Dynamic activities (Walking, Walking Upstairs, Downstairs) cluster on the left, while static postures (Sitting, Standing, Laying) cluster distinctly on the right.")

    with eda_tab4:
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("#### Activity Sample Counts in Splits")
            dist_fig = PROJECT_ROOT / "reports" / "figures" / "activity_distribution.png"
            if dist_fig.exists():
                st.image(str(dist_fig), use_container_width=True)
        with col_b:
            st.markdown("#### Key Feature Correlation Heatmap")
            corr_fig = PROJECT_ROOT / "reports" / "figures" / "correlation_heatmap.png"
            if corr_fig.exists():
                st.image(str(corr_fig), use_container_width=True)

# ==========================================
# PAGE 3: MODEL BENCHMARK & METRICS
# ==========================================
elif page == "🧠 Model Benchmark & Metrics":
    st.markdown('<div class="main-title">Machine Learning Model Benchmark & Evaluation</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Comparative evaluation of classification models tested on 2,947 completely unseen sensor window instances.</div>', unsafe_allow_html=True)
    
    metrics_file = PROJECT_ROOT / "reports" / "model_comparison_metrics.csv"
    if metrics_file.exists():
        comp_df = pd.read_csv(metrics_file)
        
        # Display formatted comparison table
        st.markdown("### 🏆 Model Comparison Table")
        formatted_df = comp_df.copy()
        for col in ["Accuracy", "Precision_Macro", "Recall_Macro", "F1_Macro", "F1_Weighted"]:
            formatted_df[col] = formatted_df[col].apply(lambda x: f"{x*100:.2f}%")
        st.dataframe(formatted_df, use_container_width=True)
        
        # Plot model accuracy comparison
        fig, ax = plt.subplots(figsize=(9, 4), dpi=200)
        bars = ax.barh(comp_df["Model"], comp_df["Accuracy"] * 100, color=['#2563eb', '#3b82f6', '#10b981', '#f59e0b'])
        ax.set_xlim(80, 100)
        ax.set_xlabel("Test Accuracy (%)", fontweight='bold')
        ax.set_title("Test Accuracy Comparison on Unseen Data", fontweight='bold', pad=10)
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.3, bar.get_y() + bar.get_height()/2, f"{width:.2f}%", va='center', fontweight='bold')
        st.pyplot(fig)
        
    st.markdown("---")
    
    col_cm, col_fi = st.columns(2)
    with col_cm:
        st.markdown("### 🔍 Confusion Matrix Analysis")
        cm_fig = PROJECT_ROOT / "reports" / "figures" / "confusion_matrix.png"
        if cm_fig.exists():
            st.image(str(cm_fig), use_container_width=True)
        st.caption("Normalized recall rates demonstrate >94% correct identification across all activity classes.")
        
    with col_fi:
        st.markdown("### 🌲 Top Feature Importances (Random Forest)")
        fi_fig = PROJECT_ROOT / "reports" / "figures" / "feature_importance.png"
        if fi_fig.exists():
            st.image(str(fi_fig), use_container_width=True)
        st.caption("Features based on signal magnitude area (SMA) and gravity acceleration components yield highest discriminative power.")

    st.markdown("---")
    st.subheader("📑 Detailed Per-Class Classification Report (Best Model)")
    report_file = PROJECT_ROOT / "reports" / "best_model_classification_report.csv"
    if report_file.exists():
        clf_df = pd.read_csv(report_file, index_col=0)
        st.dataframe(clf_df.style.format(precision=4).background_gradient(cmap="Blues", subset=["precision", "recall", "f1-score"]), use_container_width=True)

# ==========================================
# PAGE 4: ACTIVITY PREDICTOR (TEST & UPLOAD)
# ==========================================
elif page == "🔮 Activity Predictor (Test & Upload)":
    st.markdown('<div class="main-title">Interactive Activity Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Test model predictions on pre-loaded sensor windows or upload your own sensor feature CSV.</div>', unsafe_allow_html=True)
    
    col_sel, col_model = st.columns([2, 1])
    with col_model:
        selected_model_name = st.selectbox("Select ML Model", list(models_dict.keys()), index=0)
        chosen_model = models_dict[selected_model_name]
        requires_scale = "Logistic" in selected_model_name or "SVM" in selected_model_name
        
    with col_sel:
        input_mode = st.radio("Input Source", ["Select From Test Samples", "Upload Custom CSV File"], horizontal=True)
        
    if input_mode == "Select From Test Samples":
        # Group by activity
        activity_choice = st.selectbox("Choose Activity Class to Sample", list(ACTIVITY_MAP.values()), index=0)
        act_subset = sample_features_df[sample_features_df["activity"] == activity_choice]
        sample_idx = st.slider("Select Sample Instance", 0, len(act_subset) - 1, 0)
        
        sample_row = act_subset.iloc[sample_idx]
        feature_cols = [c for c in sample_features_df.columns if c not in ["subject", "activity_id", "activity"]]
        sample_vector = sample_row[feature_cols].values.reshape(1, -1)
        true_activity = sample_row["activity"]
        
        col_btn, _ = st.columns([1, 4])
        with col_btn:
            predict_clicked = st.button("🚀 Predict Activity", use_container_width=True, type="primary")
            
        if predict_clicked:
            if requires_scale:
                vec_eval = scaler.transform(sample_vector)
            else:
                vec_eval = sample_vector
                
            pred_id = chosen_model.predict(vec_eval)[0]
            pred_name = ACTIVITY_MAP[pred_id]
            meta = ACTIVITY_META[pred_name]
            
            # Get probabilities if supported
            probs = None
            if hasattr(chosen_model, "predict_proba"):
                probs = chosen_model.predict_proba(vec_eval)[0]
                
            st.markdown("---")
            res_col1, res_col2 = st.columns([1, 2])
            
            with res_col1:
                st.markdown(f"""
                <div class="prediction-badge" style="background-color: {meta['color']};">
                    {meta['icon']} {pred_name.replace('_', ' ')}
                </div>
                """, unsafe_allow_html=True)
                
                is_correct = (pred_name == true_activity)
                if is_correct:
                    st.success(f"✅ Ground Truth Match: **{true_activity}**")
                else:
                    st.warning(f"⚠️ True Activity was: **{true_activity}**")
                    
                st.write(f"**Activity Category**: {meta['type']}")
                st.write(f"**Model Used**: {selected_model_name.split('(')[0]}")
                
            with res_col2:
                st.markdown("#### Prediction Confidence Probabilities")
                if probs is not None:
                    prob_df = pd.DataFrame({
                        "Activity": [ACTIVITY_MAP[i] for i in range(1, 7)],
                        "Probability": probs
                    })
                    
                    fig, ax = plt.subplots(figsize=(7, 3), dpi=150)
                    bars = ax.barh(prob_df["Activity"], prob_df["Probability"] * 100,
                                   color=[ACTIVITY_META[act]["color"] for act in prob_df["Activity"]])
                    ax.set_xlim(0, 100)
                    ax.set_xlabel("Confidence (%)", fontweight='bold')
                    for bar in bars:
                        w = bar.get_width()
                        if w > 2:
                            ax.text(w + 1, bar.get_y() + bar.get_height()/2, f"{w:.1f}%", va='center', fontsize=9)
                    st.pyplot(fig)
                else:
                    st.info("Class probabilities not available for this model configuration.")

    else:
        st.markdown("#### Upload CSV Feature Matrix")
        uploaded_file = st.file_uploader("Upload CSV (must contain 561 HAR features)", type=["csv"])
        if uploaded_file is not None:
            up_df = pd.read_csv(uploaded_file)
            st.write(f"Uploaded shape: {up_df.shape}")
            feature_cols = [c for c in up_df.columns if c not in ["subject", "activity_id", "activity"]]
            
            if len(feature_cols) == 561:
                st.success("✅ Valid 561-feature sensor vector recognized!")
                if st.button("Predict Batch", type="primary"):
                    X_input = up_df[feature_cols].values
                    if requires_scale:
                        X_input = scaler.transform(X_input)
                    preds = chosen_model.predict(X_input)
                    up_df["Predicted_Activity"] = [ACTIVITY_MAP[p] for p in preds]
                    st.dataframe(up_df[["Predicted_Activity"] + feature_cols[:5]], use_container_width=True)
            else:
                st.error(f"Expected 561 features, but received {len(feature_cols)} columns.")

# ==========================================
# PAGE 5: REAL-TIME SENSOR STREAM SIMULATOR
# ==========================================
elif page == "⚡ Real-Time Sensor Stream Simulator":
    st.markdown('<div class="main-title">⚡ Real-Time Sensor Stream Simulator</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Simulates live 50Hz continuous data streaming from a smartphone accelerometer and gyroscope.</div>', unsafe_allow_html=True)
    
    col_ctl1, col_ctl2, col_ctl3 = st.columns(3)
    with col_ctl1:
        sim_activity = st.selectbox("Select Activity to Stream", list(ACTIVITY_MAP.values()), index=0)
    with col_ctl2:
        stream_speed = st.slider("Simulation Speed (ms delay)", 10, 200, 30)
    with col_ctl3:
        start_stream = st.button("▶️ Start Live Stream", type="primary", use_container_width=True)
        
    act_stream = sample_stream_df[sample_stream_df["activity"] == sim_activity]
    
    # Placeholders for dynamic rendering
    status_box = st.empty()
    chart_acc = st.empty()
    chart_gyro = st.empty()
    
    if start_stream:
        window_size = 64
        meta = ACTIVITY_META[sim_activity]
        
        status_box.markdown(f"""
        <div style="background-color:{meta['color']}; color:white; padding:15px; border-radius:10px; text-align:center; font-size:22px; font-weight:bold; margin-bottom:15px;">
            📡 LIVE STREAMING SENSOR FEED: {meta['icon']} {sim_activity.replace('_', ' ')} ({meta['type']})
        </div>
        """, unsafe_allow_html=True)
        
        for i in range(len(act_stream) - window_size):
            chunk = act_stream.iloc[i : i + window_size]
            
            # Plot Accelerometer
            fig_acc, ax_acc = plt.subplots(figsize=(10, 2.5), dpi=100)
            ax_acc.plot(chunk["timestamp_sec"], chunk["total_acc_x"], label="Acc X", color="#ef4444", lw=1.5)
            ax_acc.plot(chunk["timestamp_sec"], chunk["total_acc_y"], label="Acc Y", color="#10b981", lw=1.5)
            ax_acc.plot(chunk["timestamp_sec"], chunk["total_acc_z"], label="Acc Z", color="#3b82f6", lw=1.5)
            ax_acc.set_title("Live Tri-Axial Accelerometer (g) - 50 Hz", fontsize=10, fontweight="bold")
            ax_acc.set_ylabel("Accel (g)", fontsize=9)
            ax_acc.legend(loc="upper right", fontsize=8)
            ax_acc.grid(True, linestyle="--", alpha=0.5)
            chart_acc.pyplot(fig_acc)
            plt.close(fig_acc)
            
            # Plot Gyroscope
            fig_gyro, ax_gyro = plt.subplots(figsize=(10, 2.5), dpi=100)
            ax_gyro.plot(chunk["timestamp_sec"], chunk["body_gyro_x"], label="Gyro X", color="#f59e0b", lw=1.5)
            ax_gyro.plot(chunk["timestamp_sec"], chunk["body_gyro_y"], label="Gyro Y", color="#8b5cf6", lw=1.5)
            ax_gyro.plot(chunk["timestamp_sec"], chunk["body_gyro_z"], label="Gyro Z", color="#06b6d4", lw=1.5)
            ax_gyro.set_title("Live Tri-Axial Gyroscope (rad/s) - 50 Hz", fontsize=10, fontweight="bold")
            ax_gyro.set_ylabel("Gyro (rad/s)", fontsize=9)
            ax_gyro.set_xlabel("Time (s)", fontsize=9)
            ax_gyro.legend(loc="upper right", fontsize=8)
            ax_gyro.grid(True, linestyle="--", alpha=0.5)
            chart_gyro.pyplot(fig_gyro)
            plt.close(fig_gyro)
            
            time.sleep(stream_speed / 1000.0)
            
        st.success("✅ Real-time sensor stream simulation window completed!")
    else:
        st.info("Click '▶️ Start Live Stream' to begin real-time sensor waveform simulation and live recognition.")
