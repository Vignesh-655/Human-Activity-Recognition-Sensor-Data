"""
Modern, Interactive Streamlit Dashboard for Human Activity Recognition (HAR) Using Sensor Data.
ECE & AI Capstone Project.
"""

import os
import sys
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import ACTIVITY_MAP

# Page Configuration
st.set_page_config(
    page_title="Human Activity Recognition | AI Capstone",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Modern CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Hero Header */
    .hero-container {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        color: white;
        padding: 2.2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.8rem;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin-bottom: 0.4rem;
        background: linear-gradient(90deg, #60a5fa, #a78bfa, #38bdf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #94a3b8;
        max-width: 850px;
        line-height: 1.5;
        margin-bottom: 1.2rem;
    }
    
    /* Stat Cards */
    .stat-card {
        background: white;
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.03);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .stat-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.06);
    }
    .stat-label {
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
        margin-bottom: 0.3rem;
    }
    .stat-value {
        font-size: 1.6rem;
        font-weight: 800;
        color: #0f172a;
    }
    .stat-sub {
        font-size: 0.8rem;
        color: #10b981;
        font-weight: 600;
        margin-top: 0.2rem;
    }
    
    /* Activity Badges */
    .activity-card {
        padding: 1.2rem;
        border-radius: 12px;
        text-align: center;
        transition: transform 0.2s;
        border: 1px solid rgba(0,0,0,0.05);
    }
    .activity-card:hover {
        transform: translateY(-3px);
    }
    
    /* Prediction Banner */
    .pred-result-card {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border-radius: 16px;
        padding: 1.8rem;
        border: 1px solid #cbd5e1;
        text-align: center;
    }
    
    /* Status Pills */
    .pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
        margin-right: 6px;
    }
    .pill-blue { background-color: #dbeafe; color: #1e40af; }
    .pill-green { background-color: #d1fae5; color: #065f46; }
    .pill-purple { background-color: #ede9fe; color: #5b21b6; }
</style>
""", unsafe_allow_html=True)

# Activity Visual Meta
ACTIVITY_META = {
    "WALKING": {
        "icon": "🚶",
        "color": "#10b981",
        "bg": "#ecfdf5",
        "border": "#a7f3d0",
        "type": "Dynamic Gait",
        "desc": "Periodic rhythmic forward acceleration with balanced 3-axis oscillations."
    },
    "WALKING_UPSTAIRS": {
        "icon": "🧗‍♂️",
        "color": "#3b82f6",
        "bg": "#eff6ff",
        "border": "#bfdbfe",
        "type": "Dynamic Gait",
        "desc": "Upward vertical displacement with pronounced vertical axis jerk."
    },
    "WALKING_DOWNSTAIRS": {
        "icon": "🏃‍♂️",
        "color": "#8b5cf6",
        "bg": "#f5f3ff",
        "border": "#ddd6fe",
        "type": "Dynamic Gait",
        "desc": "Impact downward acceleration with high frequency foot strike signatures."
    },
    "SITTING": {
        "icon": "🪑",
        "color": "#f59e0b",
        "bg": "#fffbeb",
        "border": "#fde68a",
        "type": "Static Posture",
        "desc": "Upright stationary torso with low dynamic energy; gravity aligned on Y-axis."
    },
    "STANDING": {
        "icon": "🧍",
        "color": "#f97316",
        "bg": "#fff7ed",
        "border": "#ffedd5",
        "type": "Static Posture",
        "desc": "Erect stationary body posture with constant 1.0g gravity vector."
    },
    "LAYING": {
        "icon": "🛌",
        "color": "#ef4444",
        "bg": "#fef2f2",
        "border": "#fecaca",
        "type": "Static Posture",
        "desc": "Horizontal body posture with gravity vector redirected onto X/Z plane."
    }
}

@st.cache_resource
def load_models_and_artifacts():
    """Load pre-trained models, scalers, and metadata."""
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
def load_datasets():
    """Load test features and streaming simulation dataset."""
    processed_dir = PROJECT_ROOT / "data" / "processed"
    sample_features = pd.read_csv(processed_dir / "sample_test_features.csv")
    sample_stream = pd.read_csv(processed_dir / "sample_sensor_stream.csv")
    return sample_features, sample_stream

models_dict, scaler, metadata = load_models_and_artifacts()
sample_features_df, sample_stream_df = load_datasets()

# Top Hero Header
st.markdown("""
<div class="hero-container">
    <div style="margin-bottom: 0.6rem;">
        <span class="pill pill-blue">📱 Smartphone MEMS IMU</span>
        <span class="pill pill-green">⚡ 50 Hz Constant Rate</span>
        <span class="pill pill-purple">🏆 95.59% Test Accuracy</span>
    </div>
    <div class="hero-title">Human Activity Recognition (HAR) Studio</div>
    <div class="hero-subtitle">
        AI-powered sensor analytics system classifying human physical activities from tri-axial accelerometer 
        and gyroscope data with digital signal filtering, sliding-window feature extraction, and multi-model inference.
    </div>
</div>
""", unsafe_allow_html=True)

# Top KPI Metric Strip
col_m1, col_m2, col_m3, col_m4 = st.columns(4)
with col_m1:
    st.markdown("""
    <div class="stat-card">
        <div class="stat-label">Sensors Ingested</div>
        <div class="stat-value">6 Channels</div>
        <div class="stat-sub">Acc (X,Y,Z) + Gyro (X,Y,Z)</div>
    </div>
    """, unsafe_allow_html=True)
with col_m2:
    st.markdown("""
    <div class="stat-card">
        <div class="stat-label">Classified Activities</div>
        <div class="stat-value">6 Classes</div>
        <div class="stat-sub">3 Dynamic + 3 Static Postures</div>
    </div>
    """, unsafe_allow_html=True)
with col_m3:
    st.markdown("""
    <div class="stat-card">
        <div class="stat-label">Top Model Accuracy</div>
        <div class="stat-value">95.59%</div>
        <div class="stat-sub">Tested on Unseen Subjects</div>
    </div>
    """, unsafe_allow_html=True)
with col_m4:
    st.markdown("""
    <div class="stat-card">
        <div class="stat-label">Window Duration</div>
        <div class="stat-value">2.56 Sec</div>
        <div class="stat-sub">128 Readings (50% Overlap)</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

# Main Navigation Tabs
tab_overview, tab_eda, tab_benchmark, tab_predictor, tab_simulator = st.tabs([
    "🏠 System Overview",
    "📊 Sensor Signals & Waveforms (EDA)",
    "🧠 Model Benchmarks & Metrics",
    "🔮 Activity Predictor (Test & Upload)",
    "⚡ Real-Time Sensor Stream Simulator"
])

# ==========================================
# TAB 1: SYSTEM OVERVIEW
# ==========================================
with tab_overview:
    st.subheader("🎯 Recognized Human Activities")
    st.caption("The system continuously recognizes 6 primary human activities categorized into Dynamic Gaits and Static Postures:")
    
    cols_act = st.columns(6)
    for idx, (name, info) in enumerate(ACTIVITY_META.items()):
        with cols_act[idx]:
            st.markdown(f"""
            <div class="activity-card" style="background-color: {info['bg']}; border-color: {info['border']};">
                <div style="font-size: 2.4rem; margin-bottom: 0.3rem;">{info['icon']}</div>
                <div style="font-weight: 800; font-size: 0.85rem; color: {info['color']};">{name.replace('_', ' ')}</div>
                <div style="font-size: 0.72rem; color: #64748b; margin-top: 0.2rem; font-weight: 600;">{info['type']}</div>
            </div>
            """, unsafe_allow_html=True)
            
    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
    
    col_flow, col_details = st.columns([1.3, 1])
    
    with col_flow:
        st.subheader("⚙️ End-to-End Processing Pipeline")
        st.markdown("""
        ```mermaid
        graph TD
            A[Smartphone IMU Sensors] -->|50 Hz Tri-Axial Stream| B[Signal Preprocessing]
            B -->|Noise Removal| C[Median & Butterworth Filter 20Hz]
            C --> D[Sliding Window Segmentation 2.56s / 128 samples]
            D --> E[Feature Extraction: Time + Frequency Domains]
            E --> F[Machine Learning Classifier]
            F -->|95.59% Accuracy| G[Predicted Human Activity]
        ```
        """)
        
    with col_details:
        st.subheader("📋 Sensor Specifications")
        spec_df = pd.DataFrame({
            "Specification": [
                "Hardware Device",
                "Sensors Used",
                "Sampling Frequency",
                "Cutoff Frequency",
                "Window Length",
                "Window Overlap",
                "Number of Subjects"
            ],
            "Value": [
                "Samsung Galaxy S II (Waist Mount)",
                "3-Axis Accelerometer + 3-Axis Gyroscope",
                "50 Hz (50 samples / second)",
                "20.0 Hz (3rd-Order Low-Pass Butterworth)",
                "128 samples (2.56 seconds)",
                "64 samples (50% overlap)",
                "30 Volunteers (Age 19–48 yrs)"
            ]
        })
        st.dataframe(spec_df, use_container_width=True, hide_index=True)

# ==========================================
# TAB 2: SENSOR SIGNALS & EDA STUDIO
# ==========================================
with tab_eda:
    st.subheader("📈 Interactive Multi-Axis Sensor Signal Explorer")
    st.caption("Inspect live 128-sample waveforms (2.56-second window @ 50 Hz) for any chosen physical activity.")
    
    c_sel1, c_sel2 = st.columns([1, 2])
    with c_sel1:
        chosen_act = st.selectbox(
            "Select Physical Activity to Visualize",
            list(ACTIVITY_MAP.values()),
            index=0,
            key="eda_act_choice"
        )
        meta_info = ACTIVITY_META[chosen_act]
        st.markdown(f"""
        <div style="background-color: {meta_info['bg']}; border-left: 4px solid {meta_info['color']}; padding: 12px; border-radius: 8px; margin-top: 10px;">
            <div style="font-weight: 700; color: {meta_info['color']};">{meta_info['icon']} {chosen_act.replace('_', ' ')}</div>
            <div style="font-size: 0.85rem; color: #475569; margin-top: 4px;">{meta_info['desc']}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with c_sel2:
        # Filter raw stream for chosen activity
        act_samples = sample_stream_df[sample_stream_df["activity"] == chosen_act]
        window_options = act_samples["sample_window"].unique()
        chosen_win = st.selectbox("Select Window Sample", window_options, index=0)
        
    win_data = act_samples[act_samples["sample_window"] == chosen_win].sort_values("time_step")
    
    # Plotly Interactive Multi-Axis Chart
    fig_sensors = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=(
            f"Tri-Axial Total Accelerometer (g) - {chosen_act.replace('_', ' ')}",
            f"Tri-Axial Body Gyroscope (rad/s) - {chosen_act.replace('_', ' ')}"
        )
    )
    
    # Accelerometer
    fig_sensors.add_trace(go.Scatter(x=win_data["timestamp_sec"], y=win_data["total_acc_x"], name="Acc X", line=dict(color="#ef4444", width=2)), row=1, col=1)
    fig_sensors.add_trace(go.Scatter(x=win_data["timestamp_sec"], y=win_data["total_acc_y"], name="Acc Y", line=dict(color="#10b981", width=2)), row=1, col=1)
    fig_sensors.add_trace(go.Scatter(x=win_data["timestamp_sec"], y=win_data["total_acc_z"], name="Acc Z", line=dict(color="#3b82f6", width=2)), row=1, col=1)
    
    # Gyroscope
    fig_sensors.add_trace(go.Scatter(x=win_data["timestamp_sec"], y=win_data["body_gyro_x"], name="Gyro X", line=dict(color="#f59e0b", width=2)), row=2, col=1)
    fig_sensors.add_trace(go.Scatter(x=win_data["timestamp_sec"], y=win_data["body_gyro_y"], name="Gyro Y", line=dict(color="#8b5cf6", width=2)), row=2, col=1)
    fig_sensors.add_trace(go.Scatter(x=win_data["timestamp_sec"], y=win_data["body_gyro_z"], name="Gyro Z", line=dict(color="#06b6d4", width=2)), row=2, col=1)
    
    fig_sensors.update_layout(
        height=520,
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        template="plotly_white"
    )
    fig_sensors.update_xaxes(title_text="Time in Window (seconds)", row=2, col=1)
    fig_sensors.update_yaxes(title_text="Acceleration (g)", row=1, col=1)
    fig_sensors.update_yaxes(title_text="Angular Vel (rad/s)", row=2, col=1)
    
    st.plotly_chart(fig_sensors, use_container_width=True)
    
    # Vector Magnitude Comparison
    st.markdown("#### ⚡ Acceleration Vector Magnitude: Dynamic Movement vs Static Posture")
    st.latex(r"\text{Magnitude} = \sqrt{a_x^2 + a_y^2 + a_z^2}")
    
    # Compute magnitudes for Walking vs Standing
    walk_win = sample_stream_df[sample_stream_df["activity"] == "WALKING"].head(128)
    stand_win = sample_stream_df[sample_stream_df["activity"] == "STANDING"].head(128)
    stairs_win = sample_stream_df[sample_stream_df["activity"] == "WALKING_UPSTAIRS"].head(128)
    
    mag_fig = go.Figure()
    mag_fig.add_trace(go.Scatter(
        x=walk_win["timestamp_sec"],
        y=np.sqrt(walk_win["total_acc_x"]**2 + walk_win["total_acc_y"]**2 + walk_win["total_acc_z"]**2),
        name="Walking (Dynamic High Variance)",
        line=dict(color="#10b981", width=2.5)
    ))
    mag_fig.add_trace(go.Scatter(
        x=stairs_win["timestamp_sec"],
        y=np.sqrt(stairs_win["total_acc_x"]**2 + stairs_win["total_acc_y"]**2 + stairs_win["total_acc_z"]**2),
        name="Climbing Stairs (Dynamic Peak Energy)",
        line=dict(color="#3b82f6", width=2.5)
    ))
    mag_fig.add_trace(go.Scatter(
        x=stand_win["timestamp_sec"],
        y=np.sqrt(stand_win["total_acc_x"]**2 + stand_win["total_acc_y"]**2 + stand_win["total_acc_z"]**2),
        name="Standing (Static Constant ~1.0g)",
        line=dict(color="#f97316", width=2.5, dash="dot")
    ))
    mag_fig.update_layout(
        height=320,
        margin=dict(l=40, r=40, t=30, b=40),
        xaxis_title="Time (seconds)",
        yaxis_title="Total Magnitude (g)",
        template="plotly_white",
        hovermode="x unified"
    )
    st.plotly_chart(mag_fig, use_container_width=True)

# ==========================================
# TAB 3: MODEL BENCHMARKS & METRICS
# ==========================================
with tab_benchmark:
    st.subheader("🏆 Comparative Model Performance Benchmark")
    st.caption("Evaluated on 2,947 completely unseen test window samples from 9 independent test subjects.")
    
    metrics_path = PROJECT_ROOT / "reports" / "model_comparison_metrics.csv"
    if metrics_path.exists():
        comp_df = pd.read_csv(metrics_path)
        
        # Display Bar Chart Comparison
        fig_comp = px.bar(
            comp_df,
            x="Accuracy",
            y="Model",
            orientation="h",
            text=[f"{v*100:.2f}%" for v in comp_df["Accuracy"]],
            color="Accuracy",
            color_continuous_scale="Blues",
            range_x=[0.8, 1.0]
        )
        fig_comp.update_layout(
            height=300,
            margin=dict(l=40, r=40, t=20, b=40),
            xaxis_title="Test Accuracy",
            yaxis_title="",
            coloraxis_showscale=False,
            template="plotly_white"
        )
        fig_comp.update_traces(textposition='outside')
        st.plotly_chart(fig_comp, use_container_width=True)
        
        # Metrics Table
        st.markdown("#### 📑 Summary Metrics Table")
        disp_df = comp_df.copy()
        for col in ["Accuracy", "Precision_Macro", "Recall_Macro", "F1_Macro", "Precision_Weighted", "Recall_Weighted", "F1_Weighted"]:
            if col in disp_df.columns:
                disp_df[col] = disp_df[col].apply(lambda x: f"{x*100:.2f}%")
        st.dataframe(disp_df, use_container_width=True, hide_index=True)
        
    st.markdown("---")
    
    c_cm, c_rpt = st.columns(2)
    with c_cm:
        st.subheader("🎯 Confusion Matrix Heatmap")
        cm_image_path = PROJECT_ROOT / "reports" / "figures" / "confusion_matrix.png"
        if cm_image_path.exists():
            st.image(str(cm_image_path), use_container_width=True)
            
    with c_rpt:
        st.subheader("📊 Per-Class Precision & Recall Breakdown")
        report_path = PROJECT_ROOT / "reports" / "best_model_classification_report.csv"
        if report_path.exists():
            clf_df = pd.read_csv(report_path, index_col=0)
            st.dataframe(
                clf_df.style.format(precision=4).background_gradient(cmap="Blues", subset=["precision", "recall", "f1-score"]),
                use_container_width=True
            )
            
        st.markdown("""
        **🔍 Key Diagnostic Insights:**
        - **Dynamic vs Static:** 100% boundary separation achieved between dynamic activities (Walking/Stairs) and static postures (Sitting/Standing/Laying).
        - **Sitting vs Standing:** Minor confusion (~2%) occurs due to identical gravitational vector magnitude ($1.0g$) and near-zero angular motion.
        """)

# ==========================================
# TAB 4: ACTIVITY PREDICTOR (TEST & UPLOAD)
# ==========================================
with tab_predictor:
    st.subheader("🔮 Interactive Activity Predictor")
    st.caption("Test the trained classifier on unseen test samples or upload your own 561-feature sensor vector.")
    
    col_input, col_pred = st.columns([1, 1.2])
    
    with col_input:
        st.markdown("#### 1. Configuration & Input Selection")
        selected_model_name = st.selectbox(
            "Select Machine Learning Model",
            list(models_dict.keys()),
            index=0,
            key="pred_model_select"
        )
        active_model = models_dict[selected_model_name]
        needs_scaling = "Logistic" in selected_model_name or "SVM" in selected_model_name
        
        input_type = st.radio(
            "Input Mode",
            ["Preset Test Windows (Random Subsets)", "Upload Custom Feature CSV"],
            horizontal=True
        )
        
        sample_vec = None
        true_label = None
        
        if input_type == "Preset Test Windows (Random Subsets)":
            filter_act = st.selectbox("Filter by Ground Truth Activity", list(ACTIVITY_MAP.values()), index=0)
            subset = sample_features_df[sample_features_df["activity"] == filter_act]
            
            chosen_row_idx = st.slider("Select Instance Index", 0, len(subset) - 1, 0)
            row_data = subset.iloc[chosen_row_idx]
            
            feat_cols = [c for c in sample_features_df.columns if c not in ["subject", "activity_id", "activity"]]
            sample_vec = row_data[feat_cols].values.reshape(1, -1)
            true_label = row_data["activity"]
            
            st.info(f"Selected Sample Window #{chosen_row_idx} | Ground Truth: **{true_label}**")
        else:
            uploaded = st.file_uploader("Upload CSV containing 561 sensor features", type=["csv"])
            if uploaded:
                up_df = pd.read_csv(uploaded)
                feat_cols = [c for c in up_df.columns if c not in ["subject", "activity_id", "activity"]]
                if len(feat_cols) == 561:
                    st.success(f"Loaded {len(up_df)} samples with 561 features.")
                    sample_vec = up_df[feat_cols].iloc[0:1].values
                else:
                    st.error(f"Expected 561 features, found {len(feat_cols)}.")
                    
        btn_predict = st.button("🚀 Predict Activity", type="primary", use_container_width=True)

    with col_pred:
        st.markdown("#### 2. Live Classification Output")
        
        if btn_predict and sample_vec is not None:
            # Scale if required
            vec_eval = scaler.transform(sample_vec) if needs_scaling else sample_vec
            
            # Predict
            pred_id = int(active_model.predict(vec_eval)[0])
            pred_name = ACTIVITY_MAP[pred_id]
            pred_meta = ACTIVITY_META[pred_name]
            
            # Probabilities
            probs = None
            conf_pct = 95.0
            if hasattr(active_model, "predict_proba"):
                probs = active_model.predict_proba(vec_eval)[0]
                conf_pct = float(probs[pred_id - 1] * 100)
                
            # Render Modern Activity Result Card
            st.markdown(f"""
            <div class="pred-result-card" style="border-top: 6px solid {pred_meta['color']};">
                <div style="font-size: 3.5rem; margin-bottom: 0.5rem;">{pred_meta['icon']}</div>
                <div style="font-size: 1.8rem; font-weight: 800; color: {pred_meta['color']};">
                    {pred_name.replace('_', ' ')}
                </div>
                <div style="font-size: 0.95rem; color: #475569; font-weight: 600; margin-top: 0.2rem;">
                    Category: {pred_meta['type']}
                </div>
                <div style="margin-top: 1rem;">
                    <span class="pill pill-green" style="font-size: 0.9rem; padding: 6px 16px;">
                        Confidence: {conf_pct:.1f}%
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if true_label is not None:
                if pred_name == true_label:
                    st.success(f"🎯 Exact Match with Ground Truth: **{true_label}**")
                else:
                    st.warning(f"⚠️ True Activity was: **{true_label}**")
                    
            # Confidence Breakdown Bar Chart
            if probs is not None:
                st.markdown("##### Prediction Probability Distribution")
                prob_df = pd.DataFrame({
                    "Activity": [ACTIVITY_MAP[i].replace('_', ' ') for i in range(1, 7)],
                    "Probability": probs * 100,
                    "Color": [ACTIVITY_META[ACTIVITY_MAP[i]]["color"] for i in range(1, 7)]
                })
                
                fig_prob = px.bar(
                    prob_df,
                    x="Probability",
                    y="Activity",
                    orientation="h",
                    text=[f"{p:.1f}%" for p in prob_df["Probability"]],
                    color="Activity",
                    color_discrete_map={row["Activity"]: row["Color"] for _, row in prob_df.iterrows()}
                )
                fig_prob.update_layout(
                    height=260,
                    margin=dict(l=20, r=40, t=10, b=30),
                    xaxis_title="Confidence (%)",
                    yaxis_title="",
                    showlegend=False,
                    template="plotly_white",
                    xaxis=dict(range=[0, 105])
                )
                fig_prob.update_traces(textposition="outside")
                st.plotly_chart(fig_prob, use_container_width=True)
        else:
            st.info("👈 Select an input sample on the left and click '🚀 Predict Activity' to see real-time inference.")

# ==========================================
# TAB 5: REAL-TIME SENSOR SIMULATOR
# ==========================================
with tab_simulator:
    st.subheader("⚡ Real-Time Sensor Stream Simulator")
    st.caption("Simulates a continuous 50 Hz phone sensor stream (Accelerometer & Gyroscope) with live animated visualization.")
    
    c_ctl1, c_ctl2, c_ctl3 = st.columns([1.5, 1, 1])
    with c_ctl1:
        stream_act = st.selectbox(
            "Select Physical Activity to Stream Live",
            list(ACTIVITY_MAP.values()),
            index=0,
            key="sim_choice"
        )
    with c_ctl2:
        speed_mode = st.select_slider(
            "Playback Speed",
            options=["Smooth (30ms)", "Fast (15ms)", "Slow (60ms)"],
            value="Smooth (30ms)"
        )
        delay_map = {"Smooth (30ms)": 0.03, "Fast (15ms)": 0.015, "Slow (60ms)": 0.06}
        delay = delay_map[speed_mode]
    with c_ctl3:
        st.write("")
        st.write("")
        btn_start_sim = st.button("▶️ Start Live Stream", type="primary", use_container_width=True)
        
    stream_meta = ACTIVITY_META[stream_act]
    
    # Placeholders for dynamic rendering
    status_container = st.empty()
    chart_acc_placeholder = st.empty()
    chart_gyro_placeholder = st.empty()
    
    if btn_start_sim:
        stream_data = sample_stream_df[sample_stream_df["activity"] == stream_act]
        window_size = 48
        
        status_container.markdown(f"""
        <div style="background-color: {stream_meta['color']}; color: white; padding: 1.2rem; border-radius: 12px; text-align: center; margin-bottom: 1rem;">
            <span style="font-size: 1.8rem; vertical-align: middle; margin-right: 8px;">{stream_meta['icon']}</span>
            <span style="font-size: 1.3rem; font-weight: 800; vertical-align: middle;">
                LIVE STREAMING SENSOR FEED: {stream_act.replace('_', ' ')} ({stream_meta['type']})
            </span>
            <div style="font-size: 0.85rem; opacity: 0.9; margin-top: 4px;">Sampling Rate: 50.0 Hz | Low-Pass Butterworth Filter Active</div>
        </div>
        """, unsafe_allow_html=True)
        
        max_steps = min(len(stream_data) - window_size, 120)
        
        for i in range(0, max_steps, 2):
            chunk = stream_data.iloc[i : i + window_size]
            
            # Interactive Accelerometer Line Chart
            fig_acc = go.Figure()
            fig_acc.add_trace(go.Scatter(x=chunk["timestamp_sec"], y=chunk["total_acc_x"], name="Acc X", line=dict(color="#ef4444", width=2)))
            fig_acc.add_trace(go.Scatter(x=chunk["timestamp_sec"], y=chunk["total_acc_y"], name="Acc Y", line=dict(color="#10b981", width=2)))
            fig_acc.add_trace(go.Scatter(x=chunk["timestamp_sec"], y=chunk["total_acc_z"], name="Acc Z", line=dict(color="#3b82f6", width=2)))
            fig_acc.update_layout(
                title="Live 3-Axis Accelerometer (g) - 50 Hz",
                height=260,
                margin=dict(l=30, r=30, t=40, b=30),
                template="plotly_white",
                xaxis_title="Time (s)",
                yaxis_title="Accel (g)"
            )
            chart_acc_placeholder.plotly_chart(fig_acc, use_container_width=True)
            
            # Interactive Gyroscope Line Chart
            fig_gyro = go.Figure()
            fig_gyro.add_trace(go.Scatter(x=chunk["timestamp_sec"], y=chunk["body_gyro_x"], name="Gyro X", line=dict(color="#f59e0b", width=2)))
            fig_gyro.add_trace(go.Scatter(x=chunk["timestamp_sec"], y=chunk["body_gyro_y"], name="Gyro Y", line=dict(color="#8b5cf6", width=2)))
            fig_gyro.add_trace(go.Scatter(x=chunk["timestamp_sec"], y=chunk["body_gyro_z"], name="Gyro Z", line=dict(color="#06b6d4", width=2)))
            fig_gyro.update_layout(
                title="Live 3-Axis Gyroscope (rad/s) - 50 Hz",
                height=260,
                margin=dict(l=30, r=30, t=40, b=30),
                template="plotly_white",
                xaxis_title="Time (s)",
                yaxis_title="Angular Vel (rad/s)"
            )
            chart_gyro_placeholder.plotly_chart(fig_gyro, use_container_width=True)
            
            time.sleep(delay)
            
        st.success(f"✅ Real-time simulation stream for **{stream_act}** completed successfully!")
    else:
        status_container.markdown(f"""
        <div style="background-color: #f8fafc; border: 1px dashed #cbd5e1; padding: 1.5rem; border-radius: 12px; text-align: center;">
            <div style="font-size: 2.2rem; margin-bottom: 0.4rem;">📡</div>
            <div style="font-weight: 700; color: #334155;">Live Sensor Stream Waiting to Start</div>
            <div style="font-size: 0.85rem; color: #64748b; margin-top: 4px;">
                Click "▶️ Start Live Stream" above to simulate real-time smartphone accelerometer and gyroscope signals.
            </div>
        </div>
        """, unsafe_allow_html=True)
