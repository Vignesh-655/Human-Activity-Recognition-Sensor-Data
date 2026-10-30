# 📱 AI-Based Human Activity Recognition (HAR) Using Smartphone Sensor Data

[![Live Demo](https://img.shields.io/badge/Live_Demo-Streamlit_Cloud-FF4B4B.svg?style=for-the-badge&logo=streamlit)](https://human-activity-recognition-sensor-data-vignesh.streamlit.app/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

> 🚀 **Live Web App:** [https://human-activity-recognition-sensor-data-vignesh.streamlit.app/](https://human-activity-recognition-sensor-data-vignesh.streamlit.app/)  
> **Project Owner & Developer:** [Vignesh](https://github.com/Vignesh-655)  
> **Domain:** ECE & AI Capstone Project  
> An end-to-end Machine Learning and Signal Processing pipeline that classifies human physical activities using tri-axial accelerometer and gyroscope data from smartphones.

---

## 📌 Table of Contents
- [Project Overview](#-project-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Academic Theory & DSP Pipeline](#-academic-theory--dsp-pipeline)
- [Dataset Details](#-dataset-details)
- [Model Performance Benchmark](#-model-performance-benchmark)
- [College Viva-Voce Questions & Answers](#-college-viva-voce-questions--answers)
- [Repository Structure](#-repository-structure)
- [Quick Start Guide](#-quick-start-guide)
- [Author & Project Owner](#-author--project-owner)
- [License](#-license)

---

## 🎯 Project Overview
Wearable devices (smartphones, smartwatches, fitness trackers) continuously capture movement dynamics through embedded **accelerometers** and **gyroscopes**. This system processes multi-axial sensor streams to automatically detect 6 everyday physical activities:
1. 🚶 **WALKING**
2. 🧗‍♂️ **WALKING_UPSTAIRS**
3. 🏃‍♂️ **WALKING_DOWNSTAIRS**
4. 🪑 **SITTING**
5. 🧍 **STANDING**
6. 🛌 **LAYING**

---

## ✨ Key Features
- **Signal Preprocessing:** 3rd-order digital low-pass Butterworth filter ($f_c = 20\text{ Hz}$) and median filtering for noise and glitch attenuation.
- **Sliding-Window Segmentation:** 2.56-second fixed-width windows (128 readings @ 50 Hz) with 50% overlap.
- **Rich Feature Engineering:** Time-domain statistics (Mean, Std, SMA, Energy, IQR, Entropy) and Frequency-domain descriptors (FFT, Spectral Energy, Peak Frequency).
- **Multi-Model Benchmark:** Comparative evaluation across Logistic Regression, Support Vector Machine (SVM), Random Forest, and Decision Tree.
- **High Accuracy:** Achieves **95.59% test accuracy** and **0.9558 macro F1-score** on unseen test subjects.
- **Interactive Web UI:** Modern Streamlit dashboard featuring live 50Hz sensor stream simulation, batch prediction, and visual analytics.
- **Self-Contained Notebook:** Fully documented Jupyter/Colab notebook (`notebooks/HAR_Model_Training_and_EDA.ipynb`).

---

## 🏗️ System Architecture

```
┌──────────────────────┐      ┌─────────────────────────┐      ┌─────────────────────────┐
│  Smartphone Sensors  │ ───> │ Signal Preprocessing    │ ───> │ Sliding Window (2.56s)  │
│  • 3-Axial Accel (g) │      │ • Butterworth Low-Pass  │      │ • 128 sensor readings   │
│  • 3-Axial Gyro (rad)│      │ • Median Noise Filter   │      │ • 50% window overlap    │
└──────────────────────┘      └─────────────────────────┘      └─────────────────────────┘
                                                                            │
┌──────────────────────┐      ┌─────────────────────────┐                   │
│  Streamlit Web UI    │ <─── │ Multi-Class ML Model    │ <─────────────────┘
│  • Activity Badge    │      │ • Logistic Reg (95.59%) │      Feature Extraction:
│  • Confidence Gauge  │      │ • SVM RBF Kernel (95.5%)│      • Time Domain (Mean, Std, SMA)
│  • Real-time Stream  │      │ • Random Forest (92.8%) │      • Frequency Domain (FFT, Peak)
└──────────────────────┘      └─────────────────────────┘      • Vector Magnitude (Norm)
```

---

## 🔬 Academic Theory & DSP Pipeline

### 1. MEMS Sensor Physics & Working Principle
- **Capacitive Accelerometer:** Measures deflection of a microscopic proof mass suspended by silicon beams ($F = ma = kx$). The resulting differential capacitance $\Delta C = \frac{2\epsilon A}{d_0^2}x$ is converted to acceleration ($g$) across $X, Y, Z$.
- **Vibratory Gyroscope:** Detects angular rate of rotation based on the **Coriolis Effect** ($\mathbf{F}_{\text{Coriolis}} = -2m (\mathbf{\Omega} \times \mathbf{v})$), outputting rotational velocity in $\text{rad/s}$.

### 2. Digital Filtering & Decoupling Equation
- **Butterworth Filter (20 Hz):** Attenuates high-frequency sensor noise using a 3rd-order digital low-pass Butterworth filter ($f_c = 20\text{ Hz}$).
- **Gravity Decoupling:** Measured acceleration combines dynamic body movement $\mathbf{a}_{\text{body}}(t)$ and constant gravity $\mathbf{g}(t)$:
  $$\mathbf{a}_{\text{total}}(t) = \mathbf{a}_{\text{body}}(t) + \mathbf{g}(t)$$
  A low-pass filter with cutoff $f_c = 0.3\text{ Hz}$ separates static orientation from dynamic motion.

### 3. Sliding Window & Mathematical Features
- **Window Length:** 128 samples ($2.56\text{ seconds}$ @ 50 Hz), covering at least two full human strides.
- **50% Overlap:** 64 samples ($1.28\text{ seconds}$) for continuous temporal context.
- **Key Features:**
  - Signal Magnitude Area: $\text{SMA} = \frac{1}{N}\sum (|a_x| + |a_y| + |a_z|)$
  - Euclidean Vector Magnitude: $\text{VM} = \sqrt{a_x^2 + a_y^2 + a_z^2}$
  - Time-Domain: Mean, Variance, Std, MAD, IQR, Skewness, Kurtosis, Energy, Shannon Entropy.
  - Frequency-Domain (FFT): Peak Frequency, Spectral Energy.

---

## 📊 Dataset Details
The model is trained and benchmarked on the gold-standard **UCI Human Activity Recognition Using Smartphones Dataset**:
- **Subjects:** 30 human volunteers aged 19–48.
- **Sensors:** 3-axial linear acceleration and 3-axial angular velocity at 50 Hz.
- **Instances:** 10,299 window samples (7,352 train / 2,947 test).
- **Source Link:** [UCI Machine Learning Repository - HAR Dataset](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)

---

## 🏆 Model Performance Benchmark

Evaluated on 2,947 completely unseen test instances from 9 independent subjects:

| Algorithm | Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) | F1-Score (Weighted) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Logistic Regression** | **95.59%** | **0.9588** | **0.9546** | **0.9558** | **0.9558** |
| **Support Vector Machine (RBF)** | **95.49%** | **0.9559** | **0.9531** | **0.9540** | **0.9547** |
| **Random Forest (100 Trees)** | **92.77%** | **0.9293** | **0.9249** | **0.9262** | **0.9276** |
| **Decision Tree (CART)** | **86.43%** | **0.8640** | **0.8607** | **0.8611** | **0.8634** |

### Per-Class Performance (Best Model)
| Activity | Precision | Recall | F1-Score |
|:---|:---:|:---:|:---:|
| **WALKING** | 94.26% | 99.40% | 96.76% |
| **WALKING_UPSTAIRS** | 96.12% | 94.69% | 95.40% |
| **WALKING_DOWNSTAIRS** | 99.00% | 94.29% | 96.59% |
| **SITTING** | 97.07% | 87.78% | 92.19% |
| **STANDING** | 89.02% | 97.56% | 93.09% |
| **LAYING** | 99.81% | 99.07% | 99.44% |

---

## 📁 Repository Structure
```
├── app.py                             # Interactive Streamlit Web Application
├── requirements.txt                   # Project Python packages
├── README.md                          # Repository documentation
├── .gitignore                         # Files excluded from git
├── data/
│   ├── raw/UCI HAR Dataset/           # Downloaded raw sensor dataset
│   └── processed/                     # Pre-split CSV files and streaming samples
├── src/
│   ├── download_dataset.py            # Automated dataset downloader
│   ├── data_loader.py                 # Sensor data parsing and loader
│   ├── preprocessing.py               # Butterworth filter & feature scaling
│   ├── feature_extraction.py          # Sliding window & FFT feature extraction
│   ├── eda.py                         # Generates publication-grade figures
│   ├── train.py                       # Trains all 4 ML models and saves artifacts
│   ├── evaluate.py                    # Standalone diagnostic evaluation script
│   └── generate_notebook.py           # Colab / Jupyter notebook generator
├── models/
│   ├── best_har_model.pkl             # Serialized top model (Logistic Regression)
│   ├── scaler.pkl                     # Serialized StandardScaler
│   ├── model_metadata.json            # Model parameters and label dictionary
│   └── *.pkl                          # Serialized SVM, RF, and Decision Tree models
├── notebooks/
│   └── HAR_Model_Training_and_EDA.ipynb # Jupyter/Colab notebook
└── reports/
    ├── HAR_Project_Report.md          # Comprehensive formal capstone report
    ├── model_comparison_metrics.csv   # Performance metric CSV
    ├── best_model_classification_report.csv # Class-wise metrics
    └── figures/                       # Generated analysis plots
        ├── activity_distribution.png
        ├── raw_signals_by_activity.png
        ├── signal_magnitude_comparison.png
        ├── pca_activity_separation.png
        ├── correlation_heatmap.png
        ├── confusion_matrix.png
        └── feature_importance.png
```

---

## 🎓 College Viva-Voce Questions & Answers (Top 5)

<details>
<summary><b>Q1: Why is the sampling rate chosen as 50 Hz?</b></summary>
<blockquote>
Human physical body motion occurs within 0 – 20 Hz. By the Nyquist-Shannon Sampling Theorem ($f_s \ge 2 f_{\text{max}}$), a 50 Hz sample rate satisfies $50 \ge 40\text{ Hz}$, capturing natural kinematics without aliasing or excessive battery drain.
</blockquote>
</details>

<details>
<summary><b>Q2: Why separate total acceleration into body and gravitational components?</b></summary>
<blockquote>
Total acceleration combines static gravity ($1.0g$) and dynamic user movement. Gravity ($f < 0.3\text{ Hz}$) indicates device tilt orientation relative to Earth, identifying static postures (standing vs laying). Body motion ($f > 0.3\text{ Hz}$) differentiates dynamic gaits (walking vs stairs).
</blockquote>
</details>

<details>
<summary><b>Q3: What is Signal Magnitude Area (SMA)?</b></summary>
<blockquote>
$\text{SMA} = \frac{1}{N}\sum (|a_x| + |a_y| + |a_z|)$. It calculates the cumulative 3D kinetic energy expenditure, acting as an orientation-invariant indicator of movement intensity.
</blockquote>
</details>

<details>
<summary><b>Q4: Why do Logistic Regression and SVM achieve ~95.6% while Decision Tree gets ~86.4%?</b></summary>
<blockquote>
Decision Trees make orthogonal axis-aligned splits, which struggle with continuous correlated features across 561 dimensions. Logistic Regression and SVM with RBF kernels create smooth optimal hyperplanes, maximizing generalizability.
</blockquote>
</details>

<details>
<summary><b>Q5: Why do Sitting and Standing have minor misclassifications?</b></summary>
<blockquote>
Both are stationary postures with near-zero body acceleration and identical $1.0g$ gravitational magnitude. Slight pocket orientation angles create occasional decision boundary overlap.
</blockquote>
</details>

---

## 🚀 Quick Start Guide

### 1. Clone or Download Repository
```bash
git clone https://github.com/Vignesh-655/Human-Activity-Recognition-Sensor-Data.git
cd Human-Activity-Recognition-Sensor-Data
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit Dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 4. Retrain Models (Optional)
```bash
python src/train.py
```

### 5. Run Evaluation Script
```bash
python src/evaluate.py
```

---

## 👤 Author & Project Owner
- **Vignesh** — Lead Developer & Researcher
- **GitHub Profile:** [@Vignesh-655](https://github.com/Vignesh-655)
- **Live Streamlit App:** [Human Activity Recognition Studio](https://human-activity-recognition-sensor-data-vignesh.streamlit.app/)
- **Repository:** [Vignesh-655/Human-Activity-Recognition-Sensor-Data](https://github.com/Vignesh-655/Human-Activity-Recognition-Sensor-Data)

---

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

