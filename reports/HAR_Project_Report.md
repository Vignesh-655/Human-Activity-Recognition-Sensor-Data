# Human Activity Recognition (HAR) Using Smartphone Sensor Data
## Comprehensive Capstone Project Report
**Department:** Electronics & Communication Engineering / Artificial Intelligence & Data Science  
**Domain:** Wearable Computing, Sensor Signal Processing, Machine Learning  
**Project Title:** AI-Based Human Activity Recognition (HAR) Using Smartphone Tri-Axial Accelerometer and Gyroscope Data  

---

## 1. Executive Summary
Human Activity Recognition (HAR) plays a foundational role in pervasive computing, digital healthcare monitoring, ambient assisted living, athletic performance analysis, and smart fall detection systems. This capstone project develops an end-to-end artificial intelligence system capable of automatically classifying human physical activities using continuous sensor streams from smartphone inertial measurement units (IMU: tri-axial accelerometer and tri-axial gyroscope).

Using the benchmark UCI Human Activity Recognition dataset comprising 10,299 instances across 30 subjects sampled at 50 Hz, we implemented a full machine learning lifecycle:
1. Signal conditioning and noise attenuation using a 3rd-order digital low-pass Butterworth filter (20 Hz cutoff) and median filtering.
2. Sliding-window segmentation with 2.56-second time windows (128 samples) and 50% overlap.
3. Feature engineering encompassing statistical time-domain and frequency-domain (Fast Fourier Transform) descriptors.
4. Comparative benchmark of four distinct machine learning classifiers: Support Vector Machine (SVM), Random Forest, Logistic Regression, and Decision Tree.
5. Model validation on completely unseen test subjects, achieving **95.59% classification accuracy** and **0.9558 macro F1-score**.
6. Deployment of an interactive **Streamlit Web Dashboard** featuring live 50Hz sensor stream simulation, batch file inference, and visual analytics.

---

## 2. Introduction & Problem Statement

### 2.1 Background & Motivation
Modern smartphones and wearable smartwatches are equipped with embedded Micro-Electro-Mechanical Systems (MEMS) sensors—primarily 3-axis accelerometers and 3-axis gyroscopes. These sensors capture physical movement, gravitational orientation, and rotational velocity with high temporal fidelity while remaining unobtrusive to users.

Accurately recognizing human activity enables numerous transformative applications:
- **Elderly Care & Fall Detection:** Identifying sudden transitions from standing/walking to laying down.
- **Cardiovascular Health & Calorie Tracking:** Quantifying active vs sedentary intervals during rehabilitation.
- **Smart Ergonomics:** Warning computer workers of excessive prolonged sitting posture.

### 2.2 Project Scope & Objectives
The system must address the following technical challenges:
1. **Sensor Modalities:** Ingest 3-axial linear acceleration ($a_x, a_y, a_z$) and 3-axial angular velocity ($\omega_x, \omega_y, \omega_z$).
2. **Activity Classification:** Categorize six diverse physical activities:
   - Dynamic: `WALKING`, `WALKING_UPSTAIRS`, `WALKING_DOWNSTAIRS`
   - Static Postures: `SITTING`, `STANDING`, `LAYING`
3. **Data Preprocessing & Signal Filtering:** Filter out sensor drift and high-frequency noise while isolating body motion from gravitational acceleration.
4. **Sliding-Window Feature Extraction:** Convert temporal time-series windows into discriminative feature vectors.
5. **Model Generalization:** Evaluate models on unseen test subjects to prevent subject-dependent overfitting.
6. **Deployment:** Provide an accessible, user-friendly graphical dashboard and reproducible Colab/Jupyter notebook.

---

## 3. Dataset Architecture & Source Description

We utilized the gold-standard **UCI Human Activity Recognition (HAR) Dataset**.

### 3.1 Experimental Protocol
- **Subjects:** 30 healthy human volunteers aged 19–48 years.
- **Device:** Samsung Galaxy S II smartphone strapped to the waist.
- **Sensors:** Embedded tri-axial accelerometer and tri-axial gyroscope.
- **Sampling Rate:** 50 Hz constant frequency (50 sensor samples per second).
- **Video Ground Truth:** Experiments were video-recorded to manually label sensor segments with ground truth activity labels.
- **Partition:** 70% of subjects (21 subjects, 7,352 samples) allocated to the training split, and 30% of subjects (9 unseen subjects, 2,947 samples) reserved strictly for the test split.

### 3.2 Target Activity Classes
| Activity ID | Activity Name | Class Type | Biomechanical Description |
|:---:|:---|:---:|:---|
| **1** | `WALKING` | Dynamic | Continuous horizontal forward gait on level ground. |
| **2** | `WALKING_UPSTAIRS` | Dynamic | Upward vertical displacement against gravity on staircases. |
| **3** | `WALKING_DOWNSTAIRS` | Dynamic | Downward vertical displacement with gravity on staircases. |
| **4** | `SITTING` | Static Posture | Torso upright, thighs horizontal, stationary posture. |
| **5** | `STANDING` | Static Posture | Full body vertical and stationary posture. |
| **6** | `LAYING` | Static Posture | Full body horizontal resting on bed/ground. |

---

## 4. Signal Processing & Preprocessing Methodology

Raw accelerometer and gyroscope signals are subject to environmental noise, sensor tremors, and high-frequency sensor noise. 

```
Raw Sensor Stream (50 Hz)
         │
         ▼
[ Median Filter (Kernel=3) ]  ──> Suppresses abrupt sensor glitches / single-sample spikes
         │
         ▼
[ 3rd-Order Low-Pass Butterworth Filter (Cutoff = 20 Hz) ] ──> Attenuates high-frequency noise
         │
         ▼
[ Low-Pass Filter (Cutoff = 0.3 Hz) ] ──> Splits Total Acc into:
         ├── Body Acceleration (Dynamic Movement > 0.3 Hz)
         └── Gravitational Acceleration (Static Orientation <= 0.3 Hz)
```

### 4.1 Digital Butterworth Filter
Human body motion frequency rarely exceeds $15 \text{ Hz}$ to $20 \text{ Hz}$. A 3rd-order digital low-pass Butterworth filter with a $20 \text{ Hz}$ cutoff frequency (Nyquist frequency $f_N = 25 \text{ Hz}$) was applied using zero-phase forward-backward digital filtering (`scipy.signal.filtfilt`):
$$|H(f)|^2 = \frac{1}{1 + \left(\frac{f}{f_c}\right)^{2n}}$$
Where $f_c = 20\text{ Hz}$ and order $n = 3$.

### 4.2 Gravitational vs Body Acceleration Separation
The measured total acceleration vector $\mathbf{a}_{\text{total}}(t)$ consists of gravitational acceleration $\mathbf{g}(t)$ and dynamic user body acceleration $\mathbf{a}_{\text{body}}(t)$:
$$\mathbf{a}_{\text{total}}(t) = \mathbf{a}_{\text{body}}(t) + \mathbf{g}(t)$$
Because gravity produces a constant low-frequency component ($< 0.3\text{ Hz}$), a low-pass filter with a $0.3\text{ Hz}$ cutoff cleanly isolates the constant gravitational vector, while the high-pass residual yields body acceleration.

---

## 5. Sliding-Window Segmentation & Feature Engineering

### 5.1 Windowing Technique
Sensor streams are partitioned using fixed-width sliding windows:
- **Window Length:** 128 sensor readings ($128 \times \frac{1}{50\text{ Hz}} = 2.56\text{ seconds}$). A $2.56$-second window captures at least two complete human gait cycles (typical human stride is $1.0 - 1.2\text{ seconds}$).
- **Overlap:** 50% overlap (64 readings = $1.28\text{ seconds}$), ensuring temporal continuity and preventing boundary loss.

### 5.2 Mathematical Features Extracted
From each tri-axial time window, comprehensive time and frequency domain features are computed:

1. **Statistical Time-Domain Descriptors:**
   - Mean: $\mu = \frac{1}{N}\sum_{i=1}^N x_i$
   - Standard Deviation: $\sigma = \sqrt{\frac{1}{N}\sum_{i=1}^N (x_i - \mu)^2}$
   - Median Absolute Deviation (MAD): $\text{MAD} = \text{median}(|x_i - \text{median}(x)|)$
   - Minimum and Maximum values
   - Interquartile Range (IQR): $Q_3 - Q_1$
   - Signal Energy: $E = \frac{1}{N}\sum_{i=1}^N x_i^2$
   - Signal Entropy (Shannon distribution entropy)

2. **Signal Magnitude Area (SMA):**
   Reflects the cumulative energy expenditure of 3-axis motion:
   $$\text{SMA} = \frac{1}{N} \sum_{i=1}^N \left(|a_x(i)| + |a_y(i)| + |a_z(i)|\right)$$

3. **Euclidean Vector Magnitude (Norm):**
   $$\text{VM}(t) = \sqrt{a_x(t)^2 + a_y(t)^2 + a_z(t)^2}$$
   Provides orientation-invariant intensity of overall movement.

4. **Cross-Axis Correlation:**
   $$\rho_{x,y} = \frac{\sum (x_i - \mu_x)(y_i - \mu_y)}{(N-1)\sigma_x \sigma_y}$$

5. **Frequency-Domain (FFT) Descriptors:**
   Applying Fast Fourier Transform (FFT) generates spectral representations:
   - Dominant / Peak Frequency
   - Spectral Energy
   - Spectral Skewness and Kurtosis

---

## 6. Machine Learning Model Benchmark

Four diverse classification algorithms were implemented and trained:
1. **Multinomial Logistic Regression:** Convex linear classifier with $L_2$ regularization.
2. **Support Vector Machine (SVM):** Non-linear kernel-based classifier with Radial Basis Function (RBF) kernel ($C=10.0, \gamma=\text{'scale'}$).
3. **Random Forest Classifier:** Bagged ensemble of 100 decorrelated decision trees (max depth = 20).
4. **Decision Tree Classifier (CART):** Single decision tree with max depth = 12.

### 6.1 Unseen Test Set Performance Comparison
All models were evaluated on the independent test split of 2,947 instances from unseen human subjects:

| Algorithm | Test Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) | F1-Score (Weighted) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Logistic Regression** | **95.59%** | **0.9588** | **0.9546** | **0.9558** | **0.9558** |
| **Support Vector Machine (RBF)** | **95.49%** | **0.9559** | **0.9531** | **0.9540** | **0.9547** |
| **Random Forest (100 Trees)** | **92.77%** | **0.9293** | **0.9249** | **0.9262** | **0.9276** |
| **Decision Tree (CART)** | **86.43%** | **0.8640** | **0.8607** | **0.8611** | **0.8634** |

---

## 7. In-Depth Confusion Matrix & Class-Wise Analysis

### 7.1 Class-Wise Performance Metrics (Best Model)
| Activity Class | Precision | Recall | F1-Score | Support (Windows) |
|:---|:---:|:---:|:---:|:---:|
| **WALKING** | 94.26% | 99.40% | 96.76% | 496 |
| **WALKING_UPSTAIRS** | 96.12% | 94.69% | 95.40% | 471 |
| **WALKING_DOWNSTAIRS** | 99.00% | 94.29% | 96.59% | 420 |
| **SITTING** | 97.07% | 87.78% | 92.19% | 491 |
| **STANDING** | 89.02% | 97.56% | 93.09% | 532 |
| **LAYING** | 99.81% | 99.07% | 99.44% | 537 |
| **Overall / Average** | **95.59%** | **95.46%** | **95.58%** | **2,947** |

### 7.2 Biomechanical Confusion Analysis
1. **Separation of Dynamic vs Static Activities:**
   The classifier achieves virtually **100% boundary separation** between dynamic states (`WALKING`, `UPSTAIRS`, `DOWNSTAIRS`) and static states (`SITTING`, `STANDING`, `LAYING`). The total acceleration vector magnitude variance provides an impenetrable boundary.

2. **The Sitting vs. Standing Challenge:**
   - 57 instances of `SITTING` were predicted as `STANDING`.
   - 13 instances of `STANDING` were predicted as `SITTING`.
   - **Underlying Cause:** Both states feature near-zero dynamic body acceleration ($|\mathbf{a}_{\text{body}}| \approx 0$) and identical gravitational magnitude ($|\mathbf{g}| \approx 1.0g$). When a waist-mounted phone sits in a pocket or belt slightly tilted, the gravity angle on the Y-axis closely mirrors a standing stance, causing occasional boundary confusion.

3. **Stair Climbing Distinctions:**
   `WALKING_UPSTAIRS` and `WALKING_DOWNSTAIRS` are accurately distinguished ($>94\%$ recall) due to distinct angular velocity sign reversals and vertical jerk patterns.

---

## 8. Interactive Dashboard & System Demonstration

An interactive **Streamlit Application (`app.py`)** was developed to bridge machine learning modeling and end-user deployment.

### 8.1 Dashboard Features
1. **Overview & Architecture:** Explains the sensor physics, Butterworth filter pipeline, and activity taxonomy.
2. **EDA Studio:** Interactive visualization of 3-axis accelerometer and gyroscope waveforms, dynamic vs static magnitude comparison, 2D PCA cluster space, and feature correlation heatmaps.
3. **Model Benchmark & Metrics:** Interactive model performance selector, confusion matrix heatmaps (raw counts & normalized recall %), and Gini feature importance bar plots.
4. **Activity Predictor:** Allows selecting real unseen test windows or uploading custom CSV sensor logs. Features confidence probability bars and visual status badges.
5. **Real-Time Sensor Stream Simulator:** Simulates a live 50 Hz phone sensor stream with animated multi-axis waveform charts and instant activity recognition.

---

## 9. Limitations & Practical Considerations

1. **Sensor Placement Sensitivity:**
   The training dataset assumes a waist-mounted smartphone. When the smartphone is held in a hand or placed in a jacket pocket, orientation angles shift, requiring rotational invariance calibration.
2. **Sampling Rate Variations:**
   Different commercial wearables log at rates between 20 Hz and 100 Hz. Resampling using cubic spline interpolation is essential in wild deployment.
3. **Energy Efficiency on Edge Devices:**
   Running continuous FFT and 561-feature calculation drains mobile batteries. The project demonstrates that a subset of top 15 features (e.g., SMA, standard deviation, energy) retains $>92\%$ accuracy while reducing computation by $>85\%$.

---

## 10. Conclusion & Future Roadmap
This capstone project successfully demonstrates that smartphone inertial sensors paired with signal filtering and feature engineering achieve **95.59% accuracy** in real-time human activity recognition.

### Future Work
- **Deep Learning Pipelines:** Training 1D Convolutional Neural Networks (1D-CNN) and Bidirectional LSTMs directly on raw inertial time-series windows without manual feature extraction.
- **Edge Deployment:** Exporting the trained pipeline into **TensorFlow Lite (TFLite)** and **ONNX Runtime** for local on-device inference on Android and WearOS smartwatches.
- **Semi-Supervised Active Learning:** Adapting to personalized gait patterns of individual users over time.

---

## 11. Project Directory Structure
```
d:/Project/
├── app.py                             # Interactive Streamlit Web Dashboard
├── requirements.txt                   # Dependency specifications
├── README.md                          # GitHub documentation and quickstart
├── .gitignore                         # Git exclusion rules
├── data/
│   ├── raw/UCI HAR Dataset/           # Raw sensor dataset files
│   └── processed/                     # Pre-split CSVs and streaming samples
├── src/
│   ├── __init__.py                    # Package initializer
│   ├── download_dataset.py            # Automated UCI HAR downloader
│   ├── data_loader.py                 # Data parser and signal loader
│   ├── preprocessing.py               # Butterworth filter & scaler
│   ├── feature_extraction.py          # Sliding window & FFT feature extractor
│   ├── eda.py                         # Exploratory data analysis suite
│   ├── train.py                       # Model training and hyperparameter tuning
│   ├── evaluate.py                    # Standalone model evaluation & diagnostics
│   └── generate_notebook.py           # Colab notebook generator
├── models/
│   ├── best_har_model.pkl             # Serialized best classifier
│   ├── scaler.pkl                     # Serialized StandardScaler
│   ├── model_metadata.json            # Model parameters and label maps
│   ├── logistic_regression.pkl        # Logistic Regression model
│   ├── support_vector_machine_svm.pkl # Support Vector Machine model
│   ├── random_forest.pkl              # Random Forest model
│   └── decision_tree.pkl              # Decision Tree model
├── notebooks/
│   └── HAR_Model_Training_and_EDA.ipynb  # Interactive Jupyter/Colab notebook
└── reports/
    ├── HAR_Project_Report.md          # Comprehensive formal capstone report
    ├── model_comparison_metrics.csv   # Performance metrics table
    ├── best_model_classification_report.csv # Per-class classification report
    └── figures/                       # Publication-grade figures
        ├── activity_distribution.png
        ├── raw_signals_by_activity.png
        ├── signal_magnitude_comparison.png
        ├── pca_activity_separation.png
        ├── correlation_heatmap.png
        ├── confusion_matrix.png
        └── feature_importance.png
```

---
*Report prepared for ECE Capstone Assessment.*
