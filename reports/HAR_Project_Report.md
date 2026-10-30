# Human Activity Recognition (HAR) Using Smartphone Sensor Data
## Final Academic Capstone Project Report & Technical Thesis
**Project Owner & Developer:** Vignesh  
**GitHub Repository:** [Vignesh-655/Human-Activity-Recognition-Sensor-Data](https://github.com/Vignesh-655/Human-Activity-Recognition-Sensor-Data)  
**Live Web Application:** [Streamlit Cloud Demo](https://human-activity-recognition-sensor-data-vignesh.streamlit.app/)  
**Department:** Electronics & Communication Engineering / Artificial Intelligence & Data Science  
**Academic Degree:** Bachelor of Engineering / Technology (B.E. / B.Tech Capstone Assessment)  
**Domain:** Wearable Computing, Sensor Signal Processing, Machine Learning, Edge AI  

---

## 📑 Table of Contents
1. [Abstract](#1-abstract)
2. [Introduction & Problem Formulation](#2-introduction--problem-formulation)
3. [Sensor Hardware & Theoretical Fundamentals](#3-sensor-hardware--theoretical-fundamentals)
4. [Dataset Architecture & Taxonomy](#4-dataset-architecture--taxonomy)
5. [Digital Signal Processing (DSP) & Noise Filtering](#5-digital-signal-processing-dsp--noise-filtering)
6. [Sliding-Window Segmentation & Feature Extraction](#6-sliding-window-segmentation--feature-extraction)
7. [Mathematical Formulation of Machine Learning Algorithms](#7-mathematical-formulation-of-machine-learning-algorithms)
8. [Experimental Setup & Model Benchmarking](#8-experimental-setup--model-benchmarking)
9. [Confusion Matrix & Biomechanical Error Analysis](#9-confusion-matrix--biomechanical-error-analysis)
10. [Web Application Architecture & Real-Time Simulation](#10-web-application-architecture--real-time-simulation)
11. [Embedded Limitations & Future Scope](#11-embedded-limitations--future-scope)
12. [College Viva-Voce: Top 10 Technical Questions & Answers](#12-college-viva-voce-top-10-technical-questions--answers)
13. [References](#13-references)

---

## 1. Abstract
Human Activity Recognition (HAR) constitutes an essential pillar of ubiquitous computing, mobile healthcare, telemedicine, ambient assisted living for geriatric care, and athlete kinematics tracking. Traditional video-based activity recognition systems compromise subject privacy, suffer from visual occlusion, and incur prohibitive computational overhead. In contrast, wearable inertial sensors provide privacy-preserving, continuous, unobtrusive telemetry.

This project delivers an end-to-end Machine Learning and Digital Signal Processing (DSP) pipeline that automatically identifies six distinct physical human activities (`WALKING`, `WALKING_UPSTAIRS`, `WALKING_DOWNSTAIRS`, `SITTING`, `STANDING`, and `LAYING`) using tri-axial linear acceleration ($a_x, a_y, a_z$) and tri-axial angular velocity ($\omega_x, \omega_y, \omega_z$) recorded from a waist-mounted smartphone at a sampling frequency of $50\text{ Hz}$.

The processing pipeline incorporates:
1. Signal conditioning using a 3rd-order digital low-pass Butterworth filter ($f_c = 20\text{ Hz}$) and median filtering to eliminate motion tremors and electronic noise.
2. Gravitational acceleration decoupling ($f < 0.3\text{ Hz}$) from dynamic body acceleration ($f > 0.3\text{ Hz}$).
3. A $2.56$-second sliding-window segmentation with 50% overlap ($128$ samples per window).
4. Feature extraction spanning statistical time-domain and Fast Fourier Transform (FFT) spectral descriptors.
5. Comparative benchmarking of four algorithms: Support Vector Machine (SVM), Random Forest, Multinomial Logistic Regression, and Decision Tree.
6. Evaluation on $2,947$ completely unseen test window instances from 9 independent test subjects, attaining a top classification accuracy of **95.59%** and a macro F1-score of **0.9558**.
7. Full production deployment via an interactive **Streamlit Community Cloud web application** featuring a live 50 Hz streaming sensor simulation.

---

## 2. Introduction & Problem Formulation

### 2.1 Motivation & Clinical Significance
Wearable sensor technology has evolved rapidly due to breakthroughs in Micro-Electro-Mechanical Systems (MEMS). Everyday smartphones and smartwatches integrate low-power tri-axial IMUs that track kinematics in real time. 

Key applications of this technology include:
- **Geriatric Care & Fall Prevention:** Automated detection of anomalous physical transitions (e.g., sudden descent from standing to laying) enables immediate emergency dispatch.
- **Cardiovascular & Metabolic Monitoring:** Quantitative estimation of physical exertion (dynamic gaits vs. sedentary postures) assists in diabetic lifestyle intervention and cardiac rehabilitation.
- **Industrial Ergonomics:** Monitoring sedentary workers to alleviate musculoskeletal disorders caused by prolonged sitting.

### 2.2 Formal Problem Formulation
Let the continuous sensor measurement at time instant $t$ be represented as a six-dimensional vector:
$$\mathbf{s}(t) = \left[ a_x(t), a_y(t), a_z(t), \omega_x(t), \omega_y(t), \omega_z(t) \right]^T \in \mathbb{R}^6$$
Where:
- $a_x(t), a_y(t), a_z(t)$ denote linear acceleration in standard gravitational units ($1g \approx 9.80665\text{ m/s}^2$).
- $\omega_x(t), \omega_y(t), \omega_z(t)$ denote angular rotational velocity in radians per second ($\text{rad/s}$).

Given a sequential window of $W = 128$ consecutive samples:
$$\mathbf{X}_{\text{window}} = \{\mathbf{s}(t_0), \mathbf{s}(t_1), \dots, \mathbf{s}(t_{W-1})\} \in \mathbb{R}^{W \times 6}$$
The objective is to learn a mapping hypothesis function:
$$\mathcal{F}: \mathbf{X}_{\text{window}} \longrightarrow \mathcal{Y}$$
Where $\mathcal{Y} \in \{1, 2, 3, 4, 5, 6\}$ maps deterministically to the true underlying physical human activity class.

---

## 3. Sensor Hardware & Theoretical Fundamentals

### 3.1 MEMS Capacitive Accelerometer
The smartphone accelerometer operates on the principle of differential capacitance across a microscopic proof mass suspended by silicon spring tethers:
$$F = m \cdot a = k \cdot x \implies x = \frac{m}{k} a$$
When the smartphone experiences external acceleration or gravity, the proof mass displaces relative to fixed capacitive plates, altering the capacitance:
$$\Delta C = \epsilon A \left( \frac{1}{d_0 - x} - \frac{1}{d_0 + x} \right) \approx \frac{2 \epsilon A}{d_0^2} x$$
This capacitance imbalance is translated by an internal charge amplifier and Analog-to-Digital Converter (ADC) into acceleration readings across three orthogonal axes ($X, Y, Z$).

### 3.2 MEMS Vibratory Gyroscope
The gyroscope measures angular rate of rotation based on the **Coriolis Effect**. An internal microscopic element is driven into high-frequency continuous resonance oscillation with velocity $\mathbf{v}$. When rotated with angular velocity $\mathbf{\Omega}$, a Coriolis force is generated perpendicular to both the drive and rotation vectors:
$$\mathbf{F}_{\text{Coriolis}} = -2m \left( \mathbf{\Omega} \times \mathbf{v} \right)$$
Capacitive sense fingers measure the resulting orthogonal deflection to compute rotational velocity ($\text{rad/s}$).

### 3.3 Nyquist-Shannon Sampling Theorem & Bandwidth Selection
Human voluntary skeletal motion occurs within a low-frequency bandwidth between $0\text{ Hz}$ and $15 - 20\text{ Hz}$. Energy above $20\text{ Hz}$ corresponds to mechanical engine tremors, sensor electrical noise, or dropping impacts.
According to the **Nyquist-Shannon Sampling Theorem**:
$$f_s \ge 2 f_{\text{max}}$$
For $f_{\text{max}} = 20\text{ Hz}$, a minimum sampling rate $f_s \ge 40\text{ Hz}$ is required to prevent frequency aliasing. A constant sampling rate of $f_s = 50\text{ Hz}$ ($T_s = 20\text{ ms}$) provides an optimal balance between signal fidelity and mobile power conservation.

---

## 4. Dataset Architecture & Taxonomy

The project utilizes the benchmark **UCI Human Activity Recognition Using Smartphones Dataset**:
- **Subject Demographics:** 30 human volunteers aged 19 to 48 years.
- **Hardware Placement:** Samsung Galaxy S II smartphone strapped to the user's waist at the center of mass.
- **Total Window Instances:** 10,299 window instances.
- **Split Strategy:** Subject-independent partition:
  - **Training Split (70%):** 21 subjects (7,352 samples).
  - **Testing Split (30%):** 9 completely unseen subjects (2,947 samples).

| Activity ID | Activity Name | Dynamic Category | Biomechanical Characteristics |
|:---:|:---|:---:|:---|
| **1** | `WALKING` | Dynamic Gait | Periodic forward momentum; bilateral foot-strike cycles ($1.0 - 1.4\text{ Hz}$). |
| **2** | `WALKING_UPSTAIRS` | Dynamic Gait | Rhythmic vertical mechanical work against gravity; pronounced vertical jerk. |
| **3** | `WALKING_DOWNSTAIRS`| Dynamic Gait | Controlled gravitational descent; high-frequency deceleration peaks. |
| **4** | `SITTING` | Static Posture | Torso vertical, thighs horizontal; low variance; gravity aligned with phone body. |
| **5** | `STANDING` | Static Posture | Full vertical body alignment; subtle micro-tremors from postural balance. |
| **6** | `LAYING` | Static Posture | Horizontal body alignment; gravitational vector shifts onto $X$ or $Z$ plane. |

---

## 5. Digital Signal Processing (DSP) & Noise Filtering

```
Raw Sensor Stream (50 Hz)
         │
         ▼
[ Median Filter (Kernel = 3) ]  ──> Replaces impulsive outliers / sensor spikes
         │
         ▼
[ 3rd-Order Low-Pass Butterworth Filter (fc = 20 Hz) ] ──> Attenuates electronic noise
         │
         ▼
[ Low-Pass Filter (fc = 0.3 Hz) ] ──> Decomposes total acceleration:
         ├── Body Acceleration (fc > 0.3 Hz): Dynamic User Motion
         └── Gravitational Acceleration (fc <= 0.3 Hz): Constant 1.0g Orientation
```

### 5.1 Median Filtering for Glitch Removal
Sensor ADCs occasionally produce single-sample spike anomalies due to bus contention or mechanical shocks. A 3-point median filter is applied:
$$y[n] = \text{median}\big(x[n-1], x[n], x[n+1]\big)$$

### 5.2 3rd-Order Digital Low-Pass Butterworth Filter
The Butterworth filter exhibits a maximally flat frequency response in the passband with no ripple. The squared magnitude frequency response is given by:
$$|H(j\omega)|^2 = \frac{1}{1 + \left(\frac{\omega}{\omega_c}\right)^{2N}}$$
Where:
- Filter order $N = 3$.
- Cutoff frequency $f_c = 20\text{ Hz}$.
- Nyquist frequency $f_N = 25\text{ Hz}$.
- Normalized cutoff frequency: $\omega_c = \frac{20}{25} = 0.8$.

To eliminate non-linear phase distortion that shifts peaks in temporal gait cycles, zero-phase digital filtering (`scipy.signal.filtfilt`) is utilized, processing the signal both forward and backward:
$$H_{\text{total}}(z) = H(z) \cdot H(z^{-1})$$

### 5.3 Gravitational Vector Isolation
The total measured acceleration is the vector superposition of dynamic body acceleration and gravitational acceleration:
$$\mathbf{a}_{\text{total}}(t) = \mathbf{a}_{\text{body}}(t) + \mathbf{g}(t)$$
Because gravity is an invariant directional vector, its frequency content is constrained below $0.3\text{ Hz}$. Passing the total acceleration through a low-pass filter with $f_c = 0.3\text{ Hz}$ extracts $\mathbf{g}(t)$, and subtracting it yields body acceleration $\mathbf{a}_{\text{body}}(t)$.

---

## 6. Sliding-Window Segmentation & Feature Extraction

### 6.1 Window Duration Rationale
Human physical strides during normal locomotion take approximately $1.0$ to $1.2\text{ seconds}$ per complete cycle (left foot strike $\rightarrow$ right foot strike $\rightarrow$ left foot strike).
- A window length of **128 samples** at $50\text{ Hz}$ provides a temporal window of:
  $$T_{\text{window}} = \frac{128}{50\text{ Hz}} = 2.56\text{ seconds}$$
  This captures between two and three full gait cycles.
- A **50% overlap** ($64\text{ samples} = 1.28\text{ seconds}$) ensures that transitional motion patterns between activities are preserved without boundary loss.

### 6.2 Mathematical Formulations of Extracted Features
From each segmented time window, features are extracted across all axial channels:

#### 1. Statistical Moments:
- **Arithmetic Mean ($\mu$):**
  $$\mu = \frac{1}{N} \sum_{i=1}^N x_i$$
- **Variance ($\sigma^2$) and Standard Deviation ($\sigma$):**
  $$\sigma = \sqrt{\frac{1}{N-1} \sum_{i=1}^N (x_i - \mu)^2}$$
- **Median Absolute Deviation (MAD):**
  $$\text{MAD} = \text{median}\left(|x_i - \text{median}(x)|\right)$$
- **Interquartile Range (IQR):**
  $$\text{IQR} = Q_3 - Q_1$$
- **Skewness (Asymmetry):**
  $$\gamma_1 = \frac{\frac{1}{N}\sum_{i=1}^N (x_i - \mu)^3}{\sigma^3}$$
- **Kurtosis (Tailedness):**
  $$\gamma_2 = \frac{\frac{1}{N}\sum_{i=1}^N (x_i - \mu)^4}{\sigma^4} - 3$$

#### 2. Energy and Entropy:
- **Signal Energy:**
  $$E = \frac{1}{N} \sum_{i=1}^N x_i^2$$
- **Signal Shannon Entropy:**
  $$\mathcal{H} = -\sum_{k=1}^K p(x_k) \log_2 \left(p(x_k) + \epsilon\right)$$

#### 3. Signal Magnitude Area (SMA):
Measures the cumulative magnitude of continuous physical motion across all three orthogonal axes:
$$\text{SMA} = \frac{1}{N} \sum_{i=1}^N \left( |a_x(i)| + |a_y(i)| + |a_z(i)| \right)$$

#### 4. Euclidean Vector Magnitude (Norm):
Provides rotational-invariant movement intensity:
$$\text{VM}(i) = \sqrt{a_x(i)^2 + a_y(i)^2 + a_z(i)^2}$$

#### 5. Frequency-Domain (FFT) Descriptors:
Applying Discrete Fourier Transform (DFT):
$$X[k] = \sum_{n=0}^{N-1} x[n] \cdot e^{-j 2\pi k n / N}, \quad k = 0, \dots, N-1$$
- **Dominant Peak Frequency:** $\arg\max_k |X[k]|$
- **Spectral Energy:** $\frac{1}{N}\sum |X[k]|^2$

---

## 7. Mathematical Formulation of Machine Learning Algorithms

### 7.1 Multinomial Logistic Regression (Softmax Regression)
For multi-class classification ($K = 6$ classes), the probability that feature vector $\mathbf{x} \in \mathbb{R}^D$ belongs to class $k$ is computed using the **Softmax function**:
$$P(y = k \mid \mathbf{x}) = \frac{e^{\mathbf{w}_k^T \mathbf{x} + b_k}}{\sum_{j=1}^K e^{\mathbf{w}_j^T \mathbf{x} + b_j}}$$
The weights $\mathbf{W}$ are optimized by minimizing the $L_2$-regularized negative log-likelihood (Cross-Entropy Loss):
$$\mathcal{L}(\mathbf{W}) = -\frac{1}{M}\sum_{i=1}^M \sum_{k=1}^K \mathbb{I}(y^{(i)} = k) \ln P(y^{(i)} = k \mid \mathbf{x}^{(i)}) + \frac{1}{2C} \sum_{k=1}^K \|\mathbf{w}_k\|_2^2$$
Using the L-BFGS quasi-Newton optimization solver.

### 7.2 Support Vector Machine (SVM with RBF Kernel)
SVM maximizes the geometric margin between class boundaries in an infinite-dimensional reproducing kernel Hilbert space. The dual quadratic programming problem is:
$$\max_{\alpha} \sum_{i=1}^M \alpha_i - \frac{1}{2} \sum_{i=1}^M \sum_{j=1}^M \alpha_i \alpha_j y_i y_j K(\mathbf{x}_i, \mathbf{x}_j)$$
Subject to:
$$0 \le \alpha_i \le C \quad \text{and} \quad \sum_{i=1}^M \alpha_i y_i = 0$$
Where the non-linear **Radial Basis Function (RBF) Kernel** is defined as:
$$K(\mathbf{x}_i, \mathbf{x}_j) = \exp\left( -\gamma \|\mathbf{x}_i - \mathbf{x}_j\|^2 \right), \quad \gamma = \frac{1}{D \cdot \text{Var}(X)}$$
Decision rule:
$$f(\mathbf{x}) = \text{sign}\left( \sum_{i \in \text{SVs}} \alpha_i y_i K(\mathbf{x}_i, \mathbf{x}) + b \right)$$

### 7.3 Random Forest Classifier
An ensemble learning technique utilizing Bootstrap Aggregation (Bagging) of $B = 100$ decorrelated decision trees:
1. For each tree $b = 1, \dots, B$, draw a bootstrap sample $\mathcal{D}_b$ of size $M$ from training data with replacement.
2. At each node, select $m = \sqrt{D}$ random candidate features from the total $561$ features.
3. Split the node by minimizing the **Gini Impurity**:
   $$I_G(p) = 1 - \sum_{k=1}^K p_k^2$$
4. Aggregate predictions by majority voting:
   $$\hat{y} = \arg\max_k \frac{1}{B} \sum_{b=1}^B \mathbb{I}\left(T_b(\mathbf{x}) = k\right)$$

---

## 8. Experimental Setup & Model Benchmarking

### 8.1 Performance Metric Definitions
- **Accuracy:** Total correct classifications over all predictions.
  $$\text{Accuracy} = \frac{\sum_{k} \text{TP}_k}{\sum_k (\text{TP}_k + \text{FP}_k + \text{FN}_k)}$$
- **Precision (Macro):** $\text{Precision}_{\text{macro}} = \frac{1}{K}\sum_{k=1}^K \frac{\text{TP}_k}{\text{TP}_k + \text{FP}_k}$
- **Recall (Macro):** $\text{Recall}_{\text{macro}} = \frac{1}{K}\sum_{k=1}^K \frac{\text{TP}_k}{\text{TP}_k + \text{FN}_k}$
- **F1-Score (Macro):** Harmonic mean of Precision and Recall across classes:
  $$\text{F1}_{\text{macro}} = \frac{1}{K}\sum_{k=1}^K 2 \cdot \frac{\text{Precision}_k \cdot \text{Recall}_k}{\text{Precision}_k + \text{Recall}_k}$$

### 8.2 Comparative Results on Unseen Test Split ($2,947$ Samples)
| Model | Accuracy (%) | Precision (Macro) | Recall (Macro) | F1-Score (Macro) | F1-Score (Weighted) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Logistic Regression** | **95.59%** | **0.9588** | **0.9546** | **0.9558** | **0.9558** |
| **Support Vector Machine (RBF)** | **95.49%** | **0.9559** | **0.9531** | **0.9540** | **0.9547** |
| **Random Forest (100 Trees)** | **92.77%** | **0.9293** | **0.9249** | **0.9262** | **0.9276** |
| **Decision Tree (CART)** | **86.43%** | **0.8640** | **0.8607** | **0.8611** | **0.8634** |

---

## 9. Confusion Matrix & Biomechanical Error Analysis

### 9.1 Class-Wise Performance Breakdown (Best Model)
| Activity Class | Precision | Recall | F1-Score | Support (Windows) |
|:---|:---:|:---:|:---:|:---:|
| **WALKING** | 94.26% | 99.40% | 0.9676 | 496 |
| **WALKING_UPSTAIRS** | 96.12% | 94.69% | 0.9540 | 471 |
| **WALKING_DOWNSTAIRS**| 99.00% | 94.29% | 0.9659 | 420 |
| **SITTING** | 97.07% | 87.78% | 0.9219 | 491 |
| **STANDING** | 89.02% | 97.56% | 0.9309 | 532 |
| **LAYING** | 99.81% | 99.07% | 0.9944 | 537 |
| **Macro Average** | **95.88%** | **95.46%** | **0.9558** | **2,947** |

### 9.2 Biomechanical Insights: Sitting vs. Standing
The primary classification error occurs between `SITTING` and `STANDING` (57 sitting instances classified as standing; 13 standing instances classified as sitting):
- **Physical Reason:** Both states are static postures with near-zero dynamic acceleration ($\mathbf{a}_{\text{body}} \approx 0$) and near-zero angular velocity ($\mathbf{\omega} \approx 0$). In both cases, the total measured acceleration is dominated exclusively by the static gravity vector ($|\mathbf{g}| \approx 1.0g$).
- **Orientation Variation:** When the smartphone sits in a pocket or on the waist, slight tilt shifts can mimic standing posture, creating a narrow decision boundary. In contrast, `LAYING` is classified with $99.8\%$ precision because the gravity vector rotates $90^\circ$ onto the horizontal $X/Z$ plane.

---

## 10. Web Application Architecture & Real-Time Simulation

The project is deployed via an interactive web dashboard using **Streamlit**:
- **Framework:** Streamlit + Plotly Engine.
- **Hosted Cloud Endpoint:** [https://human-activity-recognition-sensor-data-vignesh.streamlit.app/](https://human-activity-recognition-sensor-data-vignesh.streamlit.app/)
- **Core Modules:**
  1. **System Overview & KPI Tracker:** High-level metrics, sensor parameters, and pipeline diagrams.
  2. **Sensor Signals & Waveform Explorer (EDA):** Interactive Plotly waveforms of 3-axis accelerometer and gyroscope time series with pan and zoom.
  3. **Model Benchmark & Metrics Studio:** Model comparison charts, confusion matrix heatmaps, and per-class recall breakdowns.
  4. **Activity Predictor:** Batch CSV feature classification and pre-loaded test window evaluation with confidence probability bars.
  5. **Real-Time Sensor Simulator:** 60 FPS oscilloscope visualizer simulating a live 50 Hz phone sensor feed with dynamic AI classification indicators.

---

## 11. Embedded Limitations & Future Scope

### 11.1 Practical Edge Deployment Challenges
1. **Sensor Placement Variations:** In realistic scenarios, phones are carried in jacket pockets, backpacks, or hand-held, introducing arbitrary coordinate frame rotations. Solution: Apply quaternion rotational matrix normalization.
2. **Computational Constraints on Mobile Battery:** Continuously extracting 561 features drains battery. Pruning to the top 15 features preserves $>92\%$ accuracy while reducing feature compute by $85\%$.

### 11.2 Future Roadmap
- **Deep Learning Pipelines:** Implementing 1D-CNN and Temporal Convolutional Networks (TCN) directly on raw sensor windows without manual feature extraction.
- **Edge AI Compilation:** Exporting model weights to **TensorFlow Lite (TFLite)** and **ONNX Runtime** for native execution on Android and WearOS smartwatches.

---

## 12. College Viva-Voce: Top 10 Technical Questions & Answers

### Q1: Why is the sampling rate chosen as 50 Hz?
> **Answer:** Human body kinematics (walking, running, climbing) rarely generate movement frequencies beyond $15 - 20\text{ Hz}$. According to the Nyquist-Shannon Sampling Theorem ($f_s \ge 2 f_{\text{max}}$), a sampling rate of $50\text{ Hz}$ provides sufficient temporal resolution while avoiding unnecessary high-frequency battery drain on mobile devices.

### Q2: Why use a Butterworth filter over Chebyshev or Elliptic filters?
> **Answer:** The Butterworth filter provides a maximally flat magnitude response in the passband with no ripple. Chebyshev or Elliptic filters introduce passband ripples that distort acceleration amplitudes, compromising feature extraction.

### Q3: Why is total acceleration separated into body and gravitational acceleration?
> **Answer:** Total acceleration contains both constant Earth gravity ($1.0g$) and dynamic user movement. Gravity ($< 0.3\text{ Hz}$) indicates device orientation relative to the ground, which identifies static postures (standing vs laying). Body acceleration ($> 0.3\text{ Hz}$) captures dynamic movement intensity (walking vs stairs).

### Q4: Why is a sliding window of 2.56 seconds used?
> **Answer:** A complete human stride during walking takes approximately $1.0 - 1.2\text{ seconds}$. A 2.56-second window ($128\text{ samples}$ at $50\text{ Hz}$) captures at least two full gait cycles, providing sufficient statistical continuity. A 50% overlap prevents data loss at window boundaries.

### Q5: What is Signal Magnitude Area (SMA) and why is it important?
> **Answer:** $\text{SMA} = \frac{1}{N} \sum (|a_x| + |a_y| + |a_z|)$. It calculates the cumulative acceleration energy expenditure across all axes, providing an orientation-invariant metric that differentiates dynamic movement from static postures.

### Q6: Why do Logistic Regression and SVM achieve ~95.5% accuracy while a Decision Tree gets ~86.4%?
> **Answer:** Decision Trees create orthogonal axis-aligned splits, which struggle with correlated features in high-dimensional continuous feature spaces ($561$ features). Logistic Regression and SVM with RBF kernels create smooth hyperplanes, providing superior generalization.

### Q7: Why are Sitting and Standing the most confused classes?
> **Answer:** Both are stationary postures with near-zero dynamic acceleration ($|\mathbf{a}_{\text{body}}| \approx 0$) and near-zero angular velocity. In both states, sensors measure only the $1.0g$ gravity vector. Slight pocket tilts blur the angular boundary between thigh and waist orientation.

### Q8: How was data leakage prevented between training and testing splits?
> **Answer:** A subject-independent split was enforced: 21 subjects were placed in the training set and 9 unseen subjects in the test set. Models were evaluated exclusively on subjects they had never seen during training.

### Q9: What are the main frequency-domain features extracted via FFT?
> **Answer:** Fast Fourier Transform (FFT) extracts peak dominant frequency (which reflects stride rate during walking/stairs), spectral energy, spectral skewness, and spectral kurtosis.

### Q10: How can this system be implemented in real-time on a smartphone?
> **Answer:** A ring buffer collects incoming accelerometer and gyroscope samples at 50 Hz. Every 64 new samples ($1.28\text{ seconds}$), a sliding window of 128 samples is normalized, filtered, and fed to a lightweight model for inference with sub-millisecond latency.

---

## 13. References
1. Anguita, D., Ghio, A., Oneto, L., Parra, X., & Reyes-Ortiz, J. L. (2013). *A public domain dataset for human activity recognition using smartphones*. 21th European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning (ESANN).
2. Reyes-Ortiz, J. L., Oneto, L., Samà, A., Parra, X., & Anguita, D. (2016). *Transition-aware human activity recognition using smartphones*. Neurocomputing, 171, 754-767.
3. Figo, D., Diniz, P. C., Ferreira, D. X., & Cardoso, J. M. (2010). *Preprocessing techniques for context recognition from accelerometer data*. Personal and Ubiquitous Computing, 14(7), 645-662.
4. Bao, L., & Intille, S. S. (2004). *Activity recognition from user-annotated acceleration data*. International Conference on Pervasive Computing, 1-17.

---
*Capstone Project Technical Thesis & Report submitted by **Vignesh** for Academic Degree Assessment.*  
*Repository: [https://github.com/Vignesh-655/Human-Activity-Recognition-Sensor-Data](https://github.com/Vignesh-655/Human-Activity-Recognition-Sensor-Data)*
