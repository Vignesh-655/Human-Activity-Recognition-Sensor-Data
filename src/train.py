"""
Model Training and Comparative Evaluation Module for Human Activity Recognition.
Trains, evaluates, and compares Random Forest, Support Vector Machine (SVM),
Decision Tree, and Logistic Regression on unseen test sensor data.
"""

import os
import sys
import json
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.preprocessing import StandardScaler

from src.data_loader import load_har_data, ACTIVITY_MAP

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

def train_and_evaluate_models():
    print("Loading UCI HAR dataset for model training...")
    X_train, y_train, sub_train, X_test, y_test, sub_test, feature_names = load_har_data()
    
    print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    
    # Scale features
    print("Fitting StandardScaler on training data...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save fitted scaler
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    print(f"Saved scaler to {MODELS_DIR / 'scaler.pkl'}")
    
    # Define models dictionary
    models = {
        "Support Vector Machine (SVM)": SVC(C=10.0, kernel='rbf', probability=True, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1),
        "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=12, random_state=42)
    }
    
    results = []
    trained_models = {}
    activity_labels = [ACTIVITY_MAP[i] for i in range(1, 7)]
    
    for name, model in models.items():
        print(f"\n==========================================")
        print(f"Training {name}...")
        
        # Fit model
        if name in ["Support Vector Machine (SVM)", "Logistic Regression"]:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            # Tree-based can work directly on unscaled or scaled; using unscaled for tree interpretability
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
        acc = accuracy_score(y_test, y_pred)
        prec_macro = precision_score(y_test, y_pred, average="macro")
        rec_macro = recall_score(y_test, y_pred, average="macro")
        f1_macro = f1_score(y_test, y_pred, average="macro")
        
        prec_wt = precision_score(y_test, y_pred, average="weighted")
        rec_wt = recall_score(y_test, y_pred, average="weighted")
        f1_wt = f1_score(y_test, y_pred, average="weighted")
        
        print(f"{name} Results on Unseen Test Set:")
        print(f"  Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
        print(f"  Precision (Macro): {prec_macro:.4f}")
        print(f"  Recall (Macro):    {rec_macro:.4f}")
        print(f"  F1-Score (Macro):  {f1_macro:.4f}")
        print(f"  F1-Score (Weighted): {f1_wt:.4f}")
        
        # Save individual model
        clean_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        model_file = MODELS_DIR / f"{clean_name}.pkl"
        joblib.dump(model, model_file)
        trained_models[name] = model
        
        cm = confusion_matrix(y_test, y_pred)
        
        results.append({
            "Model": name,
            "Accuracy": round(float(acc), 4),
            "Precision_Macro": round(float(prec_macro), 4),
            "Recall_Macro": round(float(rec_macro), 4),
            "F1_Macro": round(float(f1_macro), 4),
            "Precision_Weighted": round(float(prec_wt), 4),
            "Recall_Weighted": round(float(rec_wt), 4),
            "F1_Weighted": round(float(f1_wt), 4),
            "Confusion_Matrix": cm.tolist()
        })
        
    results_df = pd.DataFrame(results)
    results_csv = REPORTS_DIR / "model_comparison_metrics.csv"
    results_df.drop(columns=["Confusion_Matrix"]).to_csv(results_csv, index=False)
    print(f"\nSaved comparison metrics table to {results_csv}")
    
    # Save full results json
    with open(MODELS_DIR / "model_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
        
    # Identify best model by F1-Macro
    best_res = max(results, key=lambda x: x["F1_Macro"])
    best_name = best_res["Model"]
    best_model = trained_models[best_name]
    print(f"\n>>> Best Performing Model: {best_name} with Accuracy = {best_res['Accuracy']*100:.2f}%, F1 = {best_res['F1_Macro']:.4f} <<<")
    
    # Save best model
    joblib.dump(best_model, MODELS_DIR / "best_har_model.pkl")
    
    # Save model metadata
    metadata = {
        "best_model_name": best_name,
        "best_accuracy": best_res["Accuracy"],
        "best_f1_macro": best_res["F1_Macro"],
        "activity_map": ACTIVITY_MAP,
        "feature_names": feature_names,
        "requires_scaling": best_name in ["Support Vector Machine (SVM)", "Logistic Regression"]
    }
    with open(MODELS_DIR / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
        
    # Generate and save Confusion Matrix plots
    best_cm = np.array(best_res["Confusion_Matrix"])
    plot_confusion_matrices(best_cm, activity_labels, best_name)
    
    # Generate Feature Importance plot from Random Forest
    rf_model = trained_models["Random Forest"]
    plot_rf_feature_importance(rf_model, feature_names)
    
    # Detailed per-class classification report for the best model
    if best_name in ["Support Vector Machine (SVM)", "Logistic Regression"]:
        y_pred_best = best_model.predict(X_test_scaled)
    else:
        y_pred_best = best_model.predict(X_test)
        
    report_dict = classification_report(y_test, y_pred_best, target_names=activity_labels, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    report_csv = REPORTS_DIR / "best_model_classification_report.csv"
    report_df.to_csv(report_csv)
    print(f"Saved classification report to {report_csv}")

def plot_confusion_matrices(cm, labels, model_name):
    """Plot both raw counts and normalized percentage confusion matrices."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=300)
    
    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=axes[0], cbar=False)
    axes[0].set_title(f"Confusion Matrix (Raw Counts)\n{model_name}", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Predicted Activity", fontsize=11, fontweight='bold')
    axes[0].set_ylabel("True Activity", fontsize=11, fontweight='bold')
    axes[0].tick_params(axis='x', rotation=30)
    
    # Normalized percentages
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_norm, annot=True, fmt='.1%', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=axes[1], cbar=True)
    axes[1].set_title(f"Confusion Matrix (Normalized Recall %)\n{model_name}", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Predicted Activity", fontsize=11, fontweight='bold')
    axes[1].set_ylabel("True Activity", fontsize=11, fontweight='bold')
    axes[1].tick_params(axis='x', rotation=30)
    
    plt.tight_layout()
    fig_path = FIG_DIR / "confusion_matrix.png"
    plt.savefig(fig_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved: {fig_path}")

def plot_rf_feature_importance(rf_model, feature_names):
    """Plots top 15 most discriminative features determined by Random Forest."""
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1][:15]
    
    top_features = [feature_names[i] for i in indices]
    top_scores = importances[indices]
    
    plt.figure(figsize=(11, 6), dpi=300)
    y_pos = np.arange(len(top_features))
    
    plt.barh(y_pos, top_scores, align='center', color='#3498db', edgecolor='#2980b9')
    plt.yticks(y_pos, top_features, fontsize=9)
    plt.gca().invert_yaxis()  # top feature on top
    plt.xlabel("Mean Impurity Reduction (Gini Feature Importance)", fontsize=11, fontweight='bold')
    plt.title("Top 15 Most Discriminative Sensor Features (Random Forest)", fontsize=13, fontweight='bold', pad=12)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    fig_path = FIG_DIR / "feature_importance.png"
    plt.savefig(fig_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved: {fig_path}")

if __name__ == "__main__":
    train_and_evaluate_models()

