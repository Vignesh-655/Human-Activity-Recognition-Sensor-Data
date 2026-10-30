"""
Standalone Model Evaluation and Diagnostic Analysis Module for HAR.
Computes comprehensive metrics, per-class breakdowns, and error diagnostics.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from src.data_loader import load_har_data, ACTIVITY_MAP

def evaluate_best_model():
    models_dir = PROJECT_ROOT / "models"
    reports_dir = PROJECT_ROOT / "reports"
    
    with open(models_dir / "model_metadata.json", "r") as f:
        metadata = json.load(f)
        
    best_model_name = metadata["best_model_name"]
    requires_scaling = metadata["requires_scaling"]
    
    print(f"Loading best model: {best_model_name}...")
    model = joblib.load(models_dir / "best_har_model.pkl")
    scaler = joblib.load(models_dir / "scaler.pkl")
    
    _, _, _, X_test, y_test, sub_test, feature_names = load_har_data()
    
    if requires_scaling:
        X_eval = scaler.transform(X_test)
    else:
        X_eval = X_test
        
    y_pred = model.predict(X_eval)
    
    accuracy = accuracy_score(y_test, y_pred)
    target_names = [ACTIVITY_MAP[i] for i in range(1, 7)]
    
    print(f"\n=======================================================")
    print(f"FINAL TEST SET EVALUATION - {best_model_name}")
    print(f"=======================================================")
    print(f"Overall Test Accuracy: {accuracy * 100:.2f}%\n")
    
    report_str = classification_report(y_test, y_pred, target_names=target_names, digits=4)
    print(report_str)
    
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=[f"True: {name}" for name in target_names],
                             columns=[f"Pred: {name}" for name in target_names])
    print("Confusion Matrix:")
    print(cm_df)
    
    # Analyze common confusions
    print("\n--- Key Misclassification Insights ---")
    sitting_idx = 3 # 4-1
    standing_idx = 4 # 5-1
    sitting_as_standing = cm[sitting_idx, standing_idx]
    standing_as_sitting = cm[standing_idx, sitting_idx]
    print(f"• Sitting misclassified as Standing: {sitting_as_standing} instances")
    print(f"• Standing misclassified as Sitting: {standing_as_sitting} instances")
    print("• Biomechanical reason: Both activities exhibit near-zero dynamic acceleration with constant gravitational vector (1.0g). Device orientation differences account for minor boundary ambiguities.")
    
    return accuracy, cm_df

if __name__ == "__main__":
    evaluate_best_model()
