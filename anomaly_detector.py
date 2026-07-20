"""
anomaly_detector.py
---------------------------------
Behavioral risk engine using IsolationForest + temporal drift detection.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

FEATURE_NAMES = [
    "failed_login_count",
    "login_count",
    "off_hours_ratio",
    "bytes_out",
    "bytes_in",
    "unique_dst_count",
    "new_process_count",
    "privilege_change_count",
    "ot_write_command_count",
    "session_duration_std",
]

def run_anomaly_detection(df: pd.DataFrame) -> pd.DataFrame:
    # Per-entity-type IsolationForest scoring
    df["if_score"] = 0.0
    for etype, group in df.groupby("entity_type"):
        X = group[FEATURE_NAMES].values
        clf = IsolationForest(contamination=0.1, random_state=42)
        clf.fit(X)
        # Convert decision function to normalized 0-1 anomaly score
        scores = -clf.decision_function(X)
        min_s, max_s = scores.min(), scores.max()
        norm_scores = (scores - min_s) / (max_s - min_s + 1e-9)
        df.loc[group.index, "if_score"] = norm_scores

    # Temporal drift detection (global population baseline)
    X_all = df[FEATURE_NAMES].values
    mean_baseline = X_all.mean(axis=0)
    std_baseline = X_all.std(axis=0) + 1e-9
    z_scores = np.abs((X_all - mean_baseline) / std_baseline)
    df["drift_score"] = z_scores.mean(axis=1)
    
    # Global normalization for drift score
    d_min, d_max = df["drift_score"].min(), df["drift_score"].max()
    df["drift_score"] = (df["drift_score"] - d_min) / (d_max - d_min + 1e-9)

    # Combined risk score (weighted ensemble)
    df["risk_score"] = 0.7 * df["if_score"] + 0.3 * df["drift_score"]
    return df

if __name__ == "__main__":
    import pathlib
    module_dir = pathlib.Path(__file__).parent
    
    # Load raw events data
    data_path = module_dir / "entity_events.csv"
    if not data_path.exists():
        print("❌ `entity_events.csv` not found!")
        exit(1)

    df = pd.read_csv(data_path)
    result = run_anomaly_detection(df)
    
    # Save risk scores output
    result[["entity_id", "entity_type", "day", "risk_score", "if_score", "drift_score"]].to_csv(
        module_dir / "risk_scores.csv", index=False
    )
    print("✅ Anomaly detection complete. Output saved to `risk_scores.csv`.")
