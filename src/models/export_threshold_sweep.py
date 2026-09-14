import numpy as np
import pandas as pd
from src.db.connection import get_engine

def export_threshold_sweep():
    engine = get_engine()

    df = pd.read_sql(
        "SELECT actual_readmitted_30d, predicted_probability FROM predictions_export",
        engine
    )
    print(f"[1/3] Loaded {len(df):,} rows from predictions_export")

    y_true = df["actual_readmitted_30d"].to_numpy()
    y_prob = df["predicted_probability"].to_numpy()

    thresholds = np.arange(0.02, 0.96, 0.01)
    rows = []

    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        tp = int(((y_true == 1) & (preds == 1)).sum())
        fn = int(((y_true == 1) & (preds == 0)).sum())
        fp = int(((y_true == 0) & (preds == 1)).sum())
        tn = int(((y_true == 0) & (preds == 0)).sum())

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0

        rows.append({
            "threshold": round(float(t), 2),
            "recall": round(recall, 4),
            "precision": round(precision, 4),
            "true_positives": tp,
            "false_negatives": fn,
            "false_positives": fp,
            "true_negatives": tn
        })

    sweep_df = pd.DataFrame(rows)
    print(f"[2/3] Built full sweep — {len(sweep_df)} threshold values")

    sweep_df.to_sql("threshold_sweep_export", engine, if_exists="replace", index=False)
    print(f"[3/3] Wrote threshold_sweep_export to Postgres ({len(sweep_df)} rows)")

if __name__ == "__main__":
    export_threshold_sweep()