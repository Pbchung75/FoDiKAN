from __future__ import annotations
from typing import Any, Dict, List
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def summarize_group(df: pd.DataFrame, by_cols: List[str], metric_cols: List[str]) -> pd.DataFrame:
    agg = {}
    for col in metric_cols:
        agg[col] = ["mean", "std"]
    out = df.groupby(by_cols, dropna=False).agg(agg).reset_index()
    out.columns = [
        "_".join([c for c in tup if c]).rstrip("_")
        if isinstance(tup, tuple) else str(tup)
        for tup in out.columns
    ]
    return out

def compute_paired_tests(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if summary_df.empty:
        return pd.DataFrame(rows)

    work = summary_df.copy()
    work["Method"] = work["Model"] + "|" + work["Mode"]

    target_pairs = [
        ("EfficientKAN|DiffAug", "EfficientKAN|NoAug"),
        ("FastKAN|DiffAug", "FastKAN|NoAug"),
        ("FasterKAN|DiffAug", "FasterKAN|NoAug"),
        ("EfficientKAN|DiffAug-noAlign", "EfficientKAN|NoAug"),
        ("FastKAN|DiffAug-noAlign", "FastKAN|NoAug"),
        ("FasterKAN|DiffAug-noAlign", "FasterKAN|NoAug"),
        ("FasterKAN|DiffAug", "GB|NoAug"),
        ("FasterKAN|DiffAug", "XGB|NoAug"),
        ("FasterKAN|DiffAug-noAlign", "GB|NoAug"),
        ("FasterKAN|DiffAug-noAlign", "XGB|NoAug"),
    ]
    for backbone in ["EfficientKAN", "FastKAN", "FasterKAN"]:
        for aug in ["SMOTE", "BorderlineSMOTE", "KMeansSMOTE"]:
            target_pairs.append((f"{backbone}|{aug}", f"{backbone}|NoAug"))
            target_pairs.append((f"{backbone}|DiffAug", f"{backbone}|{aug}"))
            target_pairs.append((f"{backbone}|DiffAug-noAlign", f"{backbone}|{aug}"))

    for a, b in target_pairs:
        da = work[work["Method"] == a][["Dataset", "F1_mean"]].rename(columns={"F1_mean": "A"})
        db = work[work["Method"] == b][["Dataset", "F1_mean"]].rename(columns={"F1_mean": "B"})
        merged = da.merge(db, on="Dataset", how="inner")
        if merged.shape[0] == 0:
            continue
        diffs = merged["A"].values - merged["B"].values
        if np.allclose(diffs, 0.0):
            p_value = 1.0
            stat = 0.0
        else:
            try:
                stat, p_value = wilcoxon(merged["A"].values, merged["B"].values, alternative="two-sided", zero_method="wilcox")
            except Exception:
                stat, p_value = np.nan, np.nan

        rows.append({
            "Comparison": f"{a} vs {b}",
            "Method_A": a,
            "Method_B": b,
            "n_datasets": int(merged.shape[0]),
            "mean_diff": float(np.mean(diffs)),
            "median_diff": float(np.median(diffs)),
            "wilcoxon_stat": float(stat) if stat == stat else np.nan,
            "p_value": float(p_value) if p_value == p_value else np.nan,
        })

    return pd.DataFrame(rows)
