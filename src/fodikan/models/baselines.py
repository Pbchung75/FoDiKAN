"""Classical machine-learning baselines."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier

    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False


def fit_predict_baseline(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    num_classes: int,
    seed: int,
) -> np.ndarray:
    if model_name == "SVM":
        clf = SVC(C=10.0, gamma="scale", kernel="rbf", class_weight=None)
    elif model_name == "SVM-balanced":
        clf = SVC(C=10.0, gamma="scale", kernel="rbf", class_weight="balanced")
    elif model_name == "RF":
        clf = RandomForestClassifier(
            n_estimators=500,
            max_features="sqrt",
            min_samples_leaf=1,
            random_state=seed,
            n_jobs=-1,
        )
    elif model_name == "GB":
        clf = GradientBoostingClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=3,
            random_state=seed,
        )
    elif model_name == "XGB":
        if not HAS_XGBOOST:
            raise ImportError("xgboost is not installed. Install it or remove XGB from --models.")
        kwargs = dict(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            n_jobs=-1,
            tree_method="hist",
            verbosity=0,
        )
        if num_classes > 2:
            clf = XGBClassifier(
                objective="multi:softprob",
                num_class=num_classes,
                eval_metric="mlogloss",
                **kwargs,
            )
        else:
            clf = XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                **kwargs,
            )
    else:
        raise ValueError(f"Unsupported baseline model: {model_name}")

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_pred = np.asarray(y_pred)
    if y_pred.ndim > 1:
        y_pred = np.argmax(y_pred, axis=1)
    return y_pred.astype(np.int64, copy=False)
