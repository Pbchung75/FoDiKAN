"""Dataset discovery and input loading."""

from __future__ import annotations

import gzip
import io
import os
import zipfile
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def df_map(df: pd.DataFrame, func):
    try:
        return df.map(func)
    except AttributeError:
        return df.applymap(func)

def _read_csv_or_excel_robust(path: str) -> pd.DataFrame:
    with open(path, "rb") as fh:
        sig = fh.read(4)

    if sig[:2] == b"PK":
        with open(path, "rb") as fh:
            data = fh.read()
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                names = zf.namelist()
                if any(n.startswith("xl/") for n in names):
                    return pd.read_excel(io.BytesIO(data))
                csv_names = [n for n in names if n.lower().endswith(".csv")]
                txt_names = [n for n in names if n.lower().endswith((".txt", ".tsv"))]
                if csv_names:
                    with zf.open(csv_names[0]) as fh:
                        return pd.read_csv(fh, sep=None, engine="python")
                if txt_names:
                    with zf.open(txt_names[0]) as fh:
                        return pd.read_csv(fh, sep=None, engine="python")
        except zipfile.BadZipFile as exc:
            raise ValueError(f"Broken ZIP/XLSX file: {path}") from exc

    last_err = None
    for enc in ["utf-8-sig", "utf-8", "cp1252", "latin1", "iso-8859-1", "utf-16", "utf-16le", "utf-16be"]:
        try:
            return pd.read_csv(path, sep=None, engine="python", encoding=enc)
        except Exception as exc:
            last_err = exc
    try:
        return pd.read_csv(path, sep=None, engine="python", encoding="latin1", on_bad_lines="skip")
    except Exception as exc:
        raise ValueError(f"Cannot read file: {path}. Last error: {exc} | Previous: {last_err}") from exc

def load_data_from_csv_smart(path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    df = _read_csv_or_excel_robust(path)
    df = df.replace(r"^\s*$", np.nan, regex=True).dropna(how="all").reset_index(drop=True)

    if df.shape[1] < 2:
        raise ValueError(f"{os.path.basename(path)} must contain >=2 columns (features + label).")

    X_df = df.iloc[:, :-1].copy()
    y_raw = df.iloc[:, -1].values

    def is_suspicious(v: Any) -> bool:
        return isinstance(v, (dict, list)) or (isinstance(v, str) and "OrderedDict" in v)

    susp = df_map(X_df, is_suspicious)
    if bool(susp.values.any()):
        locs = np.argwhere(susp.values)[:10]
        print(f"[Warning] Suspicious cells in {os.path.basename(path)}:")
        for r, c in locs:
            print(f"  row={r}, col={X_df.columns[c]}, value={repr(X_df.iat[r, c])}")

    X_df = X_df.apply(pd.to_numeric, errors="coerce")
    all_nan_cols = X_df.columns[X_df.isna().all()].tolist()
    if all_nan_cols:
        X_df = X_df.drop(columns=all_nan_cols)

    if X_df.shape[1] == 0:
        raise ValueError(f"All feature columns became NaN after numeric conversion: {path}")

    X_df = X_df.apply(lambda s: s.fillna(s.median()), axis=0)

    X = X_df.to_numpy(dtype=np.float32, copy=False)
    y = LabelEncoder().fit_transform(y_raw).astype(np.int64, copy=False)
    if len(np.unique(y)) < 2:
        raise ValueError(f"{path}: only one class after encoding.")
    feature_names = [str(c) for c in X_df.columns]
    return X, y, feature_names

def load_dataset_dhkan_style(dataset_dir: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    path = os.path.join(dataset_dir, "data.trn.gz")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing data.trn.gz at: {path}")
    with gzip.open(path, "rt") as fh:
        df = pd.read_csv(fh, header=None, sep=r"\s+")
    if df.shape[1] < 2:
        raise ValueError(f"{path}: expected >=2 columns (features + label)")
    X = df.iloc[:, :-1].values.astype(np.float32, copy=False)
    y_raw = df.iloc[:, -1].values
    y = LabelEncoder().fit_transform(y_raw).astype(np.int64, copy=False)
    feature_names = [f"Gene_{i}" for i in range(X.shape[1])]
    return X, y, feature_names

@dataclass
class InputItem:
    kind: str
    dataset_id: str
    path: str

def dataset_id_from_filename(file_name: str) -> str:
    stem = os.path.splitext(os.path.basename(file_name))[0]
    for suf in ("_selected", "_transformed", "_merged"):
        if stem.endswith(suf):
            return stem[:-len(suf)]
    return stem

def list_input_files(base_dir: str) -> List[str]:
    out: List[str] = []
    for f in sorted(os.listdir(base_dir)):
        p = os.path.join(base_dir, f)
        if not os.path.isfile(p):
            continue
        if f.lower().endswith(("_generated.csv", "_generated.tsv", "_generated.txt")):
            continue
        if f.lower().endswith((".csv", ".tsv", ".txt", ".xlsx")):
            out.append(f)
    return out

def discover_inputs(base_dir: str, dataset_ids: Optional[List[str]] = None) -> List[InputItem]:

    items: List[InputItem] = []
    if not os.path.isdir(base_dir):
        return items

    wanted = set(dataset_ids) if dataset_ids else None

    subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    dhkan_dirs = []
    for d in subdirs:
        if os.path.isfile(os.path.join(base_dir, d, "data.trn.gz")):
            dhkan_dirs.append(d)

    if dhkan_dirs:
        dhkan_dirs = sorted(dhkan_dirs, key=lambda x: int(x) if str(x).isdigit() else str(x))
        for ds_id in dhkan_dirs:
            if wanted is not None and str(ds_id) not in wanted:
                continue
            items.append(InputItem(kind="dhkan", dataset_id=str(ds_id), path=os.path.join(base_dir, ds_id)))
        return items

    for f in list_input_files(base_dir):
        ds_id = dataset_id_from_filename(f)
        if wanted is not None and str(ds_id) not in wanted:
            continue
        items.append(InputItem(kind="csv", dataset_id=str(ds_id), path=os.path.join(base_dir, f)))
    return items
