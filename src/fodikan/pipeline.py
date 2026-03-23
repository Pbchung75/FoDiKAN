"""Top-level experiment runner."""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections import Counter
from dataclasses import asdict
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from fodikan.config import AUGMENT_MODES, Args, BASE_RESULTS_PATH, DEFAULT_DATA_DIR, is_diffusion_mode, parse_args
from fodikan.data.cv import choose_size_aware_cv, fixed_label_metrics, safe_train_val_split_indices
from fodikan.data.io import discover_inputs, load_data_from_csv_smart, load_dataset_dhkan_style
from fodikan.data.loaders import make_eval_loader, make_loader_with_weights
from fodikan.diffusion.augment import augment_training_set_by_mode
from fodikan.evaluation.alignment import compute_alignment_diagnostics
from fodikan.features.selection import BoMGeneSelector
from fodikan.models.baselines import fit_predict_baseline
from fodikan.models.kan import KAN_MODEL_NAMES, build_kan_model
from fodikan.results.reporting import compute_paired_tests, summarize_group
from fodikan.training.torch_trainer import (
    evaluate_torch_model,
    select_epoch_by_inner_cv,
    train_kan_fixed_epochs,
    train_kan_with_real_val,
)
from fodikan.utils.logging import TxtLogger, format_table
from fodikan.utils.repro import DEVICE, set_seed


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    run_dir = os.path.join(args.results_path, args.run_name)
    os.makedirs(run_dir, exist_ok=True)

    fs_cache_dir = os.path.join(run_dir, "_fs_cache") if not args.no_fs else None
    synth_out_dir = os.path.join(run_dir, "_synth_cache") if args.save_synthesis else None
    artifacts_out_dir = os.path.join(run_dir, "_fold_artifacts") if args.save_fold_artifacts else None

    log_file = args.log_file if args.log_file.strip() else os.path.join(run_dir, f"run_{time.strftime('%Y%m%d_%H%M%S')}.txt")
    logger = TxtLogger(log_file, also_print=True)

    try:
        logger.write("=" * 100)
        logger.write("FoDiKAN paper-aligned run")
        logger.write(f"Device: {DEVICE}")
        logger.write(f"Run dir: {run_dir}")
        logger.write(f"Default data dir: {DEFAULT_DATA_DIR}")
        logger.write(f"Default results dir: {BASE_RESULTS_PATH}")
        logger.write(f"Args: {json.dumps(asdict(args), ensure_ascii=False, indent=2)}")
        logger.write("=" * 100)

        with open(os.path.join(run_dir, "run_config.json"), "w", encoding="utf-8") as fh:
            json.dump(asdict(args), fh, ensure_ascii=False, indent=2)

        items = discover_inputs(args.data_path, dataset_ids=args.dataset_ids)
        logger.write(f"[INPUT] data_path={args.data_path}")
        logger.write(f"[INPUT] discovered {len(items)} dataset(s)")
        if len(items) == 0:
            logger.write("[FATAL] No datasets found.")
            return

        fold_rows: List[Dict[str, Any]] = []
        align_rows: List[Dict[str, Any]] = []

        for item in items:
            dataset_id = str(item.dataset_id)
            logger.write("\n" + "#" * 100)
            logger.write(f"DATASET: {dataset_id}")
            logger.write("#" * 100)

            try:
                if item.kind == "dhkan":
                    X_all, y_all, feature_names = load_dataset_dhkan_style(item.path)
                    logger.write(f"[LOAD] kind=dhkan dir={item.path}")
                else:
                    X_all, y_all, feature_names = load_data_from_csv_smart(item.path)
                    logger.write(f"[LOAD] kind=csv file={item.path}")
            except Exception as exc:
                logger.write(f"[ERROR] Cannot read dataset {dataset_id}: {exc}")
                continue

            n_samples, n_features = X_all.shape
            num_classes = int(np.max(y_all)) + 1
            class_counts = {int(k): int(v) for k, v in Counter(y_all.tolist()).items()}
            logger.write(f"n={n_samples} | p={n_features} | classes={num_classes} | counts={class_counts}")

            outer_cv, cv_name = choose_size_aware_cv(y_all, seed=args.seed)
            logger.write(f"[CV] {cv_name}")

            requested_aug_modes = []
            for mode_name in args.modes:
                if mode_name == "NoAug":
                    continue
                if is_diffusion_mode(mode_name) and args.no_diffusion:
                    continue
                if mode_name in AUGMENT_MODES:
                    requested_aug_modes.append(mode_name)
            need_any_augmentation = bool(requested_aug_modes) and any(
                m in {"EfficientKAN", "FastKAN", "FasterKAN"} for m in args.models
            )

            for fold_idx, (train_outer_idx, test_outer_idx) in enumerate(outer_cv.split(X_all, y_all), start=1):
                logger.write("\n" + "-" * 100)
                logger.write(f"Fold {fold_idx}")

                X_train_outer = X_all[train_outer_idx]
                y_train_outer = y_all[train_outer_idx]
                X_test_outer = X_all[test_outer_idx]
                y_test_outer = y_all[test_outer_idx]

                tr_rel, val_rel, split_meta = safe_train_val_split_indices(
                    y_train_outer, val_ratio=args.val_ratio, seed=args.seed + fold_idx
                )
                X_tr_raw = X_train_outer[tr_rel]
                y_tr = y_train_outer[tr_rel]
                if len(val_rel) > 0:
                    X_val_raw = X_train_outer[val_rel]
                    y_val = y_train_outer[val_rel]
                else:
                    X_val_raw = np.zeros((0, X_all.shape[1]), dtype=X_all.dtype)
                    y_val = np.zeros((0,), dtype=y_all.dtype)

                logger.write(
                    f"[SafeCV] |D_tr|={len(y_tr)} |D_val|={len(y_val)} |D_te|={len(y_test_outer)} "
                    f"| realized_val_ratio={split_meta['realized_val_ratio']:.4f} | reason={split_meta['reason']}"
                )
                logger.write(f"[SafeCV] D_tr counts={dict(Counter(y_tr.tolist()))}")
                if len(y_val) > 0:
                    logger.write(f"[SafeCV] D_val counts={dict(Counter(y_val.tolist()))}")
                logger.write(f"[SafeCV] D_te counts={dict(Counter(y_test_outer.tolist()))}")

                scaler = StandardScaler().fit(X_tr_raw)
                X_tr_std = scaler.transform(X_tr_raw)
                X_val_std = scaler.transform(X_val_raw) if len(y_val) else X_val_raw
                X_te_std = scaler.transform(X_test_outer)

                train_hash = hashlib.md5(np.asarray(train_outer_idx[tr_rel], dtype=np.int64).tobytes()).hexdigest()[:10]

                if not args.no_fs:
                    selector = BoMGeneSelector(
                        seed=args.seed,
                        cache_dir=fs_cache_dir,
                        pre_screen_topk=args.pre_screen_topk,
                        mrmr_candidate_pool=args.mrmr_candidate_pool,
                        mrmr_topk=args.mrmr_topk,
                        redundancy_threshold=args.mrmr_redundancy_threshold,
                        boruta_n_estimators=args.boruta_n_estimators,
                        boruta_max_iter=args.boruta_max_iter,
                        boruta_alpha=args.boruta_alpha,
                        min_keep=args.fs_min_keep,
                        fallback_topk=args.fs_fallback_topk,
                    )
                    selector.fit(
                        X=X_tr_std,
                        y=y_tr,
                        feature_names=feature_names,
                        dataset_id=dataset_id,
                        fold_idx=fold_idx,
                        train_hash=train_hash,
                    )
                    X_tr_fs = selector.transform(X_tr_std)
                    X_val_fs = selector.transform(X_val_std) if len(y_val) else X_val_std
                    X_te_fs = selector.transform(X_te_std)
                    kept_feature_names = selector.get_kept_names()
                    logger.write(
                        f"[FS] kept={len(kept_feature_names)} | mRMR={selector.stats_.n_mrmr} | "
                        f"Boruta={selector.stats_.n_boruta} | cache={selector.stats_.from_cache}"
                    )
                else:
                    X_tr_fs = X_tr_std
                    X_val_fs = X_val_std
                    X_te_fs = X_te_std
                    kept_feature_names = list(feature_names)
                    logger.write(f"[FS] disabled -> using all standardized features ({X_tr_fs.shape[1]})")

                outer_aug_by_mode: Dict[str, AugmentationResult] = {}
                if need_any_augmentation:
                    for aug_mode in requested_aug_modes:
                        aug_res = augment_training_set_by_mode(
                            mode=aug_mode,
                            dataset_id=dataset_id,
                            fold_idx=fold_idx,
                            X_tr_fs=X_tr_fs,
                            y_tr=y_tr,
                            feature_names=kept_feature_names,
                            args=args,
                            synth_out_dir=synth_out_dir,
                            save_synth=args.save_synthesis,
                        )
                        outer_aug_by_mode[str(aug_mode)] = aug_res
                        logger.write(
                            f"[{aug_mode}] note={aug_res.meta.get('note', '')} | "
                            f"n_syn={aug_res.meta.get('n_syn', 0)} | quota={aug_res.meta.get('quota', {})} | "
                            f"guarded={aug_res.meta.get('guarded_class_counts', {})}"
                        )
                        if aug_res.X_syn is not None and aug_res.y_syn is not None and len(aug_res.y_syn) > 0:
                            align_rows.extend(
                                compute_alignment_diagnostics(
                                    dataset_id=dataset_id,
                                    fold_idx=fold_idx,
                                    mode=str(aug_mode),
                                    X_real=X_tr_fs,
                                    y_real=y_tr,
                                    X_syn=aug_res.X_syn,
                                    y_syn=aug_res.y_syn,
                                    bins=args.alignment_bins,
                                )
                            )

                if artifacts_out_dir:
                    try:
                        os.makedirs(artifacts_out_dir, exist_ok=True)
                        primary_outer_aug_key = (
                            "DiffAug"
                            if "DiffAug" in outer_aug_by_mode
                            else ("DiffAug-noAlign" if "DiffAug-noAlign" in outer_aug_by_mode else None)
                        )
                        artifact = {
                            "dataset_id": dataset_id,
                            "fold": fold_idx,
                            "cv_name": cv_name,
                            "train_outer_size": int(len(train_outer_idx)),
                            "test_outer_size": int(len(test_outer_idx)),
                            "d_tr_size": int(len(y_tr)),
                            "d_val_size": int(len(y_val)),
                            "d_te_size": int(len(y_test_outer)),
                            "split_meta": split_meta,
                            "n_selected_features": int(X_tr_fs.shape[1]),
                            "selected_feature_names": kept_feature_names,
                            "outer_aug_meta": outer_aug_by_mode.get(primary_outer_aug_key).meta if primary_outer_aug_key else {},
                            "outer_aug_meta_by_mode": {k: v.meta for k, v in outer_aug_by_mode.items()},
                        }
                        with open(os.path.join(artifacts_out_dir, f"{dataset_id}_fold{fold_idx:03d}.json"), "w", encoding="utf-8") as fh:
                            json.dump(artifact, fh, ensure_ascii=False, indent=2)
                    except Exception:
                        pass

                for model_name in args.models:
                    is_kan = model_name in {"EfficientKAN", "FastKAN", "FasterKAN"}
                    modes_for_model = ["NoAug"]
                    if is_kan:
                        extra_modes = []
                        for mode_name in args.modes:
                            if mode_name == "NoAug":
                                continue
                            if is_diffusion_mode(mode_name) and args.no_diffusion:
                                continue
                            if mode_name in AUGMENT_MODES:
                                extra_modes.append(mode_name)
                        modes_for_model = list(dict.fromkeys(["NoAug"] + extra_modes))

                    for mode in modes_for_model:
                        if mode not in args.modes:
                            continue
                        logger.write(f"[RUN] model={model_name} | mode={mode}")
                        t0 = time.time()
                        aug_meta_for_row: Optional[Dict[str, Any]] = None

                        if not is_kan:
                            y_pred = fit_predict_baseline(
                                model_name=model_name,
                                X_train=X_tr_fs,
                                y_train=y_tr,
                                X_test=X_te_fs,
                                num_classes=num_classes,
                                seed=args.seed + fold_idx,
                            )
                            metrics = fixed_label_metrics(y_test_outer, y_pred, num_classes=num_classes)
                            best_epoch = np.nan
                            val_criterion = "n/a"
                            n_syn_used = 0
                        else:
                            aug_meta_for_row: Optional[Dict[str, Any]] = None
                            if mode != "NoAug" and mode in outer_aug_by_mode:
                                aug_res = outer_aug_by_mode[mode]
                                X_train = aug_res.X_aug
                                y_train = aug_res.y_aug
                                w_train = aug_res.w_aug
                                n_syn_used = int(aug_res.meta.get("n_syn", 0))
                                aug_meta_for_row = aug_res.meta
                            else:
                                X_train = X_tr_fs
                                y_train = y_tr
                                w_train = np.ones(len(y_tr), dtype=np.float32)
                                n_syn_used = 0
                                aug_meta_for_row = None

                            train_loader = make_loader_with_weights(
                                X_train, y_train, w_train,
                                batch_size=args.batch_size,
                                shuffle=True,
                                seed=args.seed + fold_idx,
                            )
                            test_loader = make_eval_loader(X_te_fs, y_test_outer, batch_size=args.batch_size)

                            model = build_kan_model(
                                model_name=model_name,
                                input_dim=int(X_tr_fs.shape[1]),
                                num_classes=num_classes,
                                extra_paths=args.extra_python_path,
                            )

                            if len(y_val) > 0:
                                val_loader = make_eval_loader(X_val_fs, y_val, batch_size=args.batch_size)
                                best_epoch, _, val_criterion = train_kan_with_real_val(
                                    model=model,
                                    train_loader=train_loader,
                                    val_loader=val_loader,
                                    num_classes=num_classes,
                                    lr=1e-3,
                                    weight_decay=1e-4,
                                    max_epochs=args.num_epochs,
                                    patience=args.patience,
                                )
                            else:
                                best_epoch, val_criterion = select_epoch_by_inner_cv(
                                    model_name=model_name,
                                    X_real=X_tr_fs,
                                    y_real=y_tr,
                                    feature_names=kept_feature_names,
                                    args=args,
                                    aug_mode=(mode if mode != "NoAug" else None),
                                )
                                train_kan_fixed_epochs(
                                    model=model,
                                    train_loader=train_loader,
                                    epochs=best_epoch,
                                    lr=1e-3,
                                    weight_decay=1e-4,
                                )

                            metrics = evaluate_torch_model(model, test_loader, num_classes)
                            metrics = {
                                "Accuracy": float(metrics["Accuracy"]),
                                "Precision": float(metrics["Precision"]),
                                "Recall": float(metrics["Recall"]),
                                "F1": float(metrics["F1"]),
                            }

                        train_time_sec = float(time.time() - t0)

                        row = {
                            "Dataset": dataset_id,
                            "Model": model_name,
                            "Mode": mode,
                            "Fold": int(fold_idx),
                            "CV_Type": cv_name,
                            "n_samples": int(n_samples),
                            "n_features_raw": int(n_features),
                            "n_classes": int(num_classes),
                            "n_features_selected": int(X_tr_fs.shape[1]),
                            "n_syn": int(n_syn_used),
                            "d_tr_size": int(len(y_tr)),
                            "d_val_size": int(len(y_val)),
                            "d_te_size": int(len(y_test_outer)),
                            "realized_val_ratio": float(split_meta["realized_val_ratio"]),
                            "val_classes_missing": int(num_classes - len(np.unique(y_val))) if len(y_val) else int(num_classes),
                            "test_classes_missing": int(num_classes - len(np.unique(y_test_outer))),
                            "rare_guard_threshold": int(args.min_real_for_synthesis),
                            "rare_guarded_classes": json.dumps(aug_meta_for_row.get("guarded_class_counts", {}), ensure_ascii=False) if aug_meta_for_row is not None else "",
                            "BestEpoch": best_epoch,
                            "ValCriterion": val_criterion,
                            "TrainTime_sec": train_time_sec,
                            **metrics,
                        }
                        fold_rows.append(row)
                        logger.write(
                            f"[DONE] {model_name}/{mode} | F1={metrics['F1']:.4f} | "
                            f"Acc={metrics['Accuracy']:.4f} | nSyn={n_syn_used} | "
                            f"BestEpoch={best_epoch} | ValCriterion={val_criterion}"
                        )

        fold_df = pd.DataFrame(fold_rows)
        if fold_df.empty:
            logger.write("[FATAL] No successful fold results were produced.")
            return

        metric_cols = [
            "n_features_selected", "n_syn",
            "Accuracy", "Precision", "Recall", "F1",
            "TrainTime_sec",
        ]
        summary_df = summarize_group(
            fold_df,
            by_cols=["Dataset", "Model", "Mode", "CV_Type", "n_samples", "n_features_raw", "n_classes"],
            metric_cols=metric_cols,
        )
        summary_df = summary_df.rename(columns={
            "n_features_selected_mean": "n_features_mean",
            "n_features_selected_std": "n_features_std",
            "n_syn_mean": "n_syn_mean",
            "n_syn_std": "n_syn_std",
            "Accuracy_mean": "Accuracy_mean",
            "Accuracy_std": "Accuracy_std",
            "Precision_mean": "Precision_mean",
            "Precision_std": "Precision_std",
            "Recall_mean": "Recall_mean",
            "Recall_std": "Recall_std",
            "F1_mean": "F1_mean",
            "F1_std": "F1_std",
            "TrainTime_sec_mean": "TrainTime_mean_sec",
            "TrainTime_sec_std": "TrainTime_std_sec",
        })

        paired_df = compute_paired_tests(summary_df)
        align_df = pd.DataFrame(align_rows)

        fold_path = os.path.join(run_dir, "fold_details.csv")
        summary_path = os.path.join(run_dir, "dataset_summary.csv")
        paired_path = os.path.join(run_dir, "paired_tests.csv")
        align_path = os.path.join(run_dir, "alignment_diagnostics.csv")

        fold_df.to_csv(fold_path, index=False, encoding="utf-8-sig")
        summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
        paired_df.to_csv(paired_path, index=False, encoding="utf-8-sig")
        align_df.to_csv(align_path, index=False, encoding="utf-8-sig")

        logger.write("\n" + "=" * 100)
        logger.write("RUN SUMMARY")
        logger.write("=" * 100)

        try:
            display_cols = ["Dataset", "Model", "Mode", "F1_mean", "F1_std", "Accuracy_mean", "n_features_mean", "n_syn_mean"]
            show = summary_df[display_cols].copy().sort_values(["Dataset", "Model", "Mode"]).reset_index(drop=True)
            rows = []
            for _, r in show.iterrows():
                rows.append([
                    r["Dataset"],
                    r["Model"],
                    r["Mode"],
                    f"{r['F1_mean']:.4f}",
                    f"{r['F1_std']:.4f}" if pd.notna(r["F1_std"]) else "nan",
                    f"{r['Accuracy_mean']:.4f}",
                    f"{r['n_features_mean']:.1f}",
                    f"{r['n_syn_mean']:.1f}",
                ])
            logger.write(format_table(rows, headers=["Dataset", "Model", "Mode", "F1_mean", "F1_std", "Acc_mean", "nFeat_mean", "nSyn_mean"]))
        except Exception:
            pass

        if not paired_df.empty:
            logger.write("\n[Paired Wilcoxon tests]")
            rows = []
            for _, r in paired_df.iterrows():
                rows.append([
                    r["Comparison"],
                    str(int(r["n_datasets"])),
                    f"{r['mean_diff']:.4f}",
                    f"{r['median_diff']:.4f}",
                    f"{r['p_value']:.6f}" if pd.notna(r["p_value"]) else "nan",
                ])
            logger.write(format_table(rows, headers=["Comparison", "n", "mean_diff", "median_diff", "p_value"]))

        logger.write("\nSaved files:")
        logger.write(f"  {fold_path}")
        logger.write(f"  {summary_path}")
        logger.write(f"  {paired_path}")
        logger.write(f"  {align_path}")
        logger.write(f"  {log_file}")

    finally:
        logger.close()
