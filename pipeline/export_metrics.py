"""Ekspor folder `metrics/` berisi seluruh metrik evaluasi lengkap (JSON, CSV,
Markdown README, dan grafik PNG resolusi tinggi) untuk versi v001, v002, serta
perbandingan langsung lintas versi.

Jalankan:
  python -m pipeline.export_metrics
"""
if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import shutil
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

import sms_utils
from pipeline.config import (
    FRAUD,
    LABELS,
    PROJECT_ROOT,
    TARGET_NAMES,
    configure_utf8_stdout,
    get_version_config,
)
from pipeline.visualize import plot_metrics_summary, plot_version_comparison

METRICS_ROOT = PROJECT_ROOT / "metrics"
CLASS_IDS = sorted(LABELS.keys())
CLASS_COLORS = {"normal": "#2563EB", "penipuan": "#EF4444", "promo": "#F59E0B"}


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def compute_strategy_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    proba: np.ndarray,
    classes: list[int],
    fraud_label: int = FRAUD,
) -> dict:
    """Menghitung metrik klasifikasi super lengkap untuk satu strategi prediksi."""
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    y_bin_all = label_binarize(y_true, classes=classes)
    fraud_idx = classes.index(fraud_label)

    # Metrik per kelas
    per_class = {}
    roc_auc_per_class = {}
    pr_auc_per_class = {}
    total = int(cm.sum())

    for idx, c in enumerate(classes):
        cname = LABELS[int(c)]
        tp = int(cm[idx, idx])
        fn = int(cm[idx, :].sum() - tp)
        fp = int(cm[:, idx].sum() - tp)
        tn = int(total - tp - fp - fn)
        support = int(tp + fn)

        prec = _safe_div(tp, tp + fp)
        rec = _safe_div(tp, tp + fn)
        spec = _safe_div(tn, tn + fp)
        npv = _safe_div(tn, tn + fn)
        fpr = _safe_div(fp, fp + tn)
        fnr = _safe_div(fn, fn + tp)
        f1 = _safe_div(2 * prec * rec, prec + rec)
        f2 = float(fbeta_score(y_true == c, y_pred == c, beta=2, zero_division=0))

        try:
            r_auc = float(roc_auc_score(y_bin_all[:, idx], proba[:, idx]))
        except ValueError:
            r_auc = float("nan")
        try:
            p_auc = float(average_precision_score(y_bin_all[:, idx], proba[:, idx]))
        except ValueError:
            p_auc = float("nan")

        roc_auc_per_class[cname] = r_auc
        pr_auc_per_class[cname] = p_auc
        per_class[cname] = {
            "class_id": int(c),
            "support": support,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "f2": f2,
            "specificity": spec,
            "npv": npv,
            "fpr": fpr,
            "fnr": fnr,
            "roc_auc": r_auc,
            "pr_auc_ap": p_auc,
        }

    # Metrik global
    try:
        roc_macro = float(roc_auc_score(y_true, proba, multi_class="ovr", average="macro", labels=classes))
        roc_weighted = float(roc_auc_score(y_true, proba, multi_class="ovr", average="weighted", labels=classes))
    except ValueError:
        roc_macro, roc_weighted = float("nan"), float("nan")

    pr_macro = float(np.nanmean(list(pr_auc_per_class.values())))
    ll = float(log_loss(y_true, np.clip(proba, 1e-15, 1.0 - 1e-15), labels=classes))
    brier_fraud = float(brier_score_loss((y_true == fraud_label).astype(int), proba[:, fraud_idx]))

    n_total = int(len(y_true))
    n_correct = int((y_true == y_pred).sum())
    n_errors = n_total - n_correct

    global_metrics = {
        "n_samples": n_total,
        "n_correct": n_correct,
        "n_errors": n_errors,
        "error_rate": _safe_div(n_errors, n_total),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "weighted_recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "matthews_corrcoef": float(matthews_corrcoef(y_true, y_pred)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "log_loss": ll,
        "brier_score_penipuan": brier_fraud,
        "roc_auc_ovr_macro": roc_macro,
        "roc_auc_ovr_weighted": roc_weighted,
        "pr_auc_ap_macro": pr_macro,
    }

    fraud_m = per_class[LABELS[fraud_label]]
    fraud_binary = {
        "tp": fraud_m["tp"],
        "fp": fraud_m["fp"],
        "tn": fraud_m["tn"],
        "fn": fraud_m["fn"],
        "precision_penipuan": fraud_m["precision"],
        "recall_penipuan": fraud_m["recall"],
        "specificity_penipuan": fraud_m["specificity"],
        "npv_penipuan": fraud_m["npv"],
        "fpr_penipuan": fraud_m["fpr"],
        "fnr_penipuan": fraud_m["fnr"],
        "f1_penipuan": fraud_m["f1"],
        "f2_penipuan": fraud_m["f2"],
        "roc_auc_penipuan": fraud_m["roc_auc"],
        "pr_auc_penipuan": fraud_m["pr_auc_ap"],
    }

    return {
        "global": global_metrics,
        "fraud_binary": fraud_binary,
        "per_class": per_class,
        "confusion_matrix_counts": cm.tolist(),
        "confusion_matrix_normalized": np.round(cm_norm, 6).tolist(),
        "labels_order": [LABELS[int(c)] for c in classes],
    }


def evaluate_bundle_complete(
    bundle: dict,
    model_path: Path,
    test_df: pd.DataFrame,
    version: str,
    eval_name: str,
    dataset_label: str,
    split_counts: dict | None = None,
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Menjalankan evaluasi lengkap untuk threshold terpilih dan argmax (0.50)."""
    pipe = bundle["pipeline"]
    threshold = float(bundle["threshold"])
    fraud_label = int(bundle.get("fraud_label", FRAUD))
    classes = [int(c) for c in pipe.classes_]

    y_true = test_df["label"].astype(int).to_numpy()
    proba = pipe.predict_proba(test_df["teks"])
    pred_argmax = np.asarray([classes[i] for i in proba.argmax(axis=1)], dtype=int)
    pred_thr = sms_utils.decide_batch(proba, classes, threshold, fraud_label).astype(int)

    thr_metrics = compute_strategy_metrics(y_true, pred_thr, proba, classes, fraud_label)
    arg_metrics = compute_strategy_metrics(y_true, pred_argmax, proba, classes, fraud_label)

    # Estimasi jumlah fitur pipeline
    n_features = None
    clf = pipe.steps[-1][1]
    if hasattr(clf, "coef_"):
        n_features = int(clf.coef_.shape[1])
    elif hasattr(clf, "feature_log_prob_"):
        n_features = int(clf.feature_log_prob_.shape[1])

    fraud_idx = classes.index(fraud_label)
    err_df = test_df.assign(
        pred_threshold=pred_thr,
        pred_argmax=pred_argmax,
        p_normal=proba[:, classes.index(0)],
        p_penipuan=proba[:, fraud_idx],
        p_promo=proba[:, classes.index(2)],
    )
    err_df = err_df[err_df["label"] != err_df["pred_threshold"]].copy()
    err_df["label_name"] = err_df["label"].map(LABELS)
    err_df["pred_threshold_name"] = err_df["pred_threshold"].map(LABELS)
    err_df["pred_argmax_name"] = err_df["pred_argmax"].map(LABELS)
    err_df = err_df[
        ["teks", "label_name", "pred_threshold_name", "pred_argmax_name", "p_normal", "p_penipuan", "p_promo"]
    ].sort_values("p_penipuan", ascending=False)

    record = {
        "metadata": {
            "version": version,
            "eval_name": eval_name,
            "dataset_label": dataset_label,
            "model_name": bundle.get("model_name"),
            "data_train": bundle.get("data_train"),
            "threshold": threshold,
            "threshold_criterion": bundle.get("threshold_criterion", "f2"),
            "rule_override_apk": bool(bundle.get("rule_override", True)),
            "n_features": n_features,
            "model_size_kb": round(model_path.stat().st_size / 1024, 1) if model_path.exists() else None,
            "sklearn_version": bundle.get("sklearn_version"),
            "created_at": bundle.get("created_at"),
            "exported_at": datetime.now().isoformat(timespec="seconds"),
            "split_counts": split_counts or {},
        },
        "threshold_strategy": thr_metrics,
        "argmax_strategy": arg_metrics,
    }
    return record, y_true, pred_thr, proba, err_df


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def plot_dual_confusion_matrix(
    cm_counts: np.ndarray,
    cm_norm: np.ndarray,
    title: str,
    output_path: Path,
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4), dpi=200, facecolor="#F8FAFC")
    for ax, mat, fmt, sub in [
        (axes[0], cm_counts, "d", "Jumlah Pesan (Counts)"),
        (axes[1], cm_norm * 100, ".1f", "Persentase per Baris / Recall (%)"),
    ]:
        ax.set_facecolor("#FFFFFF")
        im = ax.imshow(mat, cmap="Blues", aspect="auto")
        ax.set_xticks(range(len(TARGET_NAMES)))
        ax.set_yticks(range(len(TARGET_NAMES)))
        ax.set_xticklabels(TARGET_NAMES, fontsize=10, fontweight="bold", color="#0F172A")
        ax.set_yticklabels(TARGET_NAMES, fontsize=10, fontweight="bold", color="#0F172A")
        ax.set_xlabel("Prediksi", fontsize=10, color="#334155")
        ax.set_ylabel("Label Sebenarnya", fontsize=10, color="#334155")
        ax.set_title(sub, fontsize=11, fontweight="bold", color="#0F172A", pad=10)

        thresh = mat.max() * 0.55 if mat.max() > 0 else 0
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat[i, j]
                txt = f"{int(val):,}" if fmt == "d" else f"{val:.1f}%"
                ax.text(
                    j, i, txt, ha="center", va="center",
                    fontsize=11, fontweight="bold",
                    color="white" if val > thresh else "#0F172A",
                )
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(title, fontsize=13.5, fontweight="bold", color="#0F172A", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, facecolor="#F8FAFC")
    plt.close(fig)
    return output_path


def plot_roc_and_pr_curves(
    y_true: np.ndarray,
    proba: np.ndarray,
    classes: list[int],
    title: str,
    output_path: Path,
) -> Path:
    y_bin = label_binarize(y_true, classes=classes)
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.9), dpi=200, facecolor="#F8FAFC")

    for ax in axes:
        ax.set_facecolor("#FFFFFF")
        ax.grid(True, linestyle="--", alpha=0.45, color="#CBD5E1")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # ROC
    axes[0].plot([0, 1], [0, 1], ls="--", color="#94A3B8", lw=1.1, label="Random (AUC = 0.500)")
    for idx, c in enumerate(classes):
        cname = LABELS[int(c)]
        fpr, tpr, _ = roc_curve(y_bin[:, idx], proba[:, idx])
        auc_v = roc_auc_score(y_bin[:, idx], proba[:, idx])
        axes[0].plot(fpr, tpr, lw=2.2, color=CLASS_COLORS[cname], label=f"{cname} (AUC = {auc_v:.4f})")
    axes[0].set_xlabel("False Positive Rate (FPR)", fontsize=10, color="#334155")
    axes[0].set_ylabel("True Positive Rate (Recall)", fontsize=10, color="#334155")
    axes[0].set_title("Kurva ROC (One-vs-Rest)", fontsize=11.5, fontweight="bold", color="#0F172A")
    axes[0].legend(loc="lower right", frameon=True, facecolor="#FFFFFF", edgecolor="#E2E8F0", fontsize=9)

    # Precision-Recall
    for idx, c in enumerate(classes):
        cname = LABELS[int(c)]
        prec, rec, _ = precision_recall_curve(y_bin[:, idx], proba[:, idx])
        ap_v = average_precision_score(y_bin[:, idx], proba[:, idx])
        axes[1].plot(rec, prec, lw=2.2, color=CLASS_COLORS[cname], label=f"{cname} (AP = {ap_v:.4f})")
    axes[1].set_xlabel("Recall", fontsize=10, color="#334155")
    axes[1].set_ylabel("Precision", fontsize=10, color="#334155")
    axes[1].set_title("Kurva Precision-Recall (One-vs-Rest)", fontsize=11.5, fontweight="bold", color="#0F172A")
    axes[1].legend(loc="lower left", frameon=True, facecolor="#FFFFFF", edgecolor="#E2E8F0", fontsize=9)
    axes[1].set_ylim(0.0, 1.04)

    fig.suptitle(title, fontsize=13.5, fontweight="bold", color="#0F172A", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, facecolor="#F8FAFC")
    plt.close(fig)
    return output_path


def plot_per_class_bars(per_class: dict, title: str, output_path: Path) -> Path:
    metrics_cols = [
        ("precision", "Precision", "#2563EB"),
        ("recall", "Recall", "#10B981"),
        ("f1", "F1-Score", "#6366F1"),
        ("specificity", "Specificity", "#F59E0B"),
    ]
    x = np.arange(len(TARGET_NAMES))
    width = 0.19

    fig, ax = plt.subplots(figsize=(10.5, 4.9), dpi=200, facecolor="#F8FAFC")
    ax.set_facecolor("#FFFFFF")

    all_vals = []
    for i, (mkey, mlabel, color) in enumerate(metrics_cols):
        vals = [per_class[cname][mkey] * 100 for cname in TARGET_NAMES]
        all_vals.extend(vals)
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, vals, width, label=mlabel, color=color, zorder=3)
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v + 1.0,
                f"{v:.1f}%",
                ha="center", va="bottom", fontsize=7.8, fontweight="bold", color="#0F172A",
            )

    y_min = 0 if min(all_vals) < 55 else 40 if min(all_vals) < 75 else 60
    ax.set_ylim(y_min, 108)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{c}\n(n = {per_class[c]['support']:,})" for c in TARGET_NAMES],
        fontsize=10.5, fontweight="bold", color="#0F172A",
    )
    ax.set_ylabel("Skor (%)", fontsize=10, color="#334155")
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, color="#CBD5E1", zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(title, fontsize=12.5, fontweight="bold", color="#0F172A", pad=16)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=4, fontsize=9.5)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, facecolor="#F8FAFC")
    plt.close(fig)
    return output_path


def plot_calibration_and_confidence(
    y_true: np.ndarray,
    proba: np.ndarray,
    classes: list[int],
    threshold: float,
    title: str,
    output_path: Path,
) -> Path:
    fraud_idx = classes.index(FRAUD)
    p_fraud = proba[:, fraud_idx]
    y_fraud = (y_true == FRAUD).astype(int)

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.6), dpi=200, facecolor="#F8FAFC")
    for ax in axes:
        ax.set_facecolor("#FFFFFF")
        ax.grid(True, linestyle="--", alpha=0.45, color="#CBD5E1")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Distribusi P(penipuan)
    bins = np.linspace(0, 1, 26)
    axes[0].hist(p_fraud[y_fraud == 0], bins=bins, alpha=0.65, color="#2563EB", label="Bukan Penipuan (normal/promo)", density=True)
    axes[0].hist(p_fraud[y_fraud == 1], bins=bins, alpha=0.65, color="#EF4444", label="Penipuan (label = 1)", density=True)
    axes[0].axvline(threshold, color="#0F172A", ls="--", lw=1.8, label=f"Threshold = {threshold:.2f}")
    axes[0].set_xlabel("Probabilitas Prediksi P(penipuan)", fontsize=10, color="#334155")
    axes[0].set_ylabel("Kepadatan (Density)", fontsize=10, color="#334155")
    axes[0].set_title("Distribusi Keyakinan Model P(penipuan)", fontsize=11.3, fontweight="bold", color="#0F172A")
    axes[0].legend(frameon=True, facecolor="#FFFFFF", edgecolor="#E2E8F0", fontsize=8.6)

    # Kurva kalibrasi (uniform bins agar seluruh rentang probabilitas 0..1 terlihat)
    n_bins = 10
    prob_true, prob_pred = calibration_curve(y_fraud, p_fraud, n_bins=n_bins, strategy="uniform")
    brier = brier_score_loss(y_fraud, p_fraud)
    axes[1].plot([0, 1], [0, 1], ls="--", color="#94A3B8", lw=1.2, label="Kalibrasi Sempurna")
    axes[1].plot(prob_pred, prob_true, marker="o", lw=2.2, color="#EF4444", label=f"Model (Brier = {brier:.4f})")
    axes[1].set_xlabel("Rata-rata Probabilitas Prediksi", fontsize=10, color="#334155")
    axes[1].set_ylabel("Proporsi Aktual Penipuan", fontsize=10, color="#334155")
    axes[1].set_title("Kurva Kalibrasi Kelas Penipuan", fontsize=11.3, fontweight="bold", color="#0F172A")
    axes[1].legend(loc="lower right", frameon=True, facecolor="#FFFFFF", edgecolor="#E2E8F0", fontsize=9)

    fig.suptitle(title, fontsize=13.5, fontweight="bold", color="#0F172A", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, facecolor="#F8FAFC")
    plt.close(fig)
    return output_path


def plot_master_comparison(
    rec_v1: dict,
    rec_v2_bench: dict,
    y_v1: np.ndarray,
    proba_v1: np.ndarray,
    proba_v2_bench: np.ndarray,
    classes: list[int],
    output_path: Path,
) -> Path:
    """Grafik komprehensif v001 vs v002 (threshold & argmax) + overlay PR-Curve penipuan."""
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.4), dpi=210, facecolor="#F8FAFC")
    for ax in axes:
        ax.set_facecolor("#FFFFFF")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Kiri: Perbandingan 3 strategi pada benchmark 162 SMS Indonesia
    metric_defs = [
        ("Accuracy", lambda r: r["global"]["accuracy"] * 100),
        ("Macro F1", lambda r: r["global"]["macro_f1"] * 100),
        ("MCC", lambda r: r["global"]["matthews_corrcoef"] * 100),
        ("Recall\nPenipuan", lambda r: r["fraud_binary"]["recall_penipuan"] * 100),
        ("Precision\nPenipuan", lambda r: r["fraud_binary"]["precision_penipuan"] * 100),
        ("F1\nPenipuan", lambda r: r["fraud_binary"]["f1_penipuan"] * 100),
    ]
    series = [
        (f"v001 (Thr {rec_v1['metadata']['threshold']:.2f})", rec_v1["threshold_strategy"], "#2563EB"),
        (f"v002 (Thr {rec_v2_bench['metadata']['threshold']:.2f})", rec_v2_bench["threshold_strategy"], "#F59E0B"),
        ("v002 (Argmax 0.50)", rec_v2_bench["argmax_strategy"], "#10B981"),
    ]

    x = np.arange(len(metric_defs))
    w = 0.25
    for idx, (slabel, strat, col) in enumerate(series):
        vals = [fn(strat) for _, fn in metric_defs]
        bars = axes[0].bar(x + (idx - 1) * w, vals, w, label=slabel, color=col, zorder=3)
        nudge = (idx - 1) * 0.03
        for bar, v in zip(bars, vals):
            axes[0].text(
                bar.get_x() + bar.get_width() / 2 + nudge,
                v + 0.9,
                f"{v:.1f}",
                ha="center", va="bottom", fontsize=6.9, fontweight="bold", color="#0F172A",
            )

    axes[0].set_ylim(60, 108)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([m for m, _ in metric_defs], fontsize=8.8, fontweight="bold", color="#0F172A")
    axes[0].set_ylabel("Skor (%)", fontsize=9.8, color="#334155")
    axes[0].yaxis.grid(True, linestyle="--", alpha=0.5, color="#CBD5E1", zorder=0)
    axes[0].set_title("Perbandingan Strategi pada Benchmark 162 SMS Indonesia", fontsize=11.3, fontweight="bold", color="#0F172A", pad=26)
    axes[0].legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.09), ncol=3, fontsize=8.3, columnspacing=1.0)

    # Kanan: Overlay PR Curve khusus kelas Penipuan (v001 vs v002)
    fraud_idx = classes.index(FRAUD)
    y_fraud = (y_v1 == FRAUD).astype(int)
    p1, r1, _ = precision_recall_curve(y_fraud, proba_v1[:, fraud_idx])
    p2, r2, _ = precision_recall_curve(y_fraud, proba_v2_bench[:, fraud_idx])
    ap1 = average_precision_score(y_fraud, proba_v1[:, fraud_idx])
    ap2 = average_precision_score(y_fraud, proba_v2_bench[:, fraud_idx])

    axes[1].plot(r1, p1, lw=2.4, color="#2563EB", label=f"v001 LogReg_C1 (AP = {ap1:.4f})")
    axes[1].plot(r2, p2, lw=2.4, color="#EF4444", ls="--", label=f"v002 LogReg_C20 (AP = {ap2:.4f})")

    # Tandai titik operasi threshold masing-masing
    axes[1].scatter(
        [rec_v1["threshold_strategy"]["fraud_binary"]["recall_penipuan"]],
        [rec_v1["threshold_strategy"]["fraud_binary"]["precision_penipuan"]],
        s=75, color="#2563EB", edgecolors="white", lw=1.5, zorder=5,
        label=f"v001 @ thr {rec_v1['metadata']['threshold']:.2f}",
    )
    axes[1].scatter(
        [rec_v2_bench["threshold_strategy"]["fraud_binary"]["recall_penipuan"]],
        [rec_v2_bench["threshold_strategy"]["fraud_binary"]["precision_penipuan"]],
        s=75, color="#F59E0B", edgecolors="white", lw=1.5, zorder=5,
        label=f"v002 @ thr {rec_v2_bench['metadata']['threshold']:.2f}",
    )
    axes[1].scatter(
        [rec_v2_bench["argmax_strategy"]["fraud_binary"]["recall_penipuan"]],
        [rec_v2_bench["argmax_strategy"]["fraud_binary"]["precision_penipuan"]],
        s=75, color="#10B981", edgecolors="white", lw=1.5, zorder=5,
        label="v002 @ argmax 0.50",
    )

    axes[1].set_xlim(0.0, 1.03)
    axes[1].set_ylim(0.5, 1.03)
    axes[1].set_xlabel("Recall Penipuan", fontsize=9.8, color="#334155")
    axes[1].set_ylabel("Precision Penipuan", fontsize=9.8, color="#334155")
    axes[1].grid(True, linestyle="--", alpha=0.45, color="#CBD5E1")
    axes[1].set_title("Precision-Recall Curve Kelas Penipuan (v001 vs v002)", fontsize=11.3, fontweight="bold", color="#0F172A", pad=12)
    axes[1].legend(loc="lower left", frameon=True, facecolor="#FFFFFF", edgecolor="#E2E8F0", fontsize=8.4)

    fig.suptitle("Evaluasi Lintas Versi — Benchmark Test Set v001 (162 SMS Indonesia, Bebas Leakage)",
                 fontsize=13.5, fontweight="bold", color="#0F172A", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=210, facecolor="#F8FAFC")
    plt.close(fig)
    return output_path


def _save_eval_tables(record: dict, err_df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics_full.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Classification report CSV (threshold strategy)
    strat = record["threshold_strategy"]
    rows = []
    for cname in TARGET_NAMES:
        pc = strat["per_class"][cname]
        rows.append({"kelas": cname, **pc})
    rows.append({
        "kelas": "macro_avg",
        "support": strat["global"]["n_samples"],
        "precision": strat["global"]["macro_precision"],
        "recall": strat["global"]["macro_recall"],
        "f1": strat["global"]["macro_f1"],
        "roc_auc": strat["global"]["roc_auc_ovr_macro"],
        "pr_auc_ap": strat["global"]["pr_auc_ap_macro"],
    })
    rows.append({
        "kelas": "weighted_avg",
        "support": strat["global"]["n_samples"],
        "precision": strat["global"]["weighted_precision"],
        "recall": strat["global"]["weighted_recall"],
        "f1": strat["global"]["weighted_f1"],
        "roc_auc": strat["global"]["roc_auc_ovr_weighted"],
    })
    pd.DataFrame(rows).to_csv(out_dir / "classification_report.csv", index=False, encoding="utf-8")

    # Confusion matrix CSV (counts + normalized)
    cm_c = pd.DataFrame(strat["confusion_matrix_counts"], index=TARGET_NAMES, columns=[f"pred_{c}" for c in TARGET_NAMES])
    cm_n = pd.DataFrame(strat["confusion_matrix_normalized"], index=TARGET_NAMES, columns=[f"pred_{c}_pct" for c in TARGET_NAMES])
    cm_both = pd.concat([cm_c, (cm_n * 100).round(2)], axis=1)
    cm_both.index.name = "actual"
    cm_both.to_csv(out_dir / "confusion_matrix.csv", encoding="utf-8")

    # Test errors CSV
    err_df.to_csv(out_dir / "test_errors.csv", index=False, encoding="utf-8")


def _read_split_counts(splits_dir: Path) -> dict:
    out = {}
    for name in ("train", "val", "train_full", "test"):
        p = splits_dir / f"{name}.csv"
        if p.exists():
            df = pd.read_csv(p, usecols=["label"])
            vc = df["label"].map(LABELS).value_counts().to_dict()
            out[name] = {"total": int(len(df)), **{k: int(vc.get(k, 0)) for k in TARGET_NAMES}}
    return out


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _to_dashboard_summary(record: dict) -> dict:
    meta = record["metadata"]
    t_g = record["threshold_strategy"]["global"]
    t_f = record["threshold_strategy"]["fraud_binary"]
    a_g = record["argmax_strategy"]["global"]
    a_f = record["argmax_strategy"]["fraud_binary"]
    return {
        "version": meta["version"],
        "eval_split": meta["eval_name"],
        "dataset_label": meta["dataset_label"],
        "model_name": meta["model_name"],
        "data_train": meta["data_train"],
        "threshold": meta["threshold"],
        "n_test": t_g["n_samples"],
        "n_errors": t_g["n_errors"],
        "metrics_test_threshold": {
            "accuracy": t_g["accuracy"],
            "macro_f1": t_g["macro_f1"],
            "recall_penipuan": t_f["recall_penipuan"],
            "precision_penipuan": t_f["precision_penipuan"],
            "f1_penipuan": t_f["f1_penipuan"],
        },
        "metrics_test_argmax": {
            "accuracy": a_g["accuracy"],
            "macro_f1": a_g["macro_f1"],
            "recall_penipuan": a_f["recall_penipuan"],
            "precision_penipuan": a_f["precision_penipuan"],
            "f1_penipuan": a_f["f1_penipuan"],
        },
    }


def _summary_row(label: str, record: dict, strategy_key: str) -> dict:
    meta = record["metadata"]
    s = record[strategy_key]
    g = s["global"]
    f = s["fraud_binary"]
    return {
        "evaluasi": label,
        "versi": meta["version"],
        "model": meta["model_name"],
        "strategi": f"threshold_{meta['threshold']:.2f}" if strategy_key == "threshold_strategy" else "argmax_0.50",
        "n_test": g["n_samples"],
        "n_errors": g["n_errors"],
        "error_rate": round(g["error_rate"], 6),
        "accuracy": round(g["accuracy"], 6),
        "balanced_accuracy": round(g["balanced_accuracy"], 6),
        "macro_precision": round(g["macro_precision"], 6),
        "macro_recall": round(g["macro_recall"], 6),
        "macro_f1": round(g["macro_f1"], 6),
        "weighted_f1": round(g["weighted_f1"], 6),
        "mcc": round(g["matthews_corrcoef"], 6),
        "cohen_kappa": round(g["cohen_kappa"], 6),
        "log_loss": round(g["log_loss"], 6),
        "brier_penipuan": round(g["brier_score_penipuan"], 6),
        "roc_auc_macro": round(g["roc_auc_ovr_macro"], 6),
        "pr_auc_macro": round(g["pr_auc_ap_macro"], 6),
        "recall_penipuan": round(f["recall_penipuan"], 6),
        "precision_penipuan": round(f["precision_penipuan"], 6),
        "specificity_penipuan": round(f["specificity_penipuan"], 6),
        "fpr_penipuan": round(f["fpr_penipuan"], 6),
        "fnr_penipuan": round(f["fnr_penipuan"], 6),
        "f1_penipuan": round(f["f1_penipuan"], 6),
        "f2_penipuan": round(f["f2_penipuan"], 6),
        "roc_auc_penipuan": round(f["roc_auc_penipuan"], 6),
        "pr_auc_penipuan": round(f["pr_auc_penipuan"], 6),
    }


def _write_readme(
    out_root: Path,
    summary_df: pd.DataFrame,
    rec_v1: dict,
    rec_v2: dict,
    rec_v2_bench: dict,
) -> Path:
    def pct(v: float) -> str:
        return f"{v * 100:.2f}%"

    def per_class_table(per_class: dict) -> str:
        lines = [
            "| Kelas | Support | TP | FP | FN | Precision | Recall | F1-Score | Specificity | ROC-AUC | PR-AUC (AP) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for cname in TARGET_NAMES:
            pc = per_class[cname]
            lines.append(
                f"| **{cname}** | {pc['support']:,} | {pc['tp']:,} | {pc['fp']:,} | {pc['fn']:,} | "
                f"{pct(pc['precision'])} | {pct(pc['recall'])} | {pct(pc['f1'])} | "
                f"{pct(pc['specificity'])} | {pc['roc_auc']:.4f} | {pc['pr_auc_ap']:.4f} |"
            )
        return "\n".join(lines)

    def cm_table(strat: dict) -> str:
        cm_c = strat["confusion_matrix_counts"]
        cm_n = strat["confusion_matrix_normalized"]
        lines = [
            "| Aktual \\ Prediksi | normal | penipuan | promo | Recall (%) |",
            "|---|---:|---:|---:|---:|",
        ]
        for i, cname in enumerate(TARGET_NAMES):
            lines.append(
                f"| **{cname}** | {cm_c[i][0]:,} | {cm_c[i][1]:,} | {cm_c[i][2]:,} | {cm_n[i][i] * 100:.2f}% |"
            )
        return "\n".join(lines)

    v1_t = rec_v1["threshold_strategy"]
    v2_t = rec_v2["threshold_strategy"]
    v2b_t = rec_v2_bench["threshold_strategy"]
    v2b_a = rec_v2_bench["argmax_strategy"]

    md = f"""# Laporan Metrik Lengkap — Detektor Pesan Penipuan SMS/WA

Dibuat otomatis pada: `{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}`

Folder ini menghimpun seluruh metrik evaluasi lengkap (global, per-kelas, biner kelas penipuan, probabilistik ROC-AUC / PR-AUC / Log Loss / Brier Score, MCC, Cohen's Kappa, Confusion Matrix, analisis error, dan grafik PNG) untuk **v001** dan **v002**.

---

## 1. Ringkasan Perbandingan Utama

| Metrik | v001 (`LogReg_C1`, thr {rec_v1['metadata']['threshold']:.2f})<br>Test v001 (`n=162`) | v002 (`LogReg_C20`, thr {rec_v2['metadata']['threshold']:.2f})<br>Test Gabungan (`n=4.892`) | v002 (`LogReg_C20`, thr {rec_v2_bench['metadata']['threshold']:.2f})<br>Benchmark v001 (`n=162`) | v002 (`LogReg_C20`, argmax 0.50)<br>Benchmark v001 (`n=162`) |
|---|---:|---:|---:|---:|
| **Accuracy** | {pct(v1_t['global']['accuracy'])} | {pct(v2_t['global']['accuracy'])} | {pct(v2b_t['global']['accuracy'])} | {pct(v2b_a['global']['accuracy'])} |
| **Balanced Accuracy** | {pct(v1_t['global']['balanced_accuracy'])} | {pct(v2_t['global']['balanced_accuracy'])} | {pct(v2b_t['global']['balanced_accuracy'])} | {pct(v2b_a['global']['balanced_accuracy'])} |
| **Macro F1** | {pct(v1_t['global']['macro_f1'])} | {pct(v2_t['global']['macro_f1'])} | {pct(v2b_t['global']['macro_f1'])} | {pct(v2b_a['global']['macro_f1'])} |
| **Weighted F1** | {pct(v1_t['global']['weighted_f1'])} | {pct(v2_t['global']['weighted_f1'])} | {pct(v2b_t['global']['weighted_f1'])} | {pct(v2b_a['global']['weighted_f1'])} |
| **Matthews Corr Coef (MCC)** | {v1_t['global']['matthews_corrcoef']:.4f} | {v2_t['global']['matthews_corrcoef']:.4f} | {v2b_t['global']['matthews_corrcoef']:.4f} | {v2b_a['global']['matthews_corrcoef']:.4f} |
| **Cohen's Kappa** | {v1_t['global']['cohen_kappa']:.4f} | {v2_t['global']['cohen_kappa']:.4f} | {v2b_t['global']['cohen_kappa']:.4f} | {v2b_a['global']['cohen_kappa']:.4f} |
| **ROC-AUC (OvR Macro)** | {v1_t['global']['roc_auc_ovr_macro']:.4f} | {v2_t['global']['roc_auc_ovr_macro']:.4f} | {v2b_t['global']['roc_auc_ovr_macro']:.4f} | {v2b_a['global']['roc_auc_ovr_macro']:.4f} |
| **PR-AUC / AP (Macro)** | {v1_t['global']['pr_auc_ap_macro']:.4f} | {v2_t['global']['pr_auc_ap_macro']:.4f} | {v2b_t['global']['pr_auc_ap_macro']:.4f} | {v2b_a['global']['pr_auc_ap_macro']:.4f} |
| **Log Loss** | {v1_t['global']['log_loss']:.4f} | {v2_t['global']['log_loss']:.4f} | {v2b_t['global']['log_loss']:.4f} | {v2b_a['global']['log_loss']:.4f} |
| **Brier Score (Penipuan)** | {v1_t['global']['brier_score_penipuan']:.4f} | {v2_t['global']['brier_score_penipuan']:.4f} | {v2b_t['global']['brier_score_penipuan']:.4f} | {v2b_a['global']['brier_score_penipuan']:.4f} |
| **Recall Penipuan (TPR)** | {pct(v1_t['fraud_binary']['recall_penipuan'])} | {pct(v2_t['fraud_binary']['recall_penipuan'])} | {pct(v2b_t['fraud_binary']['recall_penipuan'])} | {pct(v2b_a['fraud_binary']['recall_penipuan'])} |
| **Precision Penipuan (PPV)** | {pct(v1_t['fraud_binary']['precision_penipuan'])} | {pct(v2_t['fraud_binary']['precision_penipuan'])} | {pct(v2b_t['fraud_binary']['precision_penipuan'])} | {pct(v2b_a['fraud_binary']['precision_penipuan'])} |
| **Specificity Penipuan (TNR)** | {pct(v1_t['fraud_binary']['specificity_penipuan'])} | {pct(v2_t['fraud_binary']['specificity_penipuan'])} | {pct(v2b_t['fraud_binary']['specificity_penipuan'])} | {pct(v2b_a['fraud_binary']['specificity_penipuan'])} |
| **False Negative Rate (FNR Penipuan)** | {pct(v1_t['fraud_binary']['fnr_penipuan'])} | {pct(v2_t['fraud_binary']['fnr_penipuan'])} | {pct(v2b_t['fraud_binary']['fnr_penipuan'])} | {pct(v2b_a['fraud_binary']['fnr_penipuan'])} |
| **False Positive Rate (FPR Penipuan)** | {pct(v1_t['fraud_binary']['fpr_penipuan'])} | {pct(v2_t['fraud_binary']['fpr_penipuan'])} | {pct(v2b_t['fraud_binary']['fpr_penipuan'])} | {pct(v2b_a['fraud_binary']['fpr_penipuan'])} |
| **F1 Penipuan** | {pct(v1_t['fraud_binary']['f1_penipuan'])} | {pct(v2_t['fraud_binary']['f1_penipuan'])} | {pct(v2b_t['fraud_binary']['f1_penipuan'])} | {pct(v2b_a['fraud_binary']['f1_penipuan'])} |
| **F2 Penipuan** | {pct(v1_t['fraud_binary']['f2_penipuan'])} | {pct(v2_t['fraud_binary']['f2_penipuan'])} | {pct(v2b_t['fraud_binary']['f2_penipuan'])} | {pct(v2b_a['fraud_binary']['f2_penipuan'])} |
| **Jumlah Error** | {v1_t['global']['n_errors']} / {v1_t['global']['n_samples']} | {v2_t['global']['n_errors']} / {v2_t['global']['n_samples']} | {v2b_t['global']['n_errors']} / {v2b_t['global']['n_samples']} | {v2b_a['global']['n_errors']} / {v2b_a['global']['n_samples']} |

### Grafik Perbandingan Lintas Versi

![Perbandingan Strategi & PR Curve](comparison/comparison_all_strategies.png)
![Perbandingan v001 vs v002](comparison/comparison_v001_vs_v002.png)

---

## 2. Detail v001 (`dataset_sms_spam_v1.csv`, `n = 162`)

### Metrik per Kelas (v001)
{per_class_table(v1_t['per_class'])}

### Confusion Matrix (v001)
{cm_table(v1_t)}

![Dashboard v001](v001/plots/dashboard_ringkasan.png)
![Confusion Matrix v001](v001/plots/confusion_matrix.png)
![ROC & PR Curves v001](v001/plots/roc_pr_curves.png)
![Per Class Metrics v001](v001/plots/per_class_metrics.png)
![Calibration & Confidence v001](v001/plots/calibration_confidence.png)

---

## 3. Detail v002 (`dataset_sms_spam_v1.csv` + `SpamDam_twitter_data.csv`, `n = 4.892`)

### Metrik per Kelas (v002 — Test Gabungan)
{per_class_table(v2_t['per_class'])}

### Confusion Matrix (v002 — Test Gabungan)
{cm_table(v2_t)}

![Dashboard v002](v002/plots/dashboard_ringkasan.png)
![Confusion Matrix v002](v002/plots/confusion_matrix.png)
![ROC & PR Curves v002](v002/plots/roc_pr_curves.png)
![Per Class Metrics v002](v002/plots/per_class_metrics.png)
![Calibration & Confidence v002](v002/plots/calibration_confidence.png)

### Benchmark v002 pada Test Set v001 (`n = 162`, Bebas Leakage)
{per_class_table(v2b_t['per_class'])}

![Dashboard Benchmark v001](v002/plots/dashboard_benchmark_v001.png)
![Confusion Matrix Benchmark v001](v002/plots/confusion_matrix_benchmark_v001.png)
![ROC & PR Curves Benchmark v001](v002/plots/roc_pr_curves_benchmark_v001.png)

---

## 4. Daftar Isi Folder `metrics/`

- `all_metrics_complete.json` — Master JSON berisi seluruh metrik lengkap v001, v002 (test gabungan), dan v002 (benchmark v001).
- `summary_table.csv` — Tabel ringkas lintas versi & strategi (threshold & argmax).
- `per_class_metrics.csv` — Precision, Recall, F1, F2, Specificity, FPR, FNR, ROC-AUC, PR-AUC untuk tiap kelas (`normal`, `penipuan`, `promo`).
- `comparison_v001_vs_v002.csv` — Tabel selisih (delta) langsung v001 vs v002 pada benchmark 162 SMS Indonesia.
- `v001/` — JSON, CSV, analisis error, dan 6 grafik PNG untuk v001.
- `v002/` — JSON, CSV, analisis error, folder `benchmark_v001/`, dan 9 grafik PNG untuk v002.
- `comparison/` — JSON & 2 grafik PNG perbandingan langsung lintas versi.
"""
    readme_path = out_root / "README.md"
    readme_path.write_text(md, encoding="utf-8")
    return readme_path


def export_all_metrics(out_root: Path = METRICS_ROOT) -> dict:
    """Menghitung dan mengekspor seluruh metrik lengkap ke folder `metrics/`."""
    out_root.mkdir(parents=True, exist_ok=True)
    cfg_v1 = get_version_config("v001")
    cfg_v2 = get_version_config("v002")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InconsistentVersionWarning)
        bundle_v1 = joblib.load(cfg_v1["model_path"])
        bundle_v2 = joblib.load(cfg_v2["model_path"])

    test_v1 = pd.read_csv(cfg_v1["splits_dir"] / "test.csv", encoding="utf-8")
    test_v1["teks"] = test_v1["teks"].astype(str)
    test_v1["label"] = test_v1["label"].astype(int)

    test_v2 = pd.read_csv(cfg_v2["splits_dir"] / "test.csv", encoding="utf-8")
    test_v2["teks"] = test_v2["teks"].astype(str)
    test_v2["label"] = test_v2["label"].astype(int)

    classes = [int(c) for c in bundle_v2["pipeline"].classes_]

    # 1. Evaluasi lengkap v001
    print("Menghitung metrik lengkap v001...")
    rec_v1, y_v1, _, proba_v1, err_v1 = evaluate_bundle_complete(
        bundle=bundle_v1,
        model_path=cfg_v1["model_path"],
        test_df=test_v1,
        version="v001",
        eval_name="test set v001 (162 SMS)",
        dataset_label=cfg_v1["dataset_label"],
        split_counts=_read_split_counts(cfg_v1["splits_dir"]),
    )
    v1_dir = out_root / "v001"
    v1_plots = v1_dir / "plots"
    _save_eval_tables(rec_v1, err_v1, v1_dir)
    _copy_if_exists(cfg_v1["reports_dir"] / "model_comparison.csv", v1_dir / "model_comparison.csv")
    _copy_if_exists(cfg_v1["reports_dir"] / "threshold_curve.csv", v1_dir / "threshold_curve.csv")
    _copy_if_exists(cfg_v1["plots_dir"] / "threshold_curve.png", v1_plots / "threshold_curve.png")

    plot_metrics_summary(_to_dashboard_summary(rec_v1), output_path=v1_plots / "dashboard_ringkasan.png")
    plot_dual_confusion_matrix(
        np.asarray(rec_v1["threshold_strategy"]["confusion_matrix_counts"]),
        np.asarray(rec_v1["threshold_strategy"]["confusion_matrix_normalized"]),
        f"Confusion Matrix v001 ({rec_v1['metadata']['model_name']}, thr {rec_v1['metadata']['threshold']:.2f})",
        v1_plots / "confusion_matrix.png",
    )
    plot_roc_and_pr_curves(y_v1, proba_v1, classes, "Kurva ROC & Precision-Recall — v001 (Test Set 162 SMS)", v1_plots / "roc_pr_curves.png")
    plot_per_class_bars(rec_v1["threshold_strategy"]["per_class"], "Metrik Lengkap per Kelas — v001 (Test Set 162 SMS)", v1_plots / "per_class_metrics.png")
    plot_calibration_and_confidence(
        y_v1, proba_v1, classes, rec_v1["metadata"]["threshold"],
        "Distribusi Probabilitas & Kalibrasi — v001", v1_plots / "calibration_confidence.png",
    )

    # 2. Evaluasi lengkap v002 (test gabungan)
    print("Menghitung metrik lengkap v002 (test gabungan 4.892 pesan)...")
    rec_v2, y_v2, _, proba_v2, err_v2 = evaluate_bundle_complete(
        bundle=bundle_v2,
        model_path=cfg_v2["model_path"],
        test_df=test_v2,
        version="v002",
        eval_name="test gabungan v002 (4.892 pesan)",
        dataset_label=cfg_v2["dataset_label"],
        split_counts=_read_split_counts(cfg_v2["splits_dir"]),
    )
    v2_dir = out_root / "v002"
    v2_plots = v2_dir / "plots"
    _save_eval_tables(rec_v2, err_v2, v2_dir)
    _copy_if_exists(cfg_v2["reports_dir"] / "model_comparison.csv", v2_dir / "model_comparison.csv")
    _copy_if_exists(cfg_v2["reports_dir"] / "threshold_curve.csv", v2_dir / "threshold_curve.csv")
    _copy_if_exists(cfg_v2["plots_dir"] / "threshold_curve.png", v2_plots / "threshold_curve.png")

    plot_metrics_summary(_to_dashboard_summary(rec_v2), output_path=v2_plots / "dashboard_ringkasan.png")
    plot_dual_confusion_matrix(
        np.asarray(rec_v2["threshold_strategy"]["confusion_matrix_counts"]),
        np.asarray(rec_v2["threshold_strategy"]["confusion_matrix_normalized"]),
        f"Confusion Matrix v002 ({rec_v2['metadata']['model_name']}, thr {rec_v2['metadata']['threshold']:.2f})",
        v2_plots / "confusion_matrix.png",
    )
    plot_roc_and_pr_curves(y_v2, proba_v2, classes, "Kurva ROC & Precision-Recall — v002 (Test Gabungan 4.892 Pesan)", v2_plots / "roc_pr_curves.png")
    plot_per_class_bars(rec_v2["threshold_strategy"]["per_class"], "Metrik Lengkap per Kelas — v002 (Test Gabungan 4.892 Pesan)", v2_plots / "per_class_metrics.png")
    plot_calibration_and_confidence(
        y_v2, proba_v2, classes, rec_v2["metadata"]["threshold"],
        "Distribusi Probabilitas & Kalibrasi — v002", v2_plots / "calibration_confidence.png",
    )

    # 3. Evaluasi lengkap v002 pada benchmark v001 (162 SMS Indonesia)
    print("Menghitung metrik lengkap v002 pada benchmark test set v001 (162 SMS)...")
    rec_v2_bench, y_v2b, _, proba_v2b, err_v2b = evaluate_bundle_complete(
        bundle=bundle_v2,
        model_path=cfg_v2["model_path"],
        test_df=test_v1,
        version="v002",
        eval_name="benchmark test set v001 (162 SMS)",
        dataset_label="Benchmark Test Set v001 (dataset_sms_spam_v1.csv, 162 SMS)",
        split_counts=_read_split_counts(cfg_v1["splits_dir"]),
    )
    v2b_dir = v2_dir / "benchmark_v001"
    _save_eval_tables(rec_v2_bench, err_v2b, v2b_dir)

    plot_metrics_summary(_to_dashboard_summary(rec_v2_bench), output_path=v2_plots / "dashboard_benchmark_v001.png")
    plot_dual_confusion_matrix(
        np.asarray(rec_v2_bench["threshold_strategy"]["confusion_matrix_counts"]),
        np.asarray(rec_v2_bench["threshold_strategy"]["confusion_matrix_normalized"]),
        f"Confusion Matrix v002 pada Benchmark v001 (thr {rec_v2_bench['metadata']['threshold']:.2f})",
        v2_plots / "confusion_matrix_benchmark_v001.png",
    )
    plot_roc_and_pr_curves(
        y_v2b, proba_v2b, classes,
        "Kurva ROC & Precision-Recall — v002 pada Benchmark v001 (162 SMS)",
        v2_plots / "roc_pr_curves_benchmark_v001.png",
    )

    # 4. Tabel gabungan & perbandingan lintas versi
    print("Menyusun tabel gabungan & grafik perbandingan lintas versi...")
    summary_rows = [
        _summary_row("v001_test_162", rec_v1, "threshold_strategy"),
        _summary_row("v001_test_162", rec_v1, "argmax_strategy"),
        _summary_row("v002_test_gabungan_4892", rec_v2, "threshold_strategy"),
        _summary_row("v002_test_gabungan_4892", rec_v2, "argmax_strategy"),
        _summary_row("v002_benchmark_v001_162", rec_v2_bench, "threshold_strategy"),
        _summary_row("v002_benchmark_v001_162", rec_v2_bench, "argmax_strategy"),
    ]
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_root / "summary_table.csv", index=False, encoding="utf-8")

    # Per-class CSV gabungan
    pc_rows = []
    for eval_tag, rec in [
        ("v001_test_162", rec_v1),
        ("v002_test_gabungan_4892", rec_v2),
        ("v002_benchmark_v001_162", rec_v2_bench),
    ]:
        for strat_key, strat_label in [("threshold_strategy", "threshold"), ("argmax_strategy", "argmax_0.50")]:
            for cname in TARGET_NAMES:
                pc = rec[strat_key]["per_class"][cname]
                pc_rows.append({
                    "evaluasi": eval_tag,
                    "versi": rec["metadata"]["version"],
                    "strategi": strat_label,
                    "kelas": cname,
                    **pc,
                })
    pd.DataFrame(pc_rows).to_csv(out_root / "per_class_metrics.csv", index=False, encoding="utf-8")

    # Perbandingan langsung v001 vs v002 pada benchmark 162 SMS
    comp_dir = out_root / "comparison"
    comp_dir.mkdir(parents=True, exist_ok=True)
    comp_keys = [
        ("accuracy", "global"),
        ("balanced_accuracy", "global"),
        ("macro_f1", "global"),
        ("weighted_f1", "global"),
        ("matthews_corrcoef", "global"),
        ("cohen_kappa", "global"),
        ("roc_auc_ovr_macro", "global"),
        ("pr_auc_ap_macro", "global"),
        ("log_loss", "global"),
        ("brier_score_penipuan", "global"),
        ("recall_penipuan", "fraud_binary"),
        ("precision_penipuan", "fraud_binary"),
        ("specificity_penipuan", "fraud_binary"),
        ("fnr_penipuan", "fraud_binary"),
        ("fpr_penipuan", "fraud_binary"),
        ("f1_penipuan", "fraud_binary"),
        ("f2_penipuan", "fraud_binary"),
    ]
    comp_rows = []
    for mname, group_key in comp_keys:
        v1_val = rec_v1["threshold_strategy"][group_key][mname]
        v2_thr_val = rec_v2_bench["threshold_strategy"][group_key][mname]
        v2_arg_val = rec_v2_bench["argmax_strategy"][group_key][mname]
        comp_rows.append({
            "metrik": mname,
            "v001_thr_0.19": round(v1_val, 6),
            "v002_thr_0.20": round(v2_thr_val, 6),
            "delta_v002_thr_vs_v001": round(v2_thr_val - v1_val, 6),
            "v002_argmax_0.50": round(v2_arg_val, 6),
            "delta_v002_argmax_vs_v001": round(v2_arg_val - v1_val, 6),
        })
    pd.DataFrame(comp_rows).to_csv(out_root / "comparison_v001_vs_v002.csv", index=False, encoding="utf-8")

    plot_version_comparison(
        v001_summary=_to_dashboard_summary(rec_v1),
        v002_bench_summary=_to_dashboard_summary(rec_v2_bench),
        output_path=comp_dir / "comparison_v001_vs_v002.png",
    )
    plot_master_comparison(
        rec_v1=rec_v1,
        rec_v2_bench=rec_v2_bench,
        y_v1=y_v1,
        proba_v1=proba_v1,
        proba_v2_bench=proba_v2b,
        classes=classes,
        output_path=comp_dir / "comparison_all_strategies.png",
    )

    comp_json = {
        "benchmark_dataset": "outputs/splits/test.csv (162 SMS Indonesia, bebas data leakage)",
        "rows": comp_rows,
    }
    (comp_dir / "comparison_v001_vs_v002.json").write_text(
        json.dumps(comp_json, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Master JSON
    master = {
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "v001": rec_v1,
        "v002_test_gabungan": rec_v2,
        "v002_benchmark_v001": rec_v2_bench,
        "comparison_v001_vs_v002": comp_rows,
    }
    (out_root / "all_metrics_complete.json").write_text(
        json.dumps(master, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    _write_readme(out_root, summary_df, rec_v1, rec_v2, rec_v2_bench)
    print(f"\nSeluruh metrik lengkap berhasil diekspor ke: {out_root}")
    return master


if __name__ == "__main__":
    configure_utf8_stdout()
    export_all_metrics()
