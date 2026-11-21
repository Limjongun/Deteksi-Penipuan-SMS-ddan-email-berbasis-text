"""Tahap 6 — Training, perbandingan kandidat model, tuning threshold,
retraining model final pada train+val, dan penyimpanan bundle `.joblib`.
"""
if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import f1_score, fbeta_score, precision_score, recall_score

import sms_utils
from pipeline.augment import augment_train
from pipeline.config import (
    DATA_CITATION,
    DATA_LICENSE,
    DEFAULT_VERSION,
    FRAUD,
    LABELS,
    MODEL_PATH,
    PLOTS_DIR,
    REPORTS_DIR,
    SEED,
    SPLITS_DIR,
    TARGET_NAMES,
    THRESHOLD_MAX,
    THRESHOLD_MAX_IMBALANCED,
    THRESHOLD_MIN,
    THRESHOLD_STEP,
    configure_utf8_stdout,
    ensure_dirs,
    get_version_config,
    set_seed,
)
from pipeline.features import compute_metrics, get_candidates
from pipeline.preprocess import load_splits


def compare_models(
    train_df: pd.DataFrame,
    train_aug: pd.DataFrame,
    val_df: pd.DataFrame,
    seed: int = SEED,
) -> tuple[pd.DataFrame, dict]:
    """Melatih semua kandidat model pada data asli dan asli+augmentasi, lalu evaluasi di val."""
    candidates = get_candidates(seed=seed)
    train_sets = {"asli": train_df, "asli+augmentasi": train_aug}
    hasil = []
    fitted = {}

    for data_name, tr in train_sets.items():
        for model_name, factory in candidates.items():
            t0 = time.time()
            model = factory().fit(tr["teks"], tr["label"])
            pred = model.predict(val_df["teks"])
            fitted[(model_name, data_name)] = model
            elapsed = round(time.time() - t0, 2)
            row = {
                "model": model_name,
                "data_train": data_name,
                **compute_metrics(val_df["label"], pred),
                "detik": elapsed,
            }
            hasil.append(row)
            print(
                f"  [{data_name:15s}] {model_name:10s} -> "
                f"macro_f1={row['macro_f1']:.3f} | rec_penipuan={row['recall_penipuan']:.3f} ({elapsed:.1f}s)"
            )

    hasil_df = (
        pd.DataFrame(hasil)
        .sort_values(["macro_f1", "recall_penipuan"], ascending=False)
        .reset_index(drop=True)
    )
    print("\n=== Perbandingan Kandidat Model (Validation Set) ===")
    print(hasil_df.round(3).to_string(index=False))
    return hasil_df, fitted


def select_best_model(hasil_df: pd.DataFrame, fitted: dict, seed: int = SEED):
    """Memilih model terbaik yang mendukung predict_proba."""
    candidates = get_candidates(seed=seed)
    punya_proba = hasil_df[hasil_df["model"].map(lambda m: hasattr(candidates[m](), "predict_proba"))]
    best_row = punya_proba.iloc[0]
    best_model_name = str(best_row["model"])
    best_data_name = str(best_row["data_train"])
    best_model = fitted[(best_model_name, best_data_name)]
    print(f"\nModel terpilih: {best_model_name} (data train: {best_data_name})")
    return best_model_name, best_data_name, best_model


def tune_threshold(
    model,
    val_df: pd.DataFrame,
    fraud_label: int = FRAUD,
    plots_dir: Path = PLOTS_DIR,
    reports_dir: Path = REPORTS_DIR,
    save_outputs: bool = True,
) -> tuple[float, pd.DataFrame, str]:
    """Mencari threshold P(penipuan) optimal pada validation set.

    - Jika proporsi penipuan <= 70% (seperti v001): memaksimalkan F2-score kelas penipuan.
    - Jika proporsi penipuan > 70% (dataset sangat timpang seperti v002): memaksimalkan
      Macro F1 (tie-break dengan F2 penipuan) agar kelas normal & promo tetap terjaga.
    """
    classes = list(model.classes_)
    proba_val = model.predict_proba(val_df["teks"])
    fraud_ratio = float((val_df["label"] == fraud_label).mean())

    imbalanced_fraud = fraud_ratio > 0.70
    thr_max = THRESHOLD_MAX_IMBALANCED if imbalanced_fraud else THRESHOLD_MAX
    thresholds = np.round(np.arange(THRESHOLD_MIN, thr_max + THRESHOLD_STEP / 2, THRESHOLD_STEP), 2)

    kurva = []
    for thr in thresholds:
        pred = sms_utils.decide_batch(proba_val, classes, float(thr), fraud_label)
        y_bin = val_df["label"] == fraud_label
        p_bin = pred == fraud_label
        kurva.append({
            "threshold": float(thr),
            "precision": float(precision_score(y_bin, p_bin, zero_division=0)),
            "recall": float(recall_score(y_bin, p_bin, zero_division=0)),
            "f2": float(fbeta_score(y_bin, p_bin, beta=2, zero_division=0)),
            "macro_f1": float(f1_score(val_df["label"], pred, average="macro")),
        })

    kurva_df = pd.DataFrame(kurva)
    if imbalanced_fraud:
        criterion = "macro_f1"
        top_mask = np.isclose(kurva_df["macro_f1"], kurva_df["macro_f1"].max(), atol=1e-4)
        top_df = kurva_df[top_mask]
        terbaik = top_df[np.isclose(top_df["f2"], top_df["f2"].max(), atol=1e-4)]["threshold"]
    else:
        criterion = "f2"
        terbaik = kurva_df[np.isclose(kurva_df["f2"], kurva_df["f2"].max())]["threshold"]

    threshold = float(np.round(terbaik.median(), 2))

    print(f"Threshold penipuan terpilih: {threshold:.2f} (kriteria: {criterion}, proporsi fraud val: {fraud_ratio:.1%})")
    sample_thrs = sorted({0.30, 0.40, 0.50, threshold})
    ringkas_thr = kurva_df[kurva_df["threshold"].isin(sample_thrs)]
    print(ringkas_thr.round(3).to_string(index=False))

    if save_outputs:
        plots_dir.mkdir(parents=True, exist_ok=True)
        reports_dir.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(7.4, 3.8))
        for col in ("precision", "recall", "f2", "macro_f1"):
            ax.plot(kurva_df["threshold"], kurva_df[col], label=col)
        ax.axvline(threshold, ls="--", c="gray", label=f"terpilih ({threshold:.2f})")
        ax.set_xlabel("threshold P(penipuan)")
        ax.set_title("Kurva metrik di validation set")
        ax.legend(loc="best")
        plt.tight_layout()

        plot_path = plots_dir / "threshold_curve.png"
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)

        kurva_csv = reports_dir / "threshold_curve.csv"
        kurva_df.to_csv(kurva_csv, index=False)
        print(f"Grafik threshold tersimpan di : {plot_path}")
        print(f"Data kurva tersimpan di       : {kurva_csv}")

    return threshold, kurva_df, criterion


def train_final_model(
    train_full: pd.DataFrame,
    best_model_name: str,
    best_data_name: str,
    seed: int = SEED,
):
    """Melatih ulang model terbaik pada seluruh data non-test (train + validation)."""
    final_train = augment_train(train_full, seed=seed) if best_data_name == "asli+augmentasi" else train_full
    candidates = get_candidates(seed=seed)
    final_model = candidates[best_model_name]().fit(final_train["teks"], final_train["label"])
    return final_model, final_train


def build_bundle(
    final_model,
    threshold: float,
    best_model_name: str,
    best_data_name: str,
    version: str = DEFAULT_VERSION,
    citation: str = DATA_CITATION,
    metrics_test: dict | None = None,
    rule_override: bool = True,
) -> dict:
    """Menyusun dictionary bundle model yang siap disimpan dengan joblib."""
    return {
        "version": version,
        "pipeline": final_model,
        "threshold": threshold,
        "labels": LABELS,
        "fraud_label": FRAUD,
        "rule_override": rule_override,
        "model_name": best_model_name,
        "data_train": best_data_name,
        "metrics_test": metrics_test,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "sumber_data": citation,
        "lisensi_data": DATA_LICENSE,
    }


def save_bundle(bundle: dict, model_path: Path = MODEL_PATH) -> Path:
    """Menyimpan bundle model ke .joblib (terkompresi) dan melakukan uji load ulang."""
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path, compress=3)
    size_kb = model_path.stat().st_size / 1024
    print(f"\nModel tersimpan: {model_path} ({size_kb:.0f} KB)")

    # Uji load ulang persis seperti yang dilakukan bot/CLI
    loaded = joblib.load(model_path)
    uji = sms_utils.analyze(loaded, "Selamat anda menang undian, kirim pulsa 100rb untuk klaim")
    print(f"Verifikasi reload model OK -> contoh uji diprediksi: {uji['label_name']}")
    return model_path


def run_training(
    splits: dict[str, pd.DataFrame],
    seed: int = SEED,
    version: str = DEFAULT_VERSION,
    citation: str = DATA_CITATION,
    model_path: Path = MODEL_PATH,
    plots_dir: Path = PLOTS_DIR,
    reports_dir: Path = REPORTS_DIR,
    save_model: bool = True,
    save_outputs: bool = True,
) -> dict:
    """Menjalankan keseluruhan tahap training (augment -> compare -> tune -> retrain)."""
    train_df = splits["train"]
    val_df = splits["val"]
    train_full = splits["train_full"]

    train_aug = augment_train(train_df, seed=seed)
    print(
        "Train asli      :",
        train_df["label"].map(LABELS).value_counts().reindex(TARGET_NAMES).fillna(0).astype(int).to_dict(),
    )
    print(
        "Train + augment :",
        train_aug["label"].map(LABELS).value_counts().reindex(TARGET_NAMES).fillna(0).astype(int).to_dict(),
    )

    hasil_df, fitted = compare_models(train_df, train_aug, val_df, seed=seed)
    if save_outputs:
        reports_dir.mkdir(parents=True, exist_ok=True)
        hasil_df.to_csv(reports_dir / "model_comparison.csv", index=False)

    best_model_name, best_data_name, best_val_model = select_best_model(hasil_df, fitted, seed=seed)
    threshold, kurva_df, threshold_criterion = tune_threshold(
        best_val_model,
        val_df,
        fraud_label=FRAUD,
        plots_dir=plots_dir,
        reports_dir=reports_dir,
        save_outputs=save_outputs,
    )

    final_model, _ = train_final_model(train_full, best_model_name, best_data_name, seed=seed)
    bundle = build_bundle(
        final_model=final_model,
        threshold=threshold,
        best_model_name=best_model_name,
        best_data_name=best_data_name,
        version=version,
        citation=citation,
        metrics_test=None,
        rule_override=True,
    )
    bundle["threshold_criterion"] = threshold_criterion

    if save_model:
        save_bundle(bundle, model_path=model_path)

    return {
        "bundle": bundle,
        "hasil_df": hasil_df,
        "kurva_df": kurva_df,
        "best_model_name": best_model_name,
        "best_data_name": best_data_name,
        "threshold": threshold,
        "threshold_criterion": threshold_criterion,
    }


if __name__ == "__main__":
    configure_utf8_stdout()
    set_seed(SEED)
    cfg = get_version_config(DEFAULT_VERSION)
    ensure_dirs(cfg)
    splits = load_splits(cfg["splits_dir"])
    run_training(
        splits,
        seed=SEED,
        version=cfg["version"],
        citation=cfg["citation"],
        model_path=cfg["model_path"],
        plots_dir=cfg["plots_dir"],
        reports_dir=cfg["reports_dir"],
    )
