"""Tahap 2 — Exploratory Data Analysis (EDA) & ringkasan fitur buatan."""
if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

import sms_utils
from pipeline.config import (
    DATA_PATH,
    LABELS,
    PLOTS_DIR,
    REPORTS_DIR,
    STOPWORDS_ID,
    TARGET_NAMES,
    configure_utf8_stdout,
    ensure_dirs,
)
from pipeline.data_loader import load_all_datasets


def run_eda(
    df: pd.DataFrame,
    plots_dir: Path = PLOTS_DIR,
    reports_dir: Path = REPORTS_DIR,
    save_outputs: bool = True,
) -> dict:
    """Menjalankan EDA: distribusi kelas, panjang pesan, top words, contoh normalisasi,
    dan statistik rata-rata fitur buatan per kelas.
    """
    df_eda = df.copy()
    df_eda["n_kata"] = df_eda["teks"].str.split().str.len()
    df_eda["teks_norm"] = df_eda["teks"].map(sms_utils.normalize_text)

    class_counts = df_eda["label"].map(LABELS).value_counts().reindex(TARGET_NAMES)
    length_stats = df_eda.groupby(df_eda["label"].map(LABELS))["n_kata"].describe().round(1)

    print("=== EDA: Distribusi Kelas ===")
    print(class_counts.to_string())
    print("\n=== EDA: Statistik Panjang Pesan (Jumlah Kata) ===")
    print(length_stats.to_string())

    # Top 15 kata paling sering per kelas setelah normalisasi
    print("\n=== EDA: 15 Kata Paling Sering per Kelas ===")
    top_words: dict[str, list[str]] = {}
    for k, name in LABELS.items():
        cv = CountVectorizer(preprocessor=sms_utils.normalize_text, stop_words=STOPWORDS_ID)
        X = cv.fit_transform(df_eda.loc[df_eda["label"] == k, "teks"])
        freq = np.asarray(X.sum(axis=0)).ravel()
        top = [str(cv.get_feature_names_out()[i]) for i in freq.argsort()[::-1][:15]]
        top_words[name] = top
        print(f"  {name:9s}: {', '.join(top)}")

    # Contoh normalisasi
    print("\n=== EDA: Contoh Normalisasi Teks ===")
    for _, r in df_eda.groupby("label").head(2).iterrows():
        print(f"  [{LABELS[r.label]}] {r.teks}")
        print(f"     -> {r.teks_norm}")

    # Ringkasan fitur buatan (meta features)
    meta = pd.DataFrame(sms_utils.extract_features(df_eda["teks"]), columns=sms_utils.META_FEATURES)
    meta["kelas"] = df_eda["label"].map(LABELS)
    meta_summary = meta.groupby("kelas").mean().T.round(3)
    print("\n=== EDA: Rata-rata Fitur Buatan per Kelas ===")
    print(meta_summary.to_string())

    plot_path = None
    meta_csv_path = None
    if save_outputs:
        plots_dir.mkdir(parents=True, exist_ok=True)
        reports_dir.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
        class_counts.plot.bar(ax=axes[0], color=["#4C78A8", "#E45756", "#F2A541"])
        axes[0].set_title("Jumlah pesan per kelas")
        axes[0].tick_params(axis="x", rotation=0)

        data_box = [df_eda.loc[df_eda["label"] == k, "n_kata"] for k in sorted(LABELS)]
        axes[1].boxplot(data_box)
        axes[1].set_xticks(range(1, len(TARGET_NAMES) + 1), TARGET_NAMES)
        axes[1].set_title("Panjang pesan (jumlah kata)")
        plt.tight_layout()

        plot_path = plots_dir / "eda_distribusi.png"
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)

        meta_csv_path = reports_dir / "meta_features_summary.csv"
        meta_summary.to_csv(meta_csv_path)
        print(f"\nGrafik EDA tersimpan di   : {plot_path}")
        print(f"Ringkasan meta tersimpan  : {meta_csv_path}")

    return {
        "class_counts": class_counts.to_dict(),
        "top_words": top_words,
        "meta_summary": meta_summary,
        "plot_path": plot_path,
        "meta_csv_path": meta_csv_path,
    }


if __name__ == "__main__":
    configure_utf8_stdout()
    ensure_dirs()
    df = load_all_datasets(DATA_PATH)
    run_eda(df)
