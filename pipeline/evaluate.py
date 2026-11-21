"""Tahap 7 — Evaluasi test set (gabungan & benchmark v001), confusion matrix,
analisis kesalahan, interpretasi fitur terpenting, visualisasi dashboard,
dan smoke test pesan modus baru.
"""
if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, classification_report

import sms_utils
from pipeline.config import (
    DEFAULT_VERSION,
    FRAUD,
    LABELS,
    PESAN_UJI,
    PLOTS_DIR,
    REPORTS_DIR,
    TARGET_NAMES,
    configure_utf8_stdout,
    ensure_dirs,
    get_version_config,
)
from pipeline.features import compute_metrics
from pipeline.preprocess import load_splits


def evaluate_on_test(
    bundle: dict,
    test_df: pd.DataFrame,
    plots_dir: Path = PLOTS_DIR,
    reports_dir: Path = REPORTS_DIR,
    save_outputs: bool = True,
    prefix: str = "",
    title_suffix: str = "test",
    dataset_label: str | None = None,
) -> dict:
    """Evaluasi model final pada test set (murni dari model, tanpa rule_override)."""
    final_model = bundle["pipeline"]
    threshold = float(bundle["threshold"])
    fraud_label = int(bundle.get("fraud_label", FRAUD))
    model_name = bundle.get("model_name", "model")
    version = bundle.get("version", DEFAULT_VERSION)

    proba_test = final_model.predict_proba(test_df["teks"])
    classes = list(final_model.classes_)
    pred_argmax = final_model.classes_[proba_test.argmax(axis=1)]
    pred_test = sms_utils.decide_batch(proba_test, classes, threshold, fraud_label)

    print(f"\n=== Evaluasi ({title_suffix}): {model_name} [{version}] | threshold {threshold:.2f} ===")
    print(classification_report(test_df["label"], pred_test, target_names=TARGET_NAMES, digits=3))

    metrics_argmax = compute_metrics(test_df["label"], pred_argmax, fraud_label=fraud_label)
    metrics_test = compute_metrics(test_df["label"], pred_test, fraud_label=fraud_label)
    banding = pd.DataFrame({
        "argmax (0.5)": metrics_argmax,
        f"threshold {threshold:.2f}": metrics_test,
    }).round(3)
    print(banding.to_string())

    # Analisis kesalahan
    fraud_idx = classes.index(fraud_label)
    err = test_df.assign(pred=pred_test, p_penipuan=proba_test[:, fraud_idx])
    err = err[err["label"] != err["pred"]].copy()
    err["label"] = err["label"].map(LABELS)
    err["pred"] = err["pred"].map(LABELS)
    err = err[["teks", "label", "pred", "p_penipuan"]].sort_values("p_penipuan", ascending=False)

    print(f"\n=== Analisis Kesalahan ({len(err)} kesalahan dari {len(test_df)} pesan {title_suffix}) ===")
    if not err.empty:
        print(err.head(15).round(3).to_string(index=False))
        if len(err) > 15:
            print(f"... ({len(err) - 15} kesalahan lainnya disimpan di CSV)")
    else:
        print("Tidak ada kesalahan prediksi di test set.")

    summary = {
        "version": version,
        "eval_split": title_suffix,
        "dataset_label": dataset_label or bundle.get("sumber_data", ""),
        "model_name": model_name,
        "data_train": bundle.get("data_train"),
        "threshold": threshold,
        "n_test": int(len(test_df)),
        "n_errors": int(len(err)),
        "metrics_test_threshold": metrics_test,
        "metrics_test_argmax": metrics_argmax,
    }

    if save_outputs:
        plots_dir.mkdir(parents=True, exist_ok=True)
        reports_dir.mkdir(parents=True, exist_ok=True)

        cm_name = f"confusion_matrix_{prefix}.png" if prefix else "confusion_matrix_test.png"
        err_name = f"test_errors_{prefix}.csv" if prefix else "test_errors.csv"
        sum_name = f"metrics_{prefix}.json" if prefix else "metrics_summary.json"
        dash_name = f"metrics_dashboard_{prefix}.png" if prefix else "metrics_summary_dashboard.png"

        fig, ax = plt.subplots(figsize=(4.8, 4.2))
        ConfusionMatrixDisplay.from_predictions(
            test_df["label"],
            pred_test,
            display_labels=TARGET_NAMES,
            cmap="Blues",
            ax=ax,
            colorbar=False,
        )
        ax.set_title(f"Confusion matrix ({title_suffix} - {version})")
        plt.tight_layout()
        cm_path = plots_dir / cm_name
        fig.savefig(cm_path, dpi=150)
        plt.close(fig)

        err_path = reports_dir / err_name
        err.to_csv(err_path, index=False, encoding="utf-8")

        summary_path = reports_dir / sum_name
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

        from pipeline.visualize import plot_metrics_summary
        dash_path = plot_metrics_summary(summary, output_path=plots_dir / dash_name)

        print(f"\nConfusion matrix tersimpan di : {cm_path}")
        print(f"Dashboard metrik tersimpan di : {dash_path}")
        print(f"Daftar error tersimpan di     : {err_path}")
        print(f"Ringkasan metrik tersimpan di : {summary_path}")

    return {
        "summary": summary,
        "metrics_test": metrics_test,
        "metrics_argmax": metrics_argmax,
        "errors_df": err,
    }


def evaluate_against_v001_benchmark(
    bundle: dict,
    benchmark_test_path: Path | str,
    v001_metrics_path: Path | str | None,
    plots_dir: Path,
    reports_dir: Path,
) -> dict | None:
    """Mengevaluasi model baru (v002) pada test set bersih v001 dan membuat
    laporan + visualisasi perbandingan langsung v001 vs v002.
    """
    bench_path = Path(benchmark_test_path)
    if not bench_path.exists():
        print(f"Benchmark test v001 tidak ditemukan di {bench_path}, dilewati.")
        return None

    v001_test_df = pd.read_csv(bench_path, encoding="utf-8")
    v001_test_df["teks"] = v001_test_df["teks"].astype(str)
    v001_test_df["label"] = v001_test_df["label"].astype(int)

    bench_res = evaluate_on_test(
        bundle,
        v001_test_df,
        plots_dir=plots_dir,
        reports_dir=reports_dir,
        save_outputs=True,
        prefix="v001_benchmark",
        title_suffix="test set v001 (162 SMS Indonesia)",
        dataset_label="Benchmark Test Set v001 (dataset_sms_spam_v1.csv, 162 SMS)",
    )

    comparison = None
    if v001_metrics_path is not None and Path(v001_metrics_path).exists():
        v001_summary = json.loads(Path(v001_metrics_path).read_text(encoding="utf-8"))
        v002_bench_summary = bench_res["summary"]

        comparison = {
            "benchmark_dataset": "outputs/splits/test.csv (162 SMS Indonesia, bebas leakage)",
            "v001": v001_summary,
            "v002_on_v001_test": v002_bench_summary,
        }
        comp_json = reports_dir / "comparison_v001_vs_v002.json"
        comp_json.write_text(json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8")

        comp_table = pd.DataFrame({
            f"v001 ({v001_summary['model_name']}, thr {v001_summary['threshold']:.2f})": v001_summary["metrics_test_threshold"],
            f"v002 ({v002_bench_summary['model_name']}, thr {v002_bench_summary['threshold']:.2f})": v002_bench_summary["metrics_test_threshold"],
        }).round(4)
        print("\n=== Perbandingan Langsung pada Test Set v001 (162 SMS Indonesia) ===")
        print(comp_table.to_string())

        from pipeline.visualize import plot_version_comparison
        comp_plot = plot_version_comparison(
            v001_summary=v001_summary,
            v002_bench_summary=v002_bench_summary,
            output_path=plots_dir / "comparison_v001_vs_v002.png",
        )
        print(f"Laporan perbandingan tersimpan : {comp_json}")
        print(f"Grafik perbandingan tersimpan  : {comp_plot}")

    return {"benchmark": bench_res, "comparison": comparison}


def interpret_top_features(final_model, top_k: int = 15) -> dict[str, list[str]]:
    """Menampilkan kata/fitur meta yang paling mendorong prediksi tiap kelas."""
    clf = final_model.steps[-1][1]
    names = final_model[:-1].get_feature_names_out()
    top_per_class: dict[str, list[str]] = {}

    print("\n=== Interpretasi: Fitur Paling Berpengaruh per Kelas ===")
    for i, c in enumerate(clf.classes_):
        if hasattr(clf, "coef_"):
            w = clf.coef_[i]
        else:  # Multinomial NB
            others = [j for j in range(len(clf.classes_)) if j != i]
            w = clf.feature_log_prob_[i] - clf.feature_log_prob_[others].mean(axis=0)
        mask = np.array([not str(n).startswith("char__") for n in names])
        idx = np.where(mask)[0][np.argsort(-w[mask])[:top_k]]
        feats = [str(names[j]).split("__")[-1] for j in idx]
        label_name = LABELS[int(c)]
        top_per_class[label_name] = feats
        print(f"  {label_name:9s}: {', '.join(feats)}")
    return top_per_class


def run_smoke_tests(bundle: dict, messages: list[str] = PESAN_UJI) -> list[dict]:
    """Menguji bundle hibrida (model + aturan .apk) pada pesan modus baru & contoh balasan bot."""
    print("\n=== Smoke Test: Pesan Modus Baru & Contoh Sehari-hari ===")
    results = []
    for t in messages:
        r = sms_utils.analyze(bundle, t)
        results.append(r)
        prob = ", ".join(f"{k} {v:.0%}" for k, v in r["proba"].items())
        tanda = " (aturan .apk)" if r["oleh_aturan"] else ""
        print(f"{r['label_name'].upper():9s}{tanda} | {prob}")
        print(f"   {t}")
        print(f"   kata kunci: {r['kata_kunci']}\n")

    if messages:
        print("=== Contoh Format Balasan Bot ===")
        print(sms_utils.format_reply(results[0]))
    return results


def run_evaluation(
    bundle: dict,
    test_df: pd.DataFrame,
    plots_dir: Path = PLOTS_DIR,
    reports_dir: Path = REPORTS_DIR,
    save_outputs: bool = True,
    dataset_label: str | None = None,
    benchmark_v001_test: Path | str | None = None,
    benchmark_v001_metrics: Path | str | None = None,
) -> dict:
    eval_res = evaluate_on_test(
        bundle,
        test_df,
        plots_dir=plots_dir,
        reports_dir=reports_dir,
        save_outputs=save_outputs,
        dataset_label=dataset_label,
    )
    bench_out = None
    if benchmark_v001_test is not None and save_outputs:
        bench_out = evaluate_against_v001_benchmark(
            bundle=bundle,
            benchmark_test_path=benchmark_v001_test,
            v001_metrics_path=benchmark_v001_metrics,
            plots_dir=plots_dir,
            reports_dir=reports_dir,
        )
    top_feats = interpret_top_features(bundle["pipeline"])
    smoke_res = run_smoke_tests(bundle)
    return {
        **eval_res,
        "v001_benchmark": bench_out,
        "top_features": top_feats,
        "smoke_tests": smoke_res,
    }


if __name__ == "__main__":
    configure_utf8_stdout()
    cfg = get_version_config(DEFAULT_VERSION)
    ensure_dirs(cfg)
    model_path = cfg["model_path"]
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model belum ada di {model_path}. Jalankan `python -m pipeline.train` terlebih dahulu."
        )
    bundle = joblib.load(model_path)
    splits = load_splits(cfg["splits_dir"])
    res = run_evaluation(
        bundle,
        splits["test"],
        plots_dir=cfg["plots_dir"],
        reports_dir=cfg["reports_dir"],
        dataset_label=cfg["dataset_label"],
        benchmark_v001_test=cfg["benchmark_v001_test"],
        benchmark_v001_metrics=cfg["benchmark_v001_metrics"],
    )

    bundle["metrics_test"] = res["metrics_test"]
    if res.get("v001_benchmark") and res["v001_benchmark"].get("benchmark"):
        bundle["metrics_v001_benchmark"] = res["v001_benchmark"]["benchmark"]["metrics_test"]
    joblib.dump(bundle, model_path, compress=3)
    print(f"\nMetadata metrics_test diperbarui di {model_path}")
