"""Orkestrator end-to-end pipeline detektor pesan penipuan SMS/WA (mendukung versi).

Contoh pemakaian:
  python run_pipeline.py                  # default: versi v002 (Wibisono + SpamDam_twitter_data.csv)
  python run_pipeline.py --version v001   # reproduksi versi v001 (dataset_sms_spam_v1.csv saja)
  python run_pipeline.py --version v002 --skip-eda
  python run_pipeline.py --version v002 --run-indobert
"""
import argparse
from pathlib import Path

import pandas as pd
import sklearn

from pipeline.config import (
    DEFAULT_VERSION,
    SEED,
    configure_utf8_stdout,
    ensure_dirs,
    get_version_config,
    set_seed,
)
from pipeline.data_loader import load_all_datasets
from pipeline.eda import run_eda
from pipeline.evaluate import run_evaluation
from pipeline.preprocess import run_preprocess
from pipeline.train import run_training, save_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Jalankan pipeline end-to-end detektor penipuan SMS/WA."
    )
    parser.add_argument(
        "--version",
        default=DEFAULT_VERSION,
        help=f"Versi pipeline/dataset: v001, v002, atau nama kustom (default: {DEFAULT_VERSION})",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Override path dataset CSV utama (opsional)",
    )
    parser.add_argument(
        "--extra-data",
        type=Path,
        nargs="*",
        default=None,
        help="Override daftar dataset CSV tambahan (opsional)",
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        default=None,
        help="Override path output file model .joblib (opsional)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help=f"Random seed (default: {SEED})",
    )
    parser.add_argument(
        "--skip-eda",
        action="store_true",
        help="Lewati tahap EDA & grafik distribusi",
    )
    parser.add_argument(
        "--run-indobert",
        action="store_true",
        help="Jalankan pembanding opsional fine-tune IndoBERT (butuh torch & transformers)",
    )
    return parser.parse_args()


def main() -> None:
    configure_utf8_stdout()
    args = parse_args()
    set_seed(args.seed)

    cfg = get_version_config(args.version)
    ensure_dirs(cfg)

    data_path = args.data if args.data is not None else cfg["data_path"]
    extra_data = args.extra_data if args.extra_data is not None else cfg["extra_data"]
    model_path = args.model_out if args.model_out is not None else cfg["model_path"]

    print(f"scikit-learn {sklearn.__version__} | pandas {pd.__version__}")
    print(f"Versi pipeline : {cfg['version']}")
    print(f"Dataset utama  : {data_path}")
    print(f"Dataset ekstra : {[str(p) for p in extra_data] if extra_data else '[]'}")
    print(f"Output folder  : {cfg['output_dir']}")
    print(f"Model output   : {model_path}")

    # 1. Collect
    print("\n" + "=" * 60)
    print("TAHAP 1: MEMUAT DATA")
    print("=" * 60)
    df = load_all_datasets(data_path=data_path, extra_data=extra_data)
    print(f"Total data termuat: {len(df):,} baris")
    print("Komposisi per sumber:")
    print(df.groupby(["sumber", "label"]).size().unstack(fill_value=0).to_string())

    # 2. EDA
    if not args.skip_eda:
        print("\n" + "=" * 60)
        print("TAHAP 2: EXPLORATORY DATA ANALYSIS (EDA)")
        print("=" * 60)
        run_eda(df, plots_dir=cfg["plots_dir"], reports_dir=cfg["reports_dir"], save_outputs=True)

    # 3. Preprocessing, deduplikasi & split anti-leakage
    print("\n" + "=" * 60)
    print("TAHAP 3: PREPROCESSING, DEDUPLIKASI & SPLIT")
    print("=" * 60)
    splits, _ = run_preprocess(
        df,
        seed=args.seed,
        splits_dir=cfg["splits_dir"],
        locked_test_path=cfg["locked_test_path"],
        save=True,
    )

    # 4 & 5. Augmentasi, training kandidat, tuning threshold & retrain final
    print("\n" + "=" * 60)
    print("TAHAP 4: AUGMENTASI, TRAINING & TUNING THRESHOLD")
    print("=" * 60)
    train_res = run_training(
        splits=splits,
        seed=args.seed,
        version=cfg["version"],
        citation=cfg["citation"],
        model_path=model_path,
        plots_dir=cfg["plots_dir"],
        reports_dir=cfg["reports_dir"],
        save_model=False,
        save_outputs=True,
    )
    bundle = train_res["bundle"]

    # 6. Evaluasi test set (+ benchmark v001 jika ada), interpretasi & smoke test
    print("\n" + "=" * 60)
    print("TAHAP 5: EVALUASI TEST SET & INTERPRETASI")
    print("=" * 60)
    eval_res = run_evaluation(
        bundle=bundle,
        test_df=splits["test"],
        plots_dir=cfg["plots_dir"],
        reports_dir=cfg["reports_dir"],
        save_outputs=True,
        dataset_label=cfg["dataset_label"],
        benchmark_v001_test=cfg["benchmark_v001_test"],
        benchmark_v001_metrics=cfg["benchmark_v001_metrics"],
    )

    # 7. Simpan model final (dengan metrics_test & benchmark) & uji load ulang
    print("\n" + "=" * 60)
    print("TAHAP 6: SIMPAN & VERIFIKASI MODEL")
    print("=" * 60)
    bundle["metrics_test"] = eval_res["metrics_test"]
    if eval_res.get("v001_benchmark") and eval_res["v001_benchmark"].get("benchmark"):
        bundle["metrics_v001_benchmark"] = eval_res["v001_benchmark"]["benchmark"]["metrics_test"]
    save_bundle(bundle, model_path=model_path)

    # 8. Opsional: IndoBERT
    if args.run_indobert:
        print("\n" + "=" * 60)
        print("TAHAP OPSIONAL: FINE-TUNE INDOBERT")
        print("=" * 60)
        from pipeline.indobert import run_indobert

        run_indobert(
            splits=splits,
            classic_model_name=bundle["model_name"],
            classic_metrics_test=bundle["metrics_test"],
            seed=args.seed,
        )

    print(f"\nPipeline {cfg['version']} selesai dengan sukses.")


if __name__ == "__main__":
    main()
