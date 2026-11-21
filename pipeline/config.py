"""Konfigurasi terpusat dan manajemen versi (v001, v002, ...) untuk pipeline
detektor SMS spam/penipuan.

Versi aktif default: `v002` (`dataset_sms_spam_v1.csv` + `SpamDam_twitter_data.csv`),
dengan artefak terpisah di `models/v002/` dan `outputs/v002/`.
"""
import random
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Path proyek & versi
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

WIBISONO_CSV = PROJECT_ROOT / "dataset_sms_spam_v1.csv"
SPAMDAM_TWITTER_CSV = PROJECT_ROOT / "SpamDam_twitter_data.csv"

DEFAULT_VERSION = "v002"

VERSION_PRESETS: dict[str, dict] = {
    "v001": {
        "version": "v001",
        "data_path": WIBISONO_CSV,
        "extra_data": [],
        "model_dir": PROJECT_ROOT / "models",
        "output_dir": PROJECT_ROOT / "outputs",
        "locked_test_path": None,
        "benchmark_v001_test": None,
        "benchmark_v001_metrics": None,
        "dataset_label": "dataset_sms_spam_v1.csv (Rahmi & Wibisono, 2016)",
        "citation": (
            "Rahmi, F. and Wibisono, Y. (2016). Aplikasi SMS Spam Filtering pada Android "
            "menggunakan Naive Bayes, Unpublished manuscript."
        ),
    },
    "v002": {
        "version": "v002",
        "data_path": WIBISONO_CSV,
        "extra_data": [SPAMDAM_TWITTER_CSV],
        "model_dir": PROJECT_ROOT / "models" / "v002",
        "output_dir": PROJECT_ROOT / "outputs" / "v002",
        "locked_test_path": PROJECT_ROOT / "outputs" / "splits" / "test.csv",
        "benchmark_v001_test": PROJECT_ROOT / "outputs" / "splits" / "test.csv",
        "benchmark_v001_metrics": PROJECT_ROOT / "outputs" / "reports" / "metrics_summary.json",
        "dataset_label": "dataset_sms_spam_v1.csv + SpamDam_twitter_data.csv (36.668 baris)",
        "citation": (
            "Rahmi, F. and Wibisono, Y. (2016) + SpamDam Twitter Dataset (SpamDam_twitter_data.csv)."
        ),
    },
}


def get_version_config(version: str = DEFAULT_VERSION) -> dict:
    """Mengembalikan konfigurasi path & dataset untuk versi tertentu (`v001`, `v002`, atau kustom)."""
    if version in VERSION_PRESETS:
        cfg = dict(VERSION_PRESETS[version])
    else:
        cfg = {
            "version": version,
            "data_path": WIBISONO_CSV,
            "extra_data": [],
            "model_dir": PROJECT_ROOT / "models" / version,
            "output_dir": PROJECT_ROOT / "outputs" / version,
            "locked_test_path": PROJECT_ROOT / "outputs" / "splits" / "test.csv",
            "benchmark_v001_test": PROJECT_ROOT / "outputs" / "splits" / "test.csv",
            "benchmark_v001_metrics": PROJECT_ROOT / "outputs" / "reports" / "metrics_summary.json",
            "dataset_label": f"Custom dataset ({version})",
            "citation": VERSION_PRESETS["v001"]["citation"],
        }

    model_dir = Path(cfg["model_dir"])
    output_dir = Path(cfg["output_dir"])
    cfg["model_dir"] = model_dir
    cfg["model_path"] = model_dir / "sms_scam_model.joblib"
    cfg["indobert_dir"] = model_dir / "indobert"
    cfg["output_dir"] = output_dir
    cfg["splits_dir"] = output_dir / "splits"
    cfg["plots_dir"] = output_dir / "plots"
    cfg["reports_dir"] = output_dir / "reports"
    return cfg


# Konstanta default mengikuti DEFAULT_VERSION (v002)
_ACTIVE = get_version_config(DEFAULT_VERSION)
DATA_PATH: Path = _ACTIVE["data_path"]
EXTRA_DATA: list[Path] = list(_ACTIVE["extra_data"])

MODEL_DIR: Path = _ACTIVE["model_dir"]
MODEL_PATH: Path = _ACTIVE["model_path"]
INDOBERT_DIR: Path = _ACTIVE["indobert_dir"]

OUTPUT_DIR: Path = _ACTIVE["output_dir"]
SPLITS_DIR: Path = _ACTIVE["splits_dir"]
PLOTS_DIR: Path = _ACTIVE["plots_dir"]
REPORTS_DIR: Path = _ACTIVE["reports_dir"]

# ---------------------------------------------------------------------------
# Konstanta eksperimen & label
# ---------------------------------------------------------------------------
SEED = 42
NEAR_DUP_THRESHOLD = 0.80
TEST_N_SPLITS = 7   # ~14% test, ~86% train_full
VAL_N_SPLITS = 6    # ~14% val, ~72% train

LABELS = {0: "normal", 1: "penipuan", 2: "promo"}
FRAUD = 1
TARGET_NAMES = [LABELS[i] for i in sorted(LABELS)]

THRESHOLD_MIN = 0.15
THRESHOLD_MAX = 0.60
THRESHOLD_MAX_IMBALANCED = 0.85
THRESHOLD_STEP = 0.01

STOPWORDS_ID = (
    "yang dan di ke dari ini itu untuk dengan pada ada saya aku kamu anda kita kami "
    "ya yg aja saja juga sudah belum tidak bisa akan atau karena jadi kalau lagi mau sama dalam oleh "
    "nya lah kah pun dong deh sih kok nih tuh si para buat tapi biar"
).split()

DATA_CITATION = _ACTIVE["citation"]
DATA_LICENSE = "CC BY-SA 4.0 (dataset_sms_spam_v1.csv) + SpamDam Twitter Dataset"

# Pesan uji modus baru & kasus umum (untuk smoke test pasca-training)
PESAN_UJI = [
    "Assalamualaikum, ini undangan pernikahan kami. Mohon dibuka ya: Undangan_Pernikahan.apk",
    "Paket Anda dari J&T tertahan karena alamat tidak lengkap. Lihat foto resi di sini: Resi_JNT.apk",
    "SELAMAT! No Anda terpilih mendapat cek tunai Rp175jt dari Shopee. Info klik bit.ly/hadiah-shopee",
    "Mah ini aku ganti nomor. Tolong transfer 500rb dulu ke rek BRI ini ya, penting banget",
    "Bro nanti malam jadi futsal jam 8? kabarin ya",
    "Nikmati kuota 30GB cuma Rp50rb! Aktifkan sekarang di *363#",
    "Kode OTP Anda 482913. JANGAN berikan kode ini kepada siapapun termasuk pihak bank.",
    "nama1, rapat kelompok besok jam 10 di perpus ya",
]


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_dirs(cfg: dict | None = None) -> None:
    target = cfg or _ACTIVE
    for key in ("model_dir", "output_dir", "splits_dir", "plots_dir", "reports_dir"):
        Path(target[key]).mkdir(parents=True, exist_ok=True)


def configure_utf8_stdout() -> None:
    """Pastikan output terminal Windows mendukung UTF-8 (emoji & simbol)."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass
