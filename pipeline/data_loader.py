"""Tahap 1 — Collect: memuat dan memvalidasi dataset_sms_spam_v1.csv."""
if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pathlib import Path
from typing import Sequence

import pandas as pd

from pipeline.config import (
    DATA_PATH,
    EXTRA_DATA,
    LABELS,
    SEED,
    configure_utf8_stdout,
)


def load_dataset(path: Path | str, sumber: str = "wibisono") -> pd.DataFrame:
    """Membaca satu file CSV SMS spam dengan fallback encoding & normalisasi kolom."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"File dataset tidak ditemukan: {path}. "
            "Pastikan dataset_sms_spam_v1.csv berada di folder proyek."
        )

    df = None
    last_err = None
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            df = pd.read_csv(path, encoding=enc)
            break
        except UnicodeDecodeError as err:
            last_err = err
            continue

    if df is None:
        raise RuntimeError(f"Gagal membaca {path} dengan encoding utf-8/utf-8-sig/latin-1") from last_err

    df.columns = [str(c).strip() for c in df.columns]
    text_col = next(
        (c for c in df.columns if c.lower() in ("teks", "text", "sms", "pesan")),
        None,
    )
    label_col = next(
        (c for c in df.columns if c.lower() in ("label", "kelas", "class")),
        None,
    )
    if text_col is None or label_col is None:
        raise ValueError(
            f"Kolom teks/label tidak ditemukan di {path}. Kolom tersedia: {list(df.columns)}"
        )

    df = df.rename(columns={text_col: "teks", label_col: "label"})[["teks", "label"]]
    df = df.dropna(subset=["teks", "label"])
    df["teks"] = df["teks"].astype(str).str.strip()
    df = df[df["teks"].str.len() > 0]
    df["label"] = df["label"].astype(int)
    df["sumber"] = sumber
    return df.reset_index(drop=True)


def load_all_datasets(
    data_path: Path | str = DATA_PATH,
    extra_data: Sequence[Path | str] | None = None,
) -> pd.DataFrame:
    """Memuat dataset utama (`dataset_sms_spam_v1.csv`) dan opsional dataset tambahan.

    File CSV lain di workspace diabaikan kecuali dimasukkan eksplisit ke `extra_data`.
    """
    extras = EXTRA_DATA if extra_data is None else list(extra_data)
    frames = [load_dataset(Path(data_path), sumber="wibisono")]
    for p in extras:
        p_path = Path(p)
        frames.append(load_dataset(p_path, sumber=p_path.stem))

    df = pd.concat(frames, ignore_index=True)

    unknown_labels = set(df["label"].unique()) - set(LABELS)
    if unknown_labels:
        raise ValueError(f"Label tidak dikenal: {unknown_labels}. Diharapkan subset dari {set(LABELS)}")

    return df


if __name__ == "__main__":
    configure_utf8_stdout()
    df = load_all_datasets()
    print(f"Dataset dimuat dari : {DATA_PATH}")
    print(f"Jumlah data         : {len(df)}")
    print("\nDistribusi label:")
    print(df["label"].map(LABELS).value_counts().to_string())
    print("\nContoh 5 baris acak:")
    print(df.sample(5, random_state=SEED).to_string(index=False))
