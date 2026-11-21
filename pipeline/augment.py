"""Tahap 4 — Augmentasi data (khusus data train untuk menyeimbangkan kelas minoritas)."""
if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import random
import re

import pandas as pd

import sms_utils
from pipeline.config import (
    LABELS,
    SEED,
    SPLITS_DIR,
    TARGET_NAMES,
    configure_utf8_stdout,
)
from pipeline.preprocess import load_splits

ENTITAS = [
    ["Telkomsel", "Indosat", "IM3", "Tri", "XL", "Axis", "Smartfren"],
    ["Lazada", "Shopee", "Tokopedia", "Blibli", "Bukalapak"],
    ["BRI", "BCA", "Mandiri", "BNI", "BSI"],
    ["JNE", "J&T", "SiCepat", "AnterAja", "Ninja Xpress"],
    ["mobil", "motor", "laptop", "emas", "rumah"],
]
ENTITAS_RE = [
    (re.compile(r"\b(" + "|".join(map(re.escape, grp)) + r")\b", re.IGNORECASE), grp)
    for grp in ENTITAS
]
SINGKAT = {
    "dengan": "dgn",
    "yang": "yg",
    "untuk": "utk",
    "tidak": "tdk",
    "sudah": "sdh",
    "selamat": "slmt",
    "mendapatkan": "mendptkan",
    "informasi": "info",
    "hubungi": "hub",
    "nomor": "no",
    "sekarang": "skrg",
    "rekening": "rek",
    "transfer": "tf",
}
LEET = {"i": "l", "o": "0", "a": "4", "e": "3", "s": "5"}
RE_JANGAN_DIUBAH = re.compile(r"https?://|www\.|\.(com|net|id|ly|xyz|apk)\b|\d", re.IGNORECASE)


def ganti_entitas(t: str, rng: random.Random) -> str:
    for rx, grp in ENTITAS_RE:
        t = rx.sub(lambda m: rng.choice(grp), t)
    return t


def singkat(t: str, rng: random.Random, p: float = 0.5) -> str:
    return " ".join(SINGKAT.get(w.lower(), w) if rng.random() < p else w for w in t.split())


def leet(t: str, rng: random.Random, p: float = 0.04) -> str:
    out = []
    for w in t.split():
        if RE_JANGAN_DIUBAH.search(w):  # jangan rusak URL/angka
            out.append(w)
        else:
            out.append("".join(LEET[c] if c in LEET and rng.random() < p else c for c in w))
    return " ".join(out)


def dropout(t: str, rng: random.Random, p: float = 0.1) -> str:
    kata = t.split()
    if len(kata) <= 5:
        return t
    sisa = [w for w in kata if RE_JANGAN_DIUBAH.search(w) or rng.random() > p]
    return " ".join(sisa) if sisa else t


def kapital(t: str, rng: random.Random) -> str:
    return t.upper()


OPS = [singkat, leet, dropout, kapital]


def augment_text(t: str, rng: random.Random) -> str:
    t = ganti_entitas(t, rng)
    k = rng.choice([1, 1, 2])
    for op in rng.sample(OPS, k):
        t = op(t, rng)
    return t


def augment_train(
    train: pd.DataFrame,
    kelas: tuple[int, ...] = (0, 1, 2),
    seed: int = SEED,
    max_tries: int = 5,
    max_multiplier: float = 3.0,
) -> pd.DataFrame:
    """Menyeimbangkan kelas minoritas pada data train saja.

    Target tiap kelas dibatasi oleh `min(counts.max(), count_k * max_multiplier)`
    agar tidak membuat ribuan salinan sintetis berulang ketika satu kelas sangat besar.
    """
    rng = random.Random(seed)
    counts = train["label"].value_counts()
    max_count = int(counts.max())
    seen = set(train["teks"].map(sms_utils.normalize_text))
    baru = []

    for k in kelas:
        sumber = train[train["label"] == k]
        if sumber.empty:
            continue
        cur = int(counts.get(k, 0))
        class_target = min(max_count, int(round(cur * max_multiplier)))
        perlu = max(0, class_target - cur)
        tries = 0
        while perlu > 0 and tries < perlu * max_tries + 50:
            tries += 1
            asal = sumber.iloc[rng.randrange(len(sumber))]
            t = augment_text(asal["teks"], rng)
            norm = sms_utils.normalize_text(t)
            if norm in seen:
                continue
            seen.add(norm)
            baru.append({
                "teks": t,
                "label": k,
                "sumber": "augmentasi",
                "group": asal["group"],
                "is_synthetic": 1,
            })
            perlu -= 1

    base = train.assign(is_synthetic=0)
    if baru:
        out = pd.concat([base, pd.DataFrame(baru)], ignore_index=True)
    else:
        out = base.copy()
    return out.sample(frac=1, random_state=seed).reset_index(drop=True)


if __name__ == "__main__":
    configure_utf8_stdout()
    splits = load_splits(SPLITS_DIR)
    train_df = splits["train"]
    train_aug = augment_train(train_df, seed=SEED)
    print("Train asli      :", train_df["label"].map(LABELS).value_counts().reindex(TARGET_NAMES).to_dict())
    print("Train + augment :", train_aug["label"].map(LABELS).value_counts().reindex(TARGET_NAMES).to_dict())
    print("\nContoh hasil augmentasi:")
    print(train_aug[train_aug["is_synthetic"] == 1].head(8)[["teks", "label"]].to_string(index=False))
