"""Tahap 3 — Preprocessing: normalisasi, deduplikasi, pengelompokan near-duplicate
berbasis chunked sparse matmul (aman untuk puluhan ribu baris), dan split
train / validation / test anti data leakage.
"""
if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

import sms_utils
from pipeline.config import (
    DATA_PATH,
    DEFAULT_VERSION,
    EXTRA_DATA,
    LABELS,
    NEAR_DUP_THRESHOLD,
    SEED,
    SPLITS_DIR,
    TARGET_NAMES,
    TEST_N_SPLITS,
    VAL_N_SPLITS,
    configure_utf8_stdout,
    ensure_dirs,
    get_version_config,
    set_seed,
)
from pipeline.data_loader import load_all_datasets


def _chunked_near_duplicate_graph(
    texts_norm: pd.Series,
    threshold: float = NEAR_DUP_THRESHOLD,
    batch_size: int = 512,
) -> csr_matrix:
    """Menghitung graf adjacency near-duplicate secara bertahap (chunked) agar
    tidak menghabiskan RAM saat dataset mencapai puluhan ribu baris.
    """
    n = len(texts_norm)
    min_df = 2 if n > 2000 else 1
    vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=min_df,
        dtype=np.float32,
    ).fit_transform(texts_norm)

    row_parts: list[np.ndarray] = []
    col_parts: list[np.ndarray] = []

    vec_t = vec.T.tocsc()
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        sim_block = (vec[start:end] @ vec_t).tocoo()
        mask = sim_block.data >= threshold
        if not np.any(mask):
            continue
        r = sim_block.row[mask] + start
        c = sim_block.col[mask]
        # Simpan hanya sisi segitiga atas (r < c) untuk graf tak berarah
        upper = r < c
        if np.any(upper):
            row_parts.append(r[upper].astype(np.int32))
            col_parts.append(c[upper].astype(np.int32))

    if row_parts:
        rows = np.concatenate(row_parts)
        cols = np.concatenate(col_parts)
        data = np.ones(len(rows), dtype=np.int8)
        adj = csr_matrix((data, (rows, cols)), shape=(n, n))
    else:
        adj = csr_matrix((n, n), dtype=np.int8)
    return adj


def deduplicate_and_group(
    df: pd.DataFrame,
    near_dup_threshold: float = NEAR_DUP_THRESHOLD,
    batch_size: int = 512,
) -> tuple[pd.DataFrame, dict]:
    """Menormalisasi teks, membuang duplikat persis, lalu mengelompokkan
    pesan near-duplicate (cosine similarity char_wb 3-5 >= threshold) ke grup yang sama.
    """
    out = df.copy()
    out["teks_norm"] = out["teks"].map(sms_utils.normalize_text)

    n_awal = len(out)
    konflik = out.groupby("teks_norm")["label"].nunique()
    konflik = konflik[konflik > 1]
    if len(konflik):
        print(f"[PERINGATAN] {len(konflik)} teks identik dengan label BERBEDA (diprioritaskan label dataset utama/pertama):")
        print(
            out[out["teks_norm"].isin(konflik.index)]
            .sort_values("teks_norm")[["teks", "label", "sumber"]]
            .head(20)
            .to_string(index=False)
        )

    out = out.drop_duplicates("teks_norm", keep="first").reset_index(drop=True)
    n_duplikat = n_awal - len(out)
    print(f"Duplikat persis dibuang: {n_duplikat} → sisa {len(out)} pesan")

    adj = _chunked_near_duplicate_graph(
        out["teks_norm"],
        threshold=near_dup_threshold,
        batch_size=batch_size,
    )
    n_grup, out["group"] = connected_components(adj, directed=False)

    ukuran = out["group"].value_counts()
    n_multi = int((ukuran > 1).sum())
    max_grup = int(ukuran.max())
    print(f"Jumlah grup: {n_grup} | grup berisi >1 pesan: {n_multi} | grup terbesar: {max_grup} pesan")

    for g in ukuran[ukuran > 1].index[:2]:
        print("\nContoh grup near-duplicate:")
        print(out.loc[out["group"] == g, "teks"].head(3).to_string())

    stats = {
        "n_awal": n_awal,
        "n_konflik_label": int(len(konflik)),
        "n_duplikat_dibuang": n_duplikat,
        "n_sisa": len(out),
        "n_grup": int(n_grup),
        "n_grup_near_dup": n_multi,
        "ukuran_grup_terbesar": max_grup,
    }
    return out, stats


def group_split(frame: pd.DataFrame, n_splits: int, seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    if frame["label"].nunique() < 2:
        gss = GroupShuffleSplit(n_splits=1, test_size=1.0 / n_splits, random_state=seed)
        rest_idx, hold_idx = next(gss.split(frame, groups=frame["group"]))
    else:
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        rest_idx, hold_idx = next(sgkf.split(frame, frame["label"], frame["group"]))
    return frame.iloc[rest_idx].reset_index(drop=True), frame.iloc[hold_idx].reset_index(drop=True)


def split_dataset(
    df_clean: pd.DataFrame,
    seed: int = SEED,
    locked_test_path: Path | str | None = None,
) -> dict[str, pd.DataFrame]:
    """Membagi data menjadi ~72% train / ~14% val / ~14% test tanpa kebocoran grup.

    Jika `locked_test_path` diberikan (mis. test set v001), seluruh grup yang berisi
    pesan dari test set tersebut dikunci ke `test` agar evaluasi pembandingan v001 vs
    v002 100% bebas data leakage.
    """
    locked_path = Path(locked_test_path) if locked_test_path is not None else None
    if locked_path is not None and locked_path.exists():
        locked_df = pd.read_csv(locked_path, encoding="utf-8")
        locked_norms = set(locked_df["teks"].astype(str).map(sms_utils.normalize_text))
        locked_groups = set(df_clean.loc[df_clean["teks_norm"].isin(locked_norms), "group"])

        locked_test = df_clean[df_clean["group"].isin(locked_groups)].reset_index(drop=True)
        remaining = df_clean[~df_clean["group"].isin(locked_groups)].reset_index(drop=True)

        # Pisahkan BERDASARKAN GRUP (bukan baris) agar grup yang berisi campuran
        # wibisono + data tambahan tidak pernah terpecah antara train_full dan test.
        base_groups = set(remaining.loc[remaining["sumber"] == "wibisono", "group"])
        base_rest = remaining[remaining["group"].isin(base_groups)].reset_index(drop=True)
        extra_rest = remaining[~remaining["group"].isin(base_groups)].reset_index(drop=True)

        if len(extra_rest) > 0:
            extra_train, extra_test = group_split(extra_rest, n_splits=TEST_N_SPLITS, seed=seed)
            train_full = pd.concat([base_rest, extra_train], ignore_index=True)
            test_df = pd.concat([locked_test, extra_test], ignore_index=True)
        else:
            train_full = base_rest
            test_df = locked_test

        assert not (set(train_full["teks_norm"]) & locked_norms), "Kebocoran test set v001 ke train_full!"
        print(
            f"Mengunci {len(locked_groups)} grup test v001 ({len(locked_test)} pesan) "
            "ke test set agar evaluasi benchmark v001 bebas leakage."
        )
    else:
        train_full, test_df = group_split(df_clean, n_splits=TEST_N_SPLITS, seed=seed)

    train_df, val_df = group_split(train_full, n_splits=VAL_N_SPLITS, seed=seed)

    assert not (set(train_df["group"]) & set(val_df["group"])), "Kebocoran grup antara train dan val!"
    assert not (set(train_full["group"]) & set(test_df["group"])), "Kebocoran grup antara train_full dan test!"

    ringkas = pd.DataFrame({
        nama: d["label"].map(LABELS).value_counts().reindex(TARGET_NAMES).fillna(0).astype(int)
        for nama, d in [("train", train_df), ("val", val_df), ("test", test_df)]
    })
    ringkas.loc["TOTAL"] = ringkas.sum()
    print("\n=== Ringkasan Split (Anti Data Leakage) ===")
    print(ringkas.to_string())

    return {
        "train": train_df,
        "val": val_df,
        "train_full": train_full,
        "test": test_df,
    }


def save_splits(splits: dict[str, pd.DataFrame], splits_dir: Path = SPLITS_DIR) -> None:
    splits_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in splits.items():
        frame.to_csv(splits_dir / f"{name}.csv", index=False, encoding="utf-8")
    print(f"Split tersimpan di: {splits_dir}")


def load_splits(splits_dir: Path = SPLITS_DIR) -> dict[str, pd.DataFrame]:
    splits = {}
    for name in ("train", "val", "train_full", "test"):
        path = splits_dir / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"File split belum ada: {path}. Jalankan `python -m pipeline.preprocess` terlebih dahulu."
            )
        frame = pd.read_csv(path, encoding="utf-8")
        frame["teks"] = frame["teks"].astype(str)
        frame["label"] = frame["label"].astype(int)
        frame["group"] = frame["group"].astype(int)
        splits[name] = frame
    return splits


def run_preprocess(
    df: pd.DataFrame,
    near_dup_threshold: float = NEAR_DUP_THRESHOLD,
    seed: int = SEED,
    splits_dir: Path = SPLITS_DIR,
    locked_test_path: Path | str | None = None,
    save: bool = True,
) -> tuple[dict[str, pd.DataFrame], dict]:
    df_clean, stats = deduplicate_and_group(df, near_dup_threshold=near_dup_threshold)
    splits = split_dataset(df_clean, seed=seed, locked_test_path=locked_test_path)
    if save:
        save_splits(splits, splits_dir=splits_dir)
    return splits, stats


if __name__ == "__main__":
    configure_utf8_stdout()
    set_seed(SEED)
    cfg = get_version_config(DEFAULT_VERSION)
    ensure_dirs(cfg)
    df = load_all_datasets(cfg["data_path"], extra_data=cfg["extra_data"])
    run_preprocess(
        df,
        seed=SEED,
        splits_dir=cfg["splits_dir"],
        locked_test_path=cfg["locked_test_path"],
    )
