"""Utilitas bersama untuk pipeline training, inferensi CLI, dan bot Telegram
detektor pesan penipuan SMS/WA.

Fungsi di sini dipakai di dalam pipeline scikit-learn, jadi file ini harus
tetap ada di root proyek (satu folder dengan bot.py, predict.py, dan
run_pipeline.py) supaya model .joblib bisa di-load dengan benar.
"""
import math
import re
import unicodedata

import numpy as np

# ---------------------------------------------------------------------------
# 1. Normalisasi teks
# ---------------------------------------------------------------------------
RE_URL = re.compile(
    r"(?:https?://|www\.)\S+"
    r"|\b[\w\-]+(?:\.[\w\-]+)*\.(?:com|net|org|id|xyz|site|website|online|info|link|top|"
    r"club|ly|me|co|biz|shop|store|live|icu|vip|cc)\b(?:/\S*)?",
    re.IGNORECASE,
)
RE_PHONE = re.compile(r"(?:\+62|\b62|\b0)\s?8\d{1,3}[\s\-]?\d{3,4}[\s\-]?\d{2,5}")
RE_USSD = re.compile(r"\*\d+(?:\*\d+)*#")
RE_APK = re.compile(r"\.apk\b|\bapk\b", re.IGNORECASE)
RE_NON_WORD = re.compile(r"[^\w\s]")
RE_REPEAT = re.compile(r"(\w)\1{2,}")
RE_SPACE = re.compile(r"\s+")

# Kamus singkatan/slang -> bentuk baku. Silakan ditambah sendiri.
SLANG = {
    "yg": "yang", "dgn": "dengan", "dg": "dengan", "utk": "untuk", "u": "untuk",
    "tdk": "tidak", "gak": "tidak", "ga": "tidak", "gk": "tidak", "nggak": "tidak", "ngga": "tidak",
    "sdh": "sudah", "udh": "sudah", "udah": "sudah", "blm": "belum", "bs": "bisa", "bsa": "bisa",
    "krn": "karena", "karna": "karena", "sy": "saya", "aq": "aku", "km": "kamu", "kmu": "kamu",
    "dr": "dari", "pd": "pada", "jg": "juga", "tsb": "tersebut", "slmt": "selamat",
    "dpt": "dapat", "dptkan": "dapatkan", "mendptkan": "mendapatkan", "hub": "hubungi",
    "tlp": "telepon", "telp": "telepon", "no": "nomor", "rek": "rekening", "tf": "transfer",
    "bgt": "banget", "aja": "saja", "lg": "lagi", "trs": "terus", "tp": "tapi", "kmrn": "kemarin",
    "skrg": "sekarang", "org": "orang", "bln": "bulan", "thn": "tahun", "hr": "hari",
    "mnt": "menit", "rb": "ribu", "jt": "juta", "infonya": "info",
}

# Token pengganti hasil masking (tanpa <> agar tidak dibuang tokenizer TF-IDF)
TOKEN_URL = "urltoken"
TOKEN_PHONE = "nomortoken"
TOKEN_NUM = "angkatoken"
TOKEN_USSD = "ussdtoken"
TOKEN_LABEL = {
    TOKEN_URL: "[link]",
    TOKEN_PHONE: "[nomor HP]",
    TOKEN_NUM: "[angka]",
    TOKEN_USSD: "[kode USSD]",
}


def _mask_token(tok: str) -> str:
    """Token yang mayoritas angka (nominal, PIN, kode) -> angkatoken.
    Token 'leet' seperti h4diah atau nama1 dibiarkan karena itu sinyal.
    """
    digits = sum(c.isdigit() for c in tok)
    if digits and digits / len(tok) >= 0.5:
        return TOKEN_NUM
    return SLANG.get(tok, tok)


def normalize_text(text: str) -> str:
    t = unicodedata.normalize("NFKC", str(text)).lower()
    t = RE_URL.sub(f" {TOKEN_URL} ", t)
    t = RE_USSD.sub(f" {TOKEN_USSD} ", t)
    t = RE_PHONE.sub(f" {TOKEN_PHONE} ", t)
    t = RE_NON_WORD.sub(" ", t)
    t = RE_REPEAT.sub(r"\1\1", t)
    tokens = [_mask_token(tok) for tok in t.split()]
    return RE_SPACE.sub(" ", " ".join(tokens)).strip()


# ---------------------------------------------------------------------------
# 2. Fitur buatan sendiri (dihitung dari teks MENTAH)
# ---------------------------------------------------------------------------
KEYWORDS = {
    "kw_hadiah": r"selamat|hadiah|menang|pemenang|undian|terpilih|cek tunai|mobil|beruntung",
    "kw_uang": r"transfer|\btf\b|rekening|\brek\b|kirim|pulsa|saldo|bayar|biaya|tagihan|pinjam",
    "kw_aksi": r"klik|kunjungi|buka|download|unduh|install|instal|cek|hubungi|balas|ketik",
    "kw_rahasia": r"\bpin\b|kode|otp|verifikasi|password|sandi",
    "kw_promo": r"promo|diskon|bonus|paket|kuota|gratis|hemat|cashback|harga|beli|aktifkan",
    "kw_mendesak": r"segera|sekarang|hari ini|batas|terakhir|blokir|hangus|darurat|penting",
}
KEYWORD_RE = {k: re.compile(v, re.IGNORECASE) for k, v in KEYWORDS.items()}

META_FEATURES = [
    "panjang_log", "rasio_kapital", "rasio_angka", "ada_link", "ada_apk",
    "ada_nomor_hp", "ada_ussd", "jumlah_seru", *KEYWORDS.keys(),
]


def _features_one(text: str) -> list[float]:
    t = str(text)
    n = max(len(t), 1)
    letters = [c for c in t if c.isalpha()]
    feats = [
        math.log1p(len(t)),
        sum(c.isupper() for c in letters) / max(len(letters), 1),
        sum(c.isdigit() for c in t) / n,
        float(bool(RE_URL.search(t))),
        float(bool(RE_APK.search(t))),
        float(bool(RE_PHONE.search(t))),
        float(bool(RE_USSD.search(t))),
        min(t.count("!"), 5),
    ]
    feats += [len(rx.findall(t)) for rx in KEYWORD_RE.values()]
    return feats


def extract_features(texts) -> np.ndarray:
    return np.asarray([_features_one(t) for t in texts], dtype=float)


def meta_feature_names(transformer=None, input_features=None) -> np.ndarray:
    return np.asarray(META_FEATURES, dtype=object)


# ---------------------------------------------------------------------------
# 3. Aturan keputusan (threshold khusus kelas penipuan)
# ---------------------------------------------------------------------------
def decide(proba_row, classes, threshold: float, fraud_label: int = 1):
    """Penipuan kalau P(penipuan) >= threshold; selain itu ambil kelas
    lain (normal/promo) dengan probabilitas tertinggi.
    """
    classes = list(classes)
    i = classes.index(fraud_label)
    if proba_row[i] >= threshold:
        return fraud_label
    others = [j for j in range(len(classes)) if j != i]
    return classes[max(others, key=lambda j: proba_row[j])]


def decide_batch(proba, classes, threshold: float, fraud_label: int = 1) -> np.ndarray:
    return np.asarray([decide(p, classes, threshold, fraud_label) for p in proba])


# ---------------------------------------------------------------------------
# 4. Penjelasan (alasan) untuk pengguna bot / CLI
# ---------------------------------------------------------------------------
RULE_REASONS = [
    (lambda t: RE_APK.search(t), "Ada file/aplikasi .apk. Jangan dibuka atau di-install."),
    (lambda t: RE_URL.search(t), "Berisi link. Cek dulu alamat situsnya sebelum diklik."),
    (lambda t: KEYWORD_RE["kw_hadiah"].search(t), "Mengaku kamu menang hadiah atau undian."),
    (lambda t: KEYWORD_RE["kw_uang"].search(t), "Menyebut transfer, pulsa, rekening, atau biaya."),
    (lambda t: KEYWORD_RE["kw_rahasia"].search(t), "Menyebut PIN/kode/OTP. Jangan pernah bagikan ke siapa pun."),
    (lambda t: KEYWORD_RE["kw_mendesak"].search(t), "Memakai bahasa yang mendesak atau mengancam."),
]

META_LABEL = {
    "ada_link": "[ada link]",
    "ada_apk": "[ada .apk]",
    "ada_nomor_hp": "[ada nomor HP]",
    "ada_ussd": "[ada kode USSD]",
    "kw_hadiah": "[kata hadiah]",
    "kw_uang": "[kata uang/transfer]",
    "kw_aksi": "[ajakan klik/hubungi]",
    "kw_rahasia": "[kata PIN/OTP]",
    "kw_promo": "[kata promo]",
    "kw_mendesak": "[kata mendesak]",
    "jumlah_seru": "[banyak tanda seru]",
    "rasio_kapital": "[banyak huruf kapital]",
}


def rule_reasons(text: str) -> list[str]:
    t = str(text)
    return [msg for check, msg in RULE_REASONS if check(t)]


def explain_tokens(pipeline, text: str, label: int, top_k: int = 5) -> list[str]:
    """Fitur kata/meta yang paling mendorong prediksi ke `label`.
    Bekerja untuk model linear (coef_) dan Multinomial NB.
    """
    clf = pipeline.steps[-1][1]
    feats = pipeline[:-1]
    classes = list(clf.classes_)
    row = classes.index(label)
    if hasattr(clf, "coef_"):
        weights = clf.coef_[row]
    elif hasattr(clf, "feature_log_prob_"):
        flp = clf.feature_log_prob_
        others = [j for j in range(len(classes)) if j != row]
        weights = flp[row] - flp[others].mean(axis=0)
    else:
        return []
    X = feats.transform([text])
    X = X.toarray()[0] if hasattr(X, "toarray") else np.asarray(X)[0]
    names = feats.get_feature_names_out()
    contrib = X * weights
    out = []
    for idx in np.argsort(-contrib):
        if contrib[idx] <= 0 or len(out) >= top_k:
            break
        name = str(names[idx])
        prefix, _, base = name.rpartition("__")
        if prefix == "char":  # n-gram karakter sulit dibaca manusia
            continue
        if prefix == "meta":
            base = META_LABEL.get(base)
            if base is None:
                continue
        else:
            base = " ".join(TOKEN_LABEL.get(w, w) for w in base.split())
        if base not in out:
            out.append(base)
    return out


def analyze(bundle: dict, text: str) -> dict:
    pipe = bundle["pipeline"]
    labels = bundle["labels"]
    fraud = bundle.get("fraud_label", 1)
    proba = pipe.predict_proba([text])[0]
    classes = [int(c) for c in pipe.classes_]
    label = int(decide(proba, classes, bundle["threshold"], fraud))
    oleh_aturan = False
    if bundle.get("rule_override", False) and label != fraud and RE_APK.search(str(text)):
        label, oleh_aturan = fraud, True
    return {
        "label": label,
        "oleh_aturan": oleh_aturan,
        "label_name": labels[label],
        "proba": {labels[c]: float(p) for c, p in zip(classes, proba)},
        "alasan": rule_reasons(text),
        "kata_kunci": explain_tokens(pipe, text, label),
    }


STATUS = {
    "penipuan": "[PERINGATAN] MENCURIGAKAN: Kemungkinan PENIPUAN",
    "promo": "[PROMO] PROMO/IKLAN (umumnya tidak berbahaya)",
    "normal": "[AMAN] Tampak AMAN (pesan biasa)",
}


def format_reply(result: dict) -> str:
    lines = [STATUS.get(result["label_name"], result["label_name"])]
    if result.get("oleh_aturan"):
        lines.append("(ditandai oleh aturan keamanan: pesan berisi file .apk)")
    lines.append("")
    prob = " · ".join(f"{k} {v:.0%}" for k, v in result["proba"].items())
    lines.append(f"Keyakinan model: {prob}")
    if result["alasan"]:
        lines += ["", "Alasan:" if result["label_name"] == "penipuan" else "Catatan:"]
        lines += [f"• {r}" for r in result["alasan"]]
    if result["kata_kunci"]:
        lines += ["", "Paling berpengaruh: " + ", ".join(result["kata_kunci"])]
    lines += [
        "",
        "Ini prediksi otomatis, bukan jaminan. Jangan pernah membagikan PIN/OTP "
        "atau meng-install file .apk dari pesan yang tidak jelas.",
    ]
    return "\n".join(lines)
