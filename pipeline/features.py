"""Tahap 5 — Feature Engineering & definisi kandidat model + metrik evaluasi."""
if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Callable

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, MaxAbsScaler
from sklearn.svm import LinearSVC

import sms_utils
from pipeline.config import FRAUD, SEED


def build_features() -> FeatureUnion:
    """FeatureUnion gabungan: TF-IDF kata (1-2), TF-IDF char_wb (3-5), dan fitur buatan."""
    return FeatureUnion([
        (
            "word",
            TfidfVectorizer(
                preprocessor=sms_utils.normalize_text,
                ngram_range=(1, 2),
                sublinear_tf=True,
                min_df=1,
            ),
        ),
        (
            "char",
            TfidfVectorizer(
                preprocessor=sms_utils.normalize_text,
                analyzer="char_wb",
                ngram_range=(3, 5),
                sublinear_tf=True,
                min_df=2,
            ),
        ),
        (
            "meta",
            make_pipeline(
                FunctionTransformer(
                    sms_utils.extract_features,
                    feature_names_out=sms_utils.meta_feature_names,
                ),
                MaxAbsScaler(),
            ),
        ),
    ])


def get_candidates(seed: int = SEED) -> dict[str, Callable[[], Pipeline]]:
    """Daftar factory kandidat model klasik untuk perbandingan di validation set."""
    return {
        "NaiveBayes": lambda: Pipeline([
            (
                "tfidf",
                TfidfVectorizer(
                    preprocessor=sms_utils.normalize_text,
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                ),
            ),
            ("clf", MultinomialNB(alpha=0.1)),
        ]),
        "LogReg_C1": lambda: Pipeline([
            ("features", build_features()),
            ("clf", LogisticRegression(C=1, class_weight="balanced", max_iter=3000, random_state=seed)),
        ]),
        "LogReg_C5": lambda: Pipeline([
            ("features", build_features()),
            ("clf", LogisticRegression(C=5, class_weight="balanced", max_iter=3000, random_state=seed)),
        ]),
        "LogReg_C20": lambda: Pipeline([
            ("features", build_features()),
            ("clf", LogisticRegression(C=20, class_weight="balanced", max_iter=3000, random_state=seed)),
        ]),
        "LinearSVM": lambda: Pipeline([
            ("features", build_features()),
            ("clf", LinearSVC(C=0.5, class_weight="balanced", random_state=seed)),
        ]),
    }


def compute_metrics(y_true, y_pred, fraud_label: int = FRAUD) -> dict[str, float]:
    """Menghitung akurasi, macro F1, serta recall/precision/F1 khusus kelas penipuan."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "recall_penipuan": float(
            recall_score(y_true, y_pred, labels=[fraud_label], average="macro", zero_division=0)
        ),
        "precision_penipuan": float(
            precision_score(y_true, y_pred, labels=[fraud_label], average="macro", zero_division=0)
        ),
        "f1_penipuan": float(
            f1_score(y_true, y_pred, labels=[fraud_label], average="macro", zero_division=0)
        ),
    }
