"""Tahap Opsional — Fine-tune IndoBERT sebagai pembanding pada split yang sama.

Membutuhkan paket opsional: `pip install torch transformers accelerate`.
"""
if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import pandas as pd
from sklearn.metrics import classification_report, f1_score, recall_score

import sms_utils
from pipeline.config import (
    FRAUD,
    INDOBERT_DIR,
    LABELS,
    MODEL_PATH,
    SEED,
    SPLITS_DIR,
    TARGET_NAMES,
    configure_utf8_stdout,
    ensure_dirs,
    set_seed,
)
from pipeline.features import compute_metrics
from pipeline.preprocess import load_splits

MODEL_ID = "indobenchmark/indobert-base-p1"


def run_indobert(
    splits: dict[str, pd.DataFrame],
    classic_model_name: str | None = None,
    classic_metrics_test: dict | None = None,
    seed: int = SEED,
    epochs: int = 4,
) -> dict:
    """Melatih IndoBERT pada train/val dan mengevaluasinya pada test set yang sama."""
    try:
        import torch
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise ImportError(
            "Paket opsional belum terpasang. Pasang terlebih dahulu:\n"
            "  pip install torch transformers accelerate"
        ) from exc

    train_df = splits["train"]
    val_df = splits["val"]
    test_df = splits["test"]

    tok = AutoTokenizer.from_pretrained(MODEL_ID)

    class SMSDataset(torch.utils.data.Dataset):
        def __init__(self, frame: pd.DataFrame):
            self.enc = tok(list(frame["teks"].map(sms_utils.normalize_text)), truncation=True, max_length=128)
            self.labels = list(frame["label"].astype(int))

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, i):
            item = {k: v[i] for k, v in self.enc.items()}
            item["labels"] = self.labels[i]
            return item

    def compute_bert_metrics(p):
        preds = p.predictions.argmax(-1)
        return {
            "macro_f1": f1_score(p.label_ids, preds, average="macro"),
            "recall_penipuan": recall_score(p.label_ids, preds, labels=[FRAUD], average="macro", zero_division=0),
        }

    bert = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        num_labels=len(LABELS),
        id2label=LABELS,
        label2id={v: k for k, v in LABELS.items()},
    )
    args = TrainingArguments(
        output_dir=str(INDOBERT_DIR),
        num_train_epochs=epochs,
        learning_rate=3e-5,
        weight_decay=0.01,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        save_total_limit=1,
        logging_steps=20,
        report_to="none",
        seed=seed,
    )
    trainer = Trainer(
        model=bert,
        args=args,
        train_dataset=SMSDataset(train_df),
        eval_dataset=SMSDataset(val_df),
        data_collator=DataCollatorWithPadding(tok),
        compute_metrics=compute_bert_metrics,
    )
    trainer.train()

    pred_bert = trainer.predict(SMSDataset(test_df)).predictions.argmax(-1)
    bert_metrics = compute_metrics(test_df["label"], pred_bert)

    print("\n=== Evaluasi IndoBERT (Test Set) ===")
    print(classification_report(test_df["label"], pred_bert, target_names=TARGET_NAMES, digits=3))

    if classic_metrics_test:
        col_name = classic_model_name or "ModelKlasik"
        print("\n=== Perbandingan Model Klasik vs IndoBERT ===")
        print(pd.DataFrame({col_name: classic_metrics_test, "IndoBERT": bert_metrics}).round(3).to_string())

    return bert_metrics


if __name__ == "__main__":
    configure_utf8_stdout()
    set_seed(SEED)
    ensure_dirs()
    splits = load_splits(SPLITS_DIR)

    classic_name, classic_metrics = None, None
    if MODEL_PATH.exists():
        bundle = joblib.load(MODEL_PATH)
        classic_name = bundle.get("model_name")
        classic_metrics = bundle.get("metrics_test")

    run_indobert(splits, classic_model_name=classic_name, classic_metrics_test=classic_metrics, seed=SEED)
