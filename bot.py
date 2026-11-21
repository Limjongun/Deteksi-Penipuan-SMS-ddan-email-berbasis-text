"""Bot Telegram detektor pesan penipuan SMS/WA.

Cara pakai:
  1. Jalankan `python run_pipeline.py` terlebih dahulu agar model terbentuk.
  2. Buat bot lewat @BotFather di Telegram, salin token-nya.
  3. Set environment variable TELEGRAM_BOT_TOKEN (dan opsional MODEL_VERSION=v002 / MODEL_PATH):
       PowerShell : $env:TELEGRAM_BOT_TOKEN="123:ABC..."
       Linux/Mac  : export TELEGRAM_BOT_TOKEN="123:ABC..."
  4. Jalankan dari folder proyek: python bot.py
"""
import logging
import os
from pathlib import Path

import joblib
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

import sms_utils  # wajib ada di folder yang sama (dipakai di dalam pipeline model)
from pipeline.config import DEFAULT_VERSION, get_version_config

_VERSION = os.environ.get("MODEL_VERSION", DEFAULT_VERSION)
DEFAULT_MODEL = get_version_config(_VERSION)["model_path"]
MODEL_PATH = Path(os.environ.get("MODEL_PATH", str(DEFAULT_MODEL)))

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("bot")

WELCOME = (
    "Halo.\n\n"
    "Teruskan atau tempel pesan SMS/WhatsApp yang mencurigakan ke sini. "
    "Sistem akan memeriksa apakah pesan tersebut termasuk PENIPUAN, PROMO, atau pesan NORMAL.\n\n"
    "Isi pesan Anda tidak disimpan oleh sistem."
)

BUNDLE: dict | None = None


def load_bundle() -> dict:
    if not MODEL_PATH.exists():
        raise SystemExit(
            f"Model tidak ditemukan di {MODEL_PATH}. Jalankan `python run_pipeline.py --version {_VERSION}` dulu."
        )
    return joblib.load(MODEL_PATH)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.effective_message.reply_text(WELCOME)


async def cek_pesan(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.effective_message
    text = msg.text or msg.caption
    if not text:
        await msg.reply_text("Kirim pesannya dalam bentuk teks, ya. Gambar/stiker belum bisa saya baca.")
        return
    result = sms_utils.analyze(BUNDLE, text)
    # Log hanya hasil prediksi, bukan isi pesan (privasi pengguna)
    log.info("prediksi=%s proba=%s", result["label_name"], {k: round(v, 3) for k, v in result["proba"].items()})
    await msg.reply_text(sms_utils.format_reply(result))


def main() -> None:
    global BUNDLE
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise SystemExit("Set environment variable TELEGRAM_BOT_TOKEN dulu (token dari @BotFather).")
    BUNDLE = load_bundle()
    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler(["start", "help"], start))
    app.add_handler(MessageHandler((filters.TEXT | filters.CAPTION) & ~filters.COMMAND, cek_pesan))
    log.info("Bot berjalan (model: %s). Tekan Ctrl+C untuk berhenti.", MODEL_PATH)
    app.run_polling()


if __name__ == "__main__":
    main()
