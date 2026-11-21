"""CLI inferensi lokal untuk menguji pesan SMS/WA menggunakan model tersimpan.

Contoh pemakaian:
  python predict.py "SELAMAT! Anda menang undian Rp100jt, klik bit.ly/hadiah"
  python predict.py --version v001 "Bro nanti jadi futsal jam 8?"
  python predict.py --version v002 --json "Your bank account has been suspended..."
  python predict.py --interactive
"""
import argparse
import json
from pathlib import Path

import joblib

import sms_utils
from pipeline.config import DEFAULT_VERSION, configure_utf8_stdout, get_version_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prediksi pesan SMS/WA (normal / penipuan / promo).")
    parser.add_argument("text", nargs="?", help="Teks pesan yang ingin diperiksa")
    parser.add_argument(
        "--version",
        default=DEFAULT_VERSION,
        help=f"Versi model yang dipakai: v001, v002, dll (default: {DEFAULT_VERSION})",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Override path file model .joblib (opsional)",
    )
    parser.add_argument("--json", action="store_true", help="Tampilkan hasil dalam format JSON")
    parser.add_argument("-i", "--interactive", action="store_true", help="Mode interaktif di terminal")
    return parser.parse_args()


def main() -> None:
    configure_utf8_stdout()
    args = parse_args()

    model_path = args.model or get_version_config(args.version)["model_path"]
    if not model_path.exists():
        raise SystemExit(
            f"Model belum ditemukan di {model_path}. Jalankan `python run_pipeline.py --version {args.version}` terlebih dahulu."
        )

    bundle = joblib.load(model_path)

    if args.interactive or not args.text:
        print(f"Mode interaktif detektor SMS/WA ({bundle.get('version', args.version)}). Ketik 'exit' atau Ctrl+C untuk keluar.\n")
        while True:
            try:
                text = input("Masukkan pesan > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nSelesai.")
                break
            if not text or text.lower() in ("exit", "quit", "keluar"):
                break
            result = sms_utils.analyze(bundle, text)
            if args.json:
                print(json.dumps(result, indent=2, ensure_ascii=False))
            else:
                print("\n" + sms_utils.format_reply(result) + "\n")
    else:
        result = sms_utils.analyze(bundle, args.text)
        if args.json:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print(sms_utils.format_reply(result))


if __name__ == "__main__":
    main()
