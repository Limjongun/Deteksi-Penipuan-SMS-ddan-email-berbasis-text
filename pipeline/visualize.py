"""Visualisasi ringkasan metrik evaluasi (`metrics_summary.json`) dan
perbandingan antar versi (`v001` vs `v002`) ke gambar dashboard PNG.
"""
if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch
import numpy as np

from pipeline.config import (
    DEFAULT_VERSION,
    PLOTS_DIR,
    REPORTS_DIR,
    configure_utf8_stdout,
    get_version_config,
)

METRIC_ORDER = [
    ("accuracy", "Accuracy"),
    ("macro_f1", "Macro F1"),
    ("recall_penipuan", "Recall\nPenipuan"),
    ("precision_penipuan", "Precision\nPenipuan"),
    ("f1_penipuan", "F1\nPenipuan"),
]

METRIC_LABELS_SINGLE_LINE = {
    "accuracy": "Accuracy",
    "macro_f1": "Macro F1",
    "recall_penipuan": "Recall Penipuan",
    "precision_penipuan": "Precision Penipuan",
    "f1_penipuan": "F1 Penipuan",
}


def _smart_y_min(values: list[float]) -> float:
    lo = min(values)
    if lo >= 85.0:
        return 80.0
    if lo >= 70.0:
        return 60.0
    if lo >= 50.0:
        return 40.0
    return 0.0


def plot_metrics_summary(
    summary: dict,
    output_path: Path | str = PLOTS_DIR / "metrics_summary_dashboard.png",
) -> Path:
    """Membuat gambar dashboard visualisasi dari dictionary `metrics_summary.json`."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    version = summary.get("version", "v001")
    eval_split = summary.get("eval_split", "test")
    dataset_label = summary.get("dataset_label") or "dataset_sms_spam_v1.csv"
    model_name = summary.get("model_name", "Model")
    data_train = summary.get("data_train", "-")
    threshold = float(summary.get("threshold", 0.5))
    n_test = int(summary.get("n_test", 0))
    n_errors = int(summary.get("n_errors", 0))
    n_correct = max(n_test - n_errors, 0)

    m_thr = summary["metrics_test_threshold"]
    m_arg = summary["metrics_test_argmax"]

    bg_color = "#F8FAFC"
    card_bg = "#FFFFFF"
    border_col = "#E2E8F0"
    text_dark = "#0F172A"
    text_muted = "#475569"
    blue_main = "#2563EB"
    slate_ref = "#94A3B8"
    green_pos = "#10B981"
    red_neg = "#EF4444"
    amber_warn = "#F59E0B"

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.edgecolor": border_col,
        "axes.linewidth": 0.9,
    })

    fig = plt.figure(figsize=(14.5, 8.6), dpi=220, facecolor=bg_color)

    # Header
    fig.text(
        0.035, 0.955,
        f"Evaluasi Performa Model Detektor SMS Penipuan ({version})",
        fontsize=19, fontweight="bold", color=text_dark, va="top",
    )
    fig.text(
        0.035, 0.913,
        f"Model: {model_name}   •   Data Train: {data_train}   •   "
        f"Threshold Penipuan: {threshold:.2f}   •   Evaluasi: {eval_split} ({n_test:,} pesan)",
        fontsize=10.8, color=text_muted, va="top",
    )

    # Baris Atas: 4 Kartu KPI
    delta_acc = (m_thr["accuracy"] - m_arg["accuracy"]) * 100
    delta_f1 = (m_thr["macro_f1"] - m_arg["macro_f1"]) * 100
    delta_rec = (m_thr["recall_penipuan"] - m_arg["recall_penipuan"]) * 100
    err_rate = (n_errors / n_test * 100) if n_test else 0.0

    kpi_cards = [
        (
            f"ACCURACY (THR {threshold:.2f})",
            f"{m_thr['accuracy'] * 100:.2f}%",
            f"{delta_acc:+.2f} pp vs Argmax ({m_arg['accuracy'] * 100:.2f}%)",
            blue_main,
            text_muted,
        ),
        (
            f"MACRO F1 (THR {threshold:.2f})",
            f"{m_thr['macro_f1'] * 100:.2f}%",
            f"{delta_f1:+.2f} pp vs Argmax ({m_arg['macro_f1'] * 100:.2f}%)",
            "#0EA5E9",
            text_muted,
        ),
        (
            "RECALL PENIPUAN (PRIORITAS)",
            f"{m_thr['recall_penipuan'] * 100:.2f}%",
            f"{delta_rec:+.2f} pp vs Argmax ({m_arg['recall_penipuan'] * 100:.2f}%)",
            green_pos,
            "#047857" if delta_rec >= 0 else "#B45309",
        ),
        (
            "KESALAHAN TEST SET",
            f"{n_errors:,} / {n_test:,}",
            f"Error rate {err_rate:.2f}%  ({n_correct:,} prediksi benar)",
            red_neg,
            text_muted,
        ),
    ]

    card_w = 0.218
    card_h = 0.135
    card_y = 0.74
    gap = 0.019
    start_x = 0.035

    for idx, (label, val, sub, accent, sub_col) in enumerate(kpi_cards):
        cx = start_x + idx * (card_w + gap)
        ax_card = fig.add_axes([cx, card_y, card_w, card_h])
        ax_card.set_xlim(0, 1)
        ax_card.set_ylim(0, 1)
        ax_card.axis("off")

        ax_card.add_patch(
            FancyBboxPatch(
                (0.01, 0.03), 0.98, 0.94,
                boxstyle="round,pad=0.02,rounding_size=0.07",
                facecolor=card_bg, edgecolor=border_col, linewidth=1.1,
            )
        )
        ax_card.add_patch(
            FancyBboxPatch(
                (0.01, 0.05), 0.028, 0.90,
                boxstyle="round,pad=0.0,rounding_size=0.02",
                facecolor=accent, edgecolor="none",
            )
        )

        ax_card.text(0.08, 0.78, label, fontsize=8.4, fontweight="bold", color=text_muted, va="center")
        ax_card.text(0.08, 0.45, val, fontsize=19.5, fontweight="bold", color=text_dark, va="center")
        ax_card.text(0.08, 0.16, sub, fontsize=8.2, color=sub_col, va="center")

    # Panel Bawah Kiri: Grouped Bar Chart
    ax_bar_bg = fig.add_axes([0.035, 0.08, 0.565, 0.615])
    ax_bar_bg.set_xlim(0, 1)
    ax_bar_bg.set_ylim(0, 1)
    ax_bar_bg.axis("off")
    ax_bar_bg.add_patch(
        FancyBboxPatch(
            (0.002, 0.005), 0.996, 0.99,
            boxstyle="round,pad=0.005,rounding_size=0.02",
            facecolor=card_bg, edgecolor=border_col, linewidth=1.1,
        )
    )
    ax_bar_bg.text(
        0.035, 0.935,
        f"Perbandingan Metrik: Threshold {threshold:.2f} vs Argmax (0.50)",
        fontsize=12.2, fontweight="bold", color=text_dark, va="center",
    )

    ax_bar = fig.add_axes([0.085, 0.175, 0.495, 0.425])
    ax_bar.set_facecolor(card_bg)

    keys = [k for k, _ in METRIC_ORDER]
    x_labels = [lbl for _, lbl in METRIC_ORDER]
    thr_vals = [m_thr[k] * 100 for k in keys]
    arg_vals = [m_arg[k] * 100 for k in keys]
    y_min = _smart_y_min(thr_vals + arg_vals)
    y_top = 104.0 if y_min < 80 else 102.5

    x = np.arange(len(keys))
    width = 0.35

    bars_thr = ax_bar.bar(x - width / 2, thr_vals, width, color=blue_main, edgecolor="none", zorder=3)
    bars_arg = ax_bar.bar(x + width / 2, arg_vals, width, color=slate_ref, edgecolor="none", zorder=3)
    bars_thr[2].set_color(green_pos)

    ax_bar.set_ylim(y_min, y_top)
    step = 5 if y_min >= 80 else 10 if y_min >= 40 else 20
    yticks = list(range(int(y_min), 101, step))
    ax_bar.set_yticks(yticks)
    ax_bar.set_yticklabels([f"{t}%" for t in yticks], fontsize=9.3, color=text_muted)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(x_labels, fontsize=9.6, fontweight="bold", color=text_dark)
    ax_bar.yaxis.grid(True, linestyle="--", alpha=0.5, color="#CBD5E1", zorder=0)
    ax_bar.set_axisbelow(True)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)

    label_pad = (y_top - y_min) * 0.022
    for bar, val in zip(bars_thr, thr_vals):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2 - 0.025,
            val + label_pad,
            f"{val:.2f}%",
            ha="center", va="bottom", fontsize=7.9, fontweight="bold", color=text_dark,
        )
    for bar, val in zip(bars_arg, arg_vals):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2 + 0.025,
            val + label_pad,
            f"{val:.2f}%",
            ha="center", va="bottom", fontsize=7.8, color=text_muted,
        )

    legend_handles = [
        Patch(facecolor=blue_main, label=f"Threshold {threshold:.2f}"),
        Patch(facecolor=green_pos, label=f"Threshold {threshold:.2f} (Recall Prioritas)"),
        Patch(facecolor=slate_ref, label="Argmax (0.50)"),
    ]
    ax_bar.legend(
        handles=legend_handles,
        frameon=False, loc="upper left", bbox_to_anchor=(-0.02, 1.11),
        ncol=3, fontsize=8.6, handlelength=1.2, columnspacing=1.1,
    )
    fig.text(
        0.055, 0.098,
        f"Catatan: sumbu Y dimulai dari {int(y_min)}% untuk memperjelas selisih antar strategi keputusan.",
        fontsize=8.3, color=text_muted,
    )

    # Panel Bawah Kanan Atas: Donut Chart
    ax_donut_bg = fig.add_axes([0.62, 0.405, 0.345, 0.29])
    ax_donut_bg.set_xlim(0, 1)
    ax_donut_bg.set_ylim(0, 1)
    ax_donut_bg.axis("off")
    ax_donut_bg.add_patch(
        FancyBboxPatch(
            (0.004, 0.01), 0.992, 0.98,
            boxstyle="round,pad=0.005,rounding_size=0.03",
            facecolor=card_bg, edgecolor=border_col, linewidth=1.1,
        )
    )
    ax_donut_bg.text(
        0.045, 0.89,
        f"Ringkasan Prediksi (n = {n_test:,})",
        fontsize=11, fontweight="bold", color=text_dark, va="center",
    )

    ax_donut = fig.add_axes([0.635, 0.42, 0.17, 0.23])
    ax_donut.pie(
        [n_correct, n_errors],
        colors=[green_pos, red_neg],
        startangle=90,
        counterclock=False,
        wedgeprops={"width": 0.34, "edgecolor": card_bg, "linewidth": 2.5},
    )
    ax_donut.text(0, 0.08, f"{n_correct:,}/{n_test:,}", ha="center", va="center",
                  fontsize=11.5 if n_test >= 1000 else 13, fontweight="bold", color=text_dark)
    ax_donut.text(0, -0.22, "Benar", ha="center", va="center", fontsize=9, color=text_muted)
    ax_donut.set_aspect("equal")

    ax_donut_bg.scatter([0.56], [0.58], s=95, color=green_pos, zorder=3)
    ax_donut_bg.text(0.61, 0.60, f"Benar: {n_correct:,} pesan", fontsize=9.6, fontweight="bold",
                     color=text_dark, va="center")
    ax_donut_bg.text(0.61, 0.49, f"{n_correct / max(n_test, 1) * 100:.2f}% dari test set", fontsize=8.5,
                     color=text_muted, va="center")

    ax_donut_bg.scatter([0.56], [0.29], s=95, color=red_neg, zorder=3)
    ax_donut_bg.text(0.61, 0.31, f"Salah: {n_errors:,} pesan", fontsize=9.6, fontweight="bold",
                     color=text_dark, va="center")
    ax_donut_bg.text(0.61, 0.20, f"{err_rate:.2f}% error rate", fontsize=8.5,
                     color=text_muted, va="center")

    # Panel Bawah Kanan Bawah: Diverging Bar Chart Delta (pp)
    ax_delta_bg = fig.add_axes([0.62, 0.08, 0.345, 0.305])
    ax_delta_bg.set_xlim(0, 1)
    ax_delta_bg.set_ylim(0, 1)
    ax_delta_bg.axis("off")
    ax_delta_bg.add_patch(
        FancyBboxPatch(
            (0.004, 0.01), 0.992, 0.98,
            boxstyle="round,pad=0.005,rounding_size=0.03",
            facecolor=card_bg, edgecolor=border_col, linewidth=1.1,
        )
    )

    ax_delta = fig.add_axes([0.742, 0.155, 0.208, 0.175])
    ax_delta.set_facecolor(card_bg)

    delta_keys = ["recall_penipuan", "macro_f1", "accuracy", "f1_penipuan", "precision_penipuan"]
    delta_names = [METRIC_LABELS_SINGLE_LINE[k] for k in delta_keys]
    deltas = [(m_thr[k] - m_arg[k]) * 100 for k in delta_keys]
    colors_d = [green_pos if d >= 0 else amber_warn for d in deltas]

    max_abs = max(max(abs(d) for d in deltas), 2.0)
    lim = max(max_abs * 1.95, 6.5)

    y_pos = np.arange(len(delta_keys))[::-1]
    ax_delta.barh(y_pos, deltas, height=0.58, color=colors_d, zorder=3)
    ax_delta.axvline(0, color="#64748B", linewidth=1.0, zorder=4)
    ax_delta.set_xlim(-lim, lim)
    ax_delta.set_yticks(y_pos)
    ax_delta.set_yticklabels(delta_names, fontsize=8.4, color=text_dark)
    tick_val = round(max_abs, 0) if max_abs >= 3 else 2
    ax_delta.set_xticks([-tick_val, 0, tick_val])
    ax_delta.set_xticklabels([f"-{int(tick_val)}", "0", f"+{int(tick_val)} pp"], fontsize=8.0, color=text_muted)
    ax_delta.xaxis.grid(True, linestyle=":", alpha=0.6, color="#CBD5E1", zorder=0)
    ax_delta.spines["top"].set_visible(False)
    ax_delta.spines["right"].set_visible(False)
    ax_delta.spines["left"].set_visible(False)
    ax_delta.tick_params(axis="y", length=0, pad=6)

    for yp, d in zip(y_pos, deltas):
        offset = lim * 0.035 if d >= 0 else -lim * 0.035
        ha = "left" if d >= 0 else "right"
        ax_delta.text(
            d + offset, yp, f"{d:+.2f} pp",
            va="center", ha=ha, fontsize=7.9, fontweight="bold",
            color="#047857" if d >= 0 else "#B45309",
        )

    ax_delta_bg.text(
        0.045, 0.90,
        f"Dampak Threshold {threshold:.2f} vs Argmax (pp)",
        fontsize=10.6, fontweight="bold", color=text_dark, va="center",
    )
    ax_delta_bg.text(
        0.045, 0.065,
        f"Selisih Macro F1 {delta_f1:+.2f} pp | Recall Penipuan {delta_rec:+.2f} pp vs Argmax.",
        fontsize=7.9, color=text_muted, va="center",
    )

    # Footer
    fig.text(
        0.035, 0.028,
        f"Versi: {version}  |  Dataset: {dataset_label}",
        fontsize=8.3, color="#64748B",
    )

    fig.savefig(output_path, dpi=220, facecolor=bg_color)
    plt.close(fig)
    return output_path


def plot_version_comparison(
    v001_summary: dict,
    v002_bench_summary: dict,
    output_path: Path | str,
) -> Path:
    """Membuat visualisasi perbandingan langsung v001 vs v002 pada test set v001 (162 SMS)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    bg_color = "#F8FAFC"
    card_bg = "#FFFFFF"
    border_col = "#E2E8F0"
    text_dark = "#0F172A"
    text_muted = "#475569"
    col_v1 = "#64748B"
    col_v2 = "#2563EB"
    green_pos = "#10B981"
    red_neg = "#EF4444"

    m1 = v001_summary["metrics_test_threshold"]
    m2 = v002_bench_summary["metrics_test_threshold"]

    keys = [k for k, _ in METRIC_ORDER]
    x_labels = [lbl for _, lbl in METRIC_ORDER]
    v1_vals = [m1[k] * 100 for k in keys]
    v2_vals = [m2[k] * 100 for k in keys]
    deltas = [b - a for a, b in zip(v1_vals, v2_vals)]

    fig = plt.figure(figsize=(13.5, 6.8), dpi=220, facecolor=bg_color)

    fig.text(
        0.04, 0.945,
        "Perbandingan Model v001 vs v002 pada Benchmark Test Set v001 (162 SMS)",
        fontsize=17, fontweight="bold", color=text_dark, va="top",
    )
    fig.text(
        0.04, 0.895,
        f"v001: {v001_summary['model_name']} (thr {v001_summary['threshold']:.2f}, error {v001_summary['n_errors']}/162)   vs   "
        f"v002: {v002_bench_summary['model_name']} (thr {v002_bench_summary['threshold']:.2f}, error {v002_bench_summary['n_errors']}/162)",
        fontsize=10.5, color=text_muted, va="top",
    )

    # Card kiri: Grouped bar chart
    ax_bg1 = fig.add_axes([0.04, 0.10, 0.58, 0.74])
    ax_bg1.set_xlim(0, 1)
    ax_bg1.set_ylim(0, 1)
    ax_bg1.axis("off")
    ax_bg1.add_patch(
        FancyBboxPatch(
            (0.002, 0.005), 0.996, 0.99,
            boxstyle="round,pad=0.005,rounding_size=0.02",
            facecolor=card_bg, edgecolor=border_col, linewidth=1.1,
        )
    )
    ax_bg1.text(0.035, 0.935, "Perbandingan Metrik (Test Set v001)", fontsize=11.8, fontweight="bold", color=text_dark)

    ax1 = fig.add_axes([0.09, 0.19, 0.505, 0.54])
    ax1.set_facecolor(card_bg)

    x = np.arange(len(keys))
    width = 0.35
    b1 = ax1.bar(x - width / 2, v1_vals, width, color=col_v1, label=f"v001 (Wibisono saja)", zorder=3)
    b2 = ax1.bar(x + width / 2, v2_vals, width, color=col_v2, label=f"v002 (Wibisono + SpamDam)", zorder=3)

    y_min = _smart_y_min(v1_vals + v2_vals)
    y_top = 104.5 if y_min < 80 else 103.0
    ax1.set_ylim(y_min, y_top)
    step = 5 if y_min >= 80 else 10 if y_min >= 40 else 20
    yticks = list(range(int(y_min), 101, step))
    ax1.set_yticks(yticks)
    ax1.set_yticklabels([f"{t}%" for t in yticks], fontsize=9.2, color=text_muted)
    ax1.set_xticks(x)
    ax1.set_xticklabels(x_labels, fontsize=9.5, fontweight="bold", color=text_dark)
    ax1.yaxis.grid(True, linestyle="--", alpha=0.5, color="#CBD5E1", zorder=0)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    pad = (y_top - y_min) * 0.022
    for bar, val in zip(b1, v1_vals):
        ax1.text(bar.get_x() + bar.get_width() / 2, val + pad, f"{val:.1f}%",
                 ha="center", va="bottom", fontsize=8.4, color=text_muted)
    for bar, val in zip(b2, v2_vals):
        ax1.text(bar.get_x() + bar.get_width() / 2, val + pad, f"{val:.1f}%",
                 ha="center", va="bottom", fontsize=8.5, fontweight="bold", color=text_dark)

    ax1.legend(frameon=False, loc="upper left", bbox_to_anchor=(-0.02, 1.11), ncol=2, fontsize=9.0)

    # Card kanan: Delta v002 - v001
    ax_bg2 = fig.add_axes([0.64, 0.10, 0.32, 0.74])
    ax_bg2.set_xlim(0, 1)
    ax_bg2.set_ylim(0, 1)
    ax_bg2.axis("off")
    ax_bg2.add_patch(
        FancyBboxPatch(
            (0.004, 0.005), 0.992, 0.99,
            boxstyle="round,pad=0.005,rounding_size=0.025",
            facecolor=card_bg, edgecolor=border_col, linewidth=1.1,
        )
    )
    ax_bg2.text(0.055, 0.935, "Perubahan v002 vs v001 (pp)", fontsize=11.5, fontweight="bold", color=text_dark)

    ax2 = fig.add_axes([0.755, 0.22, 0.185, 0.50])
    ax2.set_facecolor(card_bg)

    y_pos = np.arange(len(keys))[::-1]
    single_names = [METRIC_LABELS_SINGLE_LINE[k] for k in keys]
    colors_d = [green_pos if d >= 0 else red_neg for d in deltas]
    max_abs = max(max(abs(d) for d in deltas), 2.0)
    lim = max(max_abs * 1.95, 6.0)

    ax2.barh(y_pos, deltas, height=0.52, color=colors_d, zorder=3)
    ax2.axvline(0, color="#64748B", linewidth=1.0, zorder=4)
    ax2.set_xlim(-lim, lim)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(single_names, fontsize=8.6, color=text_dark)
    ax2.xaxis.grid(True, linestyle=":", alpha=0.6, color="#CBD5E1", zorder=0)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.tick_params(axis="y", length=0, pad=6)

    for yp, d in zip(y_pos, deltas):
        offset = lim * 0.04 if d >= 0 else -lim * 0.04
        ha = "left" if d >= 0 else "right"
        ax2.text(d + offset, yp, f"{d:+.2f} pp", va="center", ha=ha,
                 fontsize=8.2, fontweight="bold", color="#047857" if d >= 0 else "#B91C1C")

    fig.text(
        0.04, 0.035,
        "Evaluasi bebas data leakage: grup test set v001 dikunci agar tidak masuk train/val v002.",
        fontsize=8.5, color="#64748B",
    )

    fig.savefig(output_path, dpi=220, facecolor=bg_color)
    plt.close(fig)
    return output_path


def main() -> None:
    configure_utf8_stdout()
    parser = argparse.ArgumentParser(description="Buat gambar visualisasi dari metrics_summary.json.")
    parser.add_argument(
        "--version",
        default=DEFAULT_VERSION,
        help=f"Versi artefak (default: {DEFAULT_VERSION})",
    )
    parser.add_argument("--input", type=Path, default=None, help="Path file JSON ringkasan metrik (opsional)")
    parser.add_argument("--output", type=Path, default=None, help="Path output gambar PNG (opsional)")
    args = parser.parse_args()

    cfg = get_version_config(args.version)
    in_path = args.input or (cfg["reports_dir"] / "metrics_summary.json")
    out_path = args.output or (cfg["plots_dir"] / "metrics_summary_dashboard.png")

    summary = json.loads(in_path.read_text(encoding="utf-8"))
    saved = plot_metrics_summary(summary, output_path=out_path)
    print(f"Dashboard metrik tersimpan di : {saved}")


if __name__ == "__main__":
    main()
