import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


def _load_rows(summary_path):
    rows = []
    with summary_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            parsed = {}
            for key, value in row.items():
                if key == "output_dir" or value == "":
                    parsed[key] = value
                else:
                    parsed[key] = float(value)
            rows.append(parsed)
    return sorted(rows, key=lambda item: item["conf"])


def _setup_font():
    font_path = Path(r"C:\Windows\Fonts\msyh.ttc")
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
    plt.rcParams["axes.unicode_minus"] = False


def _write_analysis(rows, analysis_path, figure_path):
    best_precision = max(rows, key=lambda item: item["precision_20"])
    best_auc = max(rows, key=lambda item: item["auc"])
    best_observation = max(rows, key=lambda item: item["observation_rate"])
    conf_list = ", ".join(f"{row['conf']:.2f}" for row in rows)
    lines = [
        "# UAVDT-S conf 扫描结果分析",
        "",
        f"- 当前 conf 组合为：{conf_list}。",
        f"- Precision@20 最优：conf={best_precision['conf']:.2f}，Precision@20={best_precision['precision_20']:.6f}。",
        f"- Success AUC 最优：conf={best_auc['conf']:.2f}，AUC={best_auc['auc']:.6f}。",
        f"- Observation Rate 最优：conf={best_observation['conf']:.2f}，Observation Rate={best_observation['observation_rate']:.6f}。",
        "- 趋势：在当前三组结果中，conf 越低，检测召回、观测率、Precision@20 和 AUC 越高，说明当前 UAVDT-S 测试更依赖低阈值召回小目标。",
        "- Found Rate 在标准 SOT 模式下通常为 1.0，只代表每帧都有输出框，不代表每帧都追准。",
        "- Longest Missing 当前三组都为 2044，说明至少有一个长片段主要依赖预测续接，后续需要按序列定位问题段。",
        "",
        f"折线图：`{figure_path}`",
    ]
    analysis_path.write_text("\n".join(lines), encoding="utf-8")


def plot_conf_sweep(summary_path, output_path, analysis_path):
    _setup_font()
    rows = _load_rows(summary_path)
    if not rows:
        raise ValueError(f"没有读取到 conf 扫描结果：{summary_path}")

    conf_values = [row["conf"] for row in rows]
    top_metrics = [
        ("precision_20", "Precision@20", "#2563eb"),
        ("auc", "Success AUC", "#dc2626"),
        ("success_0_5", "Success@0.5", "#16a34a"),
        ("observation_rate", "Observation Rate", "#9333ea"),
        ("detection_recall", "Detection Recall", "#f97316"),
    ]

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(11.5, 8.2),
        dpi=180,
        gridspec_kw={"height_ratios": [2.0, 1.2]},
    )

    ax = axes[0]
    for key, label, color in top_metrics:
        values = [row[key] for row in rows]
        ax.plot(conf_values, values, marker="o", linewidth=2.4, markersize=6, label=label, color=color)
        for x_value, y_value in zip(conf_values, values):
            ax.annotate(
                f"{y_value:.3f}",
                (x_value, y_value),
                textcoords="offset points",
                xytext=(0, 7),
                ha="center",
                fontsize=8,
                color=color,
            )

    ax.set_title("UAVDT-S 全序列：不同置信度阈值下的主要指标", fontsize=15, weight="bold")
    ax.set_xlabel("conf 阈值")
    ax.set_ylabel("比例 / AUC")
    ax.set_xticks(conf_values)
    ax.set_ylim(0.25, 1.03)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.45)
    ax.legend(loc="upper right", ncol=2, fontsize=9)

    ax2 = axes[1]
    prediction_values = [row["prediction_rate"] for row in rows]
    ax2.plot(
        conf_values,
        prediction_values,
        marker="o",
        linewidth=2.4,
        markersize=6,
        label="Prediction Rate",
        color="#0f766e",
    )
    for x_value, y_value in zip(conf_values, prediction_values):
        ax2.annotate(
            f"{y_value:.3f}",
            (x_value, y_value),
            textcoords="offset points",
            xytext=(0, 7),
            ha="center",
            fontsize=8,
            color="#0f766e",
        )
    ax2.set_xlabel("conf 阈值")
    ax2.set_ylabel("预测续接比例")
    ax2.set_xticks(conf_values)
    ax2.set_ylim(0.0, max(prediction_values) * 1.35)
    ax2.grid(True, linestyle="--", linewidth=0.6, alpha=0.45)
    ax2.legend(loc="upper left", fontsize=9)

    ax3 = ax2.twinx()
    cle_values = [row["mean_cle_found"] for row in rows]
    missing_values = [row["longest_missing_frames"] for row in rows]
    ax3.plot(conf_values, cle_values, marker="s", linewidth=2.0, markersize=5, label="Mean CLE", color="#64748b")
    ax3.plot(
        conf_values,
        missing_values,
        marker="^",
        linewidth=2.0,
        markersize=5,
        label="Longest Missing",
        color="#991b1b",
    )
    for x_value, y_value in zip(conf_values, cle_values):
        ax3.annotate(
            f"{y_value:.1f}",
            (x_value, y_value),
            textcoords="offset points",
            xytext=(0, -14),
            ha="center",
            fontsize=8,
            color="#64748b",
        )
    for x_value, y_value in zip(conf_values, missing_values):
        ax3.annotate(
            f"{int(y_value)}",
            (x_value, y_value),
            textcoords="offset points",
            xytext=(0, 7),
            ha="center",
            fontsize=8,
            color="#991b1b",
        )
    ax3.set_ylabel("像素 / 帧数")
    ax3.legend(loc="upper right", fontsize=9)

    fig.text(
        0.01,
        0.01,
        "注：Found Rate 在 SOT 模式下通常为 1.0，表示每帧都有输出框，不等同于每帧追准；"
        "若某组 FPS 为 0，多半来自 resume 跳过已完成序列，速度不用于比较。",
        fontsize=8,
        color="#475569",
    )
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(output_path)
    plt.close(fig)
    _write_analysis(rows, analysis_path, output_path)


def main():
    parser = argparse.ArgumentParser(description="Plot UAVDT-S conf sweep metrics.")
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("outputs/uavdt_sot_conf_sweep_full/conf_sweep_summary.csv"),
        help="conf_sweep_summary.csv 路径",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/uavdt_sot_conf_sweep_full/conf_sweep_metrics_line.png"),
        help="输出折线图路径",
    )
    parser.add_argument(
        "--analysis",
        type=Path,
        default=Path("outputs/uavdt_sot_conf_sweep_full/conf_sweep_analysis.md"),
        help="输出分析 Markdown 路径",
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plot_conf_sweep(args.summary, args.output, args.analysis)
    print(args.output.resolve())
    print(args.analysis.resolve())


if __name__ == "__main__":
    main()
