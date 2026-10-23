"""Draw the per-class F1 chart from the saved classification report.

Usage:
    python scripts/plot_results.py
"""

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "docs" / "results" / "classification-report.txt"
METRICS = ROOT / "docs" / "results" / "metrics.txt"
OUT = ROOT / "docs" / "images" / "per-class-f1.png"

PRETTY = {
    "american_football": "American football",
    "baseball": "Baseball",
    "basketball": "Basketball",
    "billiard_ball": "Billiard ball",
    "bowling_ball": "Bowling ball",
    "cricket_ball": "Cricket ball",
    "football": "Football",
    "golf_ball": "Golf ball",
    "hockey_ball": "Field hockey ball",
    "hockey_puck": "Hockey puck",
    "rugby_ball": "Rugby ball",
    "shuttlecock": "Shuttlecock",
    "table_tennis_ball": "Table tennis ball",
    "tennis_ball": "Tennis ball",
    "volleyball": "Volleyball",
}

ROW = re.compile(
    r"^\s*([a-z_]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+(\d+)\s*$"
)


def pretty(name):
    return PRETTY.get(name, name.replace("_", " ").title())


def parse_report(path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = ROW.match(line)
        if match:
            name, precision, recall, f1, support = match.groups()
            rows.append((name, float(precision), float(recall), float(f1), int(support)))
    if len(rows) != 15:
        raise ValueError(f"expected 15 class rows in {path}, found {len(rows)}")
    return rows


def parse_macro_f1(path):
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("Test Macro F1:"):
            return float(line.split(":", 1)[1])
    raise ValueError(f"could not find 'Test Macro F1' in {path}")


def main():
    rows = sorted(parse_report(REPORT), key=lambda r: r[3])
    macro_f1 = parse_macro_f1(METRICS)
    names = [pretty(r[0]) for r in rows]
    f1 = np.array([r[3] for r in rows])
    support = [r[4] for r in rows]

    fig, ax = plt.subplots(figsize=(10.5, 7.2))
    colors = plt.cm.RdYlGn(0.15 + 0.7 * (f1 - f1.min()) / (f1.max() - f1.min()))
    y = np.arange(len(rows))

    ax.barh(y, f1, color=colors, edgecolor="#333333", linewidth=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{n}  (n={s})" for n, s in zip(names, support)], fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_xlabel("F1 score")
    ax.set_title("Per-class F1 on the held-out test set (1,841 images)", fontsize=13,
                 pad=12)
    ax.grid(axis="x", alpha=0.3)
    ax.axvline(macro_f1, color="#333333", linestyle="--", linewidth=1,
               label=f"macro F1 = {macro_f1:.3f}")
    ax.legend(loc="lower right", fontsize=9, frameon=True)

    for yi, value in zip(y, f1):
        ax.text(0.012, yi, f"{value:.2f}", va="center", fontsize=8,
                color="white", fontweight="bold")

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
