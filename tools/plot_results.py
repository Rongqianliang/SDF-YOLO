# SPDX-License-Identifier: AGPL-3.0
"""Plot manuscript-reported mAP values from a CSV; this does not run inference."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def read_metrics(path: Path) -> list[tuple[str, float, float]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"dataset", "map50", "map50_95"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"CSV must contain columns: {', '.join(sorted(required))}")
        records = []
        seen = set()
        for line, row in enumerate(reader, start=2):
            name = (row.get("dataset") or "").strip()
            if not name or name in seen:
                raise ValueError(f"Empty or duplicate dataset at CSV line {line}")
            try:
                ap50 = float(row["map50"])
                ap = float(row["map50_95"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid mAP at CSV line {line}") from exc
            if not all(math.isfinite(v) and 0 <= v <= 1 for v in (ap50, ap)):
                raise ValueError(f"mAP must be stored as fractions in [0, 1], line {line}")
            seen.add(name)
            records.append((name, ap50 * 100, ap * 100))
    if not records:
        raise ValueError("CSV contains no data rows")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("results/paper_metrics.csv"))
    parser.add_argument("--out", type=Path, default=Path("outputs/paper_metrics.png"))
    args = parser.parse_args()
    try:
        records = read_metrics(args.csv)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(max(7.2, len(records) * 1.8), 4.4), dpi=200)
    width = 0.32
    for column, offset, label, color in (
        (1, -width / 2, "mAP50", "#3979A8"),
        (2, width / 2, "mAP50-95", "#C2733A"),
    ):
        bars = ax.bar(
            [i + offset for i in range(len(records))],
            [row[column] for row in records],
            width, label=label, color=color,
        )
        ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=9)
    ax.set_xticks(range(len(records)), [row[0] for row in records])
    ax.set_ylim(0, 105)
    ax.set_ylabel("mAP (%)")
    ax.set_title("SDF-YOLO: manuscript-reported results")
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {args.out.resolve()}")


if __name__ == "__main__":
    main()
