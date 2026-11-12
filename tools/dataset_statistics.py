# SPDX-License-Identifier: AGPL-3.0
"""Create object-size and category-frequency statistics for YOLO datasets.

Example:
    python tools/dataset_statistics.py --dataset SIMD=data/SIMD.yaml --split test

The size bins use sqrt(box_area) in pixels, which is an equivalent square-side
measure at the original image resolution. Defaults are 32 and 96 pixels;
these descriptive bins do not compute an evaluation metric. Labels are
expected in normalized YOLO format:
    class_id center_x center_y width height
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit("PyYAML is required. Install it with: pip install pyyaml") from exc

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Dataset YAML must contain a mapping: {path}")
    return data


def resolve_path(value: str | Path, base: Path) -> Path:
    path = Path(str(value).strip().strip('"\''))
    return path if path.is_absolute() else (base / path).resolve()


def class_names(raw: Any) -> dict[int, str]:
    if isinstance(raw, list):
        result = {index: str(name) for index, name in enumerate(raw)}
    elif isinstance(raw, dict):
        if any(str(index) != str(int(index)) for index in raw):
            raise ValueError("Class IDs must be integers")
        result = {int(index): str(name) for index, name in raw.items()}
    else:
        raise ValueError("Dataset YAML requires a names list or mapping")
    if not result or set(result) != set(range(len(result))):
        raise ValueError("Class IDs must be consecutive integers starting at zero")
    if any(not name.strip() for name in result.values()):
        raise ValueError("Class names cannot be empty")
    if len(set(result.values())) != len(result):
        raise ValueError("Class names must be unique")
    return result


def image_paths(split_value: Any, base: Path) -> list[Path]:
    values = split_value if isinstance(split_value, list) else [split_value]
    paths: list[Path] = []
    for value in values:
        path = resolve_path(value, base)
        if path.is_file() and path.suffix.lower() in {".txt", ".list"}:
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    candidate = resolve_path(line, path.parent)
                    if not candidate.is_file() or candidate.suffix.lower() not in IMAGE_EXTENSIONS:
                        raise FileNotFoundError(f"Missing or unsupported image in {path}: {candidate}")
                    paths.append(candidate)
        elif path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            paths.append(path)
        elif path.is_dir():
            paths.extend(item for item in path.rglob("*")
                         if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS)
        else:
            raise FileNotFoundError(f"Split path does not exist or is unsupported: {path}")
    return sorted(set(paths))


def infer_label_path(image: Path) -> Path:
    parts = list(image.parts)
    lowered = [part.lower() for part in parts]
    if "images" in lowered:
        index = len(lowered) - 1 - lowered[::-1].index("images")
        parts[index] = "labels"
        return Path(*parts).with_suffix(".txt")
    return image.with_suffix(".txt")


def read_yolo_labels(label_path: Path) -> Iterable[tuple[int, float, float, float, float]]:
    if not label_path.exists():
        raise FileNotFoundError(
            f"Missing label file: {label_path}. Use an empty file for a background image."
        )
    records = []
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        fields = line.split()
        if not fields or line.lstrip().startswith("#"):
            continue
        if len(fields) != 5:
            raise ValueError(f"Expected 5 YOLO fields in {label_path}:{line_number}")
        try:
            raw_class = float(fields[0])
            cx, cy, width, height = (float(value) for value in fields[1:5])
        except ValueError as exc:
            raise ValueError(f"Invalid YOLO label in {label_path}:{line_number}") from exc
        if not all(math.isfinite(v) for v in (raw_class, cx, cy, width, height)):
            raise ValueError(f"Non-finite label value in {label_path}:{line_number}")
        if not raw_class.is_integer() or raw_class < 0:
            raise ValueError(f"Invalid class ID in {label_path}:{line_number}")
        class_id = int(raw_class)
        if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < width <= 1 and 0 < height <= 1):
            raise ValueError(f"Expected normalized coordinates and positive sizes in {label_path}:{line_number}")
        records.append((class_id, cx, cy, width, height))
    return records


def size_label(side: float, small_max: float, medium_max: float) -> str:
    if side < small_max:
        return "small"
    if side < medium_max:
        return "medium"
    return "large"


def analyse_dataset(
    name: str,
    yaml_path: Path,
    split: str,
    small_max: float,
    medium_max: float,
) -> tuple[list[dict[str, Any]], dict[int, int], dict[int, str], dict[str, int]]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise SystemExit("Pillow is required. Install it with: pip install pillow") from exc

    config = load_yaml(yaml_path)
    dataset_root = resolve_path(config.get("path", "."), yaml_path.parent)
    if split == "all":
        split_values = [config[key] for key in ("train", "val", "test") if key in config]
        if not split_values:
            raise KeyError(f"{yaml_path} has no train/val/test split")
        images = []
        for value in split_values:
            images.extend(image_paths(value, dataset_root))
        images = sorted(set(images))
    elif split not in config:
        available = ", ".join(key for key in ("train", "val", "test") if key in config)
        raise KeyError(f"{yaml_path} has no '{split}' split. Available splits: {available}")
    else:
        images = image_paths(config[split], dataset_root)
    if not images:
        raise ValueError(f"No images found for {name}, split={split}")
    names = class_names(config.get("names"))
    if "nc" in config and config["nc"] != len(names):
        raise ValueError(f"nc does not match the number of class names: {yaml_path}")
    rows: list[dict[str, Any]] = []
    class_counts: Counter[int] = Counter()
    background_images = 0

    for image_path in images:
        try:
            with Image.open(image_path) as image:
                image_width, image_height = image.size
        except Exception as exc:
            raise RuntimeError(f"Could not read image: {image_path}") from exc

        label_path = infer_label_path(image_path)
        labels = list(read_yolo_labels(label_path))
        if not labels:
            background_images += 1
        for class_id, _cx, _cy, width, height in labels:
            if class_id not in names:
                raise ValueError(f"Class ID {class_id} is not defined in {yaml_path}: {label_path}")
            box_width = width * image_width
            box_height = height * image_height
            side = math.sqrt(max(box_width * box_height, 0.0))
            category = size_label(side, small_max, medium_max)
            class_counts[class_id] += 1
            rows.append(
                {
                    "dataset": name,
                    "image": str(image_path),
                    "class_id": class_id,
                    "class_name": names.get(class_id, f"class_{class_id}"),
                    "box_width_px": box_width,
                    "box_height_px": box_height,
                    "equivalent_side_px": side,
                    "size_category": category,
                }
            )

    return rows, dict(class_counts), names, {
        "images": len(images), "background_images": background_images,
    }


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_size_distribution(rows_by_dataset: dict[str, list[dict[str, Any]]], output: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:
        raise SystemExit("Plotting requires matplotlib and numpy. Install them with: pip install matplotlib numpy") from exc

    labels = ["small", "medium", "large"]
    datasets = list(rows_by_dataset)
    values = []
    for dataset in datasets:
        counter = Counter(row["size_category"] for row in rows_by_dataset[dataset])
        total = sum(counter.values()) or 1
        values.append([100 * counter[label] / total for label in labels])

    x = np.arange(len(datasets))
    width = 0.23
    fig, ax = plt.subplots(figsize=(8.2, 4.8), dpi=220)
    for index, label in enumerate(labels):
        ax.bar(x + (index - 1) * width, [row[index] for row in values], width, label=label.title())
    ax.set_xticks(x, datasets)
    ax.set_ylabel("Objects (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Object-size distribution")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncols=3)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def plot_class_distribution(
    rows_by_dataset: dict[str, list[dict[str, Any]]], output: Path
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:
        raise SystemExit("Plotting requires matplotlib and numpy. Install them with: pip install matplotlib numpy") from exc

    datasets = list(rows_by_dataset)
    fig, axes = plt.subplots(len(datasets), 1, figsize=(9.0, max(3.4, 2.6 * len(datasets))), squeeze=False)
    for axis, dataset in zip(axes[:, 0], datasets):
        counts = Counter(row["class_name"] for row in rows_by_dataset[dataset])
        ordered = sorted(counts.items(), key=lambda item: item[1], reverse=True)
        names = [item[0] for item in ordered]
        values = [item[1] for item in ordered]
        axis.bar(np.arange(len(names)), values, color="#4C78A8")
        axis.set_title(dataset, loc="left", fontsize=10, fontweight="bold")
        axis.set_ylabel("Instances")
        axis.set_xticks(np.arange(len(names)), names, rotation=45, ha="right", fontsize=8)
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Category-frequency distribution", y=1.01)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def parse_dataset(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Use DATASET_NAME=PATH_TO_DATA_YAML")
    name, path = value.split("=", 1)
    if not name.strip() or not path.strip():
        raise argparse.ArgumentTypeError("Dataset name and YAML path cannot be empty")
    return name.strip(), Path(path.strip().strip('"\''))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", required=True, type=parse_dataset,
                        help="Repeatable DATASET_NAME=PATH_TO_DATA_YAML argument")
    parser.add_argument("--split", default="test", choices=("train", "val", "test", "all"),
                        help="Dataset split to analyse; use all for train+val+test (default: test)")
    parser.add_argument("--small-max", type=float, default=32.0,
                        help="Small-object upper equivalent-side threshold in pixels")
    parser.add_argument("--medium-max", type=float, default=96.0,
                        help="Medium-object upper equivalent-side threshold in pixels")
    parser.add_argument("--out-dir", type=Path, default=Path("dataset_statistics"),
                        help="Output directory")
    args = parser.parse_args()
    if not (math.isfinite(args.small_max) and math.isfinite(args.medium_max)
            and 0 < args.small_max < args.medium_max):
        parser.error("Require 0 < --small-max < --medium-max")
    dataset_names = [name for name, _ in args.dataset]
    if len(set(dataset_names)) != len(dataset_names):
        parser.error("Each dataset name must be unique")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows_by_dataset: dict[str, list[dict[str, Any]]] = {}
    class_counts_by_dataset: dict[str, dict[int, int]] = {}
    names_by_dataset: dict[str, dict[int, str]] = {}
    image_stats_by_dataset: dict[str, dict[str, int]] = {}

    for name, yaml_path in args.dataset:
        try:
            rows, counts, names, image_stats = analyse_dataset(
                name, yaml_path.resolve(), args.split, args.small_max, args.medium_max
            )
        except (OSError, ValueError, KeyError, RuntimeError) as exc:
            parser.error(str(exc))
        rows_by_dataset[name] = rows
        class_counts_by_dataset[name] = counts
        names_by_dataset[name] = names
        image_stats_by_dataset[name] = image_stats
        print(f"{name}: {image_stats['images']} images, {len(rows)} instances")

    all_rows = [row for rows in rows_by_dataset.values() for row in rows]
    write_csv(
        args.out_dir / "object_statistics.csv",
        all_rows,
        ["dataset", "image", "class_id", "class_name", "box_width_px", "box_height_px",
         "equivalent_side_px", "size_category"],
    )

    summary_rows = []
    class_rows = []
    for dataset, rows in rows_by_dataset.items():
        size_counts = Counter(row["size_category"] for row in rows)
        total = len(rows) or 1
        summary_rows.append({
            "dataset": dataset,
            "images": image_stats_by_dataset[dataset]["images"],
            "background_images": image_stats_by_dataset[dataset]["background_images"],
            "instances": len(rows),
            "small": size_counts["small"],
            "medium": size_counts["medium"],
            "large": size_counts["large"],
            "small_percent": round(100 * size_counts["small"] / total, 3),
            "medium_percent": round(100 * size_counts["medium"] / total, 3),
            "large_percent": round(100 * size_counts["large"] / total, 3),
        })
        for class_id in sorted(names_by_dataset[dataset]):
            count = class_counts_by_dataset[dataset].get(class_id, 0)
            class_rows.append({
                "dataset": dataset,
                "class_id": class_id,
                "class_name": names_by_dataset[dataset].get(class_id, f"class_{class_id}"),
                "instances": count,
            })

    write_csv(args.out_dir / "size_summary.csv", summary_rows,
              list(summary_rows[0].keys()) if summary_rows else ["dataset"])
    write_csv(args.out_dir / "class_counts.csv", class_rows,
              ["dataset", "class_id", "class_name", "instances"])
    plot_size_distribution(rows_by_dataset, args.out_dir / "size_distribution.png")
    plot_class_distribution(rows_by_dataset, args.out_dir / "class_distribution.png")

    metadata = {
        "split": args.split,
        "size_measure": "sqrt(box_width_px * box_height_px)",
        "resolution": "original image resolution; no detector resizing",
        "bin_boundaries": "small < small_max; small_max <= medium < medium_max; large >= medium_max",
        "image_statistics": image_stats_by_dataset,
        "small_max": args.small_max,
        "medium_max": args.medium_max,
        "datasets": {name: str(path.resolve()) for name, path in args.dataset},
    }
    (args.out_dir / "statistics_config.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved statistics to: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
