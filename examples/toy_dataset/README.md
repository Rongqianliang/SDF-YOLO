# Synthetic Example

This is a CPU-only smoke-test dataset for the statistics tool. It contains two 400 x 240 PNG images: one with three labeled rectangles and one background image with an empty label file. It is not real remote sensing data and is not used for the manuscript's experiments.

Expected statistics at the original image resolution:

| Quantity | Value |
| --- | ---: |
| Images | 2 |
| Background images | 1 |
| Objects | 3 |
| Small objects | 1 |
| Medium objects | 1 |
| Large objects | 1 |
| Class 0 instances | 2 |
| Class 1 instances | 1 |

Run from the repository root:

```bash
python tools/dataset_statistics.py --dataset Demo=examples/toy_dataset/data.yaml --split test --out-dir outputs/demo
```
