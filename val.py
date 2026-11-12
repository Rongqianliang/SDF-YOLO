"""Run one Ultralytics validation and keep the standard validation output."""

from ultralytics import YOLO


MODEL_PATH = "/home/ubuntu/rong/ultralytics-yolo11-0806/runs/train/exp768/weights/best.pt"
DATA_PATH = "/home/ubuntu/rong/ultralytics-yolo11-0624/SIMD.yaml"


if __name__ == "__main__":
    model = YOLO(MODEL_PATH)
    model.val(
        data=DATA_PATH,
        split="test",
        imgsz=640,
        batch=16,
        project="runs/val",
        name="exp",
    )
