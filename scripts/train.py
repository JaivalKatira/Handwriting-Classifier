"""
train.py: Fine-tune a YOLO classification model on handwritten "a" variants.

Expected dataset layout (relative to the project root):
    data/split_dataset/
        train/type_01 ... type_10
        val/type_01   ... type_10

Run from the project root:
    python scripts/train.py
"""

from pathlib import Path

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "split_dataset"
RUNS_DIR = PROJECT_ROOT / "runs" / "classify"

# Pre-trained nano classification weights (downloaded automatically on first run).
# Swap for "yolov8n-cls.pt" if you prefer YOLOv8.
BASE_MODEL = "yolo11n-cls.pt"


def validate_dataset(data_dir: Path) -> None:
    """Fail early with a clear message if the dataset layout is wrong."""
    for split in ("train", "val"):
        split_dir = data_dir / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Missing dataset folder: {split_dir}")

        class_dirs = sorted(p for p in split_dir.iterdir() if p.is_dir())
        print(f"[{split}] found {len(class_dirs)} classes")
        if len(class_dirs) != 10:
            print(f"  WARNING: expected 10 classes in '{split}', found {len(class_dirs)}")


def main() -> None:
    validate_dataset(DATA_DIR)

    model = YOLO(BASE_MODEL)

    # For classification, `data` is the dataset ROOT directory
    # (containing train/ and val/), not a YAML file.
    model.train(
        data=str(DATA_DIR),
        task="classify",

        # --- Core hyper-parameters for a tiny (100 image) dataset ---
        epochs=150,
        imgsz=224,
        batch=8,
        workers=2,
        patience=50,
        optimizer="AdamW",
        lr0=0.001,
        weight_decay=0.0005,
        dropout=0.2,
        seed=42,
        deterministic=True,

        # --- Augmentation ---
        # Horizontal flip is DISABLED: mirroring a handwritten "a" can turn it
        # into a different structural variation and corrupt the labels.
        fliplr=0.0,
        flipud=0.0,
        degrees=10.0,
        translate=0.1,
        scale=0.3,
        shear=2.0,
        perspective=0.0,
        hsv_h=0.0,
        hsv_s=0.3,
        hsv_v=0.3,
        erasing=0.2,
        auto_augment="randaugment",
        mosaic=0.0,
        mixup=0.0,

        # --- Output ---
        project=str(RUNS_DIR),
        name="train",
        exist_ok=False,
        plots=True,
        verbose=True,
    )

    print("\nTraining complete. Best weights are in <run_dir>/weights/best.pt")


if __name__ == "__main__":
    main()
