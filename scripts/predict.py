"""
predict.py: Classify a single handwritten "a" image using the fine-tuned model.

Usage (from the project root):
    python scripts/predict.py path/to/image.png
    python scripts/predict.py path/to/image.png --weights runs/classify/train2/weights/best.pt
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = PROJECT_ROOT / "runs" / "classify" / "train" / "weights" / "best.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict the structural type of a handwritten 'a'."
    )
    parser.add_argument("image", type=Path, help="Path to the image to classify")
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_WEIGHTS,
        help=f"Path to trained weights (default: {DEFAULT_WEIGHTS})",
    )
    parser.add_argument("--imgsz", type=int, default=224, help="Inference image size")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.image.is_file():
        raise FileNotFoundError(f"Image not found: {args.image}")
    if not args.weights.is_file():
        raise FileNotFoundError(
            f"Weights not found: {args.weights}\n"
            "Train the model first (python scripts/train.py) or pass --weights."
        )

    model = YOLO(str(args.weights))
    result = model.predict(source=str(args.image), imgsz=args.imgsz, verbose=False)[0]

    top1_idx = result.probs.top1
    top1_conf = float(result.probs.top1conf)
    top1_name = result.names[top1_idx]

    print(f"Image:      {args.image}")
    print(f"Prediction: {top1_name}")
    print(f"Confidence: {top1_conf:.2%}")


if __name__ == "__main__":
    main()
