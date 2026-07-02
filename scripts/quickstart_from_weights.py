"""Quickstart: evaluate straight from YOLO weights (no predictions on hand).

Self-contained onboarding example — it mocks a handful of images and a tiny
ground-truth table, downloads the smallest Ultralytics model, and lets
``Evaluation`` generate the predictions itself. The point to take away is how to
size ``predict_kwargs`` so inference fits your GPU.

Run:
    uv run --extra ultralytics python scripts/quickstart_from_weights.py

GPU-memory knobs (all go in ``predict_kwargs``):
    batch  — images per model.predict call; peak VRAM ≈ batch × per-image cost.
             THE lever when you OOM. Lower it (e.g. 4) on a small card.
    imgsz  — inference resolution; cost grows ~quadratically (640 → 1280 ≈ 4×).
    half   — half precision (fp16); roughly halves VRAM.
    device — "0" for the first GPU, "cpu" to force CPU.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from PIL import Image

from digital_metrics import Evaluation


def _mock_dataset(root: Path, n_images: int = 8) -> pd.DataFrame:
    """Write ``n_images`` random JPEGs and a matching ground-truth DataFrame.

    The GT boxes are dummies — the model won't hit them on random noise — but the
    schema is exactly what a real run needs: one row per GT box with ``image_name``,
    ``image_path``, a class label, and box corners.
    """
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n_images):
        path = root / f"img_{i:02d}.jpg"
        Image.fromarray(rng.integers(0, 255, (640, 640, 3), dtype=np.uint8)).save(path)
        rows.append(
            {
                "image_name": path.name,
                "image_path": str(path),
                "instance_label": "object",
                "bbox_x_tl": 100.0,
                "bbox_y_tl": 100.0,
                "bbox_x_br": 200.0,
                "bbox_y_br": 200.0,
                "split": "test",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    with TemporaryDirectory() as tmp:
        gt_df = _mock_dataset(Path(tmp))

        # preds_df=None → Evaluation runs YOLO itself from weights_path over the
        # images listed in gt_df["image_path"]. predict_kwargs are forwarded to
        # Ultralytics' model.predict; tune them to fit your GPU.
        evaluation = Evaluation(
            None,
            gt_df,
            skip_cohen_kappa=True,
            weights_path="yolo11n.pt",  # smallest model; auto-downloads on first run
            predict_kwargs={
                "batch": 4,  # ← lower this first if you OOM
                "imgsz": 640,  # ← raise for accuracy, lower for memory
                "half": True,  # ← fp16, ~half the VRAM
                "device": "0",  # ← "cpu" to skip the GPU entirely
            },
        )

        # First call generates predictions from the weights, then scores them.
        evaluation("test", find_best_confs=False)

        analyst, _ = evaluation.get_dashboards(save_to_excel=False)
        print("\nPer-class metrics (mock data — numbers are not meaningful):")
        print(analyst.to_string(index=False))


if __name__ == "__main__":
    main()
