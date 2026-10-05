"""Create tiny synthetic evaluation fixtures and an illustrative figure."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
COLS = ["image_name", "instance_label", "bbox_x_tl", "bbox_y_tl", "bbox_x_br", "bbox_y_br"]


def main() -> None:
    gt = pd.DataFrame(
        [
            ["val-positive", "cat", 0, 0, 10, 10],
            ["test-positive", "cat", 0, 0, 10, 10],
            ["test-negative", None, None, None, None, None],
        ],
        columns=COLS,
    )
    gt["split"] = ["val", "test", "test"]
    preds = pd.DataFrame(
        [
            ["val-positive", "cat", 0, 0, 10, 10, 0.9],
            ["test-positive", "cat", 0, 0, 10, 10, 0.9],
            ["test-negative", "cat", 0, 0, 10, 10, 0.4],
        ],
        columns=COLS + ["confidence"],
    )
    gt.to_csv(ROOT / "fixtures/ground_truths_all.csv")
    preds.to_csv(ROOT / "fixtures/predicts_all.csv")
    fig, ax = plt.subplots()
    ax.plot([0.5, 0.5, 1], [1, 0.5, 2 / 3], "o-")
    for r, p, t in [(0.5, 1, 0.9), (0.5, 0.5, 0.8), (1, 2 / 3, 0.7)]:
        ax.annotate(f"t={t}", (r, p), xytext=(8, -8), textcoords="offset points")
    ax.set(xlabel="Recall", ylabel="Observed precision", xlim=(0, 1.1), ylim=(0, 1.1))
    figure_path = ROOT / "docs/prf1_operating_points.svg"
    fig.savefig(figure_path)
    # Matplotlib emits trailing spaces in SVG path data; normalize the checked-in example.
    figure_path.write_text(
        "\n".join(line.rstrip() for line in figure_path.read_text().splitlines()) + "\n"
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
