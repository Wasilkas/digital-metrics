"""Tests for backend selection on Evaluation (ultralytics / torchmetrics).

The adapter that maps external ``DetectionMetrics`` onto native ``Metrics`` (so the
dashboards keep working) is torch-free and always runs. The actual backend
round-trips need the optional extras, so those are skipped unless installed.
"""

import importlib.util
import math

import numpy as np
import pandas as pd
import pytest

from digital_metrics import DetectionMetrics, Evaluation, Metrics
from digital_metrics.backends.ultralytics_metrics import (
    _conf_at_max_f1,
    _confusion_process_batch,
    _read_prf1_at_conf,
    compute_ultralytics_confusion_matrix,
)
from digital_metrics.engines import BackendEngine


def _f1(p: float, r: float) -> float:
    return 2.0 * p * r / (p + r)


def _adapt_via_engine(ev: Evaluation, det: dict[str, DetectionMetrics]) -> dict[str, Metrics]:
    """Run the (torch-free) DetectionMetrics→Metrics adapter via a BackendEngine."""
    engine = BackendEngine(
        backend="ultralytics",
        classes=ev.classes,
        confidence_optimization="per_class",
        calibrator=ev._calibrator,
    )
    return engine._adapt(det, ev.gt_df)


def test_backend_drops_unknown_classes_before_scoring(
    tiny_dataset: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    # Backends score only the GT vocabulary; the orchestrator drops out-of-vocab
    # predictions (torch-free, so no extra needed) before the external library runs.
    gt_df, preds_df = tiny_dataset
    preds_bad = preds_df.copy()
    preds_bad.loc[0, "instance_label"] = "class_z"  # not in GT vocabulary
    ev = Evaluation(preds_bad, gt_df, iou_threshold=0.5, backend="ultralytics")
    ev._define_gt("all")
    ev._drop_unknown_pred_classes()

    assert "class_z" not in set(ev._raw_preds_df["instance_label"])
    assert "class_z" not in set(ev.preds_df["instance_label"])


def test_confusion_process_batch_counts_tp_fp_fn() -> None:
    # 2 classes (a=0, b=1); matrix is (3, 3) with background at index 2.
    # GT: [a, b]; Det: [a, b]. Det 'a' matches GT 'a' (IoU 0.9); GT 'b' is missed
    # (FN); Det 'b' matches nothing (FP). Matrix is Ultralytics-oriented [pred, gt].
    matrix = np.zeros((3, 3), dtype=np.int64)
    det_classes = np.array([0, 1])
    gt_classes = np.array([0, 1])
    iou = np.array([[0.9, 0.0], [0.0, 0.0]])  # (n_gt, n_det)

    _confusion_process_batch(matrix, det_classes, gt_classes, iou, iou_thres=0.45, nc=2)

    assert matrix[0, 0] == 1  # correct: pred a, gt a
    assert matrix[2, 1] == 1  # gt b missed → background row (FN)
    assert matrix[1, 2] == 1  # pred b spurious → background col (FP)
    assert matrix.sum() == 3


def test_confusion_process_batch_empty_branches() -> None:
    # No GT → all detections are FP in the background column.
    fp_only = np.zeros((2, 2), dtype=np.int64)
    _confusion_process_batch(
        fp_only, np.array([0]), np.array([], dtype=int), np.zeros((0, 1)), 0.45, 1
    )
    assert fp_only[0, 1] == 1 and fp_only.sum() == 1

    # No detections → every GT is a FN in the background row.
    fn_only = np.zeros((2, 2), dtype=np.int64)
    _confusion_process_batch(
        fn_only, np.array([], dtype=int), np.array([0]), np.zeros((1, 0)), 0.45, 1
    )
    assert fn_only[1, 0] == 1 and fn_only.sum() == 1


def test_adapt_detection_metrics_reproduces_backend_numbers(
    tiny_dataset: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    gt_df, preds_df = tiny_dataset  # class_a x3, class_b x2, class_c x2 (all 'test')
    ev = Evaluation(preds_df, gt_df)
    ev._define_gt("all")

    det = {
        "class_a": DetectionMetrics(
            precision=0.8, recall=0.6, f1=_f1(0.8, 0.6), ap50=0.7, ap75=0.5, ap50_95=0.55
        ),
        "class_b": DetectionMetrics(
            precision=1.0, recall=0.5, f1=_f1(1.0, 0.5), ap50=0.9, ap75=0.8, ap50_95=0.6
        ),
        # class_c deliberately omitted → exercises the "missing/absent" branch.
    }

    adapted = _adapt_via_engine(ev, det)

    # precision/recall/f1/AP reproduce the backend exactly.
    a = adapted["class_a"]
    assert a.precision == pytest.approx(0.8)
    assert a.recall == pytest.approx(0.6)
    assert a.f1_score == pytest.approx(_f1(0.8, 0.6))
    assert (a.ap50, a.ap75, a.ap50_95) == pytest.approx((0.7, 0.5, 0.55))
    # Reconstructed float counts: TP=r*N, FN=N-TP, FP=TP*(1-p)/p (N=3 for class_a).
    assert a.tp == pytest.approx(1.8)
    assert a.fn == pytest.approx(1.2)
    assert a.fp == pytest.approx(0.45)
    assert a.cohen_kappa == -1
    # CIs are real proportions in [0, 1].
    assert 0.0 <= a.recall_ci_lower <= a.recall <= a.recall_ci_upper <= 1.0
    assert 0.0 <= a.precision_ci_lower <= a.precision <= a.precision_ci_upper <= 1.0

    # Class present in GT but absent from the backend output → NaN AP, zero counts.
    c = adapted["class_c"]
    assert math.isnan(c.ap50) and math.isnan(c.ap75) and math.isnan(c.ap50_95)
    assert c.tp == 0 and c.fp == 0 and c.fn == 0


def test_backend_metrics_drive_dashboards_without_cm(
    tiny_dataset: tuple[pd.DataFrame, pd.DataFrame],
    tmp_path: object,
) -> None:
    # Backend mode has no confusion matrix; the dashboards must still build.
    gt_df, preds_df = tiny_dataset
    ev = Evaluation(preds_df, gt_df)
    ev._define_gt("all")
    ev.metrics = _adapt_via_engine(
        ev,
        {
            c: DetectionMetrics(
                precision=0.8, recall=0.6, f1=_f1(0.8, 0.6), ap50=0.7, ap75=0.5, ap50_95=0.55
            )
            for c in ev.classes
        },
    )
    ev.cm = None  # native-only; cleared in backend mode

    devs, dtrk = ev.get_dashboards(
        save_to_excel=False, save_confusion_matrix=False, path=str(tmp_path)
    )
    assert not devs.empty
    assert "Недобраковка" in dtrk.columns
    assert set(devs.index) == set(ev.classes)


@pytest.mark.skipif(
    importlib.util.find_spec("ultralytics") is not None,
    reason="ultralytics installed — backend path would run instead",
)
def test_backend_selection_reaches_external_path(
    tiny_dataset: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    # backend=... → calling the evaluation dispatches to the external backend,
    # which here fails on the missing extra (proving the path is reached).
    gt_df, preds_df = tiny_dataset
    ev = Evaluation(preds_df, gt_df, backend="ultralytics")
    with pytest.raises(ImportError, match="ultralytics"):
        ev(split="test")


@pytest.mark.skipif(
    importlib.util.find_spec("torchmetrics") is not None,
    reason="torchmetrics installed — backend path would run instead",
)
def test_compute_metrics_torchmetrics_importerror(
    tiny_dataset: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    gt_df, preds_df = tiny_dataset
    ev = Evaluation(preds_df, gt_df)
    with pytest.raises(ImportError, match="torchmetrics"):
        ev.compute_metrics_torchmetrics(split="test")


@pytest.mark.skipif(
    importlib.util.find_spec("ultralytics") is not None,
    reason="ultralytics installed — box_iou would run instead",
)
def test_confusion_matrix_requires_ultralytics(
    tiny_dataset: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    gt_df, preds_df = tiny_dataset
    with pytest.raises(ImportError, match="ultralytics"):
        compute_ultralytics_confusion_matrix(gt_df, preds_df)


@pytest.mark.parametrize("backend", ["ultralytics", "torchmetrics"])
def test_backend_mode_populates_metrics_when_installed(
    backend: str,
    tiny_dataset: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    if importlib.util.find_spec(backend) is None:
        pytest.skip(f"optional backend {backend!r} not installed")

    gt_df, preds_df = tiny_dataset
    ev = Evaluation(preds_df, gt_df, backend=backend)  # type: ignore[arg-type]
    ev(split="test")

    assert set(ev.detection_metrics) <= {"class_a", "class_b", "class_c"}
    # Adapted native metrics reproduce the backend's precision/recall per class.
    for cls, dm in ev.detection_metrics.items():
        assert ev.metrics[cls].precision == pytest.approx(dm.precision, abs=1e-6)
        assert ev.metrics[cls].recall == pytest.approx(dm.recall, abs=1e-6)

    if backend == "ultralytics":
        # ultralytics backend fills the confusion matrix (classes + background).
        assert ev.cm is not None
        n = len(ev.classes)
        assert ev.cm.shape == (n + 1, n + 1)
        assert ev.class_labels == [*ev.classes, "background"]
    else:
        assert ev.cm is None  # torchmetrics has no confusion matrix


# ── Backend calibration (read P/R/F1 at the val-calibrated confidence) ──────────


def test_read_prf1_at_conf_interpolates() -> None:
    x = np.linspace(0.0, 1.0, 11)  # 0.0, 0.1, ..., 1.0
    p_curve = x.copy()  # precision rises with confidence
    r_curve = 1.0 - x  # recall falls with confidence
    f1_curve = np.full_like(x, 0.5)

    p, r, f1 = _read_prf1_at_conf(p_curve, r_curve, f1_curve, x, 0.3)

    assert p == pytest.approx(0.3)
    assert r == pytest.approx(0.7)
    assert f1 == pytest.approx(0.5)


def test_conf_at_max_f1_returns_argmax_confidence() -> None:
    x = np.linspace(0.0, 1.0, 11)
    f1 = np.array([0.0, 0.1, 0.2, 0.9, 0.4, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0])  # peak at idx 3
    assert _conf_at_max_f1(f1, x) == pytest.approx(0.3)  # x[3]


def test_backend_calibration_rejects_missing_split(
    split_dataset: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    # The calibration-split validation runs before any Ultralytics call, so this
    # raises without the extra installed.
    gt_df, preds_df = split_dataset  # has 'val' and 'test', no 'train'
    ev = Evaluation(preds_df, gt_df, backend="ultralytics")
    with pytest.raises(ValueError, match="No ground-truth rows"):
        ev(split="test", calibration_split="train")


def test_ultralytics_backend_calibration_when_installed(
    split_dataset: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    if importlib.util.find_spec("ultralytics") is None:
        pytest.skip("ultralytics not installed")

    gt_df, preds_df = split_dataset
    base = Evaluation(preds_df, gt_df, backend="ultralytics")
    base(split="test")  # self-selected operating point

    cal = Evaluation(preds_df, gt_df, backend="ultralytics")
    cal(split="test", calibration_split="val")  # operating point from val

    assert cal.detection_metrics  # populated
    assert cal.cm is not None  # confusion matrix still filled
    assert set(cal.best_confidences) >= set(cal.detection_metrics)  # thresholds recorded
    # AP is read over the full curve, so calibration must not change it.
    for cls, dm in base.detection_metrics.items():
        assert cal.detection_metrics[cls].ap50 == pytest.approx(dm.ap50)
        assert cal.detection_metrics[cls].ap50_95 == pytest.approx(dm.ap50_95)
