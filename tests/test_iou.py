import numpy as np
import pytest

from metrics.matching import compute_iou_matrix, find_duplicates_bboxes


def test_perfect_overlap() -> None:
    a = np.array([[0.0, 0.0, 100.0, 100.0]])
    iou = compute_iou_matrix(a, a)
    assert iou.shape == (1, 1)
    assert iou[0, 0] == pytest.approx(1.0)


def test_no_overlap() -> None:
    a = np.array([[0.0, 0.0, 50.0, 50.0]])
    b = np.array([[100.0, 100.0, 200.0, 200.0]])
    iou = compute_iou_matrix(a, b)
    assert iou[0, 0] == pytest.approx(0.0)


def test_partial_overlap() -> None:
    # Two 100×100 boxes offset by 50 → intersection 50×50=2500, union=17500
    a = np.array([[0.0, 0.0, 100.0, 100.0]])
    b = np.array([[50.0, 50.0, 150.0, 150.0]])
    iou = compute_iou_matrix(a, b)
    expected = 2500.0 / (10000.0 + 10000.0 - 2500.0)
    assert iou[0, 0] == pytest.approx(expected, rel=1e-5)


def test_matrix_shape() -> None:
    a = np.zeros((3, 4))
    b = np.zeros((5, 4))
    iou = compute_iou_matrix(a, b)
    assert iou.shape == (3, 5)


def test_empty_inputs() -> None:
    a = np.zeros((0, 4))
    b = np.array([[0.0, 0.0, 100.0, 100.0]])
    iou = compute_iou_matrix(a, b)
    assert iou.shape == (0, 1)


def test_find_duplicates_basic() -> None:
    # Identity IoU matrix: each box is its own duplicate if threshold < 1.0
    iou = np.eye(3)
    iou[0, 1] = 0.999  # box 1 duplicates box 0
    iou[1, 0] = 0.999
    to_remove, extra = find_duplicates_bboxes(iou, threshold=0.99)
    assert 1 in to_remove
    assert extra == []


def test_find_duplicates_no_dups() -> None:
    iou = np.eye(3)  # only self-overlap
    to_remove, _ = find_duplicates_bboxes(iou, threshold=0.99)
    # Diagonal entries are 1.0 but i==j so they are skipped (j must be > i)
    assert to_remove == []
