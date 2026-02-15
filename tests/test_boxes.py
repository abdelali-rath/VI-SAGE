"""Tests for bounding box utilities (IoU)."""

from src.utils.boxes import calculate_iou


# Use plain float comparison to avoid pytest.approx (which uses numpy)
def _approx(a, b, tol=1e-9):
    return abs(a - b) <= tol


def test_iou_identical_boxes():
    box = (10, 20, 50, 60)
    assert _approx(calculate_iou(box, box), 1.0)


def test_iou_no_overlap():
    box1 = (0, 0, 10, 10)
    box2 = (20, 20, 30, 30)
    assert calculate_iou(box1, box2) == 0.0


def test_iou_partial_overlap():
    # box1 area = 100, box2 area = 100, intersection = 25 (5x5)
    # union = 100 + 100 - 25 = 175, iou = 25/175
    box1 = (0, 0, 10, 10)
    box2 = (5, 5, 15, 15)
    iou = calculate_iou(box1, box2)
    assert 0 < iou < 1
    assert _approx(iou, 25 / 175)


def test_iou_one_inside_other():
    outer = (0, 0, 100, 100)
    inner = (25, 25, 75, 75)
    # intersection = 50*50 = 2500, area_outer = 10000, area_inner = 2500, union = 10000
    iou = calculate_iou(outer, inner)
    assert _approx(iou, 2500 / 10000)


def test_iou_boxes_with_extra_elements():
    box1 = (0, 0, 10, 10, 0.9)
    box2 = (0, 0, 10, 10)
    assert _approx(calculate_iou(box1, box2), 1.0)
