"""Metric tests. These define the agreed semantics for the whole team."""

from __future__ import annotations

import numpy as np
import pytest

from pathsnap.evaluation.metrics import (
    candidate_miss_rate,
    macro_f1,
    route_continuity,
    safe_log_loss,
    sequence_exact_match,
    sequence_f1,
    top1_accuracy,
)


def test_top1_accuracy() -> None:
    assert top1_accuracy(np.array([1, 2, 3, 4]), np.array([1, 2, 0, 0])) == 0.5


def test_macro_f1_perfect_and_imperfect() -> None:
    y = np.array([0, 1, 2, 3])
    assert macro_f1(y, y) == pytest.approx(1.0)
    assert macro_f1(y, np.array([0, 0, 0, 0])) < 0.5


def test_safe_log_loss_returns_none_when_class_unseen() -> None:
    classes = np.array([0, 1])
    proba = np.array([[1.0, 0.0], [1.0, 0.0]])
    assert safe_log_loss(np.array([0, 1]), proba, classes) is None


def test_safe_log_loss_value() -> None:
    classes = np.array([0, 1])
    proba = np.array([[0.5, 0.5], [0.5, 0.5]])
    got = safe_log_loss(np.array([0, 1]), proba, classes)
    assert got == pytest.approx(np.log(2), rel=1e-9)


def test_candidate_miss_rate() -> None:
    cands = [np.array([1, 2]), np.array([3, 4]), np.array([5])]
    truth = np.array([1, 3, 9])
    assert candidate_miss_rate(cands, truth) == pytest.approx(1 / 3)


def test_route_continuity_straight_line() -> None:
    arcs = np.array([[0, 1], [1, 2], [2, 3]])
    assert route_continuity(np.array([0, 1, 2]), arcs) == pytest.approx(1.0)


def test_route_continuity_detects_jump() -> None:
    arcs = np.array([[0, 1], [1, 2], [2, 3]])
    # arc 2 is not reachable from arc 0's head (node 1).
    assert route_continuity(np.array([0, 2]), arcs) == pytest.approx(0.0)


def test_route_continuity_single_obs() -> None:
    arcs = np.array([[0, 1]])
    assert route_continuity(np.array([0]), arcs) == 1.0


def test_sequence_exact_match() -> None:
    assert sequence_exact_match(np.array([1, 2, 3]), np.array([1, 2, 3])) == 1.0
    assert sequence_exact_match(np.array([1, 2]), np.array([1, 2, 3])) == 0.0


def test_sequence_f1_partial_overlap() -> None:
    # LCS of [1,2,3] and [1,3] is 2 -> p=1.0, r=0.667, f1=0.8
    assert sequence_f1(np.array([1, 2, 3]), np.array([1, 3])) == pytest.approx(0.8)
