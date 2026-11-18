"""Hand-checked tests for the observation-level metrics used by Model 1.

If these fail, the reported Model 1 numbers cannot be trusted.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location(
    "run_model1", Path(__file__).resolve().parents[1] / "scripts" / "run_model1.py"
)
run_model1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_model1)


def _tiny_case():
    # trip 0, observation 0: candidates arcs [5, 7], scores [0.9, 0.1], reference = 5
    # trip 0, observation 1: candidates arcs [3, 4], scores [0.2, 0.6], reference absent
    group = np.array([0, 0, 0, 0])
    obs = np.array([0, 0, 1, 1])
    arc = np.array([5, 7, 3, 4])
    y = np.array([1, 0, 0, 0])
    score = np.array([0.9, 0.1, 0.2, 0.6])
    return group, obs, arc, y, score


def test_observation_table_argmax_and_reference() -> None:
    table = run_model1.observation_table(*_tiny_case())
    assert len(table) == 2
    first = table.loc[0 * run_model1.OBS_KEY_STRIDE + 0]
    second = table.loc[0 * run_model1.OBS_KEY_STRIDE + 1]
    assert first["pred_arc"] == 5 and first["top1"]
    assert second["pred_arc"] == 4 and not second["top1"]
    assert bool(first["has_ref"]) and not bool(second["has_ref"])
    assert first["p_ref"] == pytest.approx(0.9)


def test_evaluate_matches_hand_computation() -> None:
    m = run_model1.evaluate(run_model1.observation_table(*_tiny_case()))
    assert m["n_obs"] == 2
    assert m["top1_accuracy"] == pytest.approx(0.5)            # 1 of 2 observations
    assert m["candidate_miss_rate"] == pytest.approx(0.5)      # 1 of 2 reference arcs absent
    assert m["n_obs_covered"] == 1
    assert m["macro_f1"] == pytest.approx(1.0)                 # only covered obs: 5 predicted as 5
    assert m["log_loss"] == pytest.approx(-np.log(0.9))        # renormalised p(ref) on covered obs


def test_scores_are_renormalised_within_each_observation() -> None:
    group = np.array([0, 0])
    obs = np.array([0, 0])
    arc = np.array([1, 2])
    y = np.array([0, 1])
    score = np.array([3.0, 1.0])            # raw sum 4 -> p(ref)=0.25
    table = run_model1.observation_table(group, obs, arc, y, score)
    assert table["p_ref"].iloc[0] == pytest.approx(0.25)
