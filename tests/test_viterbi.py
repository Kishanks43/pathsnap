"""Tests for the HMM/Viterbi decoder and the classifier-only baseline (Person 4).

Uses a tiny hand-built road map so every expected answer can be checked by hand:

    node0 --arc0--> node1 --arc1--> node2 --arc2--> node3      (connected road)
    node4 --arc3--> node5                                       (decoy, disconnected)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pathsnap.hmm.viterbi import (
    DecoderConfig,
    NetworkIndex,
    classifier_only,
    transition_logprob,
    viterbi_decode,
)
from pathsnap.io.schema import CandidateSet, RoadNetwork, Track


def _network() -> RoadNetwork:
    nodes = np.array(
        [
            [0.0000, 0.0000],
            [0.0005, 0.0000],
            [0.0010, 0.0000],
            [0.0015, 0.0000],
            [0.0005, 0.0005],
            [0.0010, 0.0005],
        ]
    )
    arcs = np.array([[0, 1], [1, 2], [2, 3], [4, 5]], dtype=np.int64)
    return RoadNetwork(trip_id="toy", nodes=nodes, arcs=arcs)


def _track(lons: list[float], timestamps: list[float] | None = None) -> Track:
    ts = timestamps if timestamps is not None else list(range(len(lons)))
    df = pd.DataFrame({"lon": lons, "lat": [0.0] * len(lons), "timestamp": ts})
    return Track(trip_id="toy", df=df)


def _candidates(rows: list[tuple[int, int, float, float]]) -> tuple[CandidateSet, np.ndarray]:
    """rows: (obs_idx, arc_id, t_along, probability_positive)."""
    obs = np.array([r[0] for r in rows], dtype=np.int64)
    arc = np.array([r[1] for r in rows], dtype=np.int64)
    t = np.array([r[2] for r in rows], dtype=np.float64)
    p = np.array([r[3] for r in rows], dtype=np.float64)
    cand = CandidateSet(
        trip_id="toy",
        obs_idx=obs,
        arc_id=arc,
        dist_m=np.zeros(len(rows)),
        meta={"t_along": t},
    )
    proba = np.column_stack([1.0 - p, p])
    return cand, proba


# Observations 0..4 lie along the connected road: arc0, arc0, arc1, arc1, arc2.
TRUTH = [0, 0, 1, 1, 2]


def _clean_candidates(decoy_prob_at_2: float = 0.4) -> tuple[CandidateSet, np.ndarray]:
    rows = []
    for t, truth_arc in enumerate(TRUTH):
        truth_p = 1.0 - decoy_prob_at_2 if t == 2 else 0.6
        rows.append((t, truth_arc, 0.5, truth_p))
        rows.append((t, 3, 0.5, 1.0 - truth_p))  # decoy road, disconnected from the others
    return _candidates(rows)


class TestTransition:

    def test_same_road_is_cheap(self):
        index = NetworkIndex(_network())
        cfg = DecoderConfig(beta_m=50.0, same_arc_bonus=1.0)
        score = transition_logprob(index, 0, 0.2, 0, 0.8, observed_step_m=0.0, cfg=cfg)
        assert np.isfinite(score)

    def test_disconnected_roads_have_no_transition(self):
        index = NetworkIndex(_network())
        cfg = DecoderConfig()
        assert transition_logprob(index, 0, 0.5, 3, 0.5, observed_step_m=10.0, cfg=cfg) == -np.inf

    def test_connected_neighbour_beats_distant_mismatch(self):
        index = NetworkIndex(_network())
        cfg = DecoderConfig(beta_m=50.0)
        near = transition_logprob(index, 0, 0.9, 1, 0.1, observed_step_m=20.0, cfg=cfg)
        far = transition_logprob(index, 0, 0.9, 1, 0.1, observed_step_m=500.0, cfg=cfg)
        assert near > far


class TestViterbi:

    def test_output_shape_and_dtype(self):
        cand, proba = _clean_candidates()
        res = viterbi_decode("toy", cand, proba, _network(), 5, track=_track([0.0001, 0.0004, 0.0006, 0.0009, 0.0013]))
        assert res.arc_per_obs.shape == (5,)
        assert res.arc_per_obs.dtype == np.int64

    def test_recovers_clean_route(self):
        cand, proba = _clean_candidates(decoy_prob_at_2=0.4)
        res = viterbi_decode(
            "toy", cand, proba, _network(), 5,
            track=_track([0.0001, 0.0004, 0.0006, 0.0009, 0.0013]),
        )
        assert res.arc_per_obs.tolist() == TRUTH

    def test_corrects_a_confident_wrong_candidate(self):
        """Classifier prefers the disconnected decoy at obs 2; Viterbi must not."""
        cand, proba = _clean_candidates(decoy_prob_at_2=0.55)
        track = _track([0.0001, 0.0004, 0.0006, 0.0009, 0.0013])
        decoded = viterbi_decode("toy", cand, proba, _network(), 5, track=track)
        baseline = classifier_only(cand, proba, 5)
        assert baseline[2] == 3          # the classifier alone is wrong here
        assert decoded.arc_per_obs[2] == 1
        assert decoded.n_forced_emission_only == 0

    def test_observation_without_candidates_is_minus_one(self):
        cand, proba = _candidates(
            [(0, 0, 0.5, 0.9), (2, 1, 0.5, 0.9), (3, 1, 0.5, 0.9), (4, 2, 0.5, 0.9)]
        )
        res = viterbi_decode(
            "toy", cand, proba, _network(), 5,
            track=_track([0.0001, 0.0002, 0.0006, 0.0009, 0.0013]),
        )
        assert res.arc_per_obs[1] == -1
        assert res.n_skipped_obs == 1
        assert res.arc_per_obs[[0, 2, 3, 4]].tolist() == [0, 1, 1, 2]

    def test_large_gap_resets_chain_and_is_counted(self):
        cand, proba = _clean_candidates()
        track = _track([0.0001, 0.0004, 0.0006, 0.0009, 0.0013], timestamps=[0, 1, 2, 3, 3600])
        res = viterbi_decode("toy", cand, proba, _network(), 5, track=track)
        assert res.meta["gap_resets"] == 1
        assert res.arc_per_obs.shape == (5,)

    def test_zero_probability_does_not_destroy_the_path(self):
        """A true road given probability exactly 0 is floored, not -inf."""
        rows = [(0, 0, 0.5, 0.9), (1, 0, 0.5, 0.0), (2, 1, 0.5, 0.9)]
        cand, proba = _candidates(rows)
        res = viterbi_decode(
            "toy", cand, proba, _network(), 3, track=_track([0.0001, 0.0003, 0.0006])
        )
        assert np.isfinite(res.log_likelihood)
        assert res.arc_per_obs.tolist() == [0, 0, 1]

    def test_rejects_mismatched_track(self):
        cand, proba = _clean_candidates()
        with pytest.raises(ValueError):
            viterbi_decode("toy", cand, proba, _network(), 5, track=_track([0.0, 0.0001]))


class TestClassifierOnly:

    def test_argmax_per_observation(self):
        cand, proba = _candidates([(0, 0, 0.5, 0.2), (0, 3, 0.5, 0.7), (1, 1, 0.5, 0.6)])
        out = classifier_only(cand, proba, 3)
        assert out.tolist() == [3, 1, -1]
