"""HMM + Viterbi route reconstruction.

OWNER: Person 4
STATUS: implemented

WHAT THIS MODULE DOES
---------------------
The classifier scores every (GPS observation, candidate road) pair on its own.
This module picks ONE road per observation so that the whole sequence is a
plausible drive. It is a Hidden Markov Model decoded with the Viterbi algorithm.

MODEL
-----
Hidden state at observation t : the candidate road chosen for observation t.
Emission (how well a state explains the observation):
    log emission_t(k) = emission_weight * log( max(p_tk, prob_floor) )
    where p_tk is the classifier's probability that candidate k is the road.
    The floor stops an exact zero from killing the whole path (see section 10 of
    docs/P2_EXPLAINED.md: KNN can give exactly 0 to the true road).
Transition (how plausible a move from road j at t-1 to road k at t is):
    d_net  = driving distance from the position on road j to the position on road k,
             using only up to `max_hops` road segments between them.
    d_obs  = great-circle distance between the two GPS fixes.
    log trans = -|d_net - d_obs| / beta_m
                + connectivity_bonus   (different roads, connected in the map)
                + same_arc_bonus       (same road)
    Unreachable pairs (no path within max_hops) get log trans = -inf.

VITERBI RECURRENCE (log space)
------------------------------
    delta_t(k)       = log emission_t(k) + max_j [ delta_{t-1}(j) + log trans(j -> k) ]
    backpointer_t(k) = argmax_j of the bracket above
    best end state   = argmax_k delta_T(k), then backtrack.

HANDLING OF EDGE CASES (all counted and returned in DecodeResult.meta)
---------------------------------------------------------------------
- Observation with no candidate: output -1 and skip it. The chain continues from
  the previous observation, so the output still lines up with the GPS timeline.
- GPS gap (time step > max_gap_s, or non-increasing time): with reset_on_gap the
  chain restarts from emission only at the next observation.
- No valid transition into a road (disconnected map): that road restarts from its
  emission only. Each affected observation is counted.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from pathsnap.geo.distance import haversine_m
from pathsnap.io.schema import CandidateSet, RoadNetwork, Track


@dataclass
class DecoderConfig:
    """Every field is a knob whose value must be justified in the report."""

    emission_weight: float = 1.0      # exponent applied to the classifier probability
    prob_floor: float = 1e-6          # smallest probability used in log space (avoids -inf)
    beta_m: float = 50.0              # transition tolerance in metres (distance mismatch scale)
    connectivity_bonus: float = 5.0   # log-space bonus for moving to a connected, different road
    same_arc_bonus: float = 1.0       # log-space bonus for staying on the same road
    max_hops: int = 2                 # max road segments between two candidates (0 = touching only)
    max_gap_s: float = 30.0           # larger time steps are treated as GPS dropouts
    reset_on_gap: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DecodeResult:
    trip_id: str
    arc_per_obs: np.ndarray           # int64, shape (n_obs,), -1 when no candidate
    log_likelihood: float
    n_skipped_obs: int = 0
    n_forced_emission_only: int = 0
    meta: dict = field(default_factory=dict)


class NetworkIndex:
    """Precomputed road geometry for fast transition scoring within one trip.

    Built once per trip. Holds arc lengths, the tail and head node of every arc,
    and a node-to-outgoing-arcs table, so that "how far is road k from road j
    through at most `max_hops` roads" is a small graph search.
    """

    def __init__(self, network: RoadNetwork):
        arcs = np.asarray(network.arcs, dtype=np.int64)
        self.tail = arcs[:, 0]
        self.head = arcs[:, 1]
        self.length_m = network.arc_lengths_m()
        n_nodes = network.n_nodes
        order = np.argsort(self.tail, kind="stable")
        self._out_arcs = order
        self._out_start = np.searchsorted(self.tail[order], np.arange(n_nodes + 1))
        self._reach_cache: dict[tuple[int, int], dict[int, tuple[float, int]]] = {}

    def out_arcs(self, node: int) -> np.ndarray:
        return self._out_arcs[self._out_start[node] : self._out_start[node + 1]]

    def reach(self, node: int, max_hops: int) -> dict[int, tuple[float, int]]:
        """Nodes reachable from `node` within `max_hops` arcs: node -> (metres, arcs used)."""
        key = (int(node), int(max_hops))
        cached = self._reach_cache.get(key)
        if cached is not None:
            return cached
        best: dict[int, tuple[float, int]] = {int(node): (0.0, 0)}
        frontier = [(0.0, int(node), 0)]
        for _ in range(max_hops):
            next_frontier = []
            for dist, v, hops in frontier:
                for arc in self.out_arcs(v):
                    w = int(self.head[arc])
                    nd = dist + float(self.length_m[arc])
                    if w not in best or nd < best[w][0]:
                        best[w] = (nd, hops + 1)
                        next_frontier.append((nd, w, hops + 1))
            frontier = next_frontier
        self._reach_cache[key] = best
        return best

    def network_distance_m(
        self, arc_from: int, t_from: float, arc_to: int, t_to: float, max_hops: int
    ) -> float | None:
        """Driving distance from a position on arc_from to a position on arc_to.

        Returns None when arc_to cannot be reached within max_hops arcs.
        """
        if arc_from == arc_to:
            return abs(t_to - t_from) * float(self.length_m[arc_from])
        exit_m = (1.0 - t_from) * float(self.length_m[arc_from])
        entry_m = t_to * float(self.length_m[arc_to])
        reach = self.reach(int(self.head[arc_from]), max_hops)
        hit = reach.get(int(self.tail[arc_to]))
        if hit is None:
            return None
        return exit_m + hit[0] + entry_m


def transition_logprob(
    index: NetworkIndex,
    arc_from: int,
    t_from: float,
    arc_to: int,
    t_to: float,
    observed_step_m: float,
    cfg: DecoderConfig,
) -> float:
    """log P(road arc_to at t | road arc_from at t-1), up to a constant.

    Returns -inf when the two roads are not connected within cfg.max_hops.
    """
    d_net = index.network_distance_m(arc_from, t_from, arc_to, t_to, cfg.max_hops)
    if d_net is None:
        return -np.inf
    score = -abs(d_net - observed_step_m) / cfg.beta_m
    if arc_from == arc_to:
        score += cfg.same_arc_bonus
    else:
        score += cfg.connectivity_bonus
    return float(score)


def _positive_scores(proba: np.ndarray) -> np.ndarray:
    """Probability of 'this candidate is the road' from the classifier output."""
    proba = np.asarray(proba, dtype=np.float64)
    if proba.ndim == 1:
        return proba
    if proba.shape[1] == 1:
        return proba[:, 0]
    return proba[:, -1]


def viterbi_decode(
    trip_id: str,
    candidates: CandidateSet,
    proba: np.ndarray,
    network: RoadNetwork,
    n_obs: int,
    cfg: DecoderConfig | None = None,
    *,
    track: Track,
) -> DecodeResult:
    """Decode the most likely road sequence for one trip.

    Parameters
    ----------
    proba : (n_pairs, n_classes) or (n_pairs,) classifier output aligned row-for-row
            with ``candidates.obs_idx`` / ``candidates.arc_id``.
    n_obs : number of GPS observations in the trip (output length).
    track : the trip's GPS observations (coordinates and timestamps). Needed for the
            observed step distance and for gap detection. Never the reference route.

    Returns
    -------
    DecodeResult with one predicted road per observation (-1 when no candidate).
    """
    cfg = cfg or DecoderConfig()
    if len(track) != n_obs:
        raise ValueError(f"track has {len(track)} observations but n_obs={n_obs}")

    arc_per_obs = np.full(n_obs, -1, dtype=np.int64)
    pos = _positive_scores(proba)
    if len(pos) != len(candidates.obs_idx):
        raise ValueError("proba must have one row per candidate pair")

    obs_idx = np.asarray(candidates.obs_idx, dtype=np.int64)
    arc_id = np.asarray(candidates.arc_id, dtype=np.int64)
    t_along = np.asarray(candidates.meta.get("t_along", np.zeros(len(obs_idx))), dtype=np.float64)

    counts = np.bincount(obs_idx, minlength=n_obs) if obs_idx.size else np.zeros(n_obs, dtype=np.int64)
    starts = np.cumsum(counts) - counts
    valid = np.flatnonzero(counts > 0)
    n_skipped = int(n_obs - valid.size)
    if valid.size == 0:
        return DecodeResult(
            trip_id=trip_id,
            arc_per_obs=arc_per_obs,
            log_likelihood=float("-inf"),
            n_skipped_obs=n_skipped,
            meta={"segments": 0, "gap_resets": 0, "forced_steps": 0, "config": cfg.to_dict()},
        )

    index = NetworkIndex(network)
    lon, lat = track.lon, track.lat
    ts = track.timestamp

    def rows(t: int) -> slice:
        return slice(starts[t], starts[t] + counts[t])

    delta: dict[int, np.ndarray] = {}
    back: dict[int, np.ndarray] = {}
    log_likelihood = 0.0
    gap_resets = 0
    forced_steps = 0
    segments = 1
    prev_t: int | None = None

    for t in valid.tolist():
        r = rows(t)
        emission = cfg.emission_weight * np.log(np.maximum(pos[r], cfg.prob_floor))

        if prev_t is None:
            delta[t] = emission
            back[t] = np.full(counts[t], -1, dtype=np.int64)
            prev_t = t
            continue

        dt = float(ts[t] - ts[prev_t])
        is_gap = dt <= 0 or dt > cfg.max_gap_s
        if is_gap and cfg.reset_on_gap:
            log_likelihood += float(delta[prev_t].max())
            gap_resets += 1
            segments += 1
            delta[t] = emission
            back[t] = np.full(counts[t], -1, dtype=np.int64)
            prev_t = t
            continue

        d_obs = float(haversine_m(lon[prev_t], lat[prev_t], lon[t], lat[t]))
        prev_r = rows(prev_t)
        cur_r = r
        n_prev, n_cur = counts[prev_t], counts[t]
        trans = np.full((n_prev, n_cur), -np.inf)
        for j in range(n_prev):
            aj, tj = int(arc_id[prev_r.start + j]), float(t_along[prev_r.start + j])
            for k in range(n_cur):
                ak, tk = int(arc_id[cur_r.start + k]), float(t_along[cur_r.start + k])
                trans[j, k] = transition_logprob(index, aj, tj, ak, tk, d_obs, cfg)

        score = delta[prev_t][:, None] + trans
        best_prev = np.argmax(score, axis=0)
        best_score = score[best_prev, np.arange(n_cur)]
        reachable = np.isfinite(best_score)

        if not reachable.all():
            forced_steps += 1
        new_delta = np.where(reachable, best_score + emission, emission)
        new_back = np.where(reachable, best_prev, -1).astype(np.int64)
        delta[t] = new_delta
        back[t] = new_back
        prev_t = t

    # Backtrack from the best final state; a -1 pointer means the chain restarted,
    # so continue from the best state of the observation before the restart.
    last_t = int(valid[-1])
    log_likelihood += float(delta[last_t].max())
    state = int(np.argmax(delta[last_t]))
    seq = valid.tolist()
    for i in range(len(seq) - 1, -1, -1):
        t = seq[i]
        arc_per_obs[t] = int(arc_id[rows(t).start + state])
        b = int(back[t][state])
        if b >= 0:
            state = b
        elif i > 0:
            state = int(np.argmax(delta[seq[i - 1]]))

    return DecodeResult(
        trip_id=trip_id,
        arc_per_obs=arc_per_obs,
        log_likelihood=float(log_likelihood),
        n_skipped_obs=n_skipped,
        n_forced_emission_only=forced_steps,
        meta={
            "segments": segments,
            "gap_resets": gap_resets,
            "forced_steps": forced_steps,
            "config": cfg.to_dict(),
        },
    )


def classifier_only(candidates: CandidateSet, proba: np.ndarray, n_obs: int) -> np.ndarray:
    """Per-observation argmax over candidates: the baseline Viterbi must beat.

    Independent of the decoder on purpose, so the comparison is not circular.
    """
    arc_per_obs = np.full(n_obs, -1, dtype=np.int64)
    pos = _positive_scores(proba)
    obs_idx = np.asarray(candidates.obs_idx, dtype=np.int64)
    arc_id = np.asarray(candidates.arc_id, dtype=np.int64)
    if obs_idx.size == 0:
        return arc_per_obs
    order = np.lexsort((-pos, obs_idx))
    first = np.ones(order.size, dtype=bool)
    first[1:] = obs_idx[order][1:] != obs_idx[order][:-1]
    chosen = order[first]
    arc_per_obs[obs_idx[chosen]] = arc_id[chosen]
    return arc_per_obs
