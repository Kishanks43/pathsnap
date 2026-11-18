"""HMM + Viterbi route reconstruction.

OWNER: Person 4
STATUS: STUB - this is the core algorithmic contribution of your workstream.

MODEL
-----
Hidden state at step t : the road arc chosen for observation t.
                        (equivalently the candidate pair (t, arc))
Observation at step t  : the noisy GPS fix.
Emission               : P(observation | arc) - derive from the classifier's
                         candidate probability. The classifier already scores
                         (obs, arc) pairs, so a defensible mapping is
                             emission_t(k) = proba[(t, k)] ** emission_weight
                         or normalised log-odds. Whatever you choose, state it.
Transition             : P(arc_j at t | arc_i at t-1). Two ingredients:
                         (a) connectivity - is arc_j reachable from arc_i?
                         (b) plausibility - does the driving distance between
                             them match the distance implied by the GPS step and
                             the elapsed time?
                         A standard, defensible form is
                             transition = exp(-|d_network - d_observed| / beta)
                         with beta tuned on train/student-test only.

VITERBI RECURRENCE
------------------
    delta_0(k)      = log emission_0(k)
    delta_t(k)      = log emission_t(k) + max_j [ delta_{t-1}(j) + log trans(j->k) ]
    backpointer_t(k)= argmax_j of the bracket above
    final           = argmax_k delta_T(k), then backtrack.

Do it in log space. Multiply hundreds of probabilities in float64 and you get
zeros, which silently produces a garbage route.

MUST HANDLE (each one is an explicit report question)
-----------------------------------------------------
- Observation with zero candidates -> skip it, but keep it in the output so the
  result aligns with the GPS timeline; document the decision.
- No valid transition into any candidate (disconnected network) -> fall back to
  emission-only for that step and count how often this happens. That count is
  a result worth reporting.
- Large GPS gaps / dropouts -> decide whether to reset the chain, widen the
  transition tolerance, or allow a multi-hop shortest path between candidates.
  Whatever you pick, report how many observations were affected.

REQUIRED EXPERIMENT (Deliverables 5 + Models & Training 9)
----------------------------------------------------------
Compare classifier-only predictions against classifier + Viterbi on the same
trips, reporting observation-level accuracy, route continuity, and
sequence-level performance. Include at least one trip where decoding made it
*worse* and explain why - the brief asks explicitly about cases where HMM did
not help, and an honest negative case scores better than a suspiciously perfect
story.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pathsnap.io.schema import CandidateSet, RoadNetwork


@dataclass
class DecoderConfig:
    """Every field is a knob whose value must be justified in the report."""

    emission_weight: float = 1.0     # exponent applied to classifier probability
    beta_m: float = 50.0             # transition tolerance, in metres
    connectivity_bonus: float = 5.0  # log-space bonus for a connected transition
    same_arc_bonus: float = 1.0      # log-space bonus for staying on the same arc
    max_gap_s: float = 30.0          # above this, treat as a GPS dropout
    reset_on_gap: bool = True


@dataclass
class DecodeResult:
    trip_id: str
    arc_per_obs: np.ndarray                  # int64, shape (n_obs,), -1 when no candidate
    log_likelihood: float
    n_skipped_obs: int = 0
    n_forced_emission_only: int = 0
    meta: dict = field(default_factory=dict)


def transition_logprob(
    network: RoadNetwork, arc_from: int, arc_to: int, observed_step_m: float, cfg: DecoderConfig
) -> float:
    """log P(arc_to at t | arc_from at t-1)."""
    raise NotImplementedError("Person 4: implement the transition model.")


def viterbi_decode(
    trip_id: str,
    candidates: CandidateSet,
    proba: np.ndarray,
    network: RoadNetwork,
    n_obs: int,
    cfg: DecoderConfig | None = None,
) -> DecodeResult:
    """Decode the most likely arc sequence for one trip.

    Parameters
    ----------
    proba : (n_pairs, n_classes) classifier output aligned row-for-row with
            `candidates.obs_idx` / `candidates.arc_id`.
    n_obs : number of GPS observations in the trip (output length).

    Returns
    -------
    DecodeResult with one predicted arc per observation.
    """
    raise NotImplementedError("Person 4: implement Viterbi decoding.")


def classifier_only(candidates: CandidateSet, proba: np.ndarray, n_obs: int) -> np.ndarray:
    """Per-observation argmax over candidates - the baseline Viterbi must beat.

    Implement this yourself (it is ~5 lines); do not import the decoder to
    produce the baseline, otherwise the comparison is not independent.
    """
    raise NotImplementedError("Person 4: implement the classifier-only baseline.")
