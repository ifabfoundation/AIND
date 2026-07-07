"""
Method B — Rule-based diagnosis calculator (NIA-AA 2018 / Jack 2013).

Entry point
-----------
    assign_dx_rule_based(visit_data, prev_dx, time_since_prev_years)

The function assigns CN / MCI / Dementia (or 'Unknown') to a single visit
using a multi-domain weighted majority vote, then checks temporal consistency
against the previous visit diagnosis.

Design rationale
----------------
Using a single scale (e.g. MMSE alone) is insufficient for clinical staging:
- MMSE has ceiling effects in CN and floor effects in severe dementia.
- Functional impairment (FAQ, CDR-SB) is required to distinguish MCI from Dementia.
- Memory (RAVLT) and neuroimaging (Hippocampus, Entorhinal) provide independent
  biological evidence that reduces the false-positive rate for borderline cases.

The weighted vote gives CDRSB the highest influence (3.0) because it is the
primary staging instrument in NIA-AA guidelines and the best single predictor
of clinical conversion in ADNI (O'Bryant 2010). When CDRSB is missing, the
remaining domains carry the decision, and confidence is downgraded.

The temporal consistency check does NOT change the biomarker-based diagnosis;
it flags implausible or unexpected longitudinal transitions so that downstream
consumers (e.g. Method A, pipeline quality metrics) can act on them.

References
----------
- Jack et al. 2013, Lancet Neurology — cascade model
- NIA-AA 2018 (Jack et al.) — A/T/(N) framework
- O'Bryant et al. 2010, Arch Neurology — CDR-SB cutpoints on ADNI
- Zamzmi et al. 2025, Comm. Engineering — FDA Consistency criterion
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from .thresholds import (
    CN_TO_DEMENTIA_MIN_YEARS,
    DOMAIN_WEIGHTS,
    IMPLAUSIBLE_TRANSITIONS,
    THRESHOLD_MAP,
    UNEXPECTED_TRANSITIONS,
)

VALID_STAGES = ("CN", "MCI", "Dementia")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_missing(value) -> bool:
    if value is None:
        return True
    try:
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return True


def _score_domain(value, thresholds: dict) -> Optional[str]:
    """
    Maps a numeric value to 'CN', 'MCI', or 'Dementia' using the threshold dict.
    Returns None if the value is missing or the domain cannot be scored.

    Threshold convention: (lower_bound, upper_bound)
        - lower_bound is None  → no lower bound (value < upper_bound)
        - upper_bound is None  → no upper bound (value >= lower_bound)
        - both present         → lower_bound <= value < upper_bound
    """
    if _is_missing(value):
        return None
    v = float(value)
    for stage in VALID_STAGES:
        lo, hi = thresholds[stage]
        if lo is None and hi is None:
            return stage
        if lo is None:
            if v < hi:
                return stage
        elif hi is None:
            if v >= lo:
                return stage
        else:
            if lo <= v < hi:
                return stage
    # edge: value exactly on the upper bound of Dementia (e.g. CDR-SB == 4.5)
    return VALID_STAGES[-1]


def _weighted_vote(domain_votes: dict[str, Optional[str]]) -> dict[str, float]:
    """Returns weighted score for each stage."""
    scores: dict[str, float] = {s: 0.0 for s in VALID_STAGES}
    for domain, stage in domain_votes.items():
        if stage is not None:
            scores[stage] += DOMAIN_WEIGHTS.get(domain, 1.0)
    return scores


def _confidence(
    primary_stage: str,
    primary_available: bool,
    total_weight: float,
    winning_weight: float,
    n_domains_used: int,
) -> str:
    """
    Confidence levels:
    - high:   primary domain available AND winning stage has > 75 % of total weight
    - medium: primary domain available OR winning stage has 60–75 % of total weight
    - low:    primary domain missing AND < 60 % agreement, or only 1 domain available
    """
    if n_domains_used == 0:
        return "low"
    agreement = winning_weight / total_weight if total_weight > 0 else 0.0
    if primary_available and agreement >= 0.75:
        return "high"
    if primary_available or agreement >= 0.60:
        return "medium"
    return "low"


# ---------------------------------------------------------------------------
# Temporal consistency check
# ---------------------------------------------------------------------------

def _check_temporal_consistency(
    current_dx: str,
    prev_dx: Optional[str],
    time_since_prev_years: Optional[float],
) -> tuple[bool, Optional[str]]:
    """
    Returns (is_consistent, flag_message).
    is_consistent = False does NOT change the assigned DX; it is a quality flag.
    """
    if prev_dx is None or prev_dx not in VALID_STAGES:
        return True, None

    transition = (prev_dx, current_dx)

    if transition in IMPLAUSIBLE_TRANSITIONS:
        return False, (
            f"Implausible transition {prev_dx} → {current_dx}: "
            "reversal of dementia is not expected in biological AD."
        )

    if transition == ("CN", "Dementia"):
        if time_since_prev_years is not None and time_since_prev_years < CN_TO_DEMENTIA_MIN_YEARS:
            return False, (
                f"Rapid CN → Dementia transition ({time_since_prev_years:.1f} yr): "
                f"expected ≥ {CN_TO_DEMENTIA_MIN_YEARS} yr without an intermediate MCI stage."
            )
        if time_since_prev_years is None:
            return False, (
                "CN → Dementia without intermediate MCI stage: "
                "time gap unknown, flag for review."
            )

    if transition in UNEXPECTED_TRANSITIONS:
        return False, (
            f"Unexpected transition {prev_dx} → {current_dx}: "
            "possible but uncommon — flag for review."
        )

    return True, None


# ---------------------------------------------------------------------------
# Main public interface
# ---------------------------------------------------------------------------

@dataclass
class DxResult:
    dx: str                                      # 'CN', 'MCI', 'Dementia', or 'Unknown'
    confidence: str                              # 'high', 'medium', 'low'
    evidence: dict[str, Optional[str]] = field(default_factory=dict)   # per-domain vote
    weighted_scores: dict[str, float] = field(default_factory=dict)    # raw vote weights
    temporal_consistent: bool = True
    temporal_flag: Optional[str] = None
    missing_domains: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "dx":                  self.dx,
            "confidence":          self.confidence,
            "evidence":            self.evidence,
            "weighted_scores":     self.weighted_scores,
            "temporal_consistent": self.temporal_consistent,
            "temporal_flag":       self.temporal_flag,
            "missing_domains":     self.missing_domains,
        }


def assign_dx_rule_based(
    visit_data: dict,
    prev_dx: Optional[str] = None,
    time_since_prev_years: Optional[float] = None,
) -> DxResult:
    """
    Assign a diagnosis to a single visit using multi-domain clinical rules.

    Parameters
    ----------
    visit_data : dict
        Biomarker values at clinical scale (post-denormalisation).
        Recognised keys (all optional, missing values are handled gracefully):
            - CDRSB            (CDR Sum of Boxes, 0–18)
            - MMSE             (Mini-Mental State Examination, 0–30)
            - ADAS13           (ADAS-Cog 13, 0–85)
            - FAQ              (Functional Activities Questionnaire, 0–30)
            - RAVLT_immediate  (RAVLT total immediate recall, 0–75)
            - Hippocampus      (volume in mm³ or normalised ratio)
            - Entorhinal       (cortical thickness in mm)
            - MidTemp          (cortical thickness in mm)
    prev_dx : str or None
        Diagnosis from the immediately preceding visit ('CN', 'MCI', 'Dementia').
        Pass None if this is the baseline visit or if the previous DX is unknown.
    time_since_prev_years : float or None
        Time in years between the previous and current visit.
        Used only for the CN → Dementia temporal check.

    Returns
    -------
    DxResult
        Structured result with dx, confidence, per-domain evidence, temporal flags,
        and list of missing domains.

    Examples
    --------
    >>> result = assign_dx_rule_based(
    ...     {"MMSE": 22, "CDRSB": 2.5, "FAQ": 3},
    ...     prev_dx="CN",
    ...     time_since_prev_years=1.5,
    ... )
    >>> result.dx
    'MCI'
    >>> result.confidence
    'high'
    """
    # -- Score each domain -------------------------------------------------
    domain_votes: dict[str, Optional[str]] = {}
    missing_domains: list[str] = []

    for domain, thresholds in THRESHOLD_MAP.items():
        value = visit_data.get(domain)
        vote = _score_domain(value, thresholds)
        if vote is None:
            missing_domains.append(domain)
        else:
            domain_votes[domain] = vote

    if not domain_votes:
        return DxResult(
            dx="Unknown",
            confidence="low",
            missing_domains=missing_domains,
        )

    # -- Weighted vote -----------------------------------------------------
    scores = _weighted_vote(domain_votes)
    total_weight = sum(DOMAIN_WEIGHTS[d] for d in domain_votes)
    winning_stage = max(scores, key=lambda s: (scores[s], VALID_STAGES.index(s)))
    winning_weight = scores[winning_stage]

    primary_domain = "CDRSB"
    primary_available = primary_domain in domain_votes

    conf = _confidence(
        primary_stage=winning_stage,
        primary_available=primary_available,
        total_weight=total_weight,
        winning_weight=winning_weight,
        n_domains_used=len(domain_votes),
    )

    # -- Temporal consistency ----------------------------------------------
    is_consistent, flag_msg = _check_temporal_consistency(
        winning_stage, prev_dx, time_since_prev_years
    )

    return DxResult(
        dx=winning_stage,
        confidence=conf,
        evidence=domain_votes,
        weighted_scores={s: round(scores[s], 2) for s in VALID_STAGES},
        temporal_consistent=is_consistent,
        temporal_flag=flag_msg,
        missing_domains=missing_domains,
    )


# ---------------------------------------------------------------------------
# Batch interface for DataFrames
# ---------------------------------------------------------------------------

def assign_dx_batch(
    df: pd.DataFrame,
    id_col: str = "ID",
    time_col: str = "TIME",
    prev_dx_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Apply assign_dx_rule_based row-by-row on a longitudinal DataFrame, propagating
    the computed DX as the previous-visit DX for the next visit of the same patient.

    Parameters
    ----------
    df : pd.DataFrame
        Must be sorted by (id_col, time_col) ascending.
        Columns matching THRESHOLD_MAP keys will be used automatically.
    id_col : str
        Patient identifier column.
    time_col : str
        Visit time column (numeric, same unit — used to compute time gaps in years
        only if the column contains float-year values, e.g. age or study year).
    prev_dx_col : str or None
        If provided, use this column as the initial previous-DX for each patient's
        first visit instead of None.

    Returns
    -------
    pd.DataFrame
        Original DataFrame with these new columns appended:
        - DX_rules           : assigned diagnosis
        - DX_rules_confidence: 'high' / 'medium' / 'low'
        - DX_rules_temporal_consistent : bool
        - DX_rules_temporal_flag       : str or NaN
        - DX_rules_missing_domains     : list[str]
    """
    df = df.sort_values([id_col, time_col]).copy()

    dx_col_out        = []
    conf_col_out      = []
    temporal_ok_out   = []
    temporal_flag_out = []
    missing_out       = []

    prev_dx_by_patient: dict = {}
    prev_time_by_patient: dict = {}

    for _, row in df.iterrows():
        patient_id = row[id_col]
        current_time = row[time_col]

        prev_dx: Optional[str] = prev_dx_by_patient.get(patient_id)
        prev_time: Optional[float] = prev_time_by_patient.get(patient_id)

        # Override with external prev_dx column on baseline visit (first visit only)
        if prev_dx is None and prev_dx_col and prev_dx_col in row.index:
            candidate = row[prev_dx_col]
            if not _is_missing(candidate) and candidate in VALID_STAGES:
                prev_dx = candidate

        time_gap: Optional[float] = None
        if prev_time is not None and not _is_missing(current_time):
            gap = float(current_time) - float(prev_time)
            if gap > 0:
                time_gap = gap   # assumes TIME is already in years

        visit_data = {
            col: row[col]
            for col in THRESHOLD_MAP
            if col in row.index
        }

        result = assign_dx_rule_based(visit_data, prev_dx, time_gap)

        dx_col_out.append(result.dx)
        conf_col_out.append(result.confidence)
        temporal_ok_out.append(result.temporal_consistent)
        temporal_flag_out.append(result.temporal_flag)
        missing_out.append(result.missing_domains)

        # Propagate computed DX (not the input DX) as prev for next visit
        if result.dx != "Unknown":
            prev_dx_by_patient[patient_id] = result.dx
        prev_time_by_patient[patient_id] = current_time

    df["DX_rules"]                      = dx_col_out
    df["DX_rules_confidence"]           = conf_col_out
    df["DX_rules_temporal_consistent"]  = temporal_ok_out
    df["DX_rules_temporal_flag"]        = temporal_flag_out
    df["DX_rules_missing_domains"]      = missing_out

    return df
