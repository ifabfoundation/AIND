"""
NIA-AA clinical resolution utilities — row-level helpers, no decision logic.

This module answers "given this row and config.py, what education band / memory
test / gate applies?" It never decides CN / MCI / Dementia — that happens in
dx1_nia_clinical.py, which composes these functions the same way dx_rule_based.py
composes thresholds.py.

Reference: docs/adni_diagnosis_classifier_brief.md, sections 1-2 (amended — see
the note on the single-reference decision replacing ADNI-phase-aware cutoffs).
"""

from __future__ import annotations

import math
from typing import Optional

from .config import (
    MEMORY_IMPAIRMENT_CUTOFFS,
    CDRSB_SEVERITY,
    CDRSB_TO_CDRGLOBAL,
    COLUMN_MAP,
    EDUCATION_BAND_DEFAULT,
    EDUCATION_BANDS,
    EXCLUSION_GATES,
    FAQ_THRESHOLDS,
    MMSE_GATES,
    MEMORY_TEST_PRIORITY,
    RAVLT_IMMEDIATE_CUTOFFS,
)


def is_missing(value) -> bool:
    if value is None:
        return True
    try:
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return True


def get_value(row, key: str):
    """Look up a plan-internal variable name via COLUMN_MAP, then read from row."""
    column = COLUMN_MAP.get(key, key)
    if hasattr(row, "get"):
        return row.get(column)
    return row[column] if column in row else None


def resolve_education_band(educ_years) -> tuple[str, bool]:
    """
    Returns (band_key, used_default). used_default=True when educ_years is
    missing and EDUCATION_BAND_DEFAULT was used instead (drives
    FLAGS["ASSUMED_EDUC_BAND"]).
    """
    if is_missing(educ_years):
        return EDUCATION_BAND_DEFAULT, True
    years = float(educ_years)
    for lo, hi, band in EDUCATION_BANDS:
        if lo <= years <= hi:
            return band, False
    return EDUCATION_BAND_DEFAULT, True


def resolve_memory_test(row) -> tuple[Optional[str], Optional[float], bool]:
    """
    Iterates MEMORY_TEST_PRIORITY (Plan A: LDELTOTAL, Plan B: RAVLT_immediate).
    Returns (test_name, value, is_fallback).
    is_fallback=True when RAVLT_immediate is used because LDELTOTAL is absent
    (drives FLAGS["MEMORY_FALLBACK_RAVLT"]).
    Returns (None, None, False) when both are absent
    (drives FLAGS["MISSING_MEMORY_TEST"]).
    """
    for index, test_name in enumerate(MEMORY_TEST_PRIORITY):
        value = get_value(row, test_name)
        if not is_missing(value):
            return test_name, float(value), index > 0
    return None, None, False


def memory_status(test_name: str, value: float, band: str) -> str:
    """
    Maps a memory test score to "normal" / "mci" / "ad" (LDELTOTAL, via
    MEMORY_IMPAIRMENT_CUTOFFS) or "normal" / "impaired" (RAVLT_immediate).
    """
    if test_name == "LDELTOTAL":
        cutoffs = MEMORY_IMPAIRMENT_CUTOFFS[band]
        # ad_max/mci_max checked before cn_min: the table deliberately overlaps
        # with cn_min (brief section 2.2/2.7) — e.g. band "16+" has cn_min=9 and
        # mci_max=10, so 9-10 must resolve to "mci" here, letting the CDR-based
        # decision hierarchy (and its tie-break flags) decide, rather than this
        # function silently defaulting to "normal".
        ad_max = cutoffs.get("ad_max")
        impaired_upper = cutoffs.get("mci_max")
        if ad_max is not None and value <= ad_max:
            return "ad"
        if impaired_upper is not None and value <= impaired_upper:
            return "mci"
        return "normal"

    if test_name == "RAVLT_immediate":
        cutoffs = RAVLT_IMMEDIATE_CUTOFFS[band]
        if value >= cutoffs["cn_min"]:
            return "normal"
        return "impaired"

    raise ValueError(f"Unknown memory test: {test_name}")


def mmse_in_range(mmse, category: str) -> Optional[bool]:
    """
    category is "cn_mci" or "ad". Returns None (gate skipped) if mmse is
    missing (drives FLAGS["MISSING_MMSE_GATE_SKIPPED"]).
    """
    if is_missing(mmse):
        return None
    lo, hi = MMSE_GATES[category]
    return lo <= float(mmse) <= hi


def faq_functional_impairment(faq) -> Optional[bool]:
    """Returns None when FAQ is missing (drives FLAGS["TIE_FAQ_MISSING"] upstream)."""
    if is_missing(faq):
        return None
    return float(faq) >= FAQ_THRESHOLDS["functional_impairment"]


def cdrsb_severity_band(cdrsb) -> Optional[str]:
    """Returns the CDRSB_SEVERITY band name, or None if cdrsb is missing/out of range."""
    if is_missing(cdrsb):
        return None
    value = float(cdrsb)
    for band, (lo, hi) in CDRSB_SEVERITY.items():
        if lo <= value < hi:
            return band
    return None


def derive_cdrglobal_from_cdrsb(cdrsb) -> Optional[float]:
    """
    Approximates CDR Global from CDR Sum of Boxes via config.CDRSB_TO_CDRGLOBAL
    (O'Bryant et al. 2008, Arch Neurology). Returns None if cdrsb is missing or
    outside all published bounds (drives FLAGS["MISSING_CDGLOBAL"] upstream,
    same as an outright missing CDGLOBAL).
    """
    if is_missing(cdrsb):
        return None
    value = float(cdrsb)
    for lo, hi, cdrglobal in CDRSB_TO_CDRGLOBAL:
        if lo <= value <= hi:
            return cdrglobal
    return None


def check_exclusions(row, apply_exclusions: bool) -> tuple[bool, list[str]]:
    """
    Returns (excluded, flags) using EXCLUSION_GATES (GDTOTAL / HMSCORE).
    No-op (False, []) when apply_exclusions=False.
    """
    if not apply_exclusions:
        return False, []

    flags: list[str] = []
    excluded = False

    gdtotal = get_value(row, "GDTOTAL")
    if not is_missing(gdtotal) and float(gdtotal) > EXCLUSION_GATES["GDS_MAX"]:
        excluded = True
        flags.append("EXCL_GDS")

    hmscore = get_value(row, "HMSCORE")
    if not is_missing(hmscore) and float(hmscore) > EXCLUSION_GATES["HACHINSKI_MAX"]:
        excluded = True
        flags.append("EXCL_HACHINSKI")

    return excluded, flags
