"""
Diagnosis 3 — combined clinical + biological staging (NIA-AA 2024, Jack et al.).

Combines Diagnosis 1's clinical stage (dx1_nia_clinical.py) with Diagnosis 2's
ATN status (dx2_nia_atn.py) to assign one of config.NIA_AA_2024_STAGES (1-6). This
does not replace either axis — it is a third, independent output, per the
two-axis design in docs/adni_diagnosis_classifier_brief.md.

Call order: assign_dx1_batch -> assign_dx2_batch -> assign_dx3_batch (or merge
two separately-computed frames on id_col/time_col before calling this module).
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from .config import FLAG_JOIN_SEP, NIA_AA_2024_STAGES
from .nia_protocol_resolver import cdrsb_severity_band


def resolve_severity(cdrsb) -> tuple[Optional[str], list[str]]:
    """
    Maps CDRSB_SEVERITY bands onto NIA_AA_2024_STAGES' severity keys
    (mild/moderate/severe). The "questionable" band (0.5-2.5) has no direct
    stage 4-6 counterpart — folded into "mild" with a flag documenting the
    approximation, per user decision.
    """
    band = cdrsb_severity_band(cdrsb)
    if band is None:
        return None, []
    if band == "questionable":
        return "mild", ["SEVERITY_QUESTIONABLE_FOLDED_TO_MILD"]
    return band, []


def combine_stage(
    clinical_dx: Optional[str],
    A: Optional[bool],
    T: Optional[bool],
    severity: Optional[str],
) -> Optional[int]:
    """
    Looks up NIA_AA_2024_STAGES by matching clinical + amyloid + tau (+
    severity for stages 4-6). Returns None (not "Unknown") when biomarkers
    are missing or clinical_dx is Unknown/Excluded — the framework only
    stages subjects confirmed on the amyloid-positive AD continuum.
    """
    if A is not True or clinical_dx not in ("CN", "MCI", "Dementia"):
        return None

    for stage, info in NIA_AA_2024_STAGES.items():
        if info["clinical"] != clinical_dx:
            continue
        if info["amyloid"] is not None and info["amyloid"] != A:
            continue
        if info["tau"] is not None:
            if T is None or info["tau"] != T:
                continue
        if info["severity"] is not None and severity != info["severity"]:
            continue
        return stage

    return None


def assign_dx3_batch(
    df: pd.DataFrame,
    dx1_col: str = "DX1_clinical",
    dx2_a_col: str = "DX2_A",
    dx2_t_col: str = "DX2_T",
    cdrsb_col: str = "CDRSB",
) -> pd.DataFrame:
    """
    Requires df to already carry DX1/DX2 output columns. Appends:
        DX3_stage (1-6 or None), DX3_label, DX3_flags
    """
    stage_out: list[Optional[int]] = []
    label_out: list[Optional[str]] = []
    flags_out: list[Optional[str]] = []

    for _, row in df.iterrows():
        clinical_dx = row.get(dx1_col)
        A = row.get(dx2_a_col)
        T = row.get(dx2_t_col)
        cdrsb = row.get(cdrsb_col)

        severity, flags = resolve_severity(cdrsb)
        stage = combine_stage(clinical_dx, A, T, severity)

        stage_out.append(stage)
        label_out.append(NIA_AA_2024_STAGES[stage]["label"] if stage is not None else None)
        flags_out.append(FLAG_JOIN_SEP.join(flags) if flags else None)

    df = df.copy()
    df["DX3_stage"] = stage_out
    df["DX3_label"] = label_out
    df["DX3_flags"] = flags_out
    return df
