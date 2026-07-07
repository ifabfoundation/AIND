"""
Diagnosis 2 — pure biological ATN status (NIA-AA 2018, Jack et al.).

Computed entirely from CSF / PET / volume biomarkers, independently of the
clinical diagnosis (dx1_nia_clinical.py). This module never receives a
clinical-dx input and must never be used to override Diagnosis 1 — that is
the two-axis rule from docs/adni_diagnosis_classifier_brief.md.

Method (assay/tracer/FreeSurfer version) is resolved once per dataset: an
explicit function parameter takes precedence, otherwise it is read from file
metadata (metadata.custom.CSF_filter / PET_filter / VOLUMES_filter — see
WP-2/Data_cleaning/automated_merge.py), otherwise it falls back to
config.SYNTHETIC_DEFAULTS. Today's merged datasets use one method per file,
so `metadata` is passed once per assign_dx2_batch call, not read per row.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from .config import (
    AMYLOID_PET_CUTOFFS,
    CSF_CUTOFFS,
    FLAG_JOIN_SEP,
    FREESURFER_VERSION_ALIASES,
    FREESURFER_VERSION_DEFAULT,
    METHOD_METADATA_KEYS,
    NEUROIMAGING_CUTOFFS,
    SYNTHETIC_DEFAULTS,
    TAU_PET_CUTOFFS,
)
from .nia_protocol_resolver import get_value, is_missing


def resolve_method(
    param_value,
    metadata: Optional[dict],
    metadata_key: str,
    flags_out: list[str],
) -> tuple[Optional[str], str]:
    """
    Generic resolver: explicit parameter > file metadata > None (caller
    applies its own default). Returns (value, source) where
    source in {"parameter", "metadata", "default"}.
    """
    if param_value is not None:
        if metadata and metadata.get(metadata_key) is not None:
            if str(metadata[metadata_key]).upper() != str(param_value).upper():
                flags_out.append("METHOD_PARAM_METADATA_MISMATCH")
        return param_value, "parameter"

    if metadata and metadata.get(metadata_key) is not None:
        flags_out.append("METHOD_FROM_METADATA")
        return metadata[metadata_key], "metadata"

    return None, "default"


def resolve_csf_assay(
    csf_assay: Optional[str] = None,
    metadata: Optional[dict] = None,
    flags_out: Optional[list[str]] = None,
) -> tuple[str, str]:
    flags_out = flags_out if flags_out is not None else []
    value, source = resolve_method(csf_assay, metadata, METHOD_METADATA_KEYS["csf_assay"], flags_out)
    if value is None:
        value, source = SYNTHETIC_DEFAULTS["csf_assay"], "default"
    return str(value).upper(), source


def resolve_amyloid_tracer(
    amyloid_pet_tracer: Optional[str] = None,
    metadata: Optional[dict] = None,
    flags_out: Optional[list[str]] = None,
) -> tuple[str, str]:
    flags_out = flags_out if flags_out is not None else []
    value, source = resolve_method(
        amyloid_pet_tracer, metadata, METHOD_METADATA_KEYS["amyloid_pet_tracer"], flags_out
    )
    if value is None:
        value, source = SYNTHETIC_DEFAULTS["amyloid_pet_tracer"], "default"
    resolved = str(value).upper()
    alias_target = AMYLOID_PET_CUTOFFS.get(resolved)
    if isinstance(alias_target, str):   # e.g. "AV45" -> "FBP"
        resolved = alias_target
    return resolved, source


def resolve_tau_tracer(
    tau_pet_tracer: Optional[str] = None,
    metadata: Optional[dict] = None,
    flags_out: Optional[list[str]] = None,
) -> tuple[str, str]:
    """
    No dedicated metadata key exists yet for tau tracer (PET_filter only
    distinguishes the amyloid tracer) — resolved from parameter or default only.
    """
    if tau_pet_tracer is not None:
        value, source = tau_pet_tracer, "parameter"
    else:
        value, source = SYNTHETIC_DEFAULTS["tau_pet_tracer"], "default"
    resolved = str(value).upper()
    alias_target = TAU_PET_CUTOFFS.get(resolved)
    if isinstance(alias_target, str):   # e.g. "AV1451" -> "FTP"
        resolved = alias_target
    return resolved, source


def resolve_volume_method(
    volume_scale: Optional[str] = None,
    freesurfer_version: Optional[str] = None,
    metadata: Optional[dict] = None,
    flags_out: Optional[list[str]] = None,
) -> tuple[str, str, str]:
    flags_out = flags_out if flags_out is not None else []
    scale = volume_scale or SYNTHETIC_DEFAULTS["volume_scale"]

    fs_raw, source = resolve_method(
        freesurfer_version, metadata, METHOD_METADATA_KEYS["volume_fs_version"], flags_out
    )
    if fs_raw is None:
        fs_key, source = FREESURFER_VERSION_DEFAULT, "default"
    else:
        fs_key = FREESURFER_VERSION_ALIASES.get(str(fs_raw), FREESURFER_VERSION_DEFAULT)

    return scale, fs_key, source


def _compute_amyloid(row, csf_key: str, pet_key: str, flags: list[str]) -> Optional[bool]:
    pet_val: Optional[bool] = None
    suvr = get_value(row, "SUMMARY_SUVR")
    if not is_missing(suvr) and pet_key in AMYLOID_PET_CUTOFFS:
        cutoff = AMYLOID_PET_CUTOFFS[pet_key]["SUMMARY_SUVR"]["pos_above"]
        pet_val = float(suvr) >= cutoff

    csf_val: Optional[bool] = None
    ab42 = get_value(row, "AB42_CSF")
    if not is_missing(ab42):
        entry = CSF_CUTOFFS.get(csf_key, {}).get("AB42_CSF")
        if entry and entry.get("pos_below") is not None:
            csf_val = float(ab42) < entry["pos_below"]

    if pet_val is not None and csf_val is not None:
        if pet_val != csf_val:
            flags.append("CONFLICTING_AMYLOID_PET_CSF")
        return pet_val   # PET is primary on conflict, per brief section 1
    return pet_val if pet_val is not None else csf_val


def _compute_tau(row, csf_key: str, tau_key: str) -> Optional[bool]:
    tau_val: Optional[bool] = None
    suvr = get_value(row, "TAU_METAROI")
    if not is_missing(suvr) and tau_key in TAU_PET_CUTOFFS:
        entry = TAU_PET_CUTOFFS[tau_key].get("TAU_METAROI")
        if entry and entry.get("pos_above") is not None:
            tau_val = float(suvr) > entry["pos_above"]

    csf_val: Optional[bool] = None
    ptau = get_value(row, "PT181_CSF")
    if not is_missing(ptau):
        entry = CSF_CUTOFFS.get(csf_key, {}).get("PT181_CSF")
        if entry and entry.get("pos_above") is not None:
            csf_val = float(ptau) > entry["pos_above"]

    return tau_val if tau_val is not None else csf_val   # PET primary, same convention as amyloid


def _compute_neurodegeneration(row, csf_key: str, vol_scale: str, fs_key: str, flags: list[str]) -> Optional[bool]:
    csf_val: Optional[bool] = None
    ttau = get_value(row, "TTAU_CSF")
    if not is_missing(ttau):
        entry = CSF_CUTOFFS.get(csf_key, {}).get("TTAU_CSF")
        if entry and entry.get("pos_above") is not None:
            csf_val = float(ttau) > entry["pos_above"]

    vol_val: Optional[bool] = None
    hippocampus = get_value(row, "Hippocampus")
    if not is_missing(hippocampus):
        if vol_scale == "pct_icv":
            fs_cutoffs = NEUROIMAGING_CUTOFFS["pct_icv"].get(fs_key)
            if fs_cutoffs is None:
                flags.append("FS_VERSION_CUTOFFS_UNCONFIRMED")
                hc_cutoff = None
            else:
                hc_cutoff = fs_cutoffs.get("Hippocampus")
                if hc_cutoff is None:
                    flags.append("FS_VERSION_CUTOFFS_UNCONFIRMED")
                elif fs_cutoffs.get("is_proxy"):
                    flags.append("FS_VERSION_CUTOFF_PROXY_FROM_5.1")
        else:
            hc_cutoff = NEUROIMAGING_CUTOFFS.get(vol_scale, {}).get("Hippocampus")
        if hc_cutoff:
            vol_val = float(hippocampus) < hc_cutoff["atrophy_below"]

    return csf_val if csf_val is not None else vol_val


def _atn_label(A: Optional[bool], T: Optional[bool], N: Optional[bool]) -> str:
    """Derived label per NIA-AA 2018 (brief section 3): A- / A+ / A+T+ / etc."""
    if A is None:
        return "A?"
    if not A:
        return "A-"
    label = "A+"
    label += "T+" if T else ("T-" if T is False else "T?")
    if N is True:
        label += "N+"
    elif N is False:
        label += "N-"
    return label


def classify_atn(
    row,
    csf_assay: Optional[str] = None,
    amyloid_pet_tracer: Optional[str] = None,
    tau_pet_tracer: Optional[str] = None,
    volume_scale: Optional[str] = None,
    freesurfer_version: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> dict:
    """
    Assigns A/T/N status for a single row. Returns:
        {"A": bool|None, "T": bool|None, "N": bool|None,
         "ATN_label": str, "flags": list[str], "method_source": dict}
    """
    flags: list[str] = []

    csf_key, csf_source = resolve_csf_assay(csf_assay, metadata, flags)
    pet_key, pet_source = resolve_amyloid_tracer(amyloid_pet_tracer, metadata, flags)
    tau_key, tau_source = resolve_tau_tracer(tau_pet_tracer, metadata, flags)
    vol_scale, fs_key, vol_source = resolve_volume_method(volume_scale, freesurfer_version, metadata, flags)

    A = _compute_amyloid(row, csf_key, pet_key, flags)
    T = _compute_tau(row, csf_key, tau_key)
    N = _compute_neurodegeneration(row, csf_key, vol_scale, fs_key, flags)

    if A is None and T is None and N is None:
        flags.append("NO_BIOMARKERS")
    elif A is None or T is None or N is None:
        flags.append("PARTIAL_ATN")

    return {
        "A": A,
        "T": T,
        "N": N,
        "ATN_label": _atn_label(A, T, N),
        "flags": flags,
        "method_source": {
            "csf_assay": csf_source,
            "amyloid_pet_tracer": pet_source,
            "tau_pet_tracer": tau_source,
            "volume": vol_source,
        },
    }


def assign_dx2_batch(
    df: pd.DataFrame,
    metadata: Optional[dict] = None,
    csf_assay: Optional[str] = None,
    amyloid_pet_tracer: Optional[str] = None,
    tau_pet_tracer: Optional[str] = None,
    volume_scale: Optional[str] = None,
    freesurfer_version: Optional[str] = None,
) -> pd.DataFrame:
    """
    Applies classify_atn row-by-row using ONE resolved method set for the
    whole DataFrame (today's merged files carry a single CSF/PET/volume
    method per file — see resolve_method). Appends:
        DX2_A, DX2_T, DX2_N, DX2_ATN_label, DX2_flags,
        DX2_method_source, DX2_method_uniform
    `DX2_method_uniform` is always True today (placeholder for a future
    per-row resolution path if a dataset ever mixes methods within one file).
    """
    a_out: list[Optional[bool]] = []
    t_out: list[Optional[bool]] = []
    n_out: list[Optional[bool]] = []
    label_out: list[str] = []
    flags_out: list[Optional[str]] = []
    source_out: list[dict] = []

    for _, row in df.iterrows():
        result = classify_atn(
            row,
            csf_assay=csf_assay,
            amyloid_pet_tracer=amyloid_pet_tracer,
            tau_pet_tracer=tau_pet_tracer,
            volume_scale=volume_scale,
            freesurfer_version=freesurfer_version,
            metadata=metadata,
        )
        a_out.append(result["A"])
        t_out.append(result["T"])
        n_out.append(result["N"])
        label_out.append(result["ATN_label"])
        flags_out.append(FLAG_JOIN_SEP.join(result["flags"]) if result["flags"] else None)
        source_out.append(result["method_source"])

    df = df.copy()
    df["DX2_A"] = a_out
    df["DX2_T"] = t_out
    df["DX2_N"] = n_out
    df["DX2_ATN_label"] = label_out
    df["DX2_flags"] = flags_out
    df["DX2_method_source"] = source_out
    df["DX2_method_uniform"] = True
    return df
