"""
Diagnosis 1 — pure clinical syndromic classifier (CN / MCI / Dementia).

Reproduces the ADNI clinical trial algorithm / NIA-AA 2011 clinical criteria
(McKhann / Albert / Sperling 2011), protocol-aware via config.ADNI_LM_CUTOFFS
and config.ADNI_MMSE_GATES. This is Diagnosis 1 of the two-axis design in
docs/adni_diagnosis_classifier_brief.md: it never reads biomarkers and is
never overridden by the biological axis (dx2_nia_atn.py / dx3_nia_combined.py).

Decision hierarchy: docs/adni_diagnosis_classifier_brief.md, sections 2.6-2.7.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from .config import ADNI_LM_CUTOFFS, ADNI_MMSE_GATES, DX_LABELS, FLAG_JOIN_SEP, SYNTHETIC_DEFAULTS
from .nia_protocol_resolver import (
    check_exclusions,
    derive_cdrglobal_from_cdrsb,
    faq_functional_impairment,
    get_value,
    is_missing,
    memory_status,
    mmse_in_range,
    resolve_education_band,
    resolve_memory_test,
    resolve_protocol,
)


def _emci_lmci_split(protocol: str, band: str, test_name: Optional[str], value: Optional[float]) -> Optional[str]:
    """
    ADNIGO2-only MCI granularity (EMCI vs LMCI), via LDELTOTAL bands in
    ADNI_LM_CUTOFFS["ADNIGO2"]. Returns None when it cannot be determined
    (e.g. RAVLT fallback active — RAVLT has no EMCI/LMCI split in config.py).
    SMC is not computed: no subjective-memory-complaint variable is available
    in the current column set (see DX_LABELS comment).
    """
    if protocol != "ADNIGO2" or test_name != "LDELTOTAL" or value is None:
        return None
    cutoffs = ADNI_LM_CUTOFFS["ADNIGO2"][band]
    if cutoffs["emci_min"] <= value <= cutoffs["emci_max"]:
        return "EMCI"
    if value <= cutoffs["lmci_max"]:
        return "LMCI"
    return None


def classify_syndromic(
    row,
    protocol: Optional[str] = None,
    apply_exclusions: Optional[bool] = None,
) -> dict:
    """
    Assigns a single-visit clinical diagnosis using the ADNI/NIA-AA 2011
    decision hierarchy. Returns a dict with:
        dx, dx_detailed, flags (list[str]), protocol_used, memory_test_used
    """
    if apply_exclusions is None:
        apply_exclusions = SYNTHETIC_DEFAULTS["apply_exclusions"]

    flags: list[str] = []

    excluded, excl_flags = check_exclusions(row, apply_exclusions)
    flags.extend(excl_flags)
    if excluded:
        return {
            "dx": DX_LABELS["EXCLUDED"],
            "dx_detailed": DX_LABELS["EXCLUDED"],
            "flags": flags,
            "protocol_used": None,
            "memory_test_used": None,
        }

    protocol_used = protocol or resolve_protocol(row)

    educ = get_value(row, "PTEDUCAT")
    band, used_default_band = resolve_education_band(educ)
    if used_default_band:
        flags.append("ASSUMED_EDUC_BAND")

    if is_missing(get_value(row, "LDELTOTAL")):
        flags.append("MISSING_LDELTOTAL")

    test_name, mem_value, is_fallback = resolve_memory_test(row)
    if test_name is None:
        mem_status = None
        flags.append("MISSING_MEMORY_TEST")
    else:
        if is_fallback:
            flags.append("MEMORY_FALLBACK_RAVLT")
        mem_status = memory_status(test_name, mem_value, band, protocol_used)

    cdglobal = get_value(row, "CDGLOBAL")
    if is_missing(cdglobal):
        # CDGLOBAL is often missing in the merged dataset due to a WP-2 mapping
        # bug (see STATUS_DIAGNOSI_PIPELINE_SINTETICA.md, section 8bis), not a
        # true ADNI collection gap — CDRSB is usually still present for the same
        # visit. Fall back to an O'Bryant-2008-derived approximation rather than
        # giving up immediately.
        derived_cdglobal = derive_cdrglobal_from_cdrsb(get_value(row, "CDRSB"))
        if derived_cdglobal is None:
            flags.append("MISSING_CDGLOBAL")
            return {
                "dx": DX_LABELS["UNKNOWN"],
                "dx_detailed": DX_LABELS["UNKNOWN"],
                "flags": flags,
                "protocol_used": protocol_used,
                "memory_test_used": test_name,
            }
        cdglobal = derived_cdglobal
        flags.append("CDGLOBAL_DERIVED_FROM_CDRSB")
    else:
        cdglobal = float(cdglobal)

    cdmemory = get_value(row, "CDMEMORY")
    if is_missing(cdmemory):
        flags.append("MISSING_CDMEMORY")
        cdmemory_ge_05 = cdglobal >= 0.5   # substitute gate, per FLAGS registry doc
    else:
        cdmemory_ge_05 = float(cdmemory) >= 0.5

    mmse = get_value(row, "MMSE")
    mmse_cn_mci = mmse_in_range(mmse, protocol_used, "cn_mci")
    mmse_ad = mmse_in_range(mmse, protocol_used, "ad")
    if mmse_cn_mci is None:
        flags.append("MISSING_MMSE_GATE_SKIPPED")

    faq = get_value(row, "FAQ")
    func_impaired = faq_functional_impairment(faq)

    dx: str

    if cdglobal == 0:
        # MMSE well below the AD gate overrides CDR=0 — severe cognitive
        # impairment despite an apparently normal CDR is treated as Dementia.
        ad_gate_lo = ADNI_MMSE_GATES.get(protocol_used, ADNI_MMSE_GATES[SYNTHETIC_DEFAULTS["protocol"]])["ad"][0]
        if mmse is not None and not is_missing(mmse) and float(mmse) < ad_gate_lo:
            dx = DX_LABELS["DEMENTIA"]
            flags.append("TIE_MMSE_BELOW_AD_GATE_WITH_CDR0")
        elif mem_status in ("mci", "ad", "impaired"):
            # CDR is the anchor per NIA-AA (ADNI DX is fundamentally CDR-dependent).
            dx = DX_LABELS["CN"]
            flags.append("TIE_CDR0_MEM_IMPAIRED")
        else:
            dx = DX_LABELS["CN"]

    elif cdglobal == 0.5 and cdmemory_ge_05 and mem_status in ("mci", "ad", "impaired") \
            and (mmse_cn_mci is not False) and not func_impaired:
        dx = DX_LABELS["MCI"]
        if func_impaired is None:
            flags.append("TIE_FAQ_MISSING")

    elif cdglobal >= 0.5 and func_impaired and (mem_status in ("ad",) or mmse_ad is not False):
        dx = DX_LABELS["DEMENTIA"]

    elif cdglobal >= 0.5:
        # Ambiguous branch — tie-break per brief section 2.7.
        if func_impaired is None:
            dx = DX_LABELS["MCI"]
            flags.append("TIE_FAQ_MISSING")
        elif not func_impaired:
            dx = DX_LABELS["MCI"]
            flags.append("TIE_CDR_HIGH_NO_FUNC_IMPAIRMENT")
        else:
            dx = DX_LABELS["DEMENTIA"]

    else:
        dx = DX_LABELS["UNKNOWN"]
        flags.append("INSUFFICIENT_DATA")

    dx_detailed = dx
    if dx == DX_LABELS["MCI"]:
        split = _emci_lmci_split(protocol_used, band, test_name, mem_value)
        if split is not None:
            dx_detailed = DX_LABELS[split]

    return {
        "dx": dx,
        "dx_detailed": dx_detailed,
        "flags": flags,
        "protocol_used": protocol_used,
        "memory_test_used": test_name,
    }


def assign_dx1_batch(
    df: pd.DataFrame,
    id_col: str = "ID",
    time_col: str = "TIME",
    protocol_col: str = "ORIGPROT",
    apply_exclusions: Optional[bool] = None,
) -> pd.DataFrame:
    """
    Applies classify_syndromic row-by-row. Each visit is classified
    independently — Diagnosis 1 has no temporal-propagation concept (unlike
    the legacy dx_rule_based.py). Appends:
        DX1_clinical, DX1_clinical_detailed, DX1_flags,
        DX1_protocol_used, DX1_memory_test_used
    """
    dx_out: list[str] = []
    dx_detailed_out: list[str] = []
    flags_out: list[Optional[str]] = []
    protocol_out: list[Optional[str]] = []
    memory_test_out: list[Optional[str]] = []

    for _, row in df.iterrows():
        protocol = resolve_protocol(row, protocol_col)
        result = classify_syndromic(row, protocol=protocol, apply_exclusions=apply_exclusions)
        dx_out.append(result["dx"])
        dx_detailed_out.append(result["dx_detailed"])
        flags_out.append(FLAG_JOIN_SEP.join(result["flags"]) if result["flags"] else None)
        protocol_out.append(result["protocol_used"])
        memory_test_out.append(result["memory_test_used"])

    df = df.copy()
    df["DX1_clinical"] = dx_out
    df["DX1_clinical_detailed"] = dx_detailed_out
    df["DX1_flags"] = flags_out
    df["DX1_protocol_used"] = protocol_out
    df["DX1_memory_test_used"] = memory_test_out
    return df
