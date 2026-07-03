"""
Central configuration for all diagnostic thresholds — new modular system.

Design principles
-----------------
- Single source of truth: no cutoff is hardcoded outside this file.
- Plan A / Plan B memory fallback: LDELTOTAL is preferred; if absent, the
  classifiers automatically fall back to RAVLT_immediate. Both sets of cutoffs
  live here so the switch is transparent and flagged in the output.
- Method-aware: CSF, PET, and neuroimaging cutoffs are keyed by assay/tracer/
  FreeSurfer version. Unused methods are still present — switching is a
  one-parameter change in the calling code.
- Column map: maps plan-internal names to actual dataset column names, isolating
  name differences to a single location.

Relationship to thresholds.py
------------------------------
thresholds.py drives the legacy weighted-vote classifier (dx_rule_based.py) and
is intentionally unchanged. config.py drives the new protocol-aware modules
(dx_adni_protocol, dx_clinical_nia, dx_atn, dx_combined). Do not merge them.
"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
# Column mapping
# Maps internal plan-variable names → actual column names in the dataset.
# Centralising this here means the diagnostic modules never hard-code column
# names; they always look up via COLUMN_MAP.get(key, key).
# ─────────────────────────────────────────────────────────────────────────────

COLUMN_MAP: dict[str, str] = {
    # Demographics / education
    "PTEDUCAT":               "EDUCAT",
    # CDR
    "CDGLOBAL":               "CDRGLOB",
    "CDMEMORY":               "CDMEMORY",        # absent in current dataset
    "CDRSB":                  "CDRSB",
    # Memory tests — Plan A (primary) and Plan B (fallback)
    "LDELTOTAL":              "LDELTOTAL",        # absent in current dataset → triggers Plan B
    "RAVLT_immediate":        "RAVLT_immediate",
    # Other cognitive
    "MMSE":                   "MMSE",
    "MOCA":                   "MOCA",
    "FAQ":                    "FAQ",
    "ADAS13":                 "ADAS13",
    "ADAS11":                 "ADAS11",
    # Exclusion variables (absent in current dataset)
    "GDTOTAL":                "GDTOTAL",
    "HMSCORE":                "HMSCORE",
    # CSF
    "AB42_CSF":               "AB42_CSF",
    "AB40_CSF":               "AB40_CSF",
    "AB4240_CSF":             "AB4240_CSF",
    "PT181_CSF":              "PT181_CSF",
    "TTAU_CSF":               "TTAU_CSF",
    "PT181_AB42_CSF":         "PT181_AB42_CSF",
    "TTAU_AB42_CSF":          "TTAU_AB42_CSF",
    # PET — amyloid
    "SUMMARY_SUVR":           "SUMMARY_SUVR",    # amyloid composite (FBP/AV45)
    # PET — tau
    "TAU_METAROI":            "TAU_METAROI",
    "INFERIOR_TEMPORAL_SUVR": "INFERIOR_TEMPORAL_SUVR",
    "ENTORHINAL_SUVR":        "ENTORHINAL_SUVR",
    # Brain volumes (%ICV format in current dataset)
    "Hippocampus":            "Hippocampus%ICV",
    "Entorhinal_vol":         "Entorhinal%ICV",
    "Fusiform":               "Fusiform%ICV",
    "MidTemp":                "MidTemp%ICV",
    "Ventricles":             "Ventricles%ICV",
}

# ─────────────────────────────────────────────────────────────────────────────
# Memory tests
# ─────────────────────────────────────────────────────────────────────────────

# Fallback priority — first column found in the row is used.
# Plan A: LDELTOTAL (Logical Memory II Delayed, ADNI standard)
# Plan B: RAVLT_immediate (5-trial sum, used when LDELTOTAL absent)
MEMORY_TEST_PRIORITY: list[str] = ["LDELTOTAL", "RAVLT_immediate"]

# Plan A — Logical Memory II Delayed Recall, education-adjusted
# Source: Aisen et al. 2024, Alzheimer's & Dementia; ADNI Procedures Manuals
# Keys: cn_min = minimum score for CN classification
#        mci_max = maximum score consistent with MCI (≤ this = impaired memory)
ADNI_LM_CUTOFFS: dict[str, dict[str, dict[str, int]]] = {
    "ADNI1": {
        "16+":  {"cn_min": 9, "mci_max": 8,  "ad_max": 8},
        "8-15": {"cn_min": 5, "mci_max": 4,  "ad_max": 4},
        "0-7":  {"cn_min": 3, "mci_max": 2,  "ad_max": 2},
    },
    "ADNIGO2": {
        "16+":  {"cn_min": 9,  "emci_min": 9, "emci_max": 11, "lmci_max": 8, "ad_max": 8},
        "8-15": {"cn_min": 5,  "emci_min": 5, "emci_max": 9,  "lmci_max": 4, "ad_max": 4},
        "0-7":  {"cn_min": 3,  "emci_min": 3, "emci_max": 6,  "lmci_max": 2, "ad_max": 2},
    },
    "ADNI3": {
        "16+":  {"cn_min": 9, "mci_max": 10, "ad_max": 8},
        "8-15": {"cn_min": 5, "mci_max": 9,  "ad_max": 4},
        "0-7":  {"cn_min": 3, "mci_max": 6,  "ad_max": 2},
    },
    "ADNI4": {  # same criteria as ADNI3
        "16+":  {"cn_min": 9, "mci_max": 10, "ad_max": 8},
        "8-15": {"cn_min": 5, "mci_max": 9,  "ad_max": 4},
        "0-7":  {"cn_min": 3, "mci_max": 6,  "ad_max": 2},
    },
}

# Plan B — RAVLT Immediate Recall (5-trial sum, range 0–75), education-adjusted
# Applied when LDELTOTAL is absent. Cutoffs are ~1.5 SD below education-matched
# normative means adapted from Duff et al. 2008 (J Int Neuropsychol Soc) and
# Schmidt et al. 2015 (ADNI group statistics).
# NOTE: RAVLT_immediate is an immediate-recall proxy; it is less sensitive than
# delayed recall for MCI. Flag MEMORY_FALLBACK_RAVLT is emitted when this path is used.
RAVLT_IMMEDIATE_CUTOFFS: dict[str, dict[str, int]] = {
    "16+":  {"cn_min": 42, "impaired_max": 41, "ad_max": 35},
    "8-15": {"cn_min": 35, "impaired_max": 34, "ad_max": 28},
    "0-7":  {"cn_min": 28, "impaired_max": 27, "ad_max": 20},
}

# Education band boundaries (years of education → band key)
EDUCATION_BANDS: list[tuple[int, int, str]] = [
    (0,  7,  "0-7"),
    (8,  15, "8-15"),
    (16, 99, "16+"),
]
EDUCATION_BAND_DEFAULT: str = "8-15"   # used when PTEDUCAT/EDUCAT is missing

# ─────────────────────────────────────────────────────────────────────────────
# Clinical / staging gates
# ─────────────────────────────────────────────────────────────────────────────

# MMSE inclusion ranges per protocol phase — (min_inclusive, max_inclusive)
# Source: ADNI Procedures Manuals (phase-specific screening criteria)
ADNI_MMSE_GATES: dict[str, dict[str, tuple[int, int]]] = {
    "ADNI1":   {"cn_mci": (24, 30), "ad": (20, 26)},
    "ADNIGO2": {"cn_mci": (24, 30), "ad": (20, 26)},
    "ADNI3":   {"cn_mci": (24, 30), "ad": (20, 24)},
    "ADNI4":   {"cn_mci": (24, 30), "ad": (20, 24)},
}

# FAQ — Functional Activities Questionnaire (Pfeffer 1982)
FAQ_THRESHOLDS: dict[str, int] = {
    "functional_impairment": 6,    # FAQ >= 6 → functional impairment (supports Dementia)
    "mci_upper": 5,                # FAQ <= 5 consistent with preserved independence (MCI/CN)
}

# CDR-SB severity staging — O'Bryant et al. 2010, Arch Neurology
# Bounds are (lower_inclusive, upper_exclusive)
CDRSB_SEVERITY: dict[str, tuple[float, float]] = {
    "questionable": (0.5,  2.5),
    "mild":         (0.5,  4.5),
    "moderate":     (4.5,  9.5),
    "severe":       (9.5, 18.0),
}

# Exclusion criteria — applied only when apply_exclusions=True
# GDS (Geriatric Depression Scale): GDTOTAL >= 6 → probable depression → exclude
# Hachinski (vascular dementia screen): HMSCORE > 4 → exclude
EXCLUSION_GATES: dict[str, int] = {
    "GDS_MAX":       5,    # GDTOTAL <= 5 to be included
    "HACHINSKI_MAX": 4,    # HMSCORE <= 4 to be included
}

# ─────────────────────────────────────────────────────────────────────────────
# CSF biomarkers
# ─────────────────────────────────────────────────────────────────────────────
# Select via csf_assay= parameter.  Unused methods are kept for future use.
# Column names follow COLUMN_MAP (AB42_CSF, AB4240_CSF, PT181_CSF, TTAU_CSF).
# pos_below / pos_above: threshold direction for amyloid-positivity / tau-positivity.
# None = no validated cutoff available for this marker on this platform.

CSF_CUTOFFS: dict[str, dict] = {

    # ── Roche Elecsys (ADNI3/4 — current focus dataset) ──────────────────
    # Source: Hansson et al. 2018, JAMA Neurology; Bittner et al. 2016
    "ELECSYS": {
        "AB42_CSF":   {"pos_below": 980,   "unit": "pg/mL"},
        "AB4240_CSF": {"pos_below": 0.062, "unit": "ratio"},
        "PT181_CSF":  {"pos_above": 24,    "unit": "pg/mL"},
        "TTAU_CSF":   {"pos_above": 300,   "unit": "pg/mL"},
        "reference":  "Hansson 2018 JAMA Neurology; Bittner 2016",
    },

    # ── Fujirebio INNOTEST (ADNI1/GO/2) ──────────────────────────────────
    # Source: Shaw et al. 2009, Ann Neurology
    "INNOTEST": {
        "AB42_CSF":   {"pos_below": 192,  "unit": "pg/mL"},
        "AB4240_CSF": None,               # not standard output on INNOTEST
        "PT181_CSF":  {"pos_above": 23,   "unit": "pg/mL"},
        "TTAU_CSF":   {"pos_above": 93,   "unit": "pg/mL"},
        "reference":  "Shaw 2009 Ann Neurology",
    },

    # ── Fujirebio AlzBio3 / xMAP (ADNI1/GO/2 multiplex) ─────────────────
    # Same analytical platform as INNOTEST; values directly comparable.
    # Source: Olsson et al. 2016, Lancet Neurology (systematic review)
    "ALZBIO3": {
        "AB42_CSF":   {"pos_below": 192,  "unit": "pg/mL"},
        "AB4240_CSF": None,
        "PT181_CSF":  {"pos_above": 23,   "unit": "pg/mL"},
        "TTAU_CSF":   {"pos_above": 93,   "unit": "pg/mL"},
        "reference":  "Olsson 2016 Lancet Neurology — same scale as INNOTEST",
    },

    # ── Fujirebio Lumipulse G (automated, newer cohorts) ─────────────────
    # Source: Bayoumy et al. 2021, Alzheimer's Research & Therapy
    # NOTE: cutoffs vary by laboratory calibration; treat as indicative only.
    "LUMIPULSE": {
        "AB42_CSF":   {"pos_below": 669,   "unit": "pg/mL",
                       "note": "indicative — validate with local calibration"},
        "AB4240_CSF": {"pos_below": 0.081, "unit": "ratio",
                       "note": "indicative"},
        "PT181_CSF":  {"pos_above": 56.5,  "unit": "pg/mL",
                       "note": "indicative"},
        "TTAU_CSF":   {"pos_above": 404,   "unit": "pg/mL",
                       "note": "indicative"},
        "reference":  "Bayoumy 2021 Alzheimer's Research & Therapy — local calibration recommended",
    },

    # ── Mass spectrometry (Shimadzu / EUROIMMUN / IP-MS) ─────────────────
    # Best validated for Aβ42/40 ratio; absolute concentrations platform-specific.
    # Source: Palmqvist et al. 2019, Neurology
    "MASSSPECTROMETRY": {
        "AB42_CSF":   {"pos_below": None,  "unit": "pg/mL",
                       "note": "platform-specific; prefer ratio"},
        "AB4240_CSF": {"pos_below": 0.089, "unit": "ratio",
                       "note": "Palmqvist 2019"},
        "PT181_CSF":  None,               # not a standard mass-spec output
        "TTAU_CSF":   None,
        "reference":  "Palmqvist 2019 Neurology",
    },

    # ── Meso Scale Discovery (MSD / Sector Imager) ───────────────────────
    # MSD uses electrochemiluminescence; absolute values are lot-specific.
    # Universal cutoffs do not exist — local ROC calibration required.
    "MESOSCALEDISCOVERY": {
        "AB42_CSF":   {"pos_below": None, "unit": "pg/mL",
                       "note": "lot-specific — local calibration required"},
        "AB4240_CSF": {"pos_below": None, "unit": "ratio",
                       "note": "lot-specific"},
        "PT181_CSF":  {"pos_above": None, "unit": "pg/mL",
                       "note": "lot-specific"},
        "TTAU_CSF":   {"pos_above": None, "unit": "pg/mL",
                       "note": "lot-specific"},
        "reference":  "No universal published cutoff — local calibration required",
    },

    # ── ELISA (generic) ───────────────────────────────────────────────────
    # Highly kit-dependent; not recommended for cross-site comparisons.
    "ELISA": {
        "AB42_CSF":   {"pos_below": None, "unit": "pg/mL", "note": "kit-specific"},
        "AB4240_CSF": {"pos_below": None, "unit": "ratio", "note": "kit-specific"},
        "PT181_CSF":  {"pos_above": None, "unit": "pg/mL", "note": "kit-specific"},
        "TTAU_CSF":   {"pos_above": None, "unit": "pg/mL", "note": "kit-specific"},
        "reference":  "Kit-dependent — no universal cutoff",
    },

    # ── Saladax ───────────────────────────────────────────────────────────
    # NOTE: Saladax Biomedical is primarily a therapeutic drug monitoring platform.
    # No validated AD CSF cutoffs are known for this platform.
    # This entry is a placeholder — populate if a validated protocol is available.
    "SALADAX": {
        "AB42_CSF":   None,
        "AB4240_CSF": None,
        "PT181_CSF":  None,
        "TTAU_CSF":   None,
        "reference":  "No validated AD CSF cutoffs known — verify with data source",
    },

    # ── Synthetic (default for simulated/generated data, Elecsys scale) ──
    "SYNTHETIC": {
        "AB42_CSF":   {"pos_below": 980,   "unit": "pg/mL"},
        "AB4240_CSF": {"pos_below": 0.062, "unit": "ratio"},
        "PT181_CSF":  {"pos_above": 24,    "unit": "pg/mL"},
        "TTAU_CSF":   {"pos_above": 300,   "unit": "pg/mL"},
        "reference":  "Elecsys scale (Hansson 2018) — default for synthetic data",
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# PET biomarkers — amyloid
# ─────────────────────────────────────────────────────────────────────────────
# Select via amyloid_pet_tracer= parameter.
# SUMMARY_SUVR column maps to the composite cortical SUVR for the active tracer.

AMYLOID_PET_CUTOFFS: dict[str, dict] = {

    # Florbetapir / AV45 (FBP) — ADNI standard, current focus dataset
    # Source: Jagust et al. 2021, ADNI; Landau 2013
    "FBP": {
        "SUMMARY_SUVR": {"pos_above": 1.11, "unit": "SUVR",
                         "roi": "composite cortical, whole-cerebellum reference"},
        "centiloid":    {"pos_above": 24,   "unit": "CL"},
        "reference":    "Jagust 2021 ADNI; Klunk 2015 Alzheimer's & Dementia",
    },

    # Florbetaben (FBB)
    # Source: Sabri et al. 2015, NEJM; Rominger 2012
    "FBB": {
        "SUMMARY_SUVR": {"pos_above": 1.17, "unit": "SUVR",
                         "roi": "composite cortical"},
        "centiloid":    {"pos_above": 24,   "unit": "CL"},
        "reference":    "Sabri 2015 NEJM; Rominger 2012",
    },

    # AV45 is a common alias for FBP — resolved at runtime
    "AV45": "FBP",
}

# ─────────────────────────────────────────────────────────────────────────────
# PET biomarkers — tau
# ─────────────────────────────────────────────────────────────────────────────
# TAU_METAROI, INFERIOR_TEMPORAL_SUVR, ENTORHINAL_SUVR columns in dataset.
# Select via tau_pet_tracer= parameter.

TAU_PET_CUTOFFS: dict[str, dict] = {

    # Flortaucipir / AV1451 (FTP) — ADNI standard
    # Source: Schöll et al. 2016, Neuron; Jack et al. 2019, Brain
    "FTP": {
        "TAU_METAROI":            {"pos_above": 1.33, "unit": "SUVR",
                                   "note": "Braak entorhinal/fusiform/parahippocampal meta-ROI"},
        "INFERIOR_TEMPORAL_SUVR": {"pos_above": 1.36, "unit": "SUVR"},
        "ENTORHINAL_SUVR":        {"pos_above": 1.40, "unit": "SUVR"},
        "reference":              "Schöll 2016 Neuron; Jack 2019 Brain",
    },

    # MK6240 (Cerveau Technologies / Merck)
    # Source: Leuzy et al. 2020, Brain; Ossenkoppele 2021
    "MK6240": {
        "TAU_METAROI":            {"pos_above": 1.24, "unit": "SUVR",
                                   "note": "Braak meta-ROI; fewer validation studies than FTP"},
        "INFERIOR_TEMPORAL_SUVR": {"pos_above": 1.28, "unit": "SUVR",
                                   "note": "approximate — validate locally"},
        "ENTORHINAL_SUVR":        {"pos_above": 1.30, "unit": "SUVR",
                                   "note": "approximate"},
        "reference":              "Leuzy 2020 Brain",
    },

    # PI2620 (Life Molecular Imaging)
    # Emerging tracer; cutoffs under active revision.
    # Source: Kroth et al. 2019; Mueller et al. 2021
    "PI2620": {
        "TAU_METAROI":            {"pos_above": 1.23, "unit": "SUVR",
                                   "note": "preliminary — validate with local ROC before use"},
        "INFERIOR_TEMPORAL_SUVR": None,
        "ENTORHINAL_SUVR":        None,
        "reference":              "Kroth 2019; Mueller 2021 — emerging tracer, cutoffs under revision",
    },

    # AV1451 is a common alias for FTP — resolved at runtime
    "AV1451": "FTP",
}

# ─────────────────────────────────────────────────────────────────────────────
# Neuroimaging — brain volumes
# ─────────────────────────────────────────────────────────────────────────────
# Select via volume_scale= parameter.
#
# "pct_icv"  — current dataset format: volume expressed as percentage of ICV
#              (e.g. Hippocampus%ICV column stores values ~0.22–0.66)
#              Cutoffs from normalization_settings.json + Jack 2010, Brain
# "mm3"      — raw FreeSurfer output; ICV-correction strongly recommended before use
#
# Atrophy direction: below the threshold = atrophied (bad)
# Ventricles:        above the threshold = enlarged (bad, proxy for atrophy)

NEUROIMAGING_CUTOFFS: dict[str, dict] = {

    "pct_icv": {
        "Hippocampus": {"atrophy_below":   0.34, "unit": "%ICV"},
        "Entorhinal":  {"atrophy_below":   0.18, "unit": "%ICV"},
        "Fusiform":    {"atrophy_below":   0.90, "unit": "%ICV"},
        "MidTemp":     {"atrophy_below":   0.80, "unit": "%ICV"},
        "Ventricles":  {"expansion_above": 2.00, "unit": "%ICV"},
        "reference":   "Jack 2010 Brain; Risacher 2010 Arch Neurology; normalization_settings.json",
    },

    "mm3": {
        "Hippocampus": {"atrophy_below":    6000, "unit": "mm³"},
        "Entorhinal":  {"atrophy_below":    3000, "unit": "mm³"},
        "Fusiform":    {"atrophy_below":   16000, "unit": "mm³"},
        "MidTemp":     {"atrophy_below":   18000, "unit": "mm³"},
        "Ventricles":  {"expansion_above": 35000, "unit": "mm³"},
        "reference":   "Jack 2010 Brain — ICV correction strongly recommended",
    },
}

# FreeSurfer version notes
# Different pipeline versions produce systematically different absolute volumes.
# Subgrouping by version (as done here) reduces confounding. Notes below flag
# where version-specific cutoff validation may be needed.
FREESURFER_VERSION_NOTES: dict[str, str] = {
    "4.3": (
        "Early ADNI1 pipeline. Known segmentation differences from v5+. "
        "pct_icv format reduces (but does not eliminate) version effects."
    ),
    "5.1": (
        "Standard ADNI1/2/GO pipeline. pct_icv cutoffs validated on this version. "
        "Recommended baseline for cross-version comparisons."
    ),
    "7.2": (
        "ADNI4 pipeline. Improved hippocampal segmentation yields ~5–8% higher HC "
        "volumes vs v5. Consider adjusting HC atrophy_below upward if comparing across versions."
    ),
}

# ─────────────────────────────────────────────────────────────────────────────
# NIA-AA 2024 staging model (for dx_combined.py)
# ─────────────────────────────────────────────────────────────────────────────
# Jack et al. 2024, Alzheimer's & Dementia — 6-stage biological-clinical model.
# Stages 1–2: preclinical (amyloid+ but CN); 3: prodromal; 4–6: dementia by severity.

NIA_AA_2024_STAGES: dict[int, dict] = {
    1: {"clinical": "CN",       "amyloid": True,  "tau": False, "severity": None,
        "label": "Preclinical AD"},
    2: {"clinical": "CN",       "amyloid": True,  "tau": True,  "severity": None,
        "label": "Preclinical AD + tau"},
    3: {"clinical": "MCI",      "amyloid": True,  "tau": True,  "severity": None,
        "label": "Prodromal AD"},
    4: {"clinical": "Dementia", "amyloid": True,  "tau": None,  "severity": "mild",
        "label": "Mild AD dementia"},
    5: {"clinical": "Dementia", "amyloid": True,  "tau": None,  "severity": "moderate",
        "label": "Moderate AD dementia"},
    6: {"clinical": "Dementia", "amyloid": True,  "tau": None,  "severity": "severe",
        "label": "Severe AD dementia"},
}

# ─────────────────────────────────────────────────────────────────────────────
# Defaults for synthetic / simulated data
# ─────────────────────────────────────────────────────────────────────────────

SYNTHETIC_DEFAULTS: dict[str, object] = {
    "protocol":           "ADNI3",
    "csf_assay":          "SYNTHETIC",   # Elecsys scale, denormalised to clinical units
    "amyloid_pet_tracer": "FBP",
    "tau_pet_tracer":     "FTP",
    "volume_scale":       "pct_icv",     # %ICV format used in ADCourseMap pipeline
    "apply_exclusions":   False,         # GDTOTAL / HMSCORE absent in synthetic data
}

# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic output labels — shared across all modules
# ─────────────────────────────────────────────────────────────────────────────

DX_LABELS = {
    "CN":       "CN",
    "MCI":      "MCI",
    "DEMENTIA": "Dementia",
    "EMCI":     "EMCI",      # ADNIGO2 only
    "LMCI":     "LMCI",      # ADNIGO2 only
    "SMC":      "SMC",       # ADNIGO2 subjective memory complaints
    "UNKNOWN":  "Unknown",
    "EXCLUDED": "Excluded",
}

# ─────────────────────────────────────────────────────────────────────────────
# Flag registry — all flags that can appear in *_flag output columns
# ─────────────────────────────────────────────────────────────────────────────
# Each module emits zero or more of these. Multiple flags are joined with "|".

FLAGS = {
    # Memory test path
    "MEMORY_FALLBACK_RAVLT":          "LDELTOTAL absent — RAVLT_immediate used (Plan B)",
    "MISSING_MEMORY_TEST":            "Both LDELTOTAL and RAVLT_immediate absent — memory status unknown",
    # Missing primary variables
    "MISSING_CDGLOBAL":               "CDGLOBAL/CDRGLOB absent — dx=Unknown",
    "MISSING_MMSE_GATE_SKIPPED":      "MMSE absent — MMSE range gate skipped",
    "MISSING_LDELTOTAL":              "LDELTOTAL absent (informational, Plan B active)",
    "MISSING_CDMEMORY":               "CDMEMORY absent — gate replaced by CDGLOBAL",
    # Education band
    "ASSUMED_EDUC_BAND":              f"PTEDUCAT/EDUCAT absent — assumed band '{EDUCATION_BAND_DEFAULT}'",
    # Tie-break rules triggered
    "TIE_CDR0_MEM_IMPAIRED":          "CDR=0 but memory impaired — classified CN, review recommended",
    "TIE_FAQ_MISSING":                "FAQ absent in ambiguous branch — defaulted to MCI",
    "TIE_CDR_HIGH_NO_FUNC_IMPAIRMENT":"CDR>=0.5 without functional impairment — classified MCI",
    "TIE_MMSE_BELOW_AD_GATE_WITH_CDR0":"MMSE below AD gate with CDR=0 — classified Dementia",
    # Exclusion flags
    "EXCL_GDS":                       "Excluded: GDTOTAL >= 6 (depression screen)",
    "EXCL_HACHINSKI":                 "Excluded: HMSCORE > 4 (vascular dementia screen)",
    # Biomarker coverage
    "NO_BIOMARKERS":                  "No CSF or PET biomarkers available — clinical axis only",
    "PARTIAL_ATN":                    "AT(N) partially complete — some markers absent",
    "CONFLICTING_AMYLOID_PET_CSF":    "Amyloid PET and CSF Aβ42 discordant — PET used as primary",
    # Data quality
    "INSUFFICIENT_DATA":              "Insufficient data to classify — dx=Unknown",
}
