"""
Clinically validated thresholds for AD diagnosis staging — Method B (rule-based).

Sources
-------
- Jack et al. 2013, Lancet Neurology (DOI: 10.1016/S1474-4422(12)70291-0)
  Biomarker cascade model and CDR-SB staging.
- NIA-AA 2018 (Jack et al. 2018, Alzheimer's & Dementia)
  A/T/(N) framework — updated diagnostic criteria.
- O'Bryant et al. 2010, Archives of Neurology
  CDR-SB cutpoints validated on ADNI: CN ≤ 0.5, MCI ≤ 4.5, Dementia > 4.5.
- Folstein 1975 (MMSE): widely adopted thresholds 24/18.
- Pfeffer et al. 1982 (FAQ): FAQ ≥ 6 marks functional impairment consistent with Dementia.
- Rosen et al. 1984 (ADAS-Cog); ADNI-specific cutpoints from Mohs et al. 1997.
- Schmidt et al. 2015 (RAVLT on ADNI): approximate thresholds by group.
- Shaw et al. 2009 / Hansson et al. 2018: CSF Aβ42 / p-tau / total-tau cutpoints.
- Jagust et al. 2021 (ADNI): AV45 amyloid PET SUVR cutpoint 1.11.
- Landau et al. 2012: FDG-PET hypometabolism on ADNI.

Note on plasma biomarkers
-------------------------
Plasma Aβ42/40, p-tau 217, GFAP etc. are intentionally excluded: they are present
in too few subjects in the current ADNI cleaned dataset to be reliable.

Note on APOE4 and Gender
------------------------
NIA-AA 2018 explicitly states that APOE4 is a genetic RISK FACTOR, not a diagnostic
criterion. Gender affects progression rate but not the diagnostic label. Neither is
included in the weighted vote. Both are recorded as context variables in the output
for downstream analysis (e.g. cofactor stratification in Method A).

Note on Age
-----------
Age is used only as a context variable for plausibility flagging (e.g. Dementia
below age 55 → early-onset flag). It does not vote for a stage.
"""

# ---------------------------------------------------------------------------
# Domain weights for weighted majority vote
# Ordered by clinical validation strength on ADNI (literature-informed)
# ---------------------------------------------------------------------------
DOMAIN_WEIGHTS: dict[str, float] = {
    # -- Upstream biological (A/T/(N) cascade) --------------------------------
    "ABETA":           2.0,   # CSF Aβ42: amyloid (A marker) — earliest detectable change
    "PTAU":            2.0,   # CSF p-tau 181: tau pathology (T marker) — AD-specific
    "TAU":             1.0,   # CSF total tau: neurodegeneration (N marker) — less specific
    "AV45":            2.0,   # Amyloid PET SUVR — direct in-vivo amyloid visualization
    "FDG":             1.0,   # FDG-PET SUVR — hypometabolism (N marker) — less specific
    # -- Clinical downstream --------------------------------------------------
    "CDRSB":           3.0,   # Primary staging tool in NIA-AA; highest downstream weight
    "FAQ":             2.0,   # Functional impairment; critical to separate MCI from Dementia
    "MMSE":            2.0,   # Most widely used; ceiling/floor effects reduce weight vs CDRSB
    "ADAS13":          1.5,   # Sensitive for mild impairment; validated for trial eligibility
    "RAVLT_immediate": 1.0,   # Episodic memory; useful for early MCI detection
    # -- Structural neuroimaging (supporting) ---------------------------------
    "Hippocampus":     0.5,
    "Entorhinal":      0.5,
    "MidTemp":         0.5,
}

# ---------------------------------------------------------------------------
# Per-domain thresholds  →  (lower_bound, upper_bound) inclusive
# Convention: (None, x) means "value < x"; (x, None) means "value >= x"
# ---------------------------------------------------------------------------

# CDR Sum of Boxes — O'Bryant 2010, validated on ADNI
# CN ≤ 0.5 | 0.5 < MCI ≤ 4.5 | Dementia > 4.5
CDRSB: dict[str, tuple] = {
    "CN":       (None, 0.5),
    "MCI":      (0.5,  4.5),
    "Dementia": (4.5,  None),
}

# Mini-Mental State Examination — Folstein 1975
# CN ≥ 24 | 18 ≤ MCI < 24 | Dementia < 18
MMSE: dict[str, tuple] = {
    "CN":       (24,   None),
    "MCI":      (18,   24),
    "Dementia": (None, 18),
}

# ADAS-Cog 13 — Mohs 1997 / ADNI convention
# Lower score = better cognition (range 0–85 in standard, ~0–70 in ADNI)
# CN < 11 | 11 ≤ MCI ≤ 35 | Dementia > 35
ADAS13: dict[str, tuple] = {
    "CN":       (None, 11),
    "MCI":      (11,   35),
    "Dementia": (35,   None),
}

# FAQ — Functional Activities Questionnaire (Pfeffer 1982)
# CN ≤ 1 | 1 < MCI ≤ 5 | Dementia > 5
FAQ: dict[str, tuple] = {
    "CN":       (None, 2),
    "MCI":      (2,    6),
    "Dementia": (6,    None),
}

# RAVLT immediate recall — Schmidt 2015 (ADNI group means ± 1 SD)
# CN ≥ 40 | 25 ≤ MCI < 40 | Dementia < 25
# Note: performance is education/age-dependent; thresholds are approximate.
RAVLT_immediate: dict[str, tuple] = {
    "CN":       (40,   None),
    "MCI":      (25,   40),
    "Dementia": (None, 25),
}

# ---------------------------------------------------------------------------
# Neuroimaging (supporting evidence only — weight ≤ 0.5 each)
# Values are in ADNI normalised units (ratio to ICV or mm³ depending on
# the datalake level). Thresholds are approximate group medians from
# Jack 2010 (ADNI baseline). Adjust via NEUROIMAGING_THRESHOLDS if the
# data level differs.
# Convention here: higher value = LESS atrophy (not inverted)
# ---------------------------------------------------------------------------
Hippocampus: dict[str, tuple] = {
    "CN":       (6800, None),   # mm³ approx
    "MCI":      (6000, 6800),
    "Dementia": (None, 6000),
}

Entorhinal: dict[str, tuple] = {
    "CN":       (3.8,  None),   # mm (cortical thickness)
    "MCI":      (3.2,  3.8),
    "Dementia": (None, 3.2),
}

MidTemp: dict[str, tuple] = {
    "CN":       (3.0,  None),
    "MCI":      (2.5,  3.0),
    "Dementia": (None, 2.5),
}

# ---------------------------------------------------------------------------
# CSF biomarkers — upstream A/T/(N) markers (Jack 2013 cascade)
#
# IMPORTANT — platform dependency:
#   CSF Aβ42 values in ADNI come from different assay platforms across cohorts:
#     ADNI-1/GO/2  →  Innotest or AlzBio3/xMAP  (range ~50–1 000 pg/mL, cutoff ~192)
#     ADNI-3       →  Elecsys (Roche)            (range ~200–4 000 pg/mL, cutoff ~980)
#   Thresholds below use the Elecsys scale (ADNI-3 standard).
#   If your dataset uses a different platform, rescale or update the values here.
#   Similarly for TAU and PTAU (Elecsys cutoffs: ~300 and ~24 pg/mL respectively).
# ---------------------------------------------------------------------------

# CSF Aβ42 (amyloid — A marker)
# Higher ABETA = better (more clearance, less pathology).
# Cutpoint 980 pg/mL (Elecsys): Shaw 2009, updated by Hansson 2018.
ABETA: dict[str, tuple] = {
    "CN":       (980,  None),   # A− (amyloid negative)
    "MCI":      (600,  980),    # A+ moderate load
    "Dementia": (None, 600),    # A+ high load
}

# CSF phosphorylated tau 181 (tau pathology — T marker, AD-specific)
# Higher PTAU = worse. Cutpoint 24 pg/mL (Elecsys, Hansson 2018).
PTAU: dict[str, tuple] = {
    "CN":       (None, 24),     # T−
    "MCI":      (24,   40),     # T+ moderate
    "Dementia": (40,   None),   # T+ high
}

# CSF total tau (neurodegeneration — N marker, less AD-specific)
# Higher TAU = worse. Cutpoint 300 pg/mL (Elecsys).
TAU: dict[str, tuple] = {
    "CN":       (None, 300),
    "MCI":      (300,  500),
    "Dementia": (500,  None),
}

# ---------------------------------------------------------------------------
# PET biomarkers
# ---------------------------------------------------------------------------

# AV45 amyloid PET (florbetapir SUVR, whole-cerebellum reference)
# Higher AV45 = more amyloid deposition = worse.
# Cutpoint 1.11 SUVR (Jagust 2021 / ADNI standard).
AV45: dict[str, tuple] = {
    "CN":       (None, 1.11),   # amyloid negative
    "MCI":      (1.11, 1.40),   # amyloid positive, moderate
    "Dementia": (1.40, None),   # amyloid positive, high load
}

# FDG-PET (composite SUVR of AD-signature regions: angular, temporal, posterior cingulate)
# Lower FDG = more hypometabolism = worse (N marker, less AD-specific than amyloid).
# Cutpoints approximate from Landau 2012 (ADNI baseline group means).
FDG: dict[str, tuple] = {
    "CN":       (1.30, None),
    "MCI":      (1.10, 1.30),
    "Dementia": (None, 1.10),
}

# ---------------------------------------------------------------------------
# Age context — NOT a voting domain
# Used only to flag implausible diagnoses given the patient's age.
# ---------------------------------------------------------------------------
EARLY_ONSET_DEMENTIA_AGE: float = 55.0   # Dementia below this age → early-onset flag
VERY_OLD_AGE: float = 88.0               # Normal aging effects may inflate false positives

# Convenience map — used by dx_rule_based.py to iterate over domains
THRESHOLD_MAP: dict[str, dict] = {
    # upstream biological
    "ABETA":           ABETA,
    "PTAU":            PTAU,
    "TAU":             TAU,
    "AV45":            AV45,
    "FDG":             FDG,
    # clinical downstream
    "CDRSB":           CDRSB,
    "MMSE":            MMSE,
    "ADAS13":          ADAS13,
    "FAQ":             FAQ,
    "RAVLT_immediate": RAVLT_immediate,
    # structural neuroimaging
    "Hippocampus":     Hippocampus,
    "Entorhinal":      Entorhinal,
    "MidTemp":         MidTemp,
}

# ---------------------------------------------------------------------------
# Temporal consistency rules
# Based on Jack 2018 (NIA-AA) and Petersen 2016 (MCI natural history)
# ---------------------------------------------------------------------------

# Transitions that are biologically implausible in AD
IMPLAUSIBLE_TRANSITIONS: set[tuple[str, str]] = {
    ("Dementia", "CN"),   # impossible without intervention
    ("Dementia", "MCI"),  # extreme reversal — functionally implausible
}

# Transitions that are uncommon but not impossible (reversion or rapid jump)
UNEXPECTED_TRANSITIONS: set[tuple[str, str]] = {
    ("MCI", "CN"),         # reversion possible but flag for review
}

# CN → Dementia without prior MCI is suspicious if time gap is short
CN_TO_DEMENTIA_MIN_YEARS: float = 2.0
