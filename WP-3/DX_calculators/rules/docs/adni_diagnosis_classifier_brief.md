# ADNI-style Diagnosis Classifier — Build Brief

> Handoff spec for Claude Code. Goal: implement two classifiers that assign a
> CN / MCI / AD label to **synthetic, ADNI-like patients**, plus a separate
> biological (ATN) qualifier. This file is the single source of truth for the
> decisions and cutoffs agreed so far. Read it fully before writing code.

---

## 0. Project objective

We have generated **synthetic patients** with ADNI-like features. We want to label
them with a diagnosis (CN / MCI / AD) using **two independent methods**, then
compare them:

1. **Rule-based classifier** — reproduces the ADNI clinical screening/diagnosis
   algorithm (which is, in spirit, the NIA-AA 2011 *clinical* syndromic logic).
2. **ML classifier** — trained on **real ADNI ground-truth labels** (from `DXSUM`
   + `ADNIMERGE`, normalized to CN / MCI / AD).

Both classifiers must output on the **same syndromic axis** (CN/MCI/AD) so the
comparison is apples-to-apples.

A **third, separate output** is the **ATN biological qualifier** (Amyloid / Tau /
Neurodegeneration), computed from biomarkers. This does **not** replace CN/MCI/AD;
it enriches it (e.g. "MCI, A+T+" vs "MCI, A−").

---

## 1. Key design decisions (already made — do not re-litigate)

- **Two axes, kept separate.**
  - *Syndromic axis*: CN / MCI / AD. This is what both classifiers predict and
    what we compare.
  - *Biological axis*: A / T / N status. Computed in parallel, never used to
    override the syndromic label.
- **Rule-based target = ADNI clinical algorithm**, not the biological NIA-AA
  2018/2024 framework. Reason: the ML is trained on ADNI *clinical* labels, so the
  rule-based must speak the same language to be comparable. (NIA-AA 2011 *clinical*
  criteria and the ADNI algorithm are effectively the same family; NIA-AA
  2018/2024 are biological and live on the ATN axis instead.)
- **Ground truth = `DXSUM` + `ADNIMERGE`**, normalized to 3 classes (CN/MCI/AD).
  Use the **current clinical diagnosis at each visit**, *not* the conversion/
  reversion transition coding. SMC/EMCI/LMCI are collapsed: CN+SMC → CN,
  EMCI+LMCI → MCI.
- **Cutoffs are NOT constant across ADNI phases.** The Logical Memory II cutoff for
  MCI in particular changes a lot between ADNI1 / GO-2 / 3-4. Two valid strategies:
  - *Protocol-aware* (for real ADNI rows): apply the cutoffs of the subject's phase
    via `ORIGPROT` / `COLPROT`.
  - *Single reference* (for synthetic patients, which have no phase): **use the
    ADNI3 cutoff set** (3 categories, inclusive MCI band). This is the default for
    the synthetic cohort.
- **Biomarker cutoffs must be calibrated to the synthetic generator's scale.** Do
  NOT hardcode ADNI assay thresholds blindly — they depend on assay/pipeline. Put
  them in a config dict, with literature values as placeholders to be confirmed.
- **Irreducible ceiling.** ADNI labels are *clinician judgment*, not a deterministic
  formula. A perfect rule-based classifier will still NOT match ADNI labels 100%.
  This is expected; do not "tune the rules" to chase 100% agreement.

---

## 2. Rule-based logic — syndromic axis (CN / MCI / AD)

### 2.1 Variables consumed (core diagnostic set)

| Concept | Synthetic/ADNI variable (confirm names in our data) | Use |
|---|---|---|
| CDR global | `CDGLOBAL` | Primary anchor: 0 / 0.5 / ≥1 |
| CDR memory box | `CDMEMORY` | Must be ≥ 0.5 to qualify MCI |
| CDR sum of boxes | `CDRSB` (a.k.a. `CDRSOB`) | Severity feature (optional in rules) |
| MMSE total | `MMSE` (ADNIMERGE) / `MMSCORE` (raw) | Range gate |
| Logical Memory II delayed | `LDELTOTAL` (immediate: `LIMMTOTAL`) | Episodic memory; education-adjusted cutoff |
| Functional Activities Questionnaire | `FAQ` (ADNIMERGE) / `FAQTOTAL` (raw) | MCI vs dementia (independence) |
| Geriatric Depression Scale (15-item) | `GDTOTAL` | Exclusion gate (< 6) |
| Modified Hachinski | `HMSCORE` | Exclusion gate (≤ 4) |
| Years of education | `PTEDUCAT` | Drives LM cutoff bands |
| Age | `AGE` | Eligibility (55–90), adjustment |

### 2.2 Education-adjusted Logical Memory II (LDELTOTAL) cutoffs, by phase

Education bands: **16+ years / 8–15 years / 0–7 years**. Max score = 25.

| Category | ADNI1 | ADNI-GO / ADNI2 | ADNI3 / ADNI4 (← use for synthetic) |
|---|---|---|---|
| **CN (normal memory)** | ≥9 / ≥5 / ≥3 | ≥9 / ≥5 / ≥3 | ≥9 / ≥5 / ≥3 |
| **MCI (impaired memory)** | ≤8 / ≤4 / ≤2 | EMCI: 9–11 / 5–9 / 3–6 (band); LMCI: ≤8 / ≤4 / ≤2 | <11 / ≤9 / ≤6 (i.e. ≤10 / ≤9 / ≤6) |
| **AD (impaired memory)** | ≤8 / ≤4 / ≤2 | ≤8 / ≤4 / ≤2 | ≤8 / ≤4 / ≤2 |

> Note: ADNI4 mirrors ADNI3 (verify against ADNI4 in-clinic protocol if needed).

### 2.3 MMSE range gate, by phase

| Category | ADNI1 / GO / 2 | ADNI3 / 4 |
|---|---|---|
| CN / MCI | 24–30 | 24–30 |
| AD | 20–26 | 20–24 (exceptions 24–25 for <8 yrs education) |

### 2.4 CDR mapping (all phases)

| Category | CDR global | Memory box |
|---|---|---|
| CN | 0 | 0 |
| MCI | 0.5 | ≥ 0.5 |
| AD | 0.5 or 1.0 | n/a |

### 2.5 Eligibility gates (all categories)

- `GDTOTAL` < 6  (else: depression confound)
- `HMSCORE` ≤ 4  (else: vascular etiology)
- Age 55–90
- For synthetic data these can be applied as flags rather than hard exclusions —
  decide per use case.

### 2.6 Decision hierarchy (ADNI3 reference — default for synthetic)

Apply in order; the first matching branch wins. Overlaps MUST have a deterministic
tie-break (documented below).

```
INPUTS: CDGLOBAL, CDMEMORY, MMSE, LDELTOTAL, FAQ, PTEDUCAT, (GDTOTAL, HMSCORE)

# 1. Compute education-adjusted memory status from LDELTOTAL
mem_status = memory_status(LDELTOTAL, PTEDUCAT, phase="ADNI3")
#   -> "normal"  if LDELTOTAL >= CN cutoff
#   -> "mci"     if within MCI band (<= MCI cutoff and > AD cutoff)
#   -> "ad"      if LDELTOTAL <= AD cutoff

# 2. Functional impairment flag (independence lost -> points to dementia)
func_impaired = FAQ >= FAQ_DEMENTIA_THRESHOLD   # NO official cutoff; calibratable

# 3. Branches
IF CDGLOBAL == 0 AND mem_status == "normal" AND 24 <= MMSE <= 30:
    -> "CN"

ELIF CDGLOBAL == 0.5 AND CDMEMORY >= 0.5 AND mem_status in {"mci","ad"} \
     AND 24 <= MMSE <= 30 AND not func_impaired:
    -> "MCI"

ELIF CDGLOBAL >= 0.5 AND (mem_status == "ad" OR 20 <= MMSE <= 24) \
     AND func_impaired:
    -> "AD"

ELSE:
    -> tie_break(...)   # see 2.7
```

### 2.7 Tie-break / ambiguity rules (document the choices)

The ADNI3 ranges overlap by design (CN and MCI both allow MMSE 24–30; MCI and AD
both allow CDR 0.5). The separators are:

- **CN vs MCI**: CDR (0 vs 0.5) + memory (normal vs impaired). If conflict, prefer
  CDR as the anchor (ADNI documentation states CDR and DX are fundamentally
  dependent).
- **MCI vs AD**: **functional independence** (FAQ) + meeting dementia criteria. A
  subject with impaired memory but preserved independence is MCI; loss of
  independence + dementia-level cognition is AD. When `func_impaired` is true and
  cognition is in AD range → AD; otherwise → MCI.
- Emit a `dx_rule_flag` column noting any case that fell through to tie-break, so we
  can audit disagreement with the ML later.

### 2.8 Proposed function signatures

```python
def memory_status(ldeltotal: float, educ_years: float,
                  phase: str = "ADNI3") -> str: ...
    # returns "normal" | "mci" | "ad"

def classify_syndromic(row: pd.Series,
                       phase: str = "ADNI3",
                       faq_dementia_threshold: float = ...) -> str: ...
    # returns "CN" | "MCI" | "AD"  (+ optional flag)

def apply_syndromic(df: pd.DataFrame, **kwargs) -> pd.DataFrame: ...
    # adds columns: dx_rule, dx_rule_flag
```

---

## 3. ATN qualifier — biological axis (separate output)

Computed independently from biomarkers. Output is A/T/N booleans + a derived label.
**All cutoffs go in a config dict and MUST be calibrated to our synthetic scale.**

| Axis | Concept | Candidate variable(s) | Placeholder cutoff (CONFIRM) |
|---|---|---|---|
| **A** (amyloid) | CSF Aβ42 or Aβ42/40, or amyloid PET | `ABETA`; `AV45` (florbetapir SUVR), `FBB` (florbetaben) | CSF Aβ42 low (e.g. <192 INNO-BIA, or <~980 Elecsys); PET SUVR >~1.11 / Centiloid >~20–25 |
| **T** (tau) | CSF p-tau181, or tau PET | `PTAU`; tau PET (flortaucipir SUVR) | p-tau high (assay-dependent); tau PET SUVR above region threshold |
| **N** (neurodegeneration) | Hippocampal/entorhinal volume, FDG-PET, or t-tau | `Hippocampus`, `Entorhinal`, `WholeBrain`, `Ventricles`; `FDG`; `TAU` (t-tau) | volume below age-norm; FDG hypometabolism; t-tau high |

Derived label (NIA-AA 2018 logic):
- `A−` → not Alzheimer's continuum.
- `A+` → "Alzheimer's pathologic change".
- `A+ T+` → "Alzheimer's disease" (regardless of N).
- N stages severity.

```python
def classify_atn(row: pd.Series, cutoffs: dict) -> dict: ...
    # returns {"A": bool, "T": bool, "N": bool, "atn_label": str}
```

---

## 4. Output schema

Final per-patient DataFrame:

| column | source | notes |
|---|---|---|
| `subject_id` | synthetic | key |
| `dx_rule` | rule-based (§2) | CN / MCI / AD |
| `dx_rule_flag` | rule-based | tie-break / gate flags |
| `dx_ml` | ML model (later) | CN / MCI / AD |
| `A`, `T`, `N` | ATN (§3) | bool |
| `atn_label` | ATN (§3) | derived |

Keep `dx_rule` and `dx_ml` strictly comparable (same labels, same axis).

---

## 5. Implementation notes

- **Stack**: Python + pandas. Vectorize where possible; keep `classify_syndromic`
  pure/row-wise first for clarity, then optionally vectorize.
- **Config-driven cutoffs**: put all phase cutoffs and biomarker thresholds in a
  single `config.py` / dict, so calibration is a one-file change.
- **Phase handling**: `classify_syndromic(phase=...)` should support "ADNI1",
  "ADNIGO2", "ADNI3". Default "ADNI3" for synthetic. For real ADNI rows, derive
  phase from `ORIGPROT`/`COLPROT`.
- **GDS is the 15-item version in ADNI** (max 15), cutoff < 6. Do not assume the
  30-item scale.
- **Education bands** are inclusive: 0–7, 8–15, 16+ years.

---

## 6. TODO — confirm BEFORE coding

1. **Synthetic data dictionary**: exact column names + numeric scales for every
   variable in §2.1 and §3. (Variable names above are ADNI conventions; map them to
   our synthetic columns.)
2. **Biomarker scale**: which assay/units the synthetic generator emulates →
   determines A/T/N cutoffs. Set them in config.
3. **FAQ functional-impairment threshold**: no official ADNI cutoff. Choose and
   document (literature commonly treats FAQ ≥ 6–9, or item-level dependence, as
   functional impairment). This is the main MCI↔AD lever.
4. **Protocol strategy for synthetic**: confirm ADNI3 reference is acceptable
   (recommended) vs running all three phase variants for sensitivity analysis.

---

## 7. Caveats to keep in mind

- The MCI class is **heterogeneous**: it merges EMCI-origin (milder, ~1.0 SD,
  ~45% amyloid-positive) and LMCI/ADNI1-origin (~1.5 SD, ~63% amyloid-positive).
  Clinically equivalent, biologically not. Relevant if we later stratify.
- Do **not** interpret rule-vs-ML disagreement as pure error — part of it is the
  clinician-judgment ceiling (§1) and part is genuine biological signal surfaced by
  the ATN axis.
- Variable name strings (`GDTOTAL`, `HMSCORE`, `CDGLOBAL`, raw vs ADNIMERGE) should
  be re-checked against the actual data dictionary of whatever table we load.
