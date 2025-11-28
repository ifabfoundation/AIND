# Test di Stazionarietà per Validazione Dati Sintetici

## Obiettivo

I test di stazionarietà sono stati implementati per validare che i dati sintetici catturino correttamente la **progressione della malattia** presente nei dati reali.

### Scenario Critico da Rilevare

**Se i dati originali sono non stazionari (progressione della malattia) ma quelli generati sono stazionari (pazienti "statici"), il modello ha FALLITO nel catturare la progressione della malattia.**

In altre parole:
- ✅ **Dati reali NON stazionari** → i pazienti peggiorano nel tempo (progressione della malattia)
- ✅ **Dati sintetici NON stazionari** → il modello cattura la progressione
- ❌ **Dati reali NON stazionari, ma sintetici STAZIONARI** → il modello genera pazienti "congelati" invece di pazienti che peggiorano

## Test Implementati

### 1. ADF (Augmented Dickey-Fuller)

**Ipotesi nulla (H0):** La serie temporale è **non stazionaria** (ha una radice unitaria)

**Interpretazione:**
- `p-value < 0.05` → Rifiutiamo H0 → Serie **STAZIONARIA**
- `p-value >= 0.05` → Non possiamo rifiutare H0 → Serie **NON STAZIONARIA**

### 2. KPSS (Kwiatkowski-Phillips-Schmidt-Shin)

**Ipotesi nulla (H0):** La serie temporale è **stazionaria**

**Interpretazione (opposta ad ADF!):**
- `p-value < 0.05` → Rifiutiamo H0 → Serie **NON STAZIONARIA**
- `p-value >= 0.05` → Non possiamo rifiutare H0 → Serie **STAZIONARIA**

### Perché Entrambi i Test?

ADF e KPSS sono complementari:
- **ADF** è più potente per rilevare non-stazionarietà con trend
- **KPSS** è più conservativo e rileva la stazionarietà
- Usarli insieme fornisce una validazione più robusta

## Struttura dei File

```
validation/
├── stationarity_tests.py              # Modulo con funzioni per ADF e KPSS
├── validate_data.py                   # Classe SyntheticDataValidator (modificata)
├── validate_with_preprocessing.py     # Pipeline principale (modificata)
└── README_STATIONARITY_TESTS.md       # Questa documentazione
```

## Come Usare

### 1. Eseguire la Pipeline Completa

```bash
cd validation
python validate_with_preprocessing.py
```

La pipeline esegue automaticamente:
1. Caricamento dati dal datalake
2. Pre-processing e allineamento
3. Pre-validation check
4. **Test di stazionarietà per tutte le variabili numeriche**
5. Altri test di validazione (fidelity, utility, privacy)

### 2. Output Generati

#### File di Testo Riassuntivi

Per ogni variabile testata (es. MMSE, ADAS11), viene generato:

```
validation_plots/
├── stationarity_MMSE_summary.txt      # Riepilogo dettagliato per MMSE
├── stationarity_ADAS11_summary.txt    # Riepilogo dettagliato per ADAS11
└── ...
```

Esempio contenuto:
```
================================================================================
STATIONARITY TESTS SUMMARY
================================================================================

Objective:
--------------------------------------------------------------------------------
Validate that synthetic data captures disease progression dynamics.
CRITICAL: If real data is non-stationary (disease progression) but
synthetic data is stationary (static patients), the model has FAILED.

================================================================================
AUGMENTED DICKEY-FULLER (ADF) TEST
================================================================================
Null Hypothesis: Time series is non-stationary
Interpretation: p < 0.05 → Stationary (reject H0)

Status: PASSED

Real Data:
  Total patients tested: 250
  Stationary: 45 (18.0%)
  Non-stationary: 205 (82.0%)

Synthetic Data:
  Total patients tested: 250
  Stationary: 52 (20.8%)
  Non-stationary: 198 (79.2%)

Difference: 2.8%

Interpretation:
GOOD: Synthetic data matches real data stationarity patterns.
The model successfully captures disease progression dynamics.

...
```

#### Grafici Visuali

Per ogni variabile:

```
validation_plots/
├── stationarity_MMSE.png              # Grafico comparativo per MMSE
├── stationarity_ADAS11.png            # Grafico comparativo per ADAS11
└── ...
```

Ogni grafico contiene 4 pannelli:
1. **Distribuzione p-values ADF**: Istogramma dei p-values del test ADF
2. **Distribuzione p-values KPSS**: Istogramma dei p-values del test KPSS
3. **Confronto ADF**: Barre che mostrano % stazionari vs non-stazionari (ADF)
4. **Confronto KPSS**: Barre che mostrano % stazionari vs non-stazionari (KPSS)

#### Report JSON

Il file `validation_report.json` include una sezione `stationarity`:

```json
{
  "stationarity": {
    "MMSE": {
      "adf": {
        "real_stationary_pct": 18.0,
        "synth_stationary_pct": 20.8,
        "difference_pct": 2.8,
        "status": "PASSED",
        "model_failed": false
      },
      "kpss": {
        "real_stationary_pct": 15.2,
        "synth_stationary_pct": 17.6,
        "difference_pct": 2.4,
        "status": "PASSED",
        "model_failed": false
      }
    }
  }
}
```

## Interpretazione dei Risultati

### ✅ Test PASSED (Successo)

```
Status: PASSED

Real Data: 18% stationary, 82% non-stationary
Synthetic Data: 21% stationary, 79% non-stationary
Difference: 3%

Interpretation: Synthetic data matches real data stationarity patterns.
```

**Significato:** Il modello cattura correttamente la progressione della malattia.

### ⚠️ Test WARNING (Attenzione)

```
Status: WARNING

Real Data: 20% stationary, 80% non-stationary
Synthetic Data: 42% stationary, 58% non-stationary
Difference: 22%

Interpretation: Moderate difference in stationarity patterns.
```

**Significato:** C'è una differenza moderata. Rivedere i singoli pazienti e considerare il fine-tuning del modello.

### ❌ Test FAILED (Fallimento Critico)

```
Status: FAILED

Real Data: 15% stationary, 85% non-stationary
Synthetic Data: 65% stationary, 35% non-stationary
Difference: 50%

Interpretation: CRITICAL FAILURE: Real data shows disease progression
(non-stationary) but synthetic data is static (stationary).
The model failed to capture the temporal dynamics.
```

**Significato:** **Il modello NON sta catturando la progressione della malattia!** Sta generando pazienti statici invece di pazienti che peggiorano.

**Azione richiesta:** Rivedere l'architettura del modello, in particolare la componente temporale.

## Dettagli Tecnici

### Test Per-Patient vs Aggregato

Attualmente implementato: **Test per-patient**
- Ogni paziente viene testato individualmente
- I risultati sono aggregati per calcolare le percentuali
- Più robusto per dati longitudinali con pazientidi diversi

Potenzialmente implementabile: **Test aggregato**
- Media di tutti i pazienti per ogni time point
- Utile per trend generali di popolazione

### Requisiti Minimi

Per ogni paziente:
- **Minimo 3 osservazioni temporali** per eseguire i test
- Pazienti con meno di 3 osservazioni vengono saltati

### Variabili Testate

Automaticamente testa tutte le variabili numeriche **tranne**:
- Colonna ID (`ID_VAR`)
- Colonna temporale (`TIME_VAR`)

Tipiche variabili testate:
- MMSE (Mini-Mental State Examination)
- ADAS11, ADAS13 (Alzheimer's Disease Assessment Scale)
- CDRSB (Clinical Dementia Rating Sum of Boxes)
- Biomarkers (ABETA, TAU, PTAU)
- Volumi cerebrali (Hippocampus, Ventricles, etc.)

## Configurazione

Nel file `validate_with_preprocessing.py`:

```python
# Datalake settings
FILE_CODE = 'ADNIMERGE'  # Your real data file code
LEVEL = 'cleaned_03'     # Datalake level

# Column settings
ID_VAR = 'ID'            # ID column
TIME_VAR = 'TIME'        # Time column for stationarity tests
```

## Troubleshooting

### Problema: "Stationarity tests skipped"

**Causa:** Modulo `stationarity_tests.py` non trovato o `statsmodels` non installato

**Soluzione:**
```bash
pip install statsmodels
```

### Problema: "ID variable not found"

**Causa:** `ID_VAR` non corrisponde al nome della colonna nel dataset

**Soluzione:** Verifica il nome esatto della colonna ID nei tuoi dati

### Problema: "Time variable not found"

**Causa:** `TIME_VAR` non corrisponde al nome della colonna nel dataset

**Soluzione:** Verifica il nome esatto della colonna temporale

### Problema: "Insufficient data for comparison"

**Causa:** Troppo pochi pazienti hanno almeno 3 osservazioni temporali

**Soluzione:**
- Verifica la qualità dei dati
- Considera di ridurre il threshold minimo (richiede modifica codice)

## Riferimenti

### Documentazione Statsmodels
- [ADF Test](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.adfuller.html)
- [KPSS Test](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.kpss.html)

### Teoria
- Dickey, D. A., & Fuller, W. A. (1979). "Distribution of the estimators for autoregressive time series with a unit root"
- Kwiatkowski, D., et al. (1992). "Testing the null hypothesis of stationarity against the alternative of a unit root"

## Contatti

Per domande o problemi:
- Apri un issue nel repository
- Contatta il team IFAB

---

**Versione:** 1.0
**Data:** 2025
**Autore:** IFAB Team
