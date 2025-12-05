# Synthetic Data Validation Pipeline

Pipeline completa per la validazione di dati sintetici longitudinali, sviluppata nell'ambito del progetto AIND per la validazione di dati clinici simulati relativi alla progressione dell'Alzheimer (dataset ADNIMERGE).

---

## Indice

1. [Introduzione](#introduzione)
2. [Background Teorico](#background-teorico)
3. [Architettura della Pipeline](#architettura-della-pipeline)
4. [Descrizione dei Moduli](#descrizione-dei-moduli)
5. [Installazione e Requisiti](#installazione-e-requisiti)
6. [Guida all'Uso](#guida-alluso)
7. [Interpretazione dei Risultati](#interpretazione-dei-risultati)
8. [Output Generati](#output-generati)
9. [Troubleshooting](#troubleshooting)

---

## Introduzione

La validazione di dati sintetici e un passaggio critico per garantire che i dati generati siano:

- **Fedeli** ai dati reali (stesso profilo statistico)
- **Utili** per task di machine learning (utility preservata)
- **Privati** (nessuna fuga di informazioni sensibili)
- **Realistici temporalmente** (progressione della malattia catturata)

Questa pipeline implementa una validazione su **tre dimensioni principali**:

| Dimensione | Cosa Valida | Test Implementati |
|------------|-------------|-------------------|
| **Fidelity** | Somiglianza statistica | KS Test, Correlazioni |
| **Utility** | Usefulness per ML | TSTR (Train Synthetic, Test Real) |
| **Privacy** | Protezione dati | Maximum Similarity Test |
| **Temporal** | Dinamiche longitudinali | Stationarity (ADF/KPSS), ACF/PACF |

---

## Background Teorico

### 1. Kolmogorov-Smirnov (KS) Test

Il test KS confronta le distribuzioni cumulative (CDF) tra dati reali e sintetici.

**Formula:**
```
D = max|F_real(x) - F_synth(x)|
```

**Interpretazione:**
- `p-value > 0.05`: Le distribuzioni sono statisticamente simili (PASS)
- `p-value <= 0.05`: Le distribuzioni differiscono significativamente (FAIL)

**Perche e importante:**
Se i dati sintetici hanno distribuzioni diverse, qualsiasi analisi statistica sara distorta.

---

### 2. Maximum Similarity Test (Privacy)

Questo test verifica che i dati sintetici non siano "copie" dei dati reali.

**Metriche calcolate:**
- **MISS_real**: Maximum Intra-Set Similarity (quanto sono simili i record reali tra loro)
- **MISS_synth**: Maximum Intra-Set Similarity per dati sintetici
- **MCSS**: Maximum Cross-Set Similarity (quanto i record sintetici sono simili ai reali)

**Quality Score:**
```
Q = mean(MCSS) / mean(MISS_real)
```

**Interpretazione:**
- `Q <= 1.0`: PASS - I dati sintetici non sono piu simili ai reali di quanto i reali lo siano tra loro
- `Q > 1.0`: FAIL - Possibile overfitting/privacy leak

**Distanza di Gower:**
Utilizzata per gestire dati misti (numerici + categorici):
- Variabili numeriche: `sim = 1 - |diff| / range`
- Variabili categoriche: `sim = 1 se uguali, 0 altrimenti`

---

### 3. TSTR (Train Synthetic, Test Real)

Questo test valuta l'**utility** dei dati sintetici per task di machine learning.

**Metodologia:**
1. **TRTR** (baseline): Train su dati reali, Test su dati reali
2. **TSTR**: Train su dati sintetici, Test su dati reali

**Metriche:**
- Accuracy
- F1-Score

**Criterio di successo:**
```
Accuracy_degradation = TRTR_accuracy - TSTR_accuracy < 10%
```

Se la degradazione e < 10%, i dati sintetici sono utili per addestrare modelli.

---

### 4. Stationarity Tests (ADF e KPSS)

Per dati longitudinali sulla progressione dell'Alzheimer, e **critico** verificare che i dati sintetici catturino la progressione della malattia.

#### ADF (Augmented Dickey-Fuller)

**Ipotesi nulla (H0):** La serie e NON-stazionaria (ha un trend)

**Interpretazione:**
- `p-value < 0.05`: Rifiuto H0 -> Serie STAZIONARIA (nessun trend)
- `p-value >= 0.05`: Non rifiuto H0 -> Serie NON-STAZIONARIA (ha un trend)

#### KPSS (Kwiatkowski-Phillips-Schmidt-Shin)

**Ipotesi nulla (H0):** La serie e STAZIONARIA

**Interpretazione:**
- `p-value >= 0.05`: Non rifiuto H0 -> Serie STAZIONARIA
- `p-value < 0.05`: Rifiuto H0 -> Serie NON-STAZIONARIA

**CRITICO per l'Alzheimer:**
- I dati **reali** devono essere NON-stazionari (la malattia progredisce)
- Se i dati **sintetici** sono stazionari ma i reali no -> **MODEL FAILURE**
- Il modello sta generando "pazienti congelati" che non peggiorano

---

### 5. ACF/PACF (Autocorrelation Analysis)

Questi test verificano che la struttura di autocorrelazione temporale sia preservata.

#### ACF (Autocorrelation Function)

Misura la correlazione tra una serie temporale e i suoi valori ritardati (lagged).

```
ACF(k) = Corr(X_t, X_{t-k})
```

#### PACF (Partial Autocorrelation Function)

Misura la correlazione parziale, controllando per i lag intermedi.

```
PACF(k) = Corr(X_t, X_{t-k} | X_{t-1}, ..., X_{t-k+1})
```

#### Ordine AR/MA

- **AR(p)**: Modello autoregressivo di ordine p (dipende dai p valori precedenti)
- **MA(q)**: Modello a media mobile di ordine q (dipende dagli q errori precedenti)

**Importanza:**
Se i dati reali seguono un AR(1) ma i sintetici un AR(3), il modello sta overcomplicando la struttura temporale.

---

## Architettura della Pipeline

```
validate_with_preprocessing.py (Entry Point)
           |
           v
    +------+------+
    |             |
    v             v
preprocess_   pre_validation_
synthetic.py  check.py
    |             |
    v             v
  Align       Schema Check
  Data        Size Check
              Types Check
                  |
                  v
           validate_data.py
                  |
    +-------------+-------------+
    |             |             |
    v             v             v
 Fidelity     Utility       Privacy
 (KS Test)    (TSTR)        (Similarity)
    |             |             |
    v             v             v
stationarity_  acf_pacf_    Gower
tests.py       tests.py     Similarity
```

---

## Descrizione dei Moduli

### `validate_with_preprocessing.py`

**Script principale** che orchestra l'intera pipeline.

**Funzionalita:**
1. Carica dati dal datalake
2. Esegue preprocessing e allineamento
3. Esegue pre-validation check
4. Lancia la validazione completa

**Modalita disponibili:**

| Modalita | Flag | Descrizione |
|----------|------|-------------|
| Standard | (default) | Full preprocessing + Full validation |
| No-Align | `--skip-align` | No preprocessing, raw data validation |
| Age-Matching | `--age-matching` | Filtra per eta, solo fidelity tests |
| Utility-Only | `--utility-only` | Solo test TSTR |

---

### `preprocess_synthetic.py`

Gestisce l'allineamento tra dati reali e sintetici.

**Operazioni:**
1. Rimuove prefisso `generated_` dai nomi colonna
2. Rimuove colonna `generation_id`
3. Allinea le colonne tra i due dataset
4. Riordina per consistenza

**Funzioni principali:**
- `align_synthetic_with_real()`: Allineamento completo con logging
- `quick_align()`: Wrapper con impostazioni predefinite

---

### `pre_validation_check.py`

Esegue controlli preliminari **prima** della validazione completa.

**Check eseguiti:**

| Check | Cosa Verifica | Critico? |
|-------|---------------|----------|
| Schema | Colonne corrispondono | SI |
| Sample Size | Ratio 0.3x - 3.0x | SI se < 0.3x |
| Longitudinal | Struttura paziente/visite | WARNING |
| Target Distribution | Distribuzione classi | WARNING |
| Missing Data | Pattern di missing | WARNING |
| Value Ranges | Valori plausibili | WARNING |
| Data Types | Tipi consistenti | WARNING |

---

### `validate_data.py`

**Cuore della validazione** - implementa tutti i test principali.

**Classe `SyntheticDataValidator`:**

```python
validator = SyntheticDataValidator(
    real_data=real_df,
    synthetic_data=synth_df,
    categorical_features=['GENDER', 'APOE4'],
    id_var='ID',
    target_var='DX',
    time_var='TIME'
)

results = validator.run_full_validation()
```

**Metodi principali:**
- `run_full_validation()`: Esegue tutti i test
- `run_fidelity_only_validation()`: Solo KS + correlazioni
- `run_utility_only_validation()`: Solo TSTR
- `generate_visualizations()`: Genera tutti i plot
- `save_report()`: Salva report JSON

---

### `stationarity_tests.py`

Implementa test di stazionarieta per validazione temporale.

**Funzioni principali:**

```python
# Test per singolo paziente
adf_results, kpss_results = test_stationarity_per_patient(
    data, variable='MMSE', id_var='ID', time_var='TIME'
)

# Test aggregato (media tra pazienti)
adf_agg, kpss_agg = test_stationarity_aggregated(
    data, variable='MMSE', id_var='ID', time_var='TIME'
)

# Confronto tra real e synthetic
comparison = compare_stationarity(real_adf, synth_adf, test_name='ADF')
```

---

### `acf_pacf_tests.py`

Implementa analisi ACF/PACF per validazione autocorrelazione.

**Funzioni principali:**

```python
# Calcola ACF/PACF per paziente
acf_results, pacf_results = compute_acf_pacf_per_patient(
    data, variable='MMSE', id_var='ID', time_var='TIME'
)

# Rileva ordine AR/MA
orders = detect_ar_ma_order_per_patient(
    data, variable='MMSE', id_var='ID', time_var='TIME'
)

# Confronta struttura temporale
comparison = compare_acf_pacf(
    real_acf, synth_acf, real_pacf, synth_pacf
)
```

---

## Installazione e Requisiti

### Dipendenze

```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn statsmodels
```

### Dipendenze opzionali

```bash
# Per caricamento da datalake (interno IFAB)
pip install dl_client
```

### File richiesti

```
validation/
├── validate_with_preprocessing.py  # Entry point
├── validate_data.py                 # Core validation
├── preprocess_synthetic.py          # Preprocessing
├── pre_validation_check.py          # Pre-checks
├── stationarity_tests.py            # ADF/KPSS tests
├── acf_pacf_tests.py                # Autocorrelation tests
└── .dl_client.ini                   # Config datalake (opzionale)
```

---

## Guida all'Uso

### Uso Base (Modalita Standard)

```bash
cd validation
python validate_with_preprocessing.py
```

Questo esegue:
1. Caricamento dati dal datalake
2. Preprocessing/allineamento
3. Pre-validation check
4. Validazione completa (fidelity + utility + privacy + temporal)

### Modalita Disponibili

#### 1. Skip Alignment

Se i dati sono gia allineati:

```bash
python validate_with_preprocessing.py --skip-align
```

#### 2. Age Matching Mode

Filtra i dati sintetici per range di eta simile ai reali, utile per confronti piu equi:

```bash
python validate_with_preprocessing.py --age-matching
python validate_with_preprocessing.py --age-matching --age-column AGE
```

**Questo mode:**
- Filtra dati sintetici per eta +-5% del range reale
- Bilancia i dataset se differenza > 15%
- Esegue SOLO test di fidelity (KS + correlazioni)

#### 3. Utility Only Mode

Solo test TSTR:

```bash
python validate_with_preprocessing.py --utility-only --target DX
```

#### 4. Dataset Separati per Utility

```bash
# Con dataset sintetico separato per utility
python validate_with_preprocessing.py --utility-synthetic ADNIMERGEsynthetic_v2 --target DX

# Con dataset reale separato per utility (test set)
python validate_with_preprocessing.py --utility-real ADNIMERGE_test --target DX

# Entrambi separati
python validate_with_preprocessing.py \
    --utility-synthetic ADNIMERGEsynthetic_v2 \
    --utility-real ADNIMERGE_test \
    --target DX
```

### Configurazione

Modificare le costanti in `validate_with_preprocessing.py`:

```python
# Datalake settings
FILE_CODE = 'ADNIMERGE'      # Codice file reale
LEVEL = 'cleaned_03'          # Livello nel datalake

# Dataset per utility (opzionali)
UTILITY_SYNTHETIC_FILE_CODE = 'ADNIMERGE_synthetic_utility_reversed'
UTILITY_REAL_FILE_CODE = 'ADNIMERGE_to_test_utility_reversed'

# Colonne
ID_VAR = 'ID'                 # Colonna ID paziente
TIME_VAR = 'TIME'             # Colonna tempo
TARGET_VAR = 'DX'             # Target per TSTR

# Categoriche
CATEGORICAL_FEATURES = []     # Lista feature categoriche

# Output
PRECHECK_REPORT = 'pre_validation_report_aligned.txt'
VALIDATION_REPORT = 'validation_report.json'
PLOTS_DIR = 'validation_results'
```

---

## Interpretazione dei Risultati

### Report Pre-Validation

File: `pre_validation_report_aligned.txt`

```
================================================================================
PRE-VALIDATION CHECK SUMMARY
================================================================================

  Total checks performed: 7
  ✅ Passed: 5
  ⚠️  Warnings: 2
  ❌ Critical issues: 0

  ✅ PASSED CHECKS (5):
     • Schema: columns match
     • Sample size: ratio 0.95 is appropriate
     • ...

  ⚠️  WARNINGS (2):
     • Patient count differs by 12.3% (>10%)
     • 3 columns have type mismatches
```

**Azione:** Se ci sono critical issues (❌), risolverli prima di procedere.

---

### Report Validation JSON

File: `validation_report.json`

#### Sezione TSTR (Utility)

```json
{
  "tstr": {
    "tstr_accuracy": 0.6788,        // Accuracy con train synthetic
    "tstr_f1": 0.6527,              // F1 con train synthetic
    "trtr_accuracy": 0.7255,        // Accuracy baseline (train real)
    "trtr_f1": 0.6545,              // F1 baseline
    "accuracy_degradation": 0.0467, // Degradazione (< 10% = OK)
    "passed": true
  }
}
```

**Interpretazione:**
- `accuracy_degradation < 0.10` (10%): I dati sintetici sono utili per ML
- `passed: true`: Test superato

#### Sezione KS Tests (Fidelity)

```json
{
  "ks_tests": {
    "MMSE": {
      "statistic": 0.0234,
      "p_value": 0.4521,
      "passed": true
    },
    "ADAS13": {
      "statistic": 0.0567,
      "p_value": 0.0023,
      "passed": false
    }
  }
}
```

**Interpretazione:**
- `p_value > 0.05`: Distribuzioni simili (PASS)
- `p_value <= 0.05`: Distribuzioni diverse (FAIL) - investigare quella variabile

#### Sezione Similarity (Privacy)

```json
{
  "similarity": {
    "quality_score": 0.87,
    "passed_privacy": true,
    "mean_miss_real": 0.92,
    "mean_mcss": 0.80
  }
}
```

**Interpretazione:**
- `quality_score <= 1.0`: Privacy preservata
- `quality_score > 1.0`: Possibile overfitting/memory leak

---

### Report Stationarity

File: `validation_results/stationarity/stationarity_*_summary.txt`

**Sezione critica:**

```
OVERALL CONCLUSION
================================================================================
STATUS: PASSED

What this means in simple terms:
--------------------------------------------------------------------------------
  - 78.5% of REAL patients show disease progression over time
  - 75.2% of SYNTHETIC patients show disease progression
  - Difference of 3.3% is ACCEPTABLE (threshold: < 20%)

  SUCCESS: The synthetic data is NOT generating 'frozen' patients.
```

**Red Flag - MODEL FAILURE:**

```
STATUS: CRITICAL FAILURE

  - Real data: 78.5% show progression
  - Synthetic: 15.3% show progression
  - The synthetic model is generating 'frozen' patients!
```

---

### Report ACF/PACF

File: `validation_results/acf_pacf/ar_ma_orders_*_summary.txt`

```
AR Order (Autoregressive):
  Most common in Real data:      AR(1)
  Most common in Synthetic data: AR(1)
  Match: YES

MA Order (Moving Average):
  Most common in Real data:      MA(1)
  Most common in Synthetic data: MA(2)
  Match: NO
```

**Interpretazione:**
- AR match: La struttura autoregressiva e corretta
- MA mismatch: Possibile overcomplication del modello (investigare)

---

## Output Generati

### Struttura Directory

```
validation/
├── pre_validation_report_aligned.txt
├── validation_report.json
└── validation_results/
    ├── acf_pacf/
    │   ├── ar_ma_orders_MMSE_summary.txt
    │   ├── ar_ma_orders_ADAS11_summary.txt
    │   └── ...
    ├── stationarity/
    │   ├── stationarity_MMSE.png
    │   ├── stationarity_MMSE_summary.txt
    │   ├── stationarity_ADAS11.png
    │   └── ...
    ├── ks_tests/
    │   ├── ks_test_MMSE.png
    │   ├── ks_test_ADAS11.png
    │   └── ...
    ├── correlation/
    │   └── correlation_comparison.png
    ├── similarity/
    │   └── maximum_similarity_test.png
    └── utility/
        └── tstr_summary.txt
```

### Descrizione Plot

#### `ks_tests/ks_test_*.png`

3 pannelli per ogni variabile:
1. **Histograms + KDE**: Confronto distribuzioni
2. **Empirical CDFs**: Funzioni cumulative con punto di max differenza
3. **CDF Difference**: Visualizzazione della statistica KS

#### `correlation/correlation_comparison.png`

3 heatmap:
1. **Real Data Correlation**: Matrice correlazione dati reali
2. **Synthetic Data Correlation**: Matrice correlazione dati sintetici
3. **Absolute Difference**: Differenze assolute (rosso = grande differenza)

#### `similarity/maximum_similarity_test.png`

2 pannelli:
1. **Histogram**: Distribuzioni MISS_real, MISS_synth, MCSS
2. **Boxplot**: Confronto quartili

#### `stationarity/stationarity_*.png`

4 pannelli:
1. **ADF p-values**: Distribuzione p-value test ADF
2. **KPSS p-values**: Distribuzione p-value test KPSS
3. **ADF Stationarity**: % stazionari vs non-stazionari
4. **KPSS Stationarity**: % stazionari vs non-stazionari

---

## Troubleshooting

### Errore: "dl_client not found"

**Causa:** Il modulo `dl_client` non e installato (necessario per datalake IFAB).

**Soluzione:** Installare il modulo o usare file CSV locali:

```python
# In pre_validation_check.py o validate_data.py
real_data = pd.read_csv('real_data.csv')
synthetic_data = pd.read_csv('synthetic_data.csv')
```

---

### Errore: "Missing columns in synthetic data"

**Causa:** I dati sintetici hanno colonne diverse dai reali.

**Soluzione:**
1. Verificare che il preprocessing sia stato eseguito
2. Controllare che il prefisso `generated_` sia stato rimosso
3. Usare `--skip-align` se i dati sono gia allineati

---

### Warning: "Synthetic dataset too small"

**Causa:** Il ratio synth/real e < 0.3.

**Soluzione:**
- Generare piu dati sintetici
- Usare `--age-matching` per bilanciare i dataset
- Investigare perche ci sono pochi dati sintetici

---

### Stationarity: "CRITICAL FAILURE - frozen patients"

**Causa:** Il modello generativo non sta catturando la progressione temporale.

**Soluzione:**
1. Verificare che il modello usi correttamente la variabile tempo
2. Controllare i parametri del modello generativo
3. Verificare che i dati di training abbiano sufficiente variabilita temporale

---

### TSTR: "accuracy_degradation > 10%"

**Causa:** I dati sintetici non sono sufficientemente utili per ML.

**Soluzione:**
1. Verificare fidelity (KS tests) - le distribuzioni corrispondono?
2. Verificare correlazioni - le relazioni tra variabili sono preservate?
3. Potrebbe essere necessario rifinire il modello generativo

---

## Riferimenti

- **Gower Distance**: Gower, J.C. (1971). "A general coefficient of similarity and some of its properties"
- **KS Test**: Kolmogorov, A. (1933). "Sulla determinazione empirica di una legge di distribuzione"
- **ADF Test**: Dickey, D.A., Fuller, W.A. (1979). "Distribution of the estimators for autoregressive time series with a unit root"
- **KPSS Test**: Kwiatkowski, D., et al. (1992). "Testing the null hypothesis of stationarity against the alternative of a unit root"
- **TSTR Framework**: Esteban, C., et al. (2017). "Real-valued (medical) time series generation with recurrent conditional GANs"

---

## Autori

- **Raimondo Reggio** - Sviluppo e manutenzione
