# Post-Generation: Reverse Transformations & Noise Injection

Questa cartella contiene gli strumenti per:
1. **Invertire** tutte le trasformazioni applicate ai dati durante la fase di preparazione
2. **Aggiungere rumore** ai dati sintetici per migliorare la privacy usando Differential Privacy
3. **Correggere** i dati sintetici con tecniche Residual-PAR
4. **Iniettare missing values** per simulare pattern reali di dati mancanti

---

## Script Batch: `reverse_from_datalake.bat`

Il file **`reverse_from_datalake.bat`** è lo script batch principale per orchestrare la pipeline di post-generazione su Windows. Permette di eseguire in sequenza le operazioni di reverse transformation e altre post-elaborazioni sui dati sintetici.

### Contenuto attuale del batch

```batch
@echo off
::python post-generation\noise_injection_from_datalake.py --real-code ADNIMERGE_for_validation --synthetic-code ADNIMERGE_synthetic --epsilon 0.5
::python post-generation\reverse_from_datalake.py --file-code ADNIMERGE_synthetic --prefix synthetic/reversed
::python post-generation\reverse_from_datalake.py --file-code ADNIMERGE_for_validation --prefix validation/reversed
python post-generation\reverse_from_datalake.py --file-code ADNIMERGE_synthetic_utility --prefix synthetic/utility/reversed
python post-generation\reverse_from_datalake.py --file-code ADNIMERGE_to_test_utility --prefix validation/utility/reversed
::python post-generation\residual_par_correction.py --file-code-real ADNIMERGE_for_validation_reversed --file-code-leaspy ADNIMERGE_synthetic_reversed --baseline-match-feats MMSE
::python post-generation\inject_missing_values.py --file-code-real ADNIMERGE_for_validation_reversed --file-code-gen ADNIMERGE_synthetic_reversed --prefix synthetic/missing_injected
pause
```

### Come funziona

1. **Comandi commentati (`::`)**: sono comandi opzionali o alternativi che possono essere abilitati rimuovendo `::` all'inizio della riga
2. **Comandi attivi**: attualmente esegue il reverse transformation per due dataset:
   - `ADNIMERGE_synthetic_utility` → upload su `synthetic/utility/reversed`
   - `ADNIMERGE_to_test_utility` → upload su `validation/utility/reversed`
3. **`pause`**: mantiene aperta la finestra del terminale per vedere l'output

### Pipeline completa disponibile

Il batch supporta l'intera pipeline post-generazione:

```
┌─────────────────────────────────────────────────────────────┐
│              PIPELINE POST-GENERAZIONE COMPLETA             │
└─────────────────────────────────────────────────────────────┘

[1] noise_injection_from_datalake.py
    └─ Aggiunge rumore DP ai dati normalizzati [0,1]
    └─ Input: ADNIMERGE_synthetic (normalizzato)
    └─ Output: ADNIMERGE_synthetic_noisy
           ↓
[2] reverse_from_datalake.py
    └─ Denormalizza scale cliniche e volumi
    └─ Converte dummy variables → categorical
    └─ Filtra record con TIME invalido
    └─ Input: ADNIMERGE_synthetic_noisy
    └─ Output: ADNIMERGE_synthetic_noisy_reversed
           ↓
[3] residual_par_correction.py (opzionale)
    └─ Applica correzione Residual-PAR con KNN pseudo-pairing
    └─ Migliora le distribuzioni marginali
    └─ Input: dati reali + sintetici reversed
    └─ Output: dati corretti con metriche KS/DCR
           ↓
[4] inject_missing_values.py (opzionale)
    └─ Inietta missing values secondo i pattern dei dati reali
    └─ Input: dati reali + sintetici reversed
    └─ Output: dati con missing realistici
```

### Personalizzazione del batch

Per abilitare/disabilitare comandi:
- **Abilitare**: rimuovere `::` dalla riga
- **Disabilitare**: aggiungere `::` all'inizio della riga

Esempio per eseguire la pipeline completa:
```batch
@echo off
python post-generation\noise_injection_from_datalake.py --real-code ADNIMERGE --synthetic-code ADNIMERGE_synthetic --epsilon 1.0
python post-generation\reverse_from_datalake.py --file-code ADNIMERGE_synthetic_noisy --prefix synthetic/reversed
python post-generation\inject_missing_values.py --file-code-real ADNIMERGE_reversed --file-code-gen ADNIMERGE_synthetic_noisy_reversed --prefix synthetic/final
pause
```

---

## Struttura del Package

```
post-generation/
├── __init__.py                         # Package initialization
├── reverse_from_datalake.bat           # Script batch per orchestrazione pipeline (Windows)
│
├── # === REVERSE TRANSFORMATIONS ===
├── reverse_from_datalake.py            # Entry point per reverse dal datalake
├── reverse_from_local.py               # Entry point per reverse da file locali (senza datalake)
├── reverse_transformations.py          # Entry point unificato (orchestratore)
├── denormalize.py                      # Modulo per denormalizzazione valori numerici
├── dummy_to_categorical.py             # Modulo per conversione dummy→categorical
│
├── # === NOISE INJECTION (Differential Privacy) ===
├── noise_injection.py                  # Modulo con meccanismi DP (Laplace, Gaussian, Adaptive)
├── noise_injection_from_datalake.py    # Entry point per noise injection dal datalake
│
├── # === RESIDUAL-PAR CORRECTION ===
├── residual_par_correction.py          # Correzione con KNN + PAR (no stessi ID richiesti)
├── add_noise.py                        # Funzioni per Residual-PAR e allineamento dati
├── add_noise_nb.py                     # Versione notebook con funzioni KNN pseudo-pairing
│
├── # === MISSING VALUES INJECTION ===
├── inject_missing_values.py            # Inietta missing values secondo pattern reali
│
├── # === CONFIGURATION FILES ===
├── normalization_settings.json         # Impostazioni scale cliniche (MMSE, FAQ, etc.)
├── volume_values_settings.json         # Impostazioni volumi cerebrali
├── .dl_client.ini                      # Configurazione client datalake
│
├── # === OTHER ===
├── reverse_commands.txt                # Comandi di riferimento
├── create_validation_source_dataset.ipynb  # Notebook per dataset di validazione
└── README.md                           # Questa documentazione
```

---

## Descrizione Dettagliata degli Script

### 1. `reverse_from_datalake.py` - Reverse Transformations dal Datalake

Script principale per caricare dati dal datalake, applicare reverse transformations e ricaricare i risultati.

**Funzioni principali:**
- `load_from_datalake(file_code, query_params)` - Carica dataset dal datalake tramite DatalakeClient
- `get_file_metadata(object_name)` - Recupera metadata custom dal datalake
- `upload_to_datalake(df, file_name, prefix, metadata)` - Carica DataFrame sul datalake
- `reverse_datalake_data(...)` - Pipeline completa end-to-end

**Pipeline eseguita:**
1. **STEP 1**: Carica dati dal datalake usando `file_code`
2. **STEP 1.5**: Auto-detection variabili dummy (colonne con `/` e valori 0/1)
3. **STEP 2**: Applica reverse transformations (denormalizzazione + dummy→categorical)
4. **STEP 2.5**: Filtra record con TIME invalido (fuori range [0, 100])
5. **STEP 3**: Upload sul datalake con metadata aggiornati (`_reversed` suffix)
6. **STEP 4**: (opzionale) Salva copia locale

**Uso da command line:**
```bash
# Base - upload su datalake con auto-detect
python reverse_from_datalake.py --file-code ADNIMERGE_synthetic --prefix synthetic/reversed

# Con copia locale
python reverse_from_datalake.py --file-code ADNIMERGE_synthetic --prefix synthetic/reversed --save-local --output reversed.csv

# Solo locale (no upload)
python reverse_from_datalake.py --file-code ADNIMERGE_synthetic --no-upload --save-local --output reversed.csv

# Con dummy prefixes manuali
python reverse_from_datalake.py --file-code ADNIMERGE_synthetic --dummy-prefixes SEX DX APOE4

# Skip denormalizzazione (solo dummy→categorical)
python reverse_from_datalake.py --file-code ADNIMERGE_synthetic --skip-denormalization
```

**Prefissi disponibili:** `synthetic/reversed`, `validation/reversed`, `synthetic/utility/reversed`, `validation/utility/reversed`

---

### 2. `reverse_from_local.py` - Reverse Transformations da File Locali

Versione per lavorare con file CSV locali **senza dipendenza dal datalake**. Utile per test o ambienti senza accesso al datalake.

**Funzioni principali:**
- `load_from_csv(file_path)` - Carica dataset da file CSV locale
- `reverse_local_data(...)` - Pipeline completa locale

**Pipeline eseguita:**
1. **STEP 1**: Carica dati da CSV locale
2. **STEP 1.5**: Auto-detection variabili dummy
3. **STEP 2**: Applica reverse transformations
4. **STEP 3**: Salva risultato su file CSV locale

**Uso da command line:**
```bash
# Base
python reverse_from_local.py --input-file synthetic_data.csv --output reversed_data.csv

# Con settings personalizzati
python reverse_from_local.py --input-file data.csv --output reversed.csv --normalization-settings my_settings.json

# Con dummy prefixes manuali
python reverse_from_local.py --input-file data.csv --output reversed.csv --dummy-prefixes SEX DX APOE4
```

**Uso programmatico:**
```python
from reverse_from_local import reverse_local_data

df = reverse_local_data(
    input_path='synthetic_data.csv',
    output_path='reversed_data.csv',
    normalization_settings_path='normalization_settings.json'
)
```

---

### 3. `reverse_transformations.py` - Orchestratore Unificato

Entry point che orchestra tutte le operazioni di reverse transformation. Può essere usato come libreria da altri script.

**Funzione principale:**
- `reverse_all_transformations(df, ...)` - Pipeline completa di reverse

**Operazioni in sequenza:**
1. **[1/4]** Denormalizzazione scale cliniche (MMSE, FAQ, CDRSB, etc.)
2. **[2/4]** Denormalizzazione volumi cerebrali (Hippocampus, Ventricles, etc.)
3. **[3/4]** Conversione dummy → categorical
4. **[4/4]** Decodifica label encoding (opzionale)

**Uso programmatico:**
```python
from reverse_transformations import reverse_all_transformations

dummy_configs = [
    {'prefix': 'SEX', 'separator': '/'},
    {'prefix': 'DX', 'separator': '/'},
]

df_original = reverse_all_transformations(
    df_transformed,
    normalization_settings_path='normalization_settings.json',
    volume_settings_path='volume_values_settings.json',
    dummy_configs=dummy_configs
)
```

---

### 4. `denormalize.py` - Modulo Denormalizzazione

Funzioni per riportare valori numerici dalla scala normalizzata [0,1] ai valori originali.

**Funzioni:**
- `denormalize_fixed_scale(df, columns, min_max_scales)` - Per scale cliniche con arrotondamento intelligente
- `denormalize_dataset_minmax(df, columns, min_max_values)` - Per min/max generici
- `denormalize_dataset_volumes(df, columns, normalization_values)` - Per volumi cerebrali
- `load_normalization_settings(filepath)` - Carica settings da JSON

**Arrotondamento intelligente:**
- **Metodo "increasing"** (FAQ, CDRSB, ADAS): round down se decimale < 0.5, round up se ≥ 0.5
- **Metodo "inverse"** (MMSE, RAVLT, MOCA): round down se decimale ≤ 0.5, round up se > 0.5

**Formule:**
```
# Increasing (min→0, max→1)
valore_originale = valore_normalizzato × (max - min) + min

# Inverse (min→1, max→0)
valore_originale = (1 - valore_normalizzato) × (max - min) + min
```

---

### 5. `dummy_to_categorical.py` - Conversione Dummy → Categorical

Funzioni per riconvertire variabili dummy (one-hot encoded) in colonne categoriche.

**Funzioni:**
- `dummies_to_categorical(df, dummy_prefix, separator, ...)` - Converte singolo gruppo
- `multiple_dummies_to_categorical(df, dummy_configs, ...)` - Converte multipli gruppi
- `auto_detect_dummies(df, separator)` - Rileva automaticamente gruppi dummy
- `label_decode(df, column, label_mapping)` - Decodifica label encoding
- `binary_to_categorical(df, column, mapping)` - Converte binarie in categoriche

**Esempio auto-detection:**
```python
from dummy_to_categorical import auto_detect_dummies

# Rileva automaticamente le dummy variables
dummy_groups = auto_detect_dummies(df, separator='/')
# Output: {'SEX': ['SEX/M', 'SEX/F'], 'DX': ['DX/CN', 'DX/MCI', 'DX/AD']}
```

---

### 6. `noise_injection_from_datalake.py` - Noise Injection con Differential Privacy

Script per aggiungere rumore ai dati sintetici usando meccanismi di Differential Privacy.

**Funzione principale:**
- `inject_noise_datalake_data(...)` - Pipeline completa con DP noise

**Pipeline:**
1. **STEP 1**: Carica dati sintetici dal datalake
2. **STEP 2**: Carica dati reali dal datalake (per calcolare sensitivity)
3. **STEP 3**: Applica noise injection con meccanismo scelto
4. **STEP 4**: Upload risultato su datalake
5. **STEP 5**: (opzionale) Salva copia locale

**Meccanismi disponibili:**
| Meccanismo | Descrizione | Garanzia DP |
|------------|-------------|-------------|
| `gaussian` | Rumore N(0, σ²), più smooth | (ε, δ)-DP |
| `laplace` | Rumore Laplace, più "spiky" | (ε, 0)-DP |
| `adaptive` | Scala in base al DCR corrente | Variabile |

**Uso da command line:**
```bash
# Base con gaussian noise
python noise_injection_from_datalake.py --synthetic-code ADNIMERGE_synthetic --real-code ADNIMERGE --epsilon 1.0

# Rumore aggressivo per DCR alto
python noise_injection_from_datalake.py --synthetic-code ADNIMERGE_synthetic --real-code ADNIMERGE --epsilon 0.5

# Metodo adaptive con DCR corrente
python noise_injection_from_datalake.py --synthetic-code ADNIMERGE_synthetic --real-code ADNIMERGE --method adaptive --current-dcr 0.82

# Senza correlazioni temporali
python noise_injection_from_datalake.py --synthetic-code ADNIMERGE_synthetic --real-code ADNIMERGE --no-temporal
```

**Guida epsilon:**
| DCR Corrente | Epsilon Consigliato | Effetto |
|--------------|---------------------|---------|
| > 0.80 | 0.5 - 1.0 | Rumore aggressivo |
| 0.60 - 0.80 | 1.0 - 2.0 | Rumore moderato |
| 0.50 - 0.60 | 2.0 - 5.0 | Rumore leggero |

---

### 7. `residual_par_correction.py` - Correzione Residual-PAR con KNN

Script avanzato per correggere dati sintetici usando **KNN pseudo-pairing + PARSynthesizer**. Non richiede che i dati reali e sintetici abbiano gli stessi ID.

**Funzione principale:**
- `residual_par_pipeline_from_datalake(...)` - Pipeline completa con KNN + PAR

**Pipeline:**
1. **STEP 1-2**: Carica dati reali e Leaspy/sintetici dal datalake
2. **STEP 3**: Valida baseline matching features
3. **STEP 4**: Auto-detect feature columns
4. **STEP 5**: Ensure monotone time per ID
5. **STEP 6**: KNN matching + build residual dataset
6. **STEP 7**: Train PARSynthesizer sui residui
7. **STEP 8**: Correggi Leaspy con residui sintetici (nearest merge)
8. **STEP 9**: Valutazione con KS e DCR
9. **STEP 10**: Merge back alla struttura originale
10. **STEP 11**: Upload su datalake
11. **STEP 12**: (opzionale) Salva copia locale

**Uso da command line:**
```bash
# Base con baseline features
python residual_par_correction.py \
    --file-code-real ADNIMERGE_for_validation_reversed \
    --file-code-leaspy ADNIMERGE_synthetic_reversed \
    --baseline-match-feats MMSE ADAS13 Hippocampus

# Con più neighbors
python residual_par_correction.py \
    --file-code-real ADNIMERGE_for_validation_reversed \
    --file-code-leaspy ADNIMERGE_synthetic_reversed \
    --baseline-match-feats MMSE ADAS13 \
    --k-neighbors 5 \
    --par-epochs 256
```

**Output metriche:**
- **KS Score** (per feature): più basso = migliore distribuzione
- **DCR Score**: più alto = migliore privacy

---

### 8. `inject_missing_values.py` - Iniezione Missing Values

Script per iniettare valori mancanti nei dati sintetici secondo i pattern osservati nei dati reali.

**Funzioni principali:**
- `inject_missing_longitudinal(df_synth, df_real, ...)` - Inietta missing con pattern reale
- `inject_missing_pipeline(...)` - Pipeline completa dal datalake

**Come funziona:**
1. Per ogni variabile, calcola il **missing rate** dai dati reali
2. Seleziona **randomicamente** lo stesso numero di valori nei dati sintetici
3. Imposta questi valori a **NaN**

**Uso da command line:**
```bash
# Base - tutte le colonne
python inject_missing_values.py --file-code-real ADNIMERGE_reversed --file-code-gen ADNIMERGE_synthetic_reversed

# Solo colonne specifiche
python inject_missing_values.py --file-code-real ADNIMERGE_reversed --file-code-gen ADNIMERGE_synthetic_reversed --target-columns ADAS11 ADAS13 MMSE
```

---

### 9. `add_noise.py` - Funzioni Residual-PAR

Modulo con funzioni di utilità per il metodo Residual-PAR.

**Funzioni principali:**
- `build_sequential_metadata(df, id_col, time_col, ...)` - Crea metadata SDV
- `ensure_monotone_per_id(df, id_col, time_col)` - Assicura tempo monotono
- `align_real_and_leaspy(...)` - Allinea dati reali e sintetici
- `compute_residuals(...)` - Calcola residui r_t = x_real_t - x_leaspy_t
- `train_par_on_residuals(...)` - Allena PARSynthesizer sui residui
- `correct_leaspy_with_synthetic_residuals(...)` - Corregge con x' = x_leaspy + r̂
- `quantile_map_to_real(...)` - Mappatura marginale sui reali
- `ks_by_feature(...)` - Calcola KS score per feature
- `dcr_distance_to_closest_record(...)` - Calcola DCR
- `residual_par_pipeline(...)` - Pipeline completa

---

### Moduli (Riepilogo API)

- **`denormalize.py`** - Funzioni per denormalizzare valori numerici
  - `denormalize_fixed_scale()` - Denormalizza con scale fisse
  - `denormalize_dataset_minmax()` - Denormalizza con min/max dataset
  - `denormalize_dataset_volumes()` - Denormalizza volumi cerebrali

- **`dummy_to_categorical.py`** - Funzioni per convertire dummy in categoriche
  - `dummies_to_categorical()` - Converte un set di dummy
  - `multiple_dummies_to_categorical()` - Converte più set di dummy
  - `auto_detect_dummies()` - Rileva automaticamente le dummy
  - `label_decode()` - Decodifica label encoding
  - `binary_to_categorical()` - Converte binarie in categoriche

- **`reverse_transformations.py`** - Entry point che orchestra tutto
  - `reverse_all_transformations()` - Pipeline unificata completa

- **`reverse_from_datalake.py`** - Entry point per dati dal datalake
  - `load_from_datalake()` - Carica dati dal datalake
  - `get_file_metadata()` - Recupera metadata da file sul datalake
  - `upload_to_datalake()` - Carica dataframe sul datalake
  - `reverse_datalake_data()` - Pipeline completa: carica, trasforma, carica

- **`noise_injection.py`** - Modulo per noise injection usando Differential Privacy
  - `PostGenerationNoiseInjector` - Classe per meccanismi DP base (Laplace, Gaussian, Adaptive)
  - `LongitudinalNoiseInjector` - Classe per dati longitudinali (preserva correlazioni temporali)
  - `add_post_generation_noise()` - Funzione principale per aggiungere rumore

- **`noise_injection_from_datalake.py`** - Entry point per noise dal datalake
  - `load_from_datalake()` - Carica dati sintetici e reali dal datalake
  - `inject_noise_datalake_data()` - Pipeline completa: carica, aggiungi rumore, carica

## Cosa fa lo script

Lo script `reverse_transformations.py` esegue le seguenti operazioni inverse:

1. **Denormalizzazione scale cliniche** (MMSE, FAQ, MOCA, CDRSB, ADAS11, ADAS13)
   - Da [0, 1] → valori originali
   - Gestisce sia metodo "increasing" che "inverse"

2. **Denormalizzazione volumi** (Hippocampus, Ventricles, Entorhinal, ecc.)
   - Da [0, 1] → valori originali in mm³ o percentuali
   - Gestisce sia volumi assoluti che percentuali %ICV

3. **Conversione dummy → categorical**
   - Da `SEX/M`, `SEX/F` → colonna `SEX` con valori `M`, `F`
   - Da `DX/CN`, `DX/MCI`, `DX/AD` → colonna `DX` con valori `CN`, `MCI`, `AD`
   - Supporta qualsiasi separatore (/, _, -, ecc.)

4. **Decodifica label encoding** (opzionale)
   - Da interi (0, 1, 2) → categorie originali

## Pipeline Datalake (reverse_from_datalake.py)

Lo script `reverse_from_datalake.py` esegue un flusso completo end-to-end:

```
┌─────────────────────────────────────────────────────────────┐
│                  PIPELINE DATALAKE                          │
└─────────────────────────────────────────────────────────────┘

1. LOAD FROM DATALAKE
   ├─ Query file by file_code
   ├─ Download and extract ZIP
   └─ Load DataFrame
          ↓
2. REVERSE TRANSFORMATIONS
   ├─ Denormalize clinical scales (integers with custom rounding)
   ├─ Denormalize volumes
   └─ Convert dummies to categorical
          ↓
3. GET METADATA
   ├─ Retrieve original metadata
   └─ Update file_code with '_reversed' suffix
          ↓
4. UPLOAD TO DATALAKE
   ├─ Create new filename with '_reversed' suffix
   ├─ Upload to specified prefix (synthetic/reversed or validation/reversed)
   └─ Store with updated metadata

[OPTIONAL] Save local copy
```

**Default behavior:** Upload automatico sul datalake con metadata preservati e file_code modificato.

## Pipeline Noise Injection (noise_injection_from_datalake.py)

Lo script `noise_injection_from_datalake.py` implementa meccanismi di **Differential Privacy** per aggiungere rumore controllato ai dati sintetici, migliorando la privacy senza compromettere eccessivamente l'utilità.

### Cosa fa

```
┌─────────────────────────────────────────────────────────────┐
│           NOISE INJECTION PIPELINE DATALAKE                 │
└─────────────────────────────────────────────────────────────┘

1. LOAD SYNTHETIC DATA FROM DATALAKE
   ├─ Query file by file_code (e.g., ADNIMERGE_synthetic)
   ├─ Download and extract ZIP
   └─ Load DataFrame
          ↓
2. LOAD REAL DATA FROM DATALAKE
   ├─ Query file by file_code (e.g., ADNIMERGE)
   ├─ Download and extract ZIP
   └─ Load DataFrame (needed for sensitivity calculation)
          ↓
3. NOISE INJECTION
   ├─ Auto-detect ordinal features (MMSE, CDRSB, etc.)
   ├─ Calculate sensitivity from real data distributions
   ├─ Apply noise mechanism (Gaussian, Laplace, or Adaptive)
   ├─ Preserve temporal correlations (for longitudinal data)
   ├─ Round ordinal features to integers
   └─ Clip values to valid ranges
          ↓
4. GET METADATA
   ├─ Retrieve original metadata
   ├─ Update file_code with '_noisy' suffix
   └─ Add noise parameters to metadata
          ↓
5. UPLOAD TO DATALAKE
   ├─ Create new filename with '_noisy_epsX.XX' suffix
   ├─ Upload to specified prefix (synthetic/noisy)
   └─ Store with updated metadata

[OPTIONAL] Save local copy
```

### Meccanismi di rumore disponibili

1. **Gaussian Mechanism** (default, consigliato)
   - Rumore dalla distribuzione normale N(0, σ²)
   - σ calibrato per garantire (ε, δ)-differential privacy
   - Più smooth, ideale per dati continui

2. **Laplace Mechanism**
   - Rumore dalla distribuzione Laplace
   - Garantisce (ε, 0)-differential privacy (più forte)
   - Più "spiky", può creare outlier

3. **Adaptive Mechanism**
   - Scala automaticamente il rumore in base al DCR corrente
   - Se DCR > target_dcr, aumenta il rumore
   - Utile per raggiungere un DCR specifico

### Preservazione correlazioni temporali

Per **dati longitudinali**, il sistema può preservare correlazioni temporali:
- Genera rumore correlato usando processo AR(1)
- Mantiene le traiettorie individuali più coerenti
- De-correla il rumore tra pazienti diversi

### Parametri chiave

- **epsilon (ε)**: Privacy budget - controlla l'intensità del rumore
  - Più basso = più privacy (più rumore)
  - Più alto = più utilità (meno rumore)
  - Range consigliato: 0.5 - 5.0

- **delta (δ)**: Probabilità di fallimento per (ε, δ)-DP
  - Default: 1/n² (dove n = numero record dataset reale)
  - Tipicamente molto piccolo (< 1e-5)

- **method**: Meccanismo di rumore
  - `'gaussian'` (default): Più smooth
  - `'laplace'`: DP più forte
  - `'adaptive'`: Adatta automaticamente

### Guida epsilon in base al DCR

| DCR corrente | Obiettivo | Epsilon consigliato | Effetto |
|--------------|-----------|---------------------|---------|
| > 0.80 | Privacy alta | 0.5 - 1.0 | Rumore aggressivo |
| 0.60 - 0.80 | Privacy media | 1.0 - 2.0 | Rumore moderato |
| 0.50 - 0.60 | Vicino al target | 2.0 - 5.0 | Rumore leggero |

### Quali colonne ricevono rumore?

**✅ RICEVONO rumore:**
- **Scale cliniche normalizzate** [0,1]: `MMSE`, `CDRSB`, `FAQ`, `ADAS11`, `ADAS13`, `RAVLT`
  - Generate come **continue** dal modello → ricevono noise → arrotondate dopo reverse
- **Volumi cerebrali** [0,1]: `Hippocampus`, `Ventricles`, `Entorhinal`, ecc.
- **Biomarker continui**: `ABETA`, `TAU`, `PTAU`, ecc.
- Tutte le altre variabili numeriche continue

**❌ NON ricevono rumore:**
- **Dummy variables** (es: `SEX/M`, `SEX/F`, `DX/CN`, `DX/MCI`, `DX/AD`)
  - Identificate automaticamente: separatore `/` + valori binari 0/1
- **Categoriche ordinali** (es: `APOE`, `EDUCATION`, `PTGENDER`)
  - Generate come **discrete** {0, 1, 2} dal modello
  - Identificate automaticamente: ≤10 valori unici interi con range ≤20
  - Lista nota: `APOE`, `APOE4`, `EDUCATION`, `PTGENDER`, `PTEDUCAT`
- **ID paziente** (`RID`)
- **Variabile temporale** (`AGE`, `TIME`)

**Distinzione cruciale:**
- **MMSE** (scala clinica): generato continuo [0.0, 0.033, 0.067, ..., 1.0] → **SÌ rumore**
- **APOE** (categorica): generato discreto {0, 1, 2} → **NO rumore**

**Output dello script (esempio):**
```
[INFO] Auto-detected 15 dummy columns (will NOT receive noise)
      Examples: ['SEX/M', 'SEX/F', 'DX/CN', 'DX/MCI', 'DX/AD']
[INFO] Auto-detected 2 categorical ordinal variables (will NOT receive noise)
      Examples: ['APOE', 'EDUCATION']
[INFO] 24 features will receive noise
      Including ordinals (will be rounded after): ['MMSE', 'CDRSB', 'FAQ', 'ADAS11', 'ADAS13']
```

### Ordine corretto delle operazioni

```
1. Generazione Leaspy
   ↓ [Dati normalizzati [0,1] + dummy 0/1]

2. noise_injection_from_datalake
   ↓ [Noise su [0,1], NO noise su dummy]
   ↓ [Dati normalizzati CON rumore + dummy intatti]

3. reverse_from_datalake
   ↓ [Denormalizza → arrotonda ordinali → dummy→categorical]
   ↓ [Dati su scale originali + categoriche]

4. inject_missing_values (opzionale)
   ↓ [Aggiunge NA strategici]
   ↓ [Dataset finale]
```

Questo ordine preserva le garanzie di Differential Privacy grazie al principio di **post-processing immunity**.

### Basi scientifiche

Il modulo implementa meccanismi basati su:
- **Dwork et al. (2006)**: Calibrating noise to sensitivity (TCC)
- **Dwork & Roth (2014)**: Algorithmic foundations of differential privacy
- **Balle et al. (2018)**: Improving Gaussian mechanism (ICML)
- **NIST (2021)**: Guidelines for Evaluating Differential Privacy
- **SafeSynthDP (2024)**: Privacy-preserving synthetic data (arXiv:2412.20641)

## Utilizzo

### Esempio 1: Pipeline completa automatica (Entry Point)

```python
import pandas as pd
import sys
sys.path.append('post-generation')

from reverse_transformations import reverse_all_transformations

# Carica i dati trasformati
df_transformed = pd.read_csv('data_normalized.csv')

# Configura le variabili dummy da convertire
dummy_configs = [
    {'prefix': 'SEX', 'separator': '/'},
    {'prefix': 'DX', 'separator': '/'},
    {'prefix': 'APOE4', 'separator': '/'}
]

# Esegui tutte le trasformazioni inverse
df_original = reverse_all_transformations(
    df_transformed,
    normalization_settings_path='normalization_settings.json',
    volume_settings_path='volume_values_settings.json',
    dummy_configs=dummy_configs,
    separator='/'
)

# Salva i risultati
df_original.to_csv('data_original.csv', index=False)
```

### Esempio 2: Solo denormalizzazione (Modulo specifico)

```python
import sys
sys.path.append('post-generation')

from denormalize import denormalize_fixed_scale, load_normalization_settings

# Carica impostazioni
settings = load_normalization_settings('post-generation/normalization_settings.json')

# Denormalizza solo le scale cliniche
df_denorm = denormalize_fixed_scale(
    df,
    columns=['MMSE', 'FAQ', 'MOCA'],
    min_max_scales=settings
)
```

### Esempio 3: Solo conversione dummy → categorical (Modulo specifico)

```python
import sys
sys.path.append('post-generation')

from dummy_to_categorical import dummies_to_categorical, multiple_dummies_to_categorical

# Converti un singolo set di dummy
df_restored = dummies_to_categorical(
    df,
    dummy_prefix='SEX',
    separator='/',
    drop_dummies=True
)

# Converti multipli set
configs = [
    {'prefix': 'SEX', 'separator': '/'},
    {'prefix': 'DX', 'separator': '/'},
    {'prefix': 'APOE4', 'separator': '/'}
]

df_restored = multiple_dummies_to_categorical(df, configs, separator='/')
```

### Esempio 4: Auto-rilevamento dummy (Modulo specifico)

```python
import sys
sys.path.append('post-generation')

from dummy_to_categorical import auto_detect_dummies

# Rileva automaticamente tutte le variabili dummy
dummy_groups = auto_detect_dummies(df, separator='/')

print("Gruppi di dummy rilevati:")
for prefix, columns in dummy_groups.items():
    print(f"  {prefix}: {columns}")

# Output esempio:
# SEX: ['SEX/M', 'SEX/F']
# DX: ['DX/CN', 'DX/MCI', 'DX/AD']
# APOE4: ['APOE4/0', 'APOE4/1', 'APOE4/2']
```

### Esempio 5: Decodifica label encoding (Modulo specifico)

```python
import sys
sys.path.append('post-generation')

from dummy_to_categorical import label_decode

# Definisci il mapping
mapping = {0: 'CN', 1: 'MCI', 2: 'AD'}

# Decodifica
df_decoded = label_decode(df, 'DX', mapping)
```

### Esempio 6: Caricamento dal Datalake, reverse transformations e upload

**NOTA IMPORTANTE:**
- Di default, i risultati vengono caricati automaticamente sul datalake nel prefix specificato (`synthetic/reversed` o `validation/reversed`)
- **Le variabili dummy vengono rilevate automaticamente** (colonne con `/` nel nome e valori 0/1)
- È possibile specificare manualmente i `dummy_configs` per sovrascrivere l'auto-detection

#### 6a. Uso programmatico - Auto-detection dummy variables (consigliato)

```python
import sys
sys.path.append('post-generation')

from reverse_from_datalake import reverse_datalake_data

# Auto-detection: le variabili dummy vengono rilevate automaticamente
# Il sistema cerca colonne con '/' nel nome e valori binari (0/1)
df_reversed = reverse_datalake_data(
    file_code='ADNIMERGE_synthetic',
    upload_to_dl=True,  # Default: True
    prefix='synthetic/reversed'  # Prefix per upload
)

# Per dati di validazione (auto-detection)
df_reversed = reverse_datalake_data(
    file_code='ADNIMERGE_validation',
    prefix='validation/reversed'
)

print(f"Dataset reversed e caricato: {df_reversed.shape}")
```

#### 6b. Uso programmatico - Configurazione manuale dummy (override auto-detection)

```python
# Configurazione manuale dummy variables (sovrascrive auto-detection)
dummy_configs = [
    {'prefix': 'SEX', 'separator': '/'},
    {'prefix': 'DX', 'separator': '/'},
    {'prefix': 'APOE4', 'separator': '/'}
]

# Con configurazione esplicita
df_reversed = reverse_datalake_data(
    file_code='ADNIMERGE_synthetic',
    upload_to_dl=True,
    prefix='synthetic/reversed',
    dummy_configs=dummy_configs,  # Override auto-detection
    separator='/'
)
```

#### 6c. Uso programmatico - Con copia locale

```python
# Carica su datalake E salva copia locale (con auto-detection)
df_reversed = reverse_datalake_data(
    file_code='ADNIMERGE_synthetic',
    upload_to_dl=True,
    prefix='synthetic/reversed',
    save_local=True,
    output_path='reversed_data_local.csv'
)

# Solo copia locale (senza upload su datalake, con auto-detection)
df_reversed = reverse_datalake_data(
    file_code='ADNIMERGE_synthetic',
    upload_to_dl=False,
    save_local=True,
    output_path='reversed_data_local.csv'
)
```

#### 6d. Uso da command line - Upload su datalake con auto-detection

```bash
cd post-generation
conda activate AIND

# Uso base - carica su datalake con auto-detection (consigliato)
python reverse_from_datalake.py --file-code ADNIMERGE_synthetic --prefix synthetic/reversed

# Dati di validazione con auto-detection
python reverse_from_datalake.py --file-code ADNIMERGE_validation --prefix validation/reversed

# Con copia locale
python reverse_from_datalake.py --file-code ADNIMERGE_synthetic --prefix synthetic/reversed --save-local --output reversed_local.csv

# Solo locale (senza upload)
python reverse_from_datalake.py --file-code ADNIMERGE_synthetic --no-upload --save-local --output reversed_local.csv

# Con dummy prefixes personalizzati (override auto-detection)
python reverse_from_datalake.py --file-code ADNIMERGE_synthetic --prefix synthetic/reversed --dummy-prefixes SEX DX APOE4 EDUCATION

# Con parametri query aggiuntivi
python reverse_from_datalake.py --file-code ADNIMERGE_synthetic --query-param custom.level=cleaned_03 --prefix synthetic/reversed
```

#### 6e. Solo caricamento dal datalake (senza reverse)

```python
import sys
sys.path.append('post-generation')

from reverse_from_datalake import load_from_datalake

# Carica solo i dati dal datalake
# Nota: load_from_datalake ora restituisce (df, object_name, file_name)
df, object_name, file_name = load_from_datalake(
    file_code='ADNIMERGE_synthetic'
)

print(f"Dataset caricato: {df.shape}")
print(f"Object name: {object_name}")
print(f"File name: {file_name}")

# Opzionale: con parametri query aggiuntivi
df, object_name, file_name = load_from_datalake(
    file_code='ADNIMERGE_synthetic',
    query_params={'custom.level': 'cleaned_03'}
)
```

#### 6f. Gestione metadata e auto-detection

**Auto-detection variabili dummy:**

Il sistema rileva automaticamente variabili dummy cercando:
- Colonne con il separatore specificato nel nome (default: `/`)
- Valori binari (0 o 1) in tutte le righe
- Raggruppa per prefisso (es: `SEX/M`, `SEX/F` → gruppo `SEX`)

Esempio output auto-detection:
```
[STEP 1.5] Auto-detecting dummy variables...
   ✓ Auto-detected 3 dummy variable groups:
     - SEX: 2 columns
     - DX: 3 columns
     - APOE4: 3 columns
```

**Processo automatico di upload sul datalake:**

1. **Recupera metadata originali** dal file di input (usando `search_files`)
2. **Copia tutti i metadata custom**
3. **Modifica solo il `file_code`** aggiungendo suffisso `_reversed`
4. **Crea nome file** con suffisso `_reversed` (es: `ADNIMERGE_synthetic.csv` → `ADNIMERGE_synthetic_reversed.csv`)
5. **Upload nel prefix specificato** (`synthetic/reversed` o `validation/reversed`)

**Esempio metadata:**
```python
# Metadata originale
{
    'file_code': 'ADNIMERGE_synthetic',
    'level': 'cleaned_03',
    'version': '1.0',
    ...
}

# Metadata dopo reverse
{
    'file_code': 'ADNIMERGE_synthetic_reversed',  # ← Modificato
    'level': 'cleaned_03',  # ← Mantenuto
    'version': '1.0',  # ← Mantenuto
    ...
}
```

### Esempio 7: Pipeline completa post-generazione (RACCOMANDATO)

Questo è il **flusso completo consigliato** dopo la generazione con Leaspy:

```python
import sys
sys.path.append('post-generation')

from noise_injection_from_datalake import inject_noise_datalake_data
from reverse_from_datalake import reverse_datalake_data
# from inject_missing_values import inject_missing_values  # Se disponibile

# Step 1: Genera dati con Leaspy
# (I dati generati sono normalizzati [0,1] con dummy 0/1)
# Supponiamo siano già caricati su datalake come 'ADNIMERGE_synthetic'

# Step 2: Aggiungi rumore (su dati normalizzati)
print("="*60)
print("STEP 1: Noise Injection")
print("="*60)
synth_noisy = inject_noise_datalake_data(
    synthetic_file_code='ADNIMERGE_synthetic',
    real_file_code='ADNIMERGE',
    epsilon=1.0,
    method='gaussian',
    prefix='synthetic/noisy'
)
# Output su datalake: ADNIMERGE_synthetic_noisy

# Step 3: Reverse transformations
print("\n" + "="*60)
print("STEP 2: Reverse Transformations")
print("="*60)
synth_final = reverse_datalake_data(
    file_code='ADNIMERGE_synthetic_noisy',
    prefix='synthetic/final'
)
# Output su datalake: ADNIMERGE_synthetic_noisy_reversed

# Step 4 (opzionale): Inject missing values
print("\n" + "="*60)
print("STEP 3: Inject Missing Values")
print("="*60)
# synth_with_missing = inject_missing_values(synth_final, ...)

print("\n✓ Pipeline completa!")
```

#### Da command line:

```bash
cd post-generation
conda activate AIND

# Step 1: Noise injection
python noise_injection_from_datalake.py \
    --synthetic-code ADNIMERGE_synthetic \
    --real-code ADNIMERGE \
    --epsilon 1.0 \
    --prefix synthetic/noisy

# Step 2: Reverse transformations
python reverse_from_datalake.py \
    --file-code ADNIMERGE_synthetic_noisy \
    --prefix synthetic/final

# Step 3 (opzionale): Inject missing values
# python inject_missing_values.py --file-code ADNIMERGE_synthetic_noisy_reversed
```

### Esempio 8: Noise Injection dal Datalake (dettagli)

#### 8a. Uso base - Gaussian noise con parametri default

```python
import sys
sys.path.append('post-generation')

from noise_injection_from_datalake import inject_noise_datalake_data

# Uso base: carica dati sintetici e reali, aggiungi rumore, upload
synth_noisy = inject_noise_datalake_data(
    synthetic_file_code='ADNIMERGE_synthetic',
    real_file_code='ADNIMERGE',
    epsilon=1.0,
    method='gaussian'
)

print(f"Dataset con rumore: {synth_noisy.shape}")
```

#### 8b. Adaptive noise con DCR corrente

```python
# Usa metodo adaptive per raggiungere target DCR
# Se DCR corrente è alto (es: 0.82), aumenta automaticamente il rumore
synth_noisy = inject_noise_datalake_data(
    synthetic_file_code='ADNIMERGE_synthetic',
    real_file_code='ADNIMERGE',
    epsilon=1.0,
    method='adaptive',
    current_dcr=0.82,  # DCR dalla validazione
    target_dcr=0.50    # Target desiderato
)
```

#### 8c. Rumore aggressivo per alto DCR

```python
# Per DCR molto alto (>0.80), usa epsilon basso per più rumore
synth_noisy = inject_noise_datalake_data(
    synthetic_file_code='ADNIMERGE_synthetic',
    real_file_code='ADNIMERGE',
    epsilon=0.8,  # Basso = più rumore
    method='gaussian',
    preserve_temporal=True  # Preserva correlazioni temporali
)
```

#### 8d. Senza preservazione correlazioni temporali

```python
# Per dati non longitudinali o per rumore i.i.d.
synth_noisy = inject_noise_datalake_data(
    synthetic_file_code='ADNIMERGE_synthetic',
    real_file_code='ADNIMERGE',
    epsilon=1.5,
    preserve_temporal=False  # No correlazioni temporali
)
```

#### 8e. Con feature ordinali personalizzate

```python
# Specifica manualmente le feature ordinali
synth_noisy = inject_noise_datalake_data(
    synthetic_file_code='ADNIMERGE_synthetic',
    real_file_code='ADNIMERGE',
    epsilon=1.0,
    ordinal_features=['MMSE', 'CDRSB', 'FAQ', 'ADAS13'],
    continuous_features=['Hippocampus', 'Ventricles']
)
```

#### 8f. Con copia locale e parametri custom

```python
# Upload su datalake + copia locale
synth_noisy = inject_noise_datalake_data(
    synthetic_file_code='ADNIMERGE_synthetic',
    real_file_code='ADNIMERGE',
    epsilon=1.2,
    method='gaussian',
    save_local=True,
    output_path='synthetic_noisy_eps1.2.csv',
    patient_col='RID',
    time_col='AGE'
)
```

#### 8g. Solo locale (senza upload)

```python
# Non caricare su datalake, solo file locale
synth_noisy = inject_noise_datalake_data(
    synthetic_file_code='ADNIMERGE_synthetic',
    real_file_code='ADNIMERGE',
    epsilon=1.0,
    upload_to_dl=False,
    save_local=True,
    output_path='synthetic_noisy_local.csv'
)
```

#### 8h. Uso da command line

```bash
cd post-generation
conda activate AIND

# Uso base
python noise_injection_from_datalake.py --synthetic-code ADNIMERGE_synthetic --real-code ADNIMERGE --epsilon 1.0

# Con metodo adaptive e DCR corrente
python noise_injection_from_datalake.py --synthetic-code ADNIMERGE_synthetic --real-code ADNIMERGE --method adaptive --current-dcr 0.82 --epsilon 1.0

# Rumore aggressivo (epsilon basso)
python noise_injection_from_datalake.py --synthetic-code ADNIMERGE_synthetic --real-code ADNIMERGE --epsilon 0.8

# Con copia locale
python noise_injection_from_datalake.py --synthetic-code ADNIMERGE_synthetic --real-code ADNIMERGE --epsilon 1.5 --save-local --output noisy_data.csv

# Senza correlazioni temporali
python noise_injection_from_datalake.py --synthetic-code ADNIMERGE_synthetic --real-code ADNIMERGE --epsilon 1.0 --no-temporal

# Test multipli epsilon values
for eps in 0.5 0.8 1.0 1.5 2.0; do
    python noise_injection_from_datalake.py --synthetic-code ADNIMERGE_synthetic --real-code ADNIMERGE --epsilon $eps --prefix synthetic/noisy
done

# Con custom ordinal features
python noise_injection_from_datalake.py --synthetic-code ADNIMERGE_synthetic --real-code ADNIMERGE --epsilon 1.0 --ordinal-features MMSE CDRSB FAQ

# Con parametri query aggiuntivi
python noise_injection_from_datalake.py --synthetic-code ADNIMERGE_synthetic --real-code ADNIMERGE --epsilon 1.0 --synthetic-query custom.version=v2 --real-query custom.level=cleaned_03
```

#### 8i. Workflow completo: iterativo per trovare epsilon ottimale

```python
import sys
sys.path.append('post-generation')

from noise_injection_from_datalake import inject_noise_datalake_data

# Test diversi epsilon per trovare quello che raggiunge target DCR
test_epsilons = [0.5, 0.8, 1.0, 1.5, 2.0]

for eps in test_epsilons:
    print(f"\n{'='*60}")
    print(f"Testing epsilon = {eps}")
    print('='*60)

    synth_noisy = inject_noise_datalake_data(
        synthetic_file_code='ADNIMERGE_synthetic',
        real_file_code='ADNIMERGE',
        epsilon=eps,
        method='gaussian',
        prefix=f'synthetic/noisy_test',
        save_local=True,
        output_path=f'synthetic_noisy_eps{eps}.csv'
    )

    # Qui puoi validare il nuovo dataset e calcolare il DCR
    # new_dcr = validate_and_compute_dcr(real_df, synth_noisy)
    # if 0.45 <= new_dcr <= 0.55:
    #     print(f"✓ Found optimal epsilon: {eps}")
    #     break
```

## Test dello script

Per testare lo script con dati di esempio:

```bash
python reverse_transformations.py
```

Questo eseguirà un esempio completo e salverà i risultati in `example_reversed_data.csv`.

## Parametri importanti

### `handle_all_zero` (per dummy variables)

Gestisce il caso in cui tutte le dummy sono 0:
- `'missing'` (default): Imposta a NaN
- `'first'`: Usa la prima categoria
- `'none'`: Usa 'Unknown'

### `separator`

Il separatore usato nelle variabili dummy:
- `/` per `SEX/M`, `DX/CN`, ecc.
- `_` per `SEX_M`, `DX_CN`, ecc.
- `-` per `SEX-M`, `DX-CN`, ecc.

## Note

- Lo script gestisce automaticamente le colonne mancanti
- I valori fuori range vengono gestiti correttamente
- Supporta qualsiasi combinazione di trasformazioni
- È sicuro eseguirlo su dataframe parziali (con solo alcune colonne trasformate)

## Formule matematiche

### Denormalizzazione "increasing" (min→0, max→1)
```
valore_originale = valore_normalizzato × (max - min) + min
```

### Denormalizzazione "inverse" (min→1, max→0)
```
valore_originale = (1 - valore_normalizzato) × (max - min) + min
```

## Troubleshooting

**Problema**: "No dummy columns found with prefix..."
- **Soluzione**: Verifica il separatore usato (/, _, ecc.) e il prefisso corretto

**Problema**: "min and max are equal for column..."
- **Soluzione**: Normale, quella colonna aveva un solo valore nel dataset originale

**Problema**: Valori strani dopo denormalizzazione
- **Soluzione**: Verifica che i file JSON abbiano i parametri corretti per quella colonna

**Problema**: "No module named 'dl_client'"
- **Soluzione**: La libreria `dl_client` non è installata. Il modulo `reverse_from_datalake.py` richiede questa dipendenza. Puoi comunque usare gli altri moduli senza problemi.

**Problema**: "No file found with code 'XXX'"
- **Soluzione**: Verifica che il `file_code` sia corretto e che il file esista nel datalake. Controlla i parametri query se necessario.

**Problema**: "No module named 'noise_injection'"
- **Soluzione**: Assicurati di essere nella cartella `post-generation` o di aver aggiunto il path corretto con `sys.path.append('post-generation')`.

**Problema**: Il rumore aggiunto è troppo/troppo poco
- **Soluzione**: Regola il parametro `epsilon`. Valori più bassi = più rumore. Testa con diversi valori (0.5, 1.0, 2.0) e valida i risultati.

**Problema**: Il DCR non migliora nonostante il rumore
- **Soluzione**: Usa il metodo `'adaptive'` con `current_dcr` specificato, o abbassa ulteriormente `epsilon` (es: 0.5 o 0.3).

## Note sulla denormalizzazione con arrotondamento

Le scale cliniche (MMSE, FAQ, MOCA, ecc.) vengono denormalizzate e arrotondate a valori interi usando regole specifiche:

- **Metodo "increasing"** (FAQ, CDRSB, ADAS11, ADAS13):
  - Arrotonda **per difetto** se decimale **< 0.5**
  - Arrotonda **per eccesso** se decimale **≥ 0.5**

- **Metodo "inverse"** (MMSE, RAVLT, MOCA):
  - Arrotonda **per difetto** se decimale **≤ 0.5**
  - Arrotonda **per eccesso** se decimale **> 0.5**

Questo garantisce la coerenza con le scale originali che sono sempre valori interi.

## Autore

Script creato con Claude Code per il progetto AIND WP-2.
