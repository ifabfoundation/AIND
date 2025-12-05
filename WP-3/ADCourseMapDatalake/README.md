# AD Course Map Datalake

AD Course Map è un framework di modellazione statistica multivariata a effetti misti progettato per l'analisi di dati longitudinali, con particolare focus sui dati medici. È stato sviluppato per simulare la progressione di malattie, come l'Alzheimer, e per generare dati sintetici a partire da osservazioni reali.

Il progetto utilizza la libreria **Leaspy** per implementare un modello a effetti misti che apprende traiettorie di progressione della malattia, permettendo di:
- Addestrare modelli su dati longitudinali reali
- Fare predizioni su biomarker nel tempo
- Generare dataset sintetici con proprietà statistiche controllate

---

## Indice

- [Prerequisiti](#prerequisiti)
- [Installazione](#installazione)
- [Struttura del Progetto](#struttura-del-progetto)
- [Descrizione dei Moduli](#descrizione-dei-moduli)
- [Come Eseguire il Progetto](#come-eseguire-il-progetto)
- [Argomenti da Linea di Comando](#argomenti-da-linea-di-comando)
- [Struttura Cartelle Generate](#struttura-cartelle-generate)
- [Esempi di Output](#esempi-di-output)
- [Dataset e Biomarker](#dataset-e-biomarker)
- [Esecuzione su Cluster HPC](#esecuzione-su-cluster-hpc)
- [Riferimenti Bibliografici](#riferimenti-bibliografici)

---

## Prerequisiti

- **Python**: >= 3.9, < 3.12
- **Leaspy**: Framework per modelli a effetti misti longitudinali
- **PyTorch**: Per supporto GPU (opzionale)
- **Datalake Client**: Per integrazione con il datalake

---

## Installazione

### 1. Clonare il repository

```bash
git clone https://github.com/github_username/repo_name.git
cd ADCourseMapDatalake
```

### 2. Creare un ambiente conda (opzionale ma consigliato)

```bash
conda create --name adcoursemap python=3.10
conda activate adcoursemap
```

### 3. Installare le dipendenze

```bash
pip install -r requirements.txt
```

### Dipendenze principali (requirements.txt)

```
pandas==1.5.3
numpy==1.26.4
matplotlib==3.9.2
seaborn==0.13.2
leaspy==1.5.0
```

**Dipendenze aggiuntive** (non incluse in requirements.txt):
- `torch` - Per accelerazione GPU
- `dl_client` - Client per accesso al Datalake

---

## Struttura del Progetto

```
ADCourseMapDatalake/
├── bash/                                   # Script SLURM per cluster HPC
│   ├── train_job.sh                        # Job di training
│   ├── predict_job.sh                      # Job di predizione
│   └── simulate_job.sh                     # Job di generazione dati sintetici
│
├── code/                                   # Codice Python principale
│   ├── main.py                             # Entry point - orchestrazione
│   ├── model.py                            # Classe LeaspyModel
│   ├── loader.py                           # Caricamento e preprocessing dati
│   ├── train.py                            # Training e validazione
│   ├── loss.py                             # Metriche di errore (MAE)
│   ├── prediction.py                       # Predizioni su nuovi dati
│   ├── simulation.py                       # Generazione dati sintetici
│   └── utils.py                            # Funzioni di utilità
│
├── saved_models/                           # Modelli pre-addestrati
│   ├── model_parameters.json               # Parametri modello trainato
│   └── model_parameters_best.json          # Miglior modello
│
├── utils/                                  # File di configurazione
│   └── algorithm_settings_calibration.json # Impostazioni algoritmo MCMC-SAEM
│
├── results/                                # [GENERATA] Output degli esperimenti
│   └── experiment_{timestamp}/             # Cartella singolo esperimento
│       ├── logs/                           # Log e metriche di training
│       │   ├── train_noise.txt             # Rumore stimato per feature
│       │   ├── validation_results.txt      # MAE ± CI per variabile
│       │   └── calibration_parameters.csv  # Log iterazioni MCMC
│       ├── images/                         # Grafici e visualizzazioni
│       │   ├── mean_curves.png             # Traiettorie medie popolazione
│       │   ├── distribution_hist_*.png     # Istogrammi reale vs sintetico
│       │   └── CDF_hist_*.png              # CDF reale vs sintetico
│       └── weights/                        # Pesi del modello addestrato
│           ├── model_parameters.json       # Parametri modello
│           └── average_parameters.json     # Parametri paziente medio
│
├── README.md                               # Questa documentazione
└── requirements.txt                        # Dipendenze Python
```

> **Nota:** Le cartelle marcate con `[GENERATA]` vengono create automaticamente durante l'esecuzione degli script e sono escluse dal version control (vedi `.gitignore`).

---

## Descrizione dei Moduli

### main.py - Entry Point

Il file principale che orchestra l'intera pipeline. Gestisce il parsing degli argomenti da linea di comando e indirizza l'esecuzione verso una delle tre modalità:

1. **Training**: Addestramento del modello
2. **Prediction**: Predizioni su nuovi dati
3. **Simulation**: Generazione di dati sintetici

**Responsabilità principali:**
- Parsing degli argomenti CLI con `argparse`
- Inizializzazione dei seed per riproducibilità
- Creazione delle directory di output
- Istanziazione del modello Leaspy
- Routing alle diverse modalità di esecuzione

---

### model.py - Classe LeaspyModel

Wrapper attorno alla libreria Leaspy che fornisce un'interfaccia unificata per tutte le operazioni del modello.

**Metodi principali:**

| Metodo | Descrizione |
|--------|-------------|
| `initialize(data)` | Inizializza il modello con i dati |
| `cuda(device)` | Sposta il modello su GPU/CPU |
| `fit(data)` | Addestra il modello usando MCMC-SAEM |
| `save_model()` | Salva i parametri del modello in JSON |
| `personalize(data)` | Stima i parametri individuali per ogni paziente |
| `predict(ids, timepoints, data)` | Genera predizioni per timepoint specificati |
| `estimate(data_pers, data_pred)` | Personalizza su un dataset, predice su un altro |
| `forward(data)` | Pipeline completa: fit → save → visualize |
| `generate_virtual_data(data)` | Crea dati sintetici |
| `generate_virtual_data_with_diagnosis(data, dx_percentages)` | Genera dati con distribuzione diagnosi controllata |
| `generate_virtual_data_with_cofactors(data, ...)` | Genera dati con combinazioni di cofattori |

**Metodi di visualizzazione:**
- `__make_curves()` - Traiettorie medie della popolazione
- `_make_individual_curvers()` - Predizioni individuali per paziente
- `__compare_datasets_distribution()` - Confronto istogrammi e CDF

---

### loader.py - Caricamento Dati

Gestisce il caricamento e il preprocessing dei dati dal Datalake.

**Flusso di lavoro:**
1. Carica metadati dalle opzioni (level, file_code)
2. Recupera il dataset dal Datalake
3. Estrae colonne predittori e cofattori
4. Pulisce e filtra i dati
5. Converte cofattori booleani in interi
6. Applica normalizzazione (fixed-scale)
7. Restituisce DataFrame processato

**Funzioni di normalizzazione:**
- `normalize_fixed_scale()` - Normalizza in range [0,1] con bound predefiniti
- `normalize_dataset_minmax()` - Normalizzazione min-max basata sui dati
- `normalize_dataset_volumes()` - Normalizzazione volumi cerebrali basata su ICV

---

### train.py - Training e Validazione

Gestisce l'intero processo di addestramento e validazione del modello.

**Funzione principale:** `train_and_validate(dataset, model, opt, logs_path, model_path, settings_path)`

**Workflow:**
1. Split dataset in train/validation (80/20) con stratificazione per numero visite
2. Creazione dataset separati per predittori e cofattori
3. Conversione in oggetti Leaspy Data
4. Training del modello tramite `model.forward()`
5. Estrazione e salvataggio parametri medi
6. Validazione:
   - Personalizzazione su dati di validazione
   - Generazione predizioni
   - Calcolo MAE con intervalli di confidenza

**Output generati:**
- `logs/train_noise.txt` - Stime del rumore per feature
- `logs/validation_results.txt` - MAE ± CI per ogni variabile
- `weights/average_parameters.json` - Parametri del paziente medio

---

### loss.py - Metriche di Errore

Implementa il calcolo del Mean Absolute Error (MAE) con intervalli di confidenza al 95%.

**Classe:** `MAE_CI`

**Metodo:** `calculate(d_real, d_predicted)`

**Processo:**
1. Per ogni variabile:
   - Crea maschera per valori non-NaN
   - Filtra valori reali e predetti
   - Calcola residui: reale - predetto
   - Calcola MAE: mean(|residui|)
   - Calcola CI: 1.96 × std(residui) / √n

**Output:** Dizionario `{variabile: {'mae': valore, 'ci': intervallo, 'valid_samples': conteggio}}`

---

### prediction.py - Predizioni

Esegue predizioni su nuovi dati utilizzando un modello pre-addestrato.

**Funzione principale:** `make_predictions(dataset, opt, model)`

**Workflow:**
1. Carica modello pre-addestrato
2. Prepara dataset: split in predittori e cofattori
3. Crea oggetti Leaspy Data
4. Esegue `model.predict()` con range di timepoint
5. Salva predizioni su Datalake con metadati

**Parametri timepoint:**
- `--prediction_timepoints_start`: Età iniziale
- `--prediction_timepoints_end`: Età finale
- `--prediction_timepoints_step`: Passo temporale

---

### simulation.py - Generazione Dati Sintetici

Genera dataset sintetici mantenendo le proprietà statistiche dei dati reali.

**Funzione principale:** `simulate_data(model, dataset, opt)`

**Workflow:**
1. Split dataset (80/20 train/validation)
2. Prepara oggetti Leaspy Data
3. Estrae statistiche dataset (visite per soggetto, conteggio soggetti)
4. Genera dati sintetici con uno dei metodi:
   - `generate_virtual_data()` - Base
   - `generate_virtual_data_with_diagnosis()` - Con distribuzione diagnosi (CN: 30%, MCI: 45%, Dementia: 25%)
   - `generate_virtual_data_with_cofactors()` - Con combinazioni cofattori
5. Pulisce dati generati
6. Salva su Datalake:
   - Dataset sintetico
   - Dataset di validazione
7. Genera grafici confronto distribuzioni

---

### utils.py - Funzioni di Utilità

Collezione di funzioni helper per operazioni comuni.

**Funzioni principali:**

| Funzione | Descrizione |
|----------|-------------|
| `append_to_text_file()` | Aggiunge contenuto a file di testo |
| `make_json_calibration_settings_dict()` | Crea configurazione JSON per MCMC-SAEM |
| `split_dataset_in_train_val()` | Split 80/20 stratificato per numero visite |
| `load_dataset_from_datalake()` | Carica dataset via DatalakeClient |
| `load_dataset_to_datalake()` | Carica dataset su Datalake |
| `get_dataset_custom_metadata()` | Recupera metadati dal Datalake |
| `get_dataset_predictors()` | Ottiene nomi colonne predittori |
| `get_dataset_cofactors()` | Ottiene nomi colonne cofattori |
| `prepare_dataset()` | Rinomina colonne, filtra pazienti multi-visita |
| `normalize_fixed_scale()` | Normalizzazione min-max con bound fissi |
| `clean_generated_dataframe()` | Rimuove prefissi "generated_" e colonne metadata |

---

## Come Eseguire il Progetto

### Modalità 1: Training

Addestra un nuovo modello sui dati.

```bash
python code/main.py \
    --n_iter 3000 \
    --device cpu \
    --level cleaned_02 \
    --source_dimension 3 \
    --model_type logistic \
    --file_code ADNIMERGE
```

**Parametri chiave:**
- `--n_iter`: Numero iterazioni MCMC-SAEM (default: 3000)
- `--device`: cpu o cuda per GPU
- `--level`: Livello dati nel Datalake
- `--source_dimension`: Dimensioni variabilità inter-soggetto (default: 3)

**Output generato:**
- `results/experiment_{timestamp}/logs/` - Log di training
- `results/experiment_{timestamp}/weights/` - Parametri modello
- `results/experiment_{timestamp}/images/` - Grafici

---

### Modalità 2: Prediction

Genera predizioni usando un modello pre-addestrato.

```bash
python code/main.py \
    --prediction True \
    --prediction_timepoints_start 75 \
    --prediction_timepoints_end 80 \
    --prediction_timepoints_step 1 \
    --device cpu \
    --level cleaned_03 \
    --file_code ADNIMERGE
```

**Parametri chiave:**
- `--prediction True`: Abilita modalità predizione
- `--prediction_timepoints_start`: Età iniziale per predizioni
- `--prediction_timepoints_end`: Età finale per predizioni
- `--prediction_timepoints_step`: Intervallo temporale

**Output generato:**
- Predizioni salvate su Datalake con metadati

---

### Modalità 3: Simulation

Genera dataset sintetici con proprietà statistiche controllate.

```bash
python code/main.py \
    --simulation True \
    --level cleaned_03 \
    --file_code ADNIMERGE \
    --same_data_stats 1 \
    --n_gen_sub 1000 \
    --n_gen_visit 10
```

**Parametri chiave:**
- `--simulation True`: Abilita modalità simulazione
- `--same_data_stats`: Usa statistiche dei dati originali (1 = sì)
- `--n_gen_sub`: Numero soggetti da generare
- `--n_gen_visit`: Numero medio visite per soggetto

**Output generato:**
- Dataset sintetico su Datalake
- Dataset di validazione su Datalake
- Grafici confronto distribuzioni (istogrammi e CDF)

---

### Aiuto da Linea di Comando

Per visualizzare tutti gli argomenti disponibili:

```bash
python code/main.py --help
```

---

## Argomenti da Linea di Comando

| Argomento | Tipo | Default | Descrizione |
|-----------|------|---------|-------------|
| `--source_dimension` | int | 3 | Dimensioni per variabilità inter-soggetto non temporale |
| `--model_type` | str | logistic | Tipo modello: logistic, logistic_parallel, univariate_logistic |
| `--n_iter` | int | 3000 | Numero iterazioni algoritmo MCMC-SAEM |
| `--device` | str | cpu | Dispositivo: cpu o cuda |
| `--prediction` | bool | False | Abilita modalità predizione |
| `--simulation` | bool | False | Abilita modalità simulazione |
| `--level` | str | - | Livello metadati nel Datalake (es. cleaned_02, cleaned_03) |
| `--file_code` | str | - | Codice identificativo dataset (es. ADNIMERGE) |
| `--prediction_timepoints_start` | float | - | Età iniziale per predizioni |
| `--prediction_timepoints_end` | float | - | Età finale per predizioni |
| `--prediction_timepoints_step` | float | - | Passo temporale per predizioni |
| `--same_data_stats` | int | 0 | Usa statistiche dati originali per generazione |
| `--n_gen_sub` | int | - | Numero soggetti sintetici da generare |
| `--n_gen_visit` | int | - | Numero medio visite per soggetto sintetico |
| `--merge_generations` | bool | False | Combina generazioni in unico dataset |
| `--sub_data` | bool | False | Usa subset di dati per testing |
| `--sub_n` | int | - | Numero soggetti nel subset |

---

## Struttura Cartelle Generate

Durante l'esecuzione, vengono create le seguenti directory:

```
results/
└── experiment_{sub_n}_{n_iter}_{mese}_{giorno}_{ora}_{minuto}/
    ├── logs/                               # Log e metriche
    │   ├── train_noise.txt                 # Rumore stimato per feature
    │   ├── validation_results.txt          # MAE ± CI per variabile
    │   ├── calibration_parameters.csv      # Log iterazioni MCMC
    │   └── convergence_[var].png           # Grafici convergenza
    │
    ├── images/                             # Visualizzazioni
    │   ├── mean_curves.png                 # Traiettorie medie popolazione
    │   ├── distribution_hist_[var].png     # Istogrammi reale vs sintetico
    │   ├── CDF_hist_[var].png              # CDF reale vs sintetico
    │   └── predictions/
    │       └── individual_curves_[id].png  # Curve individuali
    │
    └── weights/                            # Parametri modello
        ├── model_parameters.json           # Modello addestrato
        └── average_parameters.json         # Parametri paziente medio
```

---

## Esempi di Output

### train_noise.txt

Contiene le stime del rumore percentuale per ogni feature:

```
MMSE: 12.34%
Memory: 8.56%
Language: 5.23%
Concentration: 7.89%
Praxis: 6.12%
CDRSB: 9.45%
ADAS13: 11.23%
FAQ: 7.67%
RAVLT_immediate: 10.01%
Hippocampus%ICV: 4.56%
Ventricles%ICV: 5.78%
```

---

### validation_results.txt

Contiene le metriche di validazione per ogni variabile:

```
MMSE - MAE: 2.3456 ± 0.123 (su 1250 campioni validi)
Memory - MAE: 1.2345 ± 0.087 (su 980 campioni validi)
Language - MAE: 0.8901 ± 0.056 (su 1180 campioni validi)
Concentration - MAE: 1.0234 ± 0.067 (su 1150 campioni validi)
Praxis - MAE: 0.7890 ± 0.045 (su 1200 campioni validi)
CDRSB - MAE: 1.5678 ± 0.098 (su 1100 campioni validi)
ADAS13 - MAE: 3.4567 ± 0.234 (su 950 campioni validi)
FAQ - MAE: 2.1234 ± 0.156 (su 1050 campioni validi)
RAVLT_immediate - MAE: 4.5678 ± 0.312 (su 900 campioni validi)
Hippocampus%ICV - MAE: 0.0123 ± 0.001 (su 850 campioni validi)
Ventricles%ICV - MAE: 0.0234 ± 0.002 (su 870 campioni validi)
```

---

### Output Console - Training

```
================================================================================
                           AD COURSE MAP - TRAINING
================================================================================

Initializing model...
  Model type: logistic
  Source dimension: 3
  Device: cpu

Loading data from datalake...
  Level: cleaned_02
  File code: ADNIMERGE
  Loaded 2345 subjects with 12456 visits

Preprocessing data...
  Applying fixed-scale normalization
  Filtering multi-visit patients
  Final dataset: 1890 subjects, 9876 visits

Splitting dataset...
  Train: 1512 subjects (80%)
  Validation: 378 subjects (20%)

Creating Leaspy Data objects...
  Predictors: 11 features
  Cofactors: GENDER, APOE4, DX

Starting MCMC-SAEM optimization (3000 iterations)...
  Iteration 500/3000  - Log-likelihood: -2456.78
  Iteration 1000/3000 - Log-likelihood: -1834.56
  Iteration 1500/3000 - Log-likelihood: -1456.23
  Iteration 2000/3000 - Log-likelihood: -1289.45
  Iteration 2500/3000 - Log-likelihood: -1178.90
  Iteration 3000/3000 - Log-likelihood: -1123.67

Saving model parameters...
  Saved to: results/experiment_1890_3000_12_05_14_30/weights/model_parameters.json

Generating population curves...
  Saved to: results/experiment_1890_3000_12_05_14_30/images/mean_curves.png

Validating on held-out data...
  Personalizing model to 378 subjects...
  Generating predictions...
  Calculating MAE metrics...

Validation Results:
  MMSE:             MAE = 2.35 ± 0.12
  Memory:           MAE = 1.23 ± 0.09
  Language:         MAE = 0.89 ± 0.06
  Hippocampus%ICV:  MAE = 0.01 ± 0.00

Saving validation results...
  Saved to: results/experiment_1890_3000_12_05_14_30/logs/validation_results.txt

================================================================================
                              TRAINING COMPLETE
================================================================================
```

---

### Output Console - Simulation

```
================================================================================
                          AD COURSE MAP - SIMULATION
================================================================================

Loading pretrained model...
  Model path: saved_models/model_parameters.json

Loading source data...
  Level: cleaned_03
  Loaded 2345 subjects

Extracting dataset statistics...
  Mean visits per subject: 4.2
  Std visits per subject: 2.1
  Total subjects: 2345

Generating synthetic data with diagnosis distribution...
  CN (Cognitively Normal): 30%
  MCI (Mild Cognitive Impairment): 45%
  Dementia: 25%

Generating 1000 synthetic subjects...
  Progress: [##########] 100%

Cleaning generated data...
  Removed 'generated_' prefixes
  Removed metadata columns

Saving to datalake...
  Synthetic dataset: synthetic_data_generation_utility.csv
  Validation dataset: dataset_to_test_utility.csv

Generating distribution comparison plots...
  Saved histograms to: images/distribution_hist_*.png
  Saved CDFs to: images/CDF_hist_*.png

================================================================================
                             SIMULATION COMPLETE
================================================================================
```

---

## Dataset e Biomarker

### Predittori (Biomarker)

Il modello utilizza i seguenti biomarker come variabili predittive:

**Score Cognitivi:**
| Biomarker | Descrizione |
|-----------|-------------|
| MMSE | Mini-Mental State Examination |
| Memory | Score memoria |
| Language | Score linguaggio |
| Concentration | Score concentrazione |
| Praxis | Score prassia |
| CDRSB | Clinical Dementia Rating Sum of Boxes |
| ADAS11 | Alzheimer's Disease Assessment Scale (11 items) |
| ADAS13 | Alzheimer's Disease Assessment Scale (13 items) |
| FAQ | Functional Activities Questionnaire |
| RAVLT_immediate | Rey Auditory Verbal Learning Test (immediato) |

**Imaging Cerebrale (normalizzati per ICV):**
| Biomarker | Descrizione |
|-----------|-------------|
| Ventricles%ICV | Volume ventricoli / Volume intracranico |
| Hippocampus%ICV | Volume ippocampo / Volume intracranico |
| Entorhinal%ICV | Volume corteccia entorinale / Volume intracranico |
| Fusiform%ICV | Volume giro fusiforme / Volume intracranico |
| MidTemp%ICV | Volume lobo temporale medio / Volume intracranico |

---

### Cofattori (Caratteristiche Paziente)

| Cofattore | Descrizione | Valori |
|-----------|-------------|--------|
| GENDER | Sesso del paziente | Male, Female |
| APOE4 | Status Apolipoproteina E4 | 0, 1, 2 (numero alleli) |
| DX | Diagnosi | CN (Cognitively Normal), MCI (Mild Cognitive Impairment), Dementia |

---

### Formato Dati Input

Il dataset deve essere in formato CSV con le seguenti colonne obbligatorie:

```csv
RID,AGE,MMSE,Memory,Language,Concentration,Praxis,CDRSB,...,GENDER,APOE4,DX
001,65.3,28,0.85,0.92,0.78,0.88,1.5,...,Male,1,CN
001,66.8,27,0.82,0.89,0.75,0.85,2.0,...,Male,1,MCI
002,72.1,24,0.65,0.70,0.60,0.72,4.5,...,Female,2,MCI
...
```

**Note:**
- `RID` viene rinominato in `ID` durante il preprocessing
- `AGE` viene rinominato in `TIME` (età alla visita)
- Sono richieste almeno 2 visite per paziente

---

## Esecuzione su Cluster HPC

Per eseguire su cluster HPC con SLURM, utilizzare gli script nella cartella `bash/`:

### Training Job

```bash
sbatch bash/train_job.sh
```

Contenuto tipico di `train_job.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=adcm_train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

module load python/3.10
source activate adcoursemap

python code/main.py \
    --n_iter 20000 \
    --device cpu \
    --level cleaned_03 \
    --source_dimension 3
```

### Prediction Job

```bash
sbatch bash/predict_job.sh
```

### Simulation Job

```bash
sbatch bash/simulate_job.sh
```

---

## Riferimenti Bibliografici

### Framework Matematico

- Schiratti, J., Allassonniere, S., Colliot, O., & Durrleman, S. (2016). **A Bayesian mixed-effects model to learn trajectories of changes from repeated manifold-valued observations.** [https://hal.science/hal-01540367v1](https://hal.science/hal-01540367v1)

- Schiratti, J., Allassonniere, S., Colliot, O., & Durrleman, S. (2015). **Learning spatiotemporal trajectories from manifold-valued longitudinal data.** [https://dumas.ccsd.cnrs.fr/TDS-MACS/hal-01163373v1](https://dumas.ccsd.cnrs.fr/TDS-MACS/hal-01163373v1)

### Applicazione Alzheimer

- Koval, I., Bône, A., Louis, M., Lartigue, T., Bottani, S., Marcoux, A., Samper-González, J., Burgos, N., Charlier, B., Bertrand, A., Epelbaum, S., Colliot, O., Allassonnière, S., & Durrleman, S. (2021). **AD Course Map charts Alzheimer's disease progression.** Scientific Reports, 11(1). [https://doi.org/10.1038/s41598-021-87434-1](https://doi.org/10.1038/s41598-021-87434-1)

---

## Licenza

*Da definire*

---

## Contatti

*Da definire*
