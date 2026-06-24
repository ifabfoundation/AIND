# DX_calculators/rules — Metodo B: diagnosi basata su regole cliniche (NIA-AA 2018)

## Scopo

Assegna una diagnosi (`CN` / `MCI` / `Dementia`) a ogni visita di un paziente
usando un sistema di regole derivato dai criteri NIA-AA 2018 (Jack et al.) e
dalla cascata biomarker di Jack 2013.

Questo modulo è il **Metodo B** nella strategia diagnostica di AIND WP-3:

- È il riferimento normativo ("gold standard" clinico) con cui validare il Metodo A (ML).
- Soddisfa per costruzione il criterio FDA di *Consistency* (Zamzmi et al. 2025).
- Non dipende da training data → nessun overfitting, applicabile a qualsiasi dataset ADNI-like.

---

## Perché non basta MMSE

MMSE da solo ha limitazioni note:
- **Effetto soffitto** nei soggetti CN con alta scolarità (falsi negativi MCI precoce).
- **Effetto pavimento** nei pazienti con Demenza grave (non discrimina gradi di severità).
- Non misura il **dominio funzionale** (ADL), che è requisito necessario per la diagnosi di Demenza secondo NIA-AA.

Il Metodo B usa un **voto pesato multi-dominio** che integra staging funzionale,
cognizione globale, memoria episodica e, quando disponibili, neuroimaging.

---

## File

| File | Contenuto |
|---|---|
| `thresholds.py` | Soglie cliniche per ogni biomarker, pesi dei domini, regole di consistenza temporale |
| `dx_rule_based.py` | Funzioni principali: `assign_dx_rule_based()` (singola visita) e `assign_dx_batch()` (DataFrame) |
| `__init__.py` | Esporta `assign_dx_rule_based`, `assign_dx_batch`, `DxResult` |

---

## Domini e pesi

| Dominio | Biomarker | Peso | Riferimento |
|---|---|---|---|
| Staging funzionale | `CDRSB` | 3.0 | O'Bryant 2010 / NIA-AA 2018 |
| Funzionale ADL | `FAQ` | 2.0 | Pfeffer 1982 |
| Cognitivo globale | `MMSE` | 2.0 | Folstein 1975 |
| Cognitivo sensibile | `ADAS13` | 1.5 | Mohs 1997 |
| Memoria episodica | `RAVLT_immediate` | 1.0 | Schmidt 2015 |
| Neuroimaging | `Hippocampus`, `Entorhinal`, `MidTemp` | 0.5 cad. | Jack 2010 |

Tutti i campi sono **opzionali**: i domini mancanti vengono ignorati e segnalati
in `DxResult.missing_domains`. Se nessun dominio è disponibile, `dx = 'Unknown'`.

### Confidence

| Livello | Condizione |
|---|---|
| `high` | `CDRSB` disponibile **e** > 75% del peso totale concorda |
| `medium` | `CDRSB` disponibile **oppure** accordo tra 60% e 75% |
| `low` | `CDRSB` mancante **e** accordo < 60%, o un solo dominio disponibile |

---

## Uso

### Singola visita

```python
from DX_calculators.rules import assign_dx_rule_based

result = assign_dx_rule_based(
    visit_data={
        "MMSE": 22,
        "CDRSB": 2.5,
        "FAQ": 3,
        "ADAS13": 18,
        "RAVLT_immediate": 32,
    },
    prev_dx="CN",              # diagnosi visita precedente (None se baseline)
    time_since_prev_years=1.5, # anni dall'ultima visita (None se sconosciuto)
)

print(result.dx)           # 'MCI'
print(result.confidence)   # 'high'
print(result.evidence)     # {'CDRSB': 'MCI', 'FAQ': 'MCI', 'MMSE': 'MCI', ...}
print(result.temporal_consistent)  # True
```

### Batch su DataFrame ADNI (da Datalake)

```python
import pandas as pd
from dl_client import DatalakeClient
from DX_calculators.rules import assign_dx_batch

client = DatalakeClient()
search = client.query_files({"custom.level": "cleaned_03", "custom.file_code": "ADNIMERGE"})
zip_files = client.download_file(search["object_name"], extract_zip=True)
df = zip_files[list(zip_files.keys())[0]]

# Il DataFrame deve contenere ID, TIME, e i biomarker in scala clinica
df_with_dx = assign_dx_batch(df, id_col="ID", time_col="TIME")

# Colonne aggiunte:
# DX_rules                     → 'CN' / 'MCI' / 'Dementia' / 'Unknown'
# DX_rules_confidence          → 'high' / 'medium' / 'low'
# DX_rules_temporal_consistent → bool
# DX_rules_temporal_flag       → stringa descrittiva o NaN
# DX_rules_missing_domains     → lista dei biomarker mancanti per quella visita
```

### Confronto con DX originale ADNI (validazione)

```python
# Accuratezza sul dato reale
from sklearn.metrics import classification_report

mask = df_with_dx["DX_rules"] != "Unknown"
print(classification_report(
    df_with_dx.loc[mask, "DX"],        # etichetta originale ADNI
    df_with_dx.loc[mask, "DX_rules"],  # diagnosi calcolata dal Metodo B
))
```

---

## Consistenza temporale

Il Metodo B non modifica la diagnosi basata sui biomarker in presenza di
transizioni anomale, ma aggiunge un flag di qualità per uso downstream:

| Transizione | Flag |
|---|---|
| `Dementia → CN` | Implausibile — impossibile senza intervento |
| `Dementia → MCI` | Implausibile — regressione funzionale biologicamente assente |
| `MCI → CN` | Inatteso — possibile ma raro; segnalato per revisione |
| `CN → Dementia` (< 2 anni, senza MCI intermedio) | Sospetto — salto di stadio troppo rapido |

Il flag `temporal_consistent = False` è la metrica di qualità da usare nel
confronto A vs B (se Metodo A assegna una DX coerente ma Metodo B la flagga
come temporalmente implausibile, il record è candidato all'esclusione).

---

## Riferimenti

| Paper | DOI | Rilevanza |
|---|---|---|
| Jack et al. 2013, Lancet Neurology | `10.1016/S1474-4422(12)70291-0` | Cascata biomarker — vincolo normativo |
| Jack et al. 2018, Alzheimer's & Dementia | NIA-AA 2018 | Framework A/T/(N) aggiornato |
| O'Bryant et al. 2010, Arch. Neurology | — | CDR-SB cutpoints validati su ADNI |
| Zamzmi et al. 2025, Comm. Engineering | `10.1038/s44172-025-00450-1` | FDA Consistency criterion |
| Petersen et al. 2016, JAMA | — | Storia naturale MCI e regressione |

---

## Limitazioni note

- Le soglie di neuroimaging (`Hippocampus`, `Entorhinal`, `MidTemp`) dipendono
  dall'unità di misura del datalake level usato. Verificare che i valori siano
  in mm³ (volume) o mm (spessore corticale) prima di interpretare il voto di
  quel dominio. Se l'unità è diversa, aggiornare le soglie in `thresholds.py`.
- Il range MCI è clinicamente eterogeneo. I casi borderline CN/MCI con
  confidence `low` sono candidati all'analisi SuStaIn (Young 2018) per uno
  staging biologico più preciso.
- La proporzione CN/MCI/Dementia risultante non è controllata: dipende dalla
  distribuzione dei biomarker nel dataset. Per il dato sintetico, confrontarla
  con il target (30/45/25%) dopo l'applicazione.
