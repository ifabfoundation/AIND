# DX_calculators/rules

Verisone 06/07/2026 - Chara & Claude

Sistema di classificazione diagnostica per pazienti ADNI e ADNI-like (sintetici). Assegna tre diagnosi indipendenti a partire dagli stessi dati di visita, secondo la decisione di design in [`docs/adni_diagnosis_classifier_brief.md`](docs/adni_diagnosis_classifier_brief.md): due assi separati (clinico e biologico) che non si sovrascrivono mai a vicenda, più uno stadio combinato che li unisce senza sostituirli.

> Questo file va aggiornato ad ogni cambiamento strutturale (nuovo modulo, nuova colonna di output, nuovo flag, modifica alle soglie in `config.py`). Non serve aggiornarlo per bugfix interni che non cambiano l'interfaccia pubblica.

## Le tre diagnosi

| # | Nome | Framework | Modulo | Output |
|---|---|---|---|---|
| 1 | Clinica pura | NIA-AA 2011 (McKhann/Albert/Sperling) | `dx1_nia_clinical.py` | `DX1_clinical` (CN/MCI/Dementia) |
| 2 | Biologica pura (ATN) | NIA-AA 2018 (Jack et al.) | `dx2_nia_atn.py` | `DX2_A`, `DX2_T`, `DX2_N` (booleani), `DX2_ATN_label` |
| 3 | Combinata clinico-biologica | NIA-AA 2024, 6 stadi (Jack et al.) | `dx3_nia_combined.py` | `DX3_stage` (1-6 o `None`), `DX3_label` |

La Diagnosi 2 non riceve mai la Diagnosi 1 come input (e viceversa): sono assi indipendenti. La Diagnosi 3 li combina *dopo* che sono stati calcolati entrambi, senza modificarli.

## File

```
config.py                single source of truth per tutte le soglie (memoria, MMSE, CDR, FAQ,
                          CSF, PET amiloide/tau, volumi, staging NIA-AA 2024, registro flag)
nia_clinical_resolver.py helper di risoluzione riga-per-riga (banda educazione, test
                          memoria Plan A/B, gate MMSE/FAQ, severità CDRSB) — nessuna decisione
dx1_nia_clinical.py      Diagnosi 1 — gerarchia decisionale CN/MCI/Dementia
dx2_nia_atn.py           Diagnosi 2 — stato A/T/N dai biomarcatori
dx3_nia_combined.py      Diagnosi 3 — staging NIA-AA 2024 da Diagnosi 1 + Diagnosi 2
__init__.py              esporta l'API pubblica di tutti i moduli (nuovi + legacy)
docs/                    documenti di ragionamento che hanno guidato il design (brief,
                          ricerca su come ADNI assegna la diagnosi, linee guida cliniche)
legacy/                  sistema precedente ("Metodo B", voto pesato) — vedi legacy/README.md
```

## Uso

```python
from rules import assign_dx1_batch, assign_dx2_batch, assign_dx3_batch

df = assign_dx1_batch(df)   # aggiunge DX1_*
df = assign_dx2_batch(df, metadata=file_metadata)   # aggiunge DX2_*, metadata.custom opzionale
df = assign_dx3_batch(df)   # richiede le colonne DX1_*/DX2_* già presenti — aggiunge DX3_*
```

L'ordine conta solo per `assign_dx3_batch`, che legge `DX1_clinical`, `DX2_A`, `DX2_T` dal DataFrame. `assign_dx1_batch` e `assign_dx2_batch` sono indipendenti e possono girare in qualsiasi ordine (o in parallelo su copie separate, poi unite su `id_col`/`time_col`).

Per una singola riga, le versioni non-batch sono `classify_syndromic()`, `classify_atn()`, `combine_stage()`.

## Metodo (assay/tracer/versione FreeSurfer)

Ogni funzione batch/riga accetta un parametro esplicito (`csf_assay=`, `amyloid_pet_tracer=`, `tau_pet_tracer=`, `volume_scale=`, `freesurfer_version=`). Se omesso, viene letto da `metadata.custom` del file (`CSF_filter`/`PET_filter`/`VOLUMES_filter`, convenzione definita in `WP-2/Data_cleaning/automated_merge.py`); se anche quello manca, si usa `config.SYNTHETIC_DEFAULTS`. La provenienza di ogni scelta è riportata in `DX2_method_source`.

Oggi i dataset mergiati portano un solo metodo per file (non per riga) — `DX2_method_uniform` è sempre `True`; se in futuro un dataset dovesse mischiare metodi nello stesso file, andrà aggiunta una risoluzione per riga (vedi TODO sotto).

## Flag di qualità

Ogni funzione batch aggiunge una colonna `*_flags` (es. `DX1_flags`, `DX2_flags`) con zero o più chiavi dal registro `config.FLAGS`, unite con `config.FLAG_JOIN_SEP` (`"|"`). Il registro documenta il significato di ogni flag — consultarlo prima di aggiungerne uno nuovo, per evitare duplicati.

## Stato / TODO noti

- **Soglie %ICV per FreeSurfer 4.3, 4.4, 6.0, 7.4.1** non sono validate indipendentemente — nessuna letteratura versione-specifica esiste, e nessun fattore di conversione affidabile da 5.1 esiste (Gronenschild et al. 2012, PLoS One: il bias cross-versione è struttura-dipendente, non sistematico). `dx2_nia_atn.py`/`config.py` le popolano come **proxy esplicito dei cutoff "5.1"** (`is_proxy: True`), con flag `FS_VERSION_CUTOFF_PROXY_FROM_5.1` emesso in `DX2_flags` invece di restituire `N=None` silenziosamente. La derivazione empirica di cutoff propri per versione (es. percentili sul campione CN di quella versione) resta un TODO separato, non pianificato.
- **`CDGLOBAL` (CDR Global) spesso mancante nel dataset di merge, mentre `CDRSB` è quasi sempre presente** — non è un vero gap di raccolta ADNI, è un bug di mapping nella pipeline `WP-2/Data_cleaning/` (dettagli in `STATUS_DIAGNOSI_PIPELINE_SINTETICA.md`, sezione 8bis: `CDGLOBAL` viene scartato silenziosamente dal filtro per categoria perché non è mai stato rinominato in `CDRGLOB`). Il fix corretto è a monte, in WP-2; nel frattempo `dx1_nia_clinical.py` (`classify_syndromic`) usa un fallback esplicito: quando `CDGLOBAL` manca ma `CDRSB` è disponibile, deriva un CDR Global approssimato via il crosswalk `config.CDRSB_TO_CDRGLOBAL` (O'Bryant et al. 2008, Arch Neurology — κ=0.90, 93-94% classificati correttamente in due studi indipendenti), con flag `CDGLOBAL_DERIVED_FROM_CDRSB` in `DX1_flags` per restare distinguibile da un valore osservato.
- **Diagnosi 1 non dipende più dal protocollo/fase ADNI** (risolto — era una discussione aperta in una versione precedente di questo file). `MEMORY_IMPAIRMENT_CUTOFFS` e `MMSE_GATES` in `config.py` sono ora un riferimento unico, non più selezionato per fase (`ORIGPROT`/`ADNI1`/`ADNIGO2`/`ADNI3`/`ADNI4`): i criteri NIA-AA 2011 (Albert et al. 2011, doi:10.1016/j.jalz.2011.03.008) definiscono l'impairment di memoria come soglia statistica generica ("1-1.5 SD sotto la norma età/educazione"), non un cutoff fase-specifico, quindi la selezione per fase non era mai stata un requisito NIA-AA. Per lo stesso motivo `EMCI`/`LMCI` (terminologia di arruolamento ADNI-GO/ADNI2, non categoria NIA-AA) è stato rimosso, insieme a `resolve_protocol()` e a `SMC` (mai calcolato, era comunque specifico di ADNIGO2). **Stato: provvisorio** — i valori di `MEMORY_IMPAIRMENT_CUTOFFS`/`MMSE_GATES` restano quelli del cohort ADNI3/4 (Aisen et al. 2024), solo scorporati dalla logica di fase; non sono (ancora) tabelle normative NIA-AA indipendenti da ADNI. Raffinamento futuro possibile: sostituirli con norme età-corrette pubblicate indipendentemente da ADNI (es. MOANS per Logical Memory II, Mayo/Schmidt per RAVLT) e una banda di gravità MMSE non-ADNI (es. Perneczky et al. 2006), il che richiederebbe aggiungere `AGE` come nuovo input (oggi non usato in `rules/`, pur essendo disponibile a monte nella pipeline di merge come cofattore).
- **Nessuna chiave metadata dedicata per il tracciante tau** (`PET_filter` copre solo l'amiloide) — va risolto via parametro esplicito o default finché non compare una convenzione nei dati.
- Il confronto sistematico Diagnosi 1 vs `legacy/dx_rule_based.py` ("Metodo A vs B", vedi `STATUS_DIAGNOSI_PIPELINE_SINTETICA.md`) non è ancora automatizzato — oggi esiste solo come controllo di sanità manuale nello script di verifica.

## Verifica

Non esiste ancora una suite di test formale (`pytest`) in questa cartella. La verifica va fatta con un piccolo script che importa `rules` e controlla i casi noti da `docs/adni_diagnosis_classifier_brief.md` (soglie di confine, righe CN/MCI/Dementia costruite a mano, stadi NIA-AA 2024 noti) — vedi la cronologia del progetto per un esempio completo dei casi da coprire. Ambiente consigliato: venv `aind101/` (creato per WSL/Linux — vedi nota sotto).

**Nota ambiente:** l'interprete Python su questa macchina è funzionante solo via WSL (Ubuntu). Da PowerShell/Git Bash nativi, `python`/`python3` sono solo stub del Microsoft Store. Per eseguire codice:
```bash
wsl
cd "/mnt/c/Users/<utente>/OneDrive - Net Service S.p.A/Documenti/Github/AIND"
source aind101/bin/activate
python -m pip install -r <requirements se servono>
```
