# Data_cleaning — Stato attuale della cartella

> Questo README fotografa lo stato **attuale** (pre-refactoring) della cartella, per orientarsi tra i file esistenti.
> Non è il README "finale" previsto da [CLAUDE.md](CLAUDE.md) (che verrà scritto a refactoring completato), ma un supporto per capire cosa c'è, a cosa serve, in che ordine studiarlo e cosa è probabilmente scartabile.
> La cartella `post-generation/` (pipeline indipendente, già documentata con proprio README) **non è trattata qui**.
>
> Fonti usate: [CLAUDE.md](CLAUDE.md), [PROJECT_MAP.md](PROJECT_MAP.md), `audit_call_counts.txt`, `audit_function_names.txt`, `audit_functions_raw.txt`, lettura diretta di tutti i notebook e moduli citati nella sezione 5.

---

## ⚠️ Nota di sicurezza

`.dl_client.ini` contiene le **credenziali in chiaro** (username/password) per l'accesso al datalake ed **è attualmente tracciato da git** (non è in `.gitignore`, risulta tra i file versionati). `PROJECT_MAP.md` lo segnalava già come "da verificare che non venga mai committato" — la verifica ha confermato che il rischio si è concretizzato. Da valutare con priorità: rotazione delle credenziali e rimozione del file dalla history di git.

Nota per chi non ha accesso al datalake (es. stagista): questo file serve solo a far girare i notebook contro i dati reali. Il codice si può leggere e studiare per intero senza di esso — semplicemente le celle che chiamano `DatalakeClient()` non saranno eseguibili.

---

## 1. Cosa fa questo progetto

Tre pipeline in sequenza (dettagli in [CLAUDE.md](CLAUDE.md)):

1. **Cleaning** — pulizia/normalizzazione dei dataset ADNI grezzi (notebook `adni_cleaning*.ipynb`)
2. **Merge** — unione e riconciliazione dei dataset puliti per categoria (`merge_category_wise.ipynb` + `automated_merge.py`)
3. **Post-generation** *(fuori scope di questo README)* — post-processing dei dati sintetici

I dati non sono nel repo: si leggono/scrivono da un **datalake esterno** tramite il client proprietario `dl_client`. ADNI (Alzheimer's Disease Neuroimaging Initiative) è lo studio clinico da cui provengono i dati: soggetti seguiti nel tempo con visite ripetute, su cui vengono raccolte scale cognitive, biomarcatori (liquor, plasma, PET) e immagini di risonanza magnetica (volumi cerebrali).

---

## 2. Struttura attuale (non ancora quella target di CLAUDE.md)

```
Data_cleaning/
│
├── adni_cleaning{1,2,3,4}.ipynb          # pipeline cleaning, entry point (ATTIVI)
├── adni_cleaning{1,2,3,4}_bb.ipynb       # varianti "_bb" — vedi §6, da chiarire con l'autore
│
├── merge_category_wise.ipynb             # merge fase 1 (entry point ATTIVO)
├── automated_merge.py                    # merge fase 2 (entry point ATTIVO)
├── final_merge_categories.ipynb          # merge fase 2, alternativo ad automated_merge.py
├── analyze_merge_statistics.ipynb        # diagnostica post-merge (opzionale, non è entry point)
├── merge_categories_with_filters.ipynb   # bozza incompleta — vedi §6
│
├── data_model/
│   ├── DataCleaner.py                    # classe con le funzioni di pulizia usate dai notebook cleaning
│   ├── MergerTools.py                    # classe con le funzioni di merge/riconciliazione
│   └── manage_excel_support_file.py      # funzioni per gli excel di tracciamento variabili
│
├── documents/                            # documentazione varia, no codice — vedi §6 per i dubbi
├── test_notebooks/                       # notebook esplorativi "creati e poi non molto usati" — vedi §6
├── da_cestinare/                         # cartelle vuote, letteralmente "da cestinare" — vedi §6
│
├── *.json                                # file di configurazione (cutoff, normalizzazione, range volumi...)
├── ADNI_variables_cleaned*.csv/.xlsx     # output/tracking di ciascuno step di cleaning ("support file", vedi §5.6)
├── ADNI_variables_statistics.csv/.xlsx   # tracking sul dataset grezzo
├── merge_statistics.csv                  # statistiche sui merge finali — origine non chiara, vedi §6
│
├── .dl_client.ini                        # credenziali datalake — ⚠️ vedi nota sicurezza sopra
├── prove_download.ipynb                  # notebook scratch — vedi §6
├── scan_imports.ipynb                    # notebook temporaneo per audit import — vedi §6
├── finalize_synthetic_file.ipynb         # non documentato in PROJECT_MAP — vedi §6
├── fix_dummies_submerge.ipynb            # fix mirato post-segnalazione (Bari) — utility one-off
├── fix_merge_comb.ipynb                  # fix ricalcolo età / pulizia righe post-merge — utility one-off
├── update_metadata_merge.ipynb           # utility per aggiornare metadati di file già mergati
│
├── CLAUDE.md                             # regole operative e struttura target del refactoring
├── PROJECT_MAP.md                        # mappa dettagliata file-per-file (più granulare di questo README)
└── audit_*.txt                           # output di un audit automatico delle funzioni definite nel progetto
```

A differenza della struttura target descritta in CLAUDE.md (`pipeline/`, `utils/helpers.py`, `notebooks/exploration|validation`, `tests/`), oggi **tutto è appiattito nella root**: notebook di produzione, notebook di prova e file di documentazione convivono senza separazione. Questo è esattamente il problema che il refactoring (Fasi 2-6 in CLAUDE.md, ancora da fare) dovrà risolvere.

---

## 3. Flusso dati (entry point attivi)

```
adni_cleaning1.ipynb → cleaned_01 → adni_cleaning2.ipynb → cleaned_02 → adni_cleaning3.ipynb → cleaned_03
      (per singolo file)                                ↘
                                                           adni_cleaning4.ipynb → cleaned_04
                                                                                       ↓
                                                                     merge_category_wise.ipynb  (per categoria: vol, scale, csf, plasma, pet, cofattori)
                                                                                       ↓
                                                             automated_merge.py  /  final_merge_categories.ipynb
                                                                                       ↓
                                                                   (opzionale) analyze_merge_statistics.ipynb
```

Le funzioni condivise vivono in `data_model/DataCleaner.py` (cleaning), `data_model/MergerTools.py` (merge/riconciliazione) e `data_model/manage_excel_support_file.py` (tracking excel), richiamate dai notebook sopra. La pipeline **si biforca dopo `adni_cleaning1`**: `cleaning2 → cleaning3` produce un file pulito pronto per essere usato da solo (con righe/soggetti scarsi rimossi), mentre `cleaning4` produce un file "meno pulito" ma con tutte le righe intatte, pensato per alimentare il merge (vedi §5.4 per il perché).

---

## 4. Guida di lettura per chi arriva nuovo (nessun accesso al dataset richiesto)

Il codice si può capire leggendolo, senza eseguirlo. Ordine consigliato — ogni riga della tabella presuppone le precedenti:

| # | Cosa leggere | Perché in questo punto |
|---|---|---|
| 1 | [CLAUDE.md](CLAUDE.md) | Contesto generale, regole di lavoro sul progetto, struttura target del refactoring. |
| 2 | Questo README, §1-3 | Visione d'insieme: cosa fa il progetto, come sono organizzati i file oggi, il flusso dati principale. |
| 3 | `data_model/DataCleaner.py` — solo le **firme e i docstring** dei metodi, non tutta la logica (usare §5.5 sotto come mappa) | Serve come "glossario": i notebook del punto successivo chiamano continuamente questi metodi per nome, meglio sapere già a grandi linee cosa fanno prima di vederli usati. |
| 4 | `adni_cleaning1.ipynb` — leggere per intero **una sola sezione file-specifica** (es. quella di `ADNIMERGE`, la più ricca) invece di scorrere tutte le ~25 sezioni ripetitive | È il notebook più lungo (~5600 righe) ma segue un pattern che si ripete quasi identico per ogni file grezzo — vedi §5.1. Capito il pattern su un file, il resto è variazione sul tema. |
| 5 | `adni_cleaning2.ipynb` (breve, ~300 righe) | Secondo step generico: qui si decide quali variabili/soggetti scartare e si creano le dummy. |
| 6 | `adni_cleaning3.ipynb` **e** `adni_cleaning4.ipynb` insieme, confrontandoli | Qui la pipeline si biforca (uso singolo del file vs. preparazione al merge) — leggerli insieme rende chiaro il perché della biforcazione, vedi §5.4. |
| 7 | `data_model/manage_excel_support_file.py` | Spiega cosa sono i file Excel "di supporto" (`ADNI_variables_cleaned*.xlsx`) che tutti i notebook di cleaning leggono/scrivono continuamente — senza questo pezzo molte righe di codice dei notebook sembrano magia. |
| 8 | `merge_category_wise.ipynb` | Prima fase del merge: unisce i file `cleaned_04` in 6 dataframe per categoria clinica (volumi, scale, csf, plasma, pet, cofattori). |
| 9 | `data_model/MergerTools.py` — usare la mappa in §5.7 come indice, non leggerlo linearmente | Contiene tutta la logica di matching/risoluzione conflitti richiamata dal notebook precedente. |
| 10 | `automated_merge.py` | Seconda fase del merge, automatica: combina i 6 file per categoria in tutte le combinazioni possibili di versione/metodo. È il file `.py` più corposo e attivo, vale la pena leggerlo con calma. |
| 11 | `analyze_merge_statistics.ipynb` (opzionale) | Solo se si vuole capire come si valuta "quale combinazione di merge è la migliore". |
| 12 | [PROJECT_MAP.md](PROJECT_MAP.md) | Mappa più granulare, file per file — utile come riferimento da consultare quando si incontra un file non ancora spiegato altrove. |

Cosa **non** serve leggere per capire la pipeline di produzione (ma può essere interessante come contesto storico): tutto quanto elencato in §6 come "da rimuovere" o "da valutare".

---

## 5. Cosa fa ciascun file attivo (dettaglio)

### 5.1 `adni_cleaning1.ipynb` — Step 1: pulizia per singolo file grezzo

Il notebook più lungo (~5600 righe, 382 celle). Indice completo delle sezioni, utile per saltare direttamente al file che interessa:

`ADNI` → `INIT` → `Download the raw files` → `Support file managment` → `IF SUPPORT FILE already populated` → `FILE SPECIFIC DATA CLEANING 1` →
&nbsp;&nbsp;`Abeta & Tau in CSF - Elecsys`: `UPENNBIOMK_ROCHE_ELECSYS`, `UPENNBIOMK_ADNIDIAN_ES_2017` →
&nbsp;&nbsp;`Abeta Tau altri metodi`: `EUROIMMUN`, `FUJIREBIOABETA`, `SALADAX_BIOMEDICAL`, `MESOSCALE`, `UPENNBIOMK_MASTER`, `UPENN_2DUPLC_CRM` →
&nbsp;&nbsp;`Volumi`: `UCSF Longitudinal dataset`, `UCSFFSX`, `UCSFFX7` *(sic, refuso per UCSFFSX7)*, `UCSFFSX6`, `UCSFFSX51`, `UCSFFSX51_ADNI1_3T`, `UCSDVOL` (no FreeSurfer), `UPENN_ROI_MARS` (no FreeSurfer) →
&nbsp;&nbsp;`Multiparametre dataset`: `ADNI MERGE`, `ADSP_PHC_BIOMARKER`, `ADNI_DIAN_COMPARISON` →
&nbsp;&nbsp;`Single COFACTOR files`: `APOERES`, `PTDEMOG`, `DXSUM`, `MMSE`, `ADAS 13 & 11`, `FAQ`, `CDR`, `MOCA`, più un tentativo abbandonato di ricalcolare MOCA da altri valori ("qualcosa non torna", nota dell'autore) →
&nbsp;&nbsp;`Not anymore used files`: `YASSIN_CSF & YASSIN_PLASMA`, `BLCHANGE` (ultima sezione, fine notebook)

In totale ~24 blocchi per-file distinti, tutti costruiti sullo stesso pattern (sotto).

**Il pattern ripetuto per ogni file** (questo è ciò che vale la pena capire bene, il resto è ripetizione):
1. Query datalake per `custom.level='raw'`, `custom.file_code=<codice>` → download ed estrazione.
2. `dataCleaner.replace_unknown_values(df)` — sostituisce placeholder di missing (es. `-4`, `9999`, `"Unknown"`) con NaN veri.
3. `dataCleaner.drop_if_all_none(df, columns_must_be_verified)` — scarta righe prive di ogni misura rilevante.
4. `dataCleaner.to_date_format(df, ['EXAMDATE'])` poi `dataCleaner.find_exam_code(...)` — porta la data in formato standard e deriva/consolida il codice visita (VISCODE) e il mese di visita.
5. `dataCleaner.filter_variables(df, ..., prefix='raw')` — tiene solo le colonne registrate nel file di supporto (§5.6).
6. `dataCleaner.new_variable_names(...)` — rinomina le colonne grezze nei nomi standard del progetto.
7. *(solo file CSF)* `get_abeta_tau_ratios()` + `get_ATN_profile(df, "cutoffs.json")` — calcola i rapporti Abeta/Tau e classifica il profilo Amiloide/Tau/Neurodegenerazione (A/T/N) applicando le soglie cliniche in `cutoffs.json`.
8. Aggiornamento del file di supporto (`InfoSupportFile`) e upload del file pulito su `cleaned/single_file/` con `level='cleaned_01'`.

**Esempi di logica specifica per singolo file** (ciò che *non* è generico e richiede adattamento manuale per un nuovo file grezzo):
- `ADNIMERGE`: ricalcola l'età reale a ogni visita con `add_calculated_age()`, applica una lunga serie di funzioni categoriche dedicate (`binarization_gender`, `categorize_marry`, `categorize_education`, `categorize_ethnicity`, `categorize_race`, `categorize_diagnosis`), ed estrae `FLDSTRENG`/`FSVERSION` da stringhe libere via regex.
- `ADNI_DIAN_COMPARISON`: non esistono funzioni generiche per il suo schema di codifica, quindi genere/stato civile/etnia/razza sono rimappati a mano con dizionari inline; usa `handel_same_variable_different_methods()` per riconciliare misure Abeta fatte con Elecsys e con spettrometria di massa in un'unica variabile.
- File di volumi UCSF: aggiungono `FSVERSION`/`FLDSTRENG` come costanti hardcoded (il file grezzo non le contiene), e applicano `volume_quality_filter()` più rimozione manuale delle colonne di controllo qualità (`QC`).
- `PTDEMOG` (dati anagrafici): deriva variabili calcolate ad hoc come `AGE_AD_BEG`, `AGE_AD_DX`, `AGE_COG_BEG` per aritmetica sull'anno di nascita — logica scritta una tantum nella cella, non in `DataCleaner`.
- `BLCHANGE` (ultimo blocco, ormai in "Not anymore used files"): applica un filtro riga hardcoded specifico (`VISCODE2 == 'uns1'` scartato) che non ha equivalente generico altrove.

Le funzioni di categorizzazione (§5.5) incorporano al loro interno le mappe di ricodifica specifiche ADNI — es. `binarization_gender`: female=0, male=1; `categorize_marry`: married=1, divorced=2, widowed=3, never married=0; `categorize_race`: White=5, Black=4, Asian=2, Am Indian/Alaskan=1, Hawaiian/Other PI=3, More than one=0. Sono quindi "generiche" nel senso che si possono richiamare su qualunque file con queste colonne, ma i valori numerici a cui mappano sono una convenzione di progetto da conoscere se si devono interpretare i dati puliti.

**Input**: datalake `level='raw'`, `source='ADNI'`, filtrato per `file_code`. **Output**: `cleaned/single_file/` con `level='cleaned_01'`; file di supporto `ADNI_variables_cleaned1.xlsx` aggiornato progressivamente.

### 5.2 `adni_cleaning2.ipynb` — Step 2: pulizia generica post-cleaning1

Notebook breve (~300 righe), tre sezioni: *Inizializzazione*, *Download* (da `level='cleaned_01'`, lista `file_codes` da editare a mano), *Operazioni*.

Per ogni file scaricato:
1. `remove_param_few_subjects()` — scarta variabili con troppi pochi soggetti valorizzati.
2. Conta i soggetti con più di una visita; se sono ≤10 (soglia hardcoded), imposta `multivisit=False` e non applica il passo successivo — a meno che il file non sia `PTDEMOG` o `APOERES` (eccezione hardcoded: sono dati anagrafici/genetici baseline utili anche con una sola visita per soggetto, servono per il merge).
3. Se `multivisit` è vero (o è nell'eccezione), `remove_sub_1visit()` — scarta i soggetti con una sola visita (non longitudinalmente utilizzabili).
4. `classes_to_dummies()` sulle colonne categoriche standard (`GENDER`, `MARRY`, `ETHNICITY`, `RACE`, `DX`, se presenti) — crea le variabili dummy one-hot.
5. Upload su `cleaned/single_file/` con `level='cleaned_02'`; aggiornamento `ADNI_variables_cleaned2.xlsx`.

Prima di eseguire il loop, il notebook chiede esplicitamente all'utente di **aprire e verificare a mano** l'Excel di supporto (nomi variabili, metadati) — è un checkpoint umano, non automatizzabile.

**Input**: `cleaned_01`. **Output**: `cleaned_02` + `ADNI_variables_cleaned2.xlsx`.

### 5.3 `adni_cleaning3.ipynb` — Step 3: normalizzazione volumi (post-cleaning2)

Tre sezioni: *Inizializzazione*, *Download* (da `level='cleaned_02'`, stessa lista `file_codes` hardcoded, con un blocco commentato che documenta raggruppamenti alternativi: Mixed info / Single Cofactor / Volumes / CSF), *Operazioni*.

Per ogni file: se il file di supporto lo marca come tipo `'volume'` (campo `metadati_normalizzazione`), applica `get_volumes_total()` (somma i volumi sinistro+destro in un totale, verificando che non siano già sommati) e poi `transform_volumes_as_ICV_percent()` (normalizza ogni volume come percentuale del volume cranico totale ICV, per renderlo confrontabile tra soggetti di taglia diversa). I file non-volume passano invariati. Upload finale con `level='cleaned_03'`.

**Input**: `cleaned_02`. **Output**: `cleaned_03` + `ADNI_variables_cleaned3.xlsx`. Pensato per un file usato **da solo** (non per il merge).

### 5.4 `adni_cleaning4.ipynb` — Alternativa a cleaning2+3, senza eliminare righe

Percorso alternativo: parte direttamente da `cleaned_01` (non da `cleaned_02`) e in un unico notebook applica **sia** la codifica dummy (come cleaning2) **sia** la normalizzazione volumi/ICV (come cleaning3) — ma **senza** i due passi di rimozione righe di cleaning2 (`remove_param_few_subjects`, `remove_sub_1visit`). Ogni riga/soggetto viene preservata.

**Perché la biforcazione esiste**: se l'obiettivo finale è il merge tra più file ADNI, scartare righe "per file" prima del merge creerebbe perdite di dati asimmetriche tra le fonti (un soggetto scartato in un file per pochi dati potrebbe invece avere dati buoni in un altro file da mergiare). Meglio quindi arrivare al merge con tutte le righe intatte, e applicare eventuali filtri di qualità **dopo**, in modo uniforme sul dataset unito. Da qui: **cleaning1→2→3 per un file usato da solo**, **cleaning1→4 quando il file alimenta il merge**.

**Input**: `cleaned_01`. **Output**: `cleaned_04` + `ADNI_variables_cleaned4.xlsx`. È l'output che alimenta `merge_category_wise.ipynb`.

### 5.5 `data_model/DataCleaner.py` — mappa dei metodi (classe `DataCleaner`, ~2100 righe, 46 metodi + 4 funzioni di modulo)

| Gruppo tematico | Metodi | Problema che risolve |
|---|---|---|
| Gestione file di supporto/metadati | `update_variables_support_file` (funzione di modulo), `filter_variables`, `get_file_code_metadata`, `new_variable_names`, `remove_param_few_subjects`, `update_metadati_support`, `extract_metadata_from_support`, `get_normalization_settings`, `update_self_support_file`, `file_versions_name` | Tiene sincronizzato il "dizionario dati" Excel con le colonne realmente presenti nei file, man mano che i file cambiano tra le fasi. |
| Date, visite, VISCODE | `find_exam_code`, `_handle_duplicate_dates`, `_select_by_viscode_priority`, `_calculate_visit_month`, `convert_visitcode_to_int`, `handle_f_sc_values`, `handle_null_viscode2`, `to_date_format` | ADNI codifica le visite in modo incoerente tra file (VISCODE/VISCODE2 mancanti, codici speciali 'f'/'sc', date duplicate per lo stesso soggetto): questi metodi derivano un codice/mese di visita coerente. |
| Età al follow-up | `add_calculated_age`, `age_when_missing_bl_date` | Calcola l'età ad ogni visita quando mancano data di nascita/età precisa, con fallback a cascata su ciò che è disponibile. |
| Variabili demografiche categoriche | `binarization_gender`, `categorize_marry`, `categorize_education`, `categorize_ethnicity`, `categorize_race`, `categorize_diagnosis`, `classes_to_dummies` | Normalizza campi che arrivano sia come stringa che come codice numerico incoerente tra file diversi; `classes_to_dummies` fa poi la conversione one-hot. |
| Genotipo APOE | `uniform_APOE_format`, `APOE_4_count`, `APOE_to_dummies` | L'APOE arriva in formati eterogenei (es. "3/4", "E3E4"); questi metodi lo normalizzano e derivano il conteggio di alleli a rischio (APOE4). |
| Biomarcatori CSF/PET/plasma e profilo ATN | `load_cutoffs`, `deep_update`, `get_cutoff_from_dict` (funzioni di modulo), `convert_to_dummies_ATNC_profile`, `get_abeta_tau_ratios`, `handel_same_variable_different_methods`, `ensure_method_column`, `ensure_tau_metaroi`, `get_ATN_profile` | Gli stessi biomarcatori sono misurati con piattaforme diverse (Elecsys, Lumipulse, Simoa, tracciante PET), ognuna con soglie cliniche proprie (`cutoffs.json`); questi metodi calcolano rapporti diagnostici e derivano lo stato patologico Amiloide/Tau/Neurodegenerazione. |
| Volumi cerebrali/ICV e QC immagini | `segmentation_complete_filter`, `convert_qcpass_values`, `volume_quality_filter`, `get_volumes_total`, `transform_volumes_as_ICV_percent` | Applica i controlli qualità FreeSurfer e normalizza i volumi cerebrali rispetto alla taglia del cranio (ICV), per renderli confrontabili tra soggetti. |
| Pulizia generale | `replace_unknown_values`, `drop_if_all_none`, `remove_sub_1visit`, `get_mean_row_per_visit` | Pulizia trasversale: placeholder di missing, righe vuote, soggetti con una sola visita, aggregazione di misure ripetute nella stessa visita. |

Vedi §7 per una lista di funzioni con nomi/logica sovrapposti o segnalate come dead code dagli autori stessi.

### 5.6 `data_model/manage_excel_support_file.py` — il "file di supporto"

Il **file di supporto** (`ADNI_variables_cleaned*.xlsx`, uno per fase di cleaning) è il catalogo dei metadati delle variabili lungo tutta la pipeline: per ogni variabile di ogni file ADNI registra nome originale/standardizzato, tipo, range di valori validi, quanti missing ha, quali popolazioni/coorti ADNI la coprono, se ha soggetti con visite multiple, e un flag `del` (tenere/scartare, con soglia automatica al 65% di missing). È in pratica la "memoria" delle decisioni prese su ciascuna variabile.

Funzioni principali:
- `save_df` — salva sempre in doppio formato .xlsx + .csv.
- `create_new_support_file` — crea la versione del file di supporto per la fase successiva, svuotando i metadati calcolati (da ricalcolare/verificare) ma mantenendo la traccia dei nomi.
- `update_new_support_file` — quando arrivano nuovi file o si rielabora un file, aggiorna il file di supporto senza perdere le annotazioni manuali già fatte nelle fasi precedenti.
- Classe `InfoSupportFile` — calcola e scrive le statistiche per singola variabile (`get_varible_info`, `check_missing_values`, `check_type_range_variables`), individua automaticamente la colonna che identifica la coorte ADNI (`find_population_variable`), conta soggetti totali/con visite multiple (`get_subjects_and_multiplevisits`).

In pratica, ogni notebook di cleaning, ad ogni file processato, aggiorna questo Excel — è il motivo per cui i notebook chiedono spesso di "aprire e verificare a mano" il file prima di procedere.

### 5.7 `data_model/MergerTools.py` — mappa dei metodi (classe `MergerTools`, ~2500 righe)

| Gruppo tematico | Metodi | Problema che risolve |
|---|---|---|
| Matching di visite (RID/EXAMDATE con buffer temporale) | `find_visit_matches`, `find_rid_matches`, `list_index_visit_matches`, `matrix_match`, `individual_match_and_missing_rows`, `date_matches_with_buffer`, `date_matches_with_buffer_and_extra_var`, `get_indexes_from_matches`, `verify_visit_matches`, `create_temp_merge` | Lo stesso soggetto fa la "stessa" visita in due file sorgente ma con `EXAMDATE` che differisce di qualche giorno (es. esame clinico e prelievo CSF fatti in giorni diversi): si cercano prima match esatti, poi match entro un buffer (default 80 giorni), con assegnazione che evita di riusare due volte la stessa riga. |
| Filtraggio per categoria | `CATEGORY_CONFIG`, `filter_df_category`, `calculate_visit_month` | Le 6 categorie cliniche (volumes, scale, csf, plasma, pet, cofactor) hanno ciascuna chiavi di join diverse (es. i volumi hanno anche `FSVERSION`); questi metodi selezionano solo le colonne pertinenti e ricalcolano il mese di visita. |
| Risoluzione conflitti tra valori discordanti | `get_merged_df`, `remove_duplicates_from_merged_df`, `volume_unification_criteria`, `pet_unification_criteria`, `plasma_unification_criteria`, `update_stamp_unification_criteria`, `audit_conflicts` | Quando due fonti riportano la stessa visita con valori diversi (es. MMSE 24 vs 25), sceglie quale tenere: per i volumi in base a `STATUS`/potenza del magnete, per scale/CSF in base al timestamp più recente o al trend atteso, per plasma/PET in base a quale riga è più completa. `audit_conflicts` produce un report dei conflitti prima ancora di mergiare. |
| Cofattori e variabili dummy | `get_merged_df_cofactor`, `_resolve_dummy_group_immutable`, `_resolve_dummy_group_mutable`, `_resolve_marry`, `_resolve_cohort`, `_resolve_age`, `_propagate_all_immutables`, `_validate_dummies` | I cofattori (sesso, etnia, stato civile, diagnosi...) sono dummy one-hot che devono restare coerenti per soggetto (variabili immutabili, es. razza) o per visita (mutabili, es. diagnosi che cambia nel tempo); qui si risolvono le discrepanze e si propagano i valori. |
| Deduplicazione | `remove_duplicates_from_merged_df`, `_consolida_duplicati`, `_get_row_to_drop` | Dopo un merge "outer", righe con stesso (RID, EXAMDATE) diventano duplicati da consolidare (riempiendo i NaN reciproci) e poi da ridurre a una sola. |
| Codice legacy ("OLD FUNCTIONS", fine file) | `get_merged_col_reference`, `check_trend_and_range`, `get_merged_float_range_and_trend`, `get_merged_columns_float`, `merge_paired_rows_rid_specific` | Blocco esplicitamente marcato come vecchio dagli autori — probabilmente ignorabile a meno che un notebook non più attivo lo richiami ancora. |

### 5.8 `merge_category_wise.ipynb` — Merge fase 1: unione per categoria clinica

Scarica i file `cleaned_04` raggruppati per categoria — **"categoria"** qui significa un dominio clinico (volumi cerebrali, scale cognitive/funzionali, biomarcatori CSF, plasma, PET, cofattori demografici/genetici), non un singolo file sorgente. Per la categoria scelta (variabile `category` da impostare a mano: `volumes`, `scale`, `csf`, `plasma`, `pet`, `cofactor`):
1. Filtra ogni file alle sole colonne di quella categoria (`filter_df_category`).
2. Costruisce matrici diagnostiche di sovrapposizione (quante righe di ciascun file combaciano con quante degli altri, su RID e su RID+EXAMDATE) per decidere l'ordine di merge.
3. L'utente sceglie a mano un dataframe "base" (di solito quello con più righe in comune) e una gerarchia ordinata degli altri da aggiungere uno alla volta.
4. Per ciascuno: trova i match di visita (esatti/con buffer), verifica la cardinalità dei match, poi chiama `mergeTools.get_merged_df(...)`.
5. Salva il risultato su `cleaned/merged/category` con nome tipo `VOLUMES_merged.csv`.

Va eseguito **una volta per categoria**, manualmente, prima che `automated_merge.py` possa girare.

### 5.9 `automated_merge.py` — Merge fase 2: combinazione automatica (entry point attivo)

**Cosa si intende per "combinazione"**: alcune categorie hanno varianti interne — versione FreeSurfer (`FSVERSION`, per i volumi), metodo di dosaggio (`METHOD_CSF`, `METHOD_PLASMA`, `METHOD_PET`). Una "combinazione" è una scelta specifica di questi valori (es. `FSVERSION=7.0`, `METHOD_CSF=elecsys`, `METHOD_PET=FBB`); lo script genera il prodotto cartesiano di tutti i valori disponibili (filtrati da una whitelist) e produce un file di merge per ciascuna combinazione.

**Flusso di `main()`, in ordine**:
1. `load_datasets_from_datalake` — scarica i 6 CSV per-categoria prodotti dalla fase 1.
2. `generate_all_combinations` — prodotto cartesiano dei filtri attivi (SCALE e COFACTOR non hanno varianti, quindi non moltiplicano le combinazioni).
3. Per ciascuna combinazione, `run_incremental_merge` percorre un ordine fisso di categorie (volumi → scale → csf → plasma → pet → cofattori) e le fonde una alla volta con `merge_single_pair`, che: deduplica i volumi, trova i match di visita, riduce i match a corrispondenza 1:1, allinea le date, mergia (`merge_datasets`), e deduplica eventuali righe residue senza data — sollevando un errore se una corrispondenza resta ambigua.
4. `build_enriched_metadata` + `save_merged_result` — arricchisce i metadati e carica il CSV della combinazione su `cleaned/merged/combination`, con nome `merge_<chiave>_<valore>_...csv` (es. `merge_FSVERSION_7-0_METHOD_CSF_elecsys_METHOD_PET_FBB.csv`).
5. `compute_merge_statistics` per ogni combinazione (righe, soggetti, visite per soggetto, range date, % NaN), poi `save_statistics_csv` scrive tutto in `merge_statistics.csv` (questo risponde al dubbio aperto in PROJECT_MAP su chi genera quel file: lo genera questa funzione).

### 5.10 `analyze_merge_statistics.ipynb` — diagnostica opzionale post-merge

Legge `merge_statistics.csv` e calcola: inventario dei valori di filtro disponibili, statistiche descrittive (righe/colonne/soggetti/visite/% NaN) globali e per combinazione di `FSVERSION`/`METHOD_CSF`/`METHOD_PET`, confronti tra versioni FreeSurfer e metodi di biomarcatore, e una classifica delle combinazioni con meno missing e più righe. Serve a scegliere, tra tutte le combinazioni generate da `automated_merge.py`, quella con il miglior compromesso tra dimensione campionaria e completezza — utile prima di usare un merge per il training di un modello ML.

---

## 6. File da valutare per la rimozione (proposta, nessuna eliminazione fatta)

Per regola di progetto ([CLAUDE.md](CLAUDE.md): *"non eliminare nessun file... al massimo proporre e attendere conferma"*), quanto segue è **solo una proposta** da confermare uno a uno.

### Da rimuovere con alta confidenza

| File/cartella | Perché |
|---|---|
| `da_cestinare/old_excel/` | Cartella **vuota**. Il nome stesso ("da cestinare") indica intento di rimozione già deciso. |
| `da_cestinare/tests_utils/` | Cartella vuota (contiene solo un `__pycache__` vuoto). Stesso discorso. |
| `scan_imports.ipynb` | Annotato in PROJECT_MAP come "notebook temporaneo creato per audit importi, eliminare prima del merge finale su main". |
| `merge_categories_with_filters.ipynb` | Verificato con lettura diretta (non solo l'annotazione di PROJECT_MAP) — vedi dettaglio subito sotto la tabella. |
| `test_notebooks/prove_excel.ipynb` | Annotato come "test per verificare notazioni sugli excel, probabilmente non più necessario". |
| `__pycache__/`, `data_model/__pycache__/`, `da_cestinare/tests_utils/__pycache__/` | Bytecode Python generato automaticamente, non va versionato. Andrebbero aggiunti a `.gitignore` (attualmente assente per questo pattern). |

**Dettaglio `merge_categories_with_filters.ipynb`** — perché è più di una "bozza generica":
1. **Non è eseguibile così com'è**: la cella di download ha `file_codes = []` (lista vuota), quindi la query al datalake fallisce (`Exception: Query failed: No files match the query criteria`). Di conseguenza `zip_files` non viene mai creato, ma la cella successiva lo usa comunque (`zip_files.items()`) — il notebook non gira dall'inizio alla fine nello stato in cui è salvato.
2. **Si interrompe a metà**: dopo l'intestazione "Merge pipeline" e la sezione "ADD df by df" (che descrive solo l'intenzione — unire i df uno a uno filtrando per `FSVERSION`/`METHOD`), il codice vero e proprio non c'è: restano una cella vuota e una cella markdown vuota. Il cuore del notebook — il loop di merge — non è mai stato scritto.
3. **Un commento dell'autore segnala incertezza**: sulla riga `base = 'df_4'` compare `#sembra un errore ma questi df sono definiti`, tipico di codice lasciato a metà mentre si stava ancora ragionando sulla logica.
4. **Duplica uno scopo già coperto**: l'obiettivo dichiarato nel titolo — "Merge between categories based on filters" — è esattamente ciò che fanno già, in modo completo e funzionante, `merge_category_wise.ipynb` (fase 1) e `automated_merge.py` / `final_merge_categories.ipynb` (fase 2).

In sintesi: non è "solo non finito", è un tentativo interrotto, attualmente non eseguibile, il cui scopo è già coperto da codice funzionante altrove.

### Da valutare con l'autore (probabile scarto, ma verificare prima)

| File/cartella | Motivo del dubbio |
|---|---|
| `prove_download.ipynb` | Ispezionato direttamente: è un notebook scratch — ricerche/download di prova, celle duplicate (due sezioni "Special per file strani" quasi identiche), variabili globali riusate in modo incoerente (`df_new`, `df_focus`...), nessuna struttura lineare. Utile come riferimento storico su "come si scarica da datalake", non come codice di produzione. |
| `test_notebooks/` (l'intera cartella: `aggiornamento_excel_support.ipynb`, `controll_uploaded_files.ipynb`, `study_diff_volumes.ipynb`, `update_file_small_changes.ipynb`, `update_metadata.ipynb`, `vol_range.ipynb`, incluse le immagini in `immagini/`) | PROJECT_MAP li descrive come "notebook creati e poi non molto usati". Alcuni contengono analisi potenzialmente riutilizzabili (es. `compare_methods_var.ipynb` — confronto statistico tra metodi diversi sulla stessa variabile — segnalato come "interessante per merge/harmonization future": **non buttare senza rileggere questo in particolare**). |
| `documents/ADNI_variables_statistics.csv` | Confrontato con `ADNI_variables_statistics.csv` in root: **contenuti diversi** (586 righe vs 390, aggiornato a Giu 2025 contro Dic 2025 della versione in root). Sembra una copia obsoleta pre-refactoring del file in root, ma va confermato che nessun processo legga ancora da `documents/`. |
| `documents/Data_Cleaning_Note.docx`, `documents/Documentazione modifiche dati.docx` | PROJECT_MAP li segna come "probabilmente non aggiornati". Da confrontare con lo stato corrente della pipeline prima di archiviarli. |
| `finalize_synthetic_file.ipynb` | **Non descritto in PROJECT_MAP** (possibile file aggiunto dopo l'ultima mappatura). Ispezionato: notebook funzionante che rifinisce **un singolo file sintetico specifico** (path hardcoded `subMERGE_4-3_elecsys_FBP_fxd_0`) — rinomina colonne, ricalcola `VISIT_MONTH`, ricostruisce le dummy, aggiorna metadati e ricarica sul datalake. Sembra un'operazione one-off più che un passo di pipeline riutilizzabile: da confermare con l'autore se va generalizzato, tenuto come riferimento, o è già superato da `post-generation/`. |
| `adni_cleaning{1,2,3,4}_bb.ipynb` | Varianti "_bb" dei notebook di cleaning attivi, non descritte in PROJECT_MAP né in CLAUDE.md. Generano output paralleli (`ADNI_variables_cleaned*BB.csv/.xlsx`). Da chiarire se sono una variante di dataset (es. altro sito/batch) ancora in uso o notebook di test lasciati a fianco dei principali — **ricadono comunque nel divieto di modifica dei notebook di cleaning senza approvazione esplicita** (CLAUDE.md), quindi qualunque intervento va discusso prima. |

### Utility one-off (da tenere, ma con etichetta chiara)

`fix_dummies_submerge.ipynb`, `fix_merge_comb.ipynb`, `update_metadata_merge.ipynb` non sono bozze abbandonate: risolvono problemi puntuali già emersi (es. segnalazione Bari sui dummies) e sono stati applicati ai dati. Non sono entry point ripetibili della pipeline standard — andrebbero rietichettati come "interventi storici" piuttosto che eliminati, così chi arriva dopo capisce perché esistono senza doverli rieseguire.

---

## 7. Problemi noti nel codice attivo (da tenere a mente studiandolo)

Questi non sono motivo per scartare i file (sono file attivi e funzionanti nel complesso), ma vale la pena saperli prima di fidarsi ciecamente di ogni riga durante lo studio del codice:

- **Possibile bug funzionale** in `DataCleaner.age_when_missing_bl_date` (`data_model/DataCleaner.py`, righe ~828-862): chiama `self.add_calculated_age(self, exam_date=..., ...)`, passando `self` sia come metodo bound sia come primo argomento posizionale — sembra un doppio-`self` da refactor incompleto, non uno stile voluto.
- **Type hint non validi** in `DataCleaner.convert_visitcode_to_int` (riga ~529): `value: string` usa il modulo `string` come annotazione di tipo invece di `str`, e il tipo di ritorno `int or string` non è sintassi Python valida per i type hint (viene solo interpretato come espressione booleana a tempo di import, senza documentare nulla).
- **Funzione segnata come da eliminare dall'autore stesso**: `DataCleaner.convert_to_two_bit` (riga ~1118) ha nel docstring "DEPRECABILE/ELEMINABILE — sostituita da più efficiente funzione `convert_to_ATN_profile`" — ma quella funzione non esiste con questo nome nel file (i candidati più vicini sono `convert_to_dummies_ATNC_profile` o `get_ATN_profile`). Verificare con grep sui notebook se `convert_to_two_bit` è ancora richiamata da qualche cella prima di considerarla morta.
- **Probabile funzione orfana**: `DataCleaner.to_ICV_percentage` (riga ~1619) ha un commento dell'autore che ammette che la sua logica è stata spostata direttamente dentro `transform_volumes_as_ICV_percent` per evitare confusione — la funzione standalone sembra non più chiamata.
- **Metodi privati potenzialmente mai richiamati**: `DataCleaner._select_from_complete_rows` e `_select_by_null_count_and_viscode` sembrano un ramo alternativo di logica per la scelta della riga da tenere tra duplicati, mai invocato dal percorso realmente usato da `find_exam_code` (che chiama invece `_handle_duplicate_dates`/`_select_by_viscode_priority`).
- **Refuso non bloccante**: parametro `slef` invece di `self` in `DataCleaner.handel_same_variable_different_methods` (riga ~1759) — funziona comunque in Python, ma segnala codice non rivisto; anche il nome del metodo stesso ("handel") è un refuso di "handle".
- **Nome fuorviante**: `DataCleaner.categorize_education` non categorizza nulla, si limita a convertire il valore in intero — nome incoerente con gli altri `categorize_*` che invece rimappano categorie testuali/numeriche.
- **Blocco di codice legacy dichiarato**: in `data_model/MergerTools.py`, la sezione finale marcata `################ OLD FUNCTIONS ################` (`get_merged_col_reference`, `check_trend_and_range`, `get_merged_float_range_and_trend`, `get_merged_columns_float`, `merge_paired_rows_rid_specific`) è logica di merge più vecchia e manuale, probabilmente non più nel percorso attivo.

---

## 8. Cosa manca prima di poter chiudere il refactoring (da CLAUDE.md)

- [x] Fase 1 — Mappatura iniziale (questo file + PROJECT_MAP.md)
- [ ] Fase 2 — Audit dei file `.py` (gli `audit_*.txt` in root sono un primo output automatico; §5 e §7 sopra sono un primo passo di revisione manuale)
- [ ] Fase 3 — Audit dei notebook (spunti raccolti in §5 e §6)
- [ ] Fase 4 — Rimozione codice inutile (richiede conferma esplicita file per file, vedi regole in CLAUDE.md)
- [ ] Fase 5 — Docstring e commenti (vedi le lacune segnalate in §7)
- [ ] Fase 6 — Riorganizzazione in `pipeline/`, `utils/`, `notebooks/exploration|validation/`, `tests/` e README finale
