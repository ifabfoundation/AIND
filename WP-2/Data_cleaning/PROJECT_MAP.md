# File Map di Data Cleaning
## Struttura file
### Notebook pulizia dati
./adni_cleaning1.ipynb      --> Notebook per 1 step di pulizia specifico - per ciascun file normalizza nomi variabili e variabili categoriche, genera variabili, ...
./adni_cleaning2.ipynb      --> Notebook per 2 step di pulizia generico post cleaning1 - rimuove righe, soggetti, variabili, crea dummies e quindi aggiunge i metadati al file
./adni_cleaning3.ipynb      --> Notebook per 3 step di pulizia generico post cleaning2 - specifico per i file con volumi, unisce volumi R&L e normalizza per ICV
./adni_cleaning4.ipynb      --> Notebook per 4 step di pulizia generico post cleaning1 - Fa step 2 e 3 ma senza eliminare di default le righe

### file excel 
standard info in the excels: file_name	file_code	parameter	orig_variable_code	variable_code	Unit	type_variable	classes	range	valid_values	missing_values	missing_pop	del	metadati_fattori	metadati_normalizzazione

./ADNI_variables_cleaned1.csv       --> file containing standard info regarding the dataset updated after cleaning 1
./ADNI_variables_cleaned1.xlsx      --> file containing standard info regarding the dataset updated after cleaning 1
./ADNI_variables_cleaned2.csv       --> file containing standard info regarding the dataset updated after cleaning 2
./ADNI_variables_cleaned2.xlsx      --> file containing standard info regarding the dataset updated after cleaning 2
./ADNI_variables_cleaned3.csv       --> file containing standard info regarding the dataset updated after cleaning 3      
./ADNI_variables_cleaned3.xlsx      --> file containing standard info regarding the dataset updated after cleaning 3
./ADNI_variables_cleaned4.csv       --> file containing standard info regarding the dataset updated after cleaning 1
./ADNI_variables_cleaned4.xlsx      --> file containing standard info regarding the dataset updated after cleaning 1
./ADNI_variables_statistics.csv     --> file containing standard info regarding the raw dataset
./ADNI_variables_statistics.xlsx    --> file containing standard regarding the raw dataset


### Json files
./age_recalculation_warnings.json   --> json created to visualize warnings when recalculating the age - initial age nan, no valid age
./cofattori_values_settings.json    --> json di riferimento per i range dei cofattori
./cutoffs.json                      --> json di riferimento per i cut off per biomarcatori CSF, plasma e PET specifico per strumento/mezzo
./normalization_settings.json       --> json di riferimento per i range dei cofattori
./volume_values_settings.json       --> json di riferimento per i range dei volumi


### Merge files
./automated_merge.py                    --> py per fare il merge automatico tra df di categoria, contiene tutte le funzioni
./analyze_merge_statistics.ipynb        --> serie di statistiche ordinate per valutare i merge ottenuti (credo)
./final_merge_categories.ipynb          --> notebook per fare il merge tra df di categoria, contiene tutte le funzioni
./merge_categories_with_filters.ipynb   --> bozza notebook per fare il merge tra df di categoria --> eliminabile non terminato
./merge_category_wise.ipynb             --> notebook per il merge dei file cleaned_4 in df merged per categoria (vol, scale, ...) Richiama funzioni per gerarchia delle righe da tenere
./merge_statistics.csv                  --> csv statistiche sui merge finali, richiamata in analyze_merge_statistics.ipynb ma non so da dove sia creata

### Data model
./data_model                                --> folder contenti modelli e file py con funzioni
./data_model/DataCleaner.py                 --> py con tutte le funzioni per la pulizia dei dati richiamate dai Notebook cleaning...
./data_model/manage_excel_support_file.py   --> py con funzioni per riempire gli excel che tracciano le modifiche
./data_model/MergerTools.py                 --> py con funzioni per merge di file

### Documents
./documents                                     --> documenti generali su dati processo di pulizia, ... no codice 
./documents/ADNI_variables_statistics.csv       --> csv delle informazioni statistiche sui dati grezzi (a quando aggiornato? potrebbe essere tolto)
./documents/Data_Cleaning_Note.docx             --> note e appunti su processo di pulizia (probabilmente non aggiornato)
./documents/Documentazione modifiche dati.docx  --> documentazione di modifiche dei dati (probabilmente non aggiornato)


### Notebook vari
./prove_download.ipynb          --> funzioni ricerca e download file sia singoli che liste ed assegnazione automatica a variabile, più disordine funzioni per conto nan e ricerca id_code in comune, funzione conto soggetti con 2+ visite, funzione per vedere i metadati, download speciali.
./test_merge.ipynb              --> funzione per analisi Dummies, funzioni per identificare presenza di categorie di variabili in file, funzioni aggiornamento metadati
./fix_dummies_submerge.ipynb    --> funzioni per sistemare i dummies fatta per il submerge specifico che mischia categorie per filtri. Fatto dopo segnalazione Bari.
./fix_merge_comb.ipynb          --> funzioni per il ricalcolo dell'età e valuta soggetti che hanno età iniziale nan e quelli che non hanno età valida e rimozioni righe con pochi valori. Appliato a tutti i merge unione di categorie.
./update_metadata_merge.ipynb   --> operazioni specifiche per modificare metadati di file 

./test_notebooks                --> notebooks creati e poi non molto usati
./test_notebooks/aggiornamento_excel_support.ipynb      --> per excel di cleaning 2, elimina righe di file non elencati, e aggiornare la variabile popolazione
./test_notebooks/compare_methods_var.ipynb              --> Analisi statistica di una stessa variabile in due dataset (con metodo diverso) per valutare se statisticamente sono utilizzabili, interessante per valutazioni merge e harmonization future
./test_notebooks/controll_uploaded_files.ipynb          --> funzioni per fare controllo su molteplici file, print recursivo dei metadati di ciascuno, oltre a download e delate di file e altre operazioni random
./test_notebooks/prove_excel.ipynb                      --> test per verificare il funzionamento delle notazioni sugli excel, probabilmente non più necessario
./test_notebooks/study_diff_volumes.ipynb               --> analisi diffrenze volumi da diversi file (senza contare versioni differenti), o da aggregare alle analisi statistiche o eliminabile
./test_notebooks/update_file_small_changes.ipynb        --> piccole operazione da fare sui file come, modifica dei nomi delle variabili e aggiornamento metadati
./test_notebooks/update_metadata.ipynb                  --> semplici operazioni per aggiornare i metadata o vederli, unire con altre funzioni dei metadati
./test_notebooks/vol_range.ipynb                        --> studio file con volumi, stampa dei percentili, unione file e altre analisi
./test_notebooks/immagini                               --> immagini ottenute dalle analisi di differenze dei volumi tra i file senza considerare metodo usato
./test_notebooks/immagini/boxplot_differenze_mean_in_colonna_totalsub.png
./test_notebooks/immagini/differenze_json_media&nextvisit.png


### Post generation pipeline
Cartella autonoma e già strutturata con proprio README dettagliato.
Pipeline INDIPENDENTE dal cleaning — prende in input dati sintetici già generati
esternamente e applica post-elaborazioni per migliorarne qualità e realismo.

**Scopo generale:**
1. Invertire le trasformazioni applicate durante la preparazione dei dati (reverse)
2. Aggiungere rumore con Differential Privacy per migliorare la privacy
3. Correggere i dati sintetici con tecniche Residual-PAR (opzionale)
4. Iniettare missing values per simulare pattern reali (opzionale)

**Entry point**:
./post-generation/reverse_from_datalake.bat   → orchestratore principale (Windows)
                                                 attualmente esegue reverse su:
                                                 - ADNIMERGE_synthetic_utility
                                                 - ADNIMERGE_to_test_utility
./post-generation/reverse_from_local.py       → alternativa senza datalake (per test locali)
./post-generation/reverse_transformations.py  → entry point unificato/orchestratore

**Moduli (non entry point — contengono funzioni):**
./post-generation/denormalize.py                → denormalizzazione valori numerici
./post-generation/dummy_to_categorical.py       → conversione dummy→categorical
./post-generation/noise_injection.py            → meccanismi DP (Laplace, Gaussian, Adaptive)
./post-generation/noise_injection_from_datalake.py → noise injection dal datalake
./post-generation/add_noise.py                  → funzioni per Residual-PAR e allineamento
./post-generation/add_noise_nb.py               → versione notebook con KNN pseudo-pairing
./post-generation/residual_par_correction.py    → correzione KNN + PAR
./post-generation/inject_missing_values.py      → iniezione missing values

**File di configurazione:**
./post-generation/normalization_settings.json   → impostazioni scale cliniche (MMSE, FAQ, ...)
./post-generation/volume_values_settings.json   → impostazioni volumi cerebrali

**Altri file:**
./post-generation/__init__.py                           → inizializzazione package
./post-generation/reverse_commands.txt                  → comandi di riferimento rapido
./post-generation/create_validation_source_dataset.ipynb → notebook per dataset validazione

## Entry point
### Entry point cleaning
./adni_cleaning1.ipynb      --> Notebook pulizia: 1 step di pulizia celle specifiche per file. INPUT: dati grezzi, OUTPUT: cleaned_01
./adni_cleaning2.ipynb      --> Notebook pulizia: 2 step di pulizia generico post cleaning1. INPUT: cleaned_01, OUTPUT: cleaned_02
./adni_cleaning3.ipynb      --> Notebook pulizia: 3 step di pulizia generico post cleaning2. INPUT: cleaned_02, OUTPUT: cleaned_03
./adni_cleaning4.ipynb      --> Notebook pulizia: 2 step  di pulizia generico post cleaning1 (alternativo a ./adni_cleaning2.ipynb e ./adni_cleaning2.ipynb). INPUT: cleaned_01, OUTPUT: cleaned_04

### Entry point post-generation
./post-generation/reverse_from_datalake.bat   → orchestratore principale (Windows)
                                                 attualmente esegue reverse su:
                                                 - ADNIMERGE_synthetic_utility
                                                 - ADNIMERGE_to_test_utility
./post-generation/reverse_from_local.py       → alternativa senza datalake (per test locali)
./post-generation/reverse_transformations.py  → entry point unificato/orchestratore

### Entry point merge
./merge_category_wise.ipynb             --> notebook per la prima fase di merge, da file generati da ./adni_cleaning4.ipynb INPUT: cleaned_04, OUTPUT: MERGE<CATHEGORY>
./automated_merge.py                    --> entry point ATTIVO per la seconda fase del merge in maniera automatica, prende file generati da ./merge_category_wise.ipynb INPUT: MERGE<CATHEGORY>, OUTPUT: merge_<combinazione>
./final_merge_categories.ipynb          --> notebook alternativo (valutare se metterlo in archivio) per fare la seconda fase merge, prende file generati da ./merge_category_wise.ipynb INPUT: MERGE<CATHEGORY>, OUTPUT: merge_<combinazione>
./analyze_merge_statistics.ipynb        --> NON è un entry point del flusso produttivo, è un notebook esplorativo/diagnostico da eseguire opzionalmente dopo il merge per valutarne la qualità


## Flusso dati 
### Cleaning per uso singolo dei file
cleaning1.ipynb → [file cleaned_1] → cleaning2.ipynb → [file cleaned_2] → cleaning3.ipynb → [file cleaned_3]

### Cleaning e merge file
cleaning1.ipynb → [file cleaned_1] → cleaning4.ipynb → [file cleaned_4]
                                                           ↓
                                              merge_category_wise.ipynb
                                                           ↓
                                              automated_merge.py / final_merge_categories.ipynb


### Post-generation
[1] noise_injection_from_datalake.py  → INPUT: dati sintetici normalizzati [0,1]
                                        OUTPUT: dati con rumore DP aggiunto
         ↓
[2] reverse_from_datalake.py          → INPUT: dati noisy normalizzati
                                        OUTPUT: dati denormalizzati, dummy→categorical
         ↓
[3] residual_par_correction.py        → INPUT: dati reali + sintetici reversed (OPZIONALE)
                                        OUTPUT: dati corretti con metriche KS/DCR
         ↓
[4] inject_missing_values.py          → INPUT: dati reali + sintetici reversed (OPZIONALE)
                                        OUTPUT: dati con missing values realistici


## Note e dubbi aperti
- merge_statistics.csv → non è chiaro quale script la genera, da investigare
- documents/ADNI_variables_statistics.csv → duplicato di quello in root? Verificare
- Data_Cleaning_Note.docx e Documentazione modifiche.docx → probabilmente non aggiornati, valutare se eliminare
- study_diff_volumes.ipynb → da aggregare ad analyze_merge_statistics.ipynb o eliminare?
- add_noise_nb.py → non è chiaro se è ancora usato o è una bozza superata da add_noise.py, verificare se eliminabile
- .dl_client.ini → non era nella lista originale del PROJECT_MAP, contiene credenziali del datalake — verificare che sia nel .gitignore e NON venga mai committato su GitHub
- reverse_commands.txt → non era descritto nel PROJECT_MAP originale, aggiungere come file di riferimento rapido comandi
- scan_imports.ipynb → notebook temporaneo creato per audit importi, eliminare prima del merge finale su main


## Dipendenze esterne

### Manipolazione dati
- pandas      → manipolazione e analisi dataframe
- numpy       → calcolo numerico
- dateutil    → calcolo date relative (relativedelta)

### Visualizzazione (usata nei notebook di analisi)
- matplotlib    → grafici base e patch (Patch)
- seaborn       → grafici statistici

### Statistica avanzata e ML
- statsmodels   → statistiche inter-rater (aggregate_raters), modelli sm
- pingouin      → statistiche avanzate (pg) — NUOVA, non era nei .py
- scipy       → statistiche avanzate (ks_2samp, stats, pearsonr, norm, t ai moduli usati)
- scikit-learn → KNN per Residual-PAR, preprocessing
               (NearestNeighbors, QuantileTransformer, StandardScaler)

### Dati sintetici (solo post-generation)
- sdv         → generazione dati sintetici (PARSynthesizer)
                verificare se strettamente necessario o opzionale

### Dipendenze specifiche post-generation
- dl_client   → client proprietario per accesso al datalake
                NON su PyPI — da installare separatamente
                richiede configurazione tramite .dl_client.ini

### Librerie standard Python (non da installare)
- os, sys, json, re, datetime, argparse,
  pathlib, typing, warnings, stat, string,
  itertools