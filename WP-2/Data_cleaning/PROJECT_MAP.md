# File Map di Data Cleaning
## Notebook pulizia dati
./adni_cleaning1.ipynb      --> Notebook per 1 step di pulizia specifico - per ciascun file normalizza nomi variabili e variabili categoriche, genera variabili, ...
./adni_cleaning2.ipynb      --> Notebook per 2 step di pulizia generico post cleaning1 - rimuove righe, soggetti, variabili, crea dummies e quindi aggiunge i metadati al file
./adni_cleaning3.ipynb      --> Notebook per 3 step di pulizia generico post cleaning2 - specifico per i file con volumi, unisce volumi R&L e normalizza per ICV
./adni_cleaning4.ipynb      --> Notebook per 4 step di pulizia generico post cleaning1 - Fa step 2 e 3 ma senza eliminare di default le righe

## file excel 
./ADNI_variables_cleaned1.csv       --> file containing info regarding the dataset after cleaning 1
./ADNI_variables_cleaned1.xlsx      --> file containing info regarding the dataset after cleaning 1
./ADNI_variables_cleaned2.csv       --> file containing info regarding the dataset after cleaning 2
./ADNI_variables_cleaned2.xlsx      --> file containing info regarding the dataset after cleaning 2
./ADNI_variables_cleaned3.csv       --> file containing info regarding the dataset after cleaning 3      
./ADNI_variables_cleaned3.xlsx      --> file containing info regarding the dataset after cleaning 3
./ADNI_variables_cleaned4.csv       --> file containing info regarding the dataset after cleaning 1
./ADNI_variables_cleaned4.xlsx      --> file containing info regarding the dataset after cleaning 1
./ADNI_variables_statistics.csv     --> file containing info regarding the raw dataset
./ADNI_variables_statistics.xlsx    --> file containing info regarding the raw dataset

## Json files
./age_recalculation_warnings.json   --> json created to visualize warnings when recalculating the age - initial age nan, no valid age
./cofattori_values_settings.json    --> json di riferimento per i range dei cofattori
./cutoffs.json                      --> json di riferimento per i cut off per biomarcatori CSF, plasma e PET specifico per strumento/mezzo
./normalization_settings.json       --> json di riferimento per i range dei cofattori
./volume_values_settings.json       --> json di riferimento per i range dei volumi


## Merge files
./automated_merge.py                    --> py per fare il merge automatico tra df di categoria, contiene tutte le funzioni
./analyze_merge_statistics.ipynb        --> serie di statistiche ordinate per valutare i merge ottenuti (credo)
./final_merge_categories.ipynb          --> notebook per fare il merge tra df di categoria, contiene tutte le funzioni
./merge_categories_with_filters.ipynb   --> bozza notebook per fare il merge tra df di categoria --> eliminabile non terminato
./merge_category_wise.ipynb             --> notebook per il merge dei file cleaned_4 in df merged per categoria (vol, scale, ...) Richiama funzioni per gerarchia delle righe da tenere
./merge_statistics.csv                  --> csv statistiche sui merge finali, richiamata in analyze_merge_statistics.ipynb ma non so da dove sia creata

## Data model
./data_model                                --> folder contenti modelli e file py con funzioni
./data_model/DataCleaner.py                 --> py con tutte le funzioni per la pulizia dei dati richiamate dai Notebook cleaning...
./data_model/manage_excel_support_file.py   --> py con funzioni per riempire gli excel che tracciano le modifiche
./data_model/MergerTools.py                 --> py con funzioni per merge di file

## Documents
./documents                                     --> documenti generali su dati processo di pulizia, ... no codice 
./documents/ADNI_variables_statistics.csv       --> csv delle informazioni statistiche sui dati grezzi (a quando aggiornato? potrebbe essere tolto)
./documents/Data_Cleaning_Note.docx             --> note e appunti su processo di pulizia (probabilmente non aggiornato)
./documents/Documentazione modifiche dati.docx  --> documentazione di modifiche dei dati (probabilmente non aggiornato)


## Notebook vari
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


## Post generation pipeline
./post-generation           --> cartella di Raimondo strutturata per passaggi da fare dopo la generazione dei dati sintetici, già ben organizzata e con Readme
./post-generation/__init__.py
./post-generation/add_noise.py
./post-generation/add_noise_nb.py
./post-generation/create_validation_source_dataset.ipynb
./post-generation/denormalize.py
./post-generation/dummy_to_categorical.py
./post-generation/inject_missing_values.py
./post-generation/noise_injection.py
./post-generation/noise_injection_from_datalake.py
./post-generation/normalization_settings.json
./post-generation/README.md
./post-generation/residual_par_correction.py
./post-generation/reverse_commands.txt
./post-generation/reverse_from_datalake.bat
./post-generation/reverse_from_datalake.py
./post-generation/reverse_from_local.py
./post-generation/reverse_transformations.py
./post-generation/volume_values_settings.json


