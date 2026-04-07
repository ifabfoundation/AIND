# AutomaticDownloader - Report Tecnico

## 1. Introduzione

L'**AutomaticDownloader** e' un sistema di acquisizione dati che automatizza il processo di estrazione di dati di ricerca medica dalla piattaforma web ADNI (Alzheimer's Disease Neuroimaging Initiative) e il loro deposito nell'infrastruttura Data Lake del progetto AIND.

Il sistema risolve la necessita' di scaricare in modo ripetibile e strutturato grandi volumi di dati clinici e di neuroimaging dal repository ADNI, arricchendoli con metadati (codice file, popolazione di appartenenza) e depositandoli in un Data Lake centralizzato per le successive fasi di analisi.

---

## 2. Panoramica del Sistema

### 2.1 Obiettivo

Realizzare una pipeline automatica che:
1. Identifichi i file di interesse nel portale ADNI
2. Scarichi i dati (tabelle CSV o file di imaging in formato ZIP)
3. Arricchisca i file con metadati strutturati
4. Depositi i dati nel Data Lake AIND (MinIO + MongoDB)

### 2.2 Architettura Generale

Il sistema e' composto da due macro-componenti tecnologici:

- **Componente JavaScript (Puppeteer)**: automazione del browser per l'estrazione degli identificativi dei file dal portale web ADNI
- **Componente Python**: gestione del download HTTP, elaborazione dei dati e upload verso il Data Lake

```
+-------------------+       +--------------------+       +-------------------+
|   ADNI Web Portal |  -->  | AutomaticDownloader|  -->  |    AIND DataLake  |
| ida.loni.usc.edu  |       |  (Python + Node)   |       | MinIO + MongoDB   |
+-------------------+       +--------------------+       +-------------------+
```

### 2.3 Struttura delle Directory

```
AutomaticDownloader/
├── python/                    # Script Python per il download
│   ├── app.py                # Punto di ingresso principale
│   ├── downloader.py         # Logica di download
│   ├── config.py             # Configurazione (URL, progetto)
│   ├── utils.py              # Utilita' (interfaccia utente)
│   ├── requirements.txt      # Dipendenze Python
│   ├── credentials.json      # Credenziali sessione web
│   └── .dl_client.ini        # Configurazione client Data Lake
├── javascript/                # Script di automazione browser
│   ├── getImageFilesId.js    # Estrazione ID file via Puppeteer
│   └── package.json          # Dipendenze Node.js
├── data/                      # Dati e file di supporto
│   ├── data_ids.csv          # ID dei file da scaricare
│   ├── ADNI_code_pop.csv     # Mappatura codice-popolazione
│   └── data_ids_code_pop.csv # Metadati combinati
└── test.ipynb                 # Notebook di test
```

---

## 3. Pipeline di Download: Dal Sito Web al Data Lake

La pipeline si articola in **6 fasi sequenziali**, descritte di seguito.

### 3.1 Fase 1 - Estrazione degli ID File (JavaScript)

Lo script `getImageFilesId.js` utilizza la libreria **Puppeteer** per controllare un'istanza headless di Chrome e navigare automaticamente nel portale ADNI:

1. Avvia un browser in modalita' headless
2. Naviga su `https://ida.loni.usc.edu/`
3. Accetta la cookie policy
4. Effettua il login con le credenziali
5. Seleziona il progetto ADNI dal menu
6. Naviga nella sezione "Study Files" > "Imaging" > "MR Image Analysis"
7. Cerca i file di interesse per nome
8. Estrae gli attributi `data-id` dal DOM della pagina
9. Salva gli ID nel file `data/data_ids.csv`

**Output**: un file CSV contenente la lista degli identificativi univoci dei file da scaricare.

### 3.2 Fase 2 - Mappatura dei Metadati

Il file `data/ADNI_code_pop.csv` contiene una tabella di mappatura che associa a ciascun nome di file:
- Un **codice file** (`file_code`): identificativo sintetico del tipo di dato
- Una **popolazione** (`population`): le coorti ADNI di appartenenza (1, 2, 3, 4)

Esempio:
```
file_name;file_code;population
RMT_ECOG12PT;DIG_ECOG_I;4
ASHS volume data;ASHS_VOL;1,2,3,4
```

### 3.3 Fase 3 - Autenticazione e Selezione Modalita' (Python)

Lo script `app.py` richiede all'utente (o legge da `credentials.json`):
- **JSESSION_ID**: cookie di sessione del portale ADNI
- **IDA_USC**: cookie di identificazione utente
- **Modalita'**:
  - `T` (Tables): download di tabelle in formato CSV
  - `F` (Files): download di file di imaging in formato ZIP

Le credenziali vengono salvate localmente per un riutilizzo successivo.

### 3.4 Fase 4 - Download HTTP dai Server ADNI

La classe `Downloader` gestisce la comunicazione HTTP con le API XML del portale ADNI:

**Modalita' Table (T)**:
1. Invoca l'endpoint `GET /ajax2/xml/search/downloadTables`
2. Riceve una risposta XML contenente il percorso di download
3. Costruisce l'URL completo ed effettua il download del CSV

**Modalita' Files (F)**:
1. Legge gli ID dal file `data_ids.csv`
2. Invoca l'endpoint `POST /pages/ajax/getStudyData` con la lista degli ID
3. Riceve una risposta XML con il percorso di download
4. Costruisce l'URL completo ed effettua il download dello ZIP

Entrambe le modalita' utilizzano i cookie di sessione (JSESSION_ID e IDA_USC) per l'autenticazione HTTP.

### 3.5 Fase 5 - Elaborazione e Upload nel Data Lake

Una volta scaricati, i dati vengono elaborati e depositati nel Data Lake AIND:

**Per le tabelle (CSV)**:
1. Il contenuto della risposta viene letto come BytesIO
2. Viene parsato come DataFrame pandas
3. Viene caricato nel Data Lake tramite `datalake_client.upload_dataframe()`

**Per i file di imaging (ZIP)**:
1. Si verifica se la risposta e' un singolo file o un archivio ZIP
2. Si estrae lo ZIP in una directory temporanea
3. Per ciascun file estratto:
   - Si confronta il nome con la tabella `ADNI_code_pop.csv`
   - Si arricchiscono i metadati con `file_code` e `population`
   - Si effettua l'upload individuale nel Data Lake
4. Si pulisce la directory temporanea

**Metadati associati a ciascun upload**:
```json
{
  "source": "ADNI",
  "download_type": "study_files",
  "level": "raw",
  "file_code": "<codice_file>",
  "population": ["1", "2", "3", "4"]
}
```

### 3.6 Fase 6 - Deposito nel Data Lake

Il **Data Lake AIND** e' composto da:
- **MinIO**: sistema di object storage (compatibile S3) per la conservazione dei file binari
- **MongoDB**: catalogo dei metadati, che consente query strutturate sui file depositati

Parametri di deposito:
- **Bucket**: `aind`
- **Prefix**: `raw` (indicante dati grezzi, non ancora elaborati)
- **Autenticazione**: tramite file `.dl_client.ini` (username/password)

---

## 4. Descrizione dei Componenti

### 4.1 app.py - Orchestratore

Punto di ingresso del sistema. Gestisce:
- L'interazione con l'utente per la raccolta delle credenziali
- L'inizializzazione del `Downloader`
- L'avvio di un thread separato per il feedback visivo (spinner)
- L'esecuzione del download nella modalita' selezionata
- La gestione degli errori e la pulizia delle risorse

### 4.2 downloader.py - Logica di Download

Classe principale `Downloader` con i seguenti metodi:

| Metodo | Visibilita' | Descrizione |
|--------|-------------|-------------|
| `__retrive_table_download_url()` | Privato | Ottiene URL download tabelle via API XML |
| `__retrive_study_files_download_url()` | Privato | Ottiene URL download file via API XML |
| `download_table_by_url()` | Pubblico | Scarica tabella CSV e la deposita nel Data Lake |
| `download_study_files_by_url()` | Pubblico | Scarica file ZIP, estrae, arricchisce e deposita |

### 4.3 config.py - Configurazione

Contiene le costanti di configurazione:
```python
BASE_URL = "https://ida.loni.usc.edu/"
PROJECT_NAME = "ADNI"
```

### 4.4 utils.py - Utilita'

- `get_cookies_values_from_prompt()`: gestisce il recupero delle credenziali (da file o da input utente)
- `spinner_task()`: fornisce feedback visivo durante le operazioni lunghe

### 4.5 getImageFilesId.js - Automazione Browser

Script Node.js che utilizza Puppeteer per:
- Automatizzare il login al portale ADNI
- Navigare nelle categorie di file
- Estrarre gli ID dei file dal DOM
- Salvare i risultati in formato CSV

---

## 5. Diagramma di Flusso dei Dati

```
                    ┌─────────────────────┐
                    │  Portale Web ADNI   │
                    │ ida.loni.usc.edu    │
                    └────────┬────────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              │              ▼
    ┌──────────────────┐     │    ┌──────────────────┐
    │ Puppeteer (JS)   │     │    │  API XML (HTTP)  │
    │ Estrazione ID    │     │    │  Download dati   │
    └────────┬─────────┘     │    └────────┬─────────┘
             │               │             │
             ▼               │             ▼
    ┌──────────────────┐     │    ┌──────────────────┐
    │  data_ids.csv    │─────┘    │  File CSV / ZIP  │
    └──────────────────┘          └────────┬─────────┘
                                           │
                                           ▼
                                 ┌──────────────────┐
                                 │  Elaborazione    │
                                 │  - Estrazione ZIP│
                                 │  - Matching CSV  │
                                 │  - Arricchimento │
                                 │    metadati      │
                                 └────────┬─────────┘
                                          │
                                          ▼
                               ┌────────────────────┐
                               │   AIND Data Lake   │
                               ├────────────────────┤
                               │  MinIO (file)      │
                               │  MongoDB (metadati)│
                               │  Bucket: aind      │
                               │  Prefix: raw       │
                               └────────────────────┘
```

---

## 6. Dipendenze Tecnologiche

### Python
| Libreria | Versione | Utilizzo |
|----------|----------|----------|
| lxml | >= 4.9.0 | Parsing risposte XML dalle API ADNI |
| requests | >= 2.28.0 | Richieste HTTP per download |
| pandas | >= 1.5.0 | Manipolazione CSV e DataFrame |
| prompt_toolkit | >= 3.0.0 | Input utente avanzato nel terminale |
| dl_client | locale | Client Python per il Data Lake AIND |

### JavaScript
| Libreria | Versione | Utilizzo |
|----------|----------|----------|
| puppeteer | ^24.4.0 | Controllo browser headless Chrome |
| fs | ^0.0.1 | Accesso al filesystem |

---

## 7. Autenticazione e Sicurezza

Il sistema gestisce tre livelli di autenticazione:

1. **Portale ADNI (Web Session)**: cookie `JSESSION_ID` e `IDA_USC` ottenuti manualmente dall'utente tramite il browser e forniti come input al sistema
2. **Portale ADNI (Puppeteer)**: credenziali username/password codificate nello script JavaScript per il login automatico
3. **Data Lake AIND**: credenziali username/password nel file `.dl_client.ini`, utilizzate per ottenere token JWT di autenticazione

---

## 8. Modalita' di Esecuzione

### Esecuzione manuale
```bash
# Fase 1: Estrazione ID (opzionale, se data_ids.csv gia' presente)
cd javascript/
node getImageFilesId.js

# Fase 2: Download e upload
cd python/
python app.py
```

### Flusso interattivo
```
> Insert your JSESSION_ID: [cookie di sessione]
> Insert your IDA_USC: [cookie utente]
> Insert the mode (T for Tables, F for Files): [T/F]
> Downloading... [spinner animato]
> Done!
```

---

## 9. Limiti e Considerazioni

- **Esecuzione manuale**: attualmente il sistema non e' schedulato (nessun cron job o task scheduler)
- **Sessione ADNI**: i cookie di sessione hanno durata limitata e richiedono rinnovo periodico
- **Nessun meccanismo di retry**: in caso di errore di rete, il download fallisce senza tentativi successivi
- **Nessuna deduplicazione**: file gia' presenti nel Data Lake vengono ri-caricati
- **Nessun download incrementale**: ad ogni esecuzione vengono scaricati tutti i file indicati, senza verifica di aggiornamenti

---

## 10. Conclusioni

L'AutomaticDownloader rappresenta una soluzione di data ingestion che automatizza il trasferimento di dati clinici e di neuroimaging dal repository ADNI al Data Lake AIND. Il sistema separa le responsabilita' tra automazione web (JavaScript/Puppeteer) ed elaborazione dati (Python), implementando un flusso strutturato di:

1. **Identificazione** dei file tramite web scraping
2. **Download** tramite API HTTP autenticate
3. **Arricchimento** con metadati da tabelle di lookup
4. **Deposito** in un'infrastruttura di storage distribuita (MinIO + MongoDB)

Questa architettura consente di mantenere un Data Lake aggiornato con i dati grezzi provenienti da ADNI, pronti per le successive fasi di pulizia e analisi del progetto AIND.
