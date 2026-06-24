# STATUS OPERATIVO — Pipeline Generazione Dati Sintetici AIND WP-3
## Problema della Diagnosi: Analisi, Evidenze e Ipotesi di Soluzione

**Data:** Giugno 2026  
**Versione:** 1.0  
**Scopo:** Documento operativo di contesto per lo sviluppo del codice di calcolo della diagnosi nel dataset sintetico  
**Riferimenti principali:** `Analisi_ADCourseMap_DL_code_CP.md`, `ANALISI_COFATTORI_DIAGNOSI_LETTERATURA.md`, `ROADMAP_Miglioramenti_Pipeline_Sintetica.ipynb`, `ANALISI_ADCourseMapDatalake_PostGeneration.ipynb`

---

## 1. STATO ATTUALE DELL'ALGORITMO

### 1.1 Sottosistemi e stato operativo

| Sottosistema | Dove gira | Stato | Note |
|---|---|---|---|
| **ADCourseMapDatalake** (HPC) | Cluster SLURM (`sbatch`) | 🟢 Produzione | Genera coorte sintetica sul Datalake |
| **Validazione** (`notebooks/validation/`) | PC | 🟢 Produzione | TSTR PASS, stazionarietà 6/6 |

### 1.2 Modello scientifico (Koval 2021 / Leaspy v1.5.0)

Il modello è un **modello Bayesiano non-lineare a effetti misti** (MCMC-SAEM, 3000 iterazioni, 90% burn-in). Per ogni paziente sintetico genera una traiettoria longitudinale di biomarcatori via tre parametri individuali:

- **τᵢ (time-shift):** quando inizia la malattia rispetto alla media
- **ξᵢ (log-acceleration):** velocità di progressione
- **sᵢ (sources/space-shift):** quale biomarcatore appare prima (eterogeneità inter-soggetto)

I biomarcatori generati sono normalizzati in **[0,1]** (0=sano, 1=patologico) tramite fixed-scale bounds clinici.

### 1.3 Pipeline di generazione (ADCourseMapDatalake)

**Flusso attivo in produzione:**

```
Datalake (FINALMERGE_*) → loader.py (normalizzazione) → MCMC-SAEM (training)
→ model_parameters.json → simulation.py → generate_virtual_data_with_diagnosis()
→ DataFrame sintetico [ID, TIME, biomarcatori∈[0,1], DX, GENDER, APOE4]
→ Datalake (prefix: synthetic/)
```

**Metodo attivo:** `generate_virtual_data_with_diagnosis()` con proporzioni target:
- CN = 30%, MCI = 45%, Dementia = 25%

---

## 2. STATO DEI PAZIENTI GENERATI

### 2.1 Cosa viene generato

Per ogni paziente sintetico `S00001`, `S00002`, ...:
- N soggetti (default 1000, oppure `same_data_stats=1` → stessa dimensione del training)
- N visite per soggetto (distribuzione gaussiana: media e std dal dataset reale)
- Biomarcatori cognitivi e neuroimaging normalizzati in [0,1]
- Etichette post-hoc: DX, GENDER, APOE4

### 2.2 Problemi critici nel dataset generato

#### 🔴 PROBLEMA P1 — DIAGNOSI INCOERENTE CON I BIOMARCATORI (impatto: Alto)

**Causa:** `generate_virtual_data_with_diagnosis()` usa la DX come filtro pre-KDE, ma le distribuzioni di τ e ξ per CN/MCI/Dementia si **sovrappongono** nei dati ADNI reali. La simulazione Leaspy campiona sempre dalla stessa distribuzione gaussiana globale — i parametri `cofactor_state` passati a `AlgorithmSettings('simulation')` sono **ignorati da Leaspy v1.5.0** (bug noto verificato sul sorgente).

**Risultato concreto:** Un paziente etichettato CN può avere MMSE=14 (range Demenza); un paziente etichettato Demenza può avere MMSE=27 (range CN). L'etichetta DX è puramente decorativa rispetto ai biomarcatori generati.

**Evidenza quantitativa:** Zhou et al. (2026, framework Coogee) misurano che il **45–60% dei record sintetici statisticamente plausibili fallisce la revisione clinica** per inconsistenza label-biomarker anche con modelli sofisticati.

#### 🔴 PROBLEMA P2 — COFATTORI NON CONDIZIONANO LE TRAIETTORIE (impatto: Alto)

**Causa:** GENDER, APOE4, educazione vengono assegnati post-hoc come etichette. Non influenzano né il fitting MCMC-SAEM né la simulazione. La correlazione biologicamente nota tra APOE4/sesso e la progressione AD è **persa** nel dataset sintetico.

**Evidenza quantitativa (Koval 2021):** gli effetti attesi che il dataset sintetico attuale NON preserva:
- Donne: onset ~34–45 mesi prima, progressione 1.3–1.5× più veloce
- APOE4+: onset ~45 mesi prima, progressione 1.17–1.42× più veloce
- Interazione sesso×APOE4 (Altmann 2014): effetto APOE4 più forte nelle donne (HR 1.81 vs 1.27)

#### 🟡 PROBLEMA P3 — DX VARIABILE NEL TEMPO APPIATTITA (impatto: Medio)

**Causa:** `drop_duplicates(subset=['ID'])` in `utils.py` mantiene solo la prima visita per i cofactors. I pazienti converters (CN→MCI→Demenza) vengono classificati con la DX della prima visita — le conversioni sono perse.

#### 🟡 PROBLEMA P4 — RUMORE DP E MISSING NON STRUTTURATI (impatto: Medio)

**Causa (post-generation):** L'iniezione di valori mancanti è casuale (MCAR), mentre i missing nei dati ADNI reali seguono pattern strutturati (MNAR): i soggetti più gravi abbandonano lo studio, il CSF manca nel 60% dei casi perché la puntura lombare è opzionale.

---

## 3. PIPELINE POST-GENERAZIONE (WP2 > DataCleaning > post-generation)

### 3.1 Struttura del post-processing

```
Sintetico Leaspy [normalizzato 0-1, dummy variables]
    │
    ▼ [Step 1 — opzionale] noise_injection.py
    │   → Aggiunge rumore DP (Gaussiano/Laplaciano, epsilon=1.0)
    │   → Le colonne dummy (DX/CN, DX/MCI ecc.) sono ESCLUSE dal rumore
    │
    ▼ [Step 2 — obbligatorio] reverse_transformations.py
    │   → denormalize.py: scala [0,1] → scala clinica (MMSE 0–30, CDRSB 0–18, ecc.)
    │   → dummy_to_categorical.py: DX/CN=1,DX/MCI=0,DX/AD=0 → DX='CN'
    │
    ▼ [Step 3 — opzionale] add_noise_nb.py
    │   → Residual-PAR con KNN pseudo-pairing
    │   → Migliora distribuzioni marginali preservando struttura longitudinale
    │
    ▼ [Step 4 — opzionale] inject_missing_values
        → Inserisce missing values come nel reale (attuale: casuale/MCAR)
```

### 3.2 Stato operativo del post-processing

| Componente | Stato | Problema noto |
|---|---|---|
| `denormalize.py` | ✅ Funzionante | `normalization_settings.json` copre ~650 var ADNI standard |
| `dummy_to_categorical.py` | ✅ Funzionante | Strategia `'nan'` per righe all-zero |
| `noise_injection.py` | ✅ Funzionante | Budget ε=1.0 standard per dati medici |
| `add_noise_nb.py` (Residual-PAR) | ✅ Funzionante (non deterministico) | Manca seed fisso; KNN lento su dataset grandi |
| Missing value injection | ⚠️ Funziona ma non realistica | MCAR invece di MNAR strutturato |

### 3.3 Risultati della validazione (run 2025-12-01)

| Dimensione | Test | Risultato |
|---|---|---|
| **Fidelity** | Kolmogorov-Smirnov | Non riportato esplicitamente |
| **Utility** | TSTR (Train on Synthetic, Test on Real) | **PASS** — degradazione 4.67% |
| **Privacy** | IMS, DCR, Maximum Similarity (Gower) | Non riportato esplicitamente |
| **Temporale** | Stazionarietà ADF/KPSS | **6/6** PASS |
| **Temporale** | ACF/PACF | **5/6** PASS |

**⚠️ Nota critica:** La validazione TSTR usa `--target DX` ma la DX in input è l'etichetta decorativa (incoerente con i biomarcatori). I buoni risultati TSTR potrebbero riflettere che il modello apprende la struttura dei dati ma **non** la coerenza clinica. La Consistency (Zamzmi FDA 2025) non è attualmente misurata.

---

## 4. IPOTESI ATTUALI PER IL CALCOLO DELLA DIAGNOSI

### 4.1 Posizionamento rispetto alla letteratura

La letteratura identifica **tre paradigmi** per la diagnosi nella generazione sintetica:

| Paradigma | Logica | Stato nella pipeline AIND |
|---|---|---|
| **P0 — DX come filtro** | Training filtrato per DX, etichetta assegnata | ⬅️ **Approccio attuale** |
| **P1 — DX come soglia** | DX = f(valori biomarcatori), assegnata *dopo* la generazione | ⬅️ **Target a breve termine** |
| **P2 — DX come condizione generativa** | DX è input del generatore; biomarcatori coerenti per costruzione | → Obiettivo futuro (DDPM, VAMBN) |

L'approccio attuale (P0) è il più debole; le due ipotesi di soluzione proposte (Metodo A e Metodo B) puntano entrambe al paradigma P1, con approcci diversi.

---

## 5. METODO A — CLASSIFICATORE ML ALLENATO SU ADNI REALE

### 5.1 Idea

Addestrare un classificatore (es. rete neurale, Random Forest, XGBoost) sui dati ADNI reali con:
- **Input:** biomarcatori di ogni visita (MMSE, ADAS13, CDRSB, volumi MRI, ecc.) + eventualmente DX della visita precedente
- **Output:** DX predetta (CN / MCI / AD)
- **Applicazione:** post-generazione, ogni visita sintetica viene classificata

### 5.2 Punti di forza

- **Data-driven:** impara relazioni non-lineari tra biomarcatori e DX dai dati reali
- **Longitudinale:** se si include la DX della visita precedente, cattura le transizioni di stato (CN→MCI→Demenza)
- **Validabile:** si può confrontare la distribuzione di DX predetta vs. target sul set reale tenuto fuori

### 5.3 Criticità e decisioni da prendere

**Q1 — Includere i cofattori (GENDER, APOE4, educazione)?**

> **Raccomandazione: NON includerli nell'iterazione iniziale.**
>
> I cofattori nel dataset sintetico attuale sono **incollati post-hoc** (P2 del documento — problema noto). Includere APOE4 e GENDER come feature del classificatore significherebbe allenarlo su coppie (cofattore, biomarcatore) che nel reale sono coerenti, ma nel sintetico sono casuali. Il classificatore imparerebbe una relazione che non esiste nel sintetico → le predizioni di DX sarebbero distorte. 
>
> **Eccezione:** se prima si implementa la stratificazione per sesso×APOE4 (modifica C.1 della roadmap), i cofattori diventano affidabili e possono essere inclusi.

**Q2 — Includere la DX della visita precedente?**

> **Raccomandazione: SÌ, ma con attenzione alle sequenze iniziali.**
>
> La DX precedente è una feature molto informativa (la transizione CN→Demenza senza passare per MCI è rara). Tuttavia, la DX della prima visita nel dataset sintetico è ancora decorativa (P0) — quindi la feature "DX precedente" è inaffidabile per la prima visita. Soluzione: trattare la DX della prima visita come mancante o usare un placeholder neutro, poi propagare la DX predetta nelle visite successive. Oppure valutare di calcolare la diagnosi anche alla visita 0 baseline avendo la diagnosi precedente come mancante, punto da studiare se fattibile.

**Q3 — Quale architettura?**

Opzioni (dal più semplice al più sofisticato):

| Modello | Pro | Contro | Riferimento |
|---|---|---|---|
| **XGBoost/Random Forest** | Interpretabile, robusto con dati tabellari | Non cattura dipendenze temporali tra visite | Standard ADNI classification |
| **Logistic Regression stratificata** | Molto interpretabile, allineata ai criteri clinici | Meno potente per relazioni non-lineari | Baseline classifica |
| **LSTM / GRU** | Cattura sequenze temporali, DX precedente come stato | Richiede sequenze allineate temporalmente | Maheux 2022 (minimalRNN ha questa logica) |
| **Transformer tabellare (TabPFN, SAINT)** | SOTA su dati tabellari, pochi dati | Meno interpretabile | Recente letteratura tabellare |

> **Raccomandazione pratica:** iniziare con XGBoost (singola visita) o una Random Forest con feature della visita corrente + DX precedente codificata come one-hot. Poi eventualmente upgradare a LSTM per il caso longitudinale.

### 5.4 Riferimenti di letteratura rilevanti

| Riferimento | Rilevanza |
|---|---|
| **Heeg et al. (2024, medRxiv `10.1101/2024.02.05.24302140`)** | Usa criteri ML-like (modelli R) con feature MMSE per classificare soggetti eleggibili al trial — approccio ibrido ML/regole |
| **Koval et al. (2021)** | I parametri τ e ξ personalizzati sono feature informative per la classificazione — possono essere inclusi se disponibili --> Attialmente non credo siano disponibili da verificare, altrimente può essere una modifica da implementare nel codice di ADCoruseMapDataLake, studiare meglio se davvero utili. |
| **Maheux 2022** (minimalRNN) | L'RNN-AD originale usa CrossEntropy per DX: approccio già testato su ADNI ma codice bloccato da bug |
| **Zamzmi et al. (2025, FDA)** | La DX predetta da ML deve soddisfare il criterio FDA di *Consistency* — usarlo come metrica di validazione |

---

## 6. METODO B — REGOLE CLINICHE BASATE SU LINEE GUIDA FDA/NIA-AA

### 6.1 Idea

Implementare un sistema di regole derivato dai **criteri diagnostici NIA-AA 2018** (aggiornamento dei criteri Jack 2013) per assegnare DX post-generazione sulla base dei valori effettivi dei biomarcatori.

### 6.2 Implementazione (già proposta come modifica C.2 nella roadmap)

**Soglie MMSE + CDRSB (versione semplificata, solo downstream):**

```python
def assign_dx_from_biomarkers(mmse, cdrsb=None):
    if cdrsb is not None:
        if cdrsb <= 0.5 and mmse >= 24:   return 'CN'
        elif cdrsb <= 4.0:                 return 'MCI'
        else:                              return 'Dementia'
    else:  # solo MMSE come proxy
        if mmse >= 24:                     return 'CN'
        elif mmse >= 18:                   return 'MCI'
        else:                              return 'Dementia'
```

**Dove applicarla:** `WP-2/Data_cleaning/post-generation/reverse_transformations.py`, come Step 4 dopo la denormalizzazione (sulla scala clinica, non normalizzata).

### 6.3 Estensione con la cascata AT(N) (versione completa — modifica C.4)

**Jack et al. 2013 / NIA-AA 2018:** la diagnosi biologicamente confermata richiede verifica lungo la cascata:

```
Amiloide (CSF/PET) → Tau (CSF) → Neurodegenerazione (MRI, FDG-PET) → Cognizione → Funzione
```

Un record clinicamente coerente:
- **CN:** amiloide normale E cognizione normale (MMSE ≥ 24, CDRSB ≤ 0.5)
- **MCI:** almeno un biomarcatore upstream alterato CON compromissione cognitiva lieve (18 ≤ MMSE < 24)
- **Demenza:** pattern biomarker avanzato E compromissione cognitiva+funzionale (MMSE < 18 E CDRSB > 4.0)

### 6.4 Punti di forza

- **Interpretabile e auditabile:** le regole sono esplicite e verificabili
- **Allineato agli standard normativi:** FDA 7Cs Consistency criterion (Zamzmi 2025) è soddisfatto per costruzione
- **Nessun overfitting:** non dipende dai dati di training
- **Immediato:** ~30 righe di codice, nessuna dipendenza aggiuntiva

### 6.5 Limitazioni

- **Distribuzione DX non controllata:** la proporzione CN/MCI/Demenza risultante dipende dalla distribuzione dei biomarcatori generati — potrebbe differire dai target (30/45/25%)
- **MCI è ambiguo:** il range MCI (MMSE 18–23) include casi eterogenei; le soglie MMSE/CDRSB da sole non catturano lo staging biologico completo
- **Biomarcatori upstream spesso mancanti:** amiloide e tau CSF mancano nel 60% dei soggetti ADNI — la cascata AT(N) completa è applicabile solo ai soggetti con questi dati

### 6.6 Riferimenti di letteratura rilevanti

| Riferimento | Rilevanza |
|---|---|
| **Jack et al. (2013, Lancet Neurology `10.1016/S1474-4422(12)70291-0`)** | Modello dinamico cascata biomarker — definizione normativa della DX in AD |
| **Zamzmi et al. (2025, Comm. Engineering `10.1038/s44172-025-00450-1`)** | FDA 7Cs: Consistency come criterio separato dalla Fidelity — argomento normativo per il Metodo B |
| **Zhou et al. (2026, Coogee)** | Quantifica il 45–60% di incoerenza senza regole di consistenza — quantifica il problema che Metodo B risolve |
| **SuStaIn — Young et al. (2018, Nat. Comm. `10.1038/s41467-018-05892-0`)** | Staging biologico più preciso che potrebbe estendere il Metodo B per la diagnosi di MCI |
| **Heeg et al. (2024)** | Implementa criteri MMSE+biomarcatori su ADNI per generazione trial sintetici — caso più vicino al Metodo B su ADNI |

---

## 7. STRATEGIA DI SVILUPPO: VALUTAZIONE DELL'IPOTESI COMBINATA

### 7.1 Valutazione dell'ipotesi "Metodo A + Metodo B in parallelo"

**La proposta è solida e ben motivata dalla letteratura.** Ecco il razionale per ciascuno:

**Metodo B come "gold standard" clinicamente fondato:**
Il Metodo B (regole) deve essere il riferimento normativo — è esattamente ciò che la letteratura definisce come P1 e ciò che FDA richiede con il criterio di Consistency. Non ha senso avere un dataset sintetico che viola i criteri diagnostici clinici.

**Metodo A come "validatore data-driven":**
Un classificatore ML allenato su ADNI reale fornisce una stima data-driven di DX che non dipende da soglie fisse. Se Metodo A e Metodo B **concordano**, il record è clinicamente coerente con alta confidenza. Se **discordano**, il record è candidato a revisione o esclusione.

**Il disaccordo A≠B come metrica di qualità:**
La proporzione di record dove A e B discordano è essa stessa una metrica utile — misura quanti record sintetici sono biologicamente "borderline" o incoerenti. Questo può essere usato come nuovo criterio nella suite di validazione (accanto a TSTR, KS, DCR).

### 7.2 Ordine di implementazione consigliato

| Fase | Azione | File | Complessità | Priorità |
|---|---|---|---|---|
| **0 (prerequisito)** | Verificare l'effettiva incoerenza DX-MMSE nel dataset sintetico corrente | analisi esplorativa | Bassa | **Immediata** |
| **1a** | Metodo B: `assign_dx_from_biomarkers(MMSE, CDRSB)` | `post-generation/reverse_transformations.py` | Bassa (~30 righe) | **Alta** |
| **1b** | Metodo A: addestrare XGBoost su ADNI reale, features: biomarcatori, NO cofattori | notebook di sviluppo | Media (2–4 giorni) | **Alta** |
| **2** | Cross-validazione: misurare accordo A vs B per record | script di analisi | Bassa | **Media** |
| **3** | Aggiungere DX_precedente come feature per Metodo A | upgrade notebook | Media | **Media** |
| **4** | Metodo B esteso: aggiungere cascata AT(N) se biomarcatori disponibili | `reverse_transformations.py` | Media (~50 righe) | **Media** |
| **5** | Metodo A: upgrade a LSTM/GRU per il caso longitudinale | notebook ML | Alta | **Bassa (futura)** |

### 7.3 Schema architetturale del calcolo di DX

```
Dataset sintetico post-denormalizzazione (scala clinica, valori interi)
          │
          ▼
  ┌──────────────────────────┐    ┌──────────────────────────┐
  │   METODO B (Regole)      │    │   METODO A (ML)          │
  │  assign_dx_from_biomark()│    │  clf.predict(features)   │
  │  Jack 2013 + NIA-AA 2018 │    │  XGBoost su ADNI reale   │
  └────────────┬─────────────┘    └────────────┬─────────────┘
               │                               │
               └──────────────┬────────────────┘
                              ▼
                    Accordo A == B?
                    │          │
                  Sì: DX=X   No: DX='Discordante' (flag qualità)
                              │
                    Decisione operativa:
                    - Escludere il record?
                    - Usare DX_B (normativo)?
                    - Usare DX come 'Unknown'?
```

---

## 8. BUG E FIX OPERATIVI NOTI

| ID | Bug | File | Fix |
|---|---|---|---|
| **B1** | `cofactor_state` in AlgorithmSettings ignorato da Leaspy | `model.py` | Documentare o rimuovere il parametro |
| **B2** | `sources_method='full_kde'` non implementato in Leaspy v1.5.0 | `model.py` | Rimuovere — il sampling è sempre N(0,1) normalizzato |
| **B3** | `ip` passato a `model.simulate(ip, ...)` ignorato da Leaspy | `model.py` | Documentare; la personalizzazione pre-simulazione è inutile |
| **B4** | Blocco `if/if/else` invece di `elif` in `main.py` → sovrascrive `result_path` | `main.py` | Correggere a `if/elif/else` |
| **B5** | `requirements.txt` in encoding UTF-16 | `ADCourseMapDatalake/` | Usare quello di `ADCourseMap` |
| **B6** | Path hardcoded `/home/IFAB/WORK/ADCourseMap/` | `main.py` | Usare variabile d'ambiente o argomento CLI |
| **B7** | PAR non deterministica (no seed fisso) | `add_noise*.py` | Aggiungere `random_state=42` |
| **B8** | Modello calibrato NON va sul Datalake | HPC workflow | Procedura manuale documentata in RUNBOOK |

---

## 9. DECISIONI ARCHITETTURALI PENDENTI

### 9.1 Upgrade Leaspy v1.5.0 → v2.1.0?

**Risposta: NON urgente, NON risolve i problemi di cofactori/diagnosi.**

Leaspy v2.1.0 (10 giugno 2026) ha API incompatibile → riscrittura completa di tutti i file `code/`. I cofactori rimangono analisi post-hoc anche in v2.x — il fitting MCMC-SAEM non li usa in nessuna versione. Raccomandazione: implementare C.1–C.3 su v1.5.0, pianificare l'upgrade come task separato (Q3/Q4 2026).

### 9.2 Migrazione a MultiNODEs o DPMoSt?

**MultiNODEs (Wendland & Fröhlich 2022, `10.1038/s41746-022-00666-x`):** cofattori integrati nell'ODE (`dz/dt = f(z, t, covariates)`), testato su NACC, codice pubblico. Risolverebbe L1/L2 nativamente ma richiede riscrittura completa.

**DPMoSt (Archetti 2023):** migliore ricostruzione traiettorie di Leaspy su benchmark ADNI, supporta covariabili nel fitting, include clustering sottotipi. Richiede riscrittura completa.

**Raccomandazione:** benchmark preliminare Leaspy+C.1–C.5 vs. DPMoSt/MultiNODEs prima di decidere. Target: WP-3 v2.0.

---

## 10. RIFERIMENTI PRINCIPALI

| Paper | DOI | Rilevanza |
|---|---|---|
| **Koval et al. (2021), Scientific Reports** | `10.1038/s41598-021-87434-1` | Modello base; valori attesi per validazione |
| **Jack et al. (2013), Lancet Neurology** | `10.1016/S1474-4422(12)70291-0` | Cascata biomarker — vincolo normativo per Metodo B |
| **Zamzmi et al. (2025), Comm. Engineering** | `10.1038/s44172-025-00450-1` | FDA 7Cs Consistency — argomento normativo |
| **Heeg et al. (2024), medRxiv** | `10.1101/2024.02.05.24302140` | Generazione trial sintetico ADNI con criteri DX espliciti |
| **Young et al. (2018), Nat. Comm.** | `10.1038/s41467-018-05892-0` | SuStaIn — staging biologico per MCI ambiguo |
| **Lin et al. (2020), ACM IMC** | `10.1145/3419394.3423643` | DoppelGANger — pattern per stratificazione per cofattori |
| **Xu et al. (2019), NeurIPS** | `arXiv:1907.00503` | CTGAN — distribuzione congiunta cofattori |
| **Altmann et al. (2014), Annals Neurology** | — | Interazione sesso×APOE4 — motivazione C.1 |
| **Wendland & Fröhlich (2022), npj Dig. Med.** | `10.1038/s41746-022-00666-x` | MultiNODEs — alternativa a lungo termine |
| **Achterberg et al. (2024), BMC** | `10.1186/s12874-024-02304-4` | Protocollo validazione longitudinale per sottogruppi |
| **Moslemi & Peyvandi (2025), arXiv** | `arXiv:2511.20704` | DDPM class-conditioned — paradigma P2 su NACC |

---

## 11. CHECKLIST OPERATIVA — PASSI IMMEDIATI

- [ ] **Step 0:** Analisi esplorativa su dataset sintetico esistente — misurare proporzione di record con DX incoerente con MMSE (CN con MMSE<18, Demenza con MMSE>24)
- [ ] **Step 1a:** Implementare `assign_dx_from_biomarkers()` in `reverse_transformations.py` (Metodo B, ~30 righe)
- [ ] **Step 1b:** Addestrare XGBoost su ADNI reale per classificazione DX da biomarcatori (Metodo A, senza cofattori)
- [ ] **Step 2:** Applicare entrambi i metodi al dataset sintetico e misurare il tasso di accordo
- [ ] **Step 3:** Decidere la strategia in caso di disaccordo A≠B (esclusione, DX_B normativa, flag)
- [ ] **Step 4 (dopo):** Stratificazione KDE per sesso×APOE4 (modifica C.1) — prerequisito per includere i cofattori nel Metodo A

---

*Documento creato: giugno 2026. Aggiornare progressivamente man mano che le modifiche vengono implementate.*  
*Complementare a: `ROADMAP_Miglioramenti_Pipeline_Sintetica.ipynb`, `ANALISI_COFATTORI_DIAGNOSI_LETTERATURA.md`, `ANALISI_ADCourseMapDatalake_PostGeneration.ipynb`*
