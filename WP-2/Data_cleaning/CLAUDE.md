# CLAUDE.md — Data_cleaning project context

## Scopo del progetto

Pipeline Python per la pulizia, il merge e il post-processing di dati clinici, con output CSV/Excel pronti per l'uso in contesti di Machine Learning.

Il progetto si articola in tre pipeline principali:
1. **Cleaning** — pulizia e normalizzazione dei dataset clinici grezzi
2. **Merge** — unione e riconciliazione dei dataset puliti
3. **Post-generation** — operazioni di validazione e aggiustamento su dati sintetici generati a partire dai dati puliti e mergati

---

## Struttura delle cartelle

```
Data_cleaning/
│
├── data/                 # NON presente nel progetto — i dati sono su datalake esterno
│
├── pipeline/
│   ├── cleaning.py       # Entry point pipeline di pulizia
│   ├── merge.py          # Entry point pipeline di merge
│   └── post_generation/  # Pipeline post-generazione dati sintetici (struttura esistente — non riorganizzare)
│
├── utils/
│   └── helpers.py        # Funzioni di utilità condivise tra pipeline (da consolidare qui)
│
├── notebooks/
│   ├── exploration/      # Notebook esplorativi e di debug — non sono codice di produzione
│   └── validation/       # Notebook per validazione output
│
├── tests/                # Test (se presenti o da aggiungere)
│
├── requirements.txt
├── AUDIT.md              # Inventario funzioni (generato durante il refactoring)
├── CLAUDE.md             # Questo file
└── README.md             # Da scrivere al termine del refactoring
```

---

## Convenzioni di codice

- **Linguaggio**: Python 3
- **Stile docstring**: Google style — usare sempre per tutte le funzioni
- **Type hints**: aggiungere dove possibile nelle funzioni nuove o rifattorizzate
- **Commenti inline**: solo dove la logica non è ovvia, mai ridondanti
- **Intestazione file**: ogni `.py` deve avere una riga descrittiva in cima con lo scopo del modulo

Esempio docstring Google style:
```python
def clean_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Rimuove valori nulli e normalizza i valori di una colonna.

    Args:
        df: Il DataFrame di input.
        col: Il nome della colonna da pulire.

    Returns:
        DataFrame con la colonna pulita.
    """
```

---

## Regole operative per Claude Code

### Limiti assoluti — non fare mai senza approvazione esplicita
- **Non modificare** i Jupyter Notebook usati per la pulizia dati — l'unica eccezione consentita è l'aggiunta di commenti, e solo dopo approvazione
- **Non riorganizzare** la struttura interna di `pipeline/post_generation/`
- **Non eliminare** nessun file o funzione — al massimo proporre cosa eliminare e attendere conferma
- **Non fare refactoring** di più file in un'unica sessione senza approvazione step by step

### Modalità di lavoro obbligatoria — human in the loop
Claude Code deve operare in modalità **proponi → discuti → esegui**:
1. Per ogni azione non banale, **proporre prima** cosa intende fare e attendere conferma
2. **Un file alla volta** — non procedere al file successivo senza approvazione esplicita
3. Dopo ogni modifica, **riassumere cosa è cambiato** prima di proseguire

### Altre regole
- **Non fare modifiche massive in un unico commit** — procedere file per file
- Le funzioni di utilità condivise tra pipeline vanno consolidate in `utils/helpers.py`
- I dati non sono nel repo — vengono letti da un **datalake esterno**. Non creare cartelle `data/` locali

---

## Stato attuale del refactoring

- [~] Fase 1 — Mappatura iniziale completata, da rivedere (file scoperti successivamente)
- [ ] Fase 2 — Audit dei file `.py`
- [ ] Fase 3 — Audit dei notebook
- [ ] Fase 4 — Rimozione codice inutile
- [ ] Fase 5 — Docstring e commenti
- [ ] Fase 6 — Scrittura README

---

## Note di dominio

- I dati sono di natura **clinica** — prestare attenzione a non loggare o esporre valori di colonne sensibili nei print/debug
- I dati sintetici in `post_generation` sono generati a partire dai dati in `data/processed/`
- L'output atteso è sempre in formato **CSV o Excel**
