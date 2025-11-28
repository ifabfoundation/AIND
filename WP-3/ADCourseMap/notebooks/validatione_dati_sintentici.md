# Guida completa alla validazione di dati sintetici: teoria e pratica

## Indice

1. [Introduzione e filosofia](#1-introduzione-e-filosofia)
2. [Framework teorico fondamentale](#2-framework-teorico-fondamentale)
3. [La domanda da un milione di dollari](#3-la-domanda-da-un-milione-di-dollari)
4. [Le tre dimensioni della validazione](#4-le-tre-dimensioni-della-validazione)
5. [Il Maximum Similarity Test](#5-il-maximum-similarity-test)
6. [Validazione per modelli di progressione (Leaspy/DCM)](#6-validazione-per-modelli-di-progressione-leaspy-dcm)
7. [Implementazione pratica completa](#7-implementazione-pratica-completa)
8. [Report di validazione](#8-report-di-validazione)
9. [Conclusioni e raccomandazioni](#9-conclusioni-e-raccomandazioni)

---

## 1. Introduzione e filosofia

### 1.1 Il problema centrale

Quando generiamo dati sintetici, creiamo un modello dei dati reali (osservati) e lo utilizziamo per generare nuovi dati. I dati osservati provengono da esperienze del mondo reale e possono essere considerati un **campione casuale da una "distribuzione genitore"** — la vera distribuzione sottostante che non conosceremo mai e che dobbiamo stimare.

**La domanda fondamentale:**
> *I miei dataset reali e sintetici sono campioni casuali dalla stessa distribuzione genitore?*

Se la risposta è sì, abbiamo raggiunto il jackpot:
- **Fidelity**: I dati sintetici possiedono le stesse proprietà statistiche dei dati reali
- **Utility**: Sono altrettanto utili per task come regressione o classificazione
- **Privacy**: Non c'è rischio di identificare i dati osservati

### 1.2 Il principio di non-compromissione

**⚠️ PRINCIPIO FONDAMENTALE:**

Le tre dimensioni (fidelity, utility, privacy) **non sono sullo stesso piano**. Se i dati sintetici falliscono il test di privacy (sono troppo vicini ai dati reali), significa che il modello è in **overfitting** e le metriche di fidelity e utility diventano **prive di significato**.

> Non esiste un "2 su 3". La privacy è prerequisito per tutto il resto.

---

## 2. Framework teorico fondamentale

### 2.1 Distribuzione genitore e campionamento

```
Distribuzione Genitore P(X)
           │
           ├─────────────────────┬──────────────────────┐
           │                     │                      │
    Campione Reale          Modello M̂(X)        Campione Sintetico
      (Osservato)                │                  (Generato)
           │                     │                      │
           └─────────────────────┴──────────────────────┘
                        Devono provenire da P(X)
```

**Obiettivo**: Verificare che Real ~ P(X) e Synthetic ~ P(X), e quindi Real ~ Synthetic

### 2.2 Tipologie di modelli generativi

| Approccio | Metodo | Pro | Contro |
|-----------|--------|-----|---------|
| **Copula-based** | Stima copula + marginali | Teoricamente solido | Performance mediocre su dati complessi |
| **GAN-based** | Adversarial training | Popolare, flessibile | Instabile, performance inconsistente |
| **VAE-based** | Encoder-decoder latente | Smooth latent space | Può produrre output sfocati |
| **Sequential Imputation** | Stima P(xᵢ\|x₁,...,xᵢ₋₁) | Consistente, previene overfitting | Sensibile all'ordine |
| **Bayesian Mixed-Effects** | Random + fixed effects | Modella variabilità individuale | Richiede struttura longitudinale |

**Evidenza empirica** (da studi su UCI datasets):
- Migliori performance: sequential imputation (synthpop, UNCRi)
- Performance mediocri: GAN-based (CopulaGAN, CTGAN)
- Problemi privacy frequenti: TVAE, synthpop (se mal configurato)

### 2.3 Leaspy e Disease Course Mapping

Per dati longitudinali di progressione di malattie (es. Alzheimer), **Leaspy** (LEArning Spatiotemporal Patterns) usa:

**Modello matematico:**
```
y_i(t) = g(φ⁻¹(t - τᵢ, exp(ξᵢ)) | p₀, w_i)

dove:
- τᵢ: time-shift (onset individuale)
- ξᵢ: log-acceleration (velocità progressione)
- w_i: variabilità spaziale (ordine biomarcatori)
- p₀: parametri popolazione (traiettoria media)
```

**Articoli fondamentali:**
1. Schiratti et al. (2017) - "A Bayesian mixed-effects model to learn trajectories"
2. Koval et al. (2021) - "AD Course Map charts Alzheimer's disease progression"
3. Koval et al. (2023) - "Forecasting individual progression trajectories"

---

## 3. La domanda da un milione di dollari

### 3.1 L'intuizione geometrica

Se due dataset sono campioni casuali dalla stessa distribuzione:
- Punti di un dataset dovrebbero essere **intersparsed** con quelli dell'altro
- Un punto dovrebbe essere, in media, **tanto vicino ai suoi vicini nello stesso set quanto ai vicini nell'altro set**

Se un dataset è una perturbazione dell'altro:
- Punti saranno **più vicini ai corrispondenti nell'altro set** che ai propri vicini

### 3.2 Implicazioni per la validazione

Questa intuizione porta a tre approcci complementari:

1. **Maximum Similarity Test** (globale)
2. **Distance to Closest Record** (privacy-focused)
3. **Membership Inference Attack** (adversarial)

---

## 4. Le tre dimensioni della validazione

### 4.1 Fidelity (Fedeltà statistica)

**Definizione**: I dati sintetici preservano le proprietà statistiche dei dati reali?

**Livelli di verifica:**

#### Livello 1: Distribuzioni univariate
```python
# Test di Kolmogorov-Smirnov per variabili continue
H₀: Real ~ Synthetic
Criterio: p-value > 0.05

# KL Divergence per variabili categoriche
D_KL(P || Q) = Σ P(x) log(P(x)/Q(x))
Criterio: D_KL < 0.1
```

#### Livello 2: Correlazioni bivariate
```python
# Differenza assoluta media tra matrici di correlazione
MAD = mean(|Corr_real - Corr_synth|)
Criterio: MAD < 0.15

# Meta-correlazione
r = corr(flatten(Corr_real), flatten(Corr_synth))
Criterio: r > 0.8
```

#### Livello 3: Strutture multivariate complesse
- PCA: autovalori e autovettori simili
- Manifold learning: preservazione di t-SNE/UMAP embeddings
- Covariance structure: test di Bartlett

### 4.2 Utility (Utilità pratica)

**Definizione**: I dati sintetici sono altrettanto utili per task predittivi?

**Test standard: TSTR (Train on Synthetic, Test on Real)**

```
1. Splitta Real: Train (60%), Validation (20%), Test (20%)
2. Modello TSTR: Train su Synthetic → Test su Real_test
3. Modello TRTR: Train su Real_train → Test su Real_test
4. Confronta: TSTR vs TRTR

Criterio di successo:
|Performance_TSTR - Performance_TRTR| < 5%
```

**Metriche specifiche per task:**
- Classificazione: Accuracy, F1, AUC-ROC
- Regressione: MAE, RMSE, R²
- Ranking: Spearman's ρ per feature importance

### 4.3 Privacy

**Definizione**: I dati sintetici non rivelano informazioni sui dati reali?

**Gerarchia di rischi:**

1. **Livello 1 - Disclosure diretto**: Exact matches
2. **Livello 2 - Re-identificazione**: Punti troppo vicini
3. **Livello 3 - Inferenza**: Membership inference attacks

**Test principali:**

#### Identical Match Share (IMS)
```python
IMS = |Real ∩ Synthetic| / |Synthetic|
Criterio: IMS = 0
```

#### Distance to Closest Record (DCR)
```python
Per ogni punto sintetico:
  d_train = distanza al più vicino in Train
  d_holdout = distanza al più vicino in Holdout

Ratio = #{d_train < d_holdout} / |Synthetic|
Criterio: Ratio ≤ 0.50
```

---

## 5. Il Maximum Similarity Test

### 5.1 Teoria

**Similarità di Gower** (per dati mixed-type):

```
s_Gower(x, y) = Σᵢ wᵢ · sᵢ(xᵢ, yᵢ) / Σᵢ wᵢ

dove:
- Numeriche: sᵢ = 1 - |xᵢ - yᵢ|/rangeᵢ
- Categoriche: sᵢ = 1 se xᵢ = yᵢ, 0 altrimenti
- Missing: wᵢ = 0
```

**Definizioni:**

1. **Maximum Intra-Set Similarity (MISS)**:
   Per ogni punto in un dataset, similarità al suo **nearest neighbor nello stesso dataset**

2. **Maximum Cross-Set Similarity (MCSS)**:
   Per ogni punto in Synthetic, similarità al suo **nearest neighbor in Real**

### 5.2 Il test

**Ipotesi:**

```
H₀: Real e Synthetic sono campioni da P(X)
H₁: Synthetic è perturbazione di Real
```

**Procedura:**

1. Calcola MISS_real: distribuzione similarità intra-set per Real
2. Calcola MISS_synth: distribuzione similarità intra-set per Synthetic
3. Calcola MCSS: distribuzione similarità cross-set Synthetic→Real

**Criteri di successo:**

```
✓ MISS_real ≈ MISS_synth  (distribuzione simile)
✓ MCSS ≈ MISS_real        (non troppo vicini)
✓ mean(MCSS) ≤ mean(MISS_real)  (privacy)
```

### 5.3 Metrica unificata

**Quality Score proposto:**

```
Q = mean(MCSS) / mean(MISS_real)

Interpretazione:
- Q ≈ 1.0: Perfetto
- Q < 1.0: Buono (meno simili)
- Q > 1.0: FALLITO (privacy compromessa, overfitting)
```

### 5.4 Implementazione

```python
from sklearn.metrics import pairwise_distances
import numpy as np

def gower_similarity(X, Y=None, categorical_features=None):
    """
    Calcola matrice di similarità di Gower
    
    Parameters:
    -----------
    X : array-like of shape (n_samples_X, n_features)
    Y : array-like of shape (n_samples_Y, n_features), optional
    categorical_features : list of int, indices delle feature categoriche
    
    Returns:
    --------
    similarity_matrix : array of shape (n_samples_X, n_samples_Y)
    """
    if Y is None:
        Y = X
    
    n_features = X.shape[1]
    if categorical_features is None:
        categorical_features = []
    
    similarity_matrix = np.zeros((X.shape[0], Y.shape[0]))
    
    for i in range(n_features):
        if i in categorical_features:
            # Categorical: 1 se uguali, 0 altrimenti
            feature_sim = (X[:, i, np.newaxis] == Y[:, i]).astype(float)
        else:
            # Numeric: 1 - |diff|/range
            range_i = np.ptp(np.concatenate([X[:, i], Y[:, i]]))
            if range_i > 0:
                diff = np.abs(X[:, i, np.newaxis] - Y[:, i])
                feature_sim = 1 - (diff / range_i)
            else:
                feature_sim = np.ones((X.shape[0], Y.shape[0]))
        
        similarity_matrix += feature_sim
    
    # Media delle similarità delle feature
    similarity_matrix /= n_features
    
    return similarity_matrix

def maximum_similarity_test(real_data, synthetic_data, categorical_features=None):
    """
    Esegue Maximum Similarity Test
    
    Returns:
    --------
    results : dict con MISS_real, MISS_synth, MCSS, quality_score
    """
    # Converti a numpy
    X_real = real_data.values if hasattr(real_data, 'values') else real_data
    X_synth = synthetic_data.values if hasattr(synthetic_data, 'values') else synthetic_data
    
    # 1. Maximum Intra-Set Similarity per Real
    sim_real_real = gower_similarity(X_real, X_real, categorical_features)
    np.fill_diagonal(sim_real_real, -1)  # Escludi diagonale
    miss_real = np.max(sim_real_real, axis=1)
    
    # 2. Maximum Intra-Set Similarity per Synthetic
    sim_synth_synth = gower_similarity(X_synth, X_synth, categorical_features)
    np.fill_diagonal(sim_synth_synth, -1)
    miss_synth = np.max(sim_synth_synth, axis=1)
    
    # 3. Maximum Cross-Set Similarity (Synthetic → Real)
    sim_synth_real = gower_similarity(X_synth, X_real, categorical_features)
    mcss = np.max(sim_synth_real, axis=1)
    
    # 4. Quality Score
    quality_score = np.mean(mcss) / np.mean(miss_real)
    
    # 5. Test statistico (KS test)
    from scipy import stats
    ks_real_synth = stats.ks_2samp(miss_real, miss_synth)
    ks_real_cross = stats.ks_2samp(miss_real, mcss)
    
    return {
        'miss_real': miss_real,
        'miss_synth': miss_synth,
        'mcss': mcss,
        'mean_miss_real': np.mean(miss_real),
        'mean_miss_synth': np.mean(miss_synth),
        'mean_mcss': np.mean(mcss),
        'quality_score': quality_score,
        'ks_real_vs_synth': {
            'statistic': ks_real_synth.statistic,
            'pvalue': ks_real_synth.pvalue
        },
        'ks_real_vs_cross': {
            'statistic': ks_real_cross.statistic,
            'pvalue': ks_real_cross.pvalue
        },
        'passed_privacy': quality_score <= 1.0,
        'passed_fidelity': ks_real_synth.pvalue > 0.05,
        'overall_passed': quality_score <= 1.0 and ks_real_synth.pvalue > 0.05
    }

# Visualizzazione
def plot_similarity_distributions(results):
    """
    Plotta distribuzioni di similarità
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Istogrammi
    axes[0].hist(results['miss_real'], bins=30, alpha=0.5, 
                 label='MISS Real', density=True)
    axes[0].hist(results['miss_synth'], bins=30, alpha=0.5, 
                 label='MISS Synthetic', density=True)
    axes[0].hist(results['mcss'], bins=30, alpha=0.5, 
                 label='MCSS', density=True)
    axes[0].axvline(results['mean_miss_real'], color='blue', 
                    linestyle='--', label=f"μ MISS Real: {results['mean_miss_real']:.4f}")
    axes[0].axvline(results['mean_mcss'], color='red', 
                    linestyle='--', label=f"μ MCSS: {results['mean_mcss']:.4f}")
    axes[0].set_xlabel('Similarity')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Distribution of Maximum Similarities')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Box plot
    box_data = [results['miss_real'], results['miss_synth'], results['mcss']]
    axes[1].boxplot(box_data, labels=['MISS Real', 'MISS Synth', 'MCSS'])
    axes[1].set_ylabel('Similarity')
    axes[1].set_title('Comparison of Similarity Distributions')
    axes[1].grid(alpha=0.3)
    
    # Aggiungi quality score
    fig.suptitle(f"Quality Score: {results['quality_score']:.4f} | "
                 f"Privacy: {'✓ PASS' if results['passed_privacy'] else '✗ FAIL'} | "
                 f"Fidelity: {'✓ PASS' if results['passed_fidelity'] else '✗ FAIL'}",
                 fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    return fig
```

---

## 6. Validazione per modelli di progressione (Leaspy/DCM)

### 6.1 Validazione del modello (pre-generazione)

Prima di generare dati sintetici, **valida il modello Leaspy stesso**:

#### Cross-validation con holdout
```python
# Split strategico per dati longitudinali
def split_longitudinal_data(data, train_ratio=0.6, val_ratio=0.2):
    """
    Split che mantiene tutti i timepoint di un soggetto insieme
    """
    unique_subjects = data['ID'].unique()
    n_subjects = len(unique_subjects)
    
    # Shuffle soggetti
    np.random.shuffle(unique_subjects)
    
    # Split
    n_train = int(n_subjects * train_ratio)
    n_val = int(n_subjects * val_ratio)
    
    train_ids = unique_subjects[:n_train]
    val_ids = unique_subjects[n_train:n_train+n_val]
    test_ids = unique_subjects[n_train+n_val:]
    
    train_data = data[data['ID'].isin(train_ids)]
    val_data = data[data['ID'].isin(val_ids)]
    test_data = data[data['ID'].isin(test_ids)]
    
    return train_data, val_data, test_data

# Training con MCMC-SAEM
from leaspy import Leaspy, Data, AlgorithmSettings

def train_and_validate_leaspy(train_data, val_data, test_data):
    """
    Train Leaspy e valida su holdout
    """
    # Converti in formato Leaspy
    leaspy_train = Data.from_dataframe(train_data)
    leaspy_val = Data.from_dataframe(val_data)
    leaspy_test = Data.from_dataframe(test_data)
    
    # Modello logistic
    leaspy = Leaspy("logistic")
    
    # Settings MCMC-SAEM
    settings = AlgorithmSettings('mcmc_saem', n_iter=10000, seed=42)
    
    # Fit
    leaspy.fit(leaspy_train, settings)
    
    # Valida su validation set
    val_scores = leaspy.estimate(leaspy_val)
    
    # Log-likelihood marginale
    val_loglik = leaspy.compute_log_likelihood(leaspy_val)
    
    # BIC/AIC
    n_params = len(leaspy.model.parameters)
    n_obs = len(val_data)
    bic = -2 * val_loglik + n_params * np.log(n_obs)
    aic = -2 * val_loglik + 2 * n_params
    
    return {
        'model': leaspy,
        'val_loglik': val_loglik,
        'bic': bic,
        'aic': aic,
        'val_scores': val_scores
    }
```

#### Validazione dei parametri individuali
```python
def validate_individual_parameters(leaspy_model, real_data, synthetic_data):
    """
    Confronta distribuzioni dei parametri individuali (τ, ξ, w)
    """
    from scipy import stats
    
    # Estrai parametri da dati reali
    real_leaspy = Data.from_dataframe(real_data)
    real_ip = leaspy_model.personalize(real_leaspy)
    
    # Estrai parametri da dati sintetici
    synth_leaspy = Data.from_dataframe(synthetic_data)
    synth_ip = leaspy_model.personalize(synth_leaspy)
    
    results = {}
    
    # τ (time-shift)
    tau_real = [ip['tau'] for ip in real_ip.values()]
    tau_synth = [ip['tau'] for ip in synth_ip.values()]
    ks_tau = stats.ks_2samp(tau_real, tau_synth)
    
    results['tau'] = {
        'ks_statistic': ks_tau.statistic,
        'p_value': ks_tau.pvalue,
        'mean_real': np.mean(tau_real),
        'mean_synth': np.mean(tau_synth),
        'std_real': np.std(tau_real),
        'std_synth': np.std(tau_synth),
        'passed': ks_tau.pvalue > 0.05
    }
    
    # ξ (log-acceleration)
    xi_real = [ip['xi'] for ip in real_ip.values()]
    xi_synth = [ip['xi'] for ip in synth_ip.values()]
    ks_xi = stats.ks_2samp(xi_real, xi_synth)
    
    results['xi'] = {
        'ks_statistic': ks_xi.statistic,
        'p_value': ks_xi.pvalue,
        'mean_real': np.mean(xi_real),
        'mean_synth': np.mean(xi_synth),
        'std_real': np.std(xi_real),
        'std_synth': np.std(xi_synth),
        'passed': ks_xi.pvalue > 0.05
    }
    
    # w (sources/spatial variability)
    if 'sources' in list(real_ip.values())[0]:
        sources_real = np.array([ip['sources'] for ip in real_ip.values()])
        sources_synth = np.array([ip['sources'] for ip in synth_ip.values()])
        
        for i in range(sources_real.shape[1]):
            ks_source = stats.ks_2samp(sources_real[:, i], sources_synth[:, i])
            results[f'source_{i}'] = {
                'ks_statistic': ks_source.statistic,
                'p_value': ks_source.pvalue,
                'passed': ks_source.pvalue > 0.05
            }
    
    # Overall
    all_passed = all(v.get('passed', False) for v in results.values())
    results['overall_passed'] = all_passed
    
    return results
```

### 6.2 Validazione delle traiettorie longitudinali

```python
def validate_longitudinal_trajectories(real_data, synthetic_data, 
                                        biomarkers, time_var='TIME',
                                        groupby='DIAGNOSIS'):
    """
    Confronta traiettorie medie per gruppo diagnostico
    """
    results = {}
    
    for biomarker in biomarkers:
        # Traiettorie medie reali
        real_traj = (real_data
                     .groupby([groupby, time_var])[biomarker]
                     .agg(['mean', 'std', 'count'])
                     .reset_index())
        
        # Traiettorie medie sintetiche
        synth_traj = (synthetic_data
                      .groupby([groupby, time_var])[biomarker]
                      .agg(['mean', 'std', 'count'])
                      .reset_index())
        
        # Merge per confronto
        merged = real_traj.merge(synth_traj, 
                                  on=[groupby, time_var],
                                  suffixes=('_real', '_synth'))
        
        # Calcola differenze
        merged['mean_diff'] = np.abs(merged['mean_real'] - merged['mean_synth'])
        merged['std_diff'] = np.abs(merged['std_real'] - merged['std_synth'])
        
        # Metriche aggregate
        mean_mae = merged['mean_diff'].mean()
        std_mae = merged['std_diff'].mean()
        
        # Criterio: MAE < 20% dello std reale
        threshold = 0.2 * real_data[biomarker].std()
        
        results[biomarker] = {
            'mean_trajectory_mae': mean_mae,
            'std_trajectory_mae': std_mae,
            'threshold': threshold,
            'passed': mean_mae < threshold,
            'trajectories': merged
        }
    
    # Visualizzazione
    import matplotlib.pyplot as plt
    
    n_biomarkers = len(biomarkers)
    fig, axes = plt.subplots(n_biomarkers, 1, figsize=(12, 4*n_biomarkers))
    if n_biomarkers == 1:
        axes = [axes]
    
    for idx, biomarker in enumerate(biomarkers):
        ax = axes[idx]
        traj_data = results[biomarker]['trajectories']
        
        # Plot per gruppo
        for group in traj_data[groupby].unique():
            group_data = traj_data[traj_data[groupby] == group]
            
            # Real
            ax.plot(group_data[time_var], group_data['mean_real'],
                    label=f'{group} - Real', marker='o', linewidth=2)
            ax.fill_between(group_data[time_var],
                             group_data['mean_real'] - group_data['std_real'],
                             group_data['mean_real'] + group_data['std_real'],
                             alpha=0.2)
            
            # Synthetic
            ax.plot(group_data[time_var], group_data['mean_synth'],
                    label=f'{group} - Synthetic', marker='s', 
                    linestyle='--', linewidth=2)
            ax.fill_between(group_data[time_var],
                             group_data['mean_synth'] - group_data['std_synth'],
                             group_data['mean_synth'] + group_data['std_synth'],
                             alpha=0.2)
        
        ax.set_xlabel('Time')
        ax.set_ylabel(biomarker)
        ax.set_title(f'{biomarker} - MAE: {results[biomarker]["mean_trajectory_mae"]:.4f} '
                     f'(Threshold: {results[biomarker]["threshold"]:.4f}) '
                     f'{"✓" if results[biomarker]["passed"] else "✗"}')
        ax.legend()
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    return results, fig
```

### 6.3 Validazione della sequenza di eventi

```python
def validate_biomarker_cascade(real_data, synthetic_data, 
                                 biomarkers, abnormality_thresholds):
    """
    Valida che l'ordinamento della cascata di biomarcatori sia preservato
    
    Parameters:
    -----------
    abnormality_thresholds : dict
        Soglie per considerare un biomarcatore anomalo
        es. {'MMSE': 24, 'hippocampal_volume': 3000, ...}
    """
    from collections import Counter
    
    def extract_event_sequences(data, biomarkers, thresholds):
        """Estrae sequenze di eventi per ogni soggetto"""
        sequences = []
        
        for subject_id in data['ID'].unique():
            subject_data = data[data['ID'] == subject_id].sort_values('TIME')
            events = []
            
            for bio in biomarkers:
                # Trova primo timepoint dove diventa anomalo
                if bio in thresholds:
                    threshold = thresholds[bio]
                    # Assume che valori bassi = peggio (MMSE, volumes)
                    # o valori alti = peggio (ADAS, tau)
                    anomaly_mask = subject_data[bio] < threshold
                    if anomaly_mask.any():
                        anomaly_time = subject_data[anomaly_mask]['TIME'].min()
                        events.append((bio, anomaly_time))
            
            # Ordina eventi per tempo
            events_sorted = sorted(events, key=lambda x: x[1])
            # Estrai solo nomi biomarcatori
            sequence = tuple([e[0] for e in events_sorted])
            sequences.append(sequence)
        
        return sequences
    
    # Estrai sequenze
    real_sequences = extract_event_sequences(real_data, biomarkers, 
                                              abnormality_thresholds)
    synth_sequences = extract_event_sequences(synthetic_data, biomarkers,
                                               abnormality_thresholds)
    
    # Conta frequenze
    real_counter = Counter(real_sequences)
    synth_counter = Counter(synth_sequences)
    
    # Top 10 sequenze più comuni
    top_real = real_counter.most_common(10)
    top_synth = synth_counter.most_common(10)
    
    # Calcola overlap nelle top 5
    real_top5 = set([seq for seq, _ in real_counter.most_common(5)])
    synth_top5 = set([seq for seq, _ in synth_counter.most_common(5)])
    overlap = len(real_top5 & synth_top5) / 5
    
    # Kendall's tau per ranking
    from scipy.stats import kendalltau
    common_seqs = list(set([s for s, _ in top_real]) & 
                       set([s for s, _ in top_synth]))
    if len(common_seqs) > 1:
        real_ranks = [real_counter[seq] for seq in common_seqs]
        synth_ranks = [synth_counter[seq] for seq in common_seqs]
        tau, p_value = kendalltau(real_ranks, synth_ranks)
    else:
        tau, p_value = 0, 1
    
    return {
        'top_sequences_real': top_real,
        'top_sequences_synth': top_synth,
        'overlap_top5': overlap,
        'kendall_tau': tau,
        'kendall_pvalue': p_value,
        'passed': overlap >= 0.6 and tau > 0.5
    }
```

### 6.4 Validazione predittiva

```python
def validate_predictive_accuracy(leaspy_model, test_data, 
                                  baseline_time=0, 
                                  prediction_horizons=[12, 24, 36]):
    """
    Testa accuratezza predittiva del modello su dati reali di test
    
    Parameters:
    -----------
    prediction_horizons : list
        Mesi nel futuro per cui predire (es. [12, 24, 36])
    """
    results = {}
    
    # Soggetti con baseline e follow-up
    baseline_ids = test_data[test_data['TIME'] == baseline_time]['ID'].unique()
    
    predictions_all = []
    actuals_all = []
    
    for subject_id in baseline_ids:
        subject_data = test_data[test_data['ID'] == subject_id].sort_values('TIME')
        
        # Baseline
        baseline_data = subject_data[subject_data['TIME'] == baseline_time]
        
        # Follow-up disponibili
        followup_data = subject_data[subject_data['TIME'] > baseline_time]
        
        if len(followup_data) > 0:
            # Personalizza modello su baseline
            baseline_leaspy = Data.from_dataframe(baseline_data)
            individual_params = leaspy_model.personalize(baseline_leaspy)
            
            # Predici ai timepoint disponibili
            for horizon in prediction_horizons:
                # Dati reali a questo horizon (se disponibili)
                actual_at_horizon = followup_data[
                    followup_data['TIME'] == baseline_time + horizon
                ]
                
                if len(actual_at_horizon) > 0:
                    # Predizione
                    prediction = leaspy_model.estimate(
                        individual_params[subject_id],
                        [baseline_time + horizon]
                    )
                    
                    predictions_all.append(prediction)
                    actuals_all.append(actual_at_horizon.iloc[0])
    
    # Converti in arrays
    predictions_array = np.array(predictions_all)
    actuals_array = np.array(actuals_all)
    
    # Calcola MAE per biomarker
    biomarkers = test_data.columns.drop(['ID', 'TIME'])
    for bio in biomarkers:
        mae = np.abs(predictions_array[:, bio] - actuals_array[:, bio]).mean()
        rmse = np.sqrt(((predictions_array[:, bio] - actuals_array[:, bio])**2).mean())
        
        results[bio] = {
            'mae': mae,
            'rmse': rmse
        }
    
    # Criterio di successo (specifico per Alzheimer)
    mmse_threshold = 2.0  # < 2 punti MMSE errore
    passed = results.get('MMSE', {}).get('mae', float('inf')) < mmse_threshold
    
    results['overall_passed'] = passed
    
    return results
```

### 6.5 Validazione cross-cohort

```python
def cross_cohort_validation(model, cohorts_dict, train_cohort='ADNI'):
    """
    Valida transferibilità del modello su coorti esterne
    
    Parameters:
    -----------
    cohorts_dict : dict
        {'ADNI': adni_data, 'AIBL': aibl_data, 'NACC': nacc_data, ...}
    """
    results = {}
    
    # Train su cohort principale
    train_data = cohorts_dict[train_cohort]
    train_leaspy = Data.from_dataframe(train_data)
    model.fit(train_leaspy)
    
    # Testa su altre coorti
    for cohort_name, cohort_data in cohorts_dict.items():
        if cohort_name == train_cohort:
            continue
        
        # Split test cohort
        train_ext, test_ext = train_test_split(cohort_data, test_size=0.3)
        
        # Genera synthetic da modello
        synthetic_ext = generate_synthetic_from_leaspy(model, n_samples=len(train_ext))
        
        # TSTR test
        tstr_result = tstr_validation(test_ext, synthetic_ext, target='DIAGNOSIS')
        
        results[cohort_name] = {
            'n_samples': len(cohort_data),
            'tstr_accuracy': tstr_result['tstr_metrics']['accuracy'],
            'trtr_accuracy': tstr_result['trtr_metrics']['accuracy'],
            'degradation': tstr_result['degradation']['accuracy_diff'],
            'passed': tstr_result['passed']
        }
    
    # Overall cross-cohort performance
    avg_degradation = np.mean([r['degradation'] for r in results.values()])
    all_passed = all(r['passed'] for r in results.values())
    
    results['overall'] = {
        'avg_degradation': avg_degradation,
        'all_cohorts_passed': all_passed
    }
    
    return results
```

---

## 7. Implementazione pratica completa

### 7.1 Pipeline di validazione completa

```python
class ComprehensiveSyntheticDataValidator:
    """
    Classe per validazione completa di dati sintetici
    Combina Maximum Similarity Test + metriche tradizionali + validazione Leaspy
    """
    
    def __init__(self, real_data, synthetic_data, 
                 leaspy_model=None,
                 categorical_features=None,
                 time_var='TIME',
                 id_var='ID'):
        self.real_data = real_data
        self.synthetic_data = synthetic_data
        self.leaspy_model = leaspy_model
        self.categorical_features = categorical_features or []
        self.time_var = time_var
        self.id_var = id_var
        self.results = {}
    
    def run_full_validation(self):
        """
        Esegue batteria completa di test
        """
        print("="*80)
        print("COMPREHENSIVE SYNTHETIC DATA VALIDATION")
        print("="*80)
        
        # ============= DIMENSIONE 1: FIDELITY =============
        print("\n" + "="*80)
        print("1. FIDELITY VALIDATION")
        print("="*80)
        
        self._validate_fidelity()
        
        # ============= DIMENSIONE 2: UTILITY =============
        print("\n" + "="*80)
        print("2. UTILITY VALIDATION")
        print("="*80)
        
        self._validate_utility()
        
        # ============= DIMENSIONE 3: PRIVACY =============
        print("\n" + "="*80)
        print("3. PRIVACY VALIDATION")
        print("="*80)
        
        self._validate_privacy()
        
        # ============= MAXIMUM SIMILARITY TEST =============
        print("\n" + "="*80)
        print("4. MAXIMUM SIMILARITY TEST")
        print("="*80)
        
        self._run_maximum_similarity_test()
        
        # ============= LEASPY-SPECIFIC (se applicabile) =============
        if self.leaspy_model is not None:
            print("\n" + "="*80)
            print("5. LEASPY/DCM-SPECIFIC VALIDATION")
            print("="*80)
            
            self._validate_leaspy_specific()
        
        # ============= SUMMARY =============
        self._print_summary()
        
        return self.results
    
    def _validate_fidelity(self):
        """Validazione fedeltà statistica"""
        print("\n--- 1.1 Univariate Distributions ---")
        
        numeric_cols = self.real_data.select_dtypes(include=[np.number]).columns
        numeric_cols = [c for c in numeric_cols if c not in [self.time_var, self.id_var]]
        
        univariate_results = {}
        for col in numeric_cols:
            # KS test
            ks_stat, p_value = stats.ks_2samp(
                self.real_data[col].dropna(),
                self.synthetic_data[col].dropna()
            )
            passed = p_value > 0.05
            univariate_results[col] = {
                'ks_statistic': ks_stat,
                'p_value': p_value,
                'passed': passed
            }
            
            status = "✓" if passed else "✗"
            print(f"  {col:30s}: KS={ks_stat:.4f}, p={p_value:.4f} {status}")
        
        self.results['univariate'] = univariate_results
        
        # Correlations
        print("\n--- 1.2 Correlation Structure ---")
        
        real_corr = self.real_data[numeric_cols].corr()
        synth_corr = self.synthetic_data[numeric_cols].corr()
        
        # MAD
        corr_diff = np.abs(real_corr - synth_corr)
        mad = corr_diff.mean().mean()
        
        # Meta-correlation
        real_flat = real_corr.values[np.triu_indices_from(real_corr.values, k=1)]
        synth_flat = synth_corr.values[np.triu_indices_from(synth_corr.values, k=1)]
        meta_corr = np.corrcoef(real_flat, synth_flat)[0, 1]
        
        passed = (mad < 0.15) and (meta_corr > 0.8)
        
        self.results['correlations'] = {
            'mean_abs_diff': mad,
            'meta_correlation': meta_corr,
            'passed': passed
        }
        
        status = "✓" if passed else "✗"
        print(f"  Mean Absolute Difference: {mad:.4f} (threshold: 0.15)")
        print(f"  Meta-correlation: {meta_corr:.4f} (threshold: 0.80) {status}")
    
    def _validate_utility(self):
        """Validazione utilità pratica"""
        print("\n--- 2.1 TSTR (Train on Synthetic, Test on Real) ---")
        
        # Prepara dati per classificazione (assume colonna 'target')
        if 'DIAGNOSIS' in self.real_data.columns:
            target_col = 'DIAGNOSIS'
        elif 'target' in self.real_data.columns:
            target_col = 'target'
        else:
            print("  ⚠ No target column found, skipping TSTR")
            self.results['tstr'] = {'skipped': True}
            return
        
        # Feature columns
        feature_cols = [c for c in self.real_data.columns 
                        if c not in [target_col, self.time_var, self.id_var]]
        
        # Prepara X, y
        X_real = self.real_data[feature_cols].fillna(0)
        y_real = self.real_data[target_col]
        X_synth = self.synthetic_data[feature_cols].fillna(0)
        y_synth = self.synthetic_data[target_col]
        
        # Split real
        from sklearn.model_selection import train_test_split
        X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(
            X_real, y_real, test_size=0.3, random_state=42, stratify=y_real
        )
        
        # TSTR: Train on Synthetic, Test on Real
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score, f1_score
        
        model_tstr = RandomForestClassifier(n_estimators=100, random_state=42)
        model_tstr.fit(X_synth, y_synth)
        y_pred_tstr = model_tstr.predict(X_test_real)
        
        acc_tstr = accuracy_score(y_test_real, y_pred_tstr)
        f1_tstr = f1_score(y_test_real, y_pred_tstr, average='macro')
        
        # TRTR: Train on Real, Test on Real (baseline)
        model_trtr = RandomForestClassifier(n_estimators=100, random_state=42)
        model_trtr.fit(X_train_real, y_train_real)
        y_pred_trtr = model_trtr.predict(X_test_real)
        
        acc_trtr = accuracy_score(y_test_real, y_pred_trtr)
        f1_trtr = f1_score(y_test_real, y_pred_trtr, average='macro')
        
        # Degradation
        acc_deg = acc_trtr - acc_tstr
        f1_deg = f1_trtr - f1_tstr
        
        passed = (acc_deg < 0.05) and (f1_deg < 0.05)
        
        self.results['tstr'] = {
            'tstr_accuracy': acc_tstr,
            'tstr_f1': f1_tstr,
            'trtr_accuracy': acc_trtr,
            'trtr_f1': f1_trtr,
            'accuracy_degradation': acc_deg,
            'f1_degradation': f1_deg,
            'passed': passed
        }
        
        status = "✓" if passed else "✗"
        print(f"  TSTR Accuracy: {acc_tstr:.4f} | F1: {f1_tstr:.4f}")
        print(f"  TRTR Accuracy: {acc_trtr:.4f} | F1: {f1_trtr:.4f}")
        print(f"  Degradation: Acc={acc_deg:.4f}, F1={f1_deg:.4f} {status}")
    
    def _validate_privacy(self):
        """Validazione privacy"""
        # IMS
        print("\n--- 3.1 Identical Match Share (IMS) ---")
        
        real_tuples = set(map(tuple, self.real_data.drop(columns=[self.id_var]).values))
        synth_tuples = set(map(tuple, self.synthetic_data.drop(columns=[self.id_var]).values))
        
        exact_matches = real_tuples & synth_tuples
        ims = len(exact_matches) / len(synth_tuples) if len(synth_tuples) > 0 else 0
        
        passed_ims = (ims == 0)
        
        self.results['ims'] = {
            'identical_match_share': ims,
            'num_exact_matches': len(exact_matches),
            'passed': passed_ims
        }
        
        status = "✓" if passed_ims else "✗"
        print(f"  Exact matches: {len(exact_matches)}")
        print(f"  IMS: {ims:.6f} {status}")
        
        # DCR
        print("\n--- 3.2 Distance to Closest Record (DCR) ---")
        
        # Split real into train/holdout
        n_real = len(self.real_data)
        train_size = int(0.8 * n_real)
        
        real_train = self.real_data.iloc[:train_size]
        real_holdout = self.real_data.iloc[train_size:]
        
        # Prepara per distance computation
        feature_cols = [c for c in self.real_data.columns 
                        if c not in [self.id_var, self.time_var]]
        
        from sklearn.preprocessing import StandardScaler
        from sklearn.neighbors import NearestNeighbors
        
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(real_train[feature_cols].fillna(0))
        holdout_scaled = scaler.transform(real_holdout[feature_cols].fillna(0))
        synth_scaled = scaler.transform(self.synthetic_data[feature_cols].fillna(0))
        
        # NN to train
        nn_train = NearestNeighbors(n_neighbors=1)
        nn_train.fit(train_scaled)
        dist_to_train, _ = nn_train.kneighbors(synth_scaled)
        
        # NN to holdout
        nn_holdout = NearestNeighbors(n_neighbors=1)
        nn_holdout.fit(holdout_scaled)
        dist_to_holdout, _ = nn_holdout.kneighbors(synth_scaled)
        
        # Ratio closer to train
        closer_to_train_ratio = (dist_to_train.flatten() < dist_to_holdout.flatten()).mean()
        
        passed_dcr = (closer_to_train_ratio <= 0.5)
        
        self.results['dcr'] = {
            'closer_to_train_ratio': closer_to_train_ratio,
            'mean_dist_to_train': dist_to_train.mean(),
            'mean_dist_to_holdout': dist_to_holdout.mean(),
            'passed': passed_dcr
        }
        
        status = "✓" if passed_dcr else "✗"
        print(f"  Closer to train: {closer_to_train_ratio:.4f} (target: ≤0.50) {status}")
    
    def _run_maximum_similarity_test(self):
        """Maximum Similarity Test"""
        print("\n--- 4.1 Computing Similarities ---")
        
        # Prepara feature matrix
        feature_cols = [c for c in self.real_data.columns 
                        if c not in [self.id_var, self.time_var]]
        
        # Esegui MST
        mst_results = maximum_similarity_test(
            self.real_data[feature_cols],
            self.synthetic_data[feature_cols],
            categorical_features=self.categorical_features
        )
        
        self.results['maximum_similarity_test'] = mst_results
        
        print(f"  Mean MISS (Real): {mst_results['mean_miss_real']:.4f}")
        print(f"  Mean MISS (Synth): {mst_results['mean_miss_synth']:.4f}")
        print(f"  Mean MCSS: {mst_results['mean_mcss']:.4f}")
        print(f"  Quality Score: {mst_results['quality_score']:.4f}")
        
        status_privacy = "✓" if mst_results['passed_privacy'] else "✗"
        status_fidelity = "✓" if mst_results['passed_fidelity'] else "✗"
        
        print(f"  Privacy (Q ≤ 1.0): {status_privacy}")
        print(f"  Fidelity (KS p > 0.05): {status_fidelity}")
    
    def _validate_leaspy_specific(self):
        """Validazione specifica per Leaspy"""
        print("\n--- 5.1 Individual Parameters ---")
        
        ip_results = validate_individual_parameters(
            self.leaspy_model,
            self.real_data,
            self.synthetic_data
        )
        
        self.results['individual_parameters'] = ip_results
        
        for param, values in ip_results.items():
            if param == 'overall_passed':
                continue
            status = "✓" if values['passed'] else "✗"
            print(f"  {param}: p={values['p_value']:.4f} {status}")
        
        print("\n--- 5.2 Longitudinal Trajectories ---")
        
        biomarkers = [c for c in self.real_data.columns 
                      if c not in [self.id_var, self.time_var, 'DIAGNOSIS']]
        
        traj_results, _ = validate_longitudinal_trajectories(
            self.real_data,
            self.synthetic_data,
            biomarkers[:3]  # Top 3 per brevità
        )
        
        self.results['trajectories'] = traj_results
        
        for bio, res in list(traj_results.items())[:3]:
            status = "✓" if res['passed'] else "✗"
            print(f"  {bio}: MAE={res['mean_trajectory_mae']:.4f} {status}")
    
    def _print_summary(self):
        """Stampa summary"""
        print("\n" + "="*80)
        print("VALIDATION SUMMARY")
        print("="*80)
        
        # Conta test passati
        passed_counts = {
            'Fidelity': 0,
            'Utility': 0,
            'Privacy': 0,
            'MST': 0
        }
        total_counts = {
            'Fidelity': 0,
            'Utility': 0,
            'Privacy': 0,
            'MST': 0
        }
        
        # Fidelity
        if 'univariate' in self.results:
            for res in self.results['univariate'].values():
                total_counts['Fidelity'] += 1
                if res['passed']:
                    passed_counts['Fidelity'] += 1
        if 'correlations' in self.results:
            total_counts['Fidelity'] += 1
            if self.results['correlations']['passed']:
                passed_counts['Fidelity'] += 1
        
        # Utility
        if 'tstr' in self.results and not self.results['tstr'].get('skipped'):
            total_counts['Utility'] += 1
            if self.results['tstr']['passed']:
                passed_counts['Utility'] += 1
        
        # Privacy
        if 'ims' in self.results:
            total_counts['Privacy'] += 1
            if self.results['ims']['passed']:
                passed_counts['Privacy'] += 1
        if 'dcr' in self.results:
            total_counts['Privacy'] += 1
            if self.results['dcr']['passed']:
                passed_counts['Privacy'] += 1
        
        # MST
        if 'maximum_similarity_test' in self.results:
            total_counts['MST'] += 2  # privacy + fidelity
            if self.results['maximum_similarity_test']['passed_privacy']:
                passed_counts['MST'] += 1
            if self.results['maximum_similarity_test']['passed_fidelity']:
                passed_counts['MST'] += 1
        
        # Print
        print("\nDimension-wise Results:")
        for dim in ['Fidelity', 'Utility', 'Privacy', 'MST']:
            if total_counts[dim] > 0:
                rate = passed_counts[dim] / total_counts[dim]
                status = "✓" if rate >= 0.8 else ("⚠" if rate >= 0.6 else "✗")
                print(f"  {dim:15s}: {passed_counts[dim]}/{total_counts[dim]} "
                      f"({rate:.1%}) {status}")
        
        # Overall
        total_all = sum(total_counts.values())
        passed_all = sum(passed_counts.values())
        overall_rate = passed_all / total_all if total_all > 0 else 0
        
        print(f"\nOverall: {passed_all}/{total_all} ({overall_rate:.1%})")
        
        if overall_rate >= 0.8:
            print("\n✓ SYNTHETIC DATA VALIDATED")
            print("  Quality: HIGH - Recommended for use")
        elif overall_rate >= 0.6:
            print("\n⚠ SYNTHETIC DATA PARTIALLY VALIDATED")
            print("  Quality: MEDIUM - Use with caution")
        else:
            print("\n✗ SYNTHETIC DATA FAILED VALIDATION")
            print("  Quality: LOW - Not recommended for use")
        
        # Critical check: Privacy
        privacy_passed = (passed_counts['Privacy'] == total_counts['Privacy'] and
                          self.results.get('maximum_similarity_test', {}).get('passed_privacy', False))
        
        if not privacy_passed:
            print("\n⚠️  CRITICAL: Privacy tests failed!")
            print("   Model shows signs of overfitting.")
            print("   Fidelity and Utility metrics may be unreliable.")
    
    def save_report(self, filename='validation_report.json'):
        """Salva report completo"""
        import json
        
        # Converti numpy types per JSON serialization
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2, default=convert_types)
        
        print(f"\n✓ Report saved to: {filename}")
```

### 7.2 Esempio d'uso completo

```python
# ============= ESEMPIO COMPLETO =============

import pandas as pd
import numpy as np
from leaspy import Leaspy, Data, AlgorithmSettings

# 1. Carica dati
real_data = pd.read_csv('alzheimer_longitudinal_real.csv')
synthetic_data = pd.read_csv('alzheimer_longitudinal_synthetic.csv')

# 2. Setup Leaspy (se necessario)
leaspy_model = Leaspy("logistic")
train_data = Data.from_dataframe(real_data)
settings = AlgorithmSettings('mcmc_saem', n_iter=5000)
leaspy_model.fit(train_data, settings)

# 3. Definisci feature categoriche
categorical_features = [
    real_data.columns.get_loc('APOE4'),
    real_data.columns.get_loc('SEX'),
    real_data.columns.get_loc('DIAGNOSIS')
]

# 4. Crea validator
validator = ComprehensiveSyntheticDataValidator(
    real_data=real_data,
    synthetic_data=synthetic_data,
    leaspy_model=leaspy_model,
    categorical_features=categorical_features,
    time_var='TIME',
    id_var='PTID'
)

# 5. Esegui validazione completa
results = validator.run_full_validation()

# 6. Salva report
validator.save_report('validation_results.json')

# 7. Plot Maximum Similarity Test
mst_results = results['maximum_similarity_test']
fig = plot_similarity_distributions(mst_results)
fig.savefig('maximum_similarity_test.png', dpi=300, bbox_inches='tight')

# 8. Plot traiettorie
if 'trajectories' in results:
    _, traj_fig = validate_longitudinal_trajectories(
        real_data, synthetic_data,
        biomarkers=['MMSE', 'ADAS', 'hippocampal_volume']
    )
    traj_fig.savefig('trajectories_comparison.png', dpi=300, bbox_inches='tight')
```

---

## 8. Report di validazione

### 8.1 Template di report scientifico

```markdown
# Validation Report: Synthetic Alzheimer's Disease Progression Data

## Executive Summary

- **Dataset**: ADNI Alzheimer's Disease Longitudinal Data
- **Generator**: Leaspy (Logistic Disease Course Mapping)
- **Real samples**: 866 subjects, 4,238 observations
- **Synthetic samples**: 866 subjects, 4,238 observations
- **Overall Quality Score**: [0.95] - HIGH QUALITY
- **Recommendation**: ✓ APPROVED for research use

---

## 1. Fidelity Assessment

### 1.1 Univariate Distributions (KS Test)

| Biomarker | KS Statistic | p-value | Status |
|-----------|--------------|---------|--------|
| Age | 0.0234 | 0.876 | ✓ PASS |
| MMSE | 0.0189 | 0.923 | ✓ PASS |
| ADAS-Cog | 0.0267 | 0.745 | ✓ PASS |
| Hippocampal Volume | 0.0312 | 0.634 | ✓ PASS |
| CSF Aβ42 | 0.0401 | 0.421 | ✓ PASS |
| CSF tau | 0.0356 | 0.512 | ✓ PASS |

**Result**: 6/6 biomarkers passed (100%)

### 1.2 Correlation Structure

- **Mean Absolute Difference**: 0.087 (threshold: 0.15) ✓
- **Meta-correlation**: 0.912 (threshold: 0.80) ✓

**Result**: PASSED

### 1.3 Maximum Similarity Test

- **Mean MISS (Real)**: 0.4523
- **Mean MISS (Synthetic)**: 0.4487
- **Mean MCSS**: 0.4419
- **Quality Score**: 0.977

**Interpretation**: Synthetic points are slightly LESS similar to real points than real points are to each other, indicating good privacy and no overfitting.

**Result**: ✓ PASSED

---

## 2. Utility Assessment

### 2.1 TSTR (Diagnostic Classification: CN/MCI/AD)

| Metric | TSTR | TRTR | Degradation | Status |
|--------|------|------|-------------|--------|
| Accuracy | 0.847 | 0.863 | -0.016 | ✓ |
| F1 (Macro) | 0.829 | 0.841 | -0.012 | ✓ |
| AUC-ROC | 0.912 | 0.925 | -0.013 | ✓ |

**Result**: All degradations < 5% → ✓ PASSED

### 2.2 Cognitive Decline Prediction (12-month MMSE)

| Model | MAE | RMSE | R² |
|-------|-----|------|----|
| TSTR | 1.87 | 2.43 | 0.68 |
| TRTR | 1.73 | 2.29 | 0.72 |

**Degradation**: 0.14 MMSE points (< 1.5 threshold) → ✓ PASSED

---

## 3. Privacy Assessment

### 3.1 Identical Match Share (IMS)

- **Exact matches**: 0
- **IMS**: 0.000

**Result**: ✓ PASSED (no exact replication)

### 3.2 Distance to Closest Record (DCR)

- **Closer to train**: 47.3% (target: ≤50%)
- **Mean distance to train**: 2.34
- **Mean distance to holdout**: 2.41

**Result**: ✓ PASSED (no evidence of memorization)

---

## 4. Leaspy-Specific Validation

### 4.1 Individual Parameters

| Parameter | KS Statistic | p-value | Status |
|-----------|--------------|---------|--------|
| τ (time-shift) | 0.0423 | 0.389 | ✓ |
| ξ (acceleration) | 0.0512 | 0.267 | ✓ |
| w₁ (source 1) | 0.0389 | 0.501 | ✓ |
| w₂ (source 2) | 0.0467 | 0.334 | ✓ |

**Result**: 4/4 parameters passed → ✓ PASSED

### 4.2 Biomarker Cascade

**Top 5 Event Sequences (Real)**:
1. Aβ42 → tau → hippocampus → MMSE (23.4%)
2. Aβ42 → tau → MMSE → hippocampus (18.7%)
3. tau → Aβ42 → hippocampus → MMSE (12.3%)
...

**Top 5 Event Sequences (Synthetic)**:
1. Aβ42 → tau → hippocampus → MMSE (21.8%)
2. Aβ42 → tau → MMSE → hippocampus (19.2%)
3. tau → Aβ42 → hippocampus → MMSE (11.7%)
...

- **Overlap (Top 5)**: 80%
- **Kendall's τ**: 0.87 (p < 0.001)

**Result**: ✓ PASSED (sequence preservation)

---

## 5. Overall Conclusion

| Dimension | Passed/Total | Rate | Status |
|-----------|--------------|------|--------|
| Fidelity | 8/8 | 100% | ✓ |
| Utility | 2/2 | 100% | ✓ |
| Privacy | 3/3 | 100% | ✓ |
| Leaspy | 6/6 | 100% | ✓ |
| **TOTAL** | **19/19** | **100%** | **✓** |

### Quality Rating: ★★★★★ (5/5)

**RECOMMENDATION**: This synthetic dataset is **APPROVED** for research use. It demonstrates:
- Excellent statistical fidelity to real data
- Preserved utility for predictive modeling
- Strong privacy guarantees
- Valid disease progression patterns

**Suggested Use Cases**:
- Algorithm development and benchmarking
- Exploratory data analysis
- Educational purposes
- Data sharing (privacy-preserving)

**Limitations**:
- Cannot replace real data for clinical validation
- Should be validated again if used for novel tasks
- Rare event patterns may be underrepresented
```

---

## 9. Conclusioni e raccomandazioni

### 9.1 Principi chiave

**1. La privacy è un prerequisito**
- Se privacy fallisce → modello in overfitting → tutte le altre metriche inaffidabili
- Non esiste "2 su 3": privacy deve sempre passare

**2. Quality Score come metrica unificata**
```
Q = mean(MCSS) / mean(MISS_real)

Ideale: Q ≈ 1.0 (ma ≤ 1.0)
Accettabile: 0.9 < Q < 1.0
Problematico: Q > 1.0
```

**3. Validazione multi-dimensionale**
- Fidelity: proprietà statistiche
- Utility: utilità pratica
- Privacy: protezione dati
- (Se longitudinale) Dinamiche temporali

### 9.2 Checklist finale

**Prima di usare dati sintetici:**

#### ✓ Fidelity
- [ ] Distribuzioni univariate simili (KS test p > 0.05)
- [ ] Correlazioni preservate (MAD < 0.15, meta-corr > 0.8)
- [ ] Maximum Similarity Test: MCSS ≈ MISS_real
- [ ] (Longitudinale) Traiettorie temporali simili

#### ✓ Utility
- [ ] TSTR degradation < 5%
- [ ] Feature importance coerente
- [ ] Performance su task rilevanti accettabile

#### ✓ Privacy
- [ ] Zero exact matches (IMS = 0)
- [ ] DCR ratio ≤ 0.50
- [ ] Quality Score ≤ 1.0
- [ ] Membership inference attack fallisce

#### ✓ Domain-specific (Alzheimer/Leaspy)
- [ ] Parametri individuali (τ, ξ, w) distribuiti correttamente
- [ ] Sequenza eventi biomarcatori clinicamente plausibile
- [ ] Predizioni di progressione accurate (MAE MMSE < 2.0)
- [ ] Validazione cross-cohort superata

### 9.3 Quando NON usare dati sintetici

**⚠️ I dati sintetici NON dovrebbero essere usati per:**

1. **Validazione clinica finale** di dispositivi medici o farmaci
2. **Decisioni individuali** su pazienti specifici
3. **Pubblicazione di risultati clinici** senza validazione su dati reali
4. **Sostituzione totale** dei dati reali in studi pivotali

**✓ I dati sintetici sono appropriati per:**

1. **Sviluppo di algoritmi** e proof-of-concept
2. **Benchmarking** di metodi ML
3. **Educazione e training** di personale
4. **Data sharing** quando i dati reali non possono essere condivisi
5. **Augmentation** di dataset piccoli (con cautela)

### 9.4 Raccomandazioni per generatori

**Migliori approcci** (basati su evidenza empirica):

1. **Sequential Imputation** (synthpop, UNCRi)
   - ✓ Performance consistente
   - ✓ Previene overfitting
   - ⚠ Sensibile a ordine variabili

2. **Bayesian Mixed-Effects** (Leaspy per longitudinale)
   - ✓ Modella variabilità individuale
   - ✓ Cattura dinamiche temporali
   - ⚠ Richiede struttura longitudinale

3. **Copula-based** (GaussianCopula)
   - ✓ Teoricamente solido
   - ⚠ Performance mediocre su dati complessi

**Evitare o usare con cautela:**

- **GAN-based**: Performance inconsistente, problemi training
- **VAE-based**: Rischio privacy (es. TVAE), output sfocati

### 9.5 Risorse e riferimenti

**Software/Libraries:**

- Leaspy: https://gitlab.com/icm-institute/aramislab/leaspy
- SDV (Synthetic Data Vault): https://sdv.dev
- synthpop (R): https://cran.r-project.org/package=synthpop
- Microsoft Synthetic Data Showcase: https://github.com/microsoft/synthetic-data-showcase

**Articoli chiave:**

1. **Validazione generale:**
   - Chen et al. (2019) - "The validity of synthetic clinical data" (BMC Med Inform)
   - Hernandez et al. (2023) - "Fidelity vs privacy vs utility" (iScience)

2. **Maximum Similarity Test:**
   - Articolo Medium di riferimento

3. **Leaspy/Disease progression:**
   - Schiratti et al. (2017) - Framework matematico (JMLR)
   - Koval et al. (2021) - AD Course Map (Scientific Reports)
   - Koval et al. (2023) - Forecasting (Nature Comms)

4. **Privacy:**
   - Stadler et al. (2022) - "Synthetic data" (Nature Methods)
   - El Emam & Mosquera (2020) - "Practical Synthetic Data Generation"

---

## Appendice: Glossario

| Termine | Definizione |
|---------|------------|
| **Parent Distribution** | Vera distribuzione sottostante da cui i dati osservati sono campionati |
| **Fidelity** | Grado di similarità statistica tra dati reali e sintetici |
| **Utility** | Utilità pratica dei dati sintetici per task predittivi |
| **Privacy** | Assenza di information leakage dai dati reali ai sintetici |
| **MISS** | Maximum Intra-Set Similarity - similarità a nearest neighbor nello stesso dataset |
| **MCSS** | Maximum Cross-Set Similarity - similarità a nearest neighbor nell'altro dataset |
| **Quality Score** | Ratio MCSS/MISS_real; idealmente ≤ 1.0 |
| **TSTR** | Train on Synthetic, Test on Real - test di utilità |
| **TRTR** | Train on Real, Test on Real - baseline per TSTR |
| **IMS** | Identical Match Share - frazione di record identici |
| **DCR** | Distance to Closest Record - test di privacy basato su distanze |
| **Leaspy** | LEArning Spatiotemporal Patterns - framework Bayesiano per progressione malattie |
| **DCM** | Disease Course Mapping - approccio data-driven per modellare progressione |
| **τ (tau)** | Time-shift parameter in Leaspy - onset individuale della malattia |
| **ξ (xi)** | Log-acceleration parameter in Leaspy - velocità di progressione |
| **w (sources)** | Spatial variability in Leaspy - variazioni geometriche individuali |

---

**Versione**: 1.0  
**Data**: 2025  
**Autori**: Integrazione da articolo Medium + guida Leaspy  
**Licenza**: CC BY 4.0