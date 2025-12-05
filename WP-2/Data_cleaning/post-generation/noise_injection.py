"""
Post-Generation Noise Injection for Synthetic Data Privacy Enhancement
========================================================================

Based on Differential Privacy principles and supported by:
- Dwork et al. (2006): Laplace and Gaussian mechanisms for DP
- NIST (2021): Post-processing immunity in DP synthetic data
- Balle et al. (2018): Analytical improvements to Gaussian mechanism
- SafeSynthDP (2024): Noise injection for privacy-preserving synthetic data

Questo modulo implementa meccanismi di rumore post-generazione per migliorare
il DCR (Distance to Closest Record) mantenendo l'utilità dei dati.

References:
-----------
1. Dwork, C., McSherry, F., Nissim, K., & Smith, A. (2006).
   "Calibrating noise to sensitivity in private data analysis."
   Theory of Cryptography Conference (TCC).

2. Dwork, C., & Roth, A. (2014).
   "The algorithmic foundations of differential privacy."
   Foundations and Trends in Theoretical Computer Science, 9(3-4).

3. Balle, B., & Wang, Y. X. (2018).
   "Improving the Gaussian mechanism for differential privacy."
   International Conference on Machine Learning (ICML).
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional


class PostGenerationNoiseInjector:
    """
    Implementa meccanismi di differential privacy per aggiungere rumore
    ai dati sintetici già generati.

    References:
    -----------
    - Dwork, C., et al. (2006). "Calibrating noise to sensitivity in
      private data analysis." TCC.
    - Balle, B., et al. (2018). "Improving the Gaussian mechanism for
      differential privacy: Analytical calibration and optimal denoising." ICML.
    """

    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5):
        """
        Parameters:
        -----------
        epsilon : float
            Privacy budget (smaller = more privacy, more noise)
            Typical values: 0.1 (very private) to 10 (less private)
        delta : float
            Failure probability for (ε, δ)-DP
            Typical value: 1/n^2 where n is dataset size
        """
        self.epsilon = epsilon
        self.delta = delta

    def compute_sensitivity(self,
                          feature_values: np.ndarray,
                          feature_range: Optional[Tuple[float, float]] = None) -> float:
        """
        Calcola la sensibilità L1 o L2 di una feature.

        La sensibilità è la massima variazione che può avere una feature
        quando si aggiunge/rimuove un record dal dataset.
        """
        if feature_range is not None:
            # Sensibilità basata sul range noto
            return feature_range[1] - feature_range[0]
        else:
            # Stima conservativa dalla distribuzione empirica
            return np.ptp(feature_values)  # max - min

    def laplace_mechanism(self,
                         data: np.ndarray,
                         sensitivity: float) -> np.ndarray:
        """
        Laplace Mechanism per (ε, 0)-differential privacy.

        Aggiunge rumore dalla distribuzione Laplace con scala calibrata
        alla sensibilità e al privacy budget.

        Formula: noise ~ Laplace(0, Δf/ε)
        dove Δf è la sensibilità della query

        Reference: Dwork et al. (2006)
        """
        scale = sensitivity / self.epsilon
        noise = np.random.laplace(loc=0, scale=scale, size=data.shape)
        return data + noise

    def gaussian_mechanism(self,
                          data: np.ndarray,
                          sensitivity: float) -> np.ndarray:
        """
        Gaussian Mechanism per (ε, δ)-differential privacy.

        Aggiunge rumore gaussiano con varianza calibrata per garantire
        privacy differenziale approssimata.

        Formula: noise ~ N(0, σ²)
        dove σ = (Δf/ε) * sqrt(2 * ln(1.25/δ))

        Reference: Dwork & Roth (2014), Balle et al. (2018)
        """
        # Formula analitica per σ
        sigma = (sensitivity / self.epsilon) * np.sqrt(2 * np.log(1.25 / self.delta))
        noise = np.random.normal(loc=0, scale=sigma, size=data.shape)
        return data + noise

    def adaptive_noise(self,
                      data: np.ndarray,
                      sensitivity: float,
                      target_dcr: float = 0.50,
                      current_dcr: float = None) -> np.ndarray:
        """
        Adaptive noise injection che scala il rumore in base al DCR corrente.

        Se DCR > target_dcr, aumenta il rumore per allontanare i dati sintetici
        dai dati reali.

        Parameters:
        -----------
        target_dcr : float
            Target DCR value (typically 0.45-0.55)
        current_dcr : float
            Current DCR value from validation
        """
        if current_dcr is not None and current_dcr > target_dcr:
            # Aumenta epsilon inversamente al DCR
            # DCR più alto = più rumore necessario
            scale_factor = current_dcr / target_dcr
            adjusted_epsilon = self.epsilon / scale_factor
        else:
            adjusted_epsilon = self.epsilon

        # Usa Gaussian per default (più smooth)
        old_eps = self.epsilon
        self.epsilon = adjusted_epsilon
        result = self.gaussian_mechanism(data, sensitivity)
        self.epsilon = old_eps  # Restore

        return result


class LongitudinalNoiseInjector:
    """
    Specializzato per dati longitudinali - preserva correlazioni temporali
    mentre aggiunge rumore per privacy.
    """

    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5):
        self.base_injector = PostGenerationNoiseInjector(epsilon, delta)
        self.epsilon = epsilon
        self.delta = delta

    def inject_correlated_noise(self,
                                df: pd.DataFrame,
                                patient_col: str = 'ID',
                                time_col: str = 'TIME',
                                features: List[str] = None,
                                temporal_correlation: float = 0.7) -> pd.DataFrame:
        """
        Aggiunge rumore che preserva la correlazione temporale all'interno
        dei pazienti, ma de-correlato tra pazienti.

        Questo è importante per longitudinal data dove le misure successive
        di un paziente sono naturalmente correlate.

        Parameters:
        -----------
        temporal_correlation : float
            Correlazione desiderata tra visite successive (0-1)
            Più alto = preserva meglio le traiettorie individuali
        """
        df_noisy = df.copy()

        if features is None:
            # Auto-detect numeric features
            features = df.select_dtypes(include=[np.number]).columns.tolist()
            features = [f for f in features if f not in [patient_col, time_col]]

        for feature in features:
            # Skip if feature not in dataframe
            if feature not in df.columns:
                continue

            # Calcola sensibilità
            sensitivity = self.base_injector.compute_sensitivity(df[feature].values)

            # Per ogni paziente, genera rumore correlato temporalmente
            for patient_id in df[patient_col].unique():
                mask = df[patient_col] == patient_id
                patient_data = df.loc[mask, feature].values
                n_visits = len(patient_data)

                if n_visits == 0:
                    continue

                # Genera rumore con correlazione temporale usando AR(1)
                # noise[t] = ρ * noise[t-1] + sqrt(1-ρ²) * ε[t]
                # dove ε[t] ~ N(0, σ²)
                sigma = (sensitivity / self.base_injector.epsilon) * \
                        np.sqrt(2 * np.log(1.25 / self.base_injector.delta))

                noise = np.zeros(n_visits)
                noise[0] = np.random.normal(0, sigma)

                for t in range(1, n_visits):
                    innovation = np.random.normal(0, sigma * np.sqrt(1 - temporal_correlation**2))
                    noise[t] = temporal_correlation * noise[t-1] + innovation

                df_noisy.loc[mask, feature] = patient_data + noise

        return df_noisy


def add_post_generation_noise(
    synth_df: pd.DataFrame,
    real_df: pd.DataFrame,
    current_dcr: float,
    target_dcr: float = 0.50,
    epsilon: float = 1.0,
    method: str = 'gaussian',
    preserve_temporal: bool = True,
    patient_col: str = 'ID',
    time_col: str = 'TIME',
    ordinal_features: List[str] = None,
    continuous_features: List[str] = None,
    categorical_ordinal_features: List[str] = None
) -> pd.DataFrame:
    """
    Funzione principale per aggiungere rumore post-generazione.

    Parameters:
    -----------
    synth_df : pd.DataFrame
        Dati sintetici generati da Leaspy
    real_df : pd.DataFrame
        Dati reali per calcolare sensibilità
    current_dcr : float
        DCR corrente dai risultati di validazione
    target_dcr : float
        DCR target (tipicamente 0.45-0.55)
    epsilon : float
        Privacy budget. Valori raccomandati:
        - 0.5-1.0 per DCR molto alti (>0.80) - rumore aggressivo
        - 1.0-2.0 per DCR medi (0.60-0.80) - rumore moderato
        - 2.0-5.0 per DCR vicini al target - rumore leggero
    method : str
        'gaussian', 'laplace', o 'adaptive'
    preserve_temporal : bool
        Se True, usa rumore correlato temporalmente per longitudinal data
    ordinal_features : List[str]
        Scale cliniche ordinali (es: MMSE, CDRSB)
        Queste RICEVONO rumore (sono normalizzate [0,1]) ma NON vengono arrotondate qui.
        L'arrotondamento avviene DOPO la denormalizzazione in reverse_from_datalake.py
    continuous_features : List[str]
        Features continue (default: tutte le numeriche eccetto categoriche e dummy)
    categorical_ordinal_features : List[str]
        Categoriche ordinali che NON devono ricevere rumore (es: APOE, EDUCATION)
        Se None, usa auto-detection basata su valori unici

    Returns:
    --------
    pd.DataFrame
        Dati sintetici con rumore aggiunto

    Examples:
    ---------
    >>> # Caso 1: DCR molto alto (0.82) - rumore aggressivo
    >>> synth_noisy = add_post_generation_noise(
    ...     synth_df, real_df,
    ...     current_dcr=0.82,
    ...     epsilon=0.8,  # Più basso = più rumore
    ...     method='adaptive'
    ... )

    >>> # Caso 2: DCR moderato (0.65) - rumore moderato
    >>> synth_noisy = add_post_generation_noise(
    ...     synth_df, real_df,
    ...     current_dcr=0.65,
    ...     epsilon=1.5
    ... )
    """
    synth_df_noisy = synth_df.copy()

    # Auto-detect dummy variables (to exclude from noise)
    dummy_columns = []
    for col in synth_df.columns:
        # Check if column has separator and only binary values (0 or 1)
        if '/' in col:  # Common separator for dummies
            unique_vals = synth_df[col].dropna().unique()
            if set(unique_vals).issubset({0, 0.0, 1, 1.0}):
                dummy_columns.append(col)

    if dummy_columns:
        print(f"[INFO] Auto-detected {len(dummy_columns)} dummy columns (will NOT receive noise)")
        print(f"      Examples: {dummy_columns[:5]}")

    # Auto-detect categorical ordinal variables (to exclude from noise)
    # These are variables like APOE (0,1,2), EDUCATION (discrete years), etc.
    # They are generated as discrete values, unlike clinical scales which are continuous
    if categorical_ordinal_features is not None:
        # Use user-provided list
        categorical_ordinals = [col for col in categorical_ordinal_features if col in synth_df.columns]
        if categorical_ordinals:
            print(f"[INFO] Using user-specified categorical ordinal variables (will NOT receive noise)")
            print(f"      Variables: {categorical_ordinals}")
    else:
        # Auto-detect
        categorical_ordinals = []

        # Known categorical ordinals (common in ADNI)
        known_categoricals = ['APOE', 'APOE4', 'EDUCATION', 'PTGENDER', 'PTEDUCAT']

        for col in synth_df.columns:
            # Skip if already in dummy columns or special columns
            if col in dummy_columns or col in [patient_col, time_col]:
                continue

            # Check if it's in the known list
            if col in known_categoricals:
                categorical_ordinals.append(col)
                continue

            # Auto-detect: few unique integer values (≤ 10 unique values, all integers)
            if synth_df[col].dtype in ['int64', 'int32', 'float64', 'float32']:
                unique_vals = synth_df[col].dropna().unique()
                n_unique = len(unique_vals)

                # If ≤ 10 unique values AND all are integers (or very close to integers)
                if n_unique <= 10 and n_unique > 0:
                    # Check if all values are integers (within tolerance)
                    all_integers = all(abs(val - round(val)) < 1e-6 for val in unique_vals)
                    if all_integers:
                        # Additional check: values should be small integers (not like patient IDs)
                        max_val = max(unique_vals)
                        min_val = min(unique_vals)
                        if max_val - min_val <= 20:  # Range of values is small
                            categorical_ordinals.append(col)

        if categorical_ordinals:
            print(f"[INFO] Auto-detected {len(categorical_ordinals)} categorical ordinal variables (will NOT receive noise)")
            print(f"      Examples: {categorical_ordinals}")

    # Identifica feature types
    # IMPORTANT: ordinal_features (like MMSE, CDRSB) are normalized [0,1] and SHOULD receive noise
    # They will be rounded AFTER noise injection in post-processing
    # EXCLUDE: dummy variables and categorical ordinals (APOE, EDUCATION)
    if continuous_features is None:
        continuous_features = synth_df.select_dtypes(include=[np.number]).columns.tolist()
        # Exclude: patient_col, time_col, dummy variables, and categorical ordinals
        continuous_features = [f for f in continuous_features
                             if f not in [patient_col, time_col] + dummy_columns + categorical_ordinals]

    if continuous_features:
        print(f"[INFO] {len(continuous_features)} features will receive noise")
        print(f"      Features: {continuous_features}")
        if ordinal_features:
            ordinal_in_noise = [f for f in ordinal_features if f in continuous_features]
            if ordinal_in_noise:
                print(f"      Including ordinal scales (will be rounded during reverse): {ordinal_in_noise}")

    # Crea injector
    delta = 1.0 / (len(real_df) ** 2)  # Standard delta

    if preserve_temporal and patient_col in synth_df.columns:
        injector = LongitudinalNoiseInjector(epsilon=epsilon, delta=delta)

        # Aggiungi rumore preservando correlazioni temporali
        synth_df_noisy = injector.inject_correlated_noise(
            synth_df_noisy,
            patient_col=patient_col,
            time_col=time_col,
            features=continuous_features
        )
    else:
        injector = PostGenerationNoiseInjector(epsilon=epsilon, delta=delta)

        # Aggiungi rumore i.i.d. a ogni feature
        for feature in continuous_features:
            if feature not in synth_df_noisy.columns:
                continue

            # Calcola sensibilità dalla distribuzione reale
            real_values = real_df[feature].dropna().values
            sensitivity = injector.compute_sensitivity(real_values)

            # Seleziona meccanismo
            if method == 'gaussian':
                synth_df_noisy[feature] = injector.gaussian_mechanism(
                    synth_df_noisy[feature].values, sensitivity
                )
            elif method == 'laplace':
                synth_df_noisy[feature] = injector.laplace_mechanism(
                    synth_df_noisy[feature].values, sensitivity
                )
            elif method == 'adaptive':
                synth_df_noisy[feature] = injector.adaptive_noise(
                    synth_df_noisy[feature].values,
                    sensitivity,
                    target_dcr=target_dcr,
                    current_dcr=current_dcr
                )

    # Clip all continuous features to [0, 1] since they are normalized
    # This prevents negative values or values > 1 after noise injection
    print(f"[INFO] Clipping all {len(continuous_features)} features to [0, 1] range (normalized data)")
    for feature in continuous_features:
        if feature in synth_df_noisy.columns:
            # Check if clipping was needed
            min_val = synth_df_noisy[feature].min()
            max_val = synth_df_noisy[feature].max()
            synth_df_noisy[feature] = np.clip(synth_df_noisy[feature], 0.0, 1.0)
            if min_val < 0 or max_val > 1:
                print(f"      {feature}: clipped from [{min_val:.3f}, {max_val:.3f}] to [0, 1]")

    # NOTE: Ordinal features (MMSE, CDRSB, etc.) are NOT rounded here
    # because they are still normalized [0,1] at this stage.
    # They will be rounded to integers AFTER denormalization in reverse_from_datalake.py
    #
    # The clipping to [0,1] above is sufficient to keep them in valid range.

    return synth_df_noisy
