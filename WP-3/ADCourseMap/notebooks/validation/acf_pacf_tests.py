"""
ACF/PACF Tests for Synthetic Data Validation
==============================================

This module implements ACF (Autocorrelation Function) and PACF (Partial
Autocorrelation Function) tests to validate that synthetic data captures
the temporal autocorrelation structure of real data.

Critical validation objectives:
- ACF should match between real and synthetic data
- PACF should reveal similar autoregressive structure
- If real data is AR(1) but synthetic is AR(3), model may be overfitting
- If real data has autocorrelation but synthetic doesn't, model failed to capture temporal dependencies

Tests implemented:
- ACF: Measures correlation of series with its own lagged values
- PACF: Measures partial correlation controlling for intermediate lags
- AR/MA Order Detection: Automatically identifies AR(p) and MA(q) orders using AIC/BIC
- Comparison Metrics: MSE, MAE, Correlation between real and synthetic ACF/PACF

Author: IFAB
Date: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from typing import Dict, List, Optional, Tuple
import warnings
# Suppress specific statsmodels warnings for singular matrices in PACF
warnings.filterwarnings('ignore', message='Matrix is singular')
warnings.filterwarnings('ignore', category=UserWarning, module='statsmodels')


def compute_acf_pacf_per_patient(data: pd.DataFrame,
                                  variable: str,
                                  id_var: str = 'ID',
                                  time_var: str = 'TIME',
                                  nlags: Optional[int] = None) -> Tuple[List[Dict], List[Dict]]:
    """
    Compute ACF and PACF for each patient's trajectory separately.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset with patient trajectories
    variable : str
        Variable to analyze (e.g., 'MMSE', 'ADAS11')
    id_var : str, default='ID'
        Column name for patient ID
    time_var : str, default='TIME'
        Column name for time variable
    nlags : int, optional
        Number of lags to compute. If None, uses min(10, n_obs // 2)

    Returns
    -------
    acf_results : list of dict
        ACF results for each patient
    pacf_results : list of dict
        PACF results for each patient
    """
    acf_results = []
    pacf_results = []

    # Group by patient ID
    for patient_id, patient_data in data.groupby(id_var):
        # Sort by time
        patient_data = patient_data.sort_values(time_var)

        # Get time series for this variable
        timeseries = patient_data[variable].dropna()

        # Determine number of lags
        n_obs = len(timeseries)
        if n_obs < 5:
            continue  # Need at least 5 observations for reliable PACF

        if nlags is None:
            n_lags = min(10, n_obs // 2)
        else:
            n_lags = min(nlags, n_obs // 2)

        if n_lags < 1:
            continue

        try:
            # Compute ACF
            acf_values = acf(timeseries, nlags=n_lags, fft=False)

            # Compute PACF with OLS method (more robust for short series than Yule-Walker)
            # Suppress warnings for singular matrices
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='.*singular.*')
                warnings.filterwarnings('ignore', message='.*pinv.*')
                pacf_values = pacf(timeseries, nlags=n_lags, method='ols')

            acf_results.append({
                'patient_id': patient_id,
                'variable': variable,
                'n_obs': n_obs,
                'n_lags': n_lags,
                'acf': acf_values,
                'significant_lags_acf': np.where(np.abs(acf_values[1:]) > 1.96 / np.sqrt(n_obs))[0] + 1
            })

            pacf_results.append({
                'patient_id': patient_id,
                'variable': variable,
                'n_obs': n_obs,
                'n_lags': n_lags,
                'pacf': pacf_values,
                'significant_lags_pacf': np.where(np.abs(pacf_values[1:]) > 1.96 / np.sqrt(n_obs))[0] + 1
            })

        except Exception as e:
            # Skip patients where ACF/PACF calculation fails
            continue

    return acf_results, pacf_results


def detect_ar_ma_order(timeseries: pd.Series, max_p: int = 5, max_q: int = 5) -> Dict:
    """
    Automatically detect AR and MA orders using AIC/BIC criteria.

    Parameters
    ----------
    timeseries : pd.Series
        Time series data
    max_p : int, default=5
        Maximum AR order to test
    max_q : int, default=5
        Maximum MA order to test

    Returns
    -------
    results : dict
        Dictionary with detected orders and information criteria
    """
    if len(timeseries) < 10:
        return {
            'ar_order': None,
            'ma_order': None,
            'error': 'Insufficient observations (need at least 10)'
        }

    try:
        # Suppress warnings for singular matrices during AR/MA order detection
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='.*singular.*')
            warnings.filterwarnings('ignore', message='.*pinv.*')
            warnings.filterwarnings('ignore', category=UserWarning, module='statsmodels')
            # Test AR models
            ar_results = []
            for p in range(1, min(max_p + 1, len(timeseries) // 3)):
                try:
                    model = AutoReg(timeseries, lags=p, trend='c')
                    fitted = model.fit()
                    ar_results.append({
                        'order': p,
                        'aic': fitted.aic,
                        'bic': fitted.bic
                    })
                except:
                    continue

            # Find best AR order by AIC
            best_ar_aic = None
            best_ar_bic = None
            if ar_results:
                best_ar_aic = min(ar_results, key=lambda x: x['aic'])['order']
                best_ar_bic = min(ar_results, key=lambda x: x['bic'])['order']

            # Test MA models (using ARIMA with p=0)
            ma_results = []
            for q in range(1, min(max_q + 1, len(timeseries) // 3)):
                try:
                    model = ARIMA(timeseries, order=(0, 0, q), trend='c')
                    fitted = model.fit()
                    ma_results.append({
                        'order': q,
                        'aic': fitted.aic,
                        'bic': fitted.bic
                    })
                except:
                    continue

            # Find best MA order by AIC
            best_ma_aic = None
            best_ma_bic = None
            if ma_results:
                best_ma_aic = min(ma_results, key=lambda x: x['aic'])['order']
                best_ma_bic = min(ma_results, key=lambda x: x['bic'])['order']

            return {
                'ar_order_aic': best_ar_aic,
                'ar_order_bic': best_ar_bic,
                'ma_order_aic': best_ma_aic,
                'ma_order_bic': best_ma_bic,
                'ar_models': ar_results,
                'ma_models': ma_results
            }

    except Exception as e:
        return {
            'ar_order': None,
            'ma_order': None,
            'error': str(e)
        }


def detect_ar_ma_order_per_patient(data: pd.DataFrame,
                                    variable: str,
                                    id_var: str = 'ID',
                                    time_var: str = 'TIME',
                                    max_p: int = 3,
                                    max_q: int = 3) -> List[Dict]:
    """
    Detect AR/MA orders for each patient.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset with patient trajectories
    variable : str
        Variable to analyze
    id_var : str, default='ID'
        Column name for patient ID
    time_var : str, default='TIME'
        Column name for time variable
    max_p : int, default=3
        Maximum AR order to test
    max_q : int, default=3
        Maximum MA order to test

    Returns
    -------
    results : list of dict
        AR/MA order detection results for each patient
    """
    results = []

    for patient_id, patient_data in data.groupby(id_var):
        patient_data = patient_data.sort_values(time_var)
        timeseries = patient_data[variable].dropna()

        if len(timeseries) < 10:
            continue

        order_result = detect_ar_ma_order(timeseries, max_p=max_p, max_q=max_q)
        order_result['patient_id'] = patient_id
        order_result['variable'] = variable
        order_result['n_obs'] = len(timeseries)

        results.append(order_result)

    return results


def compare_acf_pacf(real_acf_results: List[Dict],
                     synth_acf_results: List[Dict],
                     real_pacf_results: List[Dict],
                     synth_pacf_results: List[Dict]) -> Dict:
    """
    Compare ACF and PACF between real and synthetic data.

    Computes MSE, MAE, and correlation between real and synthetic ACF/PACF values.

    Parameters
    ----------
    real_acf_results : list of dict
        ACF results for real data
    synth_acf_results : list of dict
        ACF results for synthetic data
    real_pacf_results : list of dict
        PACF results for real data
    synth_pacf_results : list of dict
        PACF results for synthetic data

    Returns
    -------
    comparison : dict
        Comparison metrics and interpretation
    """
    # Collect all ACF values (excluding lag 0 which is always 1.0)
    real_acf_all = np.concatenate([r['acf'][1:] for r in real_acf_results])
    synth_acf_all = np.concatenate([r['acf'][1:] for r in synth_acf_results])

    real_pacf_all = np.concatenate([r['pacf'][1:] for r in real_pacf_results])
    synth_pacf_all = np.concatenate([r['pacf'][1:] for r in synth_pacf_results])

    # Match lengths (use minimum)
    min_len_acf = min(len(real_acf_all), len(synth_acf_all))
    min_len_pacf = min(len(real_pacf_all), len(synth_pacf_all))

    real_acf_all = real_acf_all[:min_len_acf]
    synth_acf_all = synth_acf_all[:min_len_acf]
    real_pacf_all = real_pacf_all[:min_len_pacf]
    synth_pacf_all = synth_pacf_all[:min_len_pacf]

    # Compute metrics for ACF
    mse_acf = np.mean((real_acf_all - synth_acf_all) ** 2)
    mae_acf = np.mean(np.abs(real_acf_all - synth_acf_all))
    corr_acf = np.corrcoef(real_acf_all, synth_acf_all)[0, 1] if len(real_acf_all) > 1 else 0.0

    # Compute metrics for PACF
    mse_pacf = np.mean((real_pacf_all - synth_pacf_all) ** 2)
    mae_pacf = np.mean(np.abs(real_pacf_all - synth_pacf_all))
    corr_pacf = np.corrcoef(real_pacf_all, synth_pacf_all)[0, 1] if len(real_pacf_all) > 1 else 0.0

    # Pass criteria: MSE < 0.05, Correlation > 0.7
    passed_acf = (mse_acf < 0.05) and (corr_acf > 0.7)
    passed_pacf = (mse_pacf < 0.05) and (corr_pacf > 0.7)
    passed_overall = passed_acf and passed_pacf

    # Interpretation
    if passed_overall:
        interpretation = (
            "GOOD: Synthetic data successfully captures the autocorrelation "
            "structure of real data. Both ACF and PACF match well."
        )
        status = "PASSED"
    elif passed_acf and not passed_pacf:
        interpretation = (
            "WARNING: ACF matches well but PACF differs. The synthetic model "
            "may have different autoregressive structure (e.g., AR(1) vs AR(2))."
        )
        status = "WARNING"
    elif not passed_acf and passed_pacf:
        interpretation = (
            "WARNING: PACF matches but ACF differs. Check for different "
            "moving average components or long-term dependencies."
        )
        status = "WARNING"
    else:
        interpretation = (
            "FAILED: Both ACF and PACF differ significantly. The synthetic model "
            "is not capturing the temporal autocorrelation structure correctly."
        )
        status = "FAILED"

    return {
        'acf': {
            'mse': float(mse_acf),
            'mae': float(mae_acf),
            'correlation': float(corr_acf),
            'passed': passed_acf
        },
        'pacf': {
            'mse': float(mse_pacf),
            'mae': float(mae_pacf),
            'correlation': float(corr_pacf),
            'passed': passed_pacf
        },
        'overall_passed': passed_overall,
        'status': status,
        'interpretation': interpretation,
        'n_real_patients': len(real_acf_results),
        'n_synth_patients': len(synth_acf_results)
    }


def compare_ar_ma_orders(real_orders: List[Dict],
                         synth_orders: List[Dict]) -> Dict:
    """
    Compare AR/MA order distributions between real and synthetic data.

    Parameters
    ----------
    real_orders : list of dict
        AR/MA order detection results for real data
    synth_orders : list of dict
        AR/MA order detection results for synthetic data

    Returns
    -------
    comparison : dict
        Distribution comparison and interpretation
    """
    # Count AR orders (using BIC as it's more conservative)
    real_ar_orders = [r['ar_order_bic'] for r in real_orders if r.get('ar_order_bic') is not None]
    synth_ar_orders = [r['ar_order_bic'] for r in synth_orders if r.get('ar_order_bic') is not None]

    real_ma_orders = [r['ma_order_bic'] for r in real_orders if r.get('ma_order_bic') is not None]
    synth_ma_orders = [r['ma_order_bic'] for r in synth_orders if r.get('ma_order_bic') is not None]

    # Calculate distributions
    def order_distribution(orders):
        if not orders:
            return {}
        unique, counts = np.unique(orders, return_counts=True)
        total = len(orders)
        return {int(order): float(count / total * 100) for order, count in zip(unique, counts)}

    real_ar_dist = order_distribution(real_ar_orders)
    synth_ar_dist = order_distribution(synth_ar_orders)
    real_ma_dist = order_distribution(real_ma_orders)
    synth_ma_dist = order_distribution(synth_ma_orders)

    # Most common orders
    real_ar_mode = max(real_ar_dist, key=real_ar_dist.get) if real_ar_dist else None
    synth_ar_mode = max(synth_ar_dist, key=synth_ar_dist.get) if synth_ar_dist else None
    real_ma_mode = max(real_ma_dist, key=real_ma_dist.get) if real_ma_dist else None
    synth_ma_mode = max(synth_ma_dist, key=synth_ma_dist.get) if synth_ma_dist else None

    # Check if distributions match
    ar_match = (real_ar_mode == synth_ar_mode) if (real_ar_mode and synth_ar_mode) else False
    ma_match = (real_ma_mode == synth_ma_mode) if (real_ma_mode and synth_ma_mode) else False

    # Interpretation
    if ar_match and ma_match:
        interpretation = (
            f"GOOD: Synthetic data has same AR/MA structure as real data. "
            f"Most common: AR({real_ar_mode}), MA({real_ma_mode})."
        )
        status = "PASSED"
    elif ar_match or ma_match:
        interpretation = (
            f"WARNING: Partial match. Real: AR({real_ar_mode}), MA({real_ma_mode}). "
            f"Synthetic: AR({synth_ar_mode}), MA({synth_ma_mode})."
        )
        status = "WARNING"
    else:
        interpretation = (
            f"FAILED: Different AR/MA structure. Real: AR({real_ar_mode}), MA({real_ma_mode}). "
            f"Synthetic: AR({synth_ar_mode}), MA({synth_ma_mode})."
        )
        status = "FAILED"

    return {
        'ar_orders': {
            'real_distribution': real_ar_dist,
            'synth_distribution': synth_ar_dist,
            'real_mode': real_ar_mode,
            'synth_mode': synth_ar_mode,
            'match': ar_match
        },
        'ma_orders': {
            'real_distribution': real_ma_dist,
            'synth_distribution': synth_ma_dist,
            'real_mode': real_ma_mode,
            'synth_mode': synth_ma_mode,
            'match': ma_match
        },
        'status': status,
        'interpretation': interpretation
    }


def plot_acf_pacf_comparison(real_acf: List[Dict],
                              synth_acf: List[Dict],
                              real_pacf: List[Dict],
                              synth_pacf: List[Dict],
                              comparison: Dict,
                              variable_name: str = "",
                              save_path: Optional[str] = None) -> plt.Figure:
    """
    Create comprehensive visualization of ACF/PACF comparison.

    Parameters
    ----------
    real_acf : list of dict
        ACF results for real data
    synth_acf : list of dict
        ACF results for synthetic data
    real_pacf : list of dict
        PACF results for real data
    synth_pacf : list of dict
        PACF results for synthetic data
    comparison : dict
        Comparison metrics
    variable_name : str, optional
        Name of the variable being analyzed
    save_path : str, optional
        Path to save the figure

    Returns
    -------
    fig : plt.Figure
        Matplotlib figure object
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Compute mean ACF/PACF across patients
    max_lag = min(
        max([r['n_lags'] for r in real_acf]),
        max([r['n_lags'] for r in synth_acf])
    )

    # Average ACF for real data
    real_acf_mean = np.zeros(max_lag + 1)
    for r in real_acf:
        n_lags = min(len(r['acf']), max_lag + 1)
        real_acf_mean[:n_lags] += r['acf'][:n_lags]
    real_acf_mean /= len(real_acf)

    # Average ACF for synthetic data
    synth_acf_mean = np.zeros(max_lag + 1)
    for r in synth_acf:
        n_lags = min(len(r['acf']), max_lag + 1)
        synth_acf_mean[:n_lags] += r['acf'][:n_lags]
    synth_acf_mean /= len(synth_acf)

    # Average PACF for real data
    real_pacf_mean = np.zeros(max_lag + 1)
    for r in real_pacf:
        n_lags = min(len(r['pacf']), max_lag + 1)
        real_pacf_mean[:n_lags] += r['pacf'][:n_lags]
    real_pacf_mean /= len(real_pacf)

    # Average PACF for synthetic data
    synth_pacf_mean = np.zeros(max_lag + 1)
    for r in synth_pacf:
        n_lags = min(len(r['pacf']), max_lag + 1)
        synth_pacf_mean[:n_lags] += r['pacf'][:n_lags]
    synth_pacf_mean /= len(synth_pacf)

    lags = np.arange(max_lag + 1)

    # Confidence interval (95%)
    n_avg = len(real_acf)
    conf_interval = 1.96 / np.sqrt(n_avg)

    # ========== Panel 1: Real ACF ==========
    ax = axes[0, 0]
    ax.stem(lags, real_acf_mean, linefmt='b-', markerfmt='bo', basefmt=' ')
    ax.axhline(0, color='black', linewidth=0.8)
    ax.axhline(conf_interval, color='red', linestyle='--', linewidth=1, label='95% CI')
    ax.axhline(-conf_interval, color='red', linestyle='--', linewidth=1)
    ax.set_xlabel('Lag', fontsize=12)
    ax.set_ylabel('ACF', fontsize=12)
    ax.set_title(f'Real Data - ACF\n(n={len(real_acf)} patients)', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # ========== Panel 2: Synthetic ACF ==========
    ax = axes[0, 1]
    ax.stem(lags, synth_acf_mean, linefmt='g-', markerfmt='go', basefmt=' ')
    ax.axhline(0, color='black', linewidth=0.8)
    ax.axhline(conf_interval, color='red', linestyle='--', linewidth=1, label='95% CI')
    ax.axhline(-conf_interval, color='red', linestyle='--', linewidth=1)
    ax.set_xlabel('Lag', fontsize=12)
    ax.set_ylabel('ACF', fontsize=12)
    ax.set_title(f'Synthetic Data - ACF\n(n={len(synth_acf)} patients)', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # ========== Panel 3: Real PACF ==========
    ax = axes[1, 0]
    ax.stem(lags, real_pacf_mean, linefmt='b-', markerfmt='bo', basefmt=' ')
    ax.axhline(0, color='black', linewidth=0.8)
    ax.axhline(conf_interval, color='red', linestyle='--', linewidth=1, label='95% CI')
    ax.axhline(-conf_interval, color='red', linestyle='--', linewidth=1)
    ax.set_xlabel('Lag', fontsize=12)
    ax.set_ylabel('PACF', fontsize=12)
    ax.set_title(f'Real Data - PACF\n(AR order indicator)', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # ========== Panel 4: Synthetic PACF ==========
    ax = axes[1, 1]
    ax.stem(lags, synth_pacf_mean, linefmt='g-', markerfmt='go', basefmt=' ')
    ax.axhline(0, color='black', linewidth=0.8)
    ax.axhline(conf_interval, color='red', linestyle='--', linewidth=1, label='95% CI')
    ax.axhline(-conf_interval, color='red', linestyle='--', linewidth=1)
    ax.set_xlabel('Lag', fontsize=12)
    ax.set_ylabel('PACF', fontsize=12)
    ax.set_title(f'Synthetic Data - PACF\n(AR order indicator)', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # Overall title with metrics
    status_color = {
        'PASSED': 'green',
        'WARNING': 'orange',
        'FAILED': 'red'
    }

    fig.suptitle(
        f'ACF/PACF Comparison: {variable_name}\n'
        f'ACF MSE: {comparison["acf"]["mse"]:.4f}, Corr: {comparison["acf"]["correlation"]:.3f} | '
        f'PACF MSE: {comparison["pacf"]["mse"]:.4f}, Corr: {comparison["pacf"]["correlation"]:.3f} | '
        f'Status: {comparison["status"]}',
        fontsize=14, fontweight='bold',
        color=status_color.get(comparison['status'], 'black')
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Figure saved to: {save_path}")

    return fig


def save_acf_pacf_summary(comparison_acf: Dict,
                           comparison_orders: Dict,
                           variable_name: str,
                           save_path: str = 'acf_pacf_summary.txt'):
    """
    Save detailed summary of ACF/PACF tests to text file.

    Parameters
    ----------
    comparison_acf : dict
        ACF/PACF comparison results
    comparison_orders : dict
        AR/MA order comparison results
    variable_name : str
        Name of the variable analyzed
    save_path : str
        Path to save the summary file
    """
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"ACF/PACF TESTS SUMMARY: {variable_name}\n")
        f.write("=" * 80 + "\n\n")

        f.write("Objective:\n")
        f.write("-" * 80 + "\n")
        f.write("Validate that synthetic data captures the temporal autocorrelation\n")
        f.write("structure of real data.\n\n")
        f.write("CRITICAL: If ACF/PACF patterns differ, the synthetic model may be:\n")
        f.write("  - Missing temporal dependencies (no autocorrelation)\n")
        f.write("  - Overfitting (too high AR/MA order)\n")
        f.write("  - Using wrong model structure (e.g., AR(1) vs AR(3))\n\n")

        # ACF Results
        f.write("=" * 80 + "\n")
        f.write("AUTOCORRELATION FUNCTION (ACF) COMPARISON\n")
        f.write("=" * 80 + "\n\n")

        f.write("What is ACF?\n")
        f.write("-" * 80 + "\n")
        f.write("ACF measures correlation between a time series and its lagged values.\n")
        f.write("It reveals overall temporal dependencies and patterns.\n\n")

        f.write("Metrics:\n")
        f.write(f"  MSE (Mean Squared Error):    {comparison_acf['acf']['mse']:.6f}\n")
        f.write(f"  MAE (Mean Absolute Error):   {comparison_acf['acf']['mae']:.6f}\n")
        f.write(f"  Correlation:                 {comparison_acf['acf']['correlation']:.4f}\n\n")

        f.write("Thresholds:\n")
        f.write("  MSE < 0.05:     GOOD autocorrelation match\n")
        f.write("  Correlation > 0.7: GOOD pattern similarity\n\n")

        acf_status = "PASSED" if comparison_acf['acf']['passed'] else "FAILED"
        f.write(f"ACF Status: {acf_status}\n\n")

        # PACF Results
        f.write("=" * 80 + "\n")
        f.write("PARTIAL AUTOCORRELATION FUNCTION (PACF) COMPARISON\n")
        f.write("=" * 80 + "\n\n")

        f.write("What is PACF?\n")
        f.write("-" * 80 + "\n")
        f.write("PACF measures correlation at specific lags, controlling for shorter lags.\n")
        f.write("It helps identify AR order (number of significant lags = AR order).\n\n")

        f.write("Metrics:\n")
        f.write(f"  MSE (Mean Squared Error):    {comparison_acf['pacf']['mse']:.6f}\n")
        f.write(f"  MAE (Mean Absolute Error):   {comparison_acf['pacf']['mae']:.6f}\n")
        f.write(f"  Correlation:                 {comparison_acf['pacf']['correlation']:.4f}\n\n")

        pacf_status = "PASSED" if comparison_acf['pacf']['passed'] else "FAILED"
        f.write(f"PACF Status: {pacf_status}\n\n")

        # AR/MA Order Comparison
        f.write("=" * 80 + "\n")
        f.write("AR/MA ORDER DETECTION (AIC/BIC)\n")
        f.write("=" * 80 + "\n\n")

        f.write("What is AR/MA order?\n")
        f.write("-" * 80 + "\n")
        f.write("AR(p): Autoregressive model of order p (depends on p previous values)\n")
        f.write("MA(q): Moving average model of order q (depends on q previous errors)\n\n")

        # AR Orders
        real_ar_mode = comparison_orders['ar_orders']['real_mode']
        synth_ar_mode = comparison_orders['ar_orders']['synth_mode']
        ar_match = comparison_orders['ar_orders']['match']

        f.write("AR Order (Autoregressive):\n")
        f.write(f"  Most common in Real data:      AR({real_ar_mode})\n")
        f.write(f"  Most common in Synthetic data: AR({synth_ar_mode})\n")
        f.write(f"  Match: {'YES' if ar_match else 'NO'}\n\n")

        if comparison_orders['ar_orders']['real_distribution']:
            f.write("  Real AR order distribution:\n")
            for order, pct in sorted(comparison_orders['ar_orders']['real_distribution'].items()):
                f.write(f"    AR({order}): {pct:.1f}%\n")
            f.write("\n")

        if comparison_orders['ar_orders']['synth_distribution']:
            f.write("  Synthetic AR order distribution:\n")
            for order, pct in sorted(comparison_orders['ar_orders']['synth_distribution'].items()):
                f.write(f"    AR({order}): {pct:.1f}%\n")
            f.write("\n")

        # MA Orders
        real_ma_mode = comparison_orders['ma_orders']['real_mode']
        synth_ma_mode = comparison_orders['ma_orders']['synth_mode']
        ma_match = comparison_orders['ma_orders']['match']

        f.write("MA Order (Moving Average):\n")
        f.write(f"  Most common in Real data:      MA({real_ma_mode})\n")
        f.write(f"  Most common in Synthetic data: MA({synth_ma_mode})\n")
        f.write(f"  Match: {'YES' if ma_match else 'NO'}\n\n")

        # Overall conclusion
        f.write("=" * 80 + "\n")
        f.write("OVERALL CONCLUSION\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Status: {comparison_acf['status']}\n\n")
        f.write(f"Interpretation:\n{comparison_acf['interpretation']}\n\n")
        f.write(f"AR/MA Structure: {comparison_orders['interpretation']}\n\n")

        if comparison_acf['overall_passed'] and comparison_orders['status'] == 'PASSED':
            f.write("RECOMMENDATION: Synthetic data is validated for temporal dynamics.\n")
            f.write("                The model correctly captures autocorrelation patterns.\n")
        elif comparison_acf['status'] == 'WARNING' or comparison_orders['status'] == 'WARNING':
            f.write("RECOMMENDATION: Review model configuration. There are differences in\n")
            f.write("                temporal structure that may affect longitudinal analyses.\n")
        else:
            f.write("RECOMMENDATION: CRITICAL - The synthetic model is not capturing temporal\n")
            f.write("                dependencies correctly. Revise model architecture.\n")

        f.write("=" * 80 + "\n")

    print(f"  ACF/PACF summary saved to: {save_path}")
