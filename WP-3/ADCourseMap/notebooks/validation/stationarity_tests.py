"""
Stationarity Tests for Synthetic Data Validation
=================================================

This module implements stationarity tests to validate that synthetic data
captures the disease progression characteristics of real data.

Critical validation objective:
- Real data should be non-stationary (disease progression over time)
- Synthetic data should also be non-stationary
- If real is non-stationary but synthetic is stationary → MODEL FAILED
  (generating static patients instead of patients that worsen)

Tests implemented:
- ADF (Augmented Dickey-Fuller): Tests null hypothesis of non-stationarity
- KPSS (Kwiatkowski-Phillips-Schmidt-Shin): Tests null hypothesis of stationarity

Author: IFAB
Date: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller, kpss
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


def perform_adf_test(timeseries: pd.Series, variable_name: str = "") -> Dict:
    """
    Perform Augmented Dickey-Fuller test for stationarity.

    ADF tests the null hypothesis that the series has a unit root (non-stationary).
    - If p-value < 0.05: Reject H0 → Series is STATIONARY
    - If p-value >= 0.05: Cannot reject H0 → Series is NON-STATIONARY

    Parameters
    ----------
    timeseries : pd.Series
        Time series data to test
    variable_name : str, optional
        Name of the variable being tested

    Returns
    -------
    results : dict
        Dictionary containing test statistic, p-value, and interpretation
    """
    # Remove NaN values
    ts_clean = timeseries.dropna()

    if len(ts_clean) < 3:
        return {
            'variable': variable_name,
            'test': 'ADF',
            'n_observations': len(ts_clean),
            'statistic': None,
            'p_value': None,
            'critical_values': None,
            'is_stationary': None,
            'error': 'Insufficient data points (need at least 3)'
        }

    try:
        # Perform ADF test
        # regression='c' includes constant (intercept)
        # autolag='AIC' automatically selects lag order
        adf_result = adfuller(ts_clean, regression='c', autolag='AIC')

        adf_statistic = adf_result[0]
        p_value = adf_result[1]
        n_lags = adf_result[2]
        n_obs = adf_result[3]
        critical_values = adf_result[4]

        # Interpretation: p < 0.05 → stationary
        is_stationary = p_value < 0.05

        return {
            'variable': variable_name,
            'test': 'ADF',
            'n_observations': n_obs,
            'n_lags_used': n_lags,
            'statistic': float(adf_statistic),
            'p_value': float(p_value),
            'critical_values': {k: float(v) for k, v in critical_values.items()},
            'is_stationary': is_stationary,
            'interpretation': 'Stationary' if is_stationary else 'Non-stationary'
        }

    except Exception as e:
        return {
            'variable': variable_name,
            'test': 'ADF',
            'n_observations': len(ts_clean),
            'statistic': None,
            'p_value': None,
            'critical_values': None,
            'is_stationary': None,
            'error': str(e)
        }


def perform_kpss_test(timeseries: pd.Series, variable_name: str = "",
                      regression: str = 'c') -> Dict:
    """
    Perform KPSS test for stationarity.

    KPSS tests the null hypothesis that the series is stationary.
    - If p-value < 0.05: Reject H0 → Series is NON-STATIONARY
    - If p-value >= 0.05: Cannot reject H0 → Series is STATIONARY

    Note: KPSS is opposite to ADF in terms of null hypothesis!

    Parameters
    ----------
    timeseries : pd.Series
        Time series data to test
    variable_name : str, optional
        Name of the variable being tested
    regression : str, default='c'
        Type of regression ('c' for constant, 'ct' for constant + trend)

    Returns
    -------
    results : dict
        Dictionary containing test statistic, p-value, and interpretation
    """
    # Remove NaN values
    ts_clean = timeseries.dropna()

    if len(ts_clean) < 3:
        return {
            'variable': variable_name,
            'test': 'KPSS',
            'n_observations': len(ts_clean),
            'statistic': None,
            'p_value': None,
            'critical_values': None,
            'is_stationary': None,
            'error': 'Insufficient data points (need at least 3)'
        }

    try:
        # Perform KPSS test
        # nlags='auto' automatically selects lag order
        kpss_result = kpss(ts_clean, regression=regression, nlags='auto')

        kpss_statistic = kpss_result[0]
        p_value = kpss_result[1]
        n_lags = kpss_result[2]
        critical_values = kpss_result[3]

        # Interpretation: p >= 0.05 → stationary (opposite of ADF!)
        is_stationary = p_value >= 0.05

        return {
            'variable': variable_name,
            'test': 'KPSS',
            'n_observations': len(ts_clean),
            'n_lags_used': n_lags,
            'statistic': float(kpss_statistic),
            'p_value': float(p_value),
            'critical_values': {k: float(v) for k, v in critical_values.items()},
            'is_stationary': is_stationary,
            'interpretation': 'Stationary' if is_stationary else 'Non-stationary'
        }

    except Exception as e:
        return {
            'variable': variable_name,
            'test': 'KPSS',
            'n_observations': len(ts_clean),
            'statistic': None,
            'p_value': None,
            'critical_values': None,
            'is_stationary': None,
            'error': str(e)
        }


def test_stationarity_per_patient(data: pd.DataFrame,
                                  variable: str,
                                  id_var: str = 'ID',
                                  time_var: str = 'TIME') -> Tuple[List[Dict], List[Dict]]:
    """
    Test stationarity for each patient's trajectory separately.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset with patient trajectories
    variable : str
        Variable to test (e.g., 'MMSE', 'ADAS11')
    id_var : str, default='ID'
        Column name for patient ID
    time_var : str, default='TIME'
        Column name for time variable

    Returns
    -------
    adf_results : list of dict
        ADF test results for each patient
    kpss_results : list of dict
        KPSS test results for each patient
    """
    adf_results = []
    kpss_results = []

    # Group by patient ID
    for patient_id, patient_data in data.groupby(id_var):
        # Sort by time
        patient_data = patient_data.sort_values(time_var)

        # Get time series for this variable
        timeseries = patient_data[variable]

        # Skip if insufficient data points
        if len(timeseries.dropna()) < 3:
            continue

        # Perform tests
        adf_result = perform_adf_test(timeseries, variable_name=f"{variable}_patient_{patient_id}")
        kpss_result = perform_kpss_test(timeseries, variable_name=f"{variable}_patient_{patient_id}")

        if 'error' not in adf_result:
            adf_results.append(adf_result)
        if 'error' not in kpss_result:
            kpss_results.append(kpss_result)

    return adf_results, kpss_results


def test_stationarity_aggregated(data: pd.DataFrame,
                                 variable: str,
                                 id_var: str = 'ID',
                                 time_var: str = 'TIME') -> Tuple[Dict, Dict]:
    """
    Test stationarity on aggregated time series (mean across all patients at each time point).

    Parameters
    ----------
    data : pd.DataFrame
        Dataset with patient trajectories
    variable : str
        Variable to test (e.g., 'MMSE', 'ADAS11')
    id_var : str, default='ID'
        Column name for patient ID
    time_var : str, default='TIME'
        Column name for time variable

    Returns
    -------
    adf_result : dict
        ADF test result
    kpss_result : dict
        KPSS test result
    """
    # Calculate mean across all patients at each time point
    aggregated = data.groupby(time_var)[variable].mean()

    # Perform tests
    adf_result = perform_adf_test(aggregated, variable_name=f"{variable}_aggregated")
    kpss_result = perform_kpss_test(aggregated, variable_name=f"{variable}_aggregated")

    return adf_result, kpss_result


def compare_stationarity(real_results: List[Dict],
                        synth_results: List[Dict],
                        test_name: str = 'ADF') -> Dict:
    """
    Compare stationarity test results between real and synthetic data.

    Critical check: If real is non-stationary but synthetic is stationary,
    the model has failed to capture disease progression.

    Parameters
    ----------
    real_results : list of dict
        Test results for real data
    synth_results : list of dict
        Test results for synthetic data
    test_name : str, default='ADF'
        Name of the test ('ADF' or 'KPSS')

    Returns
    -------
    comparison : dict
        Comparison statistics and interpretation
    """
    # Count stationary vs non-stationary in each dataset
    real_stationary = sum(1 for r in real_results if r.get('is_stationary') == True)
    real_nonstationary = sum(1 for r in real_results if r.get('is_stationary') == False)
    real_total = len(real_results)

    synth_stationary = sum(1 for r in synth_results if r.get('is_stationary') == True)
    synth_nonstationary = sum(1 for r in synth_results if r.get('is_stationary') == False)
    synth_total = len(synth_results)

    # Calculate proportions
    real_stationary_pct = (real_stationary / real_total * 100) if real_total > 0 else 0
    synth_stationary_pct = (synth_stationary / synth_total * 100) if synth_total > 0 else 0

    # Critical check: Real non-stationary but synthetic stationary → FAILURE
    # This means the model is generating static patients instead of disease progression
    real_mostly_nonstationary = real_stationary_pct < 50  # Most patients show progression
    synth_mostly_stationary = synth_stationary_pct >= 50  # Most patients are static

    model_failed = real_mostly_nonstationary and synth_mostly_stationary

    # Detailed interpretation
    if model_failed:
        interpretation = (
            "CRITICAL FAILURE: Real data shows disease progression (non-stationary) "
            "but synthetic data is static (stationary). The model failed to capture "
            "the temporal dynamics of disease progression."
        )
        status = "FAILED"
    elif abs(real_stationary_pct - synth_stationary_pct) < 20:
        interpretation = (
            "GOOD: Synthetic data matches real data stationarity patterns. "
            "The model successfully captures disease progression dynamics."
        )
        status = "PASSED"
    else:
        interpretation = (
            "WARNING: Moderate difference in stationarity patterns between "
            "real and synthetic data. Review individual patient trajectories."
        )
        status = "WARNING"

    return {
        'test': test_name,
        'real': {
            'total': real_total,
            'stationary': real_stationary,
            'non_stationary': real_nonstationary,
            'stationary_percentage': real_stationary_pct
        },
        'synthetic': {
            'total': synth_total,
            'stationary': synth_stationary,
            'non_stationary': synth_nonstationary,
            'stationary_percentage': synth_stationary_pct
        },
        'difference_pct': abs(real_stationary_pct - synth_stationary_pct),
        'model_failed': model_failed,
        'status': status,
        'interpretation': interpretation
    }


def plot_stationarity_comparison(real_adf: List[Dict],
                                 synth_adf: List[Dict],
                                 real_kpss: List[Dict],
                                 synth_kpss: List[Dict],
                                 comparison_adf: Dict,
                                 comparison_kpss: Dict,
                                 save_path: Optional[str] = None) -> plt.Figure:
    """
    Create comprehensive visualization of stationarity test results.

    Parameters
    ----------
    real_adf : list of dict
        ADF results for real data
    synth_adf : list of dict
        ADF results for synthetic data
    real_kpss : list of dict
        KPSS results for real data
    synth_kpss : list of dict
        KPSS results for synthetic data
    comparison_adf : dict
        ADF comparison statistics
    comparison_kpss : dict
        KPSS comparison statistics
    save_path : str, optional
        Path to save the figure

    Returns
    -------
    fig : plt.Figure
        Matplotlib figure object
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Color scheme
    colors = {'Real': '#2E86AB', 'Synthetic': '#A23B72'}

    # ========== Panel 1: ADF p-values distribution ==========
    ax = axes[0, 0]

    real_adf_pvals = [r['p_value'] for r in real_adf if r.get('p_value') is not None]
    synth_adf_pvals = [r['p_value'] for r in synth_adf if r.get('p_value') is not None]

    if real_adf_pvals and synth_adf_pvals:
        ax.hist(real_adf_pvals, bins=20, alpha=0.6, label='Real',
                color=colors['Real'], edgecolor='black')
        ax.hist(synth_adf_pvals, bins=20, alpha=0.6, label='Synthetic',
                color=colors['Synthetic'], edgecolor='black')
        ax.axvline(0.05, color='red', linestyle='--', linewidth=2,
                  label='p=0.05 threshold')
        ax.set_xlabel('ADF p-value', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('ADF Test: p-value Distribution\n(p < 0.05 -> Stationary)',
                    fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)

    # ========== Panel 2: KPSS p-values distribution ==========
    ax = axes[0, 1]

    real_kpss_pvals = [r['p_value'] for r in real_kpss if r.get('p_value') is not None]
    synth_kpss_pvals = [r['p_value'] for r in synth_kpss if r.get('p_value') is not None]

    if real_kpss_pvals and synth_kpss_pvals:
        ax.hist(real_kpss_pvals, bins=20, alpha=0.6, label='Real',
                color=colors['Real'], edgecolor='black')
        ax.hist(synth_kpss_pvals, bins=20, alpha=0.6, label='Synthetic',
                color=colors['Synthetic'], edgecolor='black')
        ax.axvline(0.05, color='red', linestyle='--', linewidth=2,
                  label='p=0.05 threshold')
        ax.set_xlabel('KPSS p-value', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('KPSS Test: p-value Distribution\n(p >= 0.05 -> Stationary)',
                    fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)

    # ========== Panel 3: ADF Stationarity Comparison ==========
    ax = axes[1, 0]

    categories = ['Real Data', 'Synthetic Data']
    stationary = [
        comparison_adf['real']['stationary_percentage'],
        comparison_adf['synthetic']['stationary_percentage']
    ]
    non_stationary = [
        100 - comparison_adf['real']['stationary_percentage'],
        100 - comparison_adf['synthetic']['stationary_percentage']
    ]

    x = np.arange(len(categories))
    width = 0.35

    ax.bar(x - width/2, stationary, width, label='Stationary',
           color='#06A77D', edgecolor='black')
    ax.bar(x + width/2, non_stationary, width, label='Non-stationary',
           color='#D62246', edgecolor='black')

    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_title('ADF Test: Stationarity Comparison', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Add percentage labels on bars
    for i, (s, ns) in enumerate(zip(stationary, non_stationary)):
        if s > 5:
            ax.text(i - width/2, s/2, f'{s:.1f}%', ha='center', va='center',
                   fontweight='bold', fontsize=10)
        if ns > 5:
            ax.text(i + width/2, s + ns/2, f'{ns:.1f}%', ha='center', va='center',
                   fontweight='bold', fontsize=10)

    # ========== Panel 4: KPSS Stationarity Comparison ==========
    ax = axes[1, 1]

    stationary_kpss = [
        comparison_kpss['real']['stationary_percentage'],
        comparison_kpss['synthetic']['stationary_percentage']
    ]
    non_stationary_kpss = [
        100 - comparison_kpss['real']['stationary_percentage'],
        100 - comparison_kpss['synthetic']['stationary_percentage']
    ]

    ax.bar(x - width/2, stationary_kpss, width, label='Stationary',
           color='#06A77D', edgecolor='black')
    ax.bar(x + width/2, non_stationary_kpss, width, label='Non-stationary',
           color='#D62246', edgecolor='black')

    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_title('KPSS Test: Stationarity Comparison', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Add percentage labels on bars
    for i, (s, ns) in enumerate(zip(stationary_kpss, non_stationary_kpss)):
        if s > 5:
            ax.text(i - width/2, s/2, f'{s:.1f}%', ha='center', va='center',
                   fontweight='bold', fontsize=10)
        if ns > 5:
            ax.text(i + width/2, s + ns/2, f'{ns:.1f}%', ha='center', va='center',
                   fontweight='bold', fontsize=10)

    # Overall title with status
    status_adf = comparison_adf['status']
    status_kpss = comparison_kpss['status']

    status_color = {
        'PASSED': 'green',
        'WARNING': 'orange',
        'FAILED': 'red'
    }

    fig.suptitle(
        f'Stationarity Tests Comparison\n'
        f'ADF: {status_adf} | KPSS: {status_kpss}',
        fontsize=16, fontweight='bold',
        color=status_color.get(status_adf, 'black')
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Figure saved to: {save_path}")

    return fig


def save_stationarity_summary(comparison_adf: Dict,
                              comparison_kpss: Dict,
                              save_path: str = 'stationarity_summary.txt'):
    """
    Save detailed summary of stationarity tests to text file.

    Parameters
    ----------
    comparison_adf : dict
        ADF comparison results
    comparison_kpss : dict
        KPSS comparison results
    save_path : str
        Path to save the summary file
    """
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("STATIONARITY TESTS SUMMARY\n")
        f.write("=" * 80 + "\n\n")

        f.write("Objective:\n")
        f.write("-" * 80 + "\n")
        f.write("Validate that synthetic data captures disease progression dynamics.\n")
        f.write("CRITICAL: If real data is non-stationary (disease progression) but\n")
        f.write("synthetic data is stationary (static patients), the model has FAILED.\n\n")

        # ADF Results
        f.write("=" * 80 + "\n")
        f.write("AUGMENTED DICKEY-FULLER (ADF) TEST\n")
        f.write("=" * 80 + "\n")
        f.write("Null Hypothesis: Time series is non-stationary\n")
        f.write("Interpretation: p < 0.05 -> Stationary (reject H0)\n\n")

        f.write(f"Status: {comparison_adf['status']}\n\n")

        f.write("Real Data:\n")
        f.write(f"  Total patients tested: {comparison_adf['real']['total']}\n")
        f.write(f"  Stationary: {comparison_adf['real']['stationary']} "
                f"({comparison_adf['real']['stationary_percentage']:.1f}%)\n")
        f.write(f"  Non-stationary: {comparison_adf['real']['non_stationary']} "
                f"({100 - comparison_adf['real']['stationary_percentage']:.1f}%)\n\n")

        f.write("Synthetic Data:\n")
        f.write(f"  Total patients tested: {comparison_adf['synthetic']['total']}\n")
        f.write(f"  Stationary: {comparison_adf['synthetic']['stationary']} "
                f"({comparison_adf['synthetic']['stationary_percentage']:.1f}%)\n")
        f.write(f"  Non-stationary: {comparison_adf['synthetic']['non_stationary']} "
                f"({100 - comparison_adf['synthetic']['stationary_percentage']:.1f}%)\n\n")

        f.write(f"Difference: {comparison_adf['difference_pct']:.1f}%\n\n")
        f.write(f"Interpretation:\n{comparison_adf['interpretation']}\n\n")

        # KPSS Results
        f.write("=" * 80 + "\n")
        f.write("KWIATKOWSKI-PHILLIPS-SCHMIDT-SHIN (KPSS) TEST\n")
        f.write("=" * 80 + "\n")
        f.write("Null Hypothesis: Time series is stationary\n")
        f.write("Interpretation: p >= 0.05 -> Stationary (cannot reject H0)\n\n")

        f.write(f"Status: {comparison_kpss['status']}\n\n")

        f.write("Real Data:\n")
        f.write(f"  Total patients tested: {comparison_kpss['real']['total']}\n")
        f.write(f"  Stationary: {comparison_kpss['real']['stationary']} "
                f"({comparison_kpss['real']['stationary_percentage']:.1f}%)\n")
        f.write(f"  Non-stationary: {comparison_kpss['real']['non_stationary']} "
                f"({100 - comparison_kpss['real']['stationary_percentage']:.1f}%)\n\n")

        f.write("Synthetic Data:\n")
        f.write(f"  Total patients tested: {comparison_kpss['synthetic']['total']}\n")
        f.write(f"  Stationary: {comparison_kpss['synthetic']['stationary']} "
                f"({comparison_kpss['synthetic']['stationary_percentage']:.1f}%)\n")
        f.write(f"  Non-stationary: {comparison_kpss['synthetic']['non_stationary']} "
                f"({100 - comparison_kpss['synthetic']['stationary_percentage']:.1f}%)\n\n")

        f.write(f"Difference: {comparison_kpss['difference_pct']:.1f}%\n\n")
        f.write(f"Interpretation:\n{comparison_kpss['interpretation']}\n\n")

        # Interpretation guide
        f.write("=" * 80 + "\n")
        f.write("INTERPRETATION GUIDE: Understanding the Results\n")
        f.write("=" * 80 + "\n\n")

        f.write("Why do ADF and KPSS give seemingly opposite results?\n")
        f.write("-" * 80 + "\n")
        f.write("This is NOT a contradiction! The two tests have OPPOSITE null hypotheses:\n\n")

        f.write("  ADF Test:\n")
        f.write("    H0 (null hypothesis) = 'The series is NON-stationary'\n")
        f.write("    If p-value > 0.05 -> Cannot reject H0 -> Series is NON-stationary\n")
        f.write("    If p-value < 0.05 -> Reject H0 -> Series is STATIONARY\n\n")

        f.write("  KPSS Test:\n")
        f.write("    H0 (null hypothesis) = 'The series is STATIONARY'\n")
        f.write("    If p-value > 0.05 -> Cannot reject H0 -> Series is STATIONARY\n")
        f.write("    If p-value < 0.05 -> Reject H0 -> Series is NON-stationary\n\n")

        f.write("What this means in practice:\n")
        f.write("-" * 80 + "\n")
        f.write("Many patients fall in a 'GREY ZONE' where:\n")
        f.write("  - ADF cannot find strong enough evidence of stationarity\n")
        f.write("    -> Classified as 'non-stationary' (shows some progression)\n")
        f.write("  - KPSS cannot find strong enough evidence of non-stationarity\n")
        f.write("    -> Classified as 'stationary' (not dramatic changes)\n\n")

        f.write("This grey zone typically contains patients with:\n")
        f.write("  - Mild/moderate disease progression (not dramatic decline)\n")
        f.write("  - Some variability but with underlying trends\n")
        f.write("  - Realistic disease dynamics (not static, not extreme)\n\n")

        # Calculate key percentages
        adf_real_nonstat = 100 - comparison_adf['real']['stationary_percentage']
        adf_synth_nonstat = 100 - comparison_adf['synthetic']['stationary_percentage']
        kpss_real_stat = comparison_kpss['real']['stationary_percentage']
        kpss_synth_stat = comparison_kpss['synthetic']['stationary_percentage']

        f.write("=" * 80 + "\n")
        f.write("WHAT MATTERS FOR YOUR VALIDATION\n")
        f.write("=" * 80 + "\n\n")

        f.write("Focus on ADF test for disease progression:\n")
        f.write("-" * 80 + "\n")
        f.write(f"  Real patients showing progression:      {adf_real_nonstat:6.1f}%\n")
        f.write(f"  Synthetic patients showing progression: {adf_synth_nonstat:6.1f}%\n")
        f.write(f"  Difference:                             {comparison_adf['difference_pct']:6.1f}%\n\n")

        if comparison_adf['difference_pct'] < 20:
            f.write("INTERPRETATION:\n")
            f.write("  The synthetic data successfully captures the disease progression\n")
            f.write("  dynamics of the real data. The model is NOT generating 'frozen'\n")
            f.write("  patients - synthetic patients progress similarly to real ones.\n\n")
        else:
            f.write("INTERPRETATION:\n")
            f.write("  There is a significant difference in progression patterns.\n")
            f.write("  Review the model's temporal dynamics component.\n\n")

        # Overall conclusion
        f.write("=" * 80 + "\n")
        f.write("OVERALL CONCLUSION\n")
        f.write("=" * 80 + "\n")

        # Calculate key metrics for conclusion
        adf_real_nonstat = 100 - comparison_adf['real']['stationary_percentage']
        adf_synth_nonstat = 100 - comparison_adf['synthetic']['stationary_percentage']

        if comparison_adf['model_failed'] or comparison_kpss['model_failed']:
            f.write("STATUS: CRITICAL FAILURE\n\n")
            f.write("The model has failed to capture disease progression dynamics.\n")
            f.write("Synthetic patients are static (stationary) while real patients\n")
            f.write("show disease progression (non-stationary).\n\n")

            f.write("Details:\n")
            f.write(f"  - Real data: {adf_real_nonstat:.1f}% show progression\n")
            f.write(f"  - Synthetic: {adf_synth_nonstat:.1f}% show progression\n")
            f.write(f"  - The synthetic model is generating 'frozen' patients!\n\n")

            f.write("RECOMMENDATION: Retrain the model with focus on temporal dynamics.\n")
        elif comparison_adf['status'] == 'PASSED' and comparison_kpss['status'] == 'PASSED':
            f.write("STATUS: PASSED\n\n")
            f.write("The model successfully captures disease progression dynamics.\n")
            f.write("Synthetic data matches real data stationarity patterns.\n\n")

            f.write("What this means in simple terms:\n")
            f.write("-" * 80 + "\n")
            f.write(f"  - {adf_real_nonstat:.1f}% of REAL patients show disease progression over time\n")
            f.write(f"  - {adf_synth_nonstat:.1f}% of SYNTHETIC patients show disease progression\n")
            f.write(f"  - Difference of {comparison_adf['difference_pct']:.1f}% is ACCEPTABLE (threshold: < 20%)\n\n")

            f.write("  SUCCESS: The synthetic data is NOT generating 'frozen' patients.\n")
            f.write("           Synthetic patients progress realistically over time,\n")
            f.write("           capturing the disease dynamics of real patients.\n\n")

            f.write("RECOMMENDATION: Synthetic data is validated for temporal dynamics.\n")
            f.write("                Safe to use for longitudinal analyses.\n")
        else:
            f.write("STATUS: WARNING\n\n")
            f.write("There are moderate differences in stationarity patterns.\n")
            f.write(f"  - Real data: {adf_real_nonstat:.1f}% show progression\n")
            f.write(f"  - Synthetic: {adf_synth_nonstat:.1f}% show progression\n")
            f.write(f"  - Difference: {comparison_adf['difference_pct']:.1f}%\n\n")
            f.write("Review individual patient trajectories for more insight.\n\n")
            f.write("RECOMMENDATION: Investigate specific cases and consider model tuning.\n")

        f.write("=" * 80 + "\n")

    print(f"  Stationarity summary saved to: {save_path}")
