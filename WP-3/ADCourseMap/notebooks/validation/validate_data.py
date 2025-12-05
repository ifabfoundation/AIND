"""
Synthetic Data Validation Script
==================================

This script implements a comprehensive validation pipeline for synthetic data generation,
covering the three main dimensions: Fidelity, Utility, and Privacy.

Based on the guide: validatione_dati_sintentici.md
Excludes Leaspy-specific validations as requested.

Author: IFAB
Date: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import json
from typing import Dict, List, Optional, Tuple
import argparse
import sys
import warnings
warnings.filterwarnings('ignore')

# Import stationarity tests
try:
    from stationarity_tests import (
        test_stationarity_per_patient,
        test_stationarity_aggregated,
        compare_stationarity,
        plot_stationarity_comparison,
        save_stationarity_summary
    )
    STATIONARITY_AVAILABLE = True
except ImportError:
    STATIONARITY_AVAILABLE = False
    print("⚠️  Warning: stationarity_tests not found. Stationarity validation disabled.")

# Import ACF/PACF tests
try:
    from acf_pacf_tests import (
        compute_acf_pacf_per_patient,
        detect_ar_ma_order_per_patient,
        compare_acf_pacf,
        compare_ar_ma_orders,
        plot_acf_pacf_comparison,
        save_acf_pacf_summary
    )
    ACF_PACF_AVAILABLE = True
except ImportError:
    ACF_PACF_AVAILABLE = False
    print("⚠️  Warning: acf_pacf_tests not found. ACF/PACF validation disabled.")

# Import datalake client (optional - will warn if not available)
try:
    from dl_client import DatalakeClient
    DATALAKE_AVAILABLE = True
except ImportError:
    DATALAKE_AVAILABLE = False
    print("⚠️  Warning: dl_client not found. Datalake loading disabled.")


# ============================================================================
# DATALAKE HELPER FUNCTIONS
# ============================================================================

def load_from_datalake(file_code: str, level: str = 'cleaned_03') -> pd.DataFrame:
    """
    Load data from datalake using dl_client.

    Parameters
    ----------
    file_code : str
        File code to search for (e.g., 'ADNIMERGE' or 'ADNIMERGEsynthetic')
    level : str, default='cleaned_03'
        Data level in the datalake

    Returns
    -------
    pd.DataFrame
        Loaded dataset

    Raises
    ------
    ImportError
        If dl_client is not available
    RuntimeError
        If file not found or loading fails
    """
    if not DATALAKE_AVAILABLE:
        raise ImportError(
            "dl_client is not available. Please install it or use CSV loading instead."
        )

    print(f"  📡 Connecting to datalake...")

    # Initialize client
    client = DatalakeClient()

    # Query for file
    print(f"  🔍 Searching for file_code='{file_code}', level='{level}'...")
    search = client.query_files(
        query={
            'custom.level': level,
            'custom.file_code': file_code
        }
    )

    if not search or 'object_name' not in search:
        raise RuntimeError(
            f"File not found in datalake: file_code='{file_code}', level='{level}'"
        )

    # Download and extract
    print(f"  ⬇️  Downloading {search['object_name']}...")
    zip_files = client.download_file(
        search['object_name'],
        extract_zip=True
    )

    if not zip_files:
        raise RuntimeError("Downloaded file is empty or could not be extracted")

    # Get first file from zip
    file_name = list(zip_files.keys())[0]
    dataset = zip_files[file_name]

    print(f"  ✅ Loaded {file_name} from datalake ({dataset.shape})")

    return dataset


def load_real_and_synthetic_from_datalake(real_file_code: str,
                                          level: str = 'cleaned_03') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load both real and synthetic datasets from datalake.

    Assumes synthetic data has file_code = real_file_code + 'synthetic'
    (e.g., 'ADNIMERGE' → 'ADNIMERGEsynthetic')

    Parameters
    ----------
    real_file_code : str
        File code for real data (e.g., 'ADNIMERGE')
    level : str, default='cleaned_03'
        Data level in the datalake

    Returns
    -------
    tuple of (pd.DataFrame, pd.DataFrame)
        (real_data, synthetic_data)
    """
    print("\n" + "=" * 80)
    print("LOADING DATA FROM DATALAKE")
    print("=" * 80)

    # Load real data
    print("\n📊 Loading REAL data:")
    real_data = load_from_datalake(real_file_code, level)

    # Load synthetic data (convention: real_file_code + 'synthetic')
    synthetic_file_code = real_file_code + 'synthetic'
    print("\n🤖 Loading SYNTHETIC data:")
    synthetic_data = load_from_datalake(synthetic_file_code, level)

    print("\n" + "=" * 80)
    print(f"✅ Successfully loaded both datasets")
    print(f"   Real: {real_data.shape}")
    print(f"   Synthetic: {synthetic_data.shape}")
    print("=" * 80)

    return real_data, synthetic_data


# ============================================================================
# STEP 1: GOWER SIMILARITY CALCULATION
# ============================================================================

def gower_similarity(X: np.ndarray, Y: Optional[np.ndarray] = None,
                     categorical_features: Optional[List[int]] = None) -> np.ndarray:
    """
    Calculate Gower similarity matrix for mixed-type data.

    Gower similarity handles both numerical and categorical features:
    - Numerical: similarity = 1 - |difference| / range
    - Categorical: similarity = 1 if equal, 0 otherwise

    Parameters
    ----------
    X : np.ndarray, shape (n_samples_X, n_features)
        First dataset
    Y : np.ndarray, shape (n_samples_Y, n_features), optional
        Second dataset. If None, computes similarity within X
    categorical_features : List[int], optional
        Indices of categorical features

    Returns
    -------
    similarity_matrix : np.ndarray, shape (n_samples_X, n_samples_Y)
        Matrix of pairwise Gower similarities
    """
    if Y is None:
        Y = X

    n_features = X.shape[1]
    if categorical_features is None:
        categorical_features = []

    similarity_matrix = np.zeros((X.shape[0], Y.shape[0]))

    # Calculate similarity for each feature
    for i in range(n_features):
        if i in categorical_features:
            # Categorical: 1 if equal, 0 otherwise
            feature_sim = (X[:, i, np.newaxis] == Y[:, i]).astype(float)
        else:
            # Numeric: 1 - |diff|/range
            range_i = np.ptp(np.concatenate([X[:, i], Y[:, i]]))
            if range_i > 0:
                diff = np.abs(X[:, i, np.newaxis] - Y[:, i])
                feature_sim = 1 - (diff / range_i)
            else:
                # All values are the same
                feature_sim = np.ones((X.shape[0], Y.shape[0]))

        similarity_matrix += feature_sim

    # Average across features
    similarity_matrix /= n_features

    return similarity_matrix


# ============================================================================
# STEP 2: MAXIMUM SIMILARITY TEST
# ============================================================================

def maximum_similarity_test(real_data: pd.DataFrame,
                           synthetic_data: pd.DataFrame,
                           categorical_features: Optional[List[int]] = None) -> Dict:
    """
    Perform Maximum Similarity Test to validate synthetic data quality.

    The test compares:
    - MISS_real: Maximum Intra-Set Similarity for real data (nearest neighbor within real)
    - MISS_synth: Maximum Intra-Set Similarity for synthetic data
    - MCSS: Maximum Cross-Set Similarity (synthetic to nearest in real)

    Quality Score = mean(MCSS) / mean(MISS_real)
    - Q H 1.0: Perfect
    - Q < 1.0: Good (synthetic points are less similar to real than expected)
    - Q > 1.0: FAILED (privacy compromised, overfitting detected)

    Parameters
    ----------
    real_data : pd.DataFrame
        Real dataset
    synthetic_data : pd.DataFrame
        Synthetic dataset
    categorical_features : List[int], optional
        Indices of categorical features

    Returns
    -------
    results : dict
        Dictionary containing MISS, MCSS values and test results
    """
    print("  Computing Gower similarities...")

    # Convert to numpy arrays
    X_real = real_data.values if hasattr(real_data, 'values') else real_data
    X_synth = synthetic_data.values if hasattr(synthetic_data, 'values') else synthetic_data

    # 1. Maximum Intra-Set Similarity for Real
    print("  Computing MISS (Real)...")
    sim_real_real = gower_similarity(X_real, X_real, categorical_features)
    np.fill_diagonal(sim_real_real, -1)  # Exclude self-similarity
    miss_real = np.max(sim_real_real, axis=1)

    # 2. Maximum Intra-Set Similarity for Synthetic
    print("  Computing MISS (Synthetic)...")
    sim_synth_synth = gower_similarity(X_synth, X_synth, categorical_features)
    np.fill_diagonal(sim_synth_synth, -1)
    miss_synth = np.max(sim_synth_synth, axis=1)

    # 3. Maximum Cross-Set Similarity (Synthetic � Real)
    print("  Computing MCSS (Cross-set)...")
    sim_synth_real = gower_similarity(X_synth, X_real, categorical_features)
    mcss = np.max(sim_synth_real, axis=1)

    # 4. Quality Score
    quality_score = np.mean(mcss) / np.mean(miss_real) if np.mean(miss_real) > 0 else 0

    # 5. Statistical tests (Kolmogorov-Smirnov)
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


# ============================================================================
# STEP 3: VISUALIZATION FUNCTIONS
# ============================================================================

def plot_similarity_distributions(results: Dict, save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot Maximum Similarity Test results.

    Creates two subplots:
    1. Histogram overlay of MISS_real, MISS_synth, and MCSS distributions
    2. Box plot comparison

    Parameters
    ----------
    results : dict
        Results from maximum_similarity_test()
    save_path : str, optional
        If provided, saves figure to this path

    Returns
    -------
    fig : plt.Figure
        Matplotlib figure object
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram overlay
    axes[0].hist(results['miss_real'], bins=30, alpha=0.5,
                 label='MISS Real', density=True, color='blue')
    axes[0].hist(results['miss_synth'], bins=30, alpha=0.5,
                 label='MISS Synthetic', density=True, color='green')
    axes[0].hist(results['mcss'], bins=30, alpha=0.5,
                 label='MCSS', density=True, color='red')
    axes[0].axvline(results['mean_miss_real'], color='blue',
                    linestyle='--', label=f"� MISS Real: {results['mean_miss_real']:.4f}")
    axes[0].axvline(results['mean_mcss'], color='red',
                    linestyle='--', label=f"� MCSS: {results['mean_mcss']:.4f}")
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

    # Add quality score to title
    privacy_status = 'PASS' if results['passed_privacy'] else 'FAIL'
    fidelity_status = 'PASS' if results['passed_fidelity'] else 'FAIL'

    fig.suptitle(f"Quality Score: {results['quality_score']:.4f} | "
                 f"Privacy: {privacy_status} | "
                 f"Fidelity: {fidelity_status}",
                 fontsize=12, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Figure saved to: {save_path}")

    return fig


def plot_correlation_comparison(real_corr: pd.DataFrame,
                                synth_corr: pd.DataFrame,
                                save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot comparison of correlation matrices between real and synthetic data.

    Parameters
    ----------
    real_corr : pd.DataFrame
        Correlation matrix of real data
    synth_corr : pd.DataFrame
        Correlation matrix of synthetic data
    save_path : str, optional
        If provided, saves figure to this path

    Returns
    -------
    fig : plt.Figure
        Matplotlib figure object
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Real correlation matrix
    sns.heatmap(real_corr, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, vmin=-1, vmax=1, ax=axes[0], square=True)
    axes[0].set_title('Real Data Correlation Matrix')

    # Synthetic correlation matrix
    sns.heatmap(synth_corr, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, vmin=-1, vmax=1, ax=axes[1], square=True)
    axes[1].set_title('Synthetic Data Correlation Matrix')

    # Difference
    diff = np.abs(real_corr - synth_corr)
    sns.heatmap(diff, annot=True, fmt='.2f', cmap='Reds',
                vmin=0, vmax=1, ax=axes[2], square=True)
    axes[2].set_title('Absolute Difference')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Figure saved to: {save_path}")

    return fig


def plot_ks_test_comparison(real_values: np.ndarray,
                            synth_values: np.ndarray,
                            variable_name: str,
                            ks_stat: float,
                            p_value: float,
                            save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot detailed comparison for KS test including histograms and CDFs.

    Creates a 3-panel figure showing:
    1. Overlaid histograms with KDE
    2. Empirical CDFs
    3. Absolute difference between CDFs (what KS test measures)

    Parameters
    ----------
    real_values : np.ndarray
        Real data values
    synth_values : np.ndarray
        Synthetic data values
    variable_name : str
        Name of the variable being compared
    ks_stat : float
        KS test statistic
    p_value : float
        KS test p-value
    save_path : str, optional
        If provided, saves figure to this path

    Returns
    -------
    fig : plt.Figure
        Matplotlib figure object
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: Histograms with KDE
    axes[0].hist(real_values, bins=30, alpha=0.5, label='Real',
                 density=True, color='blue', edgecolor='black')
    axes[0].hist(synth_values, bins=30, alpha=0.5, label='Synthetic',
                 density=True, color='orange', edgecolor='black')

    # Add KDE curves
    from scipy.stats import gaussian_kde
    if len(real_values) > 1:
        kde_real = gaussian_kde(real_values)
        x_range = np.linspace(min(real_values.min(), synth_values.min()),
                             max(real_values.max(), synth_values.max()), 200)
        axes[0].plot(x_range, kde_real(x_range), 'b-', linewidth=2, label='Real KDE')

    if len(synth_values) > 1:
        kde_synth = gaussian_kde(synth_values)
        x_range = np.linspace(min(real_values.min(), synth_values.min()),
                             max(real_values.max(), synth_values.max()), 200)
        axes[0].plot(x_range, kde_synth(x_range), 'orange', linewidth=2,
                    linestyle='--', label='Synthetic KDE')

    axes[0].set_xlabel('Value')
    axes[0].set_ylabel('Density')
    axes[0].set_title(f'Histograms + KDE')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Panel 2: Empirical CDFs
    real_sorted = np.sort(real_values)
    synth_sorted = np.sort(synth_values)

    real_cdf = np.arange(1, len(real_sorted) + 1) / len(real_sorted)
    synth_cdf = np.arange(1, len(synth_sorted) + 1) / len(synth_sorted)

    axes[1].plot(real_sorted, real_cdf, 'b-', linewidth=2, label='Real CDF')
    axes[1].plot(synth_sorted, synth_cdf, 'orange', linewidth=2,
                linestyle='--', label='Synthetic CDF')

    # Mark the maximum difference (KS statistic)
    # Find where max difference occurs
    from scipy import interpolate
    f_real = interpolate.interp1d(real_sorted, real_cdf, bounds_error=False,
                                  fill_value=(0, 1))
    f_synth = interpolate.interp1d(synth_sorted, synth_cdf, bounds_error=False,
                                   fill_value=(0, 1))

    # Evaluate on combined grid
    all_values = np.sort(np.concatenate([real_sorted, synth_sorted]))
    cdf_diff = np.abs(f_real(all_values) - f_synth(all_values))
    max_diff_idx = np.argmax(cdf_diff)
    max_diff_x = all_values[max_diff_idx]

    axes[1].axvline(max_diff_x, color='red', linestyle=':', linewidth=2,
                   label=f'Max diff at x={max_diff_x:.2f}')
    axes[1].plot([max_diff_x, max_diff_x],
                [f_real(max_diff_x), f_synth(max_diff_x)],
                'r-', linewidth=3, label=f'KS stat = {ks_stat:.4f}')

    axes[1].set_xlabel('Value')
    axes[1].set_ylabel('Cumulative Probability')
    axes[1].set_title(f'Empirical CDFs')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # Panel 3: Absolute difference between CDFs
    axes[2].fill_between(all_values, 0, cdf_diff, alpha=0.5, color='red')
    axes[2].plot(all_values, cdf_diff, 'r-', linewidth=2)
    axes[2].axhline(ks_stat, color='darkred', linestyle='--', linewidth=2,
                   label=f'KS stat = {ks_stat:.4f}')
    axes[2].set_xlabel('Value')
    axes[2].set_ylabel('|CDF_real - CDF_synth|')
    axes[2].set_title(f'CDF Difference (KS Test)')
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    # Overall title with test results
    passed = "✓ PASS" if p_value > 0.05 else "✗ FAIL"
    fig.suptitle(f'{variable_name} - KS Test: statistic={ks_stat:.4f}, p-value={p_value:.4f} {passed}',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


# ============================================================================
# STEP 4: COMPREHENSIVE VALIDATION CLASS
# ============================================================================

class SyntheticDataValidator:
    """
    Comprehensive validation class for synthetic data.

    Performs validation across three dimensions:
    1. Fidelity: Statistical similarity to real data
    2. Utility: Practical usefulness for predictive tasks
    3. Privacy: No information leakage from real to synthetic

    Usage
    -----
    validator = SyntheticDataValidator(real_data, synthetic_data)
    results = validator.run_full_validation()
    validator.save_report('validation_report.json')
    """

    def __init__(self,
                 real_data: pd.DataFrame,
                 synthetic_data: pd.DataFrame,
                 categorical_features: Optional[List[str]] = None,
                 id_var: Optional[str] = None,
                 target_var: Optional[str] = None,
                 time_var: Optional[str] = None,
                 auto_detect_categorical: bool = True,
                 generate_plots: bool = True,
                 output_dir: str = 'validation_results',
                 utility_synthetic_data: Optional[pd.DataFrame] = None,
                 utility_real_data: Optional[pd.DataFrame] = None):
        """
        Initialize validator.

        Parameters
        ----------
        real_data : pd.DataFrame
            Real dataset
        synthetic_data : pd.DataFrame
            Synthetic dataset
        categorical_features : List[str], optional
            Names of categorical feature columns. If None and auto_detect_categorical=True,
            will automatically detect columns with dtype 'object' or 'category'
        id_var : str, optional
            Name of ID column (will be excluded from validation)
        target_var : str, optional
            Name of target variable for utility validation
        time_var : str, optional
            Name of time variable for longitudinal/stationarity validation
        auto_detect_categorical : bool, default=True
            If True and categorical_features is None, automatically detects categorical columns
        generate_plots : bool, default=True
            If True, generates visualization plots for KS tests and other validations
        output_dir : str, default='validation_results'
            Directory where plots will be saved (only used if generate_plots=True)
        utility_synthetic_data : pd.DataFrame, optional
            Separate synthetic dataset to use ONLY for utility validation (TSTR test).
            If None, uses the main synthetic_data for utility validation.
            This allows using a different synthetic dataset for utility testing
            (e.g., one generated with a different method or parameters).
        utility_real_data : pd.DataFrame, optional
            Separate real dataset to use ONLY for utility validation (TSTR test).
            If None, uses the main real_data for utility validation.
            This allows using a different real dataset for utility testing
            (e.g., a held-out test set or data from a different source).
        """
        self.real_data = real_data.copy()
        self.synthetic_data = synthetic_data.copy()
        self.id_var = id_var
        self.target_var = target_var
        self.time_var = time_var
        self.results = {}
        self.generate_plots = generate_plots
        self.output_dir = output_dir

        # Store separate utility synthetic data if provided
        if utility_synthetic_data is not None:
            self.utility_synthetic_data = utility_synthetic_data.copy()
            print(f"  📊 Separate utility synthetic dataset provided: {utility_synthetic_data.shape}")
        else:
            self.utility_synthetic_data = None

        # Store separate utility real data if provided
        if utility_real_data is not None:
            self.utility_real_data = utility_real_data.copy()
            print(f"  📊 Separate utility real dataset provided: {utility_real_data.shape}")
        else:
            self.utility_real_data = None

        # Create output directory structure if plots are enabled
        if self.generate_plots:
            import os
            os.makedirs(output_dir, exist_ok=True)

            # Create subdirectories for organized output
            self.subdirs = {
                'acf_pacf': os.path.join(output_dir, 'acf_pacf'),
                'stationarity': os.path.join(output_dir, 'stationarity'),
                'ks_tests': os.path.join(output_dir, 'ks_tests'),
                'correlation': os.path.join(output_dir, 'correlation'),
                'similarity': os.path.join(output_dir, 'similarity'),
                'utility': os.path.join(output_dir, 'utility')
            }

            for subdir_name, subdir_path in self.subdirs.items():
                os.makedirs(subdir_path, exist_ok=True)

            print(f"  📁 Plot output directory: {output_dir}")
            print(f"     ├── acf_pacf/")
            print(f"     ├── stationarity/")
            print(f"     ├── ks_tests/")
            print(f"     ├── correlation/")
            print(f"     ├── similarity/")
            print(f"     └── utility/")

        # Identify feature columns
        exclude_cols = [c for c in [id_var] if c is not None]
        self.feature_cols = [c for c in real_data.columns if c not in exclude_cols]

        # Auto-detect or validate categorical features
        if categorical_features is None and auto_detect_categorical:
            # Auto-detect: object, category, or bool dtypes
            self.categorical_features = self._auto_detect_categorical()
            print(f"  🔍 Auto-detected {len(self.categorical_features)} categorical features:")
            for cat_col in self.categorical_features:
                print(f"     • {cat_col}")
        elif categorical_features is not None:
            # Validate provided categorical features
            self.categorical_features = self._validate_categorical_features(categorical_features)
        else:
            self.categorical_features = []

        # Get categorical feature indices
        self.categorical_indices = [
            self.feature_cols.index(col) for col in self.categorical_features
            if col in self.feature_cols
        ]

        # Get numeric feature columns
        self._identify_numeric_features()

        print(f"  📊 Feature summary:")
        print(f"     Total features: {len(self.feature_cols)}")
        print(f"     Categorical: {len(self.categorical_features)}")
        print(f"     Numeric: {len(self.numeric_features)}")

    def _auto_detect_categorical(self) -> List[str]:
        """
        Auto-detect categorical features based on dtype.

        Returns
        -------
        List[str]
            List of categorical feature names
        """
        categorical = []

        for col in self.feature_cols:
            # Check both datasets for consistency
            real_dtype = self.real_data[col].dtype
            synth_dtype = self.synthetic_data[col].dtype

            # Consider categorical if: object, category, or bool
            is_categorical = (
                real_dtype == 'object' or
                synth_dtype == 'object' or
                isinstance(real_dtype, pd.CategoricalDtype) or
                isinstance(synth_dtype, pd.CategoricalDtype) or
                real_dtype == 'bool' or
                synth_dtype == 'bool'
            )

            if is_categorical:
                categorical.append(col)

        return categorical

    def _validate_categorical_features(self, categorical_features: List[str]) -> List[str]:
        """
        Validate user-provided categorical features and warn about potential issues.

        Parameters
        ----------
        categorical_features : List[str]
            User-provided list of categorical feature names

        Returns
        -------
        List[str]
            Validated list of categorical features
        """
        validated = []

        for col in categorical_features:
            if col not in self.feature_cols:
                print(f"  ⚠️  Warning: '{col}' not found in feature columns, skipping")
                continue

            # Check if dtype is numeric (potential issue)
            real_dtype = self.real_data[col].dtype
            synth_dtype = self.synthetic_data[col].dtype

            if np.issubdtype(real_dtype, np.number) and np.issubdtype(synth_dtype, np.number):
                n_unique_real = self.real_data[col].nunique()
                n_unique_synth = self.synthetic_data[col].nunique()

                if n_unique_real > 20 or n_unique_synth > 20:
                    print(f"  ⚠️  Warning: '{col}' has numeric dtype with {n_unique_real}/{n_unique_synth} unique values")
                    print(f"     Consider if this should be numeric instead")

            validated.append(col)

        return validated

    def _identify_numeric_features(self):
        """
        Identify numeric features (excluding categorical ones).
        """
        # Get columns that are numeric in BOTH datasets
        real_numeric = set(
            self.real_data[self.feature_cols].select_dtypes(include=[np.number]).columns
        )
        synth_numeric = set(
            self.synthetic_data[self.feature_cols].select_dtypes(include=[np.number]).columns
        )

        # Common numeric columns, excluding categorical features
        self.numeric_features = [
            col for col in (real_numeric & synth_numeric)
            if col not in self.categorical_features
        ]

    def _impute_missing_values(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Intelligently impute missing values in numeric columns.

        Uses median imputation for numeric features (more robust than mean to outliers).
        Does NOT modify the original DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with potential missing values
        columns : List[str]
            List of column names to impute

        Returns
        -------
        pd.DataFrame
            DataFrame with imputed values (copy of original)
        """
        df_imputed = df[columns].copy()

        # Only impute if there are missing values
        if df_imputed.isnull().sum().sum() > 0:
            # Use SimpleImputer with median strategy (robust to outliers)
            imputer = SimpleImputer(strategy='median')
            df_imputed[columns] = imputer.fit_transform(df_imputed[columns])

        return df_imputed

    def run_full_validation(self) -> Dict:
        """
        Run complete validation pipeline.

        Returns
        -------
        results : dict
            Dictionary containing all validation results
        """
        print("=" * 80)
        print("COMPREHENSIVE SYNTHETIC DATA VALIDATION")
        print("=" * 80)

        # DIMENSION 1: FIDELITY
        print("\n" + "=" * 80)
        print("1. FIDELITY VALIDATION")
        print("=" * 80)
        self._validate_fidelity()

        # DIMENSION 2: UTILITY
        print("\n" + "=" * 80)
        print("2. UTILITY VALIDATION")
        print("=" * 80)
        self._validate_utility()

        # DIMENSION 3: PRIVACY
        print("\n" + "=" * 80)
        print("3. PRIVACY VALIDATION")
        print("=" * 80)
        self._validate_privacy()

        # MAXIMUM SIMILARITY TEST
        print("\n" + "=" * 80)
        print("4. MAXIMUM SIMILARITY TEST")
        print("=" * 80)
        self._run_maximum_similarity_test()

        # STATIONARITY VALIDATION
        print("\n" + "=" * 80)
        print("5. STATIONARITY VALIDATION (Disease Progression)")
        print("=" * 80)
        self._validate_stationarity()

        # ACF/PACF VALIDATION
        print("\n" + "=" * 80)
        print("6. ACF/PACF VALIDATION (Temporal Autocorrelation)")
        print("=" * 80)
        self._validate_acf_pacf()

        # SUMMARY
        self._print_summary()

        return self.results

    def run_fidelity_only_validation(self) -> Dict:
        """
        Run ONLY fidelity validation (KS test and correlation).

        This is a simplified validation mode that only checks statistical
        fidelity without running utility or privacy tests.

        Returns
        -------
        results : dict
            Dictionary containing fidelity validation results
        """
        print("=" * 80)
        print("FIDELITY-ONLY SYNTHETIC DATA VALIDATION")
        print("=" * 80)

        # DIMENSION 1: FIDELITY (KS + Correlation)
        print("\n" + "=" * 80)
        print("FIDELITY VALIDATION (KS Test + Correlation)")
        print("=" * 80)
        self._validate_fidelity()

        # SUMMARY
        self._print_fidelity_summary()

        return self.results

    def run_utility_only_validation(self) -> Dict:
        """
        Run ONLY utility validation (TSTR test).

        This is a simplified validation mode that only checks practical utility
        via the Train on Synthetic, Test on Real (TSTR) test.

        Returns
        -------
        results : dict
            Dictionary containing utility validation results
        """
        print("=" * 80)
        print("UTILITY-ONLY SYNTHETIC DATA VALIDATION")
        print("=" * 80)

        # DIMENSION 2: UTILITY (TSTR)
        print("\n" + "=" * 80)
        print("UTILITY VALIDATION (TSTR Test)")
        print("=" * 80)
        self._validate_utility()

        # SUMMARY
        self._print_utility_summary()

        return self.results

    def _print_utility_summary(self):
        """Print summary for utility-only validation and save to file."""
        import os
        from datetime import datetime

        print("\n" + "=" * 80)
        print("UTILITY VALIDATION SUMMARY")
        print("=" * 80)

        # Build summary lines for both printing and saving
        summary_lines = []
        summary_lines.append("=" * 80)
        summary_lines.append("UTILITY VALIDATION SUMMARY (TSTR Test)")
        summary_lines.append("=" * 80)
        summary_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        summary_lines.append("")

        if 'tstr' in self.results:
            tstr = self.results['tstr']
            if tstr.get('skipped'):
                reason = tstr.get('reason', 'Unknown')
                print(f"\n⚠️  TSTR Test: SKIPPED ({reason})")
                summary_lines.append(f"TSTR Test: SKIPPED ({reason})")
            else:
                status = "✅ PASS" if tstr.get('passed') else "❌ FAIL"
                status_text = "PASS" if tstr.get('passed') else "FAIL"
                print(f"\n{status} TSTR (Train on Synthetic, Test on Real)")
                print(f"   TSTR Accuracy: {tstr.get('tstr_accuracy', 0):.4f}")
                print(f"   TSTR F1-Score: {tstr.get('tstr_f1', 0):.4f}")
                print(f"   TRTR Accuracy: {tstr.get('trtr_accuracy', 0):.4f} (baseline)")
                print(f"   TRTR F1-Score: {tstr.get('trtr_f1', 0):.4f} (baseline)")
                print(f"   Accuracy Degradation: {tstr.get('accuracy_degradation', 0):.4f}")
                print(f"   F1 Degradation: {tstr.get('f1_degradation', 0):.4f}")

                summary_lines.append(f"Result: {status_text}")
                summary_lines.append(f"Pass Criteria: Degradation < 5%")
                summary_lines.append("")
                summary_lines.append("--- Performance Metrics ---")
                summary_lines.append(f"TSTR (Train on Synthetic, Test on Real):")
                summary_lines.append(f"  Accuracy: {tstr.get('tstr_accuracy', 0):.4f}")
                summary_lines.append(f"  F1-Score: {tstr.get('tstr_f1', 0):.4f}")
                summary_lines.append("")
                summary_lines.append(f"TRTR (Train on Real, Test on Real) - Baseline:")
                summary_lines.append(f"  Accuracy: {tstr.get('trtr_accuracy', 0):.4f}")
                summary_lines.append(f"  F1-Score: {tstr.get('trtr_f1', 0):.4f}")
                summary_lines.append("")
                summary_lines.append("--- Degradation ---")
                summary_lines.append(f"Accuracy Degradation: {tstr.get('accuracy_degradation', 0):.4f} ({tstr.get('accuracy_degradation', 0)*100:.2f}%)")
                summary_lines.append(f"F1 Degradation: {tstr.get('f1_degradation', 0):.4f} ({tstr.get('f1_degradation', 0)*100:.2f}%)")

                if tstr.get('split_by_patient'):
                    print(f"\n   Split by patient:")
                    print(f"     Train: {tstr.get('n_train_patients')} patients ({tstr.get('n_train_visits')} visits)")
                    print(f"     Test: {tstr.get('n_test_patients')} patients ({tstr.get('n_test_visits')} visits)")

                    summary_lines.append("")
                    summary_lines.append("--- Data Split (by patient) ---")
                    summary_lines.append(f"Train: {tstr.get('n_train_patients')} patients ({tstr.get('n_train_visits')} visits)")
                    summary_lines.append(f"Test: {tstr.get('n_test_patients')} patients ({tstr.get('n_test_visits')} visits)")

                if tstr.get('used_separate_synthetic'):
                    print(f"\n   Used separate synthetic dataset: {tstr.get('synthetic_samples_used')} samples")
                    summary_lines.append("")
                    summary_lines.append(f"Separate synthetic dataset: {tstr.get('synthetic_samples_used')} samples")
                if tstr.get('used_separate_real'):
                    print(f"   Used separate real dataset: {tstr.get('real_samples_used')} samples")
                    summary_lines.append(f"Separate real dataset: {tstr.get('real_samples_used')} samples")

                summary_lines.append("")
                summary_lines.append(f"Features used: {tstr.get('features_used', 'N/A')}")
                summary_lines.append(f"Target variable: {self.target_var}")

        summary_lines.append("")
        summary_lines.append("=" * 80)

        # Save to file if generate_plots is enabled and subdirs exist
        if self.generate_plots and hasattr(self, 'subdirs') and 'utility' in self.subdirs:
            summary_file = os.path.join(self.subdirs['utility'], 'tstr_summary.txt')
            with open(summary_file, 'w') as f:
                f.write('\n'.join(summary_lines))
            print(f"\n   📄 Summary saved to: {summary_file}")

        print("\n" + "=" * 80)

    def _validate_fidelity(self):
        """
        Validate statistical fidelity (Dimension 1).

        Tests:
        - Univariate distributions (KS test for numeric, Chi-square for categorical)
        - Correlation structure (MAD and meta-correlation)
        """
        # ========== 1.1 NUMERIC: KS TEST ==========
        print("\n--- 1.1 Univariate Distributions (Numeric - KS Test) ---")

        univariate_results = {}

        # Test numeric features with KS test
        for col in self.numeric_features:
            real_values = self.real_data[col].dropna()
            synth_values = self.synthetic_data[col].dropna()

            if len(real_values) > 0 and len(synth_values) > 0:
                # Kolmogorov-Smirnov test
                # H0: Real and Synthetic come from the same distribution
                # If p > 0.05, we cannot reject H0 (distributions are similar)
                ks_stat, p_value = stats.ks_2samp(real_values, synth_values)
                passed = p_value > 0.05

                univariate_results[col] = {
                    'type': 'numeric',
                    'test': 'ks',
                    'ks_statistic': float(ks_stat),
                    'p_value': float(p_value),
                    'passed': passed
                }

                status = "✓ PASS" if passed else "✗ FAIL"
                print(f"  {col:30s}: KS={ks_stat:.4f}, p={p_value:.4f} {status}")

                # Generate plot if enabled
                if self.generate_plots:
                    import os
                    plot_path = os.path.join(self.subdirs['ks_tests'], f'ks_test_{col}.png')
                    fig = plot_ks_test_comparison(
                        real_values.values,
                        synth_values.values,
                        col,
                        ks_stat,
                        p_value,
                        save_path=plot_path
                    )
                    plt.close(fig)
            else:
                print(f"  {col:30s}: SKIPPED (insufficient data)")

        # Report plots saved
        if self.generate_plots and len(self.numeric_features) > 0:
            import os
            print(f"\n  📊 KS test plots saved to: {self.subdirs['ks_tests']}/")

        # ========== 1.2 CATEGORICAL: CHI-SQUARE TEST ==========
        if self.categorical_features:
            print("\n--- 1.2 Univariate Distributions (Categorical - Chi-square Test) ---")

            for col in self.categorical_features:
                real_values = self.real_data[col].dropna()
                synth_values = self.synthetic_data[col].dropna()

                if len(real_values) > 0 and len(synth_values) > 0:
                    # Get frequency distributions
                    real_counts = real_values.value_counts()
                    synth_counts = synth_values.value_counts()

                    # Align categories (ensure same categories in both)
                    all_categories = sorted(set(real_counts.index) | set(synth_counts.index))
                    real_freq = np.array([real_counts.get(cat, 0) for cat in all_categories])
                    synth_freq = np.array([synth_counts.get(cat, 0) for cat in all_categories])

                    # Normalize expected frequencies based on real distribution
                    # This accounts for different sample sizes
                    real_normalized = real_freq / real_freq.sum() * synth_freq.sum()

                    # Chi-square test - avoid zero expected frequencies
                    if np.all(real_normalized > 0) and np.all(synth_freq >= 0):
                        try:
                            chi2_stat, p_value = stats.chisquare(synth_freq, f_exp=real_normalized)
                            passed = p_value > 0.05

                            univariate_results[col] = {
                                'type': 'categorical',
                                'test': 'chi2',
                                'chi2_statistic': float(chi2_stat),
                                'p_value': float(p_value),
                                'n_categories': len(all_categories),
                                'passed': passed
                            }

                            status = "✓ PASS" if passed else "✗ FAIL"
                            print(f"  {col:30s}: χ²={chi2_stat:.4f}, p={p_value:.4f} ({len(all_categories)} cat) {status}")
                        except (ValueError, ZeroDivisionError) as e:
                            print(f"  {col:30s}: SKIPPED (chi-square error: {e})")
                            univariate_results[col] = {
                                'type': 'categorical',
                                'test': 'chi2',
                                'error': str(e),
                                'passed': False
                            }
                    else:
                        print(f"  {col:30s}: SKIPPED (zero expected frequencies)")
                        univariate_results[col] = {
                            'type': 'categorical',
                            'test': 'chi2',
                            'error': 'Zero expected frequencies',
                            'passed': False
                        }
                else:
                    print(f"  {col:30s}: SKIPPED (insufficient data)")

        self.results['univariate'] = univariate_results

        # ========== 1.3 CORRELATION STRUCTURE ==========
        print("\n--- 1.3 Correlation Structure (Numeric Only) ---")

        if len(self.numeric_features) >= 2:
            # Use self.numeric_features instead of recalculating
            real_corr = self.real_data[self.numeric_features].corr()
            synth_corr = self.synthetic_data[self.numeric_features].corr()

            # Mean Absolute Difference
            corr_diff = np.abs(real_corr - synth_corr)
            mad = corr_diff.mean().mean()

            # Meta-correlation (correlation of correlations)
            real_flat = real_corr.values[np.triu_indices_from(real_corr.values, k=1)]
            synth_flat = synth_corr.values[np.triu_indices_from(synth_corr.values, k=1)]

            if len(real_flat) > 0:  # Ensure we have correlations to compare
                meta_corr = np.corrcoef(real_flat, synth_flat)[0, 1]
            else:
                meta_corr = 1.0  # Perfect correlation if only 2 features

            # Pass criteria: MAD < 0.15 AND meta_corr > 0.8
            passed = (mad < 0.15) and (meta_corr > 0.8)

            self.results['correlations'] = {
                'mean_abs_diff': float(mad),
                'meta_correlation': float(meta_corr),
                'real_corr_matrix': real_corr.to_dict(),
                'synth_corr_matrix': synth_corr.to_dict(),
                'passed': passed
            }

            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"  Mean Absolute Difference: {mad:.4f} (threshold: < 0.15)")
            print(f"  Meta-correlation: {meta_corr:.4f} (threshold: > 0.80)")
            print(f"  Result: {status}")
        else:
            print(f"  ⚠️  Skipped (need at least 2 numeric features, found {len(self.numeric_features)})")
            self.results['correlations'] = {
                'skipped': True,
                'reason': f'Insufficient numeric features ({len(self.numeric_features)} < 2)',
                'passed': True  # Don't fail if not enough features
            }

    def _validate_utility(self):
        """
        Validate practical utility (Dimension 2).

        Test: TSTR (Train on Synthetic, Test on Real)
        - Train model on synthetic data
        - Test on real data
        - Compare with TRTR baseline (Train on Real, Test on Real)
        - Success: degradation < 5%

        Note: If utility_synthetic_data was provided at initialization,
        it will be used instead of the main synthetic_data for this test.
        Similarly, if utility_real_data was provided, it will be used
        instead of the main real_data for this test.
        """
        print("\n--- 2.1 TSTR (Train on Synthetic, Test on Real) ---")

        # Determine which synthetic dataset to use for utility validation
        if self.utility_synthetic_data is not None:
            synth_data_for_utility = self.utility_synthetic_data
            print("  📊 Using SEPARATE synthetic dataset for utility validation")
        else:
            synth_data_for_utility = self.synthetic_data
            print("  📊 Using main synthetic dataset for utility validation")

        # Determine which real dataset to use for utility validation
        if self.utility_real_data is not None:
            real_data_for_utility = self.utility_real_data
            print("  📊 Using SEPARATE real dataset for utility validation")
        else:
            real_data_for_utility = self.real_data
            print("  📊 Using main real dataset for utility validation")

        if self.target_var is None:
            print("  ⚠ No target variable specified, skipping TSTR")
            self.results['tstr'] = {'skipped': True, 'reason': 'No target variable specified'}
            return

        if self.target_var not in real_data_for_utility.columns:
            print(f"  ⚠ Target variable '{self.target_var}' not found in real data, skipping TSTR")
            print(f"     Available columns: {list(real_data_for_utility.columns)[:10]}...")
            self.results['tstr'] = {'skipped': True, 'reason': f"Target '{self.target_var}' not in real data"}
            return

        # Verify target variable exists in synthetic data too
        if self.target_var not in synth_data_for_utility.columns:
            print(f"  ⚠ Target variable '{self.target_var}' not found in synthetic data, skipping TSTR")
            self.results['tstr'] = {'skipped': True, 'reason': 'Target not in synthetic data'}
            return

        # Prepare features and target - use utility real data for feature detection
        feature_cols_numeric = [c for c in self.feature_cols
                               if c != self.target_var and
                               c in real_data_for_utility.select_dtypes(include=[np.number]).columns]

        # Verify features exist in synthetic data
        synth_numeric_cols = synth_data_for_utility.select_dtypes(include=[np.number]).columns
        feature_cols_common = [c for c in feature_cols_numeric if c in synth_numeric_cols]

        if len(feature_cols_common) < len(feature_cols_numeric):
            missing = set(feature_cols_numeric) - set(feature_cols_common)
            print(f"  ⚠ Some features missing in synthetic data: {missing}")
            feature_cols_numeric = feature_cols_common

        if len(feature_cols_numeric) == 0:
            print("  ⚠ No common numeric features found, skipping TSTR")
            self.results['tstr'] = {'skipped': True, 'reason': 'No common features'}
            return

        # Use intelligent imputation (median for numeric)
        X_real = self._impute_missing_values(real_data_for_utility, feature_cols_numeric)
        y_real = real_data_for_utility[self.target_var]
        X_synth = self._impute_missing_values(synth_data_for_utility, feature_cols_numeric)
        y_synth = synth_data_for_utility[self.target_var]

        # Check if classification or regression
        is_classification = len(y_real.unique()) < 20

        if not is_classification:
            print("  ⚠ Target appears to be continuous, skipping TSTR (implement regression if needed)")
            self.results['tstr'] = {'skipped': True}
            return

        # Split real data - by patient if ID column is available
        if self.id_var is not None and self.id_var in real_data_for_utility.columns:
            # Split by patient (stratified by number of visits per patient)
            print("  Splitting data by patient ID (stratified by visit count)...")

            np.random.seed(42)
            train_percentage = 0.7  # 70% train, 30% test

            # Count visits per patient
            visits_count = real_data_for_utility.groupby(self.id_var).size()

            # Group patients by number of visits (stratification)
            patients_per_visit_count = {}
            for patient_id, count in visits_count.items():
                if count not in patients_per_visit_count:
                    patients_per_visit_count[count] = []
                patients_per_visit_count[count].append(patient_id)

            # Select train patients from each visit-count group
            train_ids = []
            for visit_count in patients_per_visit_count:
                patient_list = patients_per_visit_count[visit_count]
                n_train = max(1, int(train_percentage * len(patient_list)))
                selected = np.random.choice(patient_list, size=n_train, replace=False).tolist()
                train_ids.extend(selected)

            # Create train/test masks based on patient IDs
            train_mask = real_data_for_utility[self.id_var].isin(train_ids)
            test_mask = ~train_mask

            # Get positional indices for train/test split
            # X_real is a DataFrame, so we use iloc for positional indexing
            train_indices = real_data_for_utility.index[train_mask].tolist()
            test_indices = real_data_for_utility.index[test_mask].tolist()

            # Map to positions in the arrays
            all_indices = real_data_for_utility.index.tolist()
            train_positions = [all_indices.index(idx) for idx in train_indices]
            test_positions = [all_indices.index(idx) for idx in test_indices]

            X_train_real = X_real.iloc[train_positions]
            X_test_real = X_real.iloc[test_positions]
            y_train_real = y_real.iloc[train_positions]
            y_test_real = y_real.iloc[test_positions]

            n_train_patients = len(train_ids)
            n_test_patients = len(set(real_data_for_utility[self.id_var]) - set(train_ids))
            split_by_patient = True
            print(f"  Split: {n_train_patients} train patients ({len(X_train_real)} visits), "
                  f"{n_test_patients} test patients ({len(X_test_real)} visits)")
        else:
            # Fallback: random split on visits (original behavior)
            print("  No ID column available, using random split on visits...")
            split_by_patient = False
            n_train_patients = None
            n_test_patients = None
            try:
                X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(
                    X_real, y_real, test_size=0.3, random_state=42, stratify=y_real
                )
            except ValueError:
                # If stratification fails, do regular split
                X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(
                    X_real, y_real, test_size=0.3, random_state=42
                )

        # TSTR: Train on Synthetic, Test on Real
        print("  Training model on synthetic data...")
        model_tstr = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model_tstr.fit(X_synth, y_synth)
        y_pred_tstr = model_tstr.predict(X_test_real)

        acc_tstr = accuracy_score(y_test_real, y_pred_tstr)
        f1_tstr = f1_score(y_test_real, y_pred_tstr, average='macro')

        # TRTR: Train on Real, Test on Real (baseline)
        print("  Training model on real data (baseline)...")
        model_trtr = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model_trtr.fit(X_train_real, y_train_real)
        y_pred_trtr = model_trtr.predict(X_test_real)

        acc_trtr = accuracy_score(y_test_real, y_pred_trtr)
        f1_trtr = f1_score(y_test_real, y_pred_trtr, average='macro')

        # Calculate degradation
        acc_deg = acc_trtr - acc_tstr
        f1_deg = f1_trtr - f1_tstr

        # Pass criteria: degradation < 5%
        passed = (acc_deg < 0.05) and (f1_deg < 0.05)

        self.results['tstr'] = {
            'tstr_accuracy': float(acc_tstr),
            'tstr_f1': float(f1_tstr),
            'trtr_accuracy': float(acc_trtr),
            'trtr_f1': float(f1_trtr),
            'accuracy_degradation': float(acc_deg),
            'f1_degradation': float(f1_deg),
            'passed': passed,
            'used_separate_synthetic': self.utility_synthetic_data is not None,
            'used_separate_real': self.utility_real_data is not None,
            'synthetic_samples_used': len(synth_data_for_utility),
            'real_samples_used': len(real_data_for_utility),
            'features_used': len(feature_cols_numeric),
            'split_by_patient': split_by_patient,
            'n_train_patients': n_train_patients,
            'n_test_patients': n_test_patients,
            'n_train_visits': len(X_train_real),
            'n_test_visits': len(X_test_real)
        }

        status = " PASS" if passed else " FAIL"
        print(f"  TSTR - Accuracy: {acc_tstr:.4f} | F1: {f1_tstr:.4f}")
        print(f"  TRTR - Accuracy: {acc_trtr:.4f} | F1: {f1_trtr:.4f}")
        print(f"  Degradation - Acc: {acc_deg:.4f} | F1: {f1_deg:.4f}")
        print(f"  Result: {status}")

    def _validate_privacy(self):
        """
        Validate privacy protection (Dimension 3).

        Tests:
        1. IMS (Identical Match Share): No exact duplicates
        2. DCR (Distance to Closest Record): Synthetic not too close to real
        """
        # TEST 1: Identical Match Share
        print("\n--- 3.1 Identical Match Share (IMS) ---")

        # Prepare data for comparison (exclude ID column)
        compare_cols = [c for c in self.feature_cols if c != self.id_var]

        real_tuples = set(map(tuple, self.real_data[compare_cols].values))
        synth_tuples = set(map(tuple, self.synthetic_data[compare_cols].values))

        exact_matches = real_tuples & synth_tuples
        ims = len(exact_matches) / len(synth_tuples) if len(synth_tuples) > 0 else 0

        passed_ims = (ims == 0)

        self.results['ims'] = {
            'identical_match_share': float(ims),
            'num_exact_matches': len(exact_matches),
            'passed': passed_ims
        }

        status = " PASS" if passed_ims else " FAIL"
        print(f"  Exact matches found: {len(exact_matches)}")
        print(f"  IMS: {ims:.6f} (target: 0.00)")
        print(f"  Result: {status}")

        # TEST 2: Distance to Closest Record
        print("\n--- 3.2 Distance to Closest Record (DCR) ---")

        # Split real data into train/holdout
        n_real = len(self.real_data)
        train_size = int(0.8 * n_real)

        real_train = self.real_data.iloc[:train_size]
        real_holdout = self.real_data.iloc[train_size:]

        # Get numeric features only for distance calculation
        numeric_features = [c for c in compare_cols
                          if c in self.real_data.select_dtypes(include=[np.number]).columns]

        # Impute missing values with median (intelligent strategy)
        train_imputed = self._impute_missing_values(real_train, numeric_features)
        holdout_imputed = self._impute_missing_values(real_holdout, numeric_features)
        synth_imputed = self._impute_missing_values(self.synthetic_data, numeric_features)

        # Standardize and compute distances
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_imputed)
        holdout_scaled = scaler.transform(holdout_imputed)
        synth_scaled = scaler.transform(synth_imputed)

        # Nearest neighbor to train set
        nn_train = NearestNeighbors(n_neighbors=1)
        nn_train.fit(train_scaled)
        dist_to_train, _ = nn_train.kneighbors(synth_scaled)

        # Nearest neighbor to holdout set
        nn_holdout = NearestNeighbors(n_neighbors=1)
        nn_holdout.fit(holdout_scaled)
        dist_to_holdout, _ = nn_holdout.kneighbors(synth_scaled)

        # Calculate ratio of synthetic points closer to train than holdout
        closer_to_train_ratio = (dist_to_train.flatten() < dist_to_holdout.flatten()).mean()

        # Pass criteria: ratio <= 0.50 (no memorization)
        passed_dcr = (closer_to_train_ratio <= 0.5)

        self.results['dcr'] = {
            'closer_to_train_ratio': float(closer_to_train_ratio),
            'mean_dist_to_train': float(dist_to_train.mean()),
            'mean_dist_to_holdout': float(dist_to_holdout.mean()),
            'passed': passed_dcr
        }

        status = " PASS" if passed_dcr else " FAIL"
        print(f"  Closer to train: {closer_to_train_ratio:.4f} (target: d 0.50)")
        print(f"  Mean distance to train: {dist_to_train.mean():.4f}")
        print(f"  Mean distance to holdout: {dist_to_holdout.mean():.4f}")
        print(f"  Result: {status}")

    def _validate_stationarity(self):
        """
        Validate stationarity patterns to ensure synthetic data captures disease progression.

        Critical objective:
        - Real data should be non-stationary (disease progression)
        - Synthetic data should also be non-stationary
        - If real is non-stationary but synthetic is stationary → MODEL FAILED

        Tests:
        - ADF (Augmented Dickey-Fuller): Tests null hypothesis of non-stationarity
        - KPSS (Kwiatkowski-Phillips-Schmidt-Shin): Tests null hypothesis of stationarity
        """
        if not STATIONARITY_AVAILABLE:
            print("  ⚠️  Stationarity tests not available (stationarity_tests module not found)")
            self.results['stationarity'] = {'skipped': True, 'reason': 'Module not available'}
            return

        # Check if we have required columns
        if self.id_var is None or self.id_var not in self.real_data.columns:
            print(f"  ⚠️  ID variable '{self.id_var}' not found, skipping stationarity tests")
            self.results['stationarity'] = {'skipped': True, 'reason': 'ID variable not found'}
            return

        if self.time_var is None or self.time_var not in self.real_data.columns:
            print(f"  ⚠️  Time variable '{self.time_var}' not found, skipping stationarity tests")
            self.results['stationarity'] = {'skipped': True, 'reason': 'Time variable not found'}
            return

        # Get numeric features to test (exclude ID and time)
        test_variables = [
            col for col in self.numeric_features
            if col != self.id_var and col != self.time_var
        ]

        if len(test_variables) == 0:
            print("  ⚠️  No numeric variables found for stationarity testing")
            self.results['stationarity'] = {'skipped': True, 'reason': 'No numeric variables'}
            return

        print(f"\n  Testing stationarity for {len(test_variables)} variables:")
        for var in test_variables:
            print(f"    • {var}")

        stationarity_results = {}

        # Test each variable
        for variable in test_variables:
            print(f"\n--- Testing {variable} ---")

            # Check if variable exists in both datasets
            if variable not in self.real_data.columns or variable not in self.synthetic_data.columns:
                print(f"  ⚠️  {variable} not found in both datasets, skipping")
                continue

            # Test per-patient stationarity
            print(f"  Testing per-patient trajectories...")
            try:
                real_adf, real_kpss = test_stationarity_per_patient(
                    self.real_data, variable, self.id_var, self.time_var
                )
                synth_adf, synth_kpss = test_stationarity_per_patient(
                    self.synthetic_data, variable, self.id_var, self.time_var
                )

                print(f"    Real: {len(real_adf)} patients tested (ADF), {len(real_kpss)} (KPSS)")
                print(f"    Synthetic: {len(synth_adf)} patients tested (ADF), {len(synth_kpss)} (KPSS)")

                # Compare stationarity patterns
                if len(real_adf) > 0 and len(synth_adf) > 0:
                    comparison_adf = compare_stationarity(real_adf, synth_adf, 'ADF')
                    print(f"\n  ADF Comparison:")
                    print(f"    Real: {comparison_adf['real']['stationary_percentage']:.1f}% stationary")
                    print(f"    Synthetic: {comparison_adf['synthetic']['stationary_percentage']:.1f}% stationary")
                    print(f"    Status: {comparison_adf['status']}")
                else:
                    comparison_adf = None
                    print(f"  ⚠️  Insufficient data for ADF comparison")

                if len(real_kpss) > 0 and len(synth_kpss) > 0:
                    comparison_kpss = compare_stationarity(real_kpss, synth_kpss, 'KPSS')
                    print(f"\n  KPSS Comparison:")
                    print(f"    Real: {comparison_kpss['real']['stationary_percentage']:.1f}% stationary")
                    print(f"    Synthetic: {comparison_kpss['synthetic']['stationary_percentage']:.1f}% stationary")
                    print(f"    Status: {comparison_kpss['status']}")
                else:
                    comparison_kpss = None
                    print(f"  ⚠️  Insufficient data for KPSS comparison")

                # Store results
                stationarity_results[variable] = {
                    'real_adf': real_adf,
                    'real_kpss': real_kpss,
                    'synth_adf': synth_adf,
                    'synth_kpss': synth_kpss,
                    'comparison_adf': comparison_adf,
                    'comparison_kpss': comparison_kpss
                }

                # Generate plots if enabled
                if self.generate_plots and comparison_adf and comparison_kpss:
                    import os
                    plot_path = os.path.join(self.subdirs['stationarity'], f'stationarity_{variable}.png')
                    fig = plot_stationarity_comparison(
                        real_adf, synth_adf,
                        real_kpss, synth_kpss,
                        comparison_adf, comparison_kpss,
                        save_path=plot_path
                    )
                    plt.close(fig)

                    # Save detailed summary
                    summary_path = os.path.join(self.subdirs['stationarity'], f'stationarity_{variable}_summary.txt')
                    save_stationarity_summary(comparison_adf, comparison_kpss, summary_path)

            except Exception as e:
                print(f"  ❌ Error testing {variable}: {e}")
                import traceback
                traceback.print_exc()
                stationarity_results[variable] = {'error': str(e)}

        # Store results
        self.results['stationarity'] = stationarity_results

        # Summary
        print("\n--- Stationarity Validation Summary ---")
        failed_vars = []
        passed_vars = []

        for var, res in stationarity_results.items():
            if 'error' in res:
                continue

            comp_adf = res.get('comparison_adf')
            comp_kpss = res.get('comparison_kpss')

            if comp_adf and comp_kpss:
                if comp_adf['model_failed'] or comp_kpss['model_failed']:
                    failed_vars.append(var)
                    print(f"  ❌ {var}: FAILED (model not capturing progression)")
                elif comp_adf['status'] == 'PASSED' and comp_kpss['status'] == 'PASSED':
                    passed_vars.append(var)
                    print(f"  ✅ {var}: PASSED")
                else:
                    print(f"  ⚠️  {var}: WARNING")

        if failed_vars:
            print(f"\n  ⚠️  CRITICAL: {len(failed_vars)} variable(s) failed stationarity check")
            print(f"     The model is generating STATIC patients instead of disease progression!")
            print(f"     Failed variables: {', '.join(failed_vars)}")
        elif passed_vars:
            print(f"\n  ✅ All tested variables passed stationarity validation")
            print(f"     The model successfully captures disease progression dynamics")

    def _validate_acf_pacf(self):
        """
        Validate AR/MA orders to ensure synthetic data captures temporal structure.

        Critical objectives:
        - AR/MA orders should be consistent between real and synthetic data
        - Distribution of orders should match

        Tests:
        - AR/MA Order Detection: Automatic model order identification with AIC/BIC
        """
        if not ACF_PACF_AVAILABLE:
            print("  ⚠️  AR/MA order tests not available (acf_pacf_tests module not found)")
            self.results['acf_pacf'] = {'skipped': True, 'reason': 'Module not available'}
            return

        # Check if we have required columns
        if self.id_var is None or self.id_var not in self.real_data.columns:
            print(f"  ⚠️  ID variable '{self.id_var}' not found, skipping AR/MA order tests")
            self.results['acf_pacf'] = {'skipped': True, 'reason': 'ID variable not found'}
            return

        if self.time_var is None or self.time_var not in self.real_data.columns:
            print(f"  ⚠️  Time variable '{self.time_var}' not found, skipping AR/MA order tests")
            self.results['acf_pacf'] = {'skipped': True, 'reason': 'Time variable not found'}
            return

        # Get numeric features to test (exclude ID and time)
        test_variables = [
            col for col in self.numeric_features
            if col != self.id_var and col != self.time_var
        ]

        if len(test_variables) == 0:
            print("  ⚠️  No numeric variables found for AR/MA order testing")
            self.results['acf_pacf'] = {'skipped': True, 'reason': 'No numeric variables'}
            return

        print(f"\n  Testing AR/MA Orders for {len(test_variables)} variables:")
        for var in test_variables:
            print(f"    • {var}")

        ar_ma_results = {}

        # Test each variable
        for variable in test_variables:
            print(f"\n--- Testing {variable} ---")

            # Check if variable exists in both datasets
            if variable not in self.real_data.columns or variable not in self.synthetic_data.columns:
                print(f"  ⚠️  {variable} not found in both datasets, skipping")
                continue

            try:
                # Detect AR/MA orders
                print(f"  Detecting AR/MA orders (requires ≥10 observations per patient)...")
                real_orders = detect_ar_ma_order_per_patient(
                    self.real_data, variable, self.id_var, self.time_var, max_p=3, max_q=3
                )
                synth_orders = detect_ar_ma_order_per_patient(
                    self.synthetic_data, variable, self.id_var, self.time_var, max_p=3, max_q=3
                )

                print(f"    Real: {len(real_orders)} patients analyzed")
                print(f"    Synthetic: {len(synth_orders)} patients analyzed")

                if len(real_orders) > 0 and len(synth_orders) > 0:
                    comparison_orders = compare_ar_ma_orders(real_orders, synth_orders)
                    print(f"\n  AR/MA Order Comparison:")
                    print(f"    AR Order - Real: AR({comparison_orders['ar_orders']['real_mode']}), "
                          f"Synthetic: AR({comparison_orders['ar_orders']['synth_mode']})")
                    print(f"    MA Order - Real: MA({comparison_orders['ma_orders']['real_mode']}), "
                          f"Synthetic: MA({comparison_orders['ma_orders']['synth_mode']})")
                    print(f"    AR Match: {'YES' if comparison_orders['ar_orders']['match'] else 'NO'}")
                    print(f"    MA Match: {'YES' if comparison_orders['ma_orders']['match'] else 'NO'}")
                    print(f"    Status: {comparison_orders['status']}")

                    # Store results
                    ar_ma_results[variable] = {
                        'comparison_orders': comparison_orders,
                        'n_real_patients': len(real_orders),
                        'n_synth_patients': len(synth_orders)
                    }

                    # Save detailed summary text file
                    if self.generate_plots:
                        import os
                        summary_path = os.path.join(self.subdirs['acf_pacf'], f'ar_ma_orders_{variable}_summary.txt')
                        self._save_ar_ma_summary(comparison_orders, variable, summary_path)

                else:
                    print(f"  ⚠️  Insufficient data for AR/MA order detection")
                    ar_ma_results[variable] = {'error': 'Insufficient data (need ≥10 observations per patient)'}

            except Exception as e:
                print(f"  ❌ Error testing {variable}: {e}")
                import traceback
                traceback.print_exc()
                ar_ma_results[variable] = {'error': str(e)}

        # Store results
        self.results['acf_pacf'] = ar_ma_results

        # Summary
        print("\n--- AR/MA Order Validation Summary ---")
        failed_vars = []
        passed_vars = []
        warning_vars = []

        for var, res in ar_ma_results.items():
            if 'error' in res:
                continue

            comp_orders = res.get('comparison_orders')

            if comp_orders:
                if comp_orders['status'] == 'FAILED':
                    failed_vars.append(var)
                    print(f"  ❌ {var}: FAILED (AR/MA structure mismatch)")
                elif comp_orders['status'] == 'PASSED':
                    passed_vars.append(var)
                    print(f"  ✅ {var}: PASSED")
                else:
                    warning_vars.append(var)
                    print(f"  ⚠️  {var}: WARNING (partial match)")

        if failed_vars:
            print(f"\n  ⚠️  {len(failed_vars)} variable(s) failed AR/MA order check")
            print(f"     Failed variables: {', '.join(failed_vars)}")
        elif warning_vars:
            print(f"\n  ⚠️  {len(warning_vars)} variable(s) have warnings (partial AR/MA match)")
            print(f"     Warning variables: {', '.join(warning_vars)}")

        if passed_vars:
            print(f"\n  ✅ {len(passed_vars)} variable(s) passed AR/MA order validation")
            print(f"     The model captures the correct temporal structure")

    def _save_ar_ma_summary(self, comparison_orders: Dict, variable_name: str, save_path: str):
        """Save AR/MA order comparison summary to text file."""
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write(f"AR/MA ORDER DETECTION SUMMARY: {variable_name}\n")
            f.write("=" * 80 + "\n\n")

            f.write("Objective:\n")
            f.write("-" * 80 + "\n")
            f.write("Validate that synthetic data has the same temporal structure (AR/MA orders)\n")
            f.write("as real data.\n\n")

            f.write("What is AR/MA order?\n")
            f.write("-" * 80 + "\n")
            f.write("AR(p): Autoregressive model of order p (depends on p previous values)\n")
            f.write("MA(q): Moving average model of order q (depends on q previous errors)\n\n")

            f.write("CRITICAL: If AR/MA orders differ, the synthetic model may be:\n")
            f.write("  - Overfitting (too high order)\n")
            f.write("  - Underfitting (too low order)\n")
            f.write("  - Using wrong temporal dependencies\n\n")

            # AR Orders
            f.write("=" * 80 + "\n")
            f.write("AR ORDER (AUTOREGRESSIVE)\n")
            f.write("=" * 80 + "\n\n")

            real_ar_mode = comparison_orders['ar_orders']['real_mode']
            synth_ar_mode = comparison_orders['ar_orders']['synth_mode']
            ar_match = comparison_orders['ar_orders']['match']

            f.write(f"Most common in Real data:      AR({real_ar_mode})\n")
            f.write(f"Most common in Synthetic data: AR({synth_ar_mode})\n")
            f.write(f"Match: {'YES' if ar_match else 'NO'}\n\n")

            if comparison_orders['ar_orders']['real_distribution']:
                f.write("Real AR order distribution:\n")
                for order, pct in sorted(comparison_orders['ar_orders']['real_distribution'].items()):
                    f.write(f"  AR({order}): {pct:.1f}%\n")
                f.write("\n")

            if comparison_orders['ar_orders']['synth_distribution']:
                f.write("Synthetic AR order distribution:\n")
                for order, pct in sorted(comparison_orders['ar_orders']['synth_distribution'].items()):
                    f.write(f"  AR({order}): {pct:.1f}%\n")
                f.write("\n")

            # MA Orders
            f.write("=" * 80 + "\n")
            f.write("MA ORDER (MOVING AVERAGE)\n")
            f.write("=" * 80 + "\n\n")

            real_ma_mode = comparison_orders['ma_orders']['real_mode']
            synth_ma_mode = comparison_orders['ma_orders']['synth_mode']
            ma_match = comparison_orders['ma_orders']['match']

            f.write(f"Most common in Real data:      MA({real_ma_mode})\n")
            f.write(f"Most common in Synthetic data: MA({synth_ma_mode})\n")
            f.write(f"Match: {'YES' if ma_match else 'NO'}\n\n")

            if comparison_orders['ma_orders']['real_distribution']:
                f.write("Real MA order distribution:\n")
                for order, pct in sorted(comparison_orders['ma_orders']['real_distribution'].items()):
                    f.write(f"  MA({order}): {pct:.1f}%\n")
                f.write("\n")

            if comparison_orders['ma_orders']['synth_distribution']:
                f.write("Synthetic MA order distribution:\n")
                for order, pct in sorted(comparison_orders['ma_orders']['synth_distribution'].items()):
                    f.write(f"  MA({order}): {pct:.1f}%\n")
                f.write("\n")

            # Overall conclusion
            f.write("=" * 80 + "\n")
            f.write("CONCLUSION\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Status: {comparison_orders['status']}\n\n")
            f.write(f"Interpretation:\n{comparison_orders['interpretation']}\n\n")

            if comparison_orders['status'] == 'PASSED':
                f.write("RECOMMENDATION: Synthetic data has correct temporal structure.\n")
                f.write("                The model captures AR/MA dynamics properly.\n")
            elif comparison_orders['status'] == 'WARNING':
                f.write("RECOMMENDATION: Review model. Partial match in AR/MA structure.\n")
                f.write("                May affect some longitudinal analyses.\n")
            else:
                f.write("RECOMMENDATION: CRITICAL - Different AR/MA structure detected.\n")
                f.write("                Revise model architecture.\n")

            f.write("=" * 80 + "\n")

        print(f"  AR/MA summary saved to: {save_path}")

    def _run_maximum_similarity_test(self):
        """
        Run Maximum Similarity Test (unified metric).

        This test provides a single quality score that combines
        fidelity and privacy in one metric.
        """
        print("\n--- 4.1 Computing Maximum Similarities ---")

        # Prepare feature matrix
        compare_cols = [c for c in self.feature_cols if c != self.id_var]

        # Run Maximum Similarity Test
        mst_results = maximum_similarity_test(
            self.real_data[compare_cols],
            self.synthetic_data[compare_cols],
            categorical_features=self.categorical_indices
        )

        self.results['maximum_similarity_test'] = mst_results

        print(f"  Mean MISS (Real): {mst_results['mean_miss_real']:.4f}")
        print(f"  Mean MISS (Synth): {mst_results['mean_miss_synth']:.4f}")
        print(f"  Mean MCSS: {mst_results['mean_mcss']:.4f}")
        print(f"\n  Quality Score: {mst_results['quality_score']:.4f}")
        print(f"    - Q H 1.0: Perfect")
        print(f"    - Q < 1.0: Good (less similar)")
        print(f"    - Q > 1.0: FAILED (privacy compromised)")

        privacy_status = " PASS" if mst_results['passed_privacy'] else " FAIL"
        fidelity_status = " PASS" if mst_results['passed_fidelity'] else " FAIL"

        print(f"\n  Privacy Test (Q d 1.0): {privacy_status}")
        print(f"  Fidelity Test (KS p > 0.05): {fidelity_status}")

    def _print_fidelity_summary(self):
        """
        Print fidelity-only validation summary.
        """
        print("\n" + "=" * 80)
        print("FIDELITY VALIDATION SUMMARY")
        print("=" * 80)

        # Count passed tests for fidelity
        passed_count = 0
        total_count = 0

        # Univariate tests
        if 'univariate' in self.results:
            for res in self.results['univariate'].values():
                total_count += 1
                if res['passed']:
                    passed_count += 1

        # Correlation test
        if 'correlations' in self.results:
            total_count += 1
            if self.results['correlations']['passed']:
                passed_count += 1

        # Print results
        if total_count > 0:
            rate = passed_count / total_count
            print(f"\nFidelity Tests: {passed_count}/{total_count} passed ({rate:.1%})")

            if rate >= 0.8:
                status = "✅ EXCELLENT"
            elif rate >= 0.6:
                status = "⚠️  ACCEPTABLE"
            else:
                status = "❌ POOR"

            print(f"Overall Status: {status}")
        else:
            print("\n⚠️  No fidelity tests were performed")

        # Final recommendation
        print("\n" + "=" * 80)
        if passed_count == total_count and total_count > 0:
            print("✅ ALL FIDELITY TESTS PASSED")
            print("   Synthetic data has excellent statistical fidelity")
        elif passed_count / total_count >= 0.8 if total_count > 0 else False:
            print("✅ FIDELITY VALIDATION MOSTLY PASSED")
            print("   Synthetic data has good statistical fidelity")
        else:
            print("❌ FIDELITY VALIDATION FAILED")
            print("   Synthetic data has poor statistical fidelity")
        print("=" * 80)

    def _print_summary(self):
        """
        Print validation summary with overall assessment.
        """
        print("\n" + "=" * 80)
        print("VALIDATION SUMMARY")
        print("=" * 80)

        # Count passed tests per dimension
        passed_counts = {'Fidelity': 0, 'Utility': 0, 'Privacy': 0, 'MST': 0}
        total_counts = {'Fidelity': 0, 'Utility': 0, 'Privacy': 0, 'MST': 0}

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
            total_counts['MST'] += 2
            if self.results['maximum_similarity_test']['passed_privacy']:
                passed_counts['MST'] += 1
            if self.results['maximum_similarity_test']['passed_fidelity']:
                passed_counts['MST'] += 1

        # Print dimension-wise results
        print("\nDimension-wise Results:")
        for dim in ['Fidelity', 'Utility', 'Privacy', 'MST']:
            if total_counts[dim] > 0:
                rate = passed_counts[dim] / total_counts[dim]
                if rate >= 0.8:
                    status = " EXCELLENT"
                elif rate >= 0.6:
                    status = "� ACCEPTABLE"
                else:
                    status = " POOR"

                print(f"  {dim:15s}: {passed_counts[dim]}/{total_counts[dim]} "
                      f"({rate:.1%}) {status}")

        # Overall assessment
        total_all = sum(total_counts.values())
        passed_all = sum(passed_counts.values())
        overall_rate = passed_all / total_all if total_all > 0 else 0

        print(f"\nOverall Score: {passed_all}/{total_all} ({overall_rate:.1%})")

        # Final recommendation
        print("\n" + "=" * 80)
        if overall_rate >= 0.8:
            print(" SYNTHETIC DATA VALIDATED")
            print("Quality: HIGH - Recommended for use")
        elif overall_rate >= 0.6:
            print("� SYNTHETIC DATA PARTIALLY VALIDATED")
            print("Quality: MEDIUM - Use with caution")
        else:
            print(" SYNTHETIC DATA FAILED VALIDATION")
            print("Quality: LOW - Not recommended for use")

        # Critical privacy check
        privacy_passed = (
            passed_counts['Privacy'] == total_counts['Privacy'] and
            self.results.get('maximum_similarity_test', {}).get('passed_privacy', False)
        )

        if not privacy_passed:
            print("\n�  CRITICAL WARNING: Privacy tests failed!")
            print("   Model shows signs of overfitting.")
            print("   Fidelity and Utility metrics may be unreliable.")

        print("=" * 80)

    def save_report(self, filename: str = 'validation_report.json'):
        """
        Save validation report to JSON file.

        Parameters
        ----------
        filename : str
            Output filename
        """
        # Recursively convert all types to JSON-serializable format
        def convert_to_json_serializable(obj):
            """Recursively convert objects to JSON-serializable types."""
            if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
                return float(obj)
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.DataFrame):
                # Don't serialize DataFrames - they can cause circular refs
                return None
            elif isinstance(obj, pd.Series):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_json_serializable(item) for item in obj]
            elif isinstance(obj, (str, int, float, type(None))):
                return obj
            else:
                # Unknown type - convert to string
                return str(obj)

        # Remove large arrays and complex objects from results for JSON
        results_to_save = {}

        for key, value in self.results.items():
            if key == 'maximum_similarity_test':
                # Keep only summary statistics, not full arrays
                mst = value
                results_to_save['maximum_similarity_test'] = {
                    'mean_miss_real': float(mst['mean_miss_real']),
                    'mean_miss_synth': float(mst['mean_miss_synth']),
                    'mean_mcss': float(mst['mean_mcss']),
                    'quality_score': float(mst['quality_score']),
                    'ks_real_vs_synth': {
                        'statistic': float(mst['ks_real_vs_synth']['statistic']),
                        'pvalue': float(mst['ks_real_vs_synth']['pvalue'])
                    },
                    'ks_real_vs_cross': {
                        'statistic': float(mst['ks_real_vs_cross']['statistic']),
                        'pvalue': float(mst['ks_real_vs_cross']['pvalue'])
                    },
                    'passed_privacy': bool(mst['passed_privacy']),
                    'passed_fidelity': bool(mst['passed_fidelity']),
                    'overall_passed': bool(mst['overall_passed'])
                }
            elif key == 'correlations':
                # Remove full correlation matrices to avoid circular references
                # Keep only the summary statistics
                corr_data = {
                    'mean_abs_diff': float(value['mean_abs_diff']),
                    'meta_correlation': float(value['meta_correlation']),
                    'passed': bool(value['passed'])
                }
                results_to_save['correlations'] = corr_data
            elif key == 'stationarity':
                # For stationarity results, keep only comparison summaries
                # Skip individual patient test results to avoid huge JSON files
                stationarity_summary = {}
                for var_name, var_results in value.items():
                    if 'error' in var_results:
                        stationarity_summary[var_name] = {'error': var_results['error']}
                    elif 'skipped' in var_results:
                        stationarity_summary[var_name] = var_results
                    else:
                        # Extract only comparison summaries
                        comp_adf = var_results.get('comparison_adf')
                        comp_kpss = var_results.get('comparison_kpss')

                        summary = {}
                        if comp_adf:
                            summary['adf'] = {
                                'real_stationary_pct': float(comp_adf['real']['stationary_percentage']),
                                'synth_stationary_pct': float(comp_adf['synthetic']['stationary_percentage']),
                                'difference_pct': float(comp_adf['difference_pct']),
                                'status': comp_adf['status'],
                                'model_failed': bool(comp_adf['model_failed'])
                            }
                        if comp_kpss:
                            summary['kpss'] = {
                                'real_stationary_pct': float(comp_kpss['real']['stationary_percentage']),
                                'synth_stationary_pct': float(comp_kpss['synthetic']['stationary_percentage']),
                                'difference_pct': float(comp_kpss['difference_pct']),
                                'status': comp_kpss['status'],
                                'model_failed': bool(comp_kpss['model_failed'])
                            }

                        stationarity_summary[var_name] = summary

                results_to_save['stationarity'] = stationarity_summary
            elif key == 'acf_pacf':
                # For AR/MA order results, keep only comparison summaries
                ar_ma_summary = {}
                for var_name, var_results in value.items():
                    if 'error' in var_results:
                        ar_ma_summary[var_name] = {'error': var_results['error']}
                    elif 'skipped' in var_results:
                        ar_ma_summary[var_name] = var_results
                    else:
                        # Extract only AR/MA order comparison
                        comp_orders = var_results.get('comparison_orders')

                        summary = {}
                        if comp_orders:
                            summary['ar_ma_orders'] = {
                                'real_ar_mode': comp_orders['ar_orders']['real_mode'],
                                'synth_ar_mode': comp_orders['ar_orders']['synth_mode'],
                                'ar_match': bool(comp_orders['ar_orders']['match']),
                                'real_ar_distribution': comp_orders['ar_orders']['real_distribution'],
                                'synth_ar_distribution': comp_orders['ar_orders']['synth_distribution'],
                                'real_ma_mode': comp_orders['ma_orders']['real_mode'],
                                'synth_ma_mode': comp_orders['ma_orders']['synth_mode'],
                                'ma_match': bool(comp_orders['ma_orders']['match']),
                                'real_ma_distribution': comp_orders['ma_orders']['real_distribution'],
                                'synth_ma_distribution': comp_orders['ma_orders']['synth_distribution'],
                                'status': comp_orders['status'],
                                'interpretation': comp_orders['interpretation']
                            }
                            # Add patient counts
                            summary['n_real_patients'] = var_results.get('n_real_patients', 0)
                            summary['n_synth_patients'] = var_results.get('n_synth_patients', 0)

                        ar_ma_summary[var_name] = summary

                results_to_save['acf_pacf'] = ar_ma_summary
            else:
                # Convert other results recursively
                results_to_save[key] = convert_to_json_serializable(value)

        # Final conversion pass to ensure everything is serializable
        results_to_save = convert_to_json_serializable(results_to_save)

        with open(filename, 'w') as f:
            json.dump(results_to_save, f, indent=2)

        print(f"\n Validation report saved to: {filename}")

    def generate_visualizations(self, output_dir: str = '.'):
        """
        Generate and save all validation plots.

        Parameters
        ----------
        output_dir : str
            Directory to save plots
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        print("\nGenerating visualizations...")

        # Maximum Similarity Test plot
        if 'maximum_similarity_test' in self.results:
            similarity_dir = os.path.join(output_dir, 'similarity')
            os.makedirs(similarity_dir, exist_ok=True)
            fig = plot_similarity_distributions(
                self.results['maximum_similarity_test'],
                save_path=os.path.join(similarity_dir, 'maximum_similarity_test.png')
            )
            plt.close(fig)

        # Correlation comparison plot
        if 'correlations' in self.results:
            correlation_dir = os.path.join(output_dir, 'correlation')
            os.makedirs(correlation_dir, exist_ok=True)
            numeric_cols = self.real_data[self.feature_cols].select_dtypes(include=[np.number]).columns.tolist()
            real_corr = self.real_data[numeric_cols].corr()
            synth_corr = self.synthetic_data[numeric_cols].corr()

            fig = plot_correlation_comparison(
                real_corr, synth_corr,
                save_path=os.path.join(correlation_dir, 'correlation_comparison.png')
            )
            plt.close(fig)

        print(f" Visualizations saved to: {output_dir}")


# ============================================================================
# STEP 5: MAIN EXECUTION FUNCTION
# ============================================================================

def main():
    """
    Main function demonstrating usage of the validation pipeline.

    This is an example. Replace with your actual data paths and parameters.
    """
    print("=" * 80)
    print("SYNTHETIC DATA VALIDATION - EXAMPLE USAGE")
    print("=" * 80)

    # Example: Load your data
    # Replace these paths with your actual data files
    print("\n=� Loading data...")

    # Example data loading (uncomment and modify for your use case)
    # real_data = pd.read_csv('path/to/real_data.csv')
    # synthetic_data = pd.read_csv('path/to/synthetic_data.csv')

    # For demonstration, create synthetic example data
    np.random.seed(42)
    n_samples = 500

    real_data = pd.DataFrame({
        'ID': range(n_samples),
        'age': np.random.normal(70, 10, n_samples),
        'biomarker1': np.random.normal(100, 15, n_samples),
        'biomarker2': np.random.normal(50, 8, n_samples),
        'diagnosis': np.random.choice(['CN', 'MCI', 'AD'], n_samples)
    })

    # Generate synthetic data with slight variations
    synthetic_data = pd.DataFrame({
        'ID': range(n_samples),
        'age': np.random.normal(70, 10, n_samples) + np.random.normal(0, 2, n_samples),
        'biomarker1': np.random.normal(100, 15, n_samples) + np.random.normal(0, 3, n_samples),
        'biomarker2': np.random.normal(50, 8, n_samples) + np.random.normal(0, 1.5, n_samples),
        'diagnosis': np.random.choice(['CN', 'MCI', 'AD'], n_samples)
    })

    print(f"  Real data: {real_data.shape}")
    print(f"  Synthetic data: {synthetic_data.shape}")

    # Initialize validator
    print("\n=' Initializing validator...")
    validator = SyntheticDataValidator(
        real_data=real_data,
        synthetic_data=synthetic_data,
        categorical_features=['diagnosis'],
        id_var='ID',
        target_var='diagnosis'
    )

    # Run validation
    print("\n=� Running validation pipeline...\n")
    results = validator.run_full_validation()

    # Save results
    print("\n=� Saving results...")
    validator.save_report('validation_report.json')
    validator.generate_visualizations(output_dir='validation_results')

    print("\n Validation complete!")
    print("\nNext steps:")
    print("  1. Review validation_report.json for detailed results")
    print("  2. Check validation_results/ for visualizations")
    print("  3. If validation passed, synthetic data is ready to use")
    print("  4. If validation failed, consider:")
    print("     - Adjusting your generative model parameters")
    print("     - Using a different generation approach")
    print("     - Collecting more real data for better model training")


if __name__ == "__main__":
    # Run the main validation pipeline
    main()
