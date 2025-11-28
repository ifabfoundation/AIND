#!/usr/bin/env python
"""
Validation with Automatic Preprocessing
========================================

This script automatically:
1. Loads data from datalake
2. Preprocesses/aligns datasets (removes 'generated_' prefix) - OPTIONAL
3. Runs pre-validation check
4. Runs full validation OR fidelity-only validation (if pre-check passes)

Modes:
------
1. STANDARD MODE (default):
   - Full preprocessing and alignment
   - Complete validation (fidelity, utility, privacy)

2. NO-ALIGN MODE (--skip-align):
   - Skip preprocessing/alignment
   - Complete validation on raw data

3. AGE-MATCHING MODE (--age-matching):
   - Filters synthetic data to match age range of real data
   - Balances dataset sizes if needed (>15% difference)
   - Runs pre-validation check
   - Runs ONLY fidelity tests (KS + correlation)

Usage:
------
python validate_with_preprocessing.py                      # Standard mode
python validate_with_preprocessing.py --skip-align         # No-align mode
python validate_with_preprocessing.py --age-matching       # Age-matching mode
python validate_with_preprocessing.py --age-matching --age-column AGE  # Custom age column

Or customize in the script below.
"""

import sys
import argparse
import numpy as np
from dl_client import DatalakeClient
from preprocess_synthetic import quick_align
from pre_validation_check import load_real_and_synthetic_from_datalake, PreValidationChecker
from validate_data import SyntheticDataValidator


# ============================================================================
# UTILITY FUNCTIONS FOR AGE-MATCHING MODE
# ============================================================================

def filter_by_age_range(real_data, synthetic_data, age_column='AGE', tolerance_percentage=5):
    """
    Filter synthetic data to keep only patients with age similar to real data.

    Parameters
    ----------
    real_data : pd.DataFrame
        Real dataset
    synthetic_data : pd.DataFrame
        Synthetic dataset
    age_column : str, default='AGE'
        Name of the age column
    tolerance_percentage : float, default=5
        Percentage tolerance for age range (e.g., 5 means ±5%)

    Returns
    -------
    filtered_synthetic : pd.DataFrame
        Filtered synthetic data with similar age range
    """
    if age_column not in real_data.columns or age_column not in synthetic_data.columns:
        print(f"  ⚠️  Age column '{age_column}' not found, skipping age filtering")
        return synthetic_data

    # Calculate age statistics from real data
    real_age_min = real_data[age_column].min()
    real_age_max = real_data[age_column].max()
    real_age_range = real_age_max - real_age_min

    # Calculate tolerance
    tolerance = real_age_range * (tolerance_percentage / 100)

    # Define acceptable age range (with tolerance)
    age_min_threshold = real_age_min - tolerance
    age_max_threshold = real_age_max + tolerance

    print(f"\n  Real age range: {real_age_min:.1f} - {real_age_max:.1f}")
    print(f"  Acceptable range (±{tolerance_percentage}%): {age_min_threshold:.1f} - {age_max_threshold:.1f}")

    # Filter synthetic data
    synth_before = len(synthetic_data)
    filtered_synthetic = synthetic_data[
        (synthetic_data[age_column] >= age_min_threshold) &
        (synthetic_data[age_column] <= age_max_threshold)
    ].copy()
    synth_after = len(filtered_synthetic)

    removed = synth_before - synth_after
    removed_pct = (removed / synth_before * 100) if synth_before > 0 else 0

    print(f"  Synthetic records: {synth_before} → {synth_after} (removed {removed}, {removed_pct:.1f}%)")

    return filtered_synthetic


def balance_datasets_if_needed(real_data, synthetic_data, threshold_percentage=15, random_state=42):
    """
    Reduce real dataset if synthetic is too much smaller (>threshold% less).

    Parameters
    ----------
    real_data : pd.DataFrame
        Real dataset
    synthetic_data : pd.DataFrame
        Synthetic dataset (already filtered)
    threshold_percentage : float, default=15
        If synthetic is more than this % smaller, reduce real data too
    random_state : int, default=42
        Random seed for reproducibility

    Returns
    -------
    balanced_real : pd.DataFrame
        Balanced real data
    balanced_synthetic : pd.DataFrame
        Synthetic data (unchanged)
    """
    n_real = len(real_data)
    n_synth = len(synthetic_data)

    # Calculate the difference
    diff_pct = ((n_real - n_synth) / n_real * 100) if n_real > 0 else 0

    print(f"\n  Dataset sizes:")
    print(f"    Real: {n_real}")
    print(f"    Synthetic: {n_synth}")
    print(f"    Difference: {diff_pct:.1f}%")

    if diff_pct > threshold_percentage:
        print(f"\n  ⚠️  Synthetic is {diff_pct:.1f}% smaller (threshold: {threshold_percentage}%)")
        print(f"  → Reducing real dataset to {n_synth} records")

        # Sample real data to match synthetic size
        balanced_real = real_data.sample(n=n_synth, random_state=random_state).copy()

        print(f"  ✅ Real dataset reduced: {n_real} → {len(balanced_real)}")
        return balanced_real, synthetic_data
    else:
        print(f"  ✅ Size difference ({diff_pct:.1f}%) is acceptable (threshold: {threshold_percentage}%)")
        return real_data, synthetic_data


# ============================================================================
# CONFIGURATION - EDIT THESE VALUES
# ============================================================================

# Datalake settings
FILE_CODE = 'ADNIMERGE'  # Your real data file code
LEVEL = 'cleaned_03'     # Datalake level

# Utility validation settings (TSTR test)
# Set this to use a DIFFERENT synthetic dataset for utility validation
# If None, uses the main synthetic dataset (FILE_CODE + 'synthetic')
UTILITY_SYNTHETIC_FILE_CODE = None  # e.g., 'ADNIMERGEsynthetic_v2'

# Column settings
ID_VAR = 'ID'            # ID column
TIME_VAR = 'TIME'        # Time column (for longitudinal data)
TARGET_VAR = None        # Target variable for utility validation (e.g., 'DX', 'MMSE_category')

# Categorical features (WITHOUT 'generated_' prefix - will be matched after alignment)
CATEGORICAL_FEATURES = []

# Output settings
PRECHECK_REPORT = 'pre_validation_report_aligned.txt'
VALIDATION_REPORT = 'validation_report.json'
PLOTS_DIR = 'validation_results'


# ============================================================================
# MAIN WORKFLOW
# ============================================================================

def main(skip_align=False, age_matching=False, age_column='AGE',
         utility_synthetic_file_code=None, target_var=None):
    """
    Main validation workflow with automatic preprocessing.

    Parameters
    ----------
    skip_align : bool, default=False
        If True, skips the preprocessing/alignment step
    age_matching : bool, default=False
        If True, enables age-matching mode:
        - Filters synthetic data by age range similar to real data
        - Balances dataset sizes if needed
        - Runs only fidelity tests (KS + correlation)
    age_column : str, default='AGE'
        Name of the age column (only used if age_matching=True)
    utility_synthetic_file_code : str, optional
        File code for a SEPARATE synthetic dataset to use ONLY for utility validation.
        If None, uses the main synthetic dataset for utility validation.
    target_var : str, optional
        Target variable for utility validation. If None, uses TARGET_VAR from config.
    """
    print("=" * 80)
    if age_matching:
        print("SYNTHETIC DATA VALIDATION - AGE MATCHING MODE")
        print("(Fidelity tests only with age-filtered data)")
    elif skip_align:
        print("SYNTHETIC DATA VALIDATION (NO PREPROCESSING)")
    else:
        print("SYNTHETIC DATA VALIDATION WITH AUTOMATIC PREPROCESSING")
    print("=" * 80)

    # Step 1: Load data
    print("\n" + "=" * 80)
    print("STEP 1: LOADING DATA FROM DATALAKE")
    print("=" * 80)

    try:
        real_data, synthetic_data = load_real_and_synthetic_from_datalake(
            real_file_code=FILE_CODE,
            level=LEVEL
        )
        print(f"\n✅ Data loaded successfully")
        print(f"   Real: {real_data.shape}")
        print(f"   Synthetic: {synthetic_data.shape}")

    except Exception as e:
        print(f"\n❌ Error loading data: {e}")
        sys.exit(1)

    # Step 1.5: Load SEPARATE utility synthetic dataset (if specified)
    # Resolve utility synthetic file code (CLI arg > config > None)
    utility_synth_code = utility_synthetic_file_code or UTILITY_SYNTHETIC_FILE_CODE
    utility_synthetic_data = None

    if utility_synth_code:
        print("\n" + "-" * 40)
        print("LOADING SEPARATE UTILITY SYNTHETIC DATASET")
        print("-" * 40)

        try:
            from dl_client import DatalakeClient

            client = DatalakeClient()
            print(f"  🔍 Searching for file_code='{utility_synth_code}', level='{LEVEL}'...")

            search = client.query_files(
                query={
                    'custom.level': LEVEL,
                    'custom.file_code': utility_synth_code
                }
            )

            if not search or 'object_name' not in search:
                print(f"  ⚠️  Utility synthetic file not found: {utility_synth_code}")
                print(f"     Falling back to main synthetic dataset for utility validation")
            else:
                print(f"  ⬇️  Downloading {search['object_name']}...")
                zip_files = client.download_file(
                    search['object_name'],
                    extract_zip=True
                )

                if zip_files:
                    file_name = list(zip_files.keys())[0]
                    utility_synthetic_data = zip_files[file_name]
                    print(f"  ✅ Utility synthetic data loaded: {utility_synthetic_data.shape}")
                else:
                    print(f"  ⚠️  Failed to extract utility synthetic file")

        except Exception as e:
            print(f"  ⚠️  Error loading utility synthetic data: {e}")
            print(f"     Falling back to main synthetic dataset for utility validation")
            utility_synthetic_data = None

    # Step 2: Preprocess and align (optional)
    utility_synth_aligned = None  # Will be set if utility synthetic data exists

    if skip_align:
        print("\n" + "=" * 80)
        print("STEP 2: PREPROCESSING AND ALIGNMENT - SKIPPED")
        print("=" * 80)
        print("\n⚠️  Using raw data without alignment")
        real_aligned = real_data
        synth_aligned = synthetic_data

        # Keep utility synthetic as-is if loaded
        if utility_synthetic_data is not None:
            utility_synth_aligned = utility_synthetic_data
            print(f"   Utility Synthetic (no alignment): {utility_synth_aligned.shape}")
    else:
        print("\n" + "=" * 80)
        print("STEP 2: PREPROCESSING AND ALIGNMENT")
        print("=" * 80)

        try:
            real_aligned, synth_aligned = quick_align(real_data, synthetic_data)
            print(f"\n✅ Datasets aligned successfully")
            print(f"   Real: {real_aligned.shape}")
            print(f"   Synthetic: {synth_aligned.shape}")
            print(f"   Common columns: {len(real_aligned.columns)}")

            # Also align utility synthetic data if loaded
            if utility_synthetic_data is not None:
                print(f"\n  Aligning utility synthetic dataset...")
                _, utility_synth_aligned = quick_align(real_data, utility_synthetic_data)
                print(f"   Utility Synthetic aligned: {utility_synth_aligned.shape}")

        except Exception as e:
            print(f"\n❌ Error during preprocessing: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    # Step 2.5: Age matching mode (if enabled)
    if age_matching:
        print("\n" + "=" * 80)
        print("STEP 2.5: AGE MATCHING AND BALANCING")
        print("=" * 80)

        try:
            # Filter synthetic data by age range
            print("\n📊 Filtering synthetic data by age range...")
            synth_aligned = filter_by_age_range(
                real_aligned,
                synth_aligned,
                age_column=age_column,
                tolerance_percentage=5
            )

            # Balance datasets if needed
            print("\n⚖️  Checking if dataset balancing is needed...")
            real_aligned, synth_aligned = balance_datasets_if_needed(
                real_aligned,
                synth_aligned,
                threshold_percentage=15,
                random_state=42
            )

            print(f"\n✅ Age matching completed")
            print(f"   Final Real: {real_aligned.shape}")
            print(f"   Final Synthetic: {synth_aligned.shape}")

        except Exception as e:
            print(f"\n❌ Error during age matching: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    # Step 3: Pre-validation check
    print("\n" + "=" * 80)
    print("STEP 3: PRE-VALIDATION CHECK")
    print("=" * 80)

    try:
        checker = PreValidationChecker(
            real_data=real_aligned,
            synthetic_data=synth_aligned,
            id_var=ID_VAR,
            #target_var=TARGET_VAR,
            time_var=TIME_VAR
        )

        precheck_passed = checker.run_all_checks()
        checker.save_report(PRECHECK_REPORT)

        print(f"\n📄 Pre-check report saved to: {PRECHECK_REPORT}")

    except Exception as e:
        print(f"\n❌ Error during pre-validation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Step 4: Validation (full or fidelity-only depending on mode)
    if not precheck_passed:
        print("\n" + "=" * 80)
        print("❌ PRE-VALIDATION FAILED")
        print("=" * 80)
        print(f"\nPlease review {PRECHECK_REPORT} and fix the issues.")
        print("Not proceeding to validation.")
        sys.exit(1)

    print("\n" + "=" * 80)
    if age_matching:
        print("STEP 4: FIDELITY VALIDATION (KS + CORRELATION)")
    else:
        print("STEP 4: FULL VALIDATION")
    print("=" * 80)

    try:
        # Filter categorical features to only those present in aligned data
        categorical_features = [
            col for col in CATEGORICAL_FEATURES
            if col in real_aligned.columns
        ]

        print(f"\nCategorical features found: {len(categorical_features)}/{len(CATEGORICAL_FEATURES)}")
        if len(categorical_features) < len(CATEGORICAL_FEATURES):
            missing = set(CATEGORICAL_FEATURES) - set(categorical_features)
            print(f"⚠️  Missing categorical features: {missing}")

        # Resolve target variable (CLI arg > config > None)
        final_target_var = target_var or TARGET_VAR

        # Show utility validation configuration
        if utility_synth_aligned is not None:
            print(f"\n📊 Utility Validation Configuration:")
            print(f"   Separate synthetic dataset: YES ({utility_synth_aligned.shape[0]} samples)")
            print(f"   Target variable: {final_target_var or 'Not specified (TSTR will be skipped)'}")
        elif final_target_var:
            print(f"\n📊 Utility Validation Configuration:")
            print(f"   Separate synthetic dataset: NO (using main synthetic)")
            print(f"   Target variable: {final_target_var}")

        validator = SyntheticDataValidator(
            real_data=real_aligned,
            synthetic_data=synth_aligned,
            categorical_features=categorical_features,
            id_var=ID_VAR,
            time_var=TIME_VAR,
            target_var=final_target_var,
            utility_synthetic_data=utility_synth_aligned
        )

        # Run appropriate validation based on mode
        if age_matching:
            # Age matching mode: only fidelity tests
            results = validator.run_fidelity_only_validation()
        else:
            # Normal mode: full validation
            results = validator.run_full_validation()

        validator.save_report(VALIDATION_REPORT)

        # Generate visualizations (only for full validation or correlation plots)
        if not age_matching:
            validator.generate_visualizations(output_dir=PLOTS_DIR)
            print(f"\n📄 Validation report saved to: {VALIDATION_REPORT}")
            print(f"📊 Plots saved to: {PLOTS_DIR}/")
        else:
            # For age matching mode, generate only correlation plots
            import os
            import matplotlib.pyplot as plt
            os.makedirs(PLOTS_DIR, exist_ok=True)

            if 'correlations' in validator.results and not validator.results['correlations'].get('skipped'):
                from validate_data import plot_correlation_comparison
                numeric_cols = real_aligned.select_dtypes(include=[np.number]).columns.tolist()
                # Exclude ID column
                if ID_VAR in numeric_cols:
                    numeric_cols.remove(ID_VAR)

                if len(numeric_cols) >= 2:
                    real_corr = real_aligned[numeric_cols].corr()
                    synth_corr = synth_aligned[numeric_cols].corr()

                    fig = plot_correlation_comparison(
                        real_corr, synth_corr,
                        save_path=os.path.join(PLOTS_DIR, 'correlation_comparison.png')
                    )
                    plt.close(fig)

            print(f"\n📄 Validation report saved to: {VALIDATION_REPORT}")
            print(f"📊 Correlation plot saved to: {PLOTS_DIR}/correlation_comparison.png")

    except Exception as e:
        print(f"\n❌ Error during validation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Final summary
    print("\n" + "=" * 80)
    print("✅ VALIDATION COMPLETE")
    print("=" * 80)

    print(f"\n📋 Generated files:")
    print(f"   • {PRECHECK_REPORT}")
    print(f"   • {VALIDATION_REPORT}")
    if age_matching:
        print(f"   • {PLOTS_DIR}/correlation/correlation_comparison.png")
    else:
        print(f"   • {PLOTS_DIR}/similarity/maximum_similarity_test.png")
        print(f"   • {PLOTS_DIR}/correlation/correlation_comparison.png")
        print(f"   • {PLOTS_DIR}/ks_tests/ks_test_*.png (per variable)")
        print(f"   • {PLOTS_DIR}/stationarity/stationarity_*.png (per variable)")
        print(f"   • {PLOTS_DIR}/stationarity/stationarity_*_summary.txt")
        print(f"   • {PLOTS_DIR}/acf_pacf/ar_ma_orders_*_summary.txt (per variable)")

    print(f"\n🎯 Next steps:")
    print(f"   1. Review {VALIDATION_REPORT} for detailed metrics")
    if age_matching:
        print(f"   2. Check correlation plot in {PLOTS_DIR}/correlation/ for visual analysis")
        print(f"   3. If fidelity tests passed: synthetic data has good statistical quality!")
    else:
        print(f"   2. Check plots in {PLOTS_DIR}/ subdirectories for visual analysis:")
        print(f"      - ks_tests/: Distribution comparisons")
        print(f"      - correlation/: Correlation matrices")
        print(f"      - similarity/: Privacy metrics")
        print(f"      - stationarity/: Disease progression tests")
        print(f"      - acf_pacf/: AR/MA order analysis (temporal structure)")
        print(f"   3. If all tests passed: synthetic data is ready to use!")

    print("\n" + "=" * 80)


if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Validate synthetic data with optional preprocessing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python validate_with_preprocessing.py                     # With preprocessing (default)
  python validate_with_preprocessing.py --skip-align        # Skip preprocessing
  python validate_with_preprocessing.py --age-matching      # Age-matching mode (fidelity only)
  python validate_with_preprocessing.py --age-matching --age-column AGE  # Custom age column

  # With separate utility synthetic dataset:
  python validate_with_preprocessing.py --utility-synthetic ADNIMERGEsynthetic_v2 --target DX
  python validate_with_preprocessing.py --utility-synthetic MyOtherSynthetic --target MMSE_category
        """
    )
    parser.add_argument(
        '--skip-align',
        action='store_true',
        help='Skip preprocessing/alignment step (use raw data as-is)'
    )
    parser.add_argument(
        '--age-matching',
        action='store_true',
        help='Enable age-matching mode: filters synthetic data by age, balances datasets, runs only fidelity tests (KS + correlation)'
    )
    parser.add_argument(
        '--age-column',
        type=str,
        default='AGE',
        help='Name of the age column (default: AGE). Only used with --age-matching'
    )
    parser.add_argument(
        '--utility-synthetic',
        type=str,
        default=None,
        dest='utility_synthetic_file_code',
        help='File code for a SEPARATE synthetic dataset to use ONLY for utility validation (TSTR test). '
             'If not specified, uses the main synthetic dataset.'
    )
    parser.add_argument(
        '--target',
        type=str,
        default=None,
        dest='target_var',
        help='Target variable for utility validation (TSTR test). '
             'Must be a categorical variable with <20 unique values. '
             'Examples: DX, DIAGNOSIS, MMSE_category'
    )

    args = parser.parse_args()

    # Run main workflow
    main(
        skip_align=args.skip_align,
        age_matching=args.age_matching,
        age_column=args.age_column,
        utility_synthetic_file_code=args.utility_synthetic_file_code,
        target_var=args.target_var
    )
