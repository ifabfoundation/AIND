"""
Preprocessing Functions for Synthetic Data
===========================================

Functions to align synthetic data with real data before validation.

Common issues:
- Synthetic data has 'generated_' prefix on cofactor columns
- Synthetic data has extra 'generation_id' column
- Column order may differ

Author: IFAB
Date: 2025
"""

import pandas as pd
import numpy as np
from typing import List, Tuple


def align_synthetic_with_real(real_data: pd.DataFrame,
                               synthetic_data: pd.DataFrame,
                               remove_prefix: str = 'generated_',
                               drop_columns: List[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Align synthetic dataset with real dataset for validation.

    This function:
    1. Removes specified prefix from synthetic column names
    2. Drops specified columns from synthetic data
    3. Reorders synthetic columns to match real data
    4. Ensures both datasets have the same columns

    Parameters
    ----------
    real_data : pd.DataFrame
        Real dataset
    synthetic_data : pd.DataFrame
        Synthetic dataset (may have prefixes and extra columns)
    remove_prefix : str, default='generated_'
        Prefix to remove from synthetic column names
    drop_columns : List[str], optional
        Columns to drop from synthetic data (e.g., ['generation_id'])

    Returns
    -------
    tuple of (pd.DataFrame, pd.DataFrame)
        (aligned_real, aligned_synthetic) with matching schemas

    Examples
    --------
    >>> # Real columns: ['ID', 'TIME', 'RACE/Asian', 'GENDER/female', ...]
    >>> # Synth columns: ['ID', 'TIME', 'generated_RACE/Asian', 'generated_GENDER/female', 'generation_id', ...]
    >>> real_aligned, synth_aligned = align_synthetic_with_real(real_data, synthetic_data,
    ...                                                          remove_prefix='generated_',
    ...                                                          drop_columns=['generation_id'])
    >>> # Both now have: ['ID', 'TIME', 'RACE/Asian', 'GENDER/female', ...]
    """
    print("=" * 80)
    print("ALIGNING SYNTHETIC DATA WITH REAL DATA")
    print("=" * 80)

    # Make copies to avoid modifying originals
    real_aligned = real_data.copy()
    synth_aligned = synthetic_data.copy()

    print(f"\nOriginal shapes:")
    print(f"  Real: {real_aligned.shape}")
    print(f"  Synthetic: {synth_aligned.shape}")

    # STEP 1: Remove prefix from synthetic column names
    if remove_prefix:
        print(f"\nStep 1: Removing prefix '{remove_prefix}' from synthetic columns...")

        renamed_cols = {}
        for col in synth_aligned.columns:
            if col.startswith(remove_prefix):
                new_name = col[len(remove_prefix):]  # Remove prefix
                renamed_cols[col] = new_name
                print(f"  '{col}' → '{new_name}'")

        if renamed_cols:
            synth_aligned = synth_aligned.rename(columns=renamed_cols)
            print(f"  ✅ Renamed {len(renamed_cols)} columns")
        else:
            print(f"  ℹ️  No columns with prefix '{remove_prefix}' found")

    # STEP 2: Drop specified columns from synthetic data
    if drop_columns is None:
        drop_columns = []

    # Automatically detect 'generation_id' if present
    if 'generation_id' in synth_aligned.columns and 'generation_id' not in drop_columns:
        drop_columns.append('generation_id')

    if drop_columns:
        print(f"\nStep 2: Dropping extra columns from synthetic data...")

        columns_to_drop = [col for col in drop_columns if col in synth_aligned.columns]

        if columns_to_drop:
            synth_aligned = synth_aligned.drop(columns=columns_to_drop)
            print(f"  Dropped: {columns_to_drop}")
            print(f"  ✅ Dropped {len(columns_to_drop)} columns")
        else:
            print(f"  ℹ️  No columns to drop")

    # STEP 3: Check column alignment
    print(f"\nStep 3: Checking column alignment...")

    real_cols = set(real_aligned.columns)
    synth_cols = set(synth_aligned.columns)

    common_cols = real_cols & synth_cols
    only_in_real = real_cols - synth_cols
    only_in_synth = synth_cols - real_cols

    print(f"  Common columns: {len(common_cols)}")

    if only_in_real:
        print(f"  ⚠️  Columns only in REAL: {only_in_real}")
        print(f"     These will be dropped from real data for validation")
        real_aligned = real_aligned.drop(columns=list(only_in_real))

    if only_in_synth:
        print(f"  ⚠️  Columns only in SYNTHETIC: {only_in_synth}")
        print(f"     These will be dropped from synthetic data for validation")
        synth_aligned = synth_aligned.drop(columns=list(only_in_synth))

    # STEP 4: Reorder synthetic columns to match real data
    print(f"\nStep 4: Reordering columns to match...")

    # Get final common columns in the order they appear in real data
    final_cols = [col for col in real_aligned.columns if col in synth_aligned.columns]

    real_aligned = real_aligned[final_cols]
    synth_aligned = synth_aligned[final_cols]

    print(f"  ✅ Both datasets now have {len(final_cols)} columns in the same order")

    # STEP 5: Summary
    print("\n" + "=" * 80)
    print("ALIGNMENT COMPLETE")
    print("=" * 80)
    print(f"\nFinal shapes:")
    print(f"  Real: {real_aligned.shape}")
    print(f"  Synthetic: {synth_aligned.shape}")
    print(f"\nColumns: {list(final_cols)[:10]}{'...' if len(final_cols) > 10 else ''}")
    print(f"\n✅ Datasets are now aligned and ready for validation!")
    print("=" * 80)

    return real_aligned, synth_aligned


def quick_align(real_data: pd.DataFrame,
                synthetic_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Quick alignment with default settings for AIND project.

    Assumes:
    - Synthetic columns have 'generated_' prefix for cofactors
    - Synthetic data has 'generation_id' column to remove

    Parameters
    ----------
    real_data : pd.DataFrame
        Real dataset
    synthetic_data : pd.DataFrame
        Synthetic dataset

    Returns
    -------
    tuple of (pd.DataFrame, pd.DataFrame)
        (aligned_real, aligned_synthetic)

    Examples
    --------
    >>> from preprocess_synthetic import quick_align
    >>> real_aligned, synth_aligned = quick_align(real_data, synthetic_data)
    """
    return align_synthetic_with_real(
        real_data=real_data,
        synthetic_data=synthetic_data,
        remove_prefix='generated_',
        drop_columns=['generation_id']
    )


# ============================================================================
# INTEGRATION WITH VALIDATION SCRIPTS
# ============================================================================

def preprocess_and_validate(real_data: pd.DataFrame,
                            synthetic_data: pd.DataFrame,
                            run_precheck: bool = True,
                            run_full_validation: bool = True,
                            **validator_kwargs):
    """
    Complete workflow: preprocess + validate.

    Parameters
    ----------
    real_data : pd.DataFrame
        Real dataset
    synthetic_data : pd.DataFrame
        Synthetic dataset
    run_precheck : bool, default=True
        Whether to run pre-validation checks
    run_full_validation : bool, default=True
        Whether to run full validation
    **validator_kwargs
        Additional arguments passed to validators (e.g., id_var, target_var)

    Returns
    -------
    dict
        Validation results

    Examples
    --------
    >>> from preprocess_synthetic import preprocess_and_validate
    >>> results = preprocess_and_validate(
    ...     real_data, synthetic_data,
    ...     id_var='ID', target_var='CDRSB',
    ...     categorical_features=['APOE4', 'GENDER/female']
    ... )
    """
    from pre_validation_check import PreValidationChecker
    from validate_data import SyntheticDataValidator

    # STEP 1: Align datasets
    print("\n🔧 STEP 1: Preprocessing and Alignment")
    real_aligned, synth_aligned = quick_align(real_data, synthetic_data)

    results = {
        'preprocessing': {
            'real_shape_before': real_data.shape,
            'synth_shape_before': synthetic_data.shape,
            'real_shape_after': real_aligned.shape,
            'synth_shape_after': synth_aligned.shape,
            'columns_aligned': list(real_aligned.columns)
        }
    }

    # STEP 2: Pre-validation check
    if run_precheck:
        print("\n🔍 STEP 2: Pre-Validation Check")
        checker = PreValidationChecker(
            real_data=real_aligned,
            synthetic_data=synth_aligned,
            **{k: v for k, v in validator_kwargs.items()
               if k in ['id_var', 'target_var', 'time_var']}
        )

        precheck_passed = checker.run_all_checks()
        checker.save_report('pre_validation_report.txt')

        results['precheck_passed'] = precheck_passed

        if not precheck_passed:
            print("\n❌ Pre-validation failed. Not proceeding to full validation.")
            return results

    # STEP 3: Full validation
    if run_full_validation:
        print("\n✅ STEP 3: Full Validation")
        validator = SyntheticDataValidator(
            real_data=real_aligned,
            synthetic_data=synth_aligned,
            **validator_kwargs
        )

        validation_results = validator.run_full_validation()
        validator.save_report('validation_report.json')
        validator.generate_visualizations(output_dir='validation_plots')

        results['validation'] = validation_results

    return results


if __name__ == '__main__':
    """
    Example usage when run as script.
    """
    print(__doc__)
    print("\nThis is a utility module. Import it in your scripts:")
    print("\nExample:")
    print("  from preprocess_synthetic import quick_align")
    print("  real_aligned, synth_aligned = quick_align(real_data, synthetic_data)")
