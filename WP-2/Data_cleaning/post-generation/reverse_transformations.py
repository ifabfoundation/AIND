"""
Entry point script for reversing data transformations (denormalization and dummy-to-categorical conversion).

This script provides a unified pipeline that orchestrates:
1. Denormalize numerical features (volumes, clinical scales) - via denormalize module
2. Convert dummy variables back to categorical - via dummy_to_categorical module
3. Decode binary/label encoded variables - via dummy_to_categorical module

Author: IFAB
Date: 2025-10-13
"""

import pandas as pd
import json
from pathlib import Path

# Import denormalization functions
from denormalize import (
    denormalize_fixed_scale,
    denormalize_dataset_minmax,
    denormalize_dataset_volumes,
    load_normalization_settings
)

# Import dummy-to-categorical conversion functions
from dummy_to_categorical import (
    dummies_to_categorical,
    multiple_dummies_to_categorical,
    auto_detect_dummies,
    label_decode,
    binary_to_categorical
)


# ============================================================================
# UNIFIED PIPELINE FUNCTION
# ============================================================================

def reverse_all_transformations(df,
                                normalization_settings_path=None,
                                volume_settings_path=None,
                                dummy_configs=None,
                                label_mappings=None,
                                separator='/',
                                skip_denormalization=False):
    """
    Complete pipeline to reverse all transformations: denormalization and dummy-to-categorical.

    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe with transformed data
    normalization_settings_path : str, optional
        Path to JSON file with normalization settings for clinical scales
    volume_settings_path : str, optional
        Path to JSON file with volume normalization settings
    dummy_configs : list of dict, optional
        Configurations for dummy variable conversion
        Example: [{'prefix': 'SEX', 'separator': '/'}, {'prefix': 'DX', 'separator': '/'}]
    label_mappings : dict, optional
        Mappings for label decoding
        Example: {'DX': {0: 'CN', 1: 'MCI', 2: 'AD'}}
    separator : str, default='/'
        Default separator for dummy variables
    skip_denormalization : bool, default=False
        If True, skip denormalization steps (clinical scales and volumes)

    Returns:
    --------
    pandas.DataFrame
        The dataframe with all transformations reversed
    """
    df_result = df.copy()

    print("="*80)
    print("Starting reverse transformation pipeline...")
    print("="*80)

    # Step 1: Denormalize clinical scales
    if skip_denormalization:
        print("\n[1/4] Skipping clinical scales denormalization (skip_denormalization=True)")
    elif normalization_settings_path:
        print("\n[1/4] Denormalizing clinical scales...")
        try:
            normalization_settings = load_normalization_settings(normalization_settings_path)

            # Find which columns exist in the dataframe
            cols_to_denorm = [col for col in normalization_settings.keys() if col in df_result.columns]

            if cols_to_denorm:
                df_result = denormalize_fixed_scale(df_result, cols_to_denorm, normalization_settings)
                print(f"   ✓ Denormalized {len(cols_to_denorm)} clinical scale columns")
            else:
                print("   ⚠ No clinical scale columns found in dataframe")
        except Exception as e:
            print(f"   ✗ Error: {e}")
    else:
        print("\n[1/4] Skipping clinical scales denormalization (no settings file provided)")

    # Step 2: Denormalize volumes
    if skip_denormalization:
        print("\n[2/4] Skipping volumes denormalization (skip_denormalization=True)")
    elif volume_settings_path:
        print("\n[2/4] Denormalizing volumes...")
        try:
            volume_settings = load_normalization_settings(volume_settings_path)

            # Find which columns exist in the dataframe
            cols_to_denorm = [col for col in volume_settings.keys() if col in df_result.columns]

            if cols_to_denorm:
                df_result = denormalize_dataset_volumes(df_result, cols_to_denorm, volume_settings)
                print(f"   ✓ Denormalized {len(cols_to_denorm)} volume columns")
            else:
                print("   ⚠ No volume columns found in dataframe")
        except Exception as e:
            print(f"   ✗ Error: {e}")
    else:
        print("\n[2/4] Skipping volumes denormalization (no settings file provided)")

    # Step 3: Convert dummies to categorical
    if dummy_configs:
        print("\n[3/4] Converting dummy variables to categorical...")
        try:
            df_result = multiple_dummies_to_categorical(df_result, dummy_configs, separator=separator)
            print(f"   ✓ Converted {len(dummy_configs)} dummy variable sets to categorical")
        except Exception as e:
            print(f"   ✗ Error: {e}")
    else:
        print("\n[3/4] Skipping dummy-to-categorical conversion (no configurations provided)")

    # Step 4: Decode label encoded columns
    if label_mappings:
        print("\n[4/4] Decoding label-encoded columns...")
        try:
            for col, mapping in label_mappings.items():
                if col in df_result.columns:
                    df_result = label_decode(df_result, col, mapping)
                    print(f"   ✓ Decoded column '{col}'")
        except Exception as e:
            print(f"   ✗ Error: {e}")
    else:
        print("\n[4/4] Skipping label decoding (no mappings provided)")

    print("\n" + "="*80)
    print("Reverse transformation pipeline completed!")
    print("="*80)

    return df_result


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("REVERSE TRANSFORMATIONS - Example Usage")
    print("="*80)

    # Get current directory
    current_dir = Path(__file__).parent

    # Define paths to settings files
    normalization_settings_path = current_dir / "normalization_settings.json"
    volume_settings_path = current_dir / "volume_values_settings.json"

    # Create sample transformed data
    print("\nCreating sample transformed data...")
    df_transformed = pd.DataFrame({
        'RID': [1, 2, 3, 4, 5],
        'AGE': [65, 70, 75, 68, 72],
        # Normalized clinical scales [0, 1]
        'MMSE': [0.5, 0.7, 0.3, 0.9, 0.6],
        'FAQ': [0.2, 0.4, 0.8, 0.1, 0.5],
        'MOCA': [0.6, 0.8, 0.4, 0.9, 0.7],
        # Normalized volumes [0, 1]
        'Hippocampus%ICV': [0.7, 0.5, 0.3, 0.8, 0.6],
        'Ventricles%ICV': [0.2, 0.4, 0.7, 0.1, 0.5],
        # Dummy variables
        'SEX/M': [1, 0, 1, 0, 1],
        'SEX/F': [0, 1, 0, 1, 0],
        'DX/CN': [1, 0, 0, 1, 0],
        'DX/MCI': [0, 1, 0, 0, 1],
        'DX/AD': [0, 0, 1, 0, 0],
        'APOE4/0': [1, 0, 0, 1, 0],
        'APOE4/1': [0, 1, 0, 0, 1],
        'APOE4/2': [0, 0, 1, 0, 0]
    })

    print("\nTransformed data (normalized + dummies):")
    print(df_transformed.head())
    print(f"\nShape: {df_transformed.shape}")
    print(f"Columns: {list(df_transformed.columns)}")

    # Configure dummy variable conversion
    dummy_configs = [
        {'prefix': 'SEX', 'separator': '/'},
        {'prefix': 'DX', 'separator': '/', 'original_name': 'DX'},
        {'prefix': 'APOE4', 'separator': '/'}
    ]

    # Reverse all transformations
    print("\n" + "="*80)

    df_original = reverse_all_transformations(
        df_transformed,
        normalization_settings_path=str(normalization_settings_path) if normalization_settings_path.exists() else None,
        volume_settings_path=str(volume_settings_path) if volume_settings_path.exists() else None,
        dummy_configs=dummy_configs,
        separator='/'
    )

    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print("\nOriginal data (after reverse transformations):")
    print(df_original.head())
    print(f"\nShape: {df_original.shape}")
    print(f"Columns: {list(df_original.columns)}")

    # Save example output
    output_path = current_dir / "example_reversed_data.csv"
    df_original.to_csv(output_path, index=False)
    print(f"\n✓ Example output saved to: {output_path}")

    print("\n" + "="*80)
    print("AUTO-DETECTION EXAMPLE")
    print("="*80)

    # Demonstrate auto-detection of dummy variables
    dummy_groups = auto_detect_dummies(df_transformed, separator='/')
    print("\nAuto-detected dummy variable groups:")
    for prefix, columns in dummy_groups.items():
        print(f"  {prefix}: {columns}")
