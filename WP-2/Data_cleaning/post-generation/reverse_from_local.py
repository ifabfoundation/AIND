"""
Script to load synthetic/generated data from local CSV files and apply reverse transformations.

This script combines:
1. Data loading from local CSV files
2. Reverse transformations (denormalization + dummy-to-categorical conversion)
3. Save results locally

Usage:
    python reverse_from_local.py --input-file synthetic_data.csv --output reversed_data.csv
"""

import pandas as pd
import argparse
from pathlib import Path

# Import reverse transformation functions
from reverse_transformations import reverse_all_transformations
from dummy_to_categorical import auto_detect_dummies


def load_from_csv(file_path):
    """
    Load a dataset from a local CSV file.

    Parameters:
    -----------
    file_path : str or Path
        Path to the CSV file to load

    Returns:
    --------
    pandas.DataFrame
        The loaded dataset
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    print(f"Loading data from: {file_path}")
    df = pd.read_csv(file_path)

    print(f"Loaded successfully!")
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    return df


def reverse_local_data(input_path,
                       output_path,
                       normalization_settings_path=None,
                       dummy_configs=None,
                       label_mappings=None,
                       separator='/'):
    """
    Complete pipeline: load data from CSV, apply reverse transformations, and save locally.

    Parameters:
    -----------
    input_path : str
        Path to the input CSV file
    output_path : str
        Path to save the reversed dataset (CSV format)
    normalization_settings_path : str, optional
        Path to JSON file with normalization settings for clinical scales
    dummy_configs : list of dict, optional
        Configurations for dummy variable conversion.
        If None, dummy variables will be auto-detected (columns with separator in name and binary values 0/1)
    label_mappings : dict, optional
        Mappings for label decoding
    separator : str, default='/'
        Default separator for dummy variables

    Returns:
    --------
    pandas.DataFrame
        The dataset with all transformations reversed
    """
    print("="*80)
    print("REVERSE TRANSFORMATIONS FROM LOCAL CSV")
    print("="*80)

    # Load data from CSV
    print("\n[STEP 1] Loading data from local CSV file...")
    df = load_from_csv(input_path)

    # Auto-detect dummy variables if dummy_configs not provided
    if dummy_configs is None:
        print("\n[STEP 1.5] Auto-detecting dummy variables...")
        dummy_groups = auto_detect_dummies(df, separator=separator)

        if dummy_groups:
            print(f"   ✓ Auto-detected {len(dummy_groups)} dummy variable groups:")
            for dummy_prefix, columns in dummy_groups.items():
                print(f"     - {dummy_prefix}: {len(columns)} columns")

            # Create dummy_configs from auto-detected groups
            dummy_configs = [
                {'prefix': dummy_prefix, 'separator': separator}
                for dummy_prefix in dummy_groups.keys()
            ]
        else:
            print("   ⚠ No dummy variables auto-detected")
            dummy_configs = []

    # Apply reverse transformations (without volume settings)
    print("\n[STEP 2] Applying reverse transformations...")
    df_reversed = reverse_all_transformations(
        df,
        normalization_settings_path=normalization_settings_path,
        volume_settings_path=None,  # Volume settings removed
        dummy_configs=dummy_configs,
        label_mappings=label_mappings,
        separator=separator
    )

    # Save to local file
    print(f"\n[STEP 3] Saving reversed dataset to: {output_path}")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_reversed.to_csv(output_path, index=False)
    print(f"   ✓ Saved successfully")

    print("\n" + "="*80)
    print("PIPELINE COMPLETED")
    print("="*80)
    print(f"Final dataset shape: {df_reversed.shape}")
    print(f"Final columns: {list(df_reversed.columns)}")

    return df_reversed


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description='Load data from local CSV, apply reverse transformations, and save back'
    )

    parser.add_argument(
        '--input-file',
        type=str,
        required=True,
        help='Input CSV file path'
    )

    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output CSV file path for reversed data'
    )

    parser.add_argument(
        '--normalization-settings',
        type=str,
        default='normalization_settings.json',
        help='Path to normalization settings JSON file'
    )

    parser.add_argument(
        '--dummy-separator',
        type=str,
        default='/',
        help='Separator used in dummy variables (default: /)'
    )

    parser.add_argument(
        '--dummy-prefixes',
        type=str,
        nargs='+',
        default=None,
        help='Dummy variable prefixes to convert (default: auto-detect)'
    )

    args = parser.parse_args()

    # Get current directory
    current_dir = Path(__file__).parent

    # Build path for normalization settings file
    norm_settings_path = current_dir / args.normalization_settings

    # Build dummy configs (None triggers auto-detection)
    dummy_configs = None
    if args.dummy_prefixes:
        dummy_configs = [
            {'prefix': dummy_prefix, 'separator': args.dummy_separator}
            for dummy_prefix in args.dummy_prefixes
        ]

    # Run pipeline
    df_reversed = reverse_local_data(
        input_path=args.input_file,
        output_path=args.output,
        normalization_settings_path=str(norm_settings_path) if norm_settings_path.exists() else None,
        dummy_configs=dummy_configs,
        separator=args.dummy_separator
    )

    return df_reversed


if __name__ == "__main__":
    # Example usage when run directly
    import sys

    if len(sys.argv) == 1:
        # No arguments provided, show example
        print("="*80)
        print("EXAMPLE USAGE")
        print("="*80)
        print("\nCommand line (with auto-detect dummies):")
        print("  python reverse_from_local.py --input-file synthetic_data.csv --output reversed_data.csv")
        print("\nWith custom normalization settings:")
        print("  python reverse_from_local.py --input-file synthetic_data.csv --output reversed_data.csv --normalization-settings my_settings.json")
        print("\nWith custom dummy prefixes (override auto-detection):")
        print("  python reverse_from_local.py --input-file synthetic_data.csv --output reversed_data.csv --dummy-prefixes SEX DX APOE4 EDUCATION")
        print("\n" + "="*80)
        print("\nProgrammatic usage:")
        print("="*80)
        print("""
from reverse_from_local import reverse_local_data

# With auto-detect dummies (default)
df = reverse_local_data(
    input_path='synthetic_data.csv',
    output_path='reversed_data.csv'
)

# With explicit dummy configs (override auto-detection)
dummy_configs = [
    {'prefix': 'SEX', 'separator': '/'},
    {'prefix': 'DX', 'separator': '/'},
    {'prefix': 'APOE4', 'separator': '/'}
]

df = reverse_local_data(
    input_path='synthetic_data.csv',
    output_path='reversed_data.csv',
    normalization_settings_path='normalization_settings.json',
    dummy_configs=dummy_configs,
    separator='/'
)
        """)
        print("="*80)
    else:
        # Run with command-line arguments
        main()
