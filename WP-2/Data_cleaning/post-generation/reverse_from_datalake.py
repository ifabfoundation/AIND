"""
Script to load synthetic/generated data from datalake and apply reverse transformations.

This script combines:
1. Data loading from datalake using DatalakeClient
2. Reverse transformations (denormalization + dummy-to-categorical conversion)
3. Upload results back to datalake

Usage:
    python reverse_from_datalake.py --file-code ADNIMERGE_synthetic --prefix synthetic/reversed
"""

import pandas as pd
import argparse
import json
from pathlib import Path
from dl_client import DatalakeClient

# Import reverse transformation functions
from reverse_transformations import reverse_all_transformations
from dummy_to_categorical import auto_detect_dummies


def load_from_datalake(file_code, query_params=None):
    """
    Load a dataset from the datalake using DatalakeClient.

    Parameters:
    -----------
    file_code : str
        The file code to search for (e.g., 'ADNIMERGE_synthetic')
    query_params : dict, optional
        Additional query parameters (e.g., {'custom.level': 'cleaned_03'})

    Returns:
    --------
    tuple
        (pandas.DataFrame, str, str): dataset, object_name, file_name
    """
    print(f"Initializing DatalakeClient...")
    client = DatalakeClient()

    # Build query
    query = {'custom.file_code': file_code}
    if query_params:
        query.update(query_params)

    print(f"Searching for file with code: {file_code}")
    print(f"Query: {query}")

    # Search for file
    search = client.query_files(query=query)

    if not search:
        raise ValueError(f"No file found with code '{file_code}'")

    object_name = search.get('object_name')
    print(f"Found file: {object_name}")

    # Download file
    print("Downloading file...")
    zip_files = client.download_file(
        object_name,
        extract_zip=True
    )

    # Get the first file from the zip
    file_name = list(zip_files.keys())[0]
    dataset = zip_files[file_name]

    print(f"Loaded file: {file_name}")
    print(f"Dataset shape: {dataset.shape}")
    print(f"Columns: {list(dataset.columns)}")

    return dataset, object_name, file_name


def get_file_metadata(object_name):
    """
    Get metadata from a file in the datalake.

    Parameters:
    -----------
    object_name : str
        The object name in the datalake

    Returns:
    --------
    dict
        Custom metadata dictionary
    """
    print(f"Retrieving metadata for: {object_name}")
    client = DatalakeClient()

    metadata = client.get_metadata(object_name=object_name)
    metadata_custom = metadata['metadata']['custom']

    print(f"Retrieved metadata: {metadata_custom}")
    return metadata_custom


def upload_to_datalake(df, file_name, prefix, metadata):
    """
    Upload a dataframe to the datalake.

    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe to upload
    file_name : str
        The name of the file (with _reversed suffix)
    prefix : str
        The prefix for the datalake (e.g., 'synthetic/reversed', 'validation/reversed')
    metadata : dict
        Custom metadata dictionary

    Returns:
    --------
    bool
        True if upload successful, False otherwise
    """
    print(f"\nUploading to datalake...")
    print(f"  File name: {file_name}")
    print(f"  Prefix: {prefix}")
    print(f"  Metadata: {metadata}")

    client = DatalakeClient()

    try:
        search = client.upload_dataframe(
            df=df,
            object_name=file_name,
            prefix=prefix,
            metadata=metadata
        )

        print(f"   ✓ Upload successful!")
        return True
    except Exception as e:
        print(f'   ✗ Upload on datalake failed: {e}')
        return False


def reverse_datalake_data(file_code,
                          upload_to_dl=True,
                          prefix='synthetic/reversed',
                          save_local=False,
                          output_path=None,
                          normalization_settings_path=None,
                          volume_settings_path=None,
                          dummy_configs=None,
                          label_mappings=None,
                          separator='/',
                          skip_denormalization=False,
                          query_params=None):
    """
    Complete pipeline: load data from datalake, apply reverse transformations, and upload back.

    Parameters:
    -----------
    file_code : str
        The file code to search for in the datalake
    upload_to_dl : bool, default=True
        Whether to upload the result to datalake
    prefix : str, default='synthetic/reversed'
        Prefix for datalake upload (e.g., 'synthetic/reversed', 'validation/reversed')
    save_local : bool, default=False
        Whether to save a local copy
    output_path : str, optional
        Path to save the reversed dataset locally (CSV format)
    normalization_settings_path : str, optional
        Path to JSON file with normalization settings for clinical scales
    volume_settings_path : str, optional
        Path to JSON file with volume normalization settings
    dummy_configs : list of dict, optional
        Configurations for dummy variable conversion.
        If None, dummy variables will be auto-detected (columns with separator in name and binary values 0/1)
    label_mappings : dict, optional
        Mappings for label decoding
    separator : str, default='/'
        Default separator for dummy variables
    skip_denormalization : bool, default=False
        If True, skip denormalization steps (clinical scales and volumes)
    query_params : dict, optional
        Additional query parameters for datalake search

    Returns:
    --------
    pandas.DataFrame
        The dataset with all transformations reversed
    """
    print("="*80)
    print("REVERSE TRANSFORMATIONS FROM DATALAKE")
    print("="*80)

    # Load data from datalake
    print("\n[STEP 1] Loading data from datalake...")
    df, object_name, original_file_name = load_from_datalake(file_code, query_params)

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

    # Apply reverse transformations
    print("\n[STEP 2] Applying reverse transformations...")
    df_reversed = reverse_all_transformations(
        df,
        normalization_settings_path=normalization_settings_path,
        volume_settings_path=volume_settings_path,
        dummy_configs=dummy_configs,
        label_mappings=label_mappings,
        separator=separator,
        skip_denormalization=skip_denormalization
    )

    # Filter out records with invalid TIME values
    print("\n[STEP 2.5] Filtering records with invalid TIME values...")
    if 'TIME' in df_reversed.columns:
        initial_count = len(df_reversed)
        df_reversed = df_reversed[(df_reversed['TIME'] >= 0) & (df_reversed['TIME'] <= 100)]
        removed_count = initial_count - len(df_reversed)

        if removed_count > 0:
            print(f"   ⚠ Removed {removed_count} records with TIME outside [0, 100] range")
            print(f"   ✓ Remaining records: {len(df_reversed)}")
        else:
            print(f"   ✓ All records have valid TIME values (within [0, 100])")
    else:
        print(f"   ⚠ TIME column not found, skipping filter")

    # Upload to datalake
    if upload_to_dl:
        print("\n[STEP 3] Uploading reversed dataset to datalake...")

        # Search again with search_files to get correct object_name for metadata
        client = DatalakeClient()
        query = {'custom.file_code': file_code}
        if query_params:
            query.update(query_params)

        search_result = client.search_files(query=query)
        if not search_result or 'files' not in search_result or len(search_result['files']) == 0:
            raise ValueError(f"No file found with code '{file_code}' for metadata retrieval")

        metadata_object_name = search_result['files'][0]['object_name']

        # Get original metadata using the correct object_name from search_files
        original_metadata = get_file_metadata(metadata_object_name)

        # Create new metadata with modified file_code
        new_metadata = original_metadata.copy()
        original_file_code = new_metadata.get('file_code', file_code)
        new_metadata['file_code'] = f"{original_file_code}_reversed"

        # Create new file name with _reversed suffix
        file_name_parts = original_file_name.rsplit('.', 1)
        if len(file_name_parts) == 2:
            new_file_name = f"{file_name_parts[0]}_reversed.{file_name_parts[1]}"
        else:
            new_file_name = f"{original_file_name}_reversed"

        # Upload to datalake
        upload_success = upload_to_datalake(
            df=df_reversed,
            file_name=new_file_name,
            prefix=prefix,
            metadata=new_metadata
        )

        if not upload_success:
            print("   ⚠ Warning: Upload to datalake failed!")

    # Save local copy if requested
    if save_local and output_path:
        print(f"\n[STEP 4] Saving local copy to: {output_path}")
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
        description='Load data from datalake, apply reverse transformations, and upload back'
    )

    parser.add_argument(
        '--file-code',
        type=str,
        required=True,
        help='File code to search in datalake (e.g., ADNIMERGE_synthetic)'
    )

    parser.add_argument(
        '--prefix',
        type=str,
        default='synthetic/reversed',
        choices=['synthetic/reversed', 'validation/reversed', 'synthetic/utility/reversed', 'validation/utility/reversed'],
        help='Datalake prefix for upload (default: synthetic/reversed)'
    )

    parser.add_argument(
        '--no-upload',
        action='store_true',
        help='Do not upload to datalake'
    )

    parser.add_argument(
        '--save-local',
        action='store_true',
        help='Save a local copy of the reversed data'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output CSV file path for local copy (requires --save-local)'
    )

    parser.add_argument(
        '--normalization-settings',
        type=str,
        default='normalization_settings.json',
        help='Path to normalization settings JSON file'
    )

    parser.add_argument(
        '--volume-settings',
        type=str,
        default='volume_values_settings.json',
        help='Path to volume settings JSON file'
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

    parser.add_argument(
        '--query-param',
        type=str,
        action='append',
        help='Additional query parameters in format key=value (can be used multiple times)'
    )

    parser.add_argument(
        '--skip-denormalization',
        action='store_true',
        help='Skip denormalization steps (only convert dummies to categorical)'
    )

    args = parser.parse_args()

    # Get current directory
    current_dir = Path(__file__).parent

    # Build paths for settings files
    norm_settings_path = current_dir / args.normalization_settings
    vol_settings_path = current_dir / args.volume_settings

    # Build dummy configs (None triggers auto-detection)
    dummy_configs = None
    if args.dummy_prefixes:
        dummy_configs = [
            {'prefix': dummy_prefix, 'separator': args.dummy_separator}
            for dummy_prefix in args.dummy_prefixes
        ]

    # Parse additional query parameters
    query_params = {}
    if args.query_param:
        for param in args.query_param:
            key, value = param.split('=', 1)
            query_params[key] = value

    # Run pipeline
    df_reversed = reverse_datalake_data(
        file_code=args.file_code,
        upload_to_dl=not args.no_upload,
        prefix=args.prefix,
        save_local=args.save_local,
        output_path=args.output,
        normalization_settings_path=str(norm_settings_path) if norm_settings_path.exists() else None,
        volume_settings_path=str(vol_settings_path) if vol_settings_path.exists() else None,
        dummy_configs=dummy_configs,
        separator=args.dummy_separator,
        skip_denormalization=args.skip_denormalization,
        query_params=query_params if query_params else None
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
        print("\nCommand line (upload to datalake with auto-detect dummies):")
        print("  python reverse_from_datalake.py --file-code ADNIMERGE_synthetic --prefix synthetic/reversed")
        print("\nWith local copy:")
        print("  python reverse_from_datalake.py --file-code ADNIMERGE_synthetic --prefix synthetic/reversed --save-local --output reversed_data.csv")
        print("\nWithout datalake upload (local only):")
        print("  python reverse_from_datalake.py --file-code ADNIMERGE_synthetic --no-upload --save-local --output reversed_data.csv")
        print("\nWith validation prefix:")
        print("  python reverse_from_datalake.py --file-code ADNIMERGE_validation --prefix validation/reversed")
        print("\nWith additional query parameters:")
        print("  python reverse_from_datalake.py --file-code ADNIMERGE_synthetic --query-param custom.level=cleaned_03 --prefix synthetic/reversed")
        print("\nWith custom dummy prefixes (override auto-detection):")
        print("  python reverse_from_datalake.py --file-code ADNIMERGE_synthetic --dummy-prefixes SEX DX APOE4 EDUCATION --prefix synthetic/reversed")
        print("\nSkip denormalization (only convert dummies):")
        print("  python reverse_from_datalake.py --file-code ADNIMERGE_synthetic --skip-denormalization --prefix synthetic/reversed")
        print("\n" + "="*80)
        print("\nProgrammatic usage:")
        print("="*80)
        print("""
from reverse_from_datalake import reverse_datalake_data

# Upload to datalake with auto-detect dummies (default)
df = reverse_datalake_data(
    file_code='ADNIMERGE_synthetic',
    prefix='synthetic/reversed'
)

# With explicit dummy configs (override auto-detection)
dummy_configs = [
    {'prefix': 'SEX', 'separator': '/'},
    {'prefix': 'DX', 'separator': '/'},
    {'prefix': 'APOE4', 'separator': '/'}
]

df = reverse_datalake_data(
    file_code='ADNIMERGE_synthetic',
    upload_to_dl=True,
    prefix='synthetic/reversed',
    save_local=True,
    output_path='reversed_data.csv',
    dummy_configs=dummy_configs,
    separator='/'
)

# Validation data with auto-detect
df = reverse_datalake_data(
    file_code='ADNIMERGE_validation',
    prefix='validation/reversed'
)

# Skip denormalization (only convert dummies)
df = reverse_datalake_data(
    file_code='ADNIMERGE_synthetic',
    skip_denormalization=True,
    prefix='synthetic/reversed'
)
        """)
        print("="*80)
    else:
        # Run with command-line arguments
        main()
