"""
Script to add post-generation noise to synthetic data from datalake using Differential Privacy.

This script combines:
1. Data loading from datalake using DatalakeClient (synthetic and real data)
2. Noise injection using differential privacy mechanisms (Laplace, Gaussian, Adaptive)
3. Upload results back to datalake

Based on:
- Dwork et al. (2006): Differential Privacy principles
- NIST (2021): Post-processing immunity in DP synthetic data

Usage:
    python noise_injection_from_datalake.py --synthetic-code ADNIMERGE_synthetic --real-code ADNIMERGE --epsilon 1.0 --method gaussian
"""

import pandas as pd
import numpy as np
import argparse
import json
from pathlib import Path
from typing import List, Optional
from dl_client import DatalakeClient

# Import noise injection classes
from noise_injection import (
    PostGenerationNoiseInjector,
    LongitudinalNoiseInjector,
    add_post_generation_noise
)


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
        The name of the file (with _noisy suffix)
    prefix : str
        The prefix for the datalake (e.g., 'synthetic/noisy', 'validation/noisy')
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


def inject_noise_datalake_data(
    synthetic_file_code,
    real_file_code,
    epsilon=1.0,
    delta=None,
    method='gaussian',
    target_dcr=0.50,
    current_dcr=None,
    preserve_temporal=True,
    patient_col='ID',
    time_col='TIME',
    ordinal_features=None,
    continuous_features=None,
    categorical_ordinal_features=None,
    upload_to_dl=True,
    prefix='synthetic/noisy',
    save_local=False,
    output_path=None,
    synthetic_query_params=None,
    real_query_params=None
):
    """
    Complete pipeline: load synthetic and real data from datalake, add noise, and upload back.

    Parameters:
    -----------
    synthetic_file_code : str
        File code for synthetic data in datalake
    real_file_code : str
        File code for real data in datalake
    epsilon : float, default=1.0
        Privacy budget (smaller = more privacy, more noise)
        Typical values:
        - 0.5-1.0 for high DCR (>0.80) - aggressive noise
        - 1.0-2.0 for medium DCR (0.60-0.80) - moderate noise
        - 2.0-5.0 for DCR close to target - light noise
    delta : float, optional
        Failure probability for (ε, δ)-DP
        Default: 1/n² where n is dataset size
    method : str, default='gaussian'
        Noise injection method: 'gaussian', 'laplace', or 'adaptive'
    target_dcr : float, default=0.50
        Target DCR value (typically 0.45-0.55)
    current_dcr : float, optional
        Current DCR from validation (used with 'adaptive' method)
    preserve_temporal : bool, default=True
        If True, preserve temporal correlations in longitudinal data
    patient_col : str, default='ID'
        Column name for patient ID
    time_col : str, default='TIME'
        Column name for time variable
    ordinal_features : list of str, optional
        Ordinal features to round after noise injection
    continuous_features : list of str, optional
        Continuous features (default: all numeric non-ordinal)
    categorical_ordinal_features : list of str, optional
        Categorical ordinal variables that should NOT receive noise (e.g., APOE, EDUCATION)
        If None, uses auto-detection based on unique values
    upload_to_dl : bool, default=True
        Whether to upload result to datalake
    prefix : str, default='synthetic/noisy'
        Prefix for datalake upload
    save_local : bool, default=False
        Whether to save a local copy
    output_path : str, optional
        Path to save the noisy dataset locally (CSV format)
    synthetic_query_params : dict, optional
        Additional query parameters for synthetic data search
    real_query_params : dict, optional
        Additional query parameters for real data search

    Returns:
    --------
    pandas.DataFrame
        The synthetic dataset with noise added
    """
    print("="*80)
    print("POST-GENERATION NOISE INJECTION FROM DATALAKE")
    print("="*80)

    # Load synthetic data from datalake
    print("\n[STEP 1] Loading synthetic data from datalake...")
    synth_df, synth_object_name, synth_file_name = load_from_datalake(
        synthetic_file_code, synthetic_query_params
    )

    # Load real data from datalake
    print("\n[STEP 2] Loading real data from datalake...")
    real_df, real_object_name, real_file_name = load_from_datalake(
        real_file_code, real_query_params
    )

    # Default ordinal features if not provided
    if ordinal_features is None:
        ordinal_features = []
        # Auto-detect common clinical scales
        common_scales = ['MMSE', 'CDRSB', 'FAQ', 'ADAS11', 'ADAS13', 'RAVLT']
        for scale in common_scales:
            if scale in synth_df.columns:
                ordinal_features.append(scale)

        if ordinal_features:
            print(f"\n[INFO] Auto-detected ordinal features: {ordinal_features}")

    # Apply noise injection
    print("\n[STEP 3] Applying noise injection...")
    print(f"  Method: {method}")
    print(f"  Epsilon: {epsilon}")
    print(f"  Target DCR: {target_dcr}")
    if current_dcr:
        print(f"  Current DCR: {current_dcr}")
    print(f"  Preserve temporal correlations: {preserve_temporal}")

    # Calculate delta if not provided
    if delta is None:
        delta = 1.0 / (len(real_df) ** 2)
    print(f"  Delta: {delta:.2e}")

    synth_df_noisy = add_post_generation_noise(
        synth_df=synth_df,
        real_df=real_df,
        current_dcr=current_dcr if current_dcr else target_dcr,
        target_dcr=target_dcr,
        epsilon=epsilon,
        method=method,
        preserve_temporal=preserve_temporal,
        patient_col=patient_col,
        time_col=time_col,
        ordinal_features=ordinal_features,
        continuous_features=continuous_features,
        categorical_ordinal_features=categorical_ordinal_features
    )

    print(f"   ✓ Noise injection completed")
    print(f"   Input shape: {synth_df.shape}")
    print(f"   Output shape: {synth_df_noisy.shape}")

    # Upload to datalake
    if upload_to_dl:
        print("\n[STEP 4] Uploading noisy dataset to datalake...")

        # Get metadata using search_files
        client = DatalakeClient()
        query = {'custom.file_code': synthetic_file_code}
        if synthetic_query_params:
            query.update(synthetic_query_params)

        search_result = client.search_files(query=query)
        if not search_result or 'files' not in search_result or len(search_result['files']) == 0:
            raise ValueError(f"No file found with code '{synthetic_file_code}' for metadata retrieval")

        metadata_object_name = search_result['files'][0]['object_name']

        # Get original metadata
        original_metadata = get_file_metadata(metadata_object_name)

        # Create new metadata with modified file_code and noise parameters
        new_metadata = original_metadata.copy()
        original_file_code = new_metadata.get('file_code', synthetic_file_code)
        new_metadata['file_code'] = f"{original_file_code}_noisy"
        new_metadata['noise_injection'] = {
            'epsilon': epsilon,
            'delta': delta,
            'method': method,
            'target_dcr': target_dcr,
            'preserve_temporal': preserve_temporal
        }
        if current_dcr:
            new_metadata['noise_injection']['current_dcr'] = current_dcr

        # Create new file name with _noisy suffix (without epsilon in filename)
        file_name_parts = synth_file_name.rsplit('.', 1)
        if len(file_name_parts) == 2:
            new_file_name = f"{file_name_parts[0]}_noisy.{file_name_parts[1]}"
        else:
            new_file_name = f"{synth_file_name}_noisy"

        # Upload to datalake
        upload_success = upload_to_datalake(
            df=synth_df_noisy,
            file_name=new_file_name,
            prefix=prefix,
            metadata=new_metadata
        )

        if not upload_success:
            print("   ⚠ Warning: Upload to datalake failed!")

    # Save local copy if requested
    if save_local and output_path:
        print(f"\n[STEP 5] Saving local copy to: {output_path}")
        synth_df_noisy.to_csv(output_path, index=False)
        print(f"   ✓ Saved successfully")

    print("\n" + "="*80)
    print("PIPELINE COMPLETED")
    print("="*80)
    print(f"Final dataset shape: {synth_df_noisy.shape}")
    print(f"Final columns: {list(synth_df_noisy.columns)}")

    return synth_df_noisy


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description='Add post-generation noise to synthetic data from datalake using Differential Privacy'
    )

    # Required arguments
    parser.add_argument(
        '--synthetic-code',
        type=str,
        required=True,
        help='File code for synthetic data in datalake (e.g., ADNIMERGE_synthetic)'
    )

    parser.add_argument(
        '--real-code',
        type=str,
        required=True,
        help='File code for real data in datalake (e.g., ADNIMERGE)'
    )

    # Noise parameters
    parser.add_argument(
        '--epsilon',
        type=float,
        default=1.0,
        help='Privacy budget (default: 1.0). Smaller = more privacy/noise. Range: 0.5-5.0'
    )

    parser.add_argument(
        '--delta',
        type=float,
        default=None,
        help='Failure probability for (ε, δ)-DP (default: 1/n²)'
    )

    parser.add_argument(
        '--method',
        type=str,
        default='gaussian',
        choices=['gaussian', 'laplace', 'adaptive'],
        help='Noise injection method (default: gaussian)'
    )

    parser.add_argument(
        '--target-dcr',
        type=float,
        default=0.50,
        help='Target DCR value (default: 0.50)'
    )

    parser.add_argument(
        '--current-dcr',
        type=float,
        default=None,
        help='Current DCR from validation (used with adaptive method)'
    )

    # Data parameters
    parser.add_argument(
        '--patient-col',
        type=str,
        default='ID',
        help='Column name for patient ID (default: ID)'
    )

    parser.add_argument(
        '--time-col',
        type=str,
        default='TIME',
        help='Column name for time variable (default: TIME)'
    )

    parser.add_argument(
        '--no-temporal',
        action='store_true',
        help='Do not preserve temporal correlations'
    )

    parser.add_argument(
        '--ordinal-features',
        type=str,
        nargs='+',
        default=None,
        help='Ordinal features to round after noise (default: auto-detect common scales)'
    )

    parser.add_argument(
        '--continuous-features',
        type=str,
        nargs='+',
        default=None,
        help='Continuous features (default: all numeric non-ordinal)'
    )

    parser.add_argument(
        '--categorical-ordinal-features',
        type=str,
        nargs='+',
        default=None,
        help='Categorical ordinal variables that should NOT receive noise (e.g., APOE EDUCATION). If not specified, uses auto-detection.'
    )

    # Output parameters
    parser.add_argument(
        '--prefix',
        type=str,
        default='synthetic/noisy',
        help='Datalake prefix for upload (default: synthetic/noisy)'
    )

    parser.add_argument(
        '--no-upload',
        action='store_true',
        help='Do not upload to datalake'
    )

    parser.add_argument(
        '--save-local',
        action='store_true',
        help='Save a local copy of the noisy data'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output CSV file path for local copy (requires --save-local)'
    )

    # Query parameters
    parser.add_argument(
        '--synthetic-query',
        type=str,
        action='append',
        help='Additional query parameters for synthetic data in format key=value'
    )

    parser.add_argument(
        '--real-query',
        type=str,
        action='append',
        help='Additional query parameters for real data in format key=value'
    )

    args = parser.parse_args()

    # Parse query parameters
    synthetic_query_params = {}
    if args.synthetic_query:
        for param in args.synthetic_query:
            key, value = param.split('=', 1)
            synthetic_query_params[key] = value

    real_query_params = {}
    if args.real_query:
        for param in args.real_query:
            key, value = param.split('=', 1)
            real_query_params[key] = value

    # Run pipeline
    synth_df_noisy = inject_noise_datalake_data(
        synthetic_file_code=args.synthetic_code,
        real_file_code=args.real_code,
        epsilon=args.epsilon,
        delta=args.delta,
        method=args.method,
        target_dcr=args.target_dcr,
        current_dcr=args.current_dcr,
        preserve_temporal=not args.no_temporal,
        patient_col=args.patient_col,
        time_col=args.time_col,
        ordinal_features=args.ordinal_features,
        continuous_features=args.continuous_features,
        categorical_ordinal_features=args.categorical_ordinal_features,
        upload_to_dl=not args.no_upload,
        prefix=args.prefix,
        save_local=args.save_local,
        output_path=args.output,
        synthetic_query_params=synthetic_query_params if synthetic_query_params else None,
        real_query_params=real_query_params if real_query_params else None
    )

    return synth_df_noisy


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 1:
        # No arguments provided, show example
        print("="*80)
        print("POST-GENERATION NOISE INJECTION FROM DATALAKE")
        print("="*80)
        print("\nEXAMPLE USAGE")
        print("="*80)
        print("\nBasic usage with default parameters:")
        print("  python noise_injection_from_datalake.py --synthetic-code ADNIMERGE_synthetic --real-code ADNIMERGE")
        print("\nWith custom epsilon (more noise):")
        print("  python noise_injection_from_datalake.py --synthetic-code ADNIMERGE_synthetic --real-code ADNIMERGE --epsilon 0.8")
        print("\nWith adaptive method and current DCR:")
        print("  python noise_injection_from_datalake.py --synthetic-code ADNIMERGE_synthetic --real-code ADNIMERGE --method adaptive --current-dcr 0.82")
        print("\nWith local save:")
        print("  python noise_injection_from_datalake.py --synthetic-code ADNIMERGE_synthetic --real-code ADNIMERGE --save-local --output noisy_data.csv")
        print("\nWithout temporal preservation:")
        print("  python noise_injection_from_datalake.py --synthetic-code ADNIMERGE_synthetic --real-code ADNIMERGE --no-temporal")
        print("\nWith custom ordinal features:")
        print("  python noise_injection_from_datalake.py --synthetic-code ADNIMERGE_synthetic --real-code ADNIMERGE --ordinal-features MMSE CDRSB FAQ")
        print("\nTest multiple epsilon values:")
        print("  for eps in 0.5 1.0 1.5 2.0; do")
        print("    python noise_injection_from_datalake.py --synthetic-code ADNIMERGE_synthetic --real-code ADNIMERGE --epsilon $eps")
        print("  done")
        print("\n" + "="*80)
        print("PROGRAMMATIC USAGE")
        print("="*80)
        print("""
from noise_injection_from_datalake import inject_noise_datalake_data

# Basic usage with default parameters
synth_noisy = inject_noise_datalake_data(
    synthetic_file_code='ADNIMERGE_synthetic',
    real_file_code='ADNIMERGE',
    epsilon=1.0,
    method='gaussian'
)

# Adaptive method with current DCR
synth_noisy = inject_noise_datalake_data(
    synthetic_file_code='ADNIMERGE_synthetic',
    real_file_code='ADNIMERGE',
    epsilon=0.8,
    method='adaptive',
    current_dcr=0.82,
    target_dcr=0.50
)

# Custom configuration
synth_noisy = inject_noise_datalake_data(
    synthetic_file_code='ADNIMERGE_synthetic',
    real_file_code='ADNIMERGE',
    epsilon=1.5,
    method='gaussian',
    preserve_temporal=True,
    ordinal_features=['MMSE', 'CDRSB', 'FAQ'],
    save_local=True,
    output_path='synthetic_noisy.csv'
)
        """)
        print("="*80)
        print("\nRECOMMENDED EPSILON VALUES")
        print("="*80)
        print("""
Based on your current DCR, choose appropriate epsilon:

- DCR > 0.80 (too high): epsilon = 0.5-1.0 (aggressive noise)
- DCR 0.60-0.80 (moderate): epsilon = 1.0-2.0 (moderate noise)
- DCR 0.50-0.60 (near target): epsilon = 2.0-5.0 (light noise)

Lower epsilon = more noise = lower DCR = better privacy
Higher epsilon = less noise = higher DCR = better utility

For adaptive method, the algorithm automatically adjusts noise based on
current_dcr vs target_dcr.
        """)
        print("="*80)
    else:
        # Run with command-line arguments
        main()
