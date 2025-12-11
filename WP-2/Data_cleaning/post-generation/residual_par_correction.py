"""
Script to apply Residual-PAR correction to Leaspy synthetic data using real data from datalake.

This script uses KNN pseudo-pairing approach (does NOT require same IDs between real and synthetic):
1. KNN matching: For each Leaspy patient, find k most similar real patients using baseline features
2. Residual computation: Interpolate real neighbors' sequences to Leaspy timepoints, compute residuals
3. Residual-PAR: Train PARSynthesizer on residuals (without context variables)
4. Correction: Sample synthetic residuals and correct Leaspy with nearest merge (tolerance-based)
5. Evaluation: KS per feature and DCR (Distance To Closest Record)

Usage:
    python residual_par_correction.py \\
        --file-code-real ADNIMERGE_cleaned_03 \\
        --file-code-leaspy ADNIMERGE_leaspy_synthetic \\
        --baseline-match-feats MMSE ADAS13 Hippocampus
"""

import pandas as pd
import numpy as np
import argparse
import json
from pathlib import Path
from typing import List, Optional, Dict, Tuple

# Datalake client
from dl_client import DatalakeClient

# Import KNN pseudo-pairing functions from add_noise_nb.py
from add_noise_nb import (
    ensure_monotone_per_id,
    snapshot_baseline,
    knn_match_by_baseline_feats,
    build_residual_dataset_knn,
    train_par_on_residuals_no_context,
    correct_leaspy_with_par_residuals,
    nearest_merge_by_id,
    ks_by_feature,
    dcr_distance_to_closest_record
)


# ==========================================
# DATALAKE UTILITY FUNCTIONS
# ==========================================

def load_from_datalake(file_code, query_params=None):
    """
    Load a dataset from the datalake using DatalakeClient.

    Parameters:
    -----------
    file_code : str
        The file code to search for (e.g., 'ADNIMERGE_cleaned_03')
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
        The name of the file (with _par_corrected suffix)
    prefix : str
        The prefix for the datalake (e.g., 'synthetic/noisy')
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
    print(f"  Metadata keys: {list(metadata.keys())}")

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


# ==========================================
# FEATURE DETECTION
# ==========================================

def auto_detect_features(df, id_col='ID', time_col='TIME', exclude_cols=None):
    """
    Auto-detect feature columns (all numeric columns except ID, TIME, and specified exclusions).

    Parameters:
    -----------
    df : pandas.DataFrame
        The dataset
    id_col : str, default='ID'
        ID column name
    time_col : str, default='TIME'
        Time column name
    exclude_cols : list, optional
        Additional columns to exclude (e.g., baseline_match_feats)

    Returns:
    --------
    list
        List of feature column names
    """
    exclude = [id_col, time_col]
    if exclude_cols:
        exclude.extend(exclude_cols)

    # Get all numeric columns except excluded ones
    feature_cols = [
        col for col in df.columns
        if col not in exclude and pd.api.types.is_numeric_dtype(df[col])
    ]

    print(f"\n   Auto-detected {len(feature_cols)} feature columns:")
    for col in feature_cols[:10]:  # Show first 10
        print(f"     - {col}")
    if len(feature_cols) > 10:
        print(f"     ... and {len(feature_cols) - 10} more")

    return feature_cols


# ==========================================
# MAIN PIPELINE WITH KNN PSEUDO-PAIRING
# ==========================================

def residual_par_pipeline_from_datalake(
    file_code_real,
    file_code_leaspy,
    baseline_match_feats,
    id_col='ID',
    time_col='TIME',
    k_neighbors=3,
    merge_tolerance=0.25,
    par_epochs=128,
    use_cuda=True,
    upload_to_dl=True,
    prefix='synthetic/noisy',
    save_local=False,
    output_path=None,
    query_params_real=None,
    query_params_leaspy=None
):
    """
    Complete pipeline: load real and Leaspy data from datalake, apply Residual-PAR correction
    using KNN pseudo-pairing (does NOT require same IDs), evaluate, and upload results.

    Pipeline steps:
    1. Load real and Leaspy data from datalake
    2. Auto-detect feature columns
    3. KNN matching: find k most similar real patients for each Leaspy patient (using baseline features)
    4. Build residual dataset: interpolate real neighbors to Leaspy timepoints, compute residuals
    5. Train PAR on residuals (without context)
    6. Correct Leaspy with synthetic residuals using nearest merge (tolerance-based)
    7. Evaluate with KS and DCR
    8. Upload to datalake

    Parameters:
    -----------
    file_code_real : str
        File code for real data in datalake (e.g., 'ADNIMERGE_cleaned_03')
    file_code_leaspy : str
        File code for Leaspy synthetic data in datalake (e.g., 'ADNIMERGE_leaspy_synthetic')
    baseline_match_feats : list of str
        Features to use for KNN matching at baseline (e.g., ['MMSE', 'ADAS13', 'Hippocampus'])
    id_col : str, default='ID'
        Name of ID column
    time_col : str, default='TIME'
        Name of TIME column
    k_neighbors : int, default=3
        Number of nearest neighbors for KNN matching
    merge_tolerance : float, default=0.25
        Tolerance for nearest merge on time (e.g., 0.25 years ≈ 3 months)
    par_epochs : int, default=128
        Number of epochs for PAR training
    use_cuda : bool, default=True
        Use GPU for PAR training
    upload_to_dl : bool, default=True
        Upload results to datalake
    prefix : str, default='synthetic/noisy'
        Prefix for datalake upload
    save_local : bool, default=False
        Save local copy
    output_path : str, optional
        Path to save CSV locally
    query_params_real : dict, optional
        Additional query parameters for real data
    query_params_leaspy : dict, optional
        Additional query parameters for Leaspy data

    Returns:
    --------
    tuple
        (pandas.DataFrame, dict, float): corrected_df, ks_scores, dcr_score
    """
    print("=" * 80)
    print("RESIDUAL-PAR CORRECTION WITH KNN PSEUDO-PAIRING")
    print("=" * 80)
    print(f"\nMethod: KNN matching (k={k_neighbors}) + Residual-PAR")
    print(f"Does NOT require same IDs between real and synthetic data")

    # [STEP 1] Load REAL data from datalake
    print("\n[STEP 1] Loading REAL data from datalake...")
    df_real, object_name_real, file_name_real = load_from_datalake(
        file_code_real, query_params_real
    )

    # [STEP 2] Load LEASPY/SYNTHETIC data from datalake
    print("\n[STEP 2] Loading LEASPY/SYNTHETIC data from datalake...")
    df_leaspy, object_name_leaspy, file_name_leaspy = load_from_datalake(
        file_code_leaspy, query_params_leaspy
    )

    # [STEP 3] Validate baseline_match_feats
    print(f"\n[STEP 3] Validating baseline matching features...")
    print(f"   Baseline match features: {baseline_match_feats}")

    missing_real = [f for f in baseline_match_feats if f not in df_real.columns]
    missing_leaspy = [f for f in baseline_match_feats if f not in df_leaspy.columns]

    if missing_real:
        raise ValueError(f"Baseline features not found in REAL data: {missing_real}")
    if missing_leaspy:
        raise ValueError(f"Baseline features not found in LEASPY data: {missing_leaspy}")

    print(f"   ✓ All baseline features found in both datasets")

    # [STEP 4] Auto-detect ALL feature columns (excluding baseline_match_feats to avoid duplication)
    print("\n[STEP 4] Auto-detecting feature columns to correct...")
    all_feature_cols = auto_detect_features(
        df_real,
        id_col=id_col,
        time_col=time_col,
        exclude_cols=None  # Don't exclude baseline features, they should be corrected too
    )

    if not all_feature_cols:
        raise ValueError("No feature columns detected! Check your data.")

    # Ensure baseline_match_feats are in feature_cols
    feature_cols = list(set(all_feature_cols + baseline_match_feats))
    print(f"   ✓ Will correct {len(feature_cols)} features (including {len(baseline_match_feats)} baseline features)")

    # [STEP 5] Ensure monotone time
    print("\n[STEP 5] Ensuring monotone time per ID...")
    df_real = ensure_monotone_per_id(df_real, id_col, time_col)
    df_leaspy = ensure_monotone_per_id(df_leaspy, id_col, time_col)
    print(f"   ✓ Time is monotone for all IDs")

    # [STEP 6] KNN matching + Build residual dataset
    print(f"\n[STEP 6] KNN matching and building residual dataset...")
    print(f"   Finding {k_neighbors} nearest real neighbors for each Leaspy patient")
    print(f"   Using baseline features: {baseline_match_feats}")

    residuals = build_residual_dataset_knn(
        df_real=df_real,
        df_leaspy=df_leaspy,
        id_col=id_col,
        time_col=time_col,
        feature_cols=feature_cols,
        baseline_match_feats=baseline_match_feats,
        k=k_neighbors
    )

    if residuals.empty:
        raise RuntimeError(
            "Residual dataset is empty! Check:\n"
            "  - Column names are correct\n"
            "  - Baseline features exist in both datasets\n"
            "  - Data has valid numeric values"
        )

    print(f"   ✓ Residuals shape: {residuals.shape}")
    print(f"   ✓ Unique Leaspy IDs in residuals: {residuals[id_col].nunique()}")

    # Show residual statistics for sample features
    print(f"   ✓ Residual statistics (first 3 features):")
    for feat in feature_cols[:3]:
        if feat in residuals.columns:
            mean_val = residuals[feat].mean()
            std_val = residuals[feat].std()
            print(f"     - {feat}: mean={mean_val:.4f}, std={std_val:.4f}")

    # [STEP 7] Train PAR on residuals (without context)
    print("\n[STEP 7] Training PARSynthesizer on residuals...")
    print(f"   PAR epochs: {par_epochs}")
    print(f"   CUDA enabled: {use_cuda}")
    print(f"   Context variables: None (KNN pseudo-pairing approach)")

    par_model = train_par_on_residuals_no_context(
        residuals=residuals,
        id_col=id_col,
        time_col=time_col,
        epochs=par_epochs,
        cuda=use_cuda,
        verbose=True
    )

    print(f"   ✓ PAR model trained successfully")

    # [STEP 8] Correct Leaspy with synthetic residuals (nearest merge with tolerance)
    print("\n[STEP 8] Correcting Leaspy data with synthetic residuals...")
    print(f"   Merge tolerance: {merge_tolerance} (in {time_col} units)")
    print(f"   Merge direction: nearest")

    df_corrected = correct_leaspy_with_par_residuals(
        df_leaspy=df_leaspy,
        par=par_model,
        id_col=id_col,
        time_col=time_col,
        feature_cols=feature_cols,
        tolerance=merge_tolerance,
        direction="nearest"
    )

    print(f"   ✓ Corrected data shape: {df_corrected.shape}")

    # [STEP 9] Evaluation: KS and DCR
    print("\n[STEP 9] Evaluating corrected data...")
    print("   Computing KS statistics per feature...")

    ks_scores = ks_by_feature(
        real_df=df_real,
        synth_df=df_corrected,
        feature_cols=feature_cols
    )

    print("\n   KS Scores (lower is better):")
    print("   " + "-" * 60)
    # Sort by KS value and show all
    sorted_ks = sorted(ks_scores.items(), key=lambda x: x[1])
    for feature, ks_val in sorted_ks[:15]:  # Show first 15
        print(f"   {feature:35s}: {ks_val:.4f}")
    if len(sorted_ks) > 15:
        print(f"   ... and {len(sorted_ks) - 15} more features")
    print("   " + "-" * 60)

    mean_ks = np.mean(list(ks_scores.values()))
    median_ks = np.median(list(ks_scores.values()))
    print(f"   Mean KS: {mean_ks:.4f}")
    print(f"   Median KS: {median_ks:.4f}")

    print("\n   Computing DCR (Distance to Closest Record)...")
    dcr_score = dcr_distance_to_closest_record(
        real_df=df_real,
        synth_df=df_corrected,
        feature_cols=feature_cols
    )

    if dcr_score is not None:
        print(f"   ✓ DCR Score (higher is better for privacy): {dcr_score:.4f}")
    else:
        print(f"   ⚠ DCR Score not available (SDMetrics not installed or error)")

    # [STEP 10] Merge corrected data back with original Leaspy structure
    print("\n[STEP 10] Merging corrected features back to original structure...")

    # Keep all original columns from Leaspy, replace corrected features
    df_final = df_leaspy.copy()

    # Drop feature columns from df_final to avoid conflicts
    df_final = df_final.drop(columns=[c for c in feature_cols if c in df_final.columns], errors='ignore')

    # Prepare corrected features for merge
    df_corrected_merge = df_corrected[[id_col, time_col] + [c for c in feature_cols if c in df_corrected.columns]].copy()

    # Merge (left join to keep all original Leaspy rows)
    df_final = df_final.merge(
        df_corrected_merge,
        on=[id_col, time_col],
        how='left'
    )

    print(f"   ✓ Final dataset shape: {df_final.shape}")

    # [STEP 11] Upload to datalake
    if upload_to_dl:
        print("\n[STEP 11] Uploading corrected data to datalake...")

        # Get original Leaspy metadata using search_files
        client = DatalakeClient()
        query = {'custom.file_code': file_code_leaspy}
        if query_params_leaspy:
            query.update(query_params_leaspy)

        search_result = client.search_files(query=query)
        if not search_result or 'files' not in search_result or len(search_result['files']) == 0:
            raise ValueError(f"No file found with code '{file_code_leaspy}' for metadata retrieval")

        metadata_object_name = search_result['files'][0]['object_name']
        original_metadata = get_file_metadata(metadata_object_name)

        # Create new metadata
        new_metadata = original_metadata.copy()
        original_file_code = new_metadata.get('file_code', file_code_leaspy)
        new_metadata['file_code'] = f"{original_file_code}_par_corrected"
        new_metadata['residual_par_correction'] = {
            'method': 'knn_pseudo_pairing',
            'source_real': file_code_real,
            'source_leaspy': file_code_leaspy,
            'baseline_match_feats': baseline_match_feats,
            'k_neighbors': k_neighbors,
            'merge_tolerance': merge_tolerance,
            'par_epochs': par_epochs,
            'use_cuda': use_cuda,
            'use_context': False,
            'feature_columns': feature_cols,
            'n_features': len(feature_cols)
        }
        new_metadata['evaluation_metrics'] = {
            'ks_scores': {k: float(v) for k, v in ks_scores.items()},
            'mean_ks': float(mean_ks),
            'median_ks': float(median_ks),
            'dcr_score': float(dcr_score) if dcr_score is not None else None
        }

        # Create new file name
        file_name_parts = file_name_leaspy.rsplit('.', 1)
        if len(file_name_parts) == 2:
            new_file_name = f"{file_name_parts[0]}_par_corrected.{file_name_parts[1]}"
        else:
            new_file_name = f"{file_name_leaspy}_par_corrected"

        # Upload
        upload_success = upload_to_datalake(
            df=df_final,
            file_name=new_file_name,
            prefix=prefix,
            metadata=new_metadata
        )

        if not upload_success:
            print("   ⚠ Warning: Upload to datalake failed!")

    # [STEP 12] Save local copy if requested
    if save_local and output_path:
        print(f"\n[STEP 12] Saving local copy to: {output_path}")
        df_final.to_csv(output_path, index=False)
        print(f"   ✓ Saved successfully")

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED")
    print("=" * 80)
    print(f"Final dataset shape: {df_final.shape}")
    print(f"Mean KS score: {mean_ks:.4f}")
    print(f"Median KS score: {median_ks:.4f}")
    if dcr_score is not None:
        print(f"DCR score: {dcr_score:.4f}")
    print("=" * 80)

    return df_final, ks_scores, dcr_score


# ==========================================
# CLI MAIN FUNCTION
# ==========================================

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description='Apply Residual-PAR correction to Leaspy synthetic data using KNN pseudo-pairing (does NOT require same IDs)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Basic usage:
    python residual_par_correction.py \\
      --file-code-real ADNIMERGE_cleaned_03 \\
      --file-code-leaspy ADNIMERGE_leaspy_synthetic \\
      --baseline-match-feats MMSE ADAS13 Hippocampus

  With more neighbors and epochs:
    python residual_par_correction.py \\
      --file-code-real ADNIMERGE_cleaned_03 \\
      --file-code-leaspy ADNIMERGE_leaspy_synthetic \\
      --baseline-match-feats MMSE ADAS13 \\
      --k-neighbors 5 \\
      --par-epochs 256
        """
    )

    # Required arguments
    parser.add_argument(
        '--file-code-real',
        type=str,
        required=True,
        help='File code for REAL data in datalake (e.g., ADNIMERGE_cleaned_03)'
    )

    parser.add_argument(
        '--file-code-leaspy',
        type=str,
        required=True,
        help='File code for LEASPY/SYNTHETIC data in datalake (e.g., ADNIMERGE_leaspy_synthetic)'
    )

    parser.add_argument(
        '--baseline-match-feats',
        type=str,
        nargs='+',
        required=True,
        help='Features to use for KNN matching at baseline (e.g., MMSE ADAS13 Hippocampus)'
    )

    # Column specifications
    parser.add_argument(
        '--id-col',
        type=str,
        default='ID',
        help='Name of the ID column (default: ID)'
    )

    parser.add_argument(
        '--time-col',
        type=str,
        default='TIME',
        help='Name of the TIME column (default: TIME)'
    )

    # KNN parameters
    parser.add_argument(
        '--k-neighbors',
        type=int,
        default=3,
        help='Number of nearest neighbors for KNN matching (default: 3)'
    )

    parser.add_argument(
        '--merge-tolerance',
        type=float,
        default=0.25,
        help='Tolerance for nearest merge on time in TIME units (default: 0.25, e.g., 0.25 years ≈ 3 months)'
    )

    # PAR training parameters
    parser.add_argument(
        '--par-epochs',
        type=int,
        default=512,
        help='Number of epochs for PAR training (default: 128, more epochs = better quality)'
    )

    parser.add_argument(
        '--no-cuda',
        action='store_true',
        help='Disable CUDA/GPU for PAR training'
    )

    # Upload/save options
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
        help='Save a local copy of the corrected data'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output CSV file path for local copy (requires --save-local)'
    )

    # Query parameters
    parser.add_argument(
        '--query-param-real',
        type=str,
        action='append',
        help='Additional query parameters for REAL data in format key=value'
    )

    parser.add_argument(
        '--query-param-leaspy',
        type=str,
        action='append',
        help='Additional query parameters for LEASPY data in format key=value'
    )

    args = parser.parse_args()

    # Parse query parameters for real data
    query_params_real = {}
    if args.query_param_real:
        for param in args.query_param_real:
            key, value = param.split('=', 1)
            query_params_real[key] = value

    # Parse query parameters for Leaspy data
    query_params_leaspy = {}
    if args.query_param_leaspy:
        for param in args.query_param_leaspy:
            key, value = param.split('=', 1)
            query_params_leaspy[key] = value

    # Run pipeline
    df_corrected, ks_scores, dcr_score = residual_par_pipeline_from_datalake(
        file_code_real=args.file_code_real,
        file_code_leaspy=args.file_code_leaspy,
        baseline_match_feats=args.baseline_match_feats,
        id_col=args.id_col,
        time_col=args.time_col,
        k_neighbors=args.k_neighbors,
        merge_tolerance=args.merge_tolerance,
        par_epochs=args.par_epochs,
        use_cuda=not args.no_cuda,
        upload_to_dl=not args.no_upload,
        prefix=args.prefix,
        save_local=args.save_local,
        output_path=args.output,
        query_params_real=query_params_real if query_params_real else None,
        query_params_leaspy=query_params_leaspy if query_params_leaspy else None
    )

    return df_corrected, ks_scores, dcr_score


# ==========================================
# EXAMPLE USAGE
# ==========================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) == 1:
        # No arguments provided, show examples
        print("=" * 80)
        print("RESIDUAL-PAR CORRECTION WITH KNN PSEUDO-PAIRING")
        print("=" * 80)
        print("\nThis script uses KNN pseudo-pairing and does NOT require same IDs")
        print("between real and synthetic data.")
        print("\n" + "=" * 80)
        print("COMMAND LINE USAGE EXAMPLES")
        print("=" * 80)

        print("\n1. Basic usage (REQUIRED: --baseline-match-feats):")
        print("   python residual_par_correction.py \\")
        print("     --file-code-real ADNIMERGE_cleaned_03 \\")
        print("     --file-code-leaspy ADNIMERGE_leaspy_synthetic \\")
        print("     --baseline-match-feats MMSE ADAS13 Hippocampus")

        print("\n2. With more neighbors (k=5):")
        print("   python residual_par_correction.py \\")
        print("     --file-code-real ADNIMERGE_cleaned_03 \\")
        print("     --file-code-leaspy ADNIMERGE_leaspy_synthetic \\")
        print("     --baseline-match-feats MMSE ADAS13 \\")
        print("     --k-neighbors 5")

        print("\n3. With more PAR epochs (better quality):")
        print("   python residual_par_correction.py \\")
        print("     --file-code-real ADNIMERGE_cleaned_03 \\")
        print("     --file-code-leaspy ADNIMERGE_leaspy_synthetic \\")
        print("     --baseline-match-feats MMSE ADAS13 Hippocampus \\")
        print("     --par-epochs 256")

        print("\n4. With custom merge tolerance (e.g., 0.5 years = 6 months):")
        print("   python residual_par_correction.py \\")
        print("     --file-code-real ADNIMERGE_cleaned_03 \\")
        print("     --file-code-leaspy ADNIMERGE_leaspy_synthetic \\")
        print("     --baseline-match-feats MMSE ADAS13 \\")
        print("     --merge-tolerance 0.5")

        print("\n5. Without CUDA (CPU only):")
        print("   python residual_par_correction.py \\")
        print("     --file-code-real ADNIMERGE_cleaned_03 \\")
        print("     --file-code-leaspy ADNIMERGE_leaspy_synthetic \\")
        print("     --baseline-match-feats MMSE ADAS13 \\")
        print("     --no-cuda")

        print("\n6. With custom prefix:")
        print("   python residual_par_correction.py \\")
        print("     --file-code-real ADNIMERGE_cleaned_03 \\")
        print("     --file-code-leaspy ADNIMERGE_validation \\")
        print("     --baseline-match-feats MMSE ADAS13 \\")
        print("     --prefix validation/noisy")

        print("\n7. Save local copy without upload:")
        print("   python residual_par_correction.py \\")
        print("     --file-code-real ADNIMERGE_cleaned_03 \\")
        print("     --file-code-leaspy ADNIMERGE_leaspy_synthetic \\")
        print("     --baseline-match-feats MMSE ADAS13 \\")
        print("     --no-upload --save-local --output corrected.csv")

        print("\n8. With query parameters:")
        print("   python residual_par_correction.py \\")
        print("     --file-code-real ADNIMERGE_cleaned_03 \\")
        print("     --file-code-leaspy ADNIMERGE_leaspy_synthetic \\")
        print("     --baseline-match-feats MMSE ADAS13 \\")
        print("     --query-param-real custom.level=cleaned_03")

        print("\n" + "=" * 80)
        print("PROGRAMMATIC USAGE")
        print("=" * 80)
        print("""
from residual_par_correction import residual_par_pipeline_from_datalake

# Basic usage with KNN pseudo-pairing
df_corrected, ks_scores, dcr_score = residual_par_pipeline_from_datalake(
    file_code_real='ADNIMERGE_cleaned_03',
    file_code_leaspy='ADNIMERGE_leaspy_synthetic',
    baseline_match_feats=['MMSE', 'ADAS13', 'Hippocampus'],
    k_neighbors=3,
    merge_tolerance=0.25,
    upload_to_dl=True,
    prefix='synthetic/noisy'
)

# With more neighbors and epochs
df_corrected, ks_scores, dcr_score = residual_par_pipeline_from_datalake(
    file_code_real='ADNIMERGE_cleaned_03',
    file_code_leaspy='ADNIMERGE_leaspy_synthetic',
    baseline_match_feats=['MMSE', 'ADAS13', 'Hippocampus', 'FAQ'],
    k_neighbors=5,
    merge_tolerance=0.25,
    par_epochs=256,
    use_cuda=True,
    upload_to_dl=True,
    prefix='synthetic/noisy',
    save_local=True,
    output_path='corrected_data.csv'
)

# Print results
import numpy as np
print(f"Mean KS: {np.mean(list(ks_scores.values())):.4f}")
print(f"Median KS: {np.median(list(ks_scores.values())):.4f}")
if dcr_score:
    print(f"DCR: {dcr_score:.4f}")
        """)
        print("=" * 80)
        print("\nKEY DIFFERENCES from original approach:")
        print("  ✓ Uses KNN pseudo-pairing (does NOT require same IDs)")
        print("  ✓ Matches Leaspy patients to similar real patients using baseline features")
        print("  ✓ No context variables needed")
        print("  ✓ More realistic for truly synthetic data")
        print("=" * 80)
    else:
        # Run with command-line arguments
        main()
