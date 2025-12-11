"""
Script for denormalizing (reverse normalization) data back to original values.

This script provides functions to reverse the normalization operations performed by
normalize_fixed_scale, normalize_dataset_minmax, and normalize_dataset_volumes.
"""

import pandas as pd
import numpy as np
import json


def _round_increasing(value):
    """
    Rounding for 'increasing' method:
    - Round down if decimal < 0.5
    - Round up if decimal >= 0.5
    (Standard round() behavior)
    - Returns NaN for NaN inputs
    """
    if pd.isna(value):
        return np.nan
    return np.round(value)


def _round_inverse(value):
    """
    Rounding for 'inverse' method:
    - Round down if decimal <= 0.5
    - Round up if decimal > 0.5
    - Returns NaN for NaN inputs
    """
    if pd.isna(value):
        return np.nan
    decimal_part = value - np.floor(value)
    if decimal_part <= 0.5:
        return np.floor(value)
    else:
        return np.ceil(value)


def denormalize_fixed_scale(df, columns, min_max_scales):
    """
    Denormalize specified columns from [0, 1] range back to their original scale.
    This is the inverse operation of normalize_fixed_scale.

    Values are rounded to integers using method-specific rules:
    - 'increasing': round down if decimal < 0.5, round up if decimal >= 0.5
    - 'inverse': round down if decimal <= 0.5, round up if decimal > 0.5

    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe with normalized values to denormalize
    columns : list
        List of column names to denormalize
    min_max_scales : dict
        Dictionary mapping column names to lists containing [min_value, max_value, method]
        where method is either 'increasing' (min→0, max→1) or 'inverse' (min→1, max→0)

    Returns:
    --------
    pandas.DataFrame
        A copy of the input dataframe with denormalized columns (as integers)
    """
    df_denormalized = df.copy()

    for col in columns:
        # Get min, max values and normalization method for the column
        min_val, max_val, method = min_max_scales[col]

        # Skip denormalization if min and max are the same
        if min_val == max_val:
            print(f"Warning: min and max are equal for column {col}. Skipping denormalization.")
            continue

        # Denormalize the column based on the specified method
        if method == "inverse":
            # Inverse denormalization: 0→max, 1→min
            # normalized = 1 - (value - min) / (max - min)
            # → value = (1 - normalized) * (max - min) + min
            df_denormalized[col] = (1 - df[col]) * (max_val - min_val) + min_val
            # Round using inverse method rules
            df_denormalized[col] = df_denormalized[col].apply(_round_inverse)
        else:  # method == "increasing" or any other value defaults to increasing
            # Standard denormalization: 0→min, 1→max
            # normalized = (value - min) / (max - min)
            # → value = normalized * (max - min) + min
            df_denormalized[col] = df[col] * (max_val - min_val) + min_val
            # Round using increasing method rules
            df_denormalized[col] = df_denormalized[col].apply(_round_increasing)

        # Convert to nullable integer type (Int64) to handle NaN values
        df_denormalized[col] = df_denormalized[col].astype('Int64')

    return df_denormalized


def denormalize_dataset_minmax(df, columns, min_max_values):
    """
    Denormalize specified columns from [0, 1] range back to their original scale.
    This is the inverse operation of normalize_dataset_minmax.

    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe with normalized values to denormalize
    columns : list
        List of column names to denormalize
    min_max_values : dict
        Dictionary mapping column names to [min_value, max_value] from the original dataset
        Example: {'column1': [10, 100], 'column2': [0, 50]}

    Returns:
    --------
    pandas.DataFrame
        A copy of the input dataframe with denormalized columns
    """
    df_denormalized = df.copy()

    for col in columns:
        # Get min and max values from the provided dictionary
        min_val, max_val = min_max_values[col]

        # Skip denormalization if min and max are the same
        if min_val == max_val:
            print(f"Warning: min and max are equal for column {col}. Skipping denormalization.")
            continue

        # Denormalize the column: value = normalized * (max - min) + min
        df_denormalized[col] = df[col] * (max_val - min_val) + min_val

    return df_denormalized


def denormalize_dataset_volumes(df, columns, normalization_values):
    """
    Denormalize specified volume columns from [0, 1] range back to their original scale.
    This is the inverse operation of normalize_dataset_volumes.

    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe with normalized values to denormalize
    columns : list
        List of column names to denormalize
    normalization_values : dict
        Dictionary mapping column names to lists containing [min_value, max_value, method]
        where method is either 'increasing' (min→0, max→1) or 'inverse' (min→1, max→0)

    Returns:
    --------
    pandas.DataFrame
        A copy of the input dataframe with denormalized columns
    """
    df_denormalized = df.copy()

    for col in columns:
        if col == 'ICV' or col == "ICV%ICV":
            continue

        # Get min, max values and normalization method for the column
        min_val, max_val, method = normalization_values[col]

        # Denormalize the column based on the specified method
        if method == "inverse":
            # Inverse denormalization: 0→max, 1→min
            # normalized = 1 - (value - min) / (max - min)
            # → value = (1 - normalized) * (max - min) + min
            df_denormalized[col] = (1 - df[col]) * (max_val - min_val) + min_val
        else:  # method == "increasing" or any other value defaults to increasing
            # Standard denormalization: 0→min, 1→max
            # normalized = (value - min) / (max - min)
            # → value = normalized * (max - min) + min
            df_denormalized[col] = df[col] * (max_val - min_val) + min_val

    return df_denormalized


def load_normalization_settings(filepath):
    """
    Load normalization settings from a JSON file.

    Parameters:
    -----------
    filepath : str
        Path to the JSON file containing normalization settings

    Returns:
    --------
    dict
        Dictionary with normalization settings
    """
    with open(filepath, 'r') as f:
        return json.load(f)


# Example usage
if __name__ == "__main__":
    # Load settings files
    normalization_settings_path = r"c:\Users\RaimondoReggio\OneDrive - Net Service S.p.A\Desktop\GitHub\AIND\WP-2\Data_cleaning\normalization_settings.json"
    volume_settings_path = r"c:\Users\RaimondoReggio\OneDrive - Net Service S.p.A\Desktop\GitHub\AIND\WP-2\Data_cleaning\volume_values_settings.json"

    normalization_settings = load_normalization_settings(normalization_settings_path)
    volume_settings = load_normalization_settings(volume_settings_path)

    # Example 1: Denormalize clinical tests/scales
    # Create sample normalized data
    normalized_data = pd.DataFrame({
        'MMSE': [0.5, 0.3, 0.8],
        'RAVLT_immediate': [0.6, 0.4, 0.7],
        'FAQ': [0.2, 0.5, 0.9],
        'MOCA': [0.7, 0.5, 0.9],
        'CDRSB': [0.1, 0.3, 0.6],
        'ADAS11': [0.2, 0.4, 0.7],
        'ADAS13': [0.3, 0.5, 0.8]
    })

    print("Original normalized data (clinical tests):")
    print(normalized_data)
    print("\n")

    # Denormalize using fixed scales
    denormalized_data = denormalize_fixed_scale(
        normalized_data,
        columns=list(normalization_settings.keys()),
        min_max_scales=normalization_settings
    )

    print("Denormalized data (clinical tests):")
    print(denormalized_data)
    print("\n" + "="*80 + "\n")

    # Example 2: Denormalize volumes
    normalized_volumes = pd.DataFrame({
        'Ventricles%ICV': [0.3, 0.5, 0.7],
        'Hippocampus%ICV': [0.6, 0.4, 0.8],
        'Entorhinal%ICV': [0.5, 0.6, 0.7],
        'Ventricles': [0.2, 0.4, 0.6],
        'Hippocampus': [0.7, 0.5, 0.9]
    })

    print("Original normalized data (volumes):")
    print(normalized_volumes)
    print("\n")

    # Denormalize volumes
    denormalized_volumes = denormalize_dataset_volumes(
        normalized_volumes,
        columns=list(normalized_volumes.columns),
        normalization_values=volume_settings
    )

    print("Denormalized data (volumes):")
    print(denormalized_volumes)
    print("\n" + "="*80 + "\n")

    # Example 3: Denormalize with dataset min/max (requires saved min/max values)
    # This example shows how to use denormalize_dataset_minmax
    # You need to provide the min/max values that were used during normalization
    custom_min_max = {
        'column1': [10, 100],
        'column2': [0, 50]
    }

    normalized_custom = pd.DataFrame({
        'column1': [0.0, 0.5, 1.0],
        'column2': [0.2, 0.6, 0.9]
    })

    print("Original normalized data (custom with dataset min/max):")
    print(normalized_custom)
    print("\n")

    denormalized_custom = denormalize_dataset_minmax(
        normalized_custom,
        columns=['column1', 'column2'],
        min_max_values=custom_min_max
    )

    print("Denormalized data (custom):")
    print(denormalized_custom)
