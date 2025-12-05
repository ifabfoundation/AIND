"""
Script for converting dummy variables back to categorical variables.

This script provides functions to reverse one-hot encoding and other dummy encoding
methods, reconstructing the original categorical columns.
"""

import pandas as pd
import numpy as np
import re


def dummies_to_categorical(df, dummy_prefix, separator='_', drop_dummies=True,
                          handle_all_zero='missing', original_column_name=None):
    """
    Convert dummy variables back to a single categorical column.

    This is the inverse operation of pd.get_dummies().

    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing dummy variables
    dummy_prefix : str
        The prefix used for the dummy columns (e.g., 'SEX' for 'SEX_M', 'SEX_F')
    separator : str, default='_'
        The separator between prefix and category value
    drop_dummies : bool, default=True
        Whether to drop the dummy columns after conversion
    handle_all_zero : str, default='missing'
        How to handle rows where all dummies are 0:
        - 'missing': Set to NaN
        - 'first': Set to the first category
        - 'none': Create a 'None' or 'Unknown' category
    original_column_name : str, optional
        Name for the new categorical column. If None, uses dummy_prefix

    Returns:
    --------
    pandas.DataFrame
        A copy of the dataframe with the categorical column added/restored

    Examples:
    ---------
    # Convert SEX_M, SEX_F back to SEX column
    df = dummies_to_categorical(df, 'SEX', separator='_')

    # Convert DX_CN, DX_MCI, DX_AD back to DX column
    df = dummies_to_categorical(df, 'DX', separator='_')
    """
    df_result = df.copy()

    # Find all columns with the given prefix
    pattern = f"^{re.escape(dummy_prefix)}{re.escape(separator)}(.+)$"
    dummy_cols = [col for col in df.columns if re.match(pattern, col)]

    if not dummy_cols:
        print(f"Warning: No dummy columns found with prefix '{dummy_prefix}{separator}'")
        return df_result

    # Extract category names from column names
    categories = [col.split(separator, 1)[1] for col in dummy_cols]

    # Determine the column name for the categorical variable
    cat_col_name = original_column_name if original_column_name else dummy_prefix

    # Create the categorical column
    # For each row, find which dummy is 1 (or has max value)
    def get_category(row):
        # Get values of dummy columns
        values = row[dummy_cols].values

        # Check if all zeros
        if values.sum() == 0:
            if handle_all_zero == 'missing':
                return np.nan
            elif handle_all_zero == 'first':
                return categories[0]
            else:  # 'none'
                return 'Unknown'

        # Find the index of the maximum value (should be 1)
        max_idx = values.argmax()
        return categories[max_idx]

    df_result[cat_col_name] = df_result.apply(get_category, axis=1)

    # Drop dummy columns if requested
    if drop_dummies:
        df_result = df_result.drop(columns=dummy_cols)

    return df_result


def multiple_dummies_to_categorical(df, dummy_configs, separator='_', drop_dummies=True):
    """
    Convert multiple sets of dummy variables back to categorical columns.

    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing dummy variables
    dummy_configs : list of dict
        List of configurations for each set of dummies.
        Each dict should have:
        - 'prefix': str, the prefix for dummy columns
        - 'separator': str (optional), separator to use (overrides default)
        - 'original_name': str (optional), name for the categorical column
        - 'handle_all_zero': str (optional), how to handle all-zero rows
    separator : str, default='_'
        Default separator between prefix and category value
    drop_dummies : bool, default=True
        Whether to drop the dummy columns after conversion

    Returns:
    --------
    pandas.DataFrame
        A copy of the dataframe with categorical columns restored

    Examples:
    ---------
    configs = [
        {'prefix': 'SEX'},
        {'prefix': 'DX', 'original_name': 'Diagnosis'},
        {'prefix': 'APOE4', 'handle_all_zero': 'first'}
    ]
    df = multiple_dummies_to_categorical(df, configs)
    """
    df_result = df.copy()

    for config in dummy_configs:
        prefix = config['prefix']
        sep = config.get('separator', separator)
        original_name = config.get('original_name', None)
        handle_zeros = config.get('handle_all_zero', 'missing')

        df_result = dummies_to_categorical(
            df_result,
            dummy_prefix=prefix,
            separator=sep,
            drop_dummies=drop_dummies,
            handle_all_zero=handle_zeros,
            original_column_name=original_name
        )

    return df_result


def auto_detect_dummies(df, separator='_'):
    """
    Automatically detect groups of dummy variables in a dataframe.

    This function looks for columns that follow the pattern 'PREFIX_CATEGORY'
    and groups them by prefix.

    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe to analyze
    separator : str, default='_'
        The separator between prefix and category

    Returns:
    --------
    dict
        Dictionary mapping prefixes to lists of dummy column names

    Examples:
    ---------
    dummy_groups = auto_detect_dummies(df)
    # Returns: {'SEX': ['SEX_M', 'SEX_F'], 'DX': ['DX_CN', 'DX_MCI', 'DX_AD']}
    """
    dummy_groups = {}

    for col in df.columns:
        if separator in col:
            parts = col.split(separator, 1)
            if len(parts) == 2:
                prefix, category = parts

                # Check if values are binary (0/1) or boolean
                if df[col].dtype in [int, bool] or set(df[col].dropna().unique()).issubset({0, 1, True, False}):
                    if prefix not in dummy_groups:
                        dummy_groups[prefix] = []
                    dummy_groups[prefix].append(col)

    # Filter out groups with only one column (might not be real dummies)
    dummy_groups = {k: v for k, v in dummy_groups.items() if len(v) > 1}

    return dummy_groups


def label_decode(df, column, label_mapping, inplace=False):
    """
    Convert integer labels back to original categorical values.

    This is the inverse of LabelEncoder or manual label encoding.

    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the encoded column
    column : str
        Name of the column to decode
    label_mapping : dict
        Dictionary mapping integer labels to original categories
        Example: {0: 'CN', 1: 'MCI', 2: 'AD'}
    inplace : bool, default=False
        If True, modify the dataframe in place

    Returns:
    --------
    pandas.DataFrame
        The dataframe with decoded column (if not inplace)

    Examples:
    ---------
    # Convert DX column from 0,1,2 to 'CN','MCI','AD'
    mapping = {0: 'CN', 1: 'MCI', 2: 'AD'}
    df = label_decode(df, 'DX', mapping)
    """
    if inplace:
        df[column] = df[column].map(label_mapping)
        return df
    else:
        df_result = df.copy()
        df_result[column] = df_result[column].map(label_mapping)
        return df_result


def binary_to_categorical(df, column, mapping, inplace=False):
    """
    Convert a binary (0/1) column to categorical values.

    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the binary column
    column : str
        Name of the binary column
    mapping : dict
        Dictionary mapping binary values to categories
        Example: {0: 'Female', 1: 'Male'} or {0: 'No', 1: 'Yes'}
    inplace : bool, default=False
        If True, modify the dataframe in place

    Returns:
    --------
    pandas.DataFrame
        The dataframe with categorical column (if not inplace)

    Examples:
    ---------
    # Convert SEX from 0/1 to F/M
    df = binary_to_categorical(df, 'SEX', {0: 'F', 1: 'M'})
    """
    return label_decode(df, column, mapping, inplace)


# Example usage and testing
if __name__ == "__main__":
    print("="*80)
    print("Example 1: Convert dummy variables back to categorical")
    print("="*80)

    # Create sample data with dummy variables
    df_dummies = pd.DataFrame({
        'ID': [1, 2, 3, 4, 5],
        'AGE': [65, 70, 75, 68, 72],
        'SEX_M': [1, 0, 1, 0, 1],
        'SEX_F': [0, 1, 0, 1, 0],
        'DX_CN': [1, 0, 0, 1, 0],
        'DX_MCI': [0, 1, 0, 0, 1],
        'DX_AD': [0, 0, 1, 0, 0],
        'MMSE': [28, 24, 18, 29, 22]
    })

    print("\nOriginal data with dummies:")
    print(df_dummies)

    # Convert SEX dummies back to categorical
    df_restored = dummies_to_categorical(df_dummies, 'SEX', separator='_')
    print("\nAfter converting SEX dummies:")
    print(df_restored)

    # Convert DX dummies back to categorical
    df_restored = dummies_to_categorical(df_restored, 'DX', separator='_')
    print("\nAfter converting DX dummies:")
    print(df_restored)

    print("\n" + "="*80)
    print("Example 2: Convert multiple dummy sets at once")
    print("="*80)

    df_dummies2 = pd.DataFrame({
        'ID': [1, 2, 3, 4, 5],
        'SEX_M': [1, 0, 1, 0, 1],
        'SEX_F': [0, 1, 0, 1, 0],
        'DX_CN': [1, 0, 0, 1, 0],
        'DX_MCI': [0, 1, 0, 0, 1],
        'DX_AD': [0, 0, 1, 0, 0],
        'APOE4_0': [1, 0, 0, 1, 0],
        'APOE4_1': [0, 1, 0, 0, 1],
        'APOE4_2': [0, 0, 1, 0, 0]
    })

    print("\nOriginal data:")
    print(df_dummies2)

    configs = [
        {'prefix': 'SEX'},
        {'prefix': 'DX', 'original_name': 'Diagnosis'},
        {'prefix': 'APOE4'}
    ]

    df_restored2 = multiple_dummies_to_categorical(df_dummies2, configs)
    print("\nAfter converting all dummies:")
    print(df_restored2)

    print("\n" + "="*80)
    print("Example 3: Auto-detect dummy variables")
    print("="*80)

    dummy_groups = auto_detect_dummies(df_dummies)
    print("\nAuto-detected dummy groups:")
    for prefix, columns in dummy_groups.items():
        print(f"  {prefix}: {columns}")

    print("\n" + "="*80)
    print("Example 4: Label decoding")
    print("="*80)

    df_labels = pd.DataFrame({
        'ID': [1, 2, 3, 4, 5],
        'DX': [0, 1, 2, 0, 1],
        'SEX': [1, 0, 1, 0, 1]
    })

    print("\nOriginal data with integer labels:")
    print(df_labels)

    # Decode DX
    dx_mapping = {0: 'CN', 1: 'MCI', 2: 'AD'}
    df_decoded = label_decode(df_labels, 'DX', dx_mapping)

    # Decode SEX
    sex_mapping = {0: 'F', 1: 'M'}
    df_decoded = label_decode(df_decoded, 'SEX', sex_mapping)

    print("\nAfter decoding:")
    print(df_decoded)

    print("\n" + "="*80)
    print("Example 5: Binary to categorical")
    print("="*80)

    df_binary = pd.DataFrame({
        'ID': [1, 2, 3, 4, 5],
        'SEX': [1, 0, 1, 0, 1],
        'APOE4_positive': [0, 1, 1, 0, 1]
    })

    print("\nOriginal data with binary columns:")
    print(df_binary)

    df_converted = binary_to_categorical(df_binary, 'SEX', {0: 'Female', 1: 'Male'})
    df_converted = binary_to_categorical(df_converted, 'APOE4_positive', {0: 'No', 1: 'Yes'})

    print("\nAfter converting binary to categorical:")
    print(df_converted)
