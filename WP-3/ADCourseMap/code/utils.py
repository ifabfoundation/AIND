# ---------------------------------------------------------- DEPENDANCE --------------------------------------------------------- #

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import os
import time
import json
import argparse
from datetime import datetime
from leaspy import Leaspy, Data, Dataset, AlgorithmSettings, IndividualParameters, __watermark__
from leaspy.io.logs.visualization.plotting import Plotting
from leaspy import IndividualParameters
import torch
import torch.nn as nn

# ---------------------------------------------------------- FUNCTIONS --------------------------------------------------------- #

# Function to append text to the file
def append_to_text_file(file_name, content):
    """Append content to a text file with a newline prefix"""
    with open(file_name, "a") as file:
        file.write("\n" + content)

# Function to create a json for the calibration settings
def make_json_calibration_settings_dict(opt, json_path):
    """Create a JSON file with MCMC-SAEM calibration settings for Leaspy model"""
    # Define the algorithm settings dictionary with MCMC-SAEM configuration
    settings = {
        "name": "mcmc_saem",  # Algorithm name
        "seed": opt.manual_seed,  # Random seed for reproducibility
        "algorithm_initialization_method": None,  # Use default initialization
        "model_initialization_method": "default",  # Use default model initialization
        "device": opt.device,  # Computing device (CPU/GPU)
        "parameters": {
            "progress_bar": True,  # Show progress bar during training
            "n_iter": opt.n_iter,  # Total number of iterations
            "n_burn_in_iter": None,  # Use fraction instead of fixed number
            "n_burn_in_iter_frac": opt.n_burn_in_iter_frac,  # Fraction of iterations for burn-in
            "burn_in_step_power": opt.burn_in_step_power,  # Step size decay during burn-in
            "random_order_variables": True,  # Randomize variable update order
            "sampler_ind": "Gibbs",  # Gibbs sampler for individual parameters
            "sampler_ind_params": {
                "acceptation_history_length": 25,  # Length of acceptance rate history
                "mean_acceptation_rate_target_bounds": [0.2, 0.4],  # Target acceptance rate bounds
                "adaptive_std_factor": 0.1  # Factor for adaptive standard deviation
            },
            "sampler_pop": "Gibbs",  # Gibbs sampler for population parameters
            "sampler_pop_params": {
                "random_order_dimension": True,  # Randomize dimension update order
                "acceptation_history_length": 25,  # Length of acceptance rate history
                "mean_acceptation_rate_target_bounds": [0.2, 0.4],  # Target acceptance rate bounds
                "adaptive_std_factor": 0.1  # Factor for adaptive standard deviation
            },
            "annealing": {
                "do_annealing": False,  # Disable simulated annealing
                "initial_temperature": 100,  # Initial temperature for annealing
                "n_plateau": 10,  # Number of plateau steps
                "n_iter": None,  # Use fraction instead of fixed number
                "n_iter_frac": 0.5  # Fraction of iterations for annealing
            }
        }
    }

    # Save settings to JSON file
    with open(json_path + 'algorithm_settings_calibration.json', 'w') as file:
        json.dump(settings, file, indent=4)

# Function to split the dataset into train, validation
def split_dataset_in_train_val(dataset, percentage, n_visits_to_predict=1):
    """Split dataset into train/validation maintaining visit distribution balance

    Parameters:
    -----------
    dataset : pandas.DataFrame
        The dataset to split
    percentage : float
        Percentage of subjects to use for training (0.0 to 1.0)
    n_visits_to_predict : int, optional
        Number of last visits per patient to use for prediction (default=1)
    """

    # Count visits per ID for balanced sampling
    visits_count = dataset.groupby('ID').size()

    # Create a dictionary that groups IDs by number of visits
    # This ensures balanced sampling across different visit frequencies
    sub_per_visits = {}
    for rid, count in visits_count.items():
        if count not in sub_per_visits:
            sub_per_visits[count] = []
        sub_per_visits[count].append(rid)

    # Sample training subjects from each visit group to maintain balance
    train_id = []
    for count in sub_per_visits:
        # Randomly sample the specified percentage from each visit group
        sublist_id = np.random.choice(sub_per_visits[count], size=int(percentage*len(sub_per_visits[count])), replace=False).tolist()
        train_id += sublist_id

    # Split dataset based on selected training IDs
    df_train = dataset[dataset['ID'].isin(train_id)].copy()
    df_val = dataset[~dataset['ID'].isin(train_id)].copy()

    # Create prediction dataset: use last n_visits_to_predict visits of each validation patient
    df_to_pred = df_val.groupby('ID').tail(n_visits_to_predict).copy()  # Get last n visits of every patient

    # Create personalization dataset: all validation visits except the prediction visits
    list_index = df_to_pred.index.tolist()
    df_pers = df_val[~df_val.index.isin(list_index)].copy()  # Evaluation dataset (all but prediction visits)

    # Set multi-level index (ID, TIME) for efficient data access
    df_train = df_train.set_index(['ID', 'TIME'])
    df_pers = df_pers.set_index(['ID', 'TIME'])
    df_to_pred = df_to_pred.set_index(['ID', 'TIME'])

    return df_train, df_val, df_pers, df_to_pred

def remove_unnecessary_values_from_list(original_list, exclude):
    """
    Remove values from a list.

    Parameters:
    -----------
    original_list : list
        The original list
    exclude : list
        List of values to exclude

    Returns:
    --------
    list
        List without excluded values
    """
    return [item for item in original_list if item not in exclude]

def load_predictors_and_cofactors(json_path="../utils/predictors_and_cofactors.json"):
    """
    Load predictors and cofactors from a JSON file.
    First checks for temporary file (created by dummy variables), then falls back to original.

    Parameters:
    -----------
    json_path : str
        Path to the JSON file containing predictors and cofactors

    Returns:
    --------
    tuple
        A tuple containing (predictors, cofactors) lists
    """
    # Check if temporary file exists (created when dummy variables are used)
    temp_json_path = json_path.replace("predictors_and_cofactors.json", "predictors_and_cofactors_temp.json")

    if os.path.exists(temp_json_path):
        # Load from temporary file with updated cofactor names
        print(f"Loading predictors/cofactors from temporary file: {temp_json_path}")
        with open(temp_json_path, 'r') as file:
            data = json.load(file)
    else:
        # Load from original JSON file
        with open(json_path, 'r') as file:
            data = json.load(file)

    # Extract predictors and cofactors lists, use empty lists as default
    predictors = data.get('predictors', [])
    cofactors = data.get('cofactors', [])

    return predictors, cofactors

def prepare_dataset(dataset, id_column, time_column, exclude=[]):
    """
    Prepare the dataset for the model.

    This function does the following:
    - Rename columns to ID and TIME based on provided parameters
    - Load predictors and cofactors from JSON file
    - Select all required columns (both predictors and cofactors)
    - Make id as a string
    - Remove rows with NaN values in the TIME column
    - Count visits per ID
    - Get IDs with more than one visit
    - Filter dataset to include only patients with multiple visits

    Parameters:
    -----------
    dataset : pandas.DataFrame
        The dataset to prepare
    id_column : str
        Name of the column to use as ID
    time_column : str
        Name of the column to use as TIME
    exclude : list, optional
        List of columns to exclude

    Returns:
    --------
    pandas.DataFrame
        A copy of the input dataframe with the required columns
    """
    # Load predictors and cofactors configuration from JSON file
    predictors, cofactors = load_predictors_and_cofactors()

    # Standardize column names: rename specified columns to 'ID' and 'TIME'
    dataset = dataset.rename(columns={id_column: 'ID', time_column: 'TIME'})

    # Prepare column lists for data selection
    predictor_columns = predictors + ['ID', 'TIME']  # Include ID and TIME with predictors
    cofactor_columns = cofactors

    # Remove excluded columns if specified
    if exclude:
        predictor_columns = remove_unnecessary_values_from_list(predictor_columns, exclude)
        cofactor_columns = remove_unnecessary_values_from_list(cofactor_columns, exclude)

    # Combine all required columns and remove duplicates
    columns_to_keep = list(set(predictor_columns + cofactor_columns))

    # Filter to keep only columns that actually exist in the dataset
    available_columns = [col for col in columns_to_keep if col in dataset.columns]
    dataset = dataset[available_columns]

    # Convert ID to string type for consistency
    dataset['ID'] = dataset['ID'].astype('str')

    # Remove rows with missing TIME values (essential for longitudinal analysis)
    dataset = dataset.dropna(subset=['TIME'])

    # Count the number of visits per patient
    visits_count = dataset.groupby('ID').size()

    # Identify patients with multiple visits (required for longitudinal modeling)
    multi_visit_ids = visits_count[visits_count > 1].index.tolist()

    # Keep only patients with multiple visits for longitudinal analysis
    dataset = dataset[dataset['ID'].isin(multi_visit_ids)]

    return dataset

def convert_cofactors_to_dummy(dataset, create_dummies=True, utils_path="../utils/"):
    """
    Convert cofactors to dummy variables or integers based on the create_dummies parameter.

    This function:
    - Loads cofactors from JSON file
    - If create_dummies=True: Creates dummy variables and saves new names to temp JSON
    - If create_dummies=False: Converts boolean cofactors to integers (0,1)
    - Returns updated dataset and cofactor column names

    Parameters:
    -----------
    dataset : pandas.DataFrame
        The dataset containing cofactors
    create_dummies : bool, optional
        If True, create dummy variables. If False, only convert booleans to integers (default=True)
    utils_path : str, optional
        Path to utils folder for saving temp JSON file (default="../utils/")

    Returns:
    --------
    tuple
        (updated_dataset, cofactor_names) where cofactor_names is the list of cofactor column names
    """
    # Load cofactors configuration from JSON (ignore predictors with _)
    predictors, cofactors = load_predictors_and_cofactors()

    dataset_copy = dataset.copy()  # Work on a copy to preserve original data
    cofactor_names = []  # Track cofactor column names

    # Process each cofactor column
    for cofactor in cofactors:
        if cofactor in dataset_copy.columns:
            if create_dummies:
                # Create dummy variables for categorical cofactors
                # Get unique values for this cofactor (excluding NaN)
                unique_values = dataset_copy[cofactor].dropna().unique()

                # Create dummy variables with format: cofactor_name/category_value
                dummies = pd.get_dummies(dataset_copy[cofactor], prefix=cofactor, prefix_sep='/')

                # Convert boolean dummy variables to integer (0 and 1)
                dummies = dummies.astype(int)

                # Add new dummy columns to the dataset
                dataset_copy = pd.concat([dataset_copy, dummies], axis=1)

                # Remove the original categorical cofactor column
                dataset_copy = dataset_copy.drop(columns=[cofactor])

                # Keep track of new dummy column names for later reference
                cofactor_names.extend(dummies.columns.tolist())
            else:
                # Only convert boolean cofactors to integers, keep original column
                if dataset_copy[cofactor].dtype == 'bool':
                    # Convert boolean to integer (True->1, False->0)
                    dataset_copy[cofactor] = dataset_copy[cofactor].astype(int)

                # Keep the original cofactor name
                cofactor_names.append(cofactor)

    # If dummy variables were created, save the new column names to temp JSON file
    if create_dummies and cofactor_names:
        temp_json_data = {
            "predictors": predictors,  # Keep predictors unchanged
            "cofactors": cofactor_names  # Use new dummy variable names
        }
        temp_json_path = utils_path + "predictors_and_cofactors_temp.json"
        with open(temp_json_path, 'w') as file:
            json.dump(temp_json_data, file, indent=4)
        print(f"Temporary predictor/cofactor names saved to: {temp_json_path}")

    return dataset_copy, cofactor_names

def cleanup_temp_predictors_cofactors_file(utils_path="../utils/"):
    """
    Delete the temporary predictors and cofactors JSON file if it exists.

    Parameters:
    -----------
    utils_path : str, optional
        Path to utils folder containing temp JSON file (default="../utils/")

    Returns:
    --------
    bool
        True if file was deleted, False if file didn't exist
    """
    temp_json_path = utils_path + "predictors_and_cofactors_temp.json"

    if os.path.exists(temp_json_path):
        os.remove(temp_json_path)
        print(f"Temporary predictor/cofactor file deleted: {temp_json_path}")
        return True
    else:
        return False

def load_normalization_settings(json_path="../utils/normalization_settings.json"):
    """
    Load normalization settings from a JSON file.

    Parameters:
    -----------
    json_path : str
        Path to the JSON file containing normalization settings

    Returns:
    --------
    dict
        Dictionary containing normalization settings
    """
    # Load normalization configuration from JSON file
    with open(json_path, 'r') as file:
        data = json.load(file)
    return data

def normalize_column_min_max(column, min_val=None, max_val=None):
    """
    Normalize a column between 0 and 1 using min-max scaling.

    Parameters:
    -----------
    column : pandas.Series
        The column to normalize
    min_val : float, optional
        Minimum value for normalization. If None, uses column min
    max_val : float, optional
        Maximum value for normalization. If None, uses column max

    Returns:
    --------
    pandas.Series
        Normalized column
    """
    # Use column's own min/max if not specified
    if min_val is None:
        min_val = column.min()
    if max_val is None:
        max_val = column.max()

    # Handle edge case: if all values are the same, return neutral value (0.5)
    if max_val == min_val:
        return pd.Series([0.5] * len(column), index=column.index)

    # Apply standard min-max normalization: (x - min) / (max - min)
    return (column - min_val) / (max_val - min_val)

def normalize_column_range(column, min_val, max_val, mode='normal'):
    """
    Normalize a column between 0 and 1 using specified range and mode.

    Parameters:
    -----------
    column : pandas.Series
        The column to normalize
    min_val : float
        Minimum value of the range
    max_val : float
        Maximum value of the range
    mode : str
        'normal' or 'inverse' normalization mode

    Returns:
    --------
    pandas.Series
        Normalized column
    """
    # Handle edge case: if min equals max, return neutral value (0.5)
    if max_val == min_val:
        return pd.Series([0.5] * len(column), index=column.index)

    # Apply min-max normalization with specified range
    normalized = (column - min_val) / (max_val - min_val)

    # Apply inverse normalization if specified (flip 0s and 1s)
    if mode == 'inverse':
        normalized = 1 - normalized

    return normalized

def normalize_dataset(dataset):
    """
    Normalize dataset columns based on settings from JSON file.

    This function:
    - Loads normalization settings from JSON file
    - Normalizes columns in norm_range using specified ranges and modes
    - Normalizes columns in norm_min_max using column's own min/max
    - Returns the normalized dataset

    Parameters:
    -----------
    dataset : pandas.DataFrame
        The dataset to normalize

    Returns:
    --------
    pandas.DataFrame
        Normalized dataset
    """
    # Load normalization configuration from JSON file
    settings = load_normalization_settings()

    dataset_copy = dataset.copy()  # Work on a copy to preserve original data

    # Process columns requiring normalization with custom ranges
    norm_range = settings.get('norm_range', [])  # Columns to normalize with custom ranges
    range_values = settings.get('range_values', {})  # Custom range specifications

    # Apply custom range normalization to specified columns
    for column in norm_range:
        if column in dataset_copy.columns and column in range_values:
            min_val, max_val, mode = range_values[column]  # Extract [min, max, mode] from config
            dataset_copy[column] = normalize_column_range(
                dataset_copy[column], min_val, max_val, mode
            )

    # Apply standard min-max normalization to specified columns
    norm_min_max = settings.get('norm_min_max', [])  # Columns for standard min-max scaling

    for column in norm_min_max:
        if column in dataset_copy.columns:
            # Use column's own min/max values for normalization
            dataset_copy[column] = normalize_column_min_max(dataset_copy[column])

    return dataset_copy