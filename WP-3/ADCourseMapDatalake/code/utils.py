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
from dl_client import DatalakeClient

# ---------------------------------------------------------- FUNCTIONS --------------------------------------------------------- #

# Function to append text to the file
def append_to_text_file(file_name, content):
    with open(file_name, "a") as file:
        file.write("\n" + content)

# Function to create a json for the calibration settings
def make_json_calibration_settings_dict(opt, json_path):
    settings = {
        "name": "mcmc_saem",
        "seed": opt.manual_seed,
        "algorithm_initialization_method": None,
        "model_initialization_method": "default",
        "device": opt.device,
        "parameters": {
            "progress_bar": True,
            "n_iter": opt.n_iter,
            "n_burn_in_iter": None,
            "n_burn_in_iter_frac": opt.n_burn_in_iter_frac,
            "burn_in_step_power": opt.burn_in_step_power,
            "random_order_variables": True,
            "sampler_ind": "Gibbs",
            "sampler_ind_params": {
                "acceptation_history_length": 25,
                "mean_acceptation_rate_target_bounds": [0.25, 0.5],
                "adaptive_std_factor": 0.25
            },
            "sampler_pop": "Gibbs",
            "sampler_pop_params": {
                "random_order_dimension": True,
                "acceptation_history_length": 25,
                "mean_acceptation_rate_target_bounds": [0.25, 0.5],
                "adaptive_std_factor": 0.25
            },
            "annealing": {
                "do_annealing": True,
                "initial_temperature": 100,
                "n_plateau": 100,
                "n_iter": 500,
                "n_iter_frac": None
            }
        } 
    }

    with open(json_path + 'algorithm_settings_calibration.json', 'w') as file:
        json.dump(settings, file, indent=4)

# Function to split the dataset into train, validation
def split_dataset_in_train_val(dataset, percentage):

    # Count visits per RID
    visits_count = dataset.groupby('ID').size()
    
    # Create a dictionary that groups RIDs by number of visits
    sub_per_visits = {}
    for rid, count in visits_count.items():
        if count not in sub_per_visits:
            sub_per_visits[count] = []
        sub_per_visits[count].append(rid)

    train_id = []
    for count in sub_per_visits:
        sublist_id = np.random.choice(sub_per_visits[count], size=int(percentage*len(sub_per_visits[count])), replace=False).tolist()
        train_id += sublist_id

    df_train = dataset[dataset['ID'].isin(train_id)].copy()
    df_val = dataset[~dataset['ID'].isin(train_id)].copy()

    # Get from the test dataset a dataset to predict 
    df_to_pred = df_val.groupby('ID').tail(1).copy() # Get last visit of every patient

    list_index = df_to_pred.index.tolist()
    df_pers = df_val[~df_val.index.isin(list_index)].copy() # Evaluation dataset

    # Set multi index
    df_train = df_train.set_index(['ID', 'TIME'])
    df_pers = df_pers.set_index(['ID', 'TIME'])
    df_to_pred = df_to_pred.set_index(['ID', 'TIME'])

    return df_train, df_val, df_pers, df_to_pred

def load_dataset_from_datalake(metadata):
    client = DatalakeClient()

    search = client.query_files(
        query=metadata
    )

    zip_files = client.download_file(
        search['object_name'],  
        extract_zip=True
    )

    file_name = list(zip_files.keys())[0]
    dataset = zip_files[file_name]

    return dataset

def load_dataset_to_datalake(metadata, data, filename, prefix):
    client = DatalakeClient()

    try:
        search = client.upload_dataframe(
            df=data,
            object_name=filename,
            prefix=prefix,
            metadata=metadata
        )

        return True
    except Exception as e:
        print('Upload on datalake failed: ', e)
        return False

def get_dataset_custom_metadata(metadata):
    client = DatalakeClient()

    search = client.search_files(
        query=metadata
    )
    
    return search['files'][0]['custom']

def get_dataset_predictors(level, file_code):
    metadata = {
        'custom.level' : level,
        'custom.file_code': file_code
    }

    custom_metadata = get_dataset_custom_metadata(metadata)

    return custom_metadata['predittori']
    
def get_dataset_cofactors(level, file_code):
    metadata = {
        'custom.level' : level,
        'custom.file_code': file_code
    }

    custom_metadata = get_dataset_custom_metadata(metadata)

    return custom_metadata['cofattori'] + ['DX/CN', 'DX/Dementia', 'DX/MCI']

def get_scale_normalization_columns(level, file_code):
    metadata = {
        'custom.level' : level,
        'custom.file_code': file_code
    }

    custom_metadata = get_dataset_custom_metadata(metadata)

    return custom_metadata['norm_scala']
    
def get_dataset_minmax_normalization_columns(level, file_code):
    metadata = {
        'custom.level' : level,
        'custom.file_code': file_code
    }

    custom_metadata = get_dataset_custom_metadata(metadata)

    return custom_metadata['norm_intervallo']

def get_volumes_normalization_columns(level, file_code):
    metadata = {
        'custom.level' : level,
        'custom.file_code': file_code
    }

    custom_metadata = get_dataset_custom_metadata(metadata)

    return custom_metadata['norm_volume']

def get_volumes_normalization_values(level, file_code):
    metadata = {
        'custom.level' : level,
        'custom.file_code': file_code
    }

    custom_metadata = get_dataset_custom_metadata(metadata)

    return custom_metadata['volume_norm_values']

def get_scale_normalization_values(level, file_code):
    metadata = {
        'custom.level' : level,
        'custom.file_code': file_code
    }

    custom_metadata = get_dataset_custom_metadata(metadata)

    return custom_metadata['norm_scale_value']

def remove_unnecessary_values_from_list(list_values, unnecessary_values):
    return [value for value in list_values if value not in unnecessary_values]

def prepare_dataset(dataset, predictors, cofactors, exclude = [], prediction_mode = False):
    """
    Prepare the dataset for the model.
    
    This function does the following:
    - Rename columns if necessary (RID→ID, AGE→TIME)
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
    
    Returns:
    --------
    pandas.DataFrame
        A copy of the input dataframe with the required columns
    """
    # Rename columns if necessary (RID→ID, AGE→TIME)
    dataset = dataset.rename(columns={'RID': 'ID'})
    dataset = dataset.rename(columns={'AGE': 'TIME'})

    # Select all required columns (both predictors and cofactors)
    # Predictors
    predictor_columns = predictors + ['ID', 'TIME']
    # Cofactors
    cofactor_columns = cofactors

    if exclude:
        predictor_columns = remove_unnecessary_values_from_list(predictor_columns, exclude)
        cofactor_columns = remove_unnecessary_values_from_list(cofactor_columns, exclude)

    # Combine all columns to keep
    columns_to_keep = list(set(predictor_columns + cofactor_columns))
    
    # Check if the columns exist in the dataset
    available_columns = [col for col in columns_to_keep if col in dataset.columns]
    dataset = dataset[available_columns]
    
    # Make id as a string
    dataset['ID'] = dataset['ID'].astype('str') 
    
    # Remove rows with NaN values in the TIME column
    dataset = dataset.dropna(subset=['TIME'])
    
    if not prediction_mode:
        # Count visits per ID
        visits_count = dataset.groupby('ID').size()

        # Get IDs with more than one visit
        multi_visit_ids = visits_count[visits_count > 1].index.tolist()
        
        # Filter dataset to include only patients with multiple visits
        dataset = dataset[dataset['ID'].isin(multi_visit_ids)]

    return dataset

def normalize_fixed_scale(df, columns, min_max_scales):
    """
    Normalize specified columns of a dataframe to a range [0, 1] using fixed scale boundaries.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe to normalize
    columns : list
        List of column names to normalize
    min_max_scales : dict
        Dictionary mapping column names to lists containing [min_value, max_value, method]
        where method is either 'increasing' (min→0, max→1) or 'inverse' (min→1, max→0)
    
    Returns:
    --------
    pandas.DataFrame
        A copy of the input dataframe with normalized columns
    """
    df_normalized = df.copy()
    
    for col in columns:
        # Get min, max values and normalization method for the column
        min_val, max_val, method = min_max_scales[col]
        
        # Skip normalization if min and max are the same (avoid division by zero)
        if min_val == max_val:
            print(f"Warning: min and max are equal for column {col}. Skipping normalization.")
            continue
        
        # Normalize the column based on the specified method
        if method == "inverse":
            # Inverse normalization: min→1, max→0
            df_normalized[col] = 1 - (df[col] - min_val) / (max_val - min_val)
        else:  # method == "increasing" or any other value defaults to increasing
            # Standard normalization: min→0, max→1
            df_normalized[col] = (df[col] - min_val) / (max_val - min_val)
        
        # Clip values to [0, 1] in case there are values outside the scale
        df_normalized[col] = np.clip(df_normalized[col], 0, 1)
    
    return df_normalized


def normalize_dataset_minmax(df, columns):
    """
    Normalize specified columns of a dataframe to a range [0, 1] using min and max values from the dataset.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe to normalize
    columns : list
        List of column names to normalize
    
    Returns:
    --------
    pandas.DataFrame
        A copy of the input dataframe with normalized columns
    """
    df_normalized = df.copy()
    
    for col in columns:
        # Get min and max values from the dataset
        min_val = df[col].min()
        max_val = df[col].max()
        
        # Skip normalization if min and max are the same (avoid division by zero)
        if min_val == max_val:
            print(f"Warning: min and max are equal for column {col}. Skipping normalization.")
            continue
        
        # Normalize the column
        df_normalized[col] = (df[col] - min_val) / (max_val - min_val)
    
    return df_normalized


def normalize_dataset_volumes(df, columns, normalization_values):
    """
    Normalize specified volumes based on ICV volume and the min and max percentage values.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe to normalize
    columns : list
        List of column names to normalize
    normalization_values : dict
        Dictionary mapping column names to lists containing [min_value, max_value, method]
        where method is either 'increasing' (min→0, max→1) or 'inverse' (min→1, max→0)
    
    Returns:
    --------
    pandas.DataFrame
        A copy of the input dataframe with normalized columns
    """

    df_normalized = df.copy()
    
    for col in columns:
        if col == 'ICV' or col == "ICV%ICV":
            continue

        # Get min, max values and normalization method for the column
        min_val, max_val, method = normalization_values[col]

        # Normalize the column based on the specified method
        if method == "inverse":
            # Inverse normalization: min→1, max→0
            df_normalized[col] = 1 - (df[col] - min_val) / (max_val - min_val)
        else:  # method == "increasing" or any other value defaults to increasing
            # Standard normalization: min→0, max→1
            df_normalized[col] = (df[col] - min_val) / (max_val - min_val)
        
        # Clip values to [0, 1] in case there are values outside the scale
        df_normalized[col] = np.clip(df_normalized[col], 0, 1)
    
    return df_normalized

def clean_generated_dataframe(df_generated):
    """
    Clean the generated dataframe by removing prefixes and metadata columns.

    This function:
    - Removes 'generated_' prefix from column names (where present)
    - Removes 'generation_id' column if present
    - Returns the cleaned dataframe

    Parameters:
    -----------
    df_generated : pandas.DataFrame
        The generated synthetic dataframe to clean

    Returns:
    --------
    pandas.DataFrame
        Cleaned dataframe ready for saving
    """
    df_cleaned = df_generated.copy()  # Work on a copy to preserve original data

    # Remove 'generated_' prefix from column names
    df_cleaned.columns = [col.replace('generated_', '') if col.startswith('generated_') else col
                          for col in df_cleaned.columns]

    # Remove generation_id column if present
    if 'generation_id' in df_cleaned.columns:
        df_cleaned = df_cleaned.drop(columns=['generation_id'])

    return df_cleaned