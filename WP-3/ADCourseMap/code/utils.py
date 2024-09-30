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
                "mean_acceptation_rate_target_bounds": [0.2, 0.4],
                "adaptive_std_factor": 0.1
            },
            "sampler_pop": "Gibbs",
            "sampler_pop_params": {
                "random_order_dimension": True,
                "acceptation_history_length": 25,
                "mean_acceptation_rate_target_bounds": [0.2, 0.4],
                "adaptive_std_factor": 0.1
            },
            "annealing": {
                "do_annealing": False,
                "initial_temperature": 100,
                "n_plateau": 10,
                "n_iter": None,
                "n_iter_frac": 0.5
            }
        } 
    }

    with open(json_path + 'algorithm_settings_calibration.json', 'w') as file:
        json.dump(settings, file, indent=4)

# Function to split the dataset into train, validation
def split_dataset_in_train_val(dataset, percentage):

    # Visit Distribution
    visit_distribution = dataset.groupby('ID').count()
    visit_distribution = visit_distribution[['TIME']].rename(columns={'TIME':'N visits'})

    # Create a dictionary that collects subject IDs by number of visits
    sub_per_visits = []
    for i in range(visit_distribution['N visits'].max()+1):
        sub_per_visits.append([])

    # Populate sub_per_visits based on the number of visits for each subject
    for id in dataset['ID'].unique():
        i = int(id)
        n = visit_distribution.iloc[i][0]
        sub_per_visits[n].append(id)

    train_id = []
    for n in range(len(sub_per_visits)):
        if sub_per_visits[n]:
            sublist_id = np.random.choice(sub_per_visits[n], size=int(percentage*len(sub_per_visits[n])), replace=False).tolist()
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