import pandas as pd
from utils import (
    load_dataset_from_datalake, 
    prepare_dataset, 
    normalize_fixed_scale, 
    normalize_dataset_minmax, 
    get_dataset_predictors, 
    get_dataset_cofactors,
    get_scale_normalization_values,
    get_scale_normalization_columns,
    get_dataset_minmax_normalization_columns,
    get_volumes_normalization_columns,
    get_volumes_normalization_values,
    normalize_dataset_volumes
)

def loader(opt):

    # Load metadata
    metadata = {
        'custom.level' : opt.level,
        'custom.file_code': opt.file_code
    }

    # Load dataset
    dataset = load_dataset_from_datalake(metadata)

    # Get predictors and cofactors
    predictors = get_dataset_predictors(opt.level, opt.file_code)
    cofactors = get_dataset_cofactors(opt.level, opt.file_code)

    #dataset = prepare_dataset(dataset, predictors, cofactors, exclude=['AGE', 'DX/CN', 'DX/Dementia', 'DX/MCI', "ICV%ICV"], prediction_mode=opt.prediction)  # Prepare dataset
    dataset = prepare_dataset(dataset, predictors, cofactors, exclude=['AGE', "ICV%ICV"], prediction_mode=opt.prediction)  # Prepare dataset
    
    # Convert boolean columns to integer
    for col in cofactors:
        if col in dataset.columns and dataset[col].dtype == 'bool':
            dataset[col] = dataset[col].astype(int)

    # Normalize sub-columns with fixed scale
    column_list_to_normalize_fixed_scale = get_scale_normalization_columns(opt.level, opt.file_code)
    fixed_min_max_scales = get_scale_normalization_values(opt.level, opt.file_code)
    dataset = normalize_fixed_scale(dataset, column_list_to_normalize_fixed_scale, fixed_min_max_scales)

    # Normalize sub-columns with min-max scale
    #column_list_to_normalize_minmax_scale = get_dataset_minmax_normalization_columns(opt.level, opt.file_code)
    #dataset = normalize_dataset_minmax(dataset, column_list_to_normalize_minmax_scale)

    # Normalize volumes
    column_list_to_normalize_volumes = get_volumes_normalization_columns(opt.level, opt.file_code)
    volumes_normalization_values = get_volumes_normalization_values(opt.level, opt.file_code)
    dataset = normalize_dataset_volumes(dataset, column_list_to_normalize_volumes, volumes_normalization_values)
    
    # Handle subsets if required
    if opt.sub_data:
        sub_n = opt.sub_n

        vis_n = dataset[dataset['ID']==sub_n].index[-1]+1 # number of visits (rows)
        dataset = dataset.iloc[:vis_n]

    return dataset