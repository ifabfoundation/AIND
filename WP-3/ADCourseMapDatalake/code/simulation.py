import pandas as pd
from utils import *
from leaspy import Data

def simulate_data(model, dataset, opt):
    """
    Generate synthetic data based on existing dataset distribution.
    The dataset should already be processed by the loader (including dummy conversion if needed).

    Parameters:
    -----------
    model : LeaspyModel
        The trained Leaspy model instance
    dataset : pandas.DataFrame
        The processed dataset (already loaded and transformed by loader)
    gen_path : str
        Path to save generated synthetic data
    opt : argparse.Namespace
        Command line arguments

    Returns:
    --------
    pandas.DataFrame
        The generated synthetic dataset
    """
    print("SIMULATION...\n")

    # Set multi-index
    df_train, df_val, _, _ = split_dataset_in_train_val(dataset, 0.8)
    #df_train = dataset.set_index(['ID', 'TIME'])

    # Get predictors and cofactors
    predictor_columns = remove_unnecessary_values_from_list(get_dataset_predictors(opt.level, opt.file_code), ['DX/CN', 'DX/Dementia', 'DX/MCI', "ICV%ICV"])
    cofactor_columns = remove_unnecessary_values_from_list(get_dataset_cofactors(opt.level, opt.file_code), ['AGE'])

    # Prepare dataset for Leaspy (similar to train_val.py approach)
    print("Preparing dataset for simulation...")

    # Create predictor dataset with [ID, TIME] multi-index
    df_predictors = df_train[predictor_columns].copy()

    # Create cofactor dataset with only ID as index, one row per patient
    df_cofactors = df_train.reset_index()[['ID'] + cofactor_columns].copy()
    df_cofactors = df_cofactors.drop_duplicates(subset=['ID']).set_index('ID')

    print(f'Simulation predictors dataset shape: {df_predictors.shape}')
    print(f'Simulation cofactors dataset shape: {df_cofactors.shape}')

    # Create Leaspy Data object
    print('LEASPY SIMULATION DATASET CREATION...')
    data_sim = Data.from_dataframe(df_predictors)
    data_sim.load_cofactors(df=df_cofactors, cofactors=cofactor_columns)

    # Generate synthetic data using the processed Leaspy Data object
    print('GENERATING SYNTHETIC DATA...')
    if opt.same_data_stats == 1:
        # Get stats from validation dataset
        df_stats = df_train.reset_index()
        n_visit = df_stats.groupby('ID').count()[['TIME']].rename(columns={'TIME':'N visits'})
        n_gen_visit = n_visit['N visits'].mean()
        n_gen_std = n_visit['N visits'].std()
        n_gen_sub = len(n_visit)
    else:
        # Get stats from command line arguments
        n_gen_visit = opt.n_gen_visit
        n_gen_std = opt.n_gen_std
        n_gen_sub = opt.n_gen_sub

    #df_simu = model.generate_virtual_data_with_cofactors(data_sim, n_gen_sub, n_gen_visit, n_gen_std, opt.merge_generations)
    #df_simu = model.generate_virtual_data(data_sim)
    df_simu = model.generate_virtual_data_with_diagnosis(data_sim, dx_percentages={'CN': 0.3, 'dementia': 0.25, 'MCI': 0.45}
)

    print('SAVING SYNTHETIC DATA...\n')
    now = datetime.now()

    if isinstance(df_simu, list):
        """
        # Multiple datasets - save each separately
        for i, dataset in enumerate(df_simu):
            # Used to create different names
            data = str(now.month) + '_' + str(now.day) + '_' + str(now.hour) + '_' + str(now.minute)

            filename = f'synthetic_data_generation_{i}_{data}.csv'
            metadata = {
                'file_code': opt.file_code + '_synthetic_' + str(i),
                'pers_dataset': opt.file_code,
                'level': opt.level,
                'n_gen_sub': n_gen_sub,
                'n_gen_visit': n_gen_visit,
                'n_gen_std': n_gen_std,
            }

            load_dataset_to_datalake(metadata, dataset, filename, 'synthetic')
            print(f'Generation {i+1} saved ({len(dataset)} samples)')
        """
        print("SKIPPED THE SINGLE DATASETS FLOW")
    else:
        # Single combined dataset
        # Used to create different names
        #data = str(now.month) + '_' + str(now.day) + '_' + str(now.hour) + '_' + str(now.minute)
        data = 'utility'
        
        filename_synthetic = f'synthetic_data_generation_{data}.csv'
        metadata_synthetic = {
            'file_code': opt.file_code + '_synthetic_utility',
            'pers_dataset': opt.file_code,
            'level': opt.level,
            'n_gen_sub': n_gen_sub,
            'n_gen_visit': n_gen_visit,
            'n_gen_std': n_gen_std,
        }

        df_simu = clean_generated_dataframe(df_simu)
        df_simu = df_simu.reset_index()
        if 'index' in df_simu.columns:
            df_simu = df_simu.drop('index', axis=1)
        load_dataset_to_datalake(metadata_synthetic, df_simu, filename_synthetic, 'synthetic')

        # Save validation dataset (to validate the generation)
        #filename_source = f'dataset_for_validation.csv'
        filename_source = f'dataset_to_test_utility.csv'
        metadata_data_for_validation = {
            'file_code': opt.file_code + '_to_test_utility',
            'level': opt.level,
        }
        
        df_for_validation = data_sim.to_dataframe(cofactors=['DX/CN', 'DX/Dementia', 'DX/MCI'])
        df_for_validation = df_for_validation.reset_index()
        if 'index' in df_for_validation.columns:
            df_for_validation = df_for_validation.drop('index', axis=1)

        load_dataset_to_datalake(metadata_data_for_validation, df_for_validation, filename_source, 'validation')
        
        # Compare datasets distribution
        # model.public_compare_datasets_distribution(df_train.set_index(['ID', 'TIME']), df_simu.set_index(['ID', 'TIME']))

        """
        filename_source = f'synthetic_data_source_{data}.csv'
        metadata_data_source = {
            'file_code': opt.file_code + '_synthetic_source',
            'level': opt.level,
            'synthetic_data_path': 'synthetic/' + filename_synthetic,
        }
        load_dataset_to_datalake(metadata_data_source, dataset.reset_index(), filename_source, 'synthetic/source')
        """
        print(f'Combined synthetic data saved')
        print(f'Generated {len(df_simu)} synthetic samples')

    return df_simu