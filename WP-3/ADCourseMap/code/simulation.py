import pandas as pd
from utils import load_predictors_and_cofactors, remove_unnecessary_values_from_list
from leaspy import Data

def simulate_data(model, dataset, gen_path, opt):
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
    print('SIMULATION\n')

    dataset = dataset.set_index(['ID', 'TIME'])

    # Load predictors and cofactors (will use temp file if dummies were created)
    predictors, cofactors = load_predictors_and_cofactors()
    predictor_columns = remove_unnecessary_values_from_list(predictors, ['DX/CN', 'DX/Dementia', 'DX/MCI'])
    cofactor_columns = remove_unnecessary_values_from_list(cofactors, ['AGE'])

    # Prepare dataset for Leaspy (similar to train_val.py approach)
    print("Preparing dataset for simulation...")

    # Create predictor dataset with [ID, TIME] multi-index
    df_predictors = dataset[predictor_columns].copy()

    # Create cofactor dataset with only ID as index, one row per patient
    df_cofactors = dataset.reset_index()[['ID'] + cofactor_columns].copy()
    df_cofactors = df_cofactors.drop_duplicates(subset=['ID']).set_index('ID')

    print(f'Simulation predictors dataset shape: {df_predictors.shape}')
    print(f'Simulation cofactors dataset shape: {df_cofactors.shape}')

    # Create Leaspy Data object
    print('LEASPY SIMULATION DATASET CREATION...')
    data_sim = Data.from_dataframe(df_predictors)
    data_sim.load_cofactors(df=df_cofactors, cofactors=cofactor_columns)

    # Generate synthetic data using the processed Leaspy Data object
    print('GENERATING SYNTHETIC DATA...')
    #df_simu = model.generate_virtual_data(data_sim)
    df_simu = model.generate_virtual_data_with_cofactors(data_sim, opt.n_gen_sub, opt.n_gen_visit, opt.n_gen_std, opt.merge_generations)

    # Save synthetic data
    if isinstance(df_simu, list):
        # Multiple datasets - save each separately
        for i, dataset in enumerate(df_simu):
            filename = f'synthetic_data_generation_{i+1}.csv'
            dataset.to_csv(gen_path + filename)
            print(f'Generation {i+1} saved to: {gen_path}{filename} ({len(dataset)} samples)')
    else:
        # Single combined dataset
        df_simu.to_csv(gen_path + 'synthetic_data.csv')
        print(f'Combined synthetic data saved to: {gen_path}synthetic_data.csv')
        print(f'Generated {len(df_simu)} synthetic samples')

    return df_simu