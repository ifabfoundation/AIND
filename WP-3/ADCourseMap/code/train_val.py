from utils import *
from loss import MAE_CI
from leaspy import Data, IndividualParameters
import numpy as np
import os

def train_and_validate(dataset, model, opt, logs_path, model_path, validation_path, settings_path):
    """
    Train the model and validate predictions.
    Includes error handling to ensure temp files are cleaned up even if training fails.

    Parameters:
    -----------
    dataset : pandas.DataFrame
        The complete dataset
    model : LeaspyModel
        The Leaspy model instance
    opt : argparse.Namespace
        Command line arguments
    logs_path : str
        Path for logs
    model_path : str
        Path for model weights
    validation_path : str
        Path for validation results
    settings_path : str
        Path for settings/utils folder

    Returns:
    --------
    tuple
        (parameters, source_dimension, noise_std, mae_results)
    """
    try:
        # Create validation path if it doesn't exist
        os.makedirs(validation_path, exist_ok=True)

        # Split dataset
        df_train, df_val, df_pers, df_to_pred = split_dataset_in_train_val(dataset, 0.8)

        # Load predictors and cofactors from JSON and filter unwanted columns
        predictors, cofactors = load_predictors_and_cofactors()
        predictor_columns = remove_unnecessary_values_from_list(predictors, ['DX/CN', 'DX/Dementia', 'DX/MCI'])
        cofactor_columns = remove_unnecessary_values_from_list(cofactors, ['AGE'])

        print('TRAIN SAMPLES: ' + str(len(df_train)))
        print('PERSONALIZATION SAMPLES: ' + str(len(df_pers)))
        print('VALIDATION SAMPLES: ' + str(len(df_to_pred)))

        # Create separate datasets as requested
        print('\nCreating separate datasets with different index structures...')

        # For df_train, create two separate datasets
        # 1. Dataset with [ID, TIME] as multi-index and predictor columns
        df_train_predictors = df_train[predictor_columns].copy()

        # 2. Dataset with only ID as index and cofactor columns, one row per patient
        # Reset the index to get ID and TIME as columns
        df_train_cofactors = df_train.reset_index()[['ID'] + cofactor_columns].copy()
        # Drop duplicates to keep only one row per patient
        df_train_cofactors = df_train_cofactors.drop_duplicates(subset=['ID']).set_index('ID')

        print(f'Train predictors dataset shape: {df_train_predictors.shape}')
        print(f'Train cofactors dataset shape: {df_train_cofactors.shape}')

        # For df_pers, create two separate datasets
        # 1. Dataset with [ID, TIME] as multi-index and predictor columns
        df_pers_predictors = df_pers[predictor_columns].copy()

        # 2. Dataset with only ID as index and cofactor columns, one row per patient
        # Reset the index to get ID and TIME as columns
        df_pers_cofactors = df_pers.reset_index()[['ID'] + cofactor_columns].copy()
        # Drop duplicates to keep only one row per patient
        df_pers_cofactors = df_pers_cofactors.drop_duplicates(subset=['ID']).set_index('ID')

        print(f'Personalization predictors dataset shape: {df_pers_predictors.shape}')
        print(f'Personalization cofactors dataset shape: {df_pers_cofactors.shape}')

        # Create Data Object to use the dataset with leaspy
        print('LEASPY TRAIN DATASET CREATION...')
        data_train = Data.from_dataframe(df_train_predictors)
        data_train.load_cofactors(df=df_train_cofactors, cofactors=cofactor_columns)

        print('LEASPY PERSONALIZATION DATASET CREATION...')
        data_pers = Data.from_dataframe(df_pers_predictors)
        data_pers.load_cofactors(df=df_pers_cofactors, cofactors=cofactor_columns)

        print(data_train.to_dataframe(cofactors='all'))

        print('TRAIN\n')

        # Train data and save population curves
        parameters, source_dimension, noise_std = model.forward(data_train)

        # Get fitting (average) parameters
        mean_xi = parameters['xi_mean'].tolist()
        mean_tau = parameters['tau_mean'].tolist()
        mean_source = parameters['sources_mean'].tolist()
        number_of_sources = source_dimension
        mean_sources = [mean_source]*number_of_sources

        average_parameters = {
            'xi': mean_xi,
            'tau': mean_tau,
            'sources': mean_sources,
        }

        # Create an Individual Parameters Object to estimate biomarkers for a 'mean patient'
        ip_average = IndividualParameters()
        ip_average.add_individual_parameters('average', average_parameters)

        # Save fitting parameters
        ip_average.save(model_path + 'average_parameters.json')

        # Save training noise results - FIXED: use df_train_predictors.columns
        for idx, col in enumerate(df_train_predictors.columns):
            col_noise = round(noise_std[idx] * 100, 2)
            with open(logs_path + '/train_noise.txt', 'a') as file:
                file.write(col + ': ' + str(col_noise) + '%\n')

        print('VALIDATION\n')

        # Performance evaluation
        df_predictions = model.estimate(data_pers, df_to_pred)

        # Take only predictors to calculate MAE
        df_to_pred_predictors = df_to_pred[predictor_columns].copy()
        # Save validation predictors
        df_to_pred_predictors.to_csv(validation_path + 'validation_predictions.csv')

        # Calculate MAE
        mae_metric = MAE_CI()
        mae = mae_metric.calculate(df_to_pred_predictors, df_predictions)

        # Save validation MAE
        for name in mae.keys():
            # Check if the MAE is NaN (no valid samples)
            if np.isnan(mae[name]['mae']):
                print(f"{name} - MAE: N/A (No valid samples)")
                with open(logs_path + '/validation_results.txt', 'a', encoding='utf-8') as file:
                    file.write(f"{name} - MAE: N/A (No valid samples)\n")
            else:
                # Format the result with the number of valid samples
                result = f"{name} - MAE: {round(mae[name]['mae'], 4)} ± {round(mae[name]['ci'], 3)} (on {mae[name]['valid_samples']} valid samples)"
                print(result)
                with open(logs_path + '/validation_results.txt', 'a', encoding='utf-8') as file:
                    file.write(result + '\n')

        # Success: return results
        return parameters, source_dimension, noise_std, mae

    except Exception as e:
        raise e

    finally:
        # Always clean up temporary predictor/cofactor file, regardless of success or failure
        cleanup_temp_predictors_cofactors_file(settings_path)