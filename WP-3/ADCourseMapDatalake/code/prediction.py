from utils import *
from leaspy import Leaspy, Data, AlgorithmSettings
import numpy as np

def make_predictions(dataset, opt, model):
    """
    Load a pretrained model and make predictions on the dataset.

    Parameters:
    -----------
    dataset : pandas.DataFrame
        The dataset to make predictions on
    opt : argparse.Namespace
        Command line arguments
    model : LeaspyModel
        The Leaspy model instance (pretrained)

    Returns:
    --------
    pandas.DataFrame
        The predictions dataframe
    """
    print('PREDICTION\n')

    # Set multi-index
    df_to_pred = dataset.set_index(['ID', 'TIME'])

    # Get predictors and cofactors
    predictor_columns = remove_unnecessary_values_from_list(get_dataset_predictors(opt.level, opt.file_code), ['DX/CN', 'DX/Dementia', 'DX/MCI', "ICV%ICV"])
    cofactor_columns = remove_unnecessary_values_from_list(get_dataset_cofactors(opt.level, opt.file_code), ['AGE'])

    # Create predictor dataset with [ID, TIME] multi-index
    df_predictors = df_to_pred[predictor_columns].copy()

    # Create cofactor dataset with only ID as index, one row per patient
    df_cofactors = df_to_pred.reset_index()[['ID'] + cofactor_columns].copy()
    df_cofactors = df_cofactors.drop_duplicates(subset=['ID']).set_index('ID')

    # Create Data Object
    data_to_pred = Data.from_dataframe(df_predictors)
    data_to_pred.load_cofactors(df=df_cofactors, cofactors=cofactor_columns)

    # Unique ids of patients
    ids = df_to_pred.index.get_level_values('ID').unique()

    predictions = model.predict(ids, opt.prediction_timepoints_start, opt.prediction_timepoints_end, opt.prediction_timepoints_step, data_to_pred)
    predictions = predictions.reset_index()

    print("SAVE PREDICTIONS...\n")

    # Metadata to save predictions
    metadata = {
        'file_code': opt.file_code + '_predictions',
        'from_dataset': opt.file_code,
        'level': opt.level,
        'prediction_timepoints_start': opt.prediction_timepoints_start,
        'prediction_timepoints_end': opt.prediction_timepoints_end,
        'prediction_timepoints_step': opt.prediction_timepoints_step
    }

    # Used to create different names
    now = datetime.now()
    data = str(now.month) + '_' + str(now.day) + '_' + str(now.hour) + '_' + str(now.minute)

    # Prediction file name
    file_name = 'predictions_{}_{}.csv'.format(opt.file_code, data)

    # Save predictions
    load_dataset_to_datalake(metadata, predictions, file_name, 'predictions')
    print("PREDICTION SAVED\n")

