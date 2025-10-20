from utils import *
from leaspy import Leaspy, Data, AlgorithmSettings
import numpy as np

def make_predictions(dataset, opt, pretrained_model_path, predictions_path):
    """
    Load a pretrained model and make predictions on the dataset.

    Parameters:
    -----------
    dataset : pandas.DataFrame
        The dataset to make predictions on
    opt : argparse.Namespace
        Command line arguments
    pretrained_model_path : str
        Path to the pretrained model
    predictions_path : str
        Path to save predictions

    Returns:
    --------
    pandas.DataFrame
        The predictions dataframe
    """
    print('PREDICTION\n')

    # Set multi-index
    df_to_pred = dataset.set_index(['ID', 'TIME'])
    # Create Data Object
    data_to_pred = Data.from_dataframe(df_to_pred)
    # Load pretrained model
    pretrained_model = Leaspy.load(pretrained_model_path)
    # Set algorithm settings
    algo_setting_personalization = AlgorithmSettings('scipy_minimize')

    # Personalize
    print('PERSONALIZE...\n')
    ip = pretrained_model.personalize(data_to_pred, algo_setting_personalization)

    # Set timepoints for predictions - FIXED: use arange for step-based spacing
    if opt.prediction_timepoints_step == 1:
        # Use linspace for single step (num parameter)
        timepoints = np.linspace(opt.prediction_timepoints_start, opt.prediction_timepoints_end,
                               opt.prediction_timepoints_end - opt.prediction_timepoints_start + 1)
    else:
        # Use arange for step-based spacing
        timepoints = np.arange(opt.prediction_timepoints_start, opt.prediction_timepoints_end + 1,
                             opt.prediction_timepoints_step)

    ids = df_to_pred.index.get_level_values('ID').unique()

    # Create a dictionary with timepoints for each ID
    time_dict = {id_: timepoints for id_ in ids}

    # Estimate
    print('ESTIMATE...\n')
    predictions = pretrained_model.estimate(time_dict, ip, to_dataframe=True)

    # Save predictions
    print('SAVE PREDICTIONS...\n')
    predictions.to_csv(predictions_path + 'predictions.csv')
    print('PREDICTIONS SAVED\n')

    return predictions