import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", message=".*set_default_tensor_type.*")
warnings.filterwarnings("ignore", category=FutureWarning)

from utils import *
from loader import loader
from model import LeaspyModel
from loss import MAE_CI
from train_val import train_and_validate
from prediction import make_predictions
from simulation import simulate_data


parser = argparse.ArgumentParser()
parser.add_argument('--source_dimension', type=int, default=2, help='number of dimensions for non-temporal inter-subject variability')
parser.add_argument('--model_type', choices=['logistic', 'logistic_parallel', 'univariate_logistic'], default='logistic', help='Choose model type between: logistic, logistic_parallel, univariate_logistic')
parser.add_argument('--noise_model', choices=['gaussian_diagonal'], default='gaussian_diagonal', help='estimate the residual noise scaling per feature')
parser.add_argument('--n_iter', type=int, default=3000, help='Number of iteration of mcmc-saem algorithm')
parser.add_argument('--n_burn_in_iter_frac', type=float, default=0.9)
parser.add_argument('--burn_in_step_power', type=float, default=0.8)
parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda', help='Cuda if you want to use GPU otherwise use the value cpu. Requires that you have installed a version of torch with cuda compatibility ')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--gpu_ids', type=int, default=0, help='ids of GPUs to use')
parser.add_argument('--sub_data', type=bool, default=False, help='True if you want to use a subset of the entire dataset')
parser.add_argument('--sub_n', type=str, default='100000', help='If sub_data is true, you can choose the number of patients')
parser.add_argument('--manual_seed', type=int, help="Manual Seed. Set to 0 for reproducibility")
parser.add_argument('--dummy', type=bool, default=False, help='True if you want to convert cofactors into dummy variables')
parser.add_argument('--data', type=str, default='data.csv', help='Indicate the dataset name from which the model will be trained')
parser.add_argument('--prediction', type=bool, default=False, help='True if you want to predict new data based on the model')
parser.add_argument('--prediction_timepoints_start', type=int, default=0, help='Indicate the start timepoint for predictions')
parser.add_argument('--prediction_timepoints_end', type=int, default=100, help='Indicate the end timepoint for predictions')
parser.add_argument('--prediction_timepoints_step', type=int, default=1, help='Indicate the step for predictions')
parser.add_argument('--simulation', type=bool, default=False, help='True if you want to simulate new date based on another dataset distribution')
parser.add_argument('--merge_generations', type=bool, default=False, help='True if you want to merge all generations into a single dataset')
parser.add_argument('--n_gen_sub', type=int, default=1000, help='Number of subjects to generate')
parser.add_argument('--n_gen_visit', type=int, default=10, help='Number of visits per patient')
parser.add_argument('--n_gen_std', type=float, default=2.3, help='Standard deviation of the number of visits per patient')

opt = parser.parse_args()
opt.algo_setting_personalization = 'scipy_minimize'

# Seed
if opt.manual_seed is None:
    opt.manual_seed = np.random.randint(1, 10000)
np.random.seed(opt.manual_seed)
torch.manual_seed(opt.manual_seed)

if opt.device == "cuda" and torch.cuda.is_available():
    torch.cuda.manual_seed_all(opt.manual_seed)

# Path
base_path = "../"
data_path = base_path + 'datasets/' # Data path
cognitive_scores_data = data_path + opt.data

# Used to create different folders in case of the same runs on the same day
now = datetime.now()
data = str(now.month) + '_' + str(now.day) + '_' + str(now.hour) + '_' + str(now.minute)

result_path = base_path + f'results/experiment_{opt.sub_n}_{opt.n_iter}_{data}/'
logs_path = result_path + 'logs/' # Logs folder
image_path = result_path + 'images/' # Path of image results
gen_path = result_path + 'synthetic_data/' # Folder to save generated data
validation_path = result_path + 'validation/predictions/' # Folder to save validation results
model_path = base_path + f'weights/experiment_{opt.sub_n}_{opt.n_iter}_{data}/' # Folder to save models
settings_path = base_path + 'utils/' # Folder of configuration files
pretrained_model_path = base_path + 'saved_models/model_parameters.json'
predictions_path = result_path + f'predictions/experiment_{opt.sub_n}_{opt.n_iter}_{data}/'


os.makedirs(logs_path, exist_ok=True)
os.makedirs(image_path, exist_ok=True)
os.makedirs(model_path, exist_ok=True)
os.makedirs(gen_path, exist_ok=True)
os.makedirs(settings_path, exist_ok=True)
os.makedirs(predictions_path, exist_ok=True)
os.makedirs(validation_path, exist_ok=True)

# Creates a json containing the calibration configuration
make_json_calibration_settings_dict(opt, settings_path)

# Evaluation Metrics
mae_metric = MAE_CI() # MAE with Confidence interval

# Datasets
dataset = loader(opt, cognitive_scores_data) # Read csv

# Leaspy Custom Model
if opt.prediction or opt.simulation:
    # Load pretrained model for prediction/simulation
    model = LeaspyModel(opt, logs_path, model_path, image_path, settings_path, pretrained_model_path)
else:
    # Create new model for training
    model = LeaspyModel(opt, logs_path, model_path, image_path, settings_path)

if not opt.prediction and not opt.simulation:
    # Train and validate using the separated module
    parameters, source_dimension, noise_std, mae = train_and_validate(
        dataset, model, opt, logs_path, model_path, validation_path, settings_path
    )

if opt.prediction:
    # Make predictions using the separated module
    predictions = make_predictions(dataset, opt, pretrained_model_path, predictions_path)


# Generate new data
if opt.simulation:
    # Generate synthetic data using the separated module
    df_simu = simulate_data(model, dataset, gen_path, opt)



    

    

