from utils import *
from loader import loader
from model import LeaspyModel
from loss import MAE_CI
from train import train_and_validate
from prediction import make_predictions
from simulation import simulate_data


parser = argparse.ArgumentParser()
parser.add_argument('--source_dimension', type=int, default=3, help='number of dimensions for non-temporal inter-subject variability')
parser.add_argument('--model_type', choices=['logistic', 'logistic_parallel', 'univariate_logistic'], default='logistic', help='Choose model type between: logistic, logistic_parallel, univariate_logistic')
parser.add_argument('--noise_model', choices=['gaussian_diagonal'], default='gaussian_diagonal', help='estimate the residual noise scaling per feature')
parser.add_argument('--n_iter', type=int, default=3000, help='Number of iteration of mcmc-saem algorithm')
parser.add_argument('--n_burn_in_iter_frac', type=float, default=0.9)
parser.add_argument('--burn_in_step_power', type=float, default=0.8)
parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu', help='Cuda if you want to use GPU otherwise use the value cpu. Requires that you have installed a version of torch with cuda compatibility ')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--gpu_ids', type=int, default=0, help='ids of GPUs to use')
parser.add_argument('--sub_data', type=bool, default=False, help='True if you want to use a subset of the entire dataset')
parser.add_argument('--sub_n', type=str, default='100000', help='If sub_data is true, you can choose the number of patients')
parser.add_argument('--manual_seed', type=int, help="Manual Seed. Set to 0 for reproducibility")
parser.add_argument('--prediction', type=bool, default=False, help='True if you want to predict new data based on the model')
parser.add_argument('--prediction_timepoints_start', type=int, default=0, help='Indicate the start timepoint for predictions')
parser.add_argument('--prediction_timepoints_end', type=int, default=100, help='Indicate the end timepoint for predictions')
parser.add_argument('--prediction_timepoints_step', type=int, default=1, help='Indicate the step for predictions')
parser.add_argument('--simulation', type=bool, default=False, help='True if you want to simulate new date based on another dataset distribution')
parser.add_argument('--merge_generations', type=bool, default=True, help='True if you want to merge all generations into a single dataset')
parser.add_argument('--n_gen_sub', type=int, default=1000, help='Number of subjects to generate')
parser.add_argument('--n_gen_visit', type=int, default=10, help='Number of visits per patient')
parser.add_argument('--n_gen_std', type=float, default=2.3, help='Standard deviation of the number of visits per patient')
parser.add_argument('--same_data_stats', type=int, default=0, help='If 1, the generated data will have the same statistics as the original data')
parser.add_argument('--level', type=str, default='cleaned_02', help='Metadata field to retrieve from the datalake')
parser.add_argument('--file_code', type=str, default='ADNIMERGE', help='Metadata field: dataset code in datalake')

opt = parser.parse_args()
opt.algo_setting_personalization = 'scipy_minimize'

print('DEBUG: METADATA LEVEL: ', opt.level)
print('DEBUG: METADATA FILE CODE: ', opt.file_code)

# Seed
if opt.manual_seed is None:
    opt.manual_seed = np.random.randint(1, 10000)
np.random.seed(opt.manual_seed)
torch.manual_seed(opt.manual_seed)

if opt.device == "cuda" and torch.cuda.is_available():
    torch.cuda.manual_seed_all(opt.manual_seed)

# Path
base_path = "/home/IFAB/WORK/ADCourseMap/"

# Used to create different folders in case of the same runs on the same day
now = datetime.now()
data = str(now.month) + '_' + str(now.day) + '_' + str(now.hour) + '_' + str(now.minute)

if opt.prediction:
    result_path = base_path + f'results/experiment_{opt.sub_n}_{opt.n_iter}_{data}_prediction/'
if opt.simulation:
    result_path = base_path + f'results/experiment_{opt.sub_n}_{opt.n_iter}_{data}_simulation/'
else:
    result_path = base_path + f'results/experiment_{opt.sub_n}_{opt.n_iter}_{data}/'
    
logs_path = result_path + 'logs/' # Logs folder
image_path = result_path + 'images/' # Path of image results
model_path = base_path + f'weights/experiment_{opt.sub_n}_{opt.n_iter}_{data}/' # Folder to save models
settings_path = base_path + 'utils/' # Folder of configuration files
pretrained_model_path = base_path + 'saved_models/model_parameters.json'

os.makedirs(logs_path, exist_ok=True)
os.makedirs(image_path, exist_ok=True)
if not opt.prediction and not opt.simulation:
    os.makedirs(model_path, exist_ok=True)
os.makedirs(settings_path, exist_ok=True)

# Creates a json containing the calibration configuration
make_json_calibration_settings_dict(opt, settings_path)

# Datasets
dataset = loader(opt) # Read csv

if opt.prediction or opt.simulation:
    # Leaspy Custom Pretrained Model
    model = LeaspyModel(opt, logs_path, model_path, image_path, settings_path, pretrained_model_path)
else:
    # Leaspy Custom Model
    model = LeaspyModel(opt, logs_path, model_path, image_path, settings_path)


if not opt.prediction and not opt.simulation:
    # Train and validate
    parameters, source_dimension, noise_std, mae = train_and_validate(dataset, model, opt, logs_path, model_path, settings_path)

if opt.prediction:
    # Make predictions using the separated model
    predictions = make_predictions(dataset, opt, model)

if opt.simulation:
    df_simu = simulate_data(model, dataset, opt)


    

    

