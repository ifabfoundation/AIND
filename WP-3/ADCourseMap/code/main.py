from utils import *
from loader import loader
from model import LeaspyModel
from loss import MAE_CI


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
parser.add_argument('--simulation', type=bool, default=False, help='True if you want to simulate new date based on another dataset distribution')
parser.add_argument('--sim_data', type=str, help='Indicate the dataset path from which to sample the distribution of the dataset to be generated. If empty, a part of the original dataset will be used')

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
cognitive_scores_data = data_path + 'cognitive_scores.csv'

# Used to create different folders in case of the same runs on the same day
now = datetime.now()
data = str(now.month) + '_' + str(now.day) + '_' + str(now.hour) + '_' + str(now.minute)

result_path = base_path + f'results/experiment_{opt.sub_n}_{opt.n_iter}_{data}/'
logs_path = result_path + 'logs/' # Logs folder
image_path = result_path + 'images/' # Path of image results
gen_path = result_path + 'synthetic_data/' # Folder to save generated data
model_path = base_path + f'weights/experiment_{opt.sub_n}_{opt.n_iter}_{data}/' # Folder to save models
settings_path = base_path + 'utils/' # Folder of configuration files

os.makedirs(logs_path, exist_ok=True)
os.makedirs(image_path, exist_ok=True)
os.makedirs(model_path, exist_ok=True)
os.makedirs(gen_path, exist_ok=True)
os.makedirs(settings_path, exist_ok=True)

# Creates a json containing the calibration configuration
make_json_calibration_settings_dict(opt, settings_path)

# Evaluation Metrics
mae_metric = MAE_CI() # MAE with Confidence interval

# Leaspy Custom Model
model = LeaspyModel(opt, logs_path, model_path, image_path, settings_path)

# Datasets
dataset = loader(opt, cognitive_scores_data) # Read csv
df_train, df_val, df_pers, df_to_pred = split_dataset_in_train_val(dataset, 0.8) # Split dataset in train and validation, df_to_pred is used to validate predictions

print('TRAIN SAMPLES: ' + str(len(df_train)))
print('PERSONALIZATION SAMPLES: ' + str(len(df_pers)))
print('VALIDATION SAMPLES: ' + str(len(df_to_pred)))

# Create Data Object to use the dataset with leaspy
data_train = Data.from_dataframe(df_train)
data_pers = Data.from_dataframe(df_pers)

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

# Save training noise results
for idx, col in enumerate(df_train.columns):
    col_noise = round(noise_std[idx] * 100, 2)
    with open(logs_path + '/train_noise.txt', 'a') as file:
        file.write(col + ': ' + str(col_noise) + '%\n') 

print('VALIDATION\n')

# Performance evaluation
df_predictions = model.estimate(data_pers, df_to_pred) # Personalize and predict
mae = mae_metric.calculate(df_to_pred, df_predictions)

# Save validation MAE
for name in mae.keys():
    print(name +' - MAE: ' + str(round(mae[name]['mae'], 4))+ ' ± ' + str(round(mae[name]['ci'],3)))
    with open(logs_path + '/validation_results.txt', 'a', encoding='utf-8') as file:
        file.write(name +' - MAE: ' + str(round(mae[name]['mae'], 4))+ ' ± ' + str(round(mae[name]['ci'],3)) + '\n')

# Generate new data
if opt.simulation:

    print('SIMULATION\n')

    if opt.sim_data is not None:
        df_sim = pd.read_csv(opt.sim_data)
    else:
        df_sim = df_val.copy()

    # Generate new data
    df_simu = model.generate_virtual_data(df_sim)

    # Save new data
    df_simu.to_csv(gen_path + 'synthetic_data.csv')



    

    

