from utils import *

class LeaspyModel():
    def __init__(self, opt, logs_path, model_path, images_path, settings_path):
        self.source_dimension = opt.source_dimension
        self.model_type = opt.model_type
        self.noise_model = opt.noise_model
        self.logs_path = logs_path
        self.model_path = model_path
        self.images_path = images_path
        self.seed = opt.manual_seed
        self.algo_setting_calibration = AlgorithmSettings.load(settings_path + 'algorithm_settings_calibration.json') # Algo settings for fitting
        self.algo_setting_personalization = AlgorithmSettings(opt.algo_setting_personalization, seed=self.seed) # Algo settings for personalization
        self.algo_setting_calibration.set_logs(
            path=self.logs_path, # Creates a logs file ; if existing, ask if rewrite it
            save_periodicity=50, # Saves the values in csv files every N iterations
            console_print_periodicity=100, # Displays logs in the console/terminal every N iterations, or None
            plot_periodicity=1000, # Generates the convergence plots every N iterations
            overwrite_logs_folder=True # if True and the logs folder already exists, it entirely overwrites it
        )

        self.model = Leaspy(self.model_type, source_dimension=self.source_dimension, noise_model=self.noise_model) # Main model

        self.leasply_plot = Plotting(self.model.model) # Used to plot curvers

    def initialize(self, data):
        self.model.model.initialize(data)

    def cuda(self, device):
        self.model.model.move_to_device(device)

    def fit(self, data):
        # Fit model
        self.model.fit(data, settings=self.algo_setting_calibration)
        noise_std = self.model.model.noise_model.to_dict()['scale']
        return noise_std
    
    def save_model(self):
        # Save model
        self.model.save(self.model_path + "model_parameters.json")

    def __make_curves(self):
        # Save population curves
        ax = self.leasply_plot.average_trajectory(alpha=1, figsize=(14,6), n_std_left=2, n_std_right=8)
        plt.savefig(self.images_path + 'mean_curves.png')
        plt.clf()

    def personalize(self, data):
        # Personalize curves
        ip = self.model.personalize(data, self.algo_setting_personalization)
        return ip
    
    def estimate(self, data_pers, data_to_pred):
        ip = self.personalize(data_pers)

        # Future predictions on biomarkers of patients who have had their biomarkers personalized
        predictions = self.model.estimate(dict(zip(data_to_pred.index.get_level_values('ID'), data_to_pred.index.get_level_values('TIME'))), ip)

        # Create a dataframe from predictions
        prediction_temp = {k:v[0] for k, v in predictions.items()} # Make a well format dictionary to majke a dataframe
        df_predicted = pd.DataFrame.from_dict(prediction_temp, orient='index', columns=['MMSE', 'Memory', 'Language', 'Concentration', 'Praxis'])

        return df_predicted
    
    def __compare_datasets_distribution(self, df_real, df_virtual):
        # Compare the datasets distribution (real biomarkers vs virtual biomarkers)
        for var in df_real.columns:
            plt.hist(df_real[var], alpha=0.5, label='original distribution')
            plt.hist(df_virtual[var], alpha=0.5, label='simulated distribution')
            plt.title(var + ' - original distribution vs sumulated')
            plt.savefig(self.images_path + 'distribution_hist_real_simu_' + var + '.png')
            plt.legend()
            plt.grid()
            plt.clf()

        # Compare the datasets CDF (real biomarkers vs virtual biomarkers)
        for var in df_real.columns:
            sns.ecdfplot(df_real[var], label='original dis')
            sns.ecdfplot(df_virtual[var], label='simu dis')
            plt.title('CDF of ' + var + ' - original distribution vs sumulated')
            plt.savefig(self.images_path + 'CDF_hist_real_simu_' + var + '.png')
            plt.legend()
            plt.grid()
            plt.clf()
    
    def generate_virtual_data(self, df_data):

        # Create a Data object to use the dataset with leasply
        data = Data.from_dataframe(df_data.set_index(['ID', 'TIME']))

        # Use customization and sample random effect variables from the data distribution
        ip = self.personalize(data)
        
        # Get number of visits, mean of number of visits and std of number of visits
        n_visit = df_data.groupby('ID').count()[['TIME']].rename(columns={'TIME':'N visits'})
        visit_mean = n_visit['N visits'].mean()
        visit_std = n_visit['N visits'].std()
        pred_sub = len(n_visit)

        # Similuated data
        settings_simulate = AlgorithmSettings('simulation', seed=self.seed, number_of_subjects=pred_sub, mean_number_of_visits=visit_mean, std_number_of_visits=visit_std)
        simulated_data = self.model.simulate(ip, data, settings_simulate)

        # Make a dataframe of simulated data
        df_simu = simulated_data.data.to_dataframe().set_index(['ID', 'TIME'])
        df_real = df_data.set_index(['ID', 'TIME'])

        self.__compare_datasets_distribution(df_real, df_simu)

        return df_simu

    def forward(self, data):
        noise_std = self.fit(data) # Fit data
        self.save_model() # Save model
        self.__make_curves() # Save curves
        return self.model.model.parameters, self.model.model.source_dimension, noise_std