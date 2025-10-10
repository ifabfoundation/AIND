from utils import *

class LeaspyModel():
    def __init__(self, opt, logs_path, model_path, images_path, settings_path, pretrained_model_path=None):
        self.source_dimension = opt.source_dimension
        self.model_type = opt.model_type
        self.noise_model = opt.noise_model
        self.logs_path = logs_path
        self.model_path = model_path
        self.images_path = images_path
        self.seed = opt.manual_seed
        self.pretrained_model_path = pretrained_model_path

        if pretrained_model_path:
            # Load pretrained model
            self.model = Leaspy.load(pretrained_model_path)
        else:
            # Create new model for training
            self.algo_setting_calibration = AlgorithmSettings.load(settings_path + 'algorithm_settings_calibration.json') # Algo settings for fitting
            self.algo_setting_calibration.set_logs(
                path=self.logs_path, # Creates a logs file ; if existing, ask if rewrite it
                save_periodicity=50, # Saves the values in csv files every N iterations
                console_print_periodicity=100, # Displays logs in the console/terminal every N iterations, or None
                plot_periodicity=1000, # Generates the convergence plots every N iterations
                overwrite_logs_folder=True # if True and the logs folder already exists, it entirely overwrites it
            )
            self.model = Leaspy(self.model_type, source_dimension=self.source_dimension, noise_model=self.noise_model) # Main model

        self.algo_setting_personalization = AlgorithmSettings(opt.algo_setting_personalization, seed=self.seed) # Algo settings for personalization
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

    def personalize(self, data, generation = False):
        # Personalize curves
        if generation:
            algo_setting_personalization = AlgorithmSettings('mode_real', seed=self.seed)
            ip = self.model.personalize(data, algo_setting_personalization)
        else:
            ip = self.model.personalize(data, self.algo_setting_personalization)
        return ip
    
    def estimate(self, data_pers, data_to_pred):
        # Personalize curves
        ip = self.personalize(data_pers)

        # Create a dictionary with ID as key and list of times as value
        time_dict = {}
        for id, time in zip(data_to_pred.index.get_level_values('ID'), data_to_pred.index.get_level_values('TIME')):
            if not isinstance(time, list):
                time = [time]  # In this way time is always a list
            time_dict[id] = time

        # Future predictions on biomarkers of patients who have had their biomarkers personalized
        predictions = self.model.estimate(time_dict, ip, to_dataframe=True)

        return predictions
    
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
    
    def generate_virtual_data(self, data):
        """
        Generate virtual data based on a Leaspy Data object.

        Parameters:
        -----------
        data : leaspy.Data
            Leaspy Data object containing the dataset (with predictors and cofactors)

        Returns:
        --------
        pandas.DataFrame
            Generated synthetic dataset
        """
        # Use personalization to sample random effect variables from the data distribution
        ip = self.personalize(data, generation=True)

        # Get dataset statistics for simulation settings
        # Extract the underlying dataframe from the Data object
        df_data = data.to_dataframe()

        # Get number of visits, mean of number of visits and std of number of visits
        n_visit = df_data.groupby('ID').size()  # Use size() instead of count() for better accuracy
        visit_mean = n_visit.mean()
        visit_std = n_visit.std()
        pred_sub = len(n_visit)

        print(f"Simulation settings: {pred_sub} subjects, mean visits: {visit_mean:.2f}, std visits: {visit_std:.2f}")

        # Create simulation settings
        settings_simulate = AlgorithmSettings('simulation', seed=self.seed,
                                             number_of_subjects=pred_sub,
                                             mean_number_of_visits=visit_mean,
                                             std_number_of_visits=visit_std)

        # Generate simulated data
        simulated_data = self.model.simulate(ip, data, settings_simulate)

        # Convert to dataframe format
        df_simu = simulated_data.data.to_dataframe(cofactors='all')

        # Compare distributions (original vs simulated)
        self.__compare_datasets_distribution(df_data, df_simu)

        return df_simu

    def generate_virtual_data_with_cofactors(self, data, num_subjects, mean_visits, std_visits, merge_generations):
        """
        Generate virtual data based on random cofactor combinations.

        Parameters:
        -----------
        data : leaspy.Data
            Leaspy Data object containing the dataset (with predictors and cofactors)
        num_subjects : int
            Total number of subjects to generate across all cofactor combinations
        mean_visits : float
            Mean number of visits per subject
        std_visits : float
            Standard deviation of visits per subject
        opt : object
            Options object containing merge_generations flag

        Returns:
        --------
        pandas.DataFrame or list
            Generated synthetic dataset(s) - single DataFrame if merge_generations=True,
            list of DataFrames if merge_generations=False
        """
        import random
        import itertools
        from collections import defaultdict

        # Extract the underlying dataframe from the Data object
        df_data = data.to_dataframe(cofactors='all')

        # Get cofactor information from the data object
        cofactor_names = data.cofactors

        # Get unique combinations from existing data (keeping NaN values)
        cofactor_combinations = df_data[cofactor_names].drop_duplicates()

        # Sample random combinations from existing ones
        num_combinations = min(10, len(cofactor_combinations))
        sampled_indices = random.sample(range(len(cofactor_combinations)), num_combinations)
        selected_combinations = []

        for idx in sampled_indices:
            combination_dict = cofactor_combinations.iloc[idx].to_dict()
            selected_combinations.append(combination_dict)

        # Calculate subjects per combination
        subjects_per_combination = num_subjects // num_combinations
        remaining_subjects = num_subjects % num_combinations

        generated_datasets = []
        all_combined_data = []

        # Use personalization to sample random effect variables from the data distribution
        ip = self.personalize(data, generation=True)

        print(f"Generating {num_combinations} cofactor combinations with ~{subjects_per_combination} subjects each")

        for i, combination_dict in enumerate(selected_combinations):
            # Adjust subject count for remainder distribution
            current_subjects = subjects_per_combination + (1 if i < remaining_subjects else 0)

            print(f"Generation {i+1}/{num_combinations}: {current_subjects} subjects")

            # Create cofactor filter for AlgorithmSettings
            # Keep original combination for final dataset
            original_combination = combination_dict.copy()

            # Filter out NaN values from the combination
            filtered_combination = {k: v for k, v in combination_dict.items() if pd.notna(v)}

            # Define priority cofactors (must be kept if not NaN)
            priority_cofactors = []
            # Add GENDER cofactors
            gender_cofactors = [k for k in filtered_combination.keys() if k.startswith('GENDER/')]
            if gender_cofactors:
                priority_cofactors.extend(gender_cofactors)
            # Add APOE4 if not NaN
            if 'APOE4' in filtered_combination:
                priority_cofactors.append('APOE4')

            # Progressive cofactor reduction
            working_combination = filtered_combination.copy()
            matching_subjects = 0
            min_subjects = 10

            while len(working_combination) > 0 and matching_subjects < min_subjects:
                # Create a mask for filtering the original data
                mask = pd.Series(True, index=df_data.index)
                for cofactor_name, cofactor_value in working_combination.items():
                    mask &= (df_data[cofactor_name] == cofactor_value)

                matching_subjects = df_data[mask]['ID'].nunique() if 'ID' in df_data.columns else len(df_data[mask])

                if matching_subjects >= min_subjects:
                    break

                # Remove least priority cofactor (keep priority ones)
                cofactors_to_remove = [k for k in working_combination.keys() if k not in priority_cofactors]

                if cofactors_to_remove:
                    # Remove a random non-priority cofactor
                    cofactor_to_remove = random.choice(cofactors_to_remove)
                    del working_combination[cofactor_to_remove]
                else:
                    # If only priority cofactors remain and still not enough subjects, break
                    break

            # Create simulation settings
            if working_combination and matching_subjects >= min_subjects:
                cofactor_state = list(working_combination.keys())
                cofactor_filter = list(working_combination.values())
                print(f"  Using cofactors: {working_combination} ({matching_subjects} matching subjects)")

                settings_simulate = AlgorithmSettings('simulation',
                                                   seed=self.seed + i,
                                                   number_of_subjects=current_subjects,
                                                   mean_number_of_visits=mean_visits,
                                                   std_number_of_visits=std_visits,
                                                   cofactor=cofactor_state,
                                                   cofactor_state=cofactor_filter)
            else:
                print(f"  No sufficient cofactor filtering, generating without constraints")
                settings_simulate = AlgorithmSettings('simulation',
                                                   seed=self.seed + i,
                                                   number_of_subjects=current_subjects,
                                                   mean_number_of_visits=mean_visits,
                                                   std_number_of_visits=std_visits)

            # Generate simulated data for this cofactor combination
            simulated_data = self.model.simulate(ip, data, settings_simulate)

            # Convert to dataframe format
            df_simu = simulated_data.data.to_dataframe()

            # Add cofactor combination information to the dataframe (using original combination including NaN values)
            for cofactor_name, cofactor_value in original_combination.items():
                df_simu[f'generated_{cofactor_name}'] = cofactor_value

            # Add generation identifier
            df_simu['generation_id'] = i + 1

            if merge_generations:
                all_combined_data.append(df_simu)
            else:
                generated_datasets.append(df_simu)

        if merge_generations:
            # Combine all generations into a single DataFrame
            final_dataset = pd.concat(all_combined_data, ignore_index=True)
            print(f"Combined dataset: {len(final_dataset)} total observations")
            return final_dataset
        else:
            print(f"Generated {len(generated_datasets)} separate datasets")
            return generated_datasets

    def forward(self, data):
        noise_std = self.fit(data) # Fit data
        self.save_model() # Save model
        self.__make_curves() # Save curves
        return self.model.model.parameters, self.model.model.source_dimension, noise_std