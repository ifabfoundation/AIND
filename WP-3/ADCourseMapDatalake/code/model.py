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
        self.level = opt.level
        self.file_code = opt.file_code

        if pretrained_model_path:
            # Load pretrined model
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
        self.leaspy_plot = Plotting(self.model.model) # Used to plot curvers

    def _get_features_bounds(self):
        """
        Get features bounds dynamically from metadata.
        Combines scale normalization values and volume normalization values.
        Transforms format from [min, max, direction] to (min, max).
        """
        features_bounds = {}

        # Get scale normalization values (e.g., ADAS11, MMSE, etc.)
        scale_values = get_scale_normalization_values(self.level, self.file_code)
        for key, values in scale_values.items():
            features_bounds[key] = (values[0], values[1])

        # Get volume normalization values (e.g., Hippocampus%ICV, etc.)
        volume_values = get_volumes_normalization_values(self.level, self.file_code)
        for key, values in volume_values.items():
            features_bounds[key] = (values[0], values[1])

        return features_bounds

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
        ax = self.leaspy_plot.average_trajectory(alpha=1, 
            figsize=(14,6), 
            n_std_left=2, 
            n_std_right=8, 
            colors=['red', 'blue', 'green', 'yellow', 'black', 'orange', 'purple', 'pink', 'brown', 'gray', 'cyan','magenta'])
        plt.savefig(self.images_path + 'mean_curves.png')
        plt.clf()


    def _make_individual_curvers(self, data, ip, ids):
        # Create a plot for each ids
        for id in ids:
            ax = self.leaspy_plot.patient_trajectories(
                data=data,
                individual_parameters=ip,
                patients_idx=[id],
                reparametrized_ages=False,
                factor_future=5,  # how many years in the future
                factor_past=0.5,  # how many years in the past
                alpha=1, 
                figsize=(14,6),
            )

            # Create directory if it doesn't exist
            os.makedirs(self.images_path + 'predictions', exist_ok=True)
            plt.savefig(self.images_path + 'predictions/individual_curves_{}.png'.format(id))
            plt.clf()
            

    def personalize(self, data, generation = False):
        # Personalize curves
        if generation:
            algo_setting_personalization = AlgorithmSettings(
                name='mode_real', 
                seed=self.seed,
                n_iter=10000,                          # ← AUMENTATO
                n_burn_in_iter_frac=0.6,              # ← RIDOTTO (50% = 3500 burn-in, 3500 stima)
                burn_in_step_power=0.8,               # ✓ OK
                sampler_ind='Gibbs',                  # ✓ OK
                sampler_ind_params={
                    'acceptation_history_length': 25,    # ✓ OK
                    'mean_acceptation_rate_target_bounds': [0.25, 0.45],  # ← AUMENTATO
                    'adaptive_std_factor': 0.3          # ← AUMENTATO
                }
            )
            print('Personalization algo settings: ', algo_setting_personalization.parameters)
            ip = self.model.personalize(data, algo_setting_personalization)
        else:
            ip = self.model.personalize(data, self.algo_setting_personalization)
        return ip

    def predict(self, ids, prediction_timepoints_start, prediction_timepoints_end, prediction_timepoints_step, data):
        # Personalize curves
        ip = self.personalize(data)
        
        # Set timepoints for predictions
        if prediction_timepoints_step == 1:
            # Use linespace for single step (num parameter)
            timepoints = np.linspace(prediction_timepoints_start, prediction_timepoints_end,
                                    prediction_timepoints_end - prediction_timepoints_start + 1)
        else:
            # Use arange for step-based spacing
            timepoints = np.arange(prediction_timepoints_start, prediction_timepoints_end + 1, prediction_timepoints_step)

        # Create a dictionary with timepoints for each ID
        time_dict = {id_: timepoints for id_ in ids}

        # Prediction
        predictions = self.model.estimate(time_dict, ip, to_dataframe=True)
        predictions_data = Data.from_dataframe(predictions)

        self._make_individual_curvers(predictions_data, ip, ids)

        return predictions
        
    
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
            safe_var = var.replace('/', '_').replace('\\', '_')
            plt.hist(df_real[var], alpha=0.5, label='original distribution')
            plt.hist(df_virtual[var], alpha=0.5, label='simulated distribution')
            plt.title(var + ' - original distribution vs sumulated')
            plt.savefig(self.images_path + 'distribution_hist_real_simu_' + safe_var + '.png')
            plt.legend()
            plt.grid()
            plt.clf()

        # Compare the datasets CDF (real biomarkers vs virtual biomarkers)
        for var in df_real.columns:
            safe_var = var.replace('/', '_').replace('\\', '_')
            sns.ecdfplot(df_real[var], label='original dis')
            sns.ecdfplot(df_virtual[var], label='simu dis')
            plt.title('CDF of ' + var + ' - original distribution vs sumulated')
            plt.savefig(self.images_path + 'CDF_hist_real_simu_' + safe_var + '.png')
            plt.legend()
            plt.grid()
            plt.clf()

    def public_compare_datasets_distribution(self, df_real, df_virtual):
        self.__compare_datasets_distribution(df_real, df_virtual)

    def __calibrate_age_bounds(self, individual_params, real_df):
        """
        Calibra reparametrized_age_bounds dalla distribuzione psi reale
        
        Args:
            model: Leaspy model fitted
            individual_params: dict di parametri individuali
            real_df: DataFrame dati reali
        
        Returns:
            tuple: (psi_lower, psi_upper)
        """
        # Estrai parametri popolazione
        tau_mean = self.model.model.parameters['tau_mean']
        
        # Calcola psi per ogni osservazione
        psi_values = []
        
        for patient_id, params in individual_params.items():
            tau_i = params['tau']
            xi_i = params['xi']
            
            # Filtra osservazioni del paziente
            patient_data = real_df[real_df['ID'] == patient_id]
            
            for _, row in patient_data.iterrows():
                t = row['TIME']
                
                # Formula: psi_i(t) = exp(xi_i) * (t - tau_i) + tau_mean
                psi = np.exp(xi_i) * (t - tau_i) + tau_mean
                psi_values.append(psi)
        
        psi_values = np.array(psi_values)
        
        # Calcola percentili
        psi_lower = np.quantile(psi_values, 0.01)  # 1° percentile
        psi_upper = np.quantile(psi_values, 0.99)  # 99° percentile
        
        print(f"Psi distribution:")
        print(f"  Min: {psi_values.min():.2f}")
        print(f"  1%: {psi_lower:.2f}")
        print(f"  Median: {np.median(psi_values):.2f}")
        print(f"  99%: {psi_upper:.2f}")
        print(f"  Max: {psi_values.max():.2f}")
        
        # Arrotonda con margine
        psi_lower = np.floor(psi_lower) - 5
        psi_upper = np.ceil(psi_upper) + 5
        
        return (float(psi_lower), float(psi_upper))
    
    def generate_virtual_data(self, data):

        # Create a dataframe with all the data
        df_data = data.to_dataframe()

        # Use customization and sample random effect variables from the data distribution
        ip = self.personalize(data, generation=True)
        
        # Get number of visits, mean of number of visits and std of number of visits
        n_visit = df_data.groupby('ID').count()[['TIME']].rename(columns={'TIME':'N visits'})
        visit_mean = n_visit['N visits'].mean()
        visit_std = n_visit['N visits'].std()
        pred_sub = len(n_visit)

        # Get features bounds dynamically from metadata
        features_bounds = self._get_features_bounds()

        age_bounds = self.__calibrate_age_bounds(ip, df_data)
        print(f"Recommended reparametrized_age_bounds: {age_bounds}")

        # Similuated data
        settings_simulate = AlgorithmSettings(
            'simulation',
            seed=self.seed,
            number_of_subjects=pred_sub,
            mean_number_of_visits=visit_mean,
            std_number_of_visits=visit_std,
            #reparametrized_age_bounds=age_bounds,
            noise='model',
            features_bbounds=features_bounds,
            #bandwidth_method=2.5,
            #delay_btw_visits=0.6,
            sources_method='full_kde',
        )

        print('Simulate algo settings: ', settings_simulate.parameters)

        simulated_data = self.model.simulate(ip, data, settings_simulate)

        # Make a dataframe of simulated data
        df_simu = simulated_data.data.to_dataframe().set_index(['ID', 'TIME'])
        df_real = df_data.set_index(['ID', 'TIME'])

        self.__compare_datasets_distribution(df_real, df_simu)

        return df_simu

    def generate_virtual_data_with_diagnosis(self, data, dx_percentages=None):
        """
        Generate virtual data with specified diagnosis distribution.
        
        Args:
            data: Original dataset
            dx_percentages: Dictionary with diagnosis percentages (e.g., {'CN': 0.5, 'MCI': 0.3, 'AD': 0.2})
        """

        # Create a dataframe with all the data
        df_data = data.to_dataframe()

        # Use customization and sample random effect variables from the data distribution
        ip = self.personalize(data, generation=True)
        
        # Get number of visits, mean of number of visits and std of number of visits
        n_visit = df_data.groupby('ID').count()[['TIME']].rename(columns={'TIME':'N visits'})
        visit_mean = n_visit['N visits'].mean()
        visit_std = n_visit['N visits'].std()
        pred_sub = len(n_visit)

        # Get features bounds dynamically from metadata
        features_bounds = self._get_features_bounds()

        age_bounds = self.__calibrate_age_bounds(ip, df_data)
        print(f"Recommended reparametrized_age_bounds: {age_bounds}")

        # Map diagnosis to cofactor states
        dx_cofactor_mapping = {
            'CN':       [1, 0, 0],
            'dementia': [0, 1, 0],
            'MCI':      [0, 0, 1],
        }
        cofactor_names = ['DX/CN', 'DX/Dementia', 'DX/MCI']

        # Determine diagnosis distribution
        if dx_percentages is None:
            # Random distribution
            random_vals = np.random.random(3)
            random_vals = random_vals / random_vals.sum()
            dx_percentages = {
                'CN': random_vals[0],
                'dementia': random_vals[1],
                'MCI': random_vals[2]
            }
            print(f"Using random DX distribution: {dx_percentages}")
    
        # Validate that percentages sum to 1
        total = sum(dx_percentages.values())
        if not np.isclose(total, 1.0):
            dx_percentages = {k: v/total for k, v in dx_percentages.items()}
            print(f"Normalized DX percentages: {dx_percentages}")

        # Calculate number of subjects for each diagnosis
        n_subjects_per_dx = {}
        remaining = pred_sub
        dx_list = list(dx_percentages.keys())
        
        for i, dx in enumerate(dx_list):
            if i == len(dx_list) - 1:
                n_subjects_per_dx[dx] = remaining
            else:
                n = int(round(pred_sub * dx_percentages[dx]))
                n_subjects_per_dx[dx] = n
                remaining -= n
        
        print(f"Total subjects to generate: {pred_sub}")
        print(f"Subjects per diagnosis: {n_subjects_per_dx}")

        # Generate data for each diagnosis group
        all_simulated_data = []
        cumulative_subjects = 0

        for generation_idx, (dx, n_sub) in enumerate(n_subjects_per_dx.items()):
            if n_sub <= 0:
                # Skip this diagnosis group as it requires 0 subjects
                print(f"Skipping {dx}: 0 subjects")
                continue
            
            # Use the correct number of subjects for this diagnosis group
            cofactor_state = dx_cofactor_mapping[dx]
            
            # Simulated data for this diagnosis group
            settings_simulate = AlgorithmSettings(
                'simulation', 
                seed=self.seed, 
                number_of_subjects=n_sub, 
                mean_number_of_visits=visit_mean, 
                std_number_of_visits=visit_std,
                #reparametrized_age_bounds=age_bounds,
                noise='model',
                features_bbounds=features_bounds,
                #bandwidth_method=2.5,
                #delay_btw_visits=0.6,
                sources_method='full_kde',
                cofactor=cofactor_names,
                cofactor_state=cofactor_state,
            )

            print(f'Simulating {n_sub} subjects with DX={dx} (cofactor_state={cofactor_state})')

            # Generate simulated data for this cofactor combination
            simulated_data = self.model.simulate(ip, data, settings_simulate)

            # Convert to dataframe format
            df_simu = simulated_data.data.to_dataframe()

            # Make IDs unique across all generations
            unique_ids = df_simu['ID'].unique()
            id_mapping = {old_id: f"S{cumulative_subjects + idx:05d}" for idx, old_id in enumerate(unique_ids)}
            df_simu['ID'] = df_simu['ID'].map(id_mapping)

            # Update cumulative count for next iteration
            cumulative_subjects += len(unique_ids)
            print(f"  Generated {len(unique_ids)} unique subjects (IDs: {id_mapping[unique_ids[0]]} to {id_mapping[unique_ids[-1]]})")

            # Add cofactor combination information to the dataframe
            df_simu['generated_DX'] = dx
            for i, cof_name in enumerate(cofactor_names):
                df_simu[f'generated_{cof_name}'] = cofactor_state[i]

            # Add generation identifier
            df_simu['generation_id'] = generation_idx + 1

            all_simulated_data.append(df_simu)

        # Combine all generations into a single DataFrame
        final_dataset = pd.concat(all_simulated_data, ignore_index=True)
        print(f"\nCombined dataset: {len(final_dataset)} total observations, {cumulative_subjects} unique subjects")
        
        # Comparison with real data
        df_simu_indexed = final_dataset.set_index(['ID', 'TIME'])
        df_real = df_data.set_index(['ID', 'TIME'])
        self.__compare_datasets_distribution(df_real, df_simu_indexed)

        return final_dataset


    def generate_virtual_data_with_cofactors(self, data, num_subjects, mean_visits, std_visits, merge_generations, max_num_combinations=0):
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
        if max_num_combinations:
            num_combinations = min(max_num_combinations, len(cofactor_combinations))
        else:
            num_combinations = len(cofactor_combinations)
            
        sampled_indices = random.sample(range(len(cofactor_combinations)), num_combinations)
        selected_combinations = []

        for idx in sampled_indices:
            combination_dict = cofactor_combinations.iloc[idx].to_dict()
            selected_combinations.append(combination_dict)

        # Get features bounds dynamically from metadata
        features_bounds = self._get_features_bounds()

        # Calculate subjects per combination
        subjects_per_combination = num_subjects // num_combinations
        remaining_subjects = num_subjects % num_combinations

        generated_datasets = []
        all_combined_data = []

        # Track cumulative subject count to ensure unique IDs across generations
        cumulative_subjects = 0

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

                settings_simulate = AlgorithmSettings(
                    'simulation',
                    seed=self.seed + i,
                    number_of_subjects=current_subjects,
                    mean_number_of_visits=mean_visits,
                    std_number_of_visits=std_visits,
                    cofactor=cofactor_state,
                    cofactor_state=cofactor_filter,
                    noise='model',
                    features_bbounds=features_bounds,
                    sources_method='full_kde',
                )
            else:
                print(f"  No sufficient cofactor filtering, generating without constraints")
                settings_simulate = AlgorithmSettings(
                    'simulation',
                    seed=self.seed + i,
                    number_of_subjects=current_subjects,
                    mean_number_of_visits=mean_visits,
                    std_number_of_visits=std_visits,
                    noise='model',
                    features_bbounds=features_bounds,
                    sources_method='full_kde',
                )

            # Generate simulated data for this cofactor combination
            simulated_data = self.model.simulate(ip, data, settings_simulate)

            # Convert to dataframe format
            df_simu = simulated_data.data.to_dataframe()

            # Make IDs unique across all generations by adding offset
            # Reset index to access ID column easily
            df_simu_reset = df_simu.copy()

            # Get unique IDs and create mapping to new unique IDs
            unique_ids = df_simu_reset['ID'].unique()
            id_mapping = {old_id: f"S{cumulative_subjects + idx:05d}" for idx, old_id in enumerate(unique_ids)}

            # Apply ID mapping
            df_simu_reset['ID'] = df_simu_reset['ID'].map(id_mapping)
            df_simu = df_simu_reset.copy()

            # Update cumulative count for next iteration
            cumulative_subjects += len(unique_ids)
            print(f"  Generated {len(unique_ids)} unique subjects (IDs: {id_mapping[unique_ids[0]]} to {id_mapping[unique_ids[-1]]})")

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