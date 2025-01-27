from utils.dependencies import *

class customDataset(Dataset):
    def __init__(self, data, s_lable, s_true_lable, g_lable, static_label, step, m_wish):
        # the data should be a dataframe with already the months calculated, not necessarely as integers 
        self.data = data.copy()
        self.s_lable = s_lable
        self.s_true_lable = s_true_lable
        self.g_lable = g_lable
        self.static_label = static_label
        self.step = step
        self.m_wish = m_wish
        self.ids, self.s_subs, self.s_true_subs, self.s_masks, self.g_subs, self.g_masks, self.static_subs = self.__get_subgroup()

    # Get len
    def __len__(self):
        return len(self.ids)
    
    # Used by the dataloader
    def __get_subgroup(self):
        self.data['month_step'] = self.data['month_bl'].apply(lambda x:self.__round_to_step(x)) 
        expanded_data, data_mask = self.__expand_patient_data()

        # Normalising TIME and month, month_step
        expanded_data['TIME'] = expanded_data['TIME']/expanded_data['TIME'].max()
        
        # creating subgroups with cathegorical values (s) and contineous values (g) and relative masks
        ids = expanded_data['ID'].unique().tolist()
        s_sub = expanded_data.groupby(['ID'])[self.s_lable].apply(lambda x: x.values.tolist()).tolist()
        s_true_sub = expanded_data.groupby(['ID'])[self.s_true_lable].apply(lambda x: x.iloc[1:].values.tolist()).tolist()
        g_sub = expanded_data.groupby(['ID'])[self.g_lable].apply(lambda x: x.values.tolist()).tolist()
        s_mask = data_mask.groupby(['ID'])[self.s_true_lable].apply(lambda x: x.iloc[1:].values.tolist()).tolist()
        g_mask = data_mask.groupby(['ID'])[self.g_lable].apply(lambda x: x.iloc[1:].values.tolist()).tolist()
        static_sub = expanded_data.groupby(['ID'])[self.static_label].apply(lambda x: x.values.tolist()).tolist()
        return ids, s_sub, s_true_sub, s_mask, g_sub, g_mask, static_sub

    # Round the value of month_bl to the nearest multiple defined by step. Used to normalize temporal data
    def __round_to_step(self, month):
        module = month%self.step
        quotient = int(month/self.step)
        if(module <= self.step/2):
            new_month =  quotient * self.step
        else:
            new_month = (quotient + 1) * self.step
        
        return new_month
    
    # Adds padding to equalize batches
    def __expand_patient_data(self):
        expanded_df_list = []
        full_mask_list = []
        for patient_id, patient_df in self.data.groupby('ID'):
            month_true = patient_df['month_step'].values
            month_missing = self.m_wish[~ np.isin(self.m_wish, month_true)]
            
            # Calculates the age at the new months
            baseline_age = patient_df.loc[patient_df['month_step'] == 0, 'TIME'].iloc[0]
            
            # Creates new rows for the missing visita, with defined ID, TIME and month and all other fetures to NaN
            new_rows = []
            for month in month_missing:
                time = baseline_age + (month / 12)
                new_row = {'ID': patient_id, 'TIME': time, 'month': month, 'month_step' : month}
                for col in self.data.columns:
                    if col not in ['ID', 'TIME', 'month', 'month_step']:
                        if col in self.static_label:
                            new_row[col] = patient_df[col].iloc[0]
                        else:
                            new_row[col] = np.nan
                new_rows.append(new_row)
            # Add the new rows and re-sort the partial dataset based on the months values
            new_rows_df = pd.DataFrame(new_rows)
            patient_expanded_df = pd.concat([patient_df, new_rows_df], ignore_index=True)
            patient_expanded_df = patient_expanded_df.sort_values(by='month_step')

            # Creating a mask to know which are the True existing vlaues and False the ones that are missing in the original dataset and will later be filled up
            nan_mask_temp = ~patient_expanded_df.drop(columns=['ID']).isna()
            nan_mask = patient_expanded_df.copy()
            nan_mask.loc[:, nan_mask_temp.columns] = nan_mask_temp

            expanded_df_list.append(patient_expanded_df)
            full_mask_list.append(nan_mask)
        
        expanded_df = pd.concat(expanded_df_list, ignore_index=True)
        full_mask = pd.concat(full_mask_list, ignore_index=True)
        
        return expanded_df, full_mask
    
    def __getitem__(self, index):

        s_subs = torch.tensor(self.s_subs[index], dtype=torch.float32)
        s_true_subs = torch.tensor(self.s_true_subs[index], dtype=torch.float32)
        s_masks = torch.tensor(self.s_masks[index], dtype=torch.bool)
        g_subs = torch.tensor(self.g_subs[index], dtype=torch.float32)
        g_masks = torch.tensor(self.g_masks[index], dtype=torch.bool)
        static_subs = torch.tensor(self.static_subs[index], dtype=torch.float32)

        return self.ids[index], s_subs, s_true_subs, s_masks, g_subs, g_masks, static_subs
    
