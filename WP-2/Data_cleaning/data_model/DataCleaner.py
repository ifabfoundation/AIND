import pandas as pd
import numpy as np
import datetime
import string
from dateutil.relativedelta import relativedelta
from dl_client import DatalakeClient

class DataCleaner:
    def __init__(self, support_file_path):
        self.client = DatalakeClient()
        self.support_file = pd.read_excel(support_file_path)

    def filter_variables(self, df, file_name, type='raw'):
        '''
        This function reduces the number of columns of the dataframe based on the variables pressent in the support file.
        The support file is filtered for the file_code of the specific file from which was obtained the df.
        The file_code is extracted from the medatada of the file stored in the datalake.
        '''
        metadata = self.client.get_metadata(
            object_name = type + '/' + file_name
        )

        file_code = metadata['metadata']['custom']['file_code']
    
        support_file = self.support_file[self.support_file['file_code']==file_code]
        lst_variable = [x for x in support_file['variable_code'].unique() if x in list(df.columns)] 
        df_new = df[lst_variable]

        return df_new

    def convert_visitcode_to_int(self, value: string) -> int or string:
        # If the value is 'sc' or 'f', return the value
        if value == 'sc' or value == 'f':
            return value
        
        # If the value is 'bl', return 0
        if value == 'bl':
            return 0

        # If the value is 'm', return the month
        if len(value) > 0 and value[0] == 'm':
            return int(value.split('m')[1])
        
        # If it's already a number, try to convert it to int
        try:
            return int(float(value))
        except (ValueError, TypeError):
            pass
            
        raise ValueError('Ops, qualche caso non è stato considerato', value)

    def handle_f_sc_values(self, df: pd.DataFrame, dataset: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        Handle the 'f' and 'sc' values in the visitcode column.
        Identifies patients with 'f' or 'sc' values in the filtered dataframe,
        then updates all their occurrences in the complete dataset.
        Patients with less than 2 visits are removed from consideration.
        """
        # Verify column exists in both dataframes
        if column not in df.columns or column not in dataset.columns:
            raise ValueError(f"Column '{column}' not found in one or both dataframes")
        
        if 'PTID' not in df.columns or 'PTID' not in dataset.columns:
            raise ValueError("Column 'PTID' not found in one or both dataframes")
        
        # Create a copy of the complete dataset to avoid modifying the original
        result_df = dataset.copy()
        
        # Track patients with errors for reporting
        patients_with_errors = []
        
        # Track patients to remove (those with fewer than 2 visits)
        patients_to_remove = []
        
        # Track patients that need their 'f' or 'sc' values updated
        patients_to_update = []
        
        # First, identify patients in the filtered dataframe that have 'f' or 'sc'
        for ptid in df['PTID'].unique():
            # Get all visits for this patient from the complete dataset
            patient_data = dataset[dataset['PTID'] == ptid]
            
            # If patient has less than 2 visits, add to removal list
            if len(patient_data) < 2:
                patients_to_remove.append(ptid)
                continue
                
            # Check if patient has both 'f' and 'sc'
            unique_values = patient_data[column].unique()
            has_f = 'f' in unique_values
            has_sc = 'sc' in unique_values
            
            if has_f and has_sc:
                patients_with_errors.append(ptid)
            
            # If patient has either 'f' or 'sc', mark for update
            if has_f or has_sc:
                patients_to_update.append(ptid)
        
        # Now update all occurrences of 'f' and 'sc' in the complete dataset
        for ptid in patients_to_update:
            # Get indices for this patient in the complete result dataframe
            patient_indices = result_df.index[result_df['PTID'] == ptid]
            
            # Update rows where column value is 'f' or 'sc' to 0
            mask_f_sc = result_df.loc[patient_indices, column].isin(['f', 'sc'])
            indices_to_update = patient_indices[mask_f_sc]
            result_df.loc[indices_to_update, column] = 0
        
        # Remove patients with fewer than 2 visits from the result
        if patients_to_remove:
            rows_to_remove = result_df['PTID'].isin(patients_to_remove)
            result_df = result_df[~rows_to_remove]
            print(f"Rimossi {len(patients_to_remove)} pazienti con meno di 2 visite")
        
        # Report errors if any
        if patients_with_errors:
            print(f"Errore: {len(patients_with_errors)} pazienti hanno sia 'f' che 'sc':")
            for ptid in patients_with_errors:
                print(f"  - Paziente: {ptid}")
            return None
        
        return result_df

    def drop_if_all_none(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """
        Remove rows from dataframe where all values in the specified columns are None/NaN.
        """
        # Verify that all specified columns exist in the dataframe
        for col in columns:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in dataframe")
        
        # Create a mask for rows where all specified columns are NaN
        mask = df[columns].isna().all(axis=1)
        
        # Return dataframe with those rows dropped
        return df[~mask].copy()

    def handle_null_viscode2(self, df: pd.DataFrame, columns_to_check: list, date_column: str) -> pd.DataFrame:
        """
        Process rows where VISCODE2 is null:
        - If all values in specified columns are null, remove the row
        - Otherwise, calculate the correct month based on the previous visit date of the patient
        """
        # Create a copy of the dataframe to avoid modifying the original
        result_df = df.copy()
        
        # Verify that all required columns exist in the dataframe
        required_cols = columns_to_check + [date_column, 'VISCODE2', 'PTID']
        for col in required_cols:
            if col not in result_df.columns:
                raise ValueError(f"Column '{col}' not found in dataframe")
        
        # Convert date column to datetime if it isn't already
        result_df[date_column] = pd.to_datetime(result_df[date_column])
        
        # Get rows with null VISCODE2
        null_viscode_mask = result_df['VISCODE2'].isna()
        rows_to_process = result_df[null_viscode_mask].copy()
        
        # Create a mask for rows to drop (all values in columns_to_check are null)
        drop_mask = rows_to_process[columns_to_check].isna().all(axis=1)
        
        # Create a list to collect rows to keep with updated VISCODE2
        rows_to_update = []
        
        # Process rows where not all columns are null
        for idx, row in rows_to_process[~drop_mask].iterrows():
            patient_id = row['PTID']
            visit_date = row[date_column]
            
            # Get all visits for this patient, sorted by date
            patient_visits = result_df[result_df['PTID'] == patient_id].sort_values(by=date_column)
            
            # Find the previous visit (if any)
            prev_visits = patient_visits[patient_visits[date_column] < visit_date]
            
            if len(prev_visits) > 0:
                # Get the most recent previous visit
                prev_visit = prev_visits.iloc[-1]
                
                if pd.notna(prev_visit['VISCODE2']):
                    # Calculate date difference in months
                    date_diff = (visit_date - prev_visit[date_column]).days / 30.44  # Average days per month
                    
                    if prev_visit['VISCODE2'] == 'bl' or prev_visit['VISCODE2'] == 'sc' or prev_visit['VISCODE2'] == 'f': 
                        # If previous visit was baseline, calculate months since baseline
                        new_viscode = f"m{int(round(date_diff))}"
                    else:
                        # If previous visit had mXX format, add the months
                        prev_month = int(prev_visit['VISCODE2'].replace('m', ''))
                        new_viscode = f"m{int(round(prev_month + date_diff))}"
                    
                    # Update the row's VISCODE2
                    result_df.loc[idx, 'VISCODE2'] = new_viscode
                else:
                    # If previous visit also had null VISCODE2, we can't reliably calculate
                    # Keep the row but leave VISCODE2 as null
                    pass
            else:
                # No previous visits, might be baseline
                result_df.loc[idx, 'VISCODE2'] = 'bl'
        
        # Filter out rows with all null values in specified columns and null VISCODE2
        rows_to_drop = rows_to_process[drop_mask].index
        result_df = result_df.drop(rows_to_drop)
        
        return result_df

    def replace_unknown_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Replace 'Unknown', 'unknown', and '-4' values with NaN across all columns in a dataframe.
        """
        # Create a copy of the dataframe to avoid modifying the original
        result_df = df.copy()
        
        # Dictionary of values to replace with NaN
        replace_dict = {
            'Unknown': np.nan,
            'unknown': np.nan,
            '-4': np.nan
        }
        
        # Replace values across the entire dataframe
        result_df = result_df.replace(replace_dict)
        
        # Handle numeric columns where -4 might be stored as an integer
        for col in result_df.select_dtypes(include=['number']).columns:
            result_df[col] = result_df[col].replace(-4, np.nan)
        
        return result_df

    def add_calculated_age(self, exam_date, birth_date=None, birth_year=None, age_bl=None, bl_date=None, visit_code=None):
        # Convert exam_date to a datetime object, ensuring any invalid dates are coerced to NaT
        exam_date = pd.to_datetime(exam_date, errors='coerce').date()
        # Check if exam_date is not valid, in that case return age = None
        if exam_date is pd.NaT:
            return None
        
        # if exam_date is valid
        # Case 1: If birth_date is provided, calculate the exact age based on birth_date and exam_date
        if pd.notna(birth_date):
            birth_date = pd.to_datetime(birth_date, errors='coerce').date()
            # Calculate the difference in years and months 
            diff = relativedelta(exam_date, birth_date)
            # Calculate the age in years, rounding to the nearest month
            age = round(diff.years + (diff.months/12))
            return age
        
        # Case 2: If birth_year is provided, calculate the age by subtracting birth_year from exam_date's year
        if pd.notna(birth_year):
            birth_year = int(birth_year)
            age = float(exam_date.year - birth_year)
            return age
        
        # Case 3: If age_bl and bl_date are provided, calculate age based on baseline age (age_bl) and difference in dates
        if pd.notna(age_bl) and pd.notna(bl_date):
            age_bl = float(age_bl)
            bl_date = pd.to_datetime(bl_date, errors='coerce').date()
            diff = relativedelta(exam_date, bl_date)
            age = round(age_bl + (diff.years + (diff.months/12)),1)
            return age
        
        # Case 4: If age_bl and visit_code are provided, calculate age based on baseline age and visit code (months since the baseline visit)
        if pd.notna(age_bl) and pd.notna(visit_code):
            age_bl = float(age_bl)
            age = age_bl + visit_code/12
            return age

    def age_when_missing_bl_date(self, df, ID, ID_col, AGE_col, visit_date_col, visit_code_col):
        # Find the baseline index for the given ID where visit_code is 0 (baseline visit)
        index_bl = df[(df[ID_col] == ID) & (df[visit_code_col] == 0)].index

        # If no baseline is found, return the original dataframe
        if len(index_bl) == 0:
            return df           # Not possible to calculate age
        
        index_bl = index_bl[0]
        age_bl = df.loc[index_bl, AGE_col]
        
        # If baseline age (age_bl) is missing, return the original dataframe
        if pd.isna(age_bl):
            return df           # Not possible to calculate age

        # Get the baseline date for age calculation
        bl_date = df[visit_date_col][index_bl]
        # If the baseline date is available, calculate age using baseline date
        if pd.notna(bl_date):
            for index in df[df[AGE_col].isna()].index:
                exam_date=df[visit_date_col][index] 
                df[AGE_col][index] = add_calculated_age(exam_date=exam_date, age_bl=age_bl, bl_date=bl_date)
        
        # If the baseline date is missing, calculate age using baseline age and visit_code
        else:
            for index in df[df[AGE_col].isna()].index:
                exam_date=df[visit_date_col][index] 
                vist_code = df[visit_code_col][index]
                df[AGE_col][index] = add_calculated_age(exam_date=exam_date, age_bl=age_bl, visit_code=vist_code)
                
        return df

    def binarization_gender(self, df, col_name):
        # female : 0
        # male : 1

        # Check the most common type in the column excluding Nans
        common_type = df[col_name].dropna().map(type).value_counts().idxmax()

        # Case 1: The column contains string values ('female', 'male')
        if common_type == str:
            df[col_name] = df[col_name].str.strip().str.lower()  # cleaning strings
            # Map gender to binary values: female -> 0, male -> 1
            df[col_name] = df[col_name].map({'female': 0, 'male': 1}).astype('Int64')
        
        # Case 2: The column contains numeric values (1 for male, 2 for female)
        elif common_type in [int, float]:
            # Handle gender with numeric representation: (male) 1 -> 1, (female) 2 -> 0
            df[col_name] = df[col_name].apply(lambda x: 1 if x in [1, 1.0] else (0 if x in [2, 2.0] else np.nan)).astype('Int64')

        # Handle unexpected types (either string or numeric, or other types)
        else:
            raise ValueError(f"Tipo di dato imprevisto nella colonna '{col_name}'. Sono attesi valori stringa o numerici e non {common_type}")
        
        return df

    def categorize_marry(self, df, col_name):
        # Married = 1, Divorced = 2, Widowed = 3, Never married = 0, Unknown/ nan = nan

        # Check the most common type in the column excluding Nans
        common_type = df[col_name].dropna().map(type).value_counts().idxmax()

        # Case 1: The column contains string values ('married', 'divorced', 'widowed', 'never married')
        if common_type == str:
            df[col_name] = df[col_name].str.strip().str.lower()  # pulizia
            # Map marital status to classes: 'married' -> 1, 'divorced' -> 2, 'widowed' -> 3, 'never married' -> 0
            df[col_name] = df[col_name].map({'married' : 1, 'divorced' : 2, 'widowed' : 3, 'never married' : 0}).astype('Int64')

        # Case 2: The column contains int or float values (1 = 'married', 2 = 'divorced', 3 = 'widowed', 4= 'never married')
        elif common_type in [int, float]:
            # Map marital status to classes: 'married' -> 1, 'divorced' -> 2, 'widowed' -> 3, 'never married' -> 0
            mapping = {1: 1, 1.0: 1, 2: 2, 2.0: 2, 3: 3, 3.0: 3, 4: 0, 4.0: 0}
            df[col_name] = df[col_name].map(mapping).astype('Int64')
        
        # Handle unexpected types (either string or numeric, or other types)
        else:
            raise ValueError(f"Tipo di dato imprevisto nella colonna '{col_name}'. Sono attesi valori stringa o numerici e non {common_type}")
        
        return df

    def categorize_education(self, df, col_name):
        # Married = 1, Divorced = 2, Widowed = 3, Never married = 0, Unknown/ nan = nan
        df[col_name] = df[col_name].apply(lambda x: int(x) if pd.notna(x) and x >= 0 else np.nan)
        return df

    def categorize_ethnicity(self, df, col_name):
        # Not Hisp/Latino = 0 , Hisp/Latino = 1, Unknown Unknown/ nan = nan

        # Check the most common type in the column excluding Nans
        common_type = df[col_name].dropna().map(type).value_counts().idxmax()

        # Case 1: The column contains string values ('not hisp/latino', 'hisp/latino')
        if common_type == str:
            df[col_name] = df[col_name].str.strip().str.lower()  # pulizia
            df[col_name] = df[col_name].map({'not hisp/latino' : 0 , 'hisp/latino' : 1}).astype('Int64')

        # Case 2: The column contains int or float values (1 = 'not hisp/latino', 2 = 'hisp/latino')
        elif common_type in [int, float]:
            df[col_name] = df[col_name].apply(lambda x: 1 if x in [1, 1.0] else (0 if x in [2, 2.0] else np.nan)).astype('Int64')
        
        # Handle unexpected types (either string or numeric, or other types)
        else:
            raise ValueError(f"Tipo di dato imprevisto nella colonna '{col_name}'. Sono attesi valori stringa o numerici e non {common_type}")
        
        return df

    def categorize_race(self, df, col_name):
        # 1 = American Indian or Alaskan Native, 2 = Asian, 3 = Native Hawaiian or Other Pacific Islander, 4 = Black or African American, 5 = White, 6=More than one race, nan = Unknown
        
        # Check the most common type in the column excluding Nans
        common_type = df[col_name].dropna().map(type).value_counts().idxmax()

        # Case 1: The column contains string values ('White', 'More than one', 'Black', 'Asian', 'Am Indian/Alaskan', 'Hawaiian/Other PI')
        if common_type == str:
            df[col_name] = df[col_name].str.strip()  # pulizia
            df[col_name] = df[col_name].map({'White': 5, 'More than one': 0, 'Black': 4, 'Asian': 2, 'Am Indian/Alaskan': 1, 'Hawaiian/Other PI': 3, '5': 5, '6': 0, '4': 4, '2': 2, '1': 1, '3': 3}).astype('Int64')

        # Case 2: The column contains int or float values (5 = 'White', 0 = 'More than one', 4 = 'Black', 2 = 'Asian', 1 = 'Am Indian/Alaskan', 3 = 'Hawaiian/Other PI')
        elif common_type in [int, float]:
            mapping = {1: 1, 1.0: 1, 2: 2, 2.0: 2, 3: 3, 3.0: 3, 4: 4, 4.0: 4, 5: 5, 5.0: 5, 6: 0, 6.0: 0}
            df[col_name] = df[col_name].map(mapping).astype('Int64')
        
        # Handle unexpected types (either string or numeric, or other types)
        else:
            raise ValueError(f"Tipo di dato imprevisto nella colonna '{col_name}'. Sono attesi valori stringa o numerici e non {common_type}")
        
        return df



    def convert_to_two_bit(self, df, col_name):
        """
        Converts a column with values like 'A+T-', 'A-T-', 'A+T+', 'A-T+' into a 2-bit representation.
        First bit: 0 if A-, 1 if A+
        Second bit: 0 if T-, 1 if T+

        'A-T-' → 0
        'A-T+' → 1
        'A+T-' → 2
        'A+T+' → 3
        
        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame containing the column to convert
        col_name : str
            The name of the column to convert
            
        Returns
        -------
        pandas.DataFrame
            DataFrame with the column converted to 2-bit representation
        """
        # Create a copy of the dataframe to avoid modifying the original
        result_df = df.copy()
        
        # Define the conversion function
        def to_two_bit(value):
            if pd.isna(value):
                return np.nan
            
            # Initialize bits
            first_bit = 0  # A
            second_bit = 0  # T
            
            # Check if A is positive (A+)
            if 'A+' in value:
                first_bit = 1
            
            # Check if T is positive (T+)
            if 'T+' in value:
                second_bit = 1
            
            # Return 2-bit representation as integer
            return first_bit * 2 + second_bit
        
        # Apply conversion to the column
        result_df[col_name] = result_df[col_name].apply(to_two_bit)
        
        return result_df

    def categorize_diagnosis(self, df, col_name):
        # Cognitive Normal  = 0, MCI = 1, Dementia = 2

        # Check the most common type in the column excluding Nans
        common_type = df[col_name].dropna().map(type).value_counts().idxmax()

        # Case 1: The column contains string values ('CN', 'MCI', 'Dementia')
        if common_type == str:
            df[col_name] = df[col_name].str.strip()  # pulizia
            df[col_name] = df[col_name].map({'CN' : 0 , 'MCI' : 1, 'Dementia': 3}).astype('Int64')

        # Case 2: The column contains int or float values (1 = 'CN', 2 = 'MCI', 3 = 'Dementia')
        elif common_type in [int, float]:
            mapping = {1: 0, 1.0: 0, 2: 1, 2.0: 1, 3: 2, 3.0: 2}
            df[col_name] = df[col_name].map(mapping).astype('Int64')
        
        # Handle unexpected types (either string or numeric, or other types)
        else:
            raise ValueError(f"Tipo di dato imprevisto nella colonna '{col_name}'. Sono attesi valori stringa o numerici e non {common_type}")
        
        return df
