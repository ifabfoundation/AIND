import pandas as pd
import numpy as np
import datetime
import string
import json
import os
from dateutil.relativedelta import relativedelta
from dl_client import DatalakeClient

def boolenaizer(support_file, column_list):
    for flag_col in column_list:
        # se non tutti i valori sono boleani allora si assicura che vengano trasformati in Boleani
        support_file[flag_col] = support_file[flag_col].apply(
            lambda x: np.nan if pd.isna(x) else str(x).strip().lower() in ['true', '1', '1.0']
        )
    return support_file



def update_variables_support_file(df, support_file, file_code, variable_col='variable_code'):
    """
    Aggiorna il file di supporto sincronizzandolo con le variabili presenti nel DataFrame.
    
    Questa funzione mantiene il file di supporto aggiornato rispetto alle variabili effettivamente
    presenti nel DataFrame, aggiungendo le variabili mancanti e rimuovendo quelle non più presenti.
    
    Args:
        df (pd.DataFrame): DataFrame contenente i dati da analizzare
        support_file (pd.DataFrame): File di supporto da aggiornare
        file_code (str): Codice del file per filtrare il support file
        variable_col (str, optional): Nome della colonna che contiene i codici delle variabili. 
                                    Defaults to 'variable_code'.
    
    Returns:
        pd.DataFrame: File di supporto aggiornato con le variabili sincronizzate
        
    Note:
        - Le variabili presenti nel DataFrame ma non nel support file vengono aggiunte con valori NaN
        - Se le nuove variabili sono derivate da variabili presenti nel support file, i metadati vengono copiati dalla variabile di riferimento
        - Le variabili presenti nel support file ma non nel DataFrame vengono rimosse
        - La funzione mantiene l'ordine delle righe nel support file originale
    """
    # Inizializza updated_support_file con il support_file originale
    updated_support_file = support_file.copy()
    
    # Filtra il support file per il file_code specificato
    support_file_filtered = updated_support_file[updated_support_file['file_code'] == file_code]
    # Trova le variabili del df che mancano nel support file filtrato
    missing_vars = [col for col in df.columns if col not in support_file_filtered[variable_col].values]

    # Se ci sono variabili mancanti, aggiungile al support file (con valori NaN tranne variable_code e file_code)
    if missing_vars:
        #print(missing_vars)
        #rows_to_add, support_file_filtered = rows_for_missing_variables(missing_vars, support_file_filtered, updated_support_file, file_code)
        rows = []
        for var in missing_vars:
            # crea una nuova riga indipendente con valori nulli per tutte le colonne del support file
            new_row = {col: np.nan for col in updated_support_file.columns}
            new_row['file_name'] = support_file_filtered['file_name'].iloc[0]
            new_row['orig_variable_code'] = np.nan
            new_row['variable_code'] = var
            new_row['file_code'] = file_code
            # ottenere i metadati delle nuove variabili se derivate da variabili del support_file_filtered
            # Estrai il prefisso della variabile (prima di '/' o '%')
            var_prefix = var.split('/')[0].split('%')[0]
            # Cerca se esiste una variabile nel support file che contiene il prefisso nel suo nome
            matching_vars = support_file_filtered[support_file_filtered[variable_col].str.contains(var_prefix, na=False)]
            #print('var_prefix: ', var_prefix)
            #print('matching_vars: ', matching_vars[variable_col].tolist())
            if not matching_vars.empty:
                # Prendi la prima variabile che corrisponde al prefisso
                reference_var = matching_vars.iloc[0]
                new_row['parameter'] = reference_var['parameter']
                new_row['orig_variable_code'] = matching_vars['orig_variable_code'].tolist()

                # Copia i metadati dalla variabile di riferimento
                if 'metadati_fattori' in reference_var and pd.notna(reference_var['metadati_fattori']):
                    new_row['metadati_fattori'] = reference_var['metadati_fattori']
                if 'metadati_normalizzazione' in reference_var and pd.notna(reference_var['metadati_normalizzazione']):
                    new_row['metadati_normalizzazione'] = reference_var['metadati_normalizzazione']
                
            rows.append(new_row)
                
        rows_to_add = pd.DataFrame(rows)
        support_file_filtered = pd.concat([support_file_filtered, rows_to_add], ignore_index=True)
        # Ora aggiorna il support file originale con le nuove righe (solo se ci sono variabili mancanti)
        # trovo l'indice a cui termina il file_code corrente nel support file originale
        index = updated_support_file[updated_support_file['file_code'] == file_code].index[-1]+1
        # divisione del support file originale in due parti dopo l'indice trovato
        df_up = updated_support_file.iloc[:index]
        df_down = updated_support_file.iloc[index:]
        # Aggiungi il le nuove righe tra le due parti del support file originale ==> support file aggiornato
        updated_support_file = pd.concat([df_up, rows_to_add, df_down], ignore_index=True)
    
    # identificare le variabili extra nel support file
    # Escludiamo i valori np.nan che rappresentano variabili appena aggiunte
    extra_vars = [col for col in support_file_filtered[variable_col].values 
                 if pd.notna(col) and col not in df.columns]
    #print('extra_vars: ', extra_vars)
    # Rimuovi dal support file le righe corrispondenti alle variabili extra
    if extra_vars:
        # Trova gli indici delle righe da rimuovere
        idx_to_remove = updated_support_file[(updated_support_file['file_code'] == file_code) & (updated_support_file[variable_col].isin(extra_vars))].index 
        #print('idx_to_remove: ', idx_to_remove)
        # Rimuovi queste righe dal support file originale
        updated_support_file = updated_support_file.drop(idx_to_remove)
        # Reindicizza il DataFrame risultante
        updated_support_file = updated_support_file.reset_index(drop=True)

    return updated_support_file

    

class DataCleaner:
    def __init__(self, support_file_path=None, support_file=pd.DataFrame()):
        self.client = DatalakeClient()
        if support_file_path:
            self.support_file = pd.read_excel(support_file_path)
            self.support_file['del'] = self.support_file['del'].astype(bool)
        elif not support_file.empty:
            self.support_file = support_file
            self.support_file['del'] = self.support_file['del'].astype(bool)
        else:
            print('Need to give as imput either the file path or the file its self')

    def update_self_support_file(self, support_file):
        self.support_file = support_file
        self.support_file['del'] = self.support_file['del'].astype(bool)
        return self.support_file

    def get_file_code_metadata(self, file_name, prefix='raw'):
        metadata = self.client.get_metadata(
            object_name = prefix + '/' + file_name
        )
        metadata_costum = metadata['metadata']['custom']
        # Get the file code
        file_code = metadata_costum['file_code']
        return file_code, metadata_costum

    def filter_variables(self, df, file_name, new_var=[], remove_var=[], prefix='raw'):
        '''
        This function reduces the number of columns of the dataframe based on the variables pressent in the support file.
        The support file is filtered for the file_code of the specific file from which was obtained the df.
        The file_code is extracted from the medatada of the file stored in the datalake.
        '''
        # get the metadata of the file from the data lake
        file_code, metadata_costum = self.get_file_code_metadata(file_name, prefix)

        # filtyers the support file for the file_code
        support_file = self.support_file[self.support_file['file_code']==file_code]
        # selection of the unique variable present in both the df and the support file
        support_list = list(support_file['variable_code'].unique())
        new_list = new_var + support_list
        
        columns_to_keep = [x for x in new_list if x in list(df.columns) and x not in remove_var] 
       
        # reduction of the the df to only selected variablees/columns
        df_new = df[columns_to_keep]

        return df_new


    def segmentation_complete_filter(self, df, filter_col='STATUS'):
        '''
        Specific for imaging datasets - FreeSurfer.
        This function filters the dataframe based on the value of the filter_col.
        '''
        print('Segmentation Status: \n', df[filter_col].value_counts())
        return df[df[filter_col] == 'complete'].reset_index(drop=True)
    
    def find_exam_code(self, df, date_column = 'EXAMDATE', viscode_refernce = 'VISCODE', patient_id_column = 'RID', essential_variables: list = []):
        '''
        This function finds the exam code for each patient based on the date of the visit.
        Pipeline per ciascun paziente:
        1) Riorganizza le righe in ordine seguendo le date_column
        2) Verifica che non ci siano date uguali per lo stesso soggetto
        3) Se ci sono più righe con date uguali:
           a) Se hanno valori uguali nelle essential_variables, tieni la prima riga
           b) Altrimenti verifica quale ha meno valori nulli nelle essential_variables
           c) Se hanno lo stesso numero di valori nulli, seleziona quella con visit_reference 'bl' o che inizia per 'm'
        4) Calcola VISIT_MONTH: prima data = 0, successive = mesi di distanza dalla visita 0
        '''
        # Create a copy of the dataframe to avoid modifying the original
        start_df = df.copy()
        
        # Verifica che le colonne richieste esistano
        required_cols = [date_column, viscode_refernce, patient_id_column] + essential_variables
        for col in required_cols:
            if col not in start_df.columns:
                raise ValueError(f"Column '{col}' not found in dataframe")
        
        # Converti la colonna data in datetime
        #start_df[date_column] = pd.to_datetime(start_df[date_column])
        
        # Lista per raccogliere le righe finali
        final_rows = []
        N_patient = 0
        adopted_strategy = []
        for ptid in start_df[patient_id_column].unique():
            # 1) Riorganizza le righe in ordine seguendo le date_column
            df_patient = start_df[start_df[patient_id_column] == ptid].copy()
            df_patient = df_patient.sort_values(by=date_column).reset_index(drop=True)
            
            # 2) Verifica che non ci siano date uguali per lo stesso soggetto
            duplicate_dates = df_patient[date_column].duplicated()
            
            if duplicate_dates.any():
                # 3) Gestisci le date duplicate
                N_patient += 1
                df_patient_cleaned, adopted_strategy = self._handle_duplicate_dates(
                    df_patient, date_column, viscode_refernce, essential_variables, adopted_strategy
                )
            else:
                df_patient_cleaned = df_patient
            # 4) Calcola VISIT_MONTH
            df_patient_cleaned = self._calculate_visit_month(
                df_patient_cleaned, date_column, patient_id_column, viscode_refernce
            )

            final_rows.append(df_patient_cleaned)
        
        print('Number of patients with duplicate dates: ', N_patient)
        print('Adopted visit selection strategy:\n', pd.Series(adopted_strategy).value_counts())
        
        # Combina tutti i pazienti processati
        if final_rows:
            result_df = pd.concat(final_rows, ignore_index=True)
        else:
            result_df = start_df
            pritn('The function find_exam_code has failed, no changes have been applied')
        
        return result_df
    
    def _handle_duplicate_dates(self, df_patient, date_column, viscode_refernce, essential_variables, adopted_strategy):
        """
        Gestisce le date duplicate per un singolo paziente seguendo la logica specificata.
        """
        df_cleaned = df_patient.copy()
        
        # Trova le date duplicate
        date_groups = df_cleaned.groupby(date_column)

        for date, group in date_groups:
            if len(group) > 1:
                if essential_variables:
                    # 3a) Verifica se le righe hanno valori uguali nelle essential_variables
                    essential_values = group[essential_variables]
                    if essential_values.nunique().sum() == len(essential_variables):
                        # Valori uguali, tieni la prima riga
                        row_to_keep = group.index[0]
                        adopted_strategy.append('Equal values')
                    else:
                        # 3b) Valori diversi, verifica quale ha meno valori nulli
                        null_counts = group[essential_variables].isnull().sum(axis=1)
                        min_null_count = null_counts.min()
                        
                        candidates = group[null_counts == min_null_count]
                        
                        if len(candidates) == 1:
                            # Una sola riga con il minor numero di nulli
                            row_to_keep = candidates.index[0]
                            adopted_strategy.append('Min null values')
                        else:
                            # 3c) Stesso numero di nulli, seleziona quella con 'bl' o che inizia per 'm'
                            row_to_keep, adopted_strategy = self._select_by_viscode_priority(candidates, viscode_refernce, adopted_strategy)
                            
                else:
                    # Se non ci sono essential_variables, usa solo la logica del viscode
                    row_to_keep = self._select_by_viscode_priority(group, viscode_refernce)
                
                # Rimuovi le righe duplicate mantenendo solo quella selezionata
                rows_to_remove = group.index[group.index != row_to_keep]
                df_cleaned = df_cleaned.drop(rows_to_remove)
        
        return df_cleaned.reset_index(drop=True), adopted_strategy
    
    def _select_by_viscode_priority(self, candidates, viscode_refernce, adopted_strategy):
        """
        Seleziona la riga basandosi sulla priorità del viscode: 'bl' o che inizia per 'm'.
        """
        for idx, row in candidates.iterrows():
            viscode = row[viscode_refernce]
            if pd.notna(viscode):
                if viscode == 'bl' or (isinstance(viscode, str) and viscode.startswith('m')):
                    adopted_strategy.append('VISITCODE priority')
                    return idx, adopted_strategy
       
        adopted_strategy.append('first row, same VISITCODE')
        # Se nessuna riga soddisfa i criteri, restituisci la prima
        return candidates.index[0], adopted_strategy
    
    def _calculate_visit_month(self, df_patient, date_column, patient_id_column, viscode_refernce):
        """
        Calcola VISIT_MONTH: prima data = 0, successive = mesi di distanza dalla visita 0.
        Inserisce la colonna VISIT_MONTH subito dopo la colonna VISCODE.
        """
        df_patient = df_patient.copy()
        # se il dataframe è vuoto, restituisci il dataframe originale
        if len(df_patient) == 0:
            print('The dataframe is empty, no changes have been applied for the visit month calculation')
            return df_patient
        
        # Ordina per data
        df_patient = df_patient.sort_values(by=date_column).reset_index(drop=True)
        
        # La prima data è la baseline (mese 0)
        baseline_date = df_patient[date_column].iloc[0]
        
        # Calcola i mesi di distanza dalla baseline
        visit_months = []
        for date in df_patient[date_column]:
            if pd.notna(date):
                # Calcola la differenza in mesi
                months_diff = (date - baseline_date).days / 30.44  # Media giorni per mese
                visit_months.append(int(round(months_diff)))
            else:
                visit_months.append(np.nan)
        
        # Trova la posizione della colonna VISCODE
        viscode_position = df_patient.columns.get_loc(viscode_refernce)
        
        # Inserisci la colonna VISIT_MONTH subito dopo VISCODE
        df_patient.insert(viscode_position + 1, 'VISIT_MONTH', visit_months)
        
        return df_patient
    
    def _select_from_complete_rows(self, complete_rows, essential_variables, viscode_refernce, adopted_strategy):
        """
        Seleziona una riga dalle righe con STATUS == 'complete' applicando la logica di selezione.
        """
        if len(complete_rows) == 1:
            adopted_strategy.append('STATUS complete - single row')
            return complete_rows.index[0]
        
        # Applica la logica di selezione sulle righe complete
        if essential_variables:
            # Controlla se tutte le righe complete hanno gli stessi valori nelle essential_variables
            essential_values = complete_rows[essential_variables]
            if essential_values.nunique().sum() == len(essential_variables):
                # Valori uguali, tieni la prima riga
                adopted_strategy.append('STATUS complete - equal essential values')
                return complete_rows.index[0]
            else:
                # Valori diversi, verifica quale ha meno valori nulli
                return self._select_by_null_count_and_viscode(
                    complete_rows, essential_variables, viscode_refernce, adopted_strategy
                )
        else:
            # Se non ci sono essential_variables, usa solo la logica del viscode
            return self._select_by_viscode_priority(complete_rows, viscode_refernce, adopted_strategy)[0]
    
    def _select_by_null_count_and_viscode(self, group, essential_variables, viscode_refernce, adopted_strategy):
        """
        Seleziona una riga basandosi sul numero di valori nulli e poi sulla priorità del viscode.
        """
        if essential_variables:
            # Verifica quale ha meno valori nulli nelle essential_variables
            null_counts = group[essential_variables].isnull().sum(axis=1)
            min_null_count = null_counts.min()
            
            candidates = group[null_counts == min_null_count]
            
            if len(candidates) == 1:
                # Una sola riga con il minor numero di nulli
                adopted_strategy.append('Min null values')
                return candidates.index[0]
            else:
                # Stesso numero di nulli, seleziona quella con 'bl' o che inizia per 'm'
                return self._select_by_viscode_priority(candidates, viscode_refernce, adopted_strategy)[0]
        else:
            # Se non ci sono essential_variables, usa solo la logica del viscode
            return self._select_by_viscode_priority(group, viscode_refernce, adopted_strategy)[0]

            
    
    def convert_visitcode_to_int(self, value: string) -> int or string:
        '''
        function to convert the visit codes (strings) in to integers corresponding to the number of months from the first visist or baseline (month = 0)
        In case of non conventional visit codes ('f', 'sc') the value its self is returned and it willbe handeled by another function.
        '''
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

    def handle_f_sc_values(self, df: pd.DataFrame, dataset: pd.DataFrame, column: str, patient_id_column: str='PTID') -> pd.DataFrame:
        """
        Handle the 'f' and 'sc' values in the visitcode column.
        Identifies patients with 'f' or 'sc' values in the filtered dataframe,
        then updates all their occurrences in the complete dataset.
        Patients with less than 2 visits are removed from consideration.
        """
        # Verify column exists in both dataframes
        if column not in df.columns or column not in dataset.columns:
            raise ValueError(f"Column '{column}' not found in one or both dataframes")
        
        if patient_id_column not in df.columns or patient_id_column not in dataset.columns:
            raise ValueError(f"Column '{patient_id_column}' not found in one or both dataframes")
        
        # Create a copy of the complete dataset to avoid modifying the original
        result_df = dataset.copy()
        
        # Track patients with errors for reporting
        patients_with_errors = []
        
        # Track patients to remove (those with fewer than 2 visits)
        patients_to_remove = []
        
        # Track patients that need their 'f' or 'sc' values updated
        patients_to_update = []
        
        # First, identify patients in the filtered dataframe that have 'f' or 'sc'
        for ptid in df[patient_id_column].unique():
            # Get all visits for this patient from the complete dataset
            patient_data = dataset[dataset[patient_id_column] == ptid]
            
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
            patient_indices = result_df.index[result_df[patient_id_column] == ptid]
            
            # Update rows where column value is 'f' or 'sc' to 0
            mask_f_sc = result_df.loc[patient_indices, column].isin(['f', 'sc'])
            indices_to_update = patient_indices[mask_f_sc]
            result_df.loc[indices_to_update, column] = 0
        
        # Remove patients with fewer than 2 visits from the result
        if patients_to_remove:
            rows_to_remove = result_df[patient_id_column].isin(patients_to_remove)
            result_df = result_df[~rows_to_remove]
            #print(f"Rimossi {len(patients_to_remove)} pazienti con meno di 2 visite")
        
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

    def handle_null_viscode2(self, df: pd.DataFrame, columns_to_check: list, date_column: str, visit_code_column: str='VISCODE2', patient_id_column: str='PTID') -> pd.DataFrame:
        """
        Process rows where VISCODE2 is null:
        - If all values in specified columns are null, remove the row
        - Otherwise, calculate the correct month based on the previous visit date of the patient
        """
        # Create a copy of the dataframe to avoid modifying the original
        result_df = df.copy()
        
        # Verify that all required columns exist in the dataframe
        required_cols = columns_to_check + [date_column, visit_code_column, patient_id_column]
        for col in required_cols:
            if col not in result_df.columns:
                raise ValueError(f"Column '{col}' not found in dataframe")
        
        # Convert date column to datetime if it isn't already
        date_format = pd.to_datetime(result_df[date_column])
        result_df[date_column] = date_format.dt.date
        
        # Get rows with null VISCODE2
        null_viscode_mask = result_df[visit_code_column].isna()
        rows_to_process = result_df[null_viscode_mask].copy()
        
        # Create a mask for rows to drop (all values in columns_to_check are null)
        drop_mask = rows_to_process[columns_to_check].isna().all(axis=1)
        
        # Create a list to collect rows to keep with updated VISCODE2
        rows_to_update = []

        # Process rows where not all columns are null
        for idx, row in rows_to_process[~drop_mask].iterrows():
            patient_id = row[patient_id_column]
            visit_date = row[date_column]
            
            # Get all visits for this patient, sorted by date
            patient_visits = result_df[result_df[patient_id_column] == patient_id].sort_values(by=date_column)
            
            # Find the previous visit (if any)
            prev_visits = patient_visits[patient_visits[date_column] < visit_date]
            
            if len(prev_visits) > 0:
                # Get the most recent previous visit
                prev_visit = prev_visits.iloc[-1]
                
                if pd.notna(prev_visit[visit_code_column]):
                    # Calculate date difference in months
                    date_diff = (visit_date - prev_visit[date_column]).days / 30.44  # Average days per month
                    
                    if prev_visit[visit_code_column] == 'bl' or prev_visit[visit_code_column] == 'sc' or prev_visit[visit_code_column] == 'f': 
                        # If previous visit was baseline, calculate months since baseline
                        new_viscode = f"m{int(round(date_diff))}"
                    else:
                        # If previous visit had mXX format, add the months
                        prev_month = int(prev_visit[visit_code_column].replace('m', ''))
                        new_viscode = f"m{int(round(prev_month + date_diff))}"
                    
                    # Update the row's VISCODE2
                    result_df.loc[idx, visit_code_column] = new_viscode
                else:
                    # If previous visit also had null VISCODE2, we can't reliably calculate
                    # Keep the row but leave VISCODE2 as null
                    pass
            else:
                # No previous visits, might be baseline
                result_df.loc[idx, visit_code_column] = 'bl'
        
        # Filter out rows with all null values in specified columns and null VISCODE2
        rows_to_drop = rows_to_process[drop_mask].index
        result_df = result_df.drop(rows_to_drop)
        
        return result_df
############## MODIFICATO ##############
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
            '-4': np.nan,
            '9999.0': np.nan,
            '9999': np.nan
        }
        
        # Replace values across the entire dataframe
        result_df = result_df.replace(replace_dict)
        
        # Gestisce le colonne numeriche dove -4, -4.0, 9999, 9999.0 potrebbero essere presenti come numeri
        for col in result_df.select_dtypes(include=['number']).columns:
            result_df[col] = result_df[col].replace([-4, -4.0, 9999, 9999.0], np.nan)
        return result_df

    def to_date_format(self, df, col_list=[]):
        for col_name in col_list:
            n = df[col_name].first_valid_index()
            if n is not None and type(df[col_name][n]) is not datetime.date:
                series_date = pd.to_datetime(df[col_name], errors='coerce')
                df[col_name] = series_date.dt.date
        return df

    def add_calculated_age(self, exam_date, birth_date=None, birth_year=None, age_bl=None, bl_date=None, visit_code=None):
        '''
        Calculates the age of the subject at the visit based on different inputs, 
        in ordere the preference to calculate the subject age @ visit are:
        1. birth date
        2. birth year
        3. age at baseline and baseline date
        4. age at baseline and numbero of months from the visit (visit code)
        '''
        # Convert exam_date to a datetime object, ensuring any invalid dates are coerced to NaT
        if type(exam_date) is not datetime.date:
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
        '''
        If the baseline age and date arent saved on the same line of the visit you want to calculate the age 
        this function allows to find these data from the dataset and call the add_calculated_age function.
        '''
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
                df[AGE_col][index] = self.add_calculated_age(self, exam_date=exam_date, age_bl=age_bl, bl_date=bl_date)
        
        # If the baseline date is missing, calculate age using baseline age and visit_code
        else:
            for index in df[df[AGE_col].isna()].index:
                exam_date=df[visit_date_col][index] 
                vist_code = df[visit_code_col][index]
                df[AGE_col][index] = self.add_calculated_age(self, exam_date=exam_date, age_bl=age_bl, visit_code=vist_code)
                
        return df

    def binarization_gender(self, df, col_name):
        '''
        Function to binarize the gender of the subject, it works with strings ('female', 'male) and floats (1. male, 2. female)
        The binarization decoding is:
        female : 0
        male : 1
        '''
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
        '''
        Function to cathegorize the marital status, works for strings ('married', 'divorced', 'widowed', 'never married') and floats
        (1. married, 2. divorced, 3. widowed, 4. 'never married')

        class decodeing:
        'married' -> 1, 'divorced' -> 2, 'widowed' -> 3, 'never married' -> 0
        '''
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
        '''
        Tunrs education variable in to integers, if they are not yet integers
        '''
        df[col_name] = df[col_name].apply(lambda x: int(x) if pd.notna(x) and x >= 0 else np.nan)
        return df

    def categorize_ethnicity(self, df, col_name):
        '''
        Function to binbarize the the ethnical class, works for strings ('not hisp/latino', 'hisp/latino') and floats
        (1. not hisp/latino, 2. hisp/latino)

        class decodeing:
        1 = 'not hisp/latino', 2 = 'hisp/latino'
        '''

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
        '''
        Function to cathegorize the the ethnical class, works for strings ('White', 'More than one', 'Black', 'Asian', 'Am Indian/Alaskan', 'Hawaiian/Other PI') 
        and floats (5. White, 0. More than one, 4. Black, 2. Asian, 1. Am Indian/Alaskan, 3. Hawaiian/Other PI)

        class decodeing:
        1 = American Indian or Alaskan Native, 
        2 = Asian, 
        3 = Native Hawaiian or Other Pacific Islander, 
        4 = Black or African American, 
        5 = White, 
        6 = More than one race
        '''

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
            df[col_name] = df[col_name].map({'CN' : 0 , 'MCI' : 1, 'Dementia': 2}).astype('Int64')

        # Case 2: The column contains int or float values (1 = 'CN', 2 = 'MCI', 3 = 'Dementia')
        elif common_type in [int, float]:
            mapping = {1: 0, 1.0: 0, 2: 1, 2.0: 1, 3: 2, 3.0: 2}
            df[col_name] = df[col_name].map(mapping).astype('Int64')
        
        # Handle unexpected types (either string or numeric, or other types)
        else:
            raise ValueError(f"Tipo di dato imprevisto nella colonna '{col_name}'. Sono attesi valori stringa o numerici e non {common_type}")
        
        return df
    

    def new_variable_names(self, new_support_file_path, df, file_code):
        """
        - Apre il file di supporto da Excel (new_support_file_path)
        - Filtra il support file per il file_code fornito
        - Se ci sono variabili nel df che mancano nella colonna variable_code filtrata del support file,
          aggiunge la riga corrispondente al support file non filtrato
        - Rinomina le colonne del df secondo la colonna new_variable_code (se presente), altrimenti lascia il nome originale
        - Restituisce sia il df rinominato che il support file aggiornato
        """
        import pandas as pd
        import numpy as np

        # Carica il support file da Excel
        new_support_file = pd.read_excel(new_support_file_path)

        # Filtra il support file per il file_code fornito
        support_file_filtered = new_support_file[new_support_file['file_code'] == file_code]
        
        new_updated_support_file = update_variables_support_file(df, new_support_file, file_code, variable_col='orig_variable_code')
        '''
        ##### inizio funzione
        # Trova le variabili del df che mancano nel support file filtrato
        missing_vars = [col for col in df.columns if col not in support_file_filtered['orig_variable_code'].values]
        # Se ci sono variabili mancanti, aggiungile al support file (con valori NaN tranne variable_code e file_code)
        if missing_vars:
            #print(missing_vars)
            rows = []
            for var in missing_vars:
                new_row = {col: np.nan for col in new_support_file.columns}
                new_row['variable_code'] = var
                new_row['file_code'] = file_code
                rows.append(new_row)
            
            rows_to_add = pd.DataFrame(rows)
            support_file_filtered = pd.concat([support_file_filtered, rows_to_add], ignore_index=True)
            # Ora aggiorna il support file originale con le nuove righe (solo se ci sono variabili mancanti)
            index = new_support_file[new_support_file['file_code'] == file_code].index[-1]+1
            # Rimuovi le righe del file_code corrente dal support file originale
            df_up = new_support_file.iloc[:index]
            df_down = new_support_file.iloc[index:]
            # Aggiungi il support_file_filtered aggiornato
            new_support_file = pd.concat([df_up, rows_to_add, df_down], ignore_index=True)
        ##### fine funzione
        '''
        # Crea il mapping per la rinomina delle colonne
        rename_dict = {}
        for _, row in support_file_filtered.iterrows():
            orig = row['orig_variable_code']
            new = row['variable_code'] if pd.notna(row['variable_code']) else orig
            if orig in df.columns:
                rename_dict[orig] = new

        # Rinomina le colonne del df
        df_renamed = df.rename(columns=rename_dict)

        return df_renamed, new_updated_support_file
    
    def remove_param_few_subjects(self, df: pd.DataFrame, file_name: str, flag_col: str='del', prefix: str = 'raw'):
        """
        Rimuove i parametri che nel file di supporto hanno il valore True nella colonna specificata.

        Args:
            df (pd.DataFrame): Il dataframe di input.
            file_name (str): Il nome del file nel datalake.
            column_with_true (str): La colonna nel file di supporto da controllare per il valore True.
            type (str, optional): Il tipo di file nel datalake. Defaults to 'raw'.

        Returns:
            pd.DataFrame: Il dataframe con le colonne rimosse.
        """
        # Step 1: trovare il codice del file dai metadati
        self.file_code, self.metadata_costum = self.get_file_code_metadata(file_name, prefix)

        # Step 2: filtraggio file_supporto per il codice del file
        support_file_for_file = self.support_file[self.support_file['file_code'] == self.file_code]

        # Step 3: in questo sottogruppo identificare i variable_code che hanno True nella colonna specificata
        if flag_col not in support_file_for_file.columns:
            print(f"Attenzione: la colonna '{flag_col}' non è presente nel file di supporto per il file_code '{self.file_code}'. Nessuna colonna verrà rimossa.")
            return df
        
        # Verifica che tutti i valori della colonna flag_col siano booleani, altrimenti li converte
        if not support_file_for_file[flag_col].dropna().map(lambda x: isinstance(x, bool)).all():
            support_file_for_file = boolenaizer(support_file_for_file, column_list=[flag_col])
        
        variables_to_remove = support_file_for_file[support_file_for_file[flag_col] == True]['variable_code'].tolist()

        # Step 4: rimuovere dal df dato in input le colonne che corrispondono alle stringhe della lista appena creata
        df_cleaned = df.drop(columns=[col for col in variables_to_remove if col in df.columns], errors='ignore')
        
        # Rimuovi le righe dal support_file per il file_code corrente e le variabili da rimuovere
        mask = ~(
            (self.support_file['file_code'] == self.file_code) &
            (self.support_file['variable_code'].isin(variables_to_remove))
        )
        self.support_file = self.support_file[mask].reset_index(drop=True)

        return df_cleaned, self.file_code, self.support_file
          
    def remove_sub_1visit(self, df: pd.DataFrame, subject_id_col: str = 'RID') -> pd.DataFrame:
        """
        Rimuove dal dataframe i soggetti che hanno una sola visita.

        Args:
            df (pd.DataFrame): Il dataframe di input.
            subject_id_col (str, optional): La colonna che identifica univocamente il soggetto. Defaults to 'RID'.

        Returns:
            pd.DataFrame: Il dataframe senza i soggetti con una sola visita.
        """
        if subject_id_col not in df.columns:
            raise ValueError(f"La colonna '{subject_id_col}' non è presente nel dataframe.")

        # Conta le visite per ogni soggetto
        visit_counts = df[subject_id_col].value_counts()
        
        # Identifica i soggetti con una sola visita
        subjects_to_remove = visit_counts[visit_counts == 1].index.tolist()
        
        if subjects_to_remove:
            #print(f"Rimossi {len(subjects_to_remove)} soggetti con una sola visita.")
            # Filtra il dataframe per mantenere solo i soggetti con più di una visita
            return df[~df[subject_id_col].isin(subjects_to_remove)].copy()
        else:
            print("Nessun soggetto con una sola visita da rimuovere.")
            return df.copy()
        
    def update_metadati_support(self, updated_support_file): 
        """
        Aggiorna il file di supporto con i metadati delle nuove variabili.
        """
        # Per ogni nuova variabile, aggiorna la colonna appropriata nel self.support_file
        meta_class = [x for x in self.metadata_costum.keys() if x in ['cofattori', 'predittori', 'norm_scala', 'norm_intervallo', 'norm_volume']]
        for key in meta_class:
            if self.metadata_costum[key] == []:
                continue
            if key in ['cofattori', 'predittori']:
                colonna = 'metadati_fattori'
            elif key in ['norm_scala', 'norm_intervallo', 'norm_volume']:
                key = key.replace('norm_', '')
                colonna = 'metadati_normalizzazione'
            else:
                continue  # ignora chiavi non riconosciute

            for var in self.metadata_costum[key]:
                #print('var: ', var, 'colonna: ', key)
                # Trova la riga corrispondente alla variabile nel support_file
                idx = updated_support_file[updated_support_file['file_code'] == self.file_code][updated_support_file['variable_code'] == var].index
                #print('idx: ', idx)
                if not idx.empty:
                    updated_support_file.loc[idx, colonna] = key
        return updated_support_file
    
    def extract_metadata_from_support(self, df, new_level, file_name=None, prefix=None, updated_support_file=None):
        """
        Estrae le liste di variabili per 'fattori' e 'normalizzazione' dal file di supporto,
        filtrando per il file_code corrente dell'istanza.

        Returns:
            Dizionario: Metadati del file aggiornati con liste dei parametri cofattori/predittori e da normalizzare su scala/intervallo
        """
        cofattori = []
        predittori = []
        norm_scala = []
        norm_intervallo = []
        norm_volume = []
        if updated_support_file is not None:
            self.support_file = updated_support_file
        
        if file_name and prefix:
            self.file_code, self.metadata_costum = self.get_file_code_metadata(file_name, prefix)
        elif not hasattr(self, 'metadata_costum') or not self.metadata_costum:
            raise ValueError("L'attributo 'metadata_costum' non è stato impostato. Inserire come input il metadata_costum. Oppure eseguire prima una funzione come 'remove_param_few_subjects'.")
        if not hasattr(self, 'file_code') or not self.file_code:
            raise ValueError("L'attributo 'file_code' non è stato impostato. Inserire come input il file_code. Oppure eseguire prima una funzione come 'remove_param_few_subjects'.")

        # 1. Filtraggio file supporto sulla base del file_code
        support_filtered = self.support_file[self.support_file['file_code'] == self.file_code]

        if support_filtered.empty:
            print(f"Attenzione: nessun dato trovato nel file di supporto per il file_code '{self.file_code}'.")
            return self.metadata_costum

        # 2. Estrazione delle liste di variable_code
        if 'metadati_fattori' in support_filtered.columns:
            cofattori = support_filtered[support_filtered['metadati_fattori']=='cofattore']['variable_code'].tolist()
            predittori = support_filtered[support_filtered['metadati_fattori']=='predittore']['variable_code'].tolist()
            self.metadata_costum['cofattori'] = [x for x in df.columns if x.split('/')[0] in cofattori]
            self.metadata_costum['predittori'] = [x for x in df.columns if x.split('/')[0] in predittori]
            #print('cofattori', cofattori, '\npredittori', predittori)
        else:
            print("Attenzione: la colonna 'metadati_fattori' non è stata trovata.")
            
        if 'metadati_normalizzazione' in support_filtered.columns:
            norm_scala = support_filtered[support_filtered['metadati_normalizzazione']=='scala']['variable_code'].tolist()
            norm_intervallo = support_filtered[support_filtered['metadati_normalizzazione']=='intervallo']['variable_code'].tolist()
            norm_volume = support_filtered[support_filtered['metadati_normalizzazione']=='volume']['variable_code'].tolist()
            self.metadata_costum['norm_scala'] = [x for x in df.columns if x.split('/')[0] in norm_scala]
            self.metadata_costum['norm_intervallo'] = [x for x in df.columns if x.split('/')[0] in norm_intervallo]
            self.metadata_costum['norm_volume'] = [x for x in df.columns if x.split('/')[0] in norm_volume]
            #print('scala', norm_scala, '\nintervallo', norm_intervallo)
            if norm_scala or norm_intervallo:
                # Estrazione altri metadata per la normalizzazione da un file Jaison
                norm_metadata =  self.get_normalization_settings(df)
                self.metadata_costum['norm_scale_value'] = norm_metadata
            else:
                self.metadata_costum['norm_scale_value'] = []
            if norm_volume:
                # Estrazione altri metadata per la normalizzazione di volumi da un file Jaison
                volume_metadata = self.get_normalization_settings(df, file_name='volume_values_settings.json')
                self.metadata_costum['volume_norm_values'] = volume_metadata
            else:
                self.metadata_costum['volume_norm_values'] = []
        else:
            print("Attenzione: la colonna 'metadati_normalizzazione' non è stata trovata.")

        # aggiornamento level del file
        self.metadata_costum['level'] = new_level

        # restituzione metadati
        metadata = self.metadata_costum

        return metadata 

    def file_versions_name(self, file_name, suffix):
        import re
        
        # Se il file ha un'estensione
        if '.' in file_name:
            name, ext = file_name.rsplit('.', 1)
            
            # Controlla se il nome del file ha già un suffisso di versione (_numero)
            # Pattern per trovare suffissi come _1, _01, _23, ecc.
            version_pattern = r'_(\d+)$'
            match = re.search(version_pattern, name)
            
            if match:
                # Rimuovi il suffisso di versione esistente
                name_without_version = re.sub(version_pattern, '', name)
                new_file_name = f"{name_without_version}{suffix}.{ext}"
            else:
                # Nessun suffisso di versione esistente, aggiungi quello nuovo
                new_file_name = f"{name}{suffix}.{ext}"
        else:
            # Se il file non ha estensione, controlla comunque per suffissi di versione
            version_pattern = r'_(\d+)$'
            match = re.search(version_pattern, file_name)
            
            if match:
                # Rimuovi il suffisso di versione esistente
                name_without_version = re.sub(version_pattern, '', file_name)
                new_file_name = name_without_version + suffix
            else:
                # Nessun suffisso di versione esistente, aggiungi quello nuovo
                new_file_name = file_name + suffix
                
        return new_file_name
    
    def classes_to_dummies(self, df, col_list, prefix_step='/'):
        # col_list è la lista di variabili da rendere dummies
        # classi per mappare il passaggio da classe int a classe stringa
        classes = {'GENDER':{0 :'female', 1 :'male'},
                   'MARRY':{1: 'married', 2:'divorced', 3: 'widowed', 0:'single'},
                   'ETHNICITY': {0: 'not_latino', 1: 'latino'}, 
                   'RACE': {5: 'White', 0: 'Mixed', 4: 'Black', 2: 'Asian', 1: 'Native_american', 'Pacific': 3},
                   'DX': {0: 'CN',1: 'MCI', 2: 'Dementia'}}
        ranges = {'GENDER':[0, 1],
                   'MARRY':[0, 1, 2, 3],
                   'ETHNICITY': [0, 1], 
                   'RACE': [0, 1, 2, 3, 4, 5],
                   'DX': [0, 1, 2]}
        
        # copia la lista di variabili da rendere booleane
        col_list_new = col_list.copy()
        
        if col_list_new:
            for col in col_list_new:
                # Verifica che tutti i valori nella colonna rientrino nei range validi
                unique_values = df[col].dropna().unique()
                invalid_values = [val for val in unique_values if val not in ranges[col]]
                
                if invalid_values:
                    print(f"La colonna '{col}' ha valori non validi: {invalid_values}. I valori validi sono: {ranges[col]}")
                    continue  # Salta questa colonna se ha valori non validi
                else:
                    # Per ogni colonna in col_list, se col presente nel df
                    # trasforma i valori in interi se sono float o stringhe numeriche
                    df[col] = df[col].apply(
                        lambda x: int(float(x)) if (isinstance(x, (float, str)) and str(x).replace('.', '', 1).isdigit()) else x
                    )
                    # conversione da classe numerica in classe stringa
                    df[col] = df[col].map(classes[col])
            
            # Salva le colonne originali prima della creazione delle dummies
            original_columns = set(df.columns)
            # Conversione delle colonne di classe in variabili dummies
            df = pd.get_dummies(df, columns=col_list_new, prefix=col_list_new, prefix_sep=prefix_step)
            # Trova le nuove colonne dummies create
            new_columns = [col for col in df.columns if col not in original_columns]
        else:
            new_columns = []
        return df, new_columns
    
    def get_normalization_settings(self, df, additional_scales=None, file_name = 'normalization_settings.json'):
        '''
        This function reads a normalization_settings.json file and returns a dictionary containing
        only the keys (column names) and their respective min/max values that are actually present
        in the dataframe. Optionally accepts additional_scales parameter to add missing scales
        and updates the JSON file.
        
        Parameters:
        df (pandas.DataFrame): The dataframe to check columns against
        additional_scales (dict, optional): Dictionary with column names as keys and [min, max] arrays as values
                                           to add scales not present in the JSON file
        
        Returns:
        dict: Dictionary containing only the normalization settings for columns present in the dataframe
        '''
        # Path to the normalization settings file
        json_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), file_name)
        
        # Load existing normalization settings
        try:
            with open(json_path, 'r') as f:
                normalization_settings = json.load(f)
        except FileNotFoundError:
            print(f"Warning: {json_path} not found. Starting with empty settings.")
            normalization_settings = {}
        except json.JSONDecodeError:
            print(f"Warning: {json_path} contains invalid JSON. Starting with empty settings.")
            normalization_settings = {}
        
        # Get dataframe columns
        df_columns = set(df.columns)
        
        # Filter normalization settings to include only columns present in the dataframe
        filtered_settings = {key: value for key, value in normalization_settings.items() if key in df_columns}
        
        # Handle additional scales if provided
        if additional_scales is not None:
            # Add new scales to the filtered settings
            for column, scale_range in additional_scales.items():
                if column in df_columns:
                    filtered_settings[column] = scale_range
                    # Also add to the original settings for saving back to JSON
                    normalization_settings[column] = scale_range
            
            # Save updated settings back to JSON file
            try:
                with open(json_path, 'w') as f:
                    json.dump(normalization_settings, f, indent=4)
                print(f"Updated normalization settings saved to {json_path}")
            except Exception as e:
                print(f"Warning: Could not save updated settings to {json_path}. Error: {e}")
        
        return filtered_settings

    
    def get_volumes_total(self, df: pd.DataFrame, file_code: str, prefix: str = 'raw') -> pd.DataFrame:
        # Filter the support file for the current file code
        support_file_for_file = self.support_file[self.support_file['file_code'] == file_code]
        
        # Get the volumes columns
        volume_original_columns = support_file_for_file[support_file_for_file['metadati_normalizzazione'] == 'volume']['variable_code'].tolist()
        # lista di volumi già 'totali' quindi da tenere
        volume_list = [x for x in volume_original_columns if x[0] not in ['R', 'L']]
        # lista di volumi da sommare per ottenere il volume totale, la label però resta generica, seza R e L così da sapere già come nominare la nuova colonna
        vol_tot_labels = list({x[1:] for x in volume_original_columns if x and x[0] in ["R","L"]})
        # lista di volumi da rimuovere dal df
        volumes_to_remove = []
        for label in vol_tot_labels:
            if ['R'+label, 'L'+label] in volume_original_columns:
                df[label] = df['R'+label] + df['L'+label]
                volume_list.append(label)
                volumes_to_remove += ['R'+label,'L'+label]
        
        # rimozione dei volumi da rimuovere dal df
        df_processed = df.drop(columns=volumes_to_remove)

        return df_processed, volume_list
    
    
    def to_ICV_percentage(self, df: pd.DataFrame, ICV_column: str, volume_column: str) -> pd.DataFrame:         # messa direttamente dentro alla funzione transform_volumes_as_ICV_percent# Ho cambiato il nome per evitare confusione con la funzione transform_volumes_as_ICV_percent
        # Transform the volumes as ICV percentage
        df[volume_column] = (df[volume_column] / df[ICV_column]) * 100
        return df

    def transform_volumes_as_ICV_percent(self, df: pd.DataFrame, volume_list: list, file_code: str) -> pd.DataFrame:
        volumes_columns = [x for x in volume_list if x in df.columns]
        # Transform the volumes as ICV percentage
        if len(volumes_columns) > 0 and 'ICV' in df.columns:
            for col in volumes_columns:
                df[col+'%ICV'] = (df[col] / df['ICV']) * 100
        else:
            print(f"No volumes columns found for file code {file_code}")

        return df