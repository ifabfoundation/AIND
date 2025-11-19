import pandas as pd
import numpy as np
from dl_client import DatalakeClient
import warnings
import os
warnings.filterwarnings('ignore')


def save_df(df_to_save, output_path):
    """
    Salva un DataFrame in un file  e csv.

    Args:
        df_to_save (pd.DataFrame): Il DataFrame da salvare.
        output_path (str): Il percorso (incluso il nome del file) dove salvare il file Excel.
    """
    df_to_save.to_excel(output_path+'.xlsx', index=False)
    df_to_save.to_csv(output_path+'.csv', index=False)


def create_new_support_file(support_file, support_file_path, new_name=None, rename_column=True):
    """
    Crea una copia di `self.support_file`, aggiunge la colonna 'new_variable_code',
    salva il nuovo DataFrame in un file Excel e lo restituisce.
    Il nuovo file viene salvato nella stessa directory dell'originale, con 'new_' 
    aggiunto all'inizio del nome del file.
    """
    support_file = support_file.copy(deep=True)
    new_support_file = support_file.copy(deep=True)
    
    # Trova la posizione della colonna 'variable_code'
    variable_code_loc = new_support_file.columns.get_loc('variable_code')
    if rename_column:
        new_support_file.rename(columns={'variable_code': 'orig_variable_code'}, inplace=True)
        # Inserisce la nuova colonna con valore nullo
        new_support_file.insert(
            loc=variable_code_loc + 1,
            column='variable_code',
            value=new_support_file['orig_variable_code']
        )
        

    # Svuota le celle delle colonne specificate
    cols_to_empty = ['type_variable', 'classes', 'range', 'valid_values', 'missing_values', 'missing_pop']
    for col in cols_to_empty:
        if col in new_support_file.columns:
            new_support_file[col] = np.nan

    directory, filename = os.path.split(support_file_path)
    
    if new_name:
        new_filename = new_name
    else: 
        new_filename = 'new_' + filename
    
    new_output_path = os.path.join(directory, new_filename)
    print(f'The {new_name} file has been created.\nOpen the file and verify it, if needed update the variables names and metadata')

    save_df(new_support_file, new_output_path)     # si può rimuovere il return quando si vede che funziona in quanto mi interessa poi aprire l'excel inserire i nuovi nomi delle variabili

def update_new_support_file(support_file, new_support_file_name, processed_file=[]):
    """
    Aggiorna il file di supporto con i nuovi file
    """
    support = support_file.copy(deep=True)
    new_support_file = pd.read_excel(new_support_file_name+'.xlsx')
    #file_code nuovi, presenti nel support file originale ma non nel nuovo support file
    new_file_code = [x for x in support['file_code'].unique() if x not in new_support_file['file_code'].unique()]
    #filecode già presenti ma da riprocessare
    to_restore = [x for x in processed_file if x in new_support_file['file_code'].unique()]

    # sezione del file originale di supporto da aggiungere al nuovo support file, in base ai file_code nuovi
    section_to_add = support[support['file_code'].isin(new_file_code)]

    # Svuota le celle delle colonne specificate
    cols_to_empty = ['type_variable', 'classes', 'range', 'valid_values', 'missing_values', 'missing_pop']
    for col in cols_to_empty:
        if col in section_to_add.columns:
            section_to_add[col] = np.nan

    # aggiunge la sezione al nuovo support file
    new_support_file = pd.concat([new_support_file, section_to_add], ignore_index=True)
    print(f'The {new_support_file_name} file has been updated with the new file_code: {new_file_code}\nOpen the file and verify it, if needed update the variables names and metadata')
    
    # se però sto rièorcessando un file di cui avevo gìà preso informazioni allora porto allo stato precedente le variabili per quel file
    if to_restore:
        s1 = new_support_file.loc[new_support_file['file_code']==to_restore[0], 'variable_code']
        s2 = support.loc[support['file_code']==to_restore[0], 'variable_code']
        code_restored = []
        for file_code in to_restore:
            if not s1.equals(s2):
                code_restored.append(file_code)
                # trova sezione del support file originale per il file_code corrente
                reprocessed_file = support[support['file_code']== file_code]
                for col in cols_to_empty:
                    if col in reprocessed_file.columns:
                        reprocessed_file[col] = np.nan
                # trova gli indici per dividere il new support file originale in due parti e quindi reinserire la sezione del support file originale tra le due parti
                index_up = new_support_file[new_support_file['file_code'] == file_code].index[0]
                index_down = new_support_file[new_support_file['file_code'] == file_code].index[-1]+1
                # divisione del support file originale in due parti dopo l'indice trovato
                df_up = new_support_file.iloc[:index_up]
                df_down = new_support_file.iloc[index_down:]
                # Aggiungi il le nuove righe tra le due parti del support file originale ==> support file aggiornato
                new_support_file = pd.concat([df_up, reprocessed_file, df_down], ignore_index=True)
        print(f'The {new_support_file_name} file has restored the previous information of the file_code: {code_restored}\nOpen the file and verify it, if needed update the variables names and metadata')
    

    

    save_df(new_support_file, new_support_file_name)

class InfoSupportFile:
    def __init__(self, support_file, df, file_name, prefix='raw'):
        self.client = DatalakeClient()
        self.support_file = support_file
        self.df = df
        self.file_name = file_name
        self.population = None
        # Ottiene i metadati del file dal data lake
        metadata = self.client.get_metadata(
            object_name = prefix + '/' + self.file_name
        )
        # Estrazione del file_code dai metadati
        file_code = metadata['metadata']['custom']['file_code']
        self.file_code = file_code
        self.metadata = metadata['metadata']['custom']
        
    def filter_variables(self):
        '''
        Rimuove le righe dal file di supporto (`self.support_file`) che, per un dato `file_code`, 
        contengono variabili (`variable_code`) non presenti nelle colonne del DataFrame `df`.
        '''
        # Controlla che il file_code sia presente nel support_file
        if self.file_code not in self.support_file['file_code'].values:
            print(f"file_code '{self.file_code}' non trovato nel support_file.")
            return self.support_file, self.file_code

        # Ottiene i codici delle variabili unici dal file di supporto per il file_code corrente
        support_file_vars = self.support_file[self.support_file['file_code'] == self.file_code]['variable_code'].unique()

        # Identifica le variabili presenti nel file di supporto ma non nelle colonne del df
        vars_not_in_df = [var for var in support_file_vars if var not in self.df.columns]

        # Ottiene gli indici delle righe da rimuovere
        indices_to_remove = self.support_file[
            (self.support_file['file_code'] == self.file_code) &
            (self.support_file['variable_code'].isin(vars_not_in_df))
        ].index

        # Rimuove le righe identificate dal dataframe del file di supporto
        self.support_file.drop(indices_to_remove, inplace=True)
        return self.support_file, self.file_code
    
    def find_population_variable(self):
        """
        Identifica la variabile di popolazione nei dati ADNI.

        Args:
            df (pd.DataFrame): DataFrame contenente i dati ADNI
            file_code (str): Codice del file per l'identificazione
            support_file (pd.DataFrame): DataFrame di supporto per salvare le informazioni

        Returns:
            tuple: (str or None, pd.DataFrame) Nome della variabile di popolazione identificata e support_file aggiornato
        """
        adni_versions = ['ADNI1', 'ADNI2', 'ADNIGO', 'ADNI3', 'ADNI4']
        key_population = [col for col in self.df.columns if any(self.df[col].astype(str).str.contains(ver).any() for ver in adni_versions)]

        population = None
        warning_msg = None

        if len(key_population) == 1:
            population = key_population[0]
        elif len(key_population) == 2:
            # Se le due colonne sono identiche, usa la prima
            if self.df[key_population[0]].equals(self.df[key_population[1]]):
                population = key_population[0]
            else:
                # Cerca la colonna che varia tra i soggetti
                for key in key_population:
                    if not self.df.groupby('RID')[key].nunique().eq(1).all():
                        population = key
                if population is None:
                    warning_msg = f"{self.file_code}\nPROBLEMA: 2 chiavi di popolazione diverse, ma con tutti valori uguali"
        elif len(key_population) > 2:
            warning_msg = f"{self.file_code}\nPROBLEMA: più di 2 chiavi di popolazione trovate: {key_population}"
        else:
            warning_msg = f"{self.file_code}\nPROBLEMA: nessuna chiave di popolazione trovata"

        if warning_msg:
            print(warning_msg)

        # Aggiorna il file di supporto se necessario
        support_file_new = self.support_file.copy(deep=True)
        if population is not None:
            try:
                if population not in self.support_file[self.support_file['file_code'] == self.file_code]['variable_code'].values:
                    filtered_df = self.support_file[self.support_file['file_code'] == self.file_code]
                    if not filtered_df.empty:
                        index = filtered_df.index[0] + 1
                        file_name = filtered_df['file_name'].iloc[0]
                        new_row = pd.Series({
                            'file_name': file_name,
                            'file_code': self.file_code,
                            'parameter': 'Cohort',
                            'population': None,
                            'variable_code': population,
                            'type_variable': None,
                            'classes': None,
                            'range': None,
                            'valid_values': None,
                            'missing_values': None,
                            'missing_pop': None,
                            'del': 'keep',
                        })
                        support_file_new = pd.concat([
                            self.support_file.iloc[:index],
                            pd.DataFrame([new_row]),
                            self.support_file.iloc[index:]
                        ]).reset_index(drop=True)
                else:
                    # For the row of the population, impose 'Cohort' in the column 'parameter'
                    idx = self.support_file[
                        (self.support_file['file_code'] == self.file_code) &
                        (self.support_file['variable_code'] == population)
                    ].index
                    if not idx.empty:
                        support_file_new.loc[idx, 'parameter'] = 'Cohort'
            except Exception as e:
                print(f"Errore nell'aggiornamento del file di supporto: {e}")
                support_file_new = self.support_file
            
            self.support_file = support_file_new
            self.population = population
        return self.population, self.support_file

    def check_missing_values(self):  
        # Verifica che la colonna esista nel DataFrame
        if self.key not in self.df.columns:
            print(f"Colonna '{self.key}' non trovata nel DataFrame")
            return None, None, None, ['pop not found'], ['pop not found']
        
        # Calcola i valori mancanti
        missing_series = self.df[self.key].isna()
        if isinstance(missing_series, pd.Series):
            n_missing = int(missing_series.sum())
        else:
            n_missing = int(missing_series)
        
        n_tot = int(self.df.shape[0])
        n_valid = int(n_tot - n_missing)
        
        if self.population is not None and self.population in self.df.columns:
            pop = ['ADNI1', 'ADNIGO', 'ADNI2', 'ADNI3', 'ADNI4']
            pop_valid = self.df[self.df[self.key].isna() == False][self.population].unique().tolist()
            pop_missing = [x for x in pop if x not in pop_valid]
        else:
            pop_valid = ['pop not found']
            pop_missing = ['pop not found']
        
#        warnings.warn(
#            f"Variabile: {self.key} - Valori mancanti/totali: {n_missing}/{n_tot}, "
#            f"Popolazioni con valori mancanti: {pop_missing}"
#        )
        
        return n_tot, n_valid, n_missing, pop_valid, pop_missing
    
    def check_type_range_variables(self):
        # Verifica che la colonna esista nel DataFrame
        if self.key not in self.df.columns:
            print(f"Colonna '{self.key}' non trovata nel DataFrame")
            return None, [None], [None]
        
        n = self.df[self.key].first_valid_index()
        #print(self.key, n)
        if n is None:
            tipo = None
            raw_tipo = None
            intervallo = [None]
            classes = [None]
            print(self.key, ' non ha valori')
            return tipo, intervallo, classes
        else:
            raw_tipo = type(self.df[self.key][n])  # numpy.int64
            tipo = str(raw_tipo.__name__)
            options = self.df[self.key].unique()
            # intervallo
            if 'float' in tipo or 'int' in tipo:
                intervallo = [float(self.df[self.key].max()), float(self.df[self.key].min())]
            else:
                intervallo = [None]
            
            #classi
            if len(options) <= 10:
                classes = options
            elif tipo == 'str':
                classes = options[:5]
            else:
                classes = [None]

            return raw_tipo, intervallo, classes
    
    def get_varible_info(self, key):
        self.key = key

        if self.population is None and 'Cohort' in self.support_file[self.support_file['file_code']==self.file_code]['parameter'].values:
            ind = self.support_file[(self.support_file['file_code']==self.file_code) & (self.support_file['parameter']=='Cohort')].index[0]  
            self.population =self.support_file['variable_code'].loc[ind]

        n_tot, n_valid, n_missing, _, pop_missing = self.check_missing_values()
        tipo, intervallo, classes = self.check_type_range_variables()
        #print('===='+key+'====\n', 'tipo', tipo, '\nintervallo', intervallo, '\nclasses', classes)

        # Verifica che la variabile esista nel support file
        matching_rows = self.support_file[(self.support_file['file_code'] == self.file_code) & (self.support_file['variable_code'] == self.key)]
        if matching_rows.empty:
            print(f"Variabile '{self.key}' non trovata nel support file per il file_code '{self.file_code}'")
            return self.support_file
        
        index = matching_rows.index[0]
        self.support_file['type_variable'][index] = str(tipo)
        self.support_file['classes'][index] = ', '.join(map(str, classes))
        self.support_file['range'][index] = ', '.join(map(str, intervallo))
        self.support_file['valid_values'][index] = int(n_valid)
        self.support_file['missing_values'][index] = int(n_missing)
        self.support_file['missing_pop'][index] = ', '.join(map(str, pop_missing))
        
        if n_tot > 0:
            if n_valid / n_tot <= 0.65:
                self.support_file['del'][index] = 'drop'
            else:
                self.support_file['del'][index] = 'keep'
        else:
            # Se n_tot è 0, la colonna è vuota, quindi la marchiamo per l'eliminazione
            self.support_file['del'][index] = 'drop'
            print('usata scappatoia')

        return self.support_file

    def get_present_populations(self, file_code=None):
        """
        Estrae le popolazioni effettivamente presenti per un certo file_code dal support file.
        - Filtra new_support_file per file_code
        - Dalla colonna 'missing_pop' estrae le popolazioni che sono presenti in tutte le righe (completely missing)
        - Sottrae queste dalla lista delle possibili popolazioni
        - Restituisce la lista delle popolazioni effettivamente presenti
        """
        if file_code != None:
            self.file_code = file_code

        possible_population = ['ADNI1', 'ADNIGO', 'ADNI2', 'ADNI3', 'ADNI4']
        filtered = self.support_file[self.support_file['file_code'] == self.file_code]
        if filtered.empty or 'missing_pop' not in filtered.columns:
            return []  # Se non c'è info, restituisci lista vuoto
        if 'pop not found' in list(filtered['missing_pop'].unique()):
            print('WARNING: "Population not found" ==> need to manually adjust the metadata manually')
            return []
        # Trova tutte le popolazioni che sono presenti in tutte le righe della colonna 'missing_pop'
        missing_lists = filtered['missing_pop'].dropna().apply(lambda x: [s.strip() for s in str(x).split(',') if s.strip()])
        if missing_lists.empty:
            completely_missing_population = []
        else:
            # Intersezione di tutte le popolazioni mancanti
            completely_missing_population = set(missing_lists.iloc[0])
            for pop_list in missing_lists.iloc[1:]:
                completely_missing_population &= set(pop_list)
        # Sottrai le popolazioni completamente mancanti da quelle possibili
        present_population = [pop for pop in possible_population if pop not in completely_missing_population]
        return present_population

    def get_subjects_and_multiplevisits(self, key):
        """
        Conta il numero di soggetti che hanno la variabile key e il numero di soggetti 
        che hanno più di una visita con quella variabile valida.
        """
        # --- Validation --- #
        if key not in self.df.columns:
            raise ValueError(f"La colonna '{key}' non esiste nel DataFrame")

        if 'RID' not in self.df.columns:
            raise ValueError("La colonna 'RID' non esiste nel DataFrame")

        # --- Filter valid rows --- #
        df_filtered = self.df[self.df[key].notna()].copy()

        # Totale soggetti unici
        subject_tot = df_filtered['RID'].nunique()

        # Soggetti con più di una visita
        rid_counts = df_filtered['RID'].value_counts()
        subject_multivisit = (rid_counts >= 2).sum()

        # --- Ensure columns exist in support_file --- #
        for col in ['subj_tot', 'subj_multiple_visits']:
            if col not in self.support_file.columns:
                self.support_file[col] = None

        # --- Locate the correct row in support_file --- #
        mask = (
            (self.support_file['file_code'] == self.file_code) &
            (self.support_file['variable_code'] == key)
        )

        if not mask.any():
            raise ValueError(
                f"Nessuna riga trovata in support_file per file_code='{self.file_code}', "
                f"variable_code='{key}'"
            )

        index = self.support_file.loc[mask].index[0]

        # --- Safe, non-chained assignment --- #
        self.support_file.loc[index, 'subj_tot'] = subject_tot
        self.support_file.loc[index, 'subj_multiple_visits'] = subject_multivisit

        return self.support_file


        