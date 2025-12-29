
import stat
import pandas as pd
import numpy as np
import re
import os
import json
from typing import Tuple, List, Dict, Any, Callable

class MergerTools:

    # ==========================================================================
    # COSTANTI DI CLASSE
    # ==========================================================================
    
    COHORT_GERARCHIA = ['ADNI1', 'ADNIGO', 'ADNI2', 'ADNI3', 'ADNI4']
    
    CATEGORY_CONFIG = {
        'volumes': {
            'key_cols': ['RID', 'EXAMDATE', 'FSVERSION', 'IMAGEUID'],
            'colonne_escluse': ['MRI_SCANDATE','IMAGEUID', 'FSVERSION', 'FLDSTRENG', 'STATUS', 'ICV%ICV', 'Brain%ICV', 'Ventricles%ICV', 'Hippocampus%ICV', 'Entorhinal%ICV', 'Fusiform%ICV', 'MidTemp%ICV'],
        },
        'scale': {
            'key_cols': ['RID', 'EXAMDATE'],
            'colonne_escluse': [],
        },
        'csf': {
            'key_cols': ['RID', 'EXAMDATE', 'METHOD'],
            'colonne_escluse': ['METHOD', "AB40_CSF", "AB42_CSF", "AB4240_CSF", "PT181_CSF", "TTAU_CSF", "PT181_AB42_CSF", "TTAU_AB42_CSF" ],
        },
        'plasma': {
            'key_cols': ['RID', 'EXAMDATE', 'METHOD'],
            'colonne_escluse': [],
        },
        'pet': {
            'key_cols': ['RID', 'EXAMDATE', 'METHOD'],
            'colonne_escluse': [],
        },
        'cofactor': {
            'key_cols': ['RID', 'EXAMDATE'],
            'colonne_escluse': [],
        },
    }

    BASE_KEYS = ['RID', 'EXAMDATE', 'SCANDATE', 'VISCODE', 'VISIT_MONTH', 'COHORT', 'update_stamp', 'METHOD']

    CATEGORY_KEYS = {'volumes': ['MRI_SCANDATE','IMAGEUID', 'FSVERSION', 'FLDSTRENG', 'STATUS', 'ICV%ICV', 'Brain%ICV', 'Ventricles%ICV', 'Hippocampus%ICV', 'Entorhinal%ICV', 'Fusiform%ICV', 'MidTemp%ICV'],
                    'scale': ['MMSE', 'RAVLT_immediate', 'FAQ', 'MOCA', 'CDRSB', 'CDRGLOB', 'ADAS11', 'ADAS13'],
                    'csf': ["AB40_CSF", "AB42_CSF", "AB4240_CSF", "PT181_CSF", "TTAU_CSF", "PT181_AB42_CSF", "TTAU_AB42_CSF" ],
                    'plasma': [ "AB40_PL", "AB42_PL", "AB4240_PL", "PT181_PL", "PT217_AB42_PL", "TTAU_PL", "nPT217_PL", "PT217_nPT217_PL", "ALPHA_SYN", "NFL_PL", "GFAP"],
                    'pet': [ 'TRACER', "AMY_CENTILOIDS", "SUMMARY_SUVR", "PRECUNEUS_SUVR", "TAU_METAROI", 
                                "INFERIORPARIETAL_SUVR", "PARAHIPPOCAMPAL_SUVR", "LATERALOCCIPITAL_SUVR",
                                "MIDDLETEMPORAL_SUVR", "INFERIOR_TEMPORAL_SUVR", "ENTORHINAL_SUVR", 
                                "FUSIFORM_SUVR"],
                    'cofactor': ['AGE', 'AGE_AD_BEG', 'AGE_AD_DX', 'AGE_COG_BEG', 'AGE_bl', 'GENDER/female', 'GENDER/male', 'DX/CN', 'DX/Dementia', 'DX/MCI', 'EDUCAT',  
                                'MARRY/divorced', 'MARRY/married', 'ETHNICITY/latino', 'ETHNICITY/not_latino', 'MARRY/single', 'MARRY/widowed', 'RACE/Asian', 'RACE/Black', 
                                'RACE/Mixed', 'RACE/Native_american', 'RACE/White', 'APOE', 'APOE_4', 'DIAN_MUTATION', 'Aprofile', 'Tprofile', 'Nprofile']}
    COFACTOR_CONFIG = {
        'immutable': [
            'APOE', 'APOE_4', 'DIAN_MUTATION', 'AGE_bl'
        ],
        'immutable_single_df': ['AGE_AD_BEG', 'AGE_AD_DX', 'AGE_COG_BEG'],
        'mutable_single_df': ['Aprofile', 'Tprofile', 'Nprofile'],
        
        # Gruppi dummy per tipo
        'dummy_groups_immutable': {
            'GENDER': ['GENDER/female', 'GENDER/male'],
            'ETHNICITY': ['ETHNICITY/latino', 'ETHNICITY/not_latino'],
            'RACE': ['RACE/Asian', 'RACE/Black', 'RACE/Mixed', 'RACE/Native_american', 'RACE/White'],
        },
        'dummy_groups_mutable': {
            'DX': ['DX/CN', 'DX/MCI', 'DX/Dementia']
            # MARRY rimosso: ha logica speciale (sorgente a livello RID + forward-fill)
        },
        'dummy_marry': ['MARRY/divorced', 'MARRY/married', 'MARRY/single', 'MARRY/widowed'],
        
        'custom_resolvers': ['COHORT', 'AGE'], 
        'forward_fill_cols': ['EDUCAT']
    }
    

    def __init__(self, fix_dummy_violations: bool = False):
        # Carica ogni JSON una sola volta
        normalization_full = self.get_json_file('normalization_settings.json')
        volume_full = self.get_json_file('volume_values_settings.json')
        cofactor_full = self.get_json_file('cofattori_values_settings.json')
        
        # Filtra per categoria
        self.trend_reference = {
            'volumes': self._filter_dict(volume_full, 'volumes'),
            'scale': self._filter_dict(normalization_full, 'scale'),
            'csf': self._filter_dict(normalization_full, 'csf'),
            'plasma': self._filter_dict(normalization_full, 'plasma'),
            'pet': self._filter_dict(normalization_full, 'pet'),
            'cofactor': self._filter_dict(cofactor_full, 'cofactor'),
        }

        self.fix_dummy_violations = fix_dummy_violations
        self._resolvers: Dict[str, Callable] = {
            'COHORT': self._resolve_cohort,
            'AGE': self._resolve_age,
        }
        
        # Unione di tutti i gruppi dummy per validazione finale
        self._all_dummy_groups = {
            **self.COFACTOR_CONFIG['dummy_groups_immutable'],
            **self.COFACTOR_CONFIG['dummy_groups_mutable'],
            'MARRY': self.COFACTOR_CONFIG['dummy_marry']
        }




    def _filter_dict(self, full_dict, category):
        """Filtra dizionario per le chiavi della categoria."""
        valid_keys = self.CATEGORY_KEYS.get(category, [])
        return {k: v for k, v in full_dict.items() if k in valid_keys}

    def get_json_file(self, file_name):
        """
        Cerca il file di configurazione JSON sia nella cartella corrente
        (Data_cleaning) sia in quella superiore (WP-2) per garantire
        retrocompatibilità con vecchi percorsi.
        """
        base_dir = os.path.dirname(__file__)
        candidates = [
            os.path.join(base_dir, file_name),
            os.path.join(os.path.dirname(base_dir), file_name),
        ]

        for json_path in candidates:
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    return json.load(f)

        raise FileNotFoundError(f'File JSON non trovato. Percorsi verificati: {candidates}')

    def _convert_nan_to_none(self, obj):
        """
        Converte ricorsivamente tutti i NaN in None per la serializzazione JSON.
        """
        if isinstance(obj, (list, tuple)):
            return [self._convert_nan_to_none(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: self._convert_nan_to_none(value) for key, value in obj.items()}
        elif isinstance(obj, (np.floating, float)) and (np.isnan(obj) or pd.isna(obj)):
            return None
        elif isinstance(obj, np.ndarray):
            return [self._convert_nan_to_none(item) for item in obj]
        elif pd.isna(obj):
            return None
        else:
            return obj

    def calculate_visit_month(self,df, rid_col='RID', date_col='EXAMDATE', visit_col='VISIT_MONTH'):
        """
        Calcola VISIT_MONTH per ogni soggetto come differenza in mesi dalla prima visita.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame contenente i dati delle visite
        rid_col : str, default 'RID'
            Nome della colonna identificativo soggetto
        date_col : str, default 'EXAMDATE'
            Nome della colonna con le date delle visite
        visit_col : str, default 'VISIT_MONTH'
            Nome della colonna di output per i mesi
        
        Returns
        -------
        pd.DataFrame
            DataFrame con la colonna VISIT_MONTH aggiornata
        """
        
        # Copia per evitare modifiche inplace
        df = df.copy()
        
        # Converti in datetime se non lo è già
        if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            df[date_col] = pd.to_datetime(df[date_col])
        
        # Calcola la data della prima visita per ogni soggetto
        first_exam = df.groupby(rid_col)[date_col].transform('min')
        
        # Calcola VISIT_MONTH come differenza in mesi
        # Usa Int64 (nullable) per gestire eventuali NaN nelle date
        df[visit_col] = ((df[date_col] - first_exam).dt.days / 30.44).round().astype('Int64')
        
        return df
            
    def filter_df_category(self, df, category):
        
        base_keys = self.BASE_KEYS
        keys = self.CATEGORY_KEYS[category]

        if len(set(df.columns)&set(keys)) > 0:
            col_list = [x for x in df.columns if x in base_keys + keys]
            df_new = df[col_list].copy(deep=True)
            check_list = [x for x in col_list if x not in base_keys and x not in ['METHOD']] 
            cleaned_df = df_new.dropna(subset=check_list, how='all')
        else:
            print(f"WARNING: il df non contiene colonne di {category}")
            cleaned_df = pd.DataFrame()

        return cleaned_df

    '''
    def find_visit_matches(self,df1, df2, rid_col='RID', date_col='EXAMDATE', buffer_days=pd.Timedelta(days=0)):
        """
        Trova corrispondenze tra visite di due dataset longitudinali.
        
        Returns:
            exact_matches: {(RID, EXAMDATE): [[indici_df1], [indici_df2]]}
            buffer_matches: {(RID, EXAMDATE1, EXAMDATE2): [[indici_df1], [indici_df2]]}
        """
        # Copia per non modificare gli originali e assicura datetime
        df1 = df1.copy()
        df2 = df2.copy()
        df1[date_col] = pd.to_datetime(df1[date_col])
        df2[date_col] = pd.to_datetime(df2[date_col])
        
        exact_matches = {}
        buffer_matches = {}
        
        # Pre-raggruppa df2 per RID -> accesso O(1) invece di filtrare ogni volta
        df2_by_rid = {rid: group for rid, group in df2.groupby(rid_col)}
        
        # Itera su ogni combinazione (RID, EXAMDATE) in df1
        for (rid, date1), group1 in df1.groupby([rid_col, date_col]):
            # Se il paziente non esiste in df2, skip
            if rid not in df2_by_rid:
                continue
            
            group2_rid = df2_by_rid[rid]
            indices1 = group1.index.tolist()
            
            # 1. Cerca match ESATTO sulla data
            exact_mask = group2_rid[date_col] == date1
            
            if exact_mask.any():
                indices2 = group2_rid[exact_mask].index.tolist()
                exact_matches[(rid, date1)] = [indices1, indices2]
            else:
                # 2. Solo se NON c'è match esatto, cerca con BUFFER
                diffs = (group2_rid[date_col] - date1).abs()
                buffer_mask = (diffs > pd.Timedelta(0)) & (diffs <= buffer_days)
                
                if buffer_mask.any():
                    buffer_rows = group2_rid[buffer_mask]
                    # Raggruppa per data2 (potrebbero esserci più visite nella stessa data)
                    for date2, subgroup2 in buffer_rows.groupby(date_col):
                        indices2 = subgroup2.index.tolist()
                        buffer_matches[(rid, date1, date2)] = [indices1, indices2]
        
        return exact_matches,buffer_matches
        '''

    def find_visit_matches(self, df1, df2, rid_col='RID', date_col='EXAMDATE', buffer_days=pd.Timedelta(days=80)):
        """
        Trova corrispondenze tra visite di due dataset longitudinali.
        Ogni indice di df1 e df2 compare AL MASSIMO in un match.
        Priorità: exact match > buffer match (più vicino temporalmente).
        
        Returns:
            exact_matches: {(RID, EXAMDATE): [[indici_df1], [indici_df2]]}
            buffer_matches: {(RID, EXAMDATE1, EXAMDATE2): [[indici_df1], [indici_df2]]}
        """
        # Normalizza buffer_days
        if isinstance(buffer_days, (int, float)):
            buffer_days = pd.Timedelta(days=buffer_days)
        
        # Copia e converte date
        df1 = df1.copy()
        df2 = df2.copy()
        df1[date_col] = pd.to_datetime(df1[date_col])
        df2[date_col] = pd.to_datetime(df2[date_col])
        
        exact_matches = {}
        buffer_matches = {}
        used_idx1 = set()
        used_idx2 = set()
        
        # Pre-raggruppa per (RID, EXAMDATE) -> accesso O(1)
        df1_by_rid_date = {key: group for key, group in df1.groupby([rid_col, date_col])}
        df2_by_rid_date = {key: group for key, group in df2.groupby([rid_col, date_col])}
        
        # Pre-raggruppa le date di df2 per RID (per efficienza nel PASSO 2)
        df2_dates_by_rid = {}
        for (rid, date) in df2_by_rid_date.keys():
            if rid not in df2_dates_by_rid:
                df2_dates_by_rid[rid] = []
            df2_dates_by_rid[rid].append(date)
        
        # =====================================================
        # PASSO 1: Trova TUTTI i match ESATTI
        # =====================================================
        for (rid, date1), group1 in df1_by_rid_date.items():
            if (rid, date1) not in df2_by_rid_date:
                continue
            
            group2 = df2_by_rid_date[(rid, date1)]
            indices1 = group1.index.tolist()
            indices2 = group2.index.tolist()
            
            exact_matches[(rid, date1)] = [indices1, indices2]
            used_idx1.update(indices1)
            used_idx2.update(indices2)
        
        # =====================================================
        # PASSO 2: Raccogli TUTTI i potenziali buffer matches
        # =====================================================
        potential_buffers = []
        
        for (rid, date1) in df1_by_rid_date.keys():
            if rid not in df2_dates_by_rid:
                continue
            
            for date2 in df2_dates_by_rid[rid]:
                # Skip match esatti (già gestiti)
                if date1 == date2:
                    continue
                
                diff_td = abs(date2 - date1)
                if diff_td > buffer_days:
                    continue
                
                potential_buffers.append({
                    'rid': rid,
                    'date1': date1,
                    'date2': date2,
                    'diff_td': diff_td
                })
        
        # =====================================================
        # PASSO 3: Ordina per vicinanza e assegna greedy
        # =====================================================
        potential_buffers.sort(key=lambda x: x['diff_td'])
        
        for pb in potential_buffers:
            rid, date1, date2 = pb['rid'], pb['date1'], pb['date2']
            
            group1 = df1_by_rid_date[(rid, date1)]
            group2 = df2_by_rid_date[(rid, date2)]
            
            # Filtra indici disponibili (stesso metodo per entrambi)
            available_idx1 = group1.index.difference(used_idx1)
            available_idx2 = group2.index.difference(used_idx2)
            
            if available_idx1.empty or available_idx2.empty:
                continue
            
            buffer_matches[(rid, date1, date2)] = [available_idx1.tolist(), available_idx2.tolist()]
            used_idx1.update(available_idx1)
            used_idx2.update(available_idx2)
        
        return exact_matches, buffer_matches


    def verify_visit_matches(self, exact_matches, buffer_matches):
        """
        Verifica e stampa se ci sono sovrapposizioni tra exact e buffer matches.
        """
        def extract_indices(matches, pos):
            return {idx for indices_pair in matches.values() for idx in indices_pair[pos]}
        
        exact_idx1 = extract_indices(exact_matches, 0)
        exact_idx2 = extract_indices(exact_matches, 1)
        buffer_idx1 = extract_indices(buffer_matches, 0)
        buffer_idx2 = extract_indices(buffer_matches, 1)
        
        overlap_idx1 = exact_idx1 & buffer_idx1
        overlap_idx2 = exact_idx2 & buffer_idx2
        
        # Verifica df1
        if overlap_idx1:
            print(f"⚠️  OVERLAP df1: {len(overlap_idx1)} indici in comune")
            print(f"    Indici: {sorted(overlap_idx1)}")
        
        # Verifica df2
        if overlap_idx2:
            print(f"⚠️  OVERLAP df2: {len(overlap_idx2)} indici in comune")
            print(f"    Indici: {sorted(overlap_idx2)}")
        
        # Esito finale
        if not overlap_idx1 and not overlap_idx2:
            print("✓ Verifica matches superata: nessun overlap tra exact e buffer")

    def find_rid_matches(self, df1, df2, rid_col='RID'):
        """
        Trova corrispondenze tra righe di due dataset basandosi solo su RID.
        
        Returns:
            matches: {RID: [[indici_df1], [indici_df2]]}
        """
        matches = {}
        
        df1_grouped = df1.groupby(rid_col)
        df2_grouped = df2.groupby(rid_col)
        
        common_rids = set(df1[rid_col]) & set(df2[rid_col])
        
        for rid in common_rids:
            matches[rid] = [
                df1_grouped.get_group(rid).index.tolist(),
                df2_grouped.get_group(rid).index.tolist()
            ]
        
        return matches
        
    def list_index_visit_matches(self, matches):
        """
        Lista gli indici delle righe che matchano tra due dataframe.
        """
        index_list1 = []
        index_list2 = []
        for indici in matches.values():
            for ind1 in indici[0]:
                for ind2 in indici[1]:
                    index_list1.append(ind1)
                    index_list2.append(ind2)

        index_1 = pd.Index(index_list1)
        index_2 = pd.Index(index_list2)

        return index_1, index_2

    def matrix_match(self,dfs, df_names, df_code, columns_list=['RID'], time_buffer=pd.Timedelta(days=0)):
        """
        Crea una matrice di match per (RID, EXAMDATE) tra i dataframe in dfs.
        """
        # Assicurati che le colonne RID e EXAMDATE siano presenti e che EXAMDATE sia in formato datetime
        for df_name, df in dfs.items():
            for col in columns_list:
                if col not in df.columns:
                    print(f"AVVISO: {df_name} -> '{df_names[df_name]}' non contiene colonna {col}")

        # Matrice di match per (RID, EXAMDATE)
        keys_list = list(dfs.keys())
        match_matrix = pd.DataFrame(0, index=df_code, columns=df_code)
        columns_set = set(columns_list)
        for i, kI in enumerate(keys_list):
            for j, kJ in enumerate(keys_list):
                if 'EXAMDATE' in columns_list and ('EXAMDATE' not in dfs[kI].columns or 'EXAMDATE' not in dfs[kJ].columns):
                    match_matrix.iloc[i, j] = None  # Indica colonne mancanti
                    continue
                # Ci assicuriamo di confrontare solo se entrambe le colonne esistono nei dati
                if columns_set.issubset(dfs[kI].columns) and columns_set.issubset(dfs[kJ].columns):
                    if ['RID', 'EXAMDATE'] == columns_list:
                        match_matrix.iloc[i, j] = len(self.date_matches_with_buffer(dfs[kI], dfs[kJ], time_buffer)[0])
                    elif set(['RID', 'EXAMDATE']).issubset(set(columns_list)) and len(columns_list) > 2:
                        # Estrai le variabili extra (tutte le colonne tranne RID e EXAMDATE)
                        extra_vars = [col for col in columns_list if col not in ['RID', 'EXAMDATE']]
                        match_matrix.iloc[i, j] = len(self.date_matches_with_buffer_and_extra_var(dfs[kI], dfs[kJ], extra_vars, time_buffer)[0])
                    else:
                        s1 = set(dfs[kI][columns_list].drop_duplicates().itertuples(index=False, name=None))
                        s2 = set(dfs[kJ][columns_list].drop_duplicates().itertuples(index=False, name=None))
                        match_matrix.iloc[i, j] = len(s1 & s2)
                else:
                    match_matrix.iloc[kI, kJ] = None  # Indica colonne mancanti

        display_match_matrix = match_matrix.copy()
        mask = np.triu(np.ones(display_match_matrix.shape), k=1).astype(bool)
        display_match_matrix = display_match_matrix.mask(mask, "")
        
        return display_match_matrix

    def individual_match_and_missing_rows(self, df_0, dfs, df_names, columns_list=['RID'], time_buffer=pd.Timedelta(days=0)):
        """
        Per df_0, individua le righe (RID, EXAMDATE) che non sono presenti negli altri df in ingresso.
        """
        columns_set = set(columns_list)
        s1 = set(df_0[columns_list].drop_duplicates().itertuples(index=False, name=None)) if columns_set.issubset(df_0.columns) else set()

        for k in dfs.keys():
            if not columns_set.issubset(dfs[k].columns):
                print(f"AVVISO: '{k}' non contiene tutte le colonne {columns_list}")
                continue
            
            df_k = dfs[k].copy(deep=True)

            s2 = set(dfs[k][columns_list].drop_duplicates().itertuples(index=False, name=None))

            use_time_buffer = columns_list == ['RID', 'EXAMDATE'] and time_buffer != pd.Timedelta(days=0)

            if use_time_buffer:
                matches = self.date_matches_with_buffer(df_0, df_k, time_buffer)[0]
                matched_df0 = set(matches[['RID', 'EXAMDATE_1']].drop_duplicates().itertuples(index=False, name=None))
                matched_dfk = set(matches[['RID', 'EXAMDATE_2']].drop_duplicates().itertuples(index=False, name=None))
                both_count = len(matched_df0)
                not_in_other = s1 - matched_df0
                opposite = s2 - matched_dfk
            else:
                both = s1 & s2
                both_count = len(both)
                not_in_other = s1 - s2
                opposite = s2 - s1

            print(f"N righe {columns_list} comuni a {df_names[0]} e {df_names[k+1]}:", both_count)
            print(f"N righe {columns_list} di {df_names[0]} NON in {df_names[k+1]}:", len(not_in_other))
            print(f"N righe {columns_list} di {df_names[k+1]} NON in {df_names[0]}:", len(opposite), '\n')
    
    def date_matches_with_buffer(self, df1, df2, time_buffer=pd.Timedelta(days=0), print_info=False):

        """
        Trova match esatti + eventuali match entro buffer.
        Le date che hanno match esatto vengono *eliminate* dal confronto buffer
        per evitare accoppiamenti falsi.
        """

        # --- Step 1: Preprocessing ---
        d1 = df1[['RID', 'EXAMDATE']].drop_duplicates().copy(deep=True)
        d2 = df2[['RID', 'EXAMDATE']].drop_duplicates().copy(deep=True)

        d1['EXAMDATE'] = pd.to_datetime(d1['EXAMDATE'])
        d2['EXAMDATE'] = pd.to_datetime(d2['EXAMDATE'])

        merged = d1.merge(d2, on='RID', suffixes=('_1', '_2'))
        merged['date_diff'] = (merged['EXAMDATE_1'] - merged['EXAMDATE_2']).abs()

        # --- Step 2: Match esatti ---
        exact_matches = merged[merged['date_diff'] == pd.Timedelta(0)]

        # Lista (RID, EXAMDATE) da escludere dal buffer match
        exact_keys = set(
            exact_matches[['RID', 'EXAMDATE_1']]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        )

        # Se nessun buffer richiesto → ritorno match esatti e basta
        if time_buffer == pd.Timedelta(0):
            if print_info:
                print(f"[date_matches_with_buffer] Match esatti: {len(exact_matches)}")
            return exact_matches[['RID', 'EXAMDATE_1', 'EXAMDATE_2']].drop_duplicates()

        # --- Step 3: Rimuovere *tutte le righe* che coinvolgono una data già matchata esattamente ---
        if exact_keys:
            merged_no_exact = merged[
                ~(
                    merged.apply(lambda row:
                        (row['RID'], row['EXAMDATE_1']) in exact_keys or
                        (row['RID'], row['EXAMDATE_2']) in exact_keys,
                    axis=1)
                )
            ]
        else:
            merged_no_exact = merged.copy()

        # --- Step 4: Match entro buffer (escludendo match esatti) ---
        buffered_matches = merged_no_exact[
            (merged_no_exact['date_diff'] <= time_buffer) &
            (merged_no_exact['date_diff'] > pd.Timedelta(0))
        ]

        # --- Step 5: Unione finale ---
        matches = pd.concat([exact_matches, buffered_matches], ignore_index=True)
        matches = matches[['RID', 'EXAMDATE_1', 'EXAMDATE_2']].drop_duplicates()

        # --- Step 6: Info ---
        if print_info:
            print(f"[date_matches_with_buffer] \nExact: \t\t{len(exact_matches)}, \n"
                f"Buffer ({time_buffer}): \t{len(buffered_matches)}, \n"
                f"Totale: \t\t{len(matches)}")

        return matches, exact_matches, buffered_matches

    def date_matches_with_buffer_and_extra_var(self, df1, df2, extra_vars, time_buffer=pd.Timedelta(days=0), print_info=False):
        """
        Trova match esatti + eventuali match entro buffer considerando RID, EXAMDATE e variabili extra.
        Le date che hanno match esatto vengono *eliminate* dal confronto buffer
        per evitare accoppiamenti falsi.
        
        Args:
            df1: primo dataframe
            df2: secondo dataframe
            extra_vars: lista di nomi delle colonne extra da considerare nel matching (es. ['FSVERSION', 'IMAGEUID'])
            time_buffer: buffer temporale per il matching delle date
            print_info: se True stampa informazioni sui match
            
        Returns:
            matches: dataframe con tutti i match (esatti + buffer)
            exact_matches: dataframe con solo i match esatti
            buffered_matches: dataframe con solo i match entro buffer
        """
        
        # Normalizza extra_vars a lista se è una stringa singola (retrocompatibilità)
        if isinstance(extra_vars, str):
            extra_vars = [extra_vars]
        elif not isinstance(extra_vars, list):
            raise ValueError(f"extra_vars deve essere una stringa o una lista di stringhe, ricevuto: {type(extra_vars)}")
        
        # Verifica che tutte le colonne extra_vars esistano in entrambi i dataframe
        for var in extra_vars:
            if var not in df1.columns:
                raise ValueError(f"Colonna '{var}' non trovata in df1")
            if var not in df2.columns:
                raise ValueError(f"Colonna '{var}' non trovata in df2")
        
        # --- Step 1: Preprocessing ---
        cols_to_select = ['RID', 'EXAMDATE'] + extra_vars
        d1 = df1[cols_to_select].drop_duplicates().copy(deep=True)
        d2 = df2[cols_to_select].drop_duplicates().copy(deep=True)
        
        d1['EXAMDATE'] = pd.to_datetime(d1['EXAMDATE'])
        d2['EXAMDATE'] = pd.to_datetime(d2['EXAMDATE'])
        
        merged = d1.merge(d2, on='RID', suffixes=('_1', '_2'))
        merged['date_diff'] = (merged['EXAMDATE_1'] - merged['EXAMDATE_2']).abs()
        extra_vars1 = [f'{var}_1' for var in extra_vars]
        extra_vars2 = [f'{var}_2' for var in extra_vars]
        
        # --- Step 2: Match esatti (RID, EXAMDATE, tutte le extra_vars) ---
        # Gestione NaN: due NaN sono considerati uguali per ogni variabile extra
        # Tutte le variabili extra devono essere uguali
        extra_vars_match_conditions = []
        for var in extra_vars:
            var_match = (
                (merged[f'{var}_1'] == merged[f'{var}_2']) |
                (merged[f'{var}_1'].isna() & merged[f'{var}_2'].isna())
            )
            extra_vars_match_conditions.append(var_match)
        
        # Combina tutte le condizioni: tutte devono essere True
        if extra_vars_match_conditions:
            extra_vars_match = extra_vars_match_conditions[0]
            for condition in extra_vars_match_conditions[1:]:
                extra_vars_match = extra_vars_match & condition
        else:
            extra_vars_match = pd.Series([True] * len(merged), index=merged.index)
        
        exact_matches = merged[
            (merged['date_diff'] == pd.Timedelta(0)) &
            extra_vars_match
        ]
        
        # Lista (RID, EXAMDATE, tutte le extra_vars) da escludere dal buffer match
        exact_keys_cols = ['RID', 'EXAMDATE_1'] + extra_vars1
        exact_keys = set(
            exact_matches[exact_keys_cols]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        )
        
        # Colonne da restituire nel risultato
        result_cols = ['RID', 'EXAMDATE_1', 'EXAMDATE_2'] + extra_vars1 + extra_vars2
        
        # Se nessun buffer richiesto → ritorno match esatti e basta
        if time_buffer == pd.Timedelta(days=0):
            if print_info:
                print(f"[date_matches_with_buffer_and_extra_var] Match esatti: {len(exact_matches)}")
            if len(exact_matches) > 0:
                matches = exact_matches[result_cols].drop_duplicates()
            else:
                matches = pd.DataFrame(columns=result_cols)
            # Crea un DataFrame vuoto con le stesse colonne per buffered_matches
            buffered_matches = pd.DataFrame(columns=result_cols)
            return matches, exact_matches, buffered_matches
        
        # --- Step 3: Rimuovere *tutte le righe* che coinvolgono una combinazione già matchata esattamente ---
        if exact_keys:
            def check_exact_match(row):
                key1 = tuple([row['RID'], row['EXAMDATE_1']] + [row[var] for var in extra_vars1])
                key2 = tuple([row['RID'], row['EXAMDATE_2']] + [row[var] for var in extra_vars2])
                return key1 in exact_keys or key2 in exact_keys
            
            merged_no_exact = merged[~merged.apply(check_exact_match, axis=1)]
        else:
            merged_no_exact = merged.copy()
        
        # --- Step 4: Match entro buffer (escludendo match esatti) con tutte le extra_vars uguali ---
        # Gestione NaN: due NaN sono considerati uguali per ogni variabile extra
        # Tutte le variabili extra devono essere uguali
        extra_vars_match_buffer_conditions = []
        for var in extra_vars:
            var_match_buffer = (
                (merged_no_exact[f'{var}_1'] == merged_no_exact[f'{var}_2']) |
                (merged_no_exact[f'{var}_1'].isna() & merged_no_exact[f'{var}_2'].isna())
            )
            extra_vars_match_buffer_conditions.append(var_match_buffer)
        
        # Combina tutte le condizioni: tutte devono essere True
        if extra_vars_match_buffer_conditions:
            extra_vars_match_buffer = extra_vars_match_buffer_conditions[0]
            for condition in extra_vars_match_buffer_conditions[1:]:
                extra_vars_match_buffer = extra_vars_match_buffer & condition
        else:
            extra_vars_match_buffer = pd.Series([True] * len(merged_no_exact), index=merged_no_exact.index)
        
        buffered_matches = merged_no_exact[
            (merged_no_exact['date_diff'] <= time_buffer) &
            (merged_no_exact['date_diff'] > pd.Timedelta(0)) &
            extra_vars_match_buffer
        ]
        
        # --- Step 5: Unione finale ---
        matches = pd.concat([exact_matches, buffered_matches], ignore_index=True)
        matches = matches[result_cols].drop_duplicates()
        
        # --- Step 6: Info ---
        if print_info:
            print(f"[date_matches_with_buffer_and_extra_var] Exact: {len(exact_matches)}, "
                f"Buffer ({time_buffer}): {len(buffered_matches)}, "
                f"Totale: {len(matches)}")
        
        return matches, exact_matches, buffered_matches

    def get_indexes_from_matches(self,df1, df2, matches, columns_list=['RID', 'EXAMDATE']):
        """
        Ottiene gli indici delle righe che matchano tra due dataframe.
        """
        # Trova gli indici delle righe originali in df1 e df2
        index_df1_list = []
        index_df2_list = []
        
        if columns_list == ['RID', 'EXAMDATE']:
            for _, match_row in matches.iterrows():
                rid = match_row['RID']
                examdate_1 = match_row['EXAMDATE_1']
                examdate_2 = match_row['EXAMDATE_2']
                
                # Trova gli indici in df1 che corrispondono a questo RID e EXAMDATE
                idxs_df1 = df1[(df1['RID'] == rid) & (df1['EXAMDATE'] == examdate_1)].index.tolist()
                # Trova gli indici in df2 che corrispondono a questo RID e EXAMDATE
                idxs_df2 = df2[(df2['RID'] == rid) & (df2['EXAMDATE'] == examdate_2)].index.tolist()
                
                index_df1_list.extend(idxs_df1)
                index_df2_list.extend(idxs_df2)
            
            # Rimuovi duplicati e ordina
            index_df1 = pd.Index(index_df1_list).drop_duplicates().sort_values()
            index_df2 = pd.Index(index_df2_list).drop_duplicates().sort_values()
            
        else:
            print(f"AVVISO: 'columns_list' non supportato: {columns_list}\nModificare funzione get_indexes_from_matches")
            index_df1 = index_df1_list
            index_df2 = index_df2_list
        
        return index_df1, index_df2

    def compare_matches_with_without_buffer(self, df1, df2, time_buffer, print_info=False):
        """
        Confronta i match tra due dataframe considerando RID e EXAMDATE con buffer temporale e senza buffer temporale.
        Restituisce gli indici delle righe che matchano con buffer ma non hanno date esattamente uguali.
        
        Returns:
            index_df1: indici delle righe in df1 che matchano con buffer ma non hanno data esatta
            index_df2: indici delle righe in df2 che matchano con buffer ma non hanno data esatta
        """
        # Ottieni i match con buffer e senza buffer
        buffered_matches = self.date_matches_with_buffer(df1, df2, time_buffer, print_info=print_info)[2] 
        
        index_df1, index_df2 = self.get_indexes_from_matches(df1, df2, buffered_matches, columns_list=['RID', 'EXAMDATE'])

        if print_info:
            if len(index_df1)!=len(index_df2):
                print('Qualcosa è andato storto --> len(buffer_index1)!=len(buffer_index2)\nBisogna trovare ugual numero di indici')
            elif len(index_df1) > 0:
                print('\n\nCi sono match con TIME BUFFER --> necessario studiare riga per riga\n====> Perchè?\n1) si riferisce ad esami diversi ==> raname uno dei due EXAMDATE \n2) altenativamente ===> scegli quale EXAMDATE usare e sovrascriverlo sull\'altro per quei valori.\n')            
            else:
                print('ZERO matches con TIME BUFFER')
        
        return index_df1, index_df2
        

    def get_date_match_index(self, df1, df2, time_buffer=pd.Timedelta(days=0), print_info=False):
        """
        Ottiene gli indici delle righe che matchano tra due dataframe considerando RID e EXAMDATE con buffer temporale.
        """
        all_matches = self.date_matches_with_buffer(df1, df2, time_buffer, print_info)[0]
        index_df1, index_df2 = self.get_indexes_from_matches(df1, df2, all_matches, columns_list=['RID', 'EXAMDATE'])
        return index_df1, index_df2

    def get_row_different_for_col(self, df1, df2,index1,index2,columns_list):
        col_list = [x for x in columns_list if x in df1.columns and x in df2.columns] #più altre specifiche
        df1_focus = df1.iloc[index1][col_list].copy(deep=True)
        df2_focus = df2.iloc[index2][col_list].copy(deep=True)

        if df1_focus.reset_index(drop=True).equals(df2_focus.reset_index(drop=True)):
            print('I due dataframe hanno TUTTE le righe comuni con lo stesso contenuto nelle colonne di riferimento')
            return None, None, None
        else:
            index_diff1 = []
            index_diff2 = []
            columns_different = []
            index_equal = []
            for i1, i2 in zip(index1, index2):
                if df1_focus.iloc[i1].equals(df2_focus.iloc[i2]):
                    index_equal.append([i1, i2])
                else:
                    index_diff1.append(i1)
                    index_diff2.append(i2)
                    col_diff = []
                    for col in col_list:
                        if df1_focus.iloc[i1][col] != df2_focus.iloc[i2][col]:
                            col_diff.append([col])
                    columns_different.append(col_diff)

            return pd.Index(index_diff1), pd.Index(index_diff2), columns_different


    def create_temp_merge(self, df1, df2, all_index1, all_index2, rid=None, col_list=['VISCODE', 'EXAMDATE']):
        """
        Crea un merge temporaneo dei dataset filtrati per il soggetto RID.
        Include solo colonne RID, VISCODE ed EXAMDATE (se presenti).
        Per righe appaiate (indici in all_index1 e all_index2), mette i valori affiancati con suffissi _1 e _2.
        Per righe uniche, mette NaN nelle colonne dell'altro file.
        
        Args:
            df1: primo dataframe
            df2: secondo dataframe
            all_index1: indici appaiati del primo dataframe (indici originali)
            all_index2: indici appaiati del secondo dataframe (indici originali)
            rid: RID del soggetto da filtrare, se None, non filtra per RID
            col_list: liste di colonne da selezionare
        Returns:
            DataFrame con merge temporaneo
        """
        # Filtra per RID
        if rid is not None:
            df1_filtered = df1[df1['RID'] == rid].copy(deep=True)
            df2_filtered = df2[df2['RID'] == rid].copy(deep=True)
        else:
            df1_filtered = df1.copy(deep=True)
            df2_filtered = df2.copy(deep=True)

        # Seleziona solo colonne RID, VISCODE, EXAMDATE (se presenti)
        cols_to_select1 = []
        for col in col_list:
            if col in df1_filtered.columns:
                cols_to_select1.append(col)
        
        df1_focus = df1_filtered[cols_to_select1].copy(deep=True)
        
        cols_to_select2 = []
        for col in col_list:
            if col in df2_filtered.columns:
                cols_to_select2.append(col)
        
        df2_focus = df2_filtered[cols_to_select2].copy(deep=True)
        
        # Reset indici per facilitare il merge (mantiene mapping con indici originali)
        df1_focus = df1_focus.reset_index(drop=True)
        df2_focus = df2_focus.reset_index(drop=True)
        
        # Crea un mapping degli indici appaiati
        # all_index1 e all_index2 contengono gli indici originali dei dataframe
        # Dobbiamo mapparli agli indici dei dataframe filtrati
        paired_indices_1 = []
        paired_indices_2 = []
        
        # Mappa gli indici originali agli indici filtrati
        original_indices_1 = df1_filtered.index.tolist()
        original_indices_2 = df2_filtered.index.tolist()
        
        # Filtra all_index1 e all_index2 per includere solo quelli del RID corrente
        for orig_idx1, orig_idx2 in zip(all_index1, all_index2):
            # Verifica che gli indici appartengano ai dataframe filtrati per questo RID
            if orig_idx1 in original_indices_1 and orig_idx2 in original_indices_2:
                # Verifica anche che il RID corrisponda (doppio controllo)
                if rid is None or (df1.loc[orig_idx1, 'RID'] == rid and df2.loc[orig_idx2, 'RID'] == rid):
                    # Trova la posizione nell'array filtrato
                    filtered_idx1 = original_indices_1.index(orig_idx1)
                    filtered_idx2 = original_indices_2.index(orig_idx2)
                    paired_indices_1.append(filtered_idx1)
                    paired_indices_2.append(filtered_idx2)
       
        # Crea il merge per le righe appaiate
        paired_rows = []
        for idx1, idx2 in zip(paired_indices_1, paired_indices_2):
            row1 = df1_focus.iloc[idx1].copy()
            row2 = df2_focus.iloc[idx2].copy()
            
            # Crea una riga merged
            #merged_row = {'RID': row1['RID']}
            merged_row = {}

            for col in col_list:
                if col == 'RID':
                    merged_row[col] = row1[col]
                    continue
                elif col in row1.index:
                    merged_row[f'{col}_1'] = row1[col]
                else:
                    merged_row[f'{col}_1'] = None
                if col in row2.index:
                    merged_row[f'{col}_2'] = row2[col]
                else:
                    merged_row[f'{col}_2'] = None       
    
            paired_rows.append(merged_row)
        
        # Trova righe uniche (non appaiate)
        if rid is not None:
            all_filtered_indices_1 = set(range(len(df1_focus)))
            all_filtered_indices_2 = set(range(len(df2_focus)))
            unique_indices_1 = all_filtered_indices_1 - set(paired_indices_1)
            unique_indices_2 = all_filtered_indices_2 - set(paired_indices_2)
            
        
            # Aggiungi righe uniche di df1
            for idx1 in unique_indices_1:
                row1 = df1_focus.iloc[idx1].copy()
                merged_row = {}
                
                for col in col_list:
                    if col == 'RID':
                        merged_row[col] = row1[col]
                        continue
                    elif col in row1.index:
                        merged_row[f'{col}_1'] = row1[col]
                    else:
                        merged_row[f'{col}_1'] = None
                    merged_row[f'{col}_2'] = None
                
                paired_rows.append(merged_row)
            
            # Aggiungi righe uniche di df2
            for idx2 in unique_indices_2:
                row2 = df2_focus.iloc[idx2].copy()
                merged_row = {}
                
                for col in col_list:
                    if col == 'RID':
                        merged_row[col] = row2[col]
                        continue
                    elif col in row2.index:
                        merged_row[f'{col}_2'] = row2[col]
                    else:
                        merged_row[f'{col}_2'] = None
                    merged_row[f'{col}_1'] = None

                paired_rows.append(merged_row)
        
        # Crea il dataframe finale
        temp_merge = pd.DataFrame(paired_rows)
        #temp_merge['IDX_1'] = list(original_indices_1)
        #temp_merge['IDX_2'] = list(original_indices_2)

        return temp_merge



    def get_merged_df(self, df1, df2, category):
        config = self.CATEGORY_CONFIG.get(category, {})
        key_cols = config.get('key_cols', ['RID', 'EXAMDATE'])
        
        print(f'### merging {category}')
        print(f'df1 rows: {len(df1)}')  
        print(f'df2 rows: {len(df2)}')  
        df_merged = pd.merge(df1, df2, how='outer', suffixes=('_1', '_2'))
        df_merged = df_merged.sort_values(by=['RID', 'EXAMDATE']).reset_index(drop=True)
        print(f'-----> Merged df ({len(df_merged)} rows)')
        
        df_clean = self.remove_duplicates_from_merged_df(df_merged, category)
        print(f'-----> Final df ({len(df_clean)} rows)')
        
        return df_clean
    
    def remove_duplicates_from_merged_df(self, df, category):
        config = self.CATEGORY_CONFIG.get(category, {})
        key_cols = config.get('key_cols', ['RID', 'EXAMDATE'])
        colonne_escluse = config.get('colonne_escluse', [])
        
        # Regole di conflitto (uguali per tutte le categorie)
        regole_conflitto = {
            'COHORT': self._risolvi_cohort,
            #'VISCODE': self._risolvi_viscode,
        }
        
        df = df.copy()
        
        # STEP 1: Consolida le righe con lo stesso RID e EXAMDATE(riempi NaN, applica regole) anche non duplicati
        df = self._consolida_duplicati(df, ['RID', 'EXAMDATE'], regole_conflitto, colonne_escluse)
        
        # STEP 2: Scegli quale riga tenere per ogni gruppo di duplicati
        grouped_indices = df.groupby(key_cols).indices
        duplicate_groups = {k: v.tolist() for k, v in grouped_indices.items() if len(v) > 1}
        
        rows_to_drop = []
        for row in duplicate_groups.values():
            if len(row) > 2:
                print(f'WARNING -----> Duplicate group found with more than 2 rows: {row}')
            
            row_to_drop = self._get_row_to_drop(df, row, category)
            rows_to_drop.append(row_to_drop)
        
        print(f'\t\t-----> duplicates removed ({len(rows_to_drop)} rows)')
        df = df.drop(rows_to_drop)
        
        return df
    
    # ==========================================================================
    # METODI PRIVATI - RISOLUZIONE CONFLITTI
    # ==========================================================================
    
    def _risolvi_cohort(self, valori):
        """
        COHORT: unifica sempre al valore più alto della gerarchia.
        Restituisce una lista con lo stesso valore per tutte le righe.
        
        ADNI4 > ADNI3 > ADNI2 > ADNIGO > ADNI1 > qualsiasi altro valore
        """
        valori_validi = [v for v in valori if pd.notna(v)]
        
        if not valori_validi:
            return list(valori)  # Tutti NaN, non modificare
        
        valori_in_gerarchia = [v for v in valori_validi if v in self.COHORT_GERARCHIA]
        
        if valori_in_gerarchia:
            valore_unificato = max(valori_in_gerarchia, key=lambda x: self.COHORT_GERARCHIA.index(x))
        else:
            valore_unificato = valori_validi[0]
        
        return [valore_unificato] * len(valori)
    
    def _risolvi_viscode(self, valori):
        """
        VISCODE: logica differenziata.
        - NaN prende sempre il valore dell'altra riga
        - Valido = 'bl' oppure inizia con 'm'
        - Entrambi validi ma diversi → ognuno tiene il suo
        - Uno valido, uno no → il non valido prende il valido
        - Entrambi non validi → ognuno tiene il suo
        """
        def is_valid(v):
            if pd.isna(v):
                return False
            return v == 'bl' or str(v).startswith('m')
        
        risultato = list(valori)
        
        # Step 1: Riempi i NaN con il primo valore non-NaN
        valori_non_nan = [v for v in valori if pd.notna(v)]
        
        if not valori_non_nan:
            return risultato  # Tutti NaN, niente da fare
        
        fill_value = valori_non_nan[0]
        risultato = [fill_value if pd.isna(v) else v for v in risultato]
        
        # Step 2: Se tutti uguali, nessun conflitto
        if len(set(risultato)) == 1:
            return risultato
        
        # Step 3: Risolvi conflitti tra valori diversi
        validi_mask = [is_valid(v) for v in risultato]
        num_validi = sum(validi_mask)
        
        if num_validi == len(risultato):
            # Tutti validi ma diversi → ognuno tiene il suo
            pass
        elif num_validi > 0:
            # Alcuni validi, alcuni no → i non validi prendono il primo valido
            primo_valido = next(v for v, is_val in zip(risultato, validi_mask) if is_val)
            risultato = [v if is_valid(v) else primo_valido for v in risultato]
        else:
            # Nessun valido → ognuno tiene il suo
            pass
        
        return risultato
    
    # ==========================================================================
    # METODI PRIVATI - HELPER
    # ==========================================================================
    
    def _consolida_duplicati(self, df, key_cols, regole_conflitto=None, colonne_escluse=None):
        """
        Consolida le righe duplicate: riempie i NaN e applica regole speciali.
        
        - Colonne in colonne_escluse: non vengono toccate
        - Colonne in regole_conflitto: la funzione gestisce NaN + conflitti
        - Colonne normali: riempi NaN, conflitti restano invariati
        """
        regole_conflitto = regole_conflitto or {}
        colonne_escluse = set(colonne_escluse or [])
        
        grouped = df.groupby(key_cols)
        
        for keys, indices in grouped.indices.items():
            if len(indices) <= 1:
                continue
            
            idx_list = indices.tolist()
            
            for col in df.columns:
                if col in key_cols or col in colonne_escluse:
                    continue
                
                valori = df.loc[idx_list, col]
                valori_non_nan = valori.dropna()
                
                if col in regole_conflitto:
                    # Colonna con regola speciale
                    nuovi_valori = regole_conflitto[col](valori.tolist())
                    for i, idx in enumerate(idx_list):
                        df.loc[idx, col] = nuovi_valori[i]
                else:
                    # Colonna normale: riempi solo i NaN
                    if len(valori_non_nan) > 0:
                        valore_fill = valori_non_nan.iloc[0]
                        mask_nan = valori.isna()
                        idx_da_riempire = [idx_list[i] for i, is_nan in enumerate(mask_nan) if is_nan]
                        df.loc[idx_da_riempire, col] = valore_fill
        
        return df
    
    def _get_row_to_drop(self, df, row_indices, category):
        """
        Applica la regola specifica per categoria e restituisce l'indice da droppare.
        """
        idx_0, idx_1 = row_indices[0], row_indices[1]
        
        if category == 'volume':
            return self.volume_unification_criteria(df,idx_0, idx_1)
        elif category == 'scale':
            return self.update_stamp_unification_criteria(df,idx_0, idx_1, check_cols=self.CATEGORY_KEYS[category], category=category)
        elif category == 'csf':
            return self.update_stamp_unification_criteria(df,idx_0, idx_1, check_cols=self.CATEGORY_KEYS[category], category=category)
        elif category == 'plasma':
            return self.plasma_unification_criteria(df,idx_0, idx_1, key_cols=self.CATEGORY_KEYS[category])
        elif category == 'pet':
            return self.pet_unification_criteria(df,idx_0, idx_1, key_cols=self.CATEGORY_KEYS[category])
        else:
            return idx_1  # default: droppa la seconda

    def volume_unification_criteria(self, df, idx_1, idx_2):
        """
        Criteri di unificazione per righe di volume quando entrambe le sorgenti hanno valori.
        
        Logica di selezione basata su STATUS e FLDSTRENG:
        
        1. Se entrambi STATUS sono 'complete':
           - Se FLDSTRENG_1 < FLDSTRENG_2 (valore numerico) → scegli row_2 (FLDSTRENG maggiore)
           - Altrimenti → scegli row_1
        
        2. Se row_1 è 'complete' e row_2 è 'partial' o nan:
           → scegli row_1 (priorità a 'complete')
        
        3. Se row_1 è 'partial' o nan e row_2 è 'complete':
           → scegli row_2 (priorità a 'complete')
        
        4. Se entrambi sono 'partial' (o uno è nan):
           - Se FLDSTRENG_1 < FLDSTRENG_2 → scegli row_2 (FLDSTRENG maggiore)
           - Altrimenti → scegli row_1
        
        5. Se row_1 è nan e row_2 è 'partial':
           - Se FLDSTRENG_1 > FLDSTRENG_2 → scegli row_1
           - Altrimenti → scegli row_2
        
        6. Default:
           → scegli row_1
        
        Args:
            row_1: Serie pandas con colonne _1
            row_2: Serie pandas con colonne _2
        
        Returns:
            Serie pandas con la riga selezionata secondo i criteri
        """
        idx_to_drop = idx_2  # Default: scegli row_1
        row_1 = df.loc[idx_1].copy()
        row_2 = df.loc[idx_2].copy()
        
        def extract_fldstreng_value(val):
            """Estrae il valore numerico da FLDSTRENG"""
            if pd.isna(val):
                return float('inf')
            val_str = str(val)
            match = re.search(r'([0-9.]+)', val_str)
            if match:
                return float(match.group(1))
            return float('inf')
        
        if 'STATUS_1' in row_1.index and 'STATUS_2' in row_2.index:
            if row_1['STATUS_1'] == 'complete' and row_2['STATUS_2'] == 'complete':
                if 'FLDSTRENG_1' in row_1.index and 'FLDSTRENG_2' in row_2.index:
                    fld1_val = extract_fldstreng_value(row_1['FLDSTRENG_1'])
                    fld2_val = extract_fldstreng_value(row_2['FLDSTRENG_2'])
                    if fld1_val < fld2_val:
                        idx_to_drop = idx_1
                    else:
                        idx_to_drop = idx_2
            elif row_1['STATUS_1'] == 'complete' and (row_2['STATUS_2'] == 'partial' or pd.isna(row_2['STATUS_2'])):
                idx_to_drop = idx_2
            elif (row_1['STATUS_1'] == 'partial' or pd.isna(row_1['STATUS_1'])) and row_2['STATUS_2'] == 'complete':
                idx_to_drop = row_2
            elif row_1['STATUS_1'] == 'partial' and (row_2['STATUS_2'] == 'partial' or pd.isna(row_2['STATUS_2'])):
                if 'FLDSTRENG_1' in row_1.index and 'FLDSTRENG_2' in row_2.index:
                    fld1_val = extract_fldstreng_value(row_1['FLDSTRENG_1'])
                    fld2_val = extract_fldstreng_value(row_2['FLDSTRENG_2'])
                    if fld1_val < fld2_val:
                        idx_to_drop = idx_1
                    else:
                        idx_to_drop = idx_2
            elif pd.isna(row_1['STATUS_1']) and row_2['STATUS_2'] == 'partial':
                if 'FLDSTRENG_1' in row_1.index and 'FLDSTRENG_2' in row_2.index:
                    fld1_val = extract_fldstreng_value(row_1['FLDSTRENG_1'])
                    fld2_val = extract_fldstreng_value(row_2['FLDSTRENG_2'])
                    if fld1_val > fld2_val:
                        idx_to_drop = idx_2
                    else:
                        idx_to_drop = idx_1
            else:
                idx_to_drop = idx_2

        return idx_to_drop


    def pet_unification_criteria(self, df, idx_1, idx_2, key_cols=None):
        """
        Criteri di unificazione per righe PET quando ci sono duplicati.
        Preferisce la riga con più valori non-nulli nelle colonne PET.
        """
        row_1 = df.loc[idx_1]
        row_2 = df.loc[idx_2]

        # Conta valori non-nulli nelle colonne PET
        pet_cols = [c for c in df.columns if 'SUVR' in c or 'CENTILOID' in c or 'METAROI' in c]

        count_1 = sum(1 for c in pet_cols if c in row_1.index and pd.notna(row_1[c]))
        count_2 = sum(1 for c in pet_cols if c in row_2.index and pd.notna(row_2[c]))

        if count_2 > count_1:
            return idx_1  # droppa idx_1, tieni idx_2
        else:
            return idx_2  # droppa idx_2, tieni idx_1 (default)


    def plasma_unification_criteria(self, df, idx_1, idx_2, key_cols=None):
        """
        Criteri di unificazione per righe plasma quando ci sono duplicati.

        Logica di selezione basata su completezza:
        1. Conta i valori non-NaN nelle colonne biomarker plasma
        2. Scegli la riga con più valori (più completa)
        3. Se pari completezza → scegli row_1 (default)

        Args:
            df: DataFrame con le righe da confrontare
            idx_1: indice prima riga
            idx_2: indice seconda riga
            key_cols: lista colonne chiave (per compatibilità, non usato)

        Returns:
            idx_to_drop: indice della riga da scartare
        """
        idx_to_drop = idx_2  # Default: tieni row_1
        row_1 = df.loc[idx_1]
        row_2 = df.loc[idx_2]

        # Colonne biomarker plasma
        plasma_biomarkers = [
            "AB40_PL", "AB42_PL", "AB4240_PL",
            "PT181_PL", "PT217_PL", "nPT217_PL", "PT217_nPT217_PL",
            "TTAU_PL", "NFL_PL", "GFAP", "ALPHASYN"
        ]

        # Conta valori non-NaN per ogni riga (solo colonne presenti)
        cols_presenti = [c for c in plasma_biomarkers if c in df.columns]

        count_1 = row_1[cols_presenti].notna().sum()
        count_2 = row_2[cols_presenti].notna().sum()

        if count_2 > count_1:
            idx_to_drop = idx_1  # row_2 più completa → scarta row_1
        # Se count_1 >= count_2 → tieni row_1 (default)

        return idx_to_drop


    def _get_previous_row(self, df, idx_1, rid, examdate):
        """
        Trova la riga precedente con stesso RID e EXAMDATE inferiore.
        Prova idx_1-1, se EXAMDATE uguale prova idx_1-2.
        """
        loc = df.index.get_loc(idx_1)
        
        for offset in [1, 2]:
            if loc - offset < 0:
                return None
            
            row_prev = df.iloc[loc - offset]
            
            if row_prev['RID'] != rid:
                return None
            
            if row_prev['EXAMDATE'] < examdate:
                return row_prev
            
            if row_prev['EXAMDATE'] > examdate:
                return None
        
        return None


    def _check_trend(self, val, val_prev, trend):
        """Verifica se val rispetta il trend rispetto a val_prev."""
        if pd.isna(val) or pd.isna(val_prev):
            return False
        
        if trend == 'increasing':
            return val >= val_prev
        elif trend == 'inverse':
            return val <= val_prev
        
        return False


    def _build_check_lists(self, row_1, row_2, row_prev, check_cols, category):
        """
        Costruisce check_1 e check_2.
        True se: valori uguali tra row_1/row_2, oppure rispetta il trend.
        False se: valore NaN, oppure non rispetta il trend.
        """
        check_1 = []
        check_2 = []
        
        for col in check_cols:
            if col not in row_1.index or col not in row_2.index:
                continue
            
            val_1 = row_1[col]
            val_2 = row_2[col]
            
            # Valori uguali (incluso entrambi NaN) → True per entrambi
            both_nan = pd.isna(val_1) and pd.isna(val_2)
            if both_nan or val_1 == val_2:
                check_1.append(True)
                check_2.append(True)
                continue
            
            # Valori diversi → verifica trend
            val_prev = row_prev[col] if row_prev is not None else None
            trend = self.trend_reference[category].get(col, [None, None])[1]
            
            check_1.append(False if pd.isna(val_1) else self._check_trend(val_1, val_prev, trend))
            check_2.append(False if pd.isna(val_2) else self._check_trend(val_2, val_prev, trend))
        
        return check_1, check_2


    def update_stamp_unification_criteria(self, df, idx_1, idx_2, check_cols, category):
        """
        Determina quale riga droppare tra due righe duplicate.
        
        Criteri di decisione (in ordine di priorità):
        ─────────────────────────────────────────────
        1. UPDATE_STAMP (se presente):
        - Tiene la riga con timestamp più recente
        - Se uno mancante → tiene quello valido
        
        2. Se timestamp uguali o entrambi mancanti → CONFRONTO TREND:
        a) Trova riga precedente (stesso RID, EXAMDATE inferiore)
        b) Per ogni colonna in check_cols:
            - Valori uguali tra row_1 e row_2 → True per entrambe
            - Valore NaN → False
            - Altrimenti verifica rispetto del trend (da JSON):
                · 'increasing': val >= val_prev
                · 'inverse': val <= val_prev
        c) Droppa la riga con meno True
        
        3. Default / pareggio / riga precedente non trovata → drop idx_2
        
        Parameters
        ----------
        df : pd.DataFrame
        idx_1, idx_2 : index
            Indici delle due righe duplicate
        check_cols : list
            Colonne da verificare per il confronto trend
        
        Returns
        -------
        idx_to_drop : index
        """
        idx_to_drop = idx_2  # Default
        row_1 = df.loc[idx_1].copy()
        row_2 = df.loc[idx_2].copy()

        # === CRITERIO 1: UPDATE_STAMP ===
        if 'update_stamp' in row_1.index and 'update_stamp' in row_2.index:
            stamp_1 = pd.to_datetime(row_1['update_stamp'])
            stamp_2 = pd.to_datetime(row_2['update_stamp'])
            
            stamp_1_missing = pd.isna(stamp_1)
            stamp_2_missing = pd.isna(stamp_2)
            
            both_missing = stamp_1_missing and stamp_2_missing
            both_equal = (not stamp_1_missing) and (not stamp_2_missing) and (stamp_1 == stamp_2)
            
            if (both_missing or both_equal) and category == 'scale':
                # === CRITERIO 2: CONFRONTO TREND ===
                rid = row_1['RID']
                examdate = row_1['EXAMDATE']
                
                row_prev = self._get_previous_row(df, idx_1, rid, examdate)
                
                if row_prev is not None:
                    check_1, check_2 = self._build_check_lists(row_1, row_2, row_prev, check_cols, category)
                    
                    if sum(check_1) < sum(check_2):
                        idx_to_drop = idx_1
            elif category == 'csf':
                # Conta i NaN nelle colonne check_cols per entrambe le righe
                nan_count_1 = row_1[check_cols].isna().sum()
                nan_count_2 = row_2[check_cols].isna().sum()

                # Droppa la riga con più NaN
                if nan_count_1 > nan_count_2:
                    idx_to_drop = idx_1
                
            elif stamp_1_missing:
                idx_to_drop = idx_1
            elif stamp_2_missing:
                idx_to_drop = idx_2
            elif stamp_2 > stamp_1:
                idx_to_drop = idx_1

        return idx_to_drop


    def audit_conflicts(self, df1, df2, keys=['RID', 'EXAMDATE'], vars_to_check=None):
        """
        Report dettagliato PRIMA di fare merge
        """
        merged = df1.merge(df2, on=keys, suffixes=('_1', '_2'), how='inner')
        
        report = []
        for var in vars_to_check:
            v1, v2 = f'{var}_1', f'{var}_2'
            if v1 in merged.columns and v2 in merged.columns:
                conflicts = merged[(merged[v1] != merged[v2]) & 
                                merged[v1].notna() & 
                                merged[v2].notna()]
                
                if len(conflicts) > 0:
                    report.append({
                        'var': var,
                        'n_conflicts': len(conflicts),
                        'examples': conflicts[['RID', 'EXAMDATE', v1, v2]].head(5).to_dict('records')
                    })
        
        return report


    def get_merged_df_cofactor(self, df1: pd.DataFrame, df2: pd.DataFrame, log_path: str = 'cofactor_merge_log.json') -> pd.DataFrame:
        """
        Merge specifico per cofactor. df1 ha sempre precedenza.
        
        Args:
            df1: DataFrame primario (precedenza)
            df2: DataFrame secondario
            log_path: Path per il log JSON
            
        Returns:
            DataFrame merged con conflitti risolti
        """
        print(f'### Merging cofactor')
        print(f'    df1: {len(df1)} rows, \n    df2: {len(df2)} rows')
        
       # 1. Merge e sort iniziale
        # Auto-detect: se EXAMDATE manca in uno dei df, merge solo su RID
        if 'EXAMDATE' in df1.columns and 'EXAMDATE' in df2.columns:
            merge_keys = ['RID', 'EXAMDATE']
        else:
            merge_keys = ['RID']
            print('Merge only on RID')

        df = pd.merge(df1, df2, on=merge_keys, how='outer', suffixes=('_1', '_2'))

        if 'EXAMDATE' in df.columns:
            df = df.sort_values(['RID', 'EXAMDATE']).reset_index(drop=True)
        else:
            df = df.sort_values(['RID']).reset_index(drop=True)
        print(f'    Merged: {len(df)} rows')
        
        log = {'conflicts': [], 'warnings': []}
        
        # 2. Identifica colonne duplicate
        vars_duplicated = {col[:-2] for col in df.columns if col.endswith('_1')}
        
        # 3. Risolvi gruppi dummy IMMUTABILI (logica a livello RID)
        for group_name, group_cols in self.COFACTOR_CONFIG['dummy_groups_immutable'].items():
            group_in_merge = [c for c in group_cols if c in vars_duplicated]
            if group_in_merge:
                df, group_conflicts = self._resolve_dummy_group_immutable(df, group_name, group_cols)
                if group_conflicts:
                    log['conflicts'].extend(group_conflicts)
                vars_duplicated -= set(group_cols)
        
        # 4. Risolvi gruppi dummy MUTABILI (logica a livello riga)
        for group_name, group_cols in self.COFACTOR_CONFIG['dummy_groups_mutable'].items():
            group_in_merge = [c for c in group_cols if c in vars_duplicated]
            if group_in_merge:
                df, group_conflicts = self._resolve_dummy_group_mutable(df, group_name, group_cols)
                if group_conflicts:
                    log['conflicts'].extend(group_conflicts)
                vars_duplicated -= set(group_cols)
        
        # 5. Risolvi MARRY (logica speciale: sorgente a livello RID + forward-fill)
        marry_cols = self.COFACTOR_CONFIG['dummy_marry']
        marry_in_merge = [c for c in marry_cols if c in vars_duplicated]
        if marry_in_merge:
            df, marry_conflicts, marry_warnings = self._resolve_marry(df, marry_cols)
            if marry_conflicts:
                log['conflicts'].extend(marry_conflicts)
            if marry_warnings:
                log['warnings'].extend(marry_warnings)
            vars_duplicated -= set(marry_cols)
        
        # 6. Risolvi le altre colonne duplicate
        for var in vars_duplicated:
            col_1, col_2 = f'{var}_1', f'{var}_2'
            
            if col_2 not in df.columns:
                df[var] = df[col_1]
                continue
            
            if var in self._resolvers:
                df, var_conflicts = self._resolvers[var](df, var, col_1, col_2)
            elif var in self.COFACTOR_CONFIG['immutable']:
                df, var_conflicts = self._resolve_immutable(df, var, col_1, col_2)
            else:
                df, var_conflicts = self._resolve_default(df, var, col_1, col_2)
            
            if var_conflicts:
                log['conflicts'].extend(var_conflicts)
        
        # 7. Propaga immutabili single_df
        df = self._propagate_all_immutables(df)

        # 8. Forward-fill colonne specifiche (EDUCAT)
        df = self._forward_fill_columns(df)

        # 9. Valida dummy
        df, dummy_warnings = self._validate_dummies(df)
        log['warnings'].extend(dummy_warnings)
        
        # 10. Cleanup
        df = df.drop(columns=[c for c in df.columns if c.endswith(('_1', '_2'))], errors='ignore')
        print(f'    Final: {len(df)} rows')

        # 11. Salva log
        self._save_log(log, log_path)
        
        return df
    
    # ==========================================================================
    # RESOLVER GRUPPI DUMMY
    # ==========================================================================
    
    def _resolve_dummy_group_immutable(self, df: pd.DataFrame, group_name: str, group_cols: List[str]) -> Tuple[pd.DataFrame, List[Dict]]:
        """
        Risolve un gruppo dummy IMMUTABILE a livello RID.
        
        Logica:
        1. df1 ha almeno un True E coerente su tutte le visite → usa df1
        2. df1 non coerente, ma df2 coerente → usa df2
        3. Entrambi inconsistenti → moda df1
        
        Il valore scelto viene propagato a TUTTE le righe del RID.
        """
        conflicts = []
        
        cols_1 = [f'{c}_1' for c in group_cols if f'{c}_1' in df.columns]
        cols_2 = [f'{c}_2' for c in group_cols if f'{c}_2' in df.columns]
        
        if not cols_1 and not cols_2:
            for col in group_cols:
                df[col] = 0
            return df, conflicts
        
        # Inizializza colonne output
        for col in group_cols:
            df[col] = 0
        
        # Processa per RID
        for rid in df['RID'].unique():
            mask_rid = df['RID'] == rid
            df_rid = df.loc[mask_rid]
            
            pattern_1 = self._get_dummy_pattern(df_rid, cols_1, group_cols)
            pattern_2 = self._get_dummy_pattern(df_rid, cols_2, group_cols)
            
            consistent_1, value_1 = self._check_consistency(pattern_1)
            consistent_2, value_2 = self._check_consistency(pattern_2)
            
            final_value = None
            decision = None
            
            if consistent_1 and value_1 is not None:
                final_value = value_1
                decision = 'used_df1'
                
                if consistent_2 and value_2 is not None and value_2 != value_1:
                    conflicts.append({
                        'variable': group_name,
                        'type': 'cross_validation_mismatch',
                        'RID': int(rid),
                        'df1_value': value_1,
                        'df2_value': value_2,
                        'decision': decision
                    })
                    
            elif consistent_2 and value_2 is not None:
                final_value = value_2
                decision = 'used_df2_fallback'
                
                if not consistent_1 and any(v is not None for v in pattern_1):
                    conflicts.append({
                        'variable': group_name,
                        'type': 'df1_inconsistent',
                        'RID': int(rid),
                        'df1_values': list(set(v for v in pattern_1 if v is not None)),
                        'decision': decision
                    })
                    
            elif any(v is not None for v in pattern_1):
                final_value = self._get_mode(pattern_1)
                decision = 'used_df1_mode'
                
                conflicts.append({
                    'variable': group_name,
                    'type': 'both_inconsistent_or_df2_empty',
                    'RID': int(rid),
                    'df1_values': list(set(v for v in pattern_1 if v is not None)),
                    'decision': decision
                })
                
            elif any(v is not None for v in pattern_2):
                final_value = value_2 if consistent_2 else self._get_mode(pattern_2)
                decision = 'used_df2_only_source'
            
            if final_value is not None:
                df.loc[mask_rid, final_value] = 1
        
        return df, conflicts
    
    def _resolve_dummy_group_mutable(self, df: pd.DataFrame, group_name: str, group_cols: List[str]) -> Tuple[pd.DataFrame, List[Dict]]:
        """
        Risolve un gruppo dummy MUTABILE a livello riga.
        
        Logica (per ogni riga):
        - df1 ha almeno un True → usa tutto da df1
        - df1 non ha True ma df2 sì → usa tutto da df2
        - Nessuno ha True → tutto 0
        """
        conflicts = []
        
        cols_1 = [f'{c}_1' for c in group_cols if f'{c}_1' in df.columns]
        cols_2 = [f'{c}_2' for c in group_cols if f'{c}_2' in df.columns]
        
        if not cols_1 and not cols_2:
            for col in group_cols:
                df[col] = 0
            return df, conflicts
        
        if cols_1:
            df1_has_value = df[cols_1].fillna(0).any(axis=1)
        else:
            df1_has_value = pd.Series(False, index=df.index)
            
        if cols_2:
            df2_has_value = df[cols_2].fillna(0).any(axis=1)
        else:
            df2_has_value = pd.Series(False, index=df.index)
        
        use_df1 = df1_has_value
        use_df2 = ~df1_has_value & df2_has_value
        
        # Log conflitti
        both_have = df1_has_value & df2_has_value
        if both_have.any():
            differs = pd.Series(False, index=df.index)
            for col in group_cols:
                col_1, col_2 = f'{col}_1', f'{col}_2'
                if col_1 in df.columns and col_2 in df.columns:
                    col_diff = (df[col_1].fillna(0) != df[col_2].fillna(0)) & both_have
                    differs |= col_diff
            
            if differs.any():
                conflicts.append({
                    'variable': group_name,
                    'type': 'row_mismatch_used_df1',
                    'n_conflicts': int(differs.sum()),
                    'sample_RIDs': df.loc[differs, 'RID'].head(5).tolist()
                })
        
        for col in group_cols:
            col_1, col_2 = f'{col}_1', f'{col}_2'
            df[col] = 0
            
            if col_1 in df.columns:
                df.loc[use_df1, col] = df.loc[use_df1, col_1].fillna(0)
            
            if col_2 in df.columns:
                df.loc[use_df2, col] = df.loc[use_df2, col_2].fillna(0)
        
        return df, conflicts
    
    def _resolve_marry(self, df: pd.DataFrame, marry_cols: List[str]) -> Tuple[pd.DataFrame, List[Dict], List[Dict]]:
        """
        Risolve MARRY con logica speciale:
        - Sorgente scelta a livello RID (df1 ha precedenza)
        - Forward-fill all'interno del RID
        
        Logica per RID:
        - df1 ha almeno un valore MARRY → usa SOLO df1 + forward-fill
        - df1 non ha valori MARRY → usa df2 + forward-fill
        """
        conflicts = []
        warnings = []
        
        cols_1 = [f'{c}_1' for c in marry_cols if f'{c}_1' in df.columns]
        cols_2 = [f'{c}_2' for c in marry_cols if f'{c}_2' in df.columns]
        
        if not cols_1 and not cols_2:
            for col in marry_cols:
                df[col] = 0
            return df, conflicts, warnings
        
        # Inizializza colonne output
        for col in marry_cols:
            df[col] = 0
        
        # Assicura ordinamento per forward-fill
        df = df.sort_values(['RID', 'EXAMDATE']).reset_index(drop=True)
        
        # Determina per ogni RID quale sorgente usare
        rids_use_df1 = set()
        rids_use_df2 = set()
        rids_no_data = set()
        
        for rid in df['RID'].unique():
            mask_rid = df['RID'] == rid
            
            # df1 ha almeno un True per questo RID?
            if cols_1:
                df1_has_any = df.loc[mask_rid, cols_1].fillna(0).any().any()
            else:
                df1_has_any = False
            
            # df2 ha almeno un True per questo RID?
            if cols_2:
                df2_has_any = df.loc[mask_rid, cols_2].fillna(0).any().any()
            else:
                df2_has_any = False
            
            if df1_has_any:
                rids_use_df1.add(rid)
            elif df2_has_any:
                rids_use_df2.add(rid)
            else:
                rids_no_data.add(rid)
        
        # Log conflitti (RID dove entrambi hanno dati ma si usa df1)
        rids_both_have = rids_use_df1 & {rid for rid in df['RID'].unique() 
                                          if cols_2 and df.loc[df['RID'] == rid, cols_2].fillna(0).any().any()}
        if rids_both_have:
            conflicts.append({
                'variable': 'MARRY',
                'type': 'source_precedence_df1',
                'n_rids': len(rids_both_have),
                'sample_RIDs': list(rids_both_have)[:5],
                'note': 'df2 values ignored for these RIDs'
            })
        
        # Copia valori dalla sorgente corretta
        for rid in rids_use_df1:
            mask_rid = df['RID'] == rid
            for col in marry_cols:
                col_1 = f'{col}_1'
                if col_1 in df.columns:
                    df.loc[mask_rid, col] = df.loc[mask_rid, col_1].fillna(0)
        
        for rid in rids_use_df2:
            mask_rid = df['RID'] == rid
            for col in marry_cols:
                col_2 = f'{col}_2'
                if col_2 in df.columns:
                    df.loc[mask_rid, col] = df.loc[mask_rid, col_2].fillna(0)
        
        # Forward-fill per RID
        for col in marry_cols:
            # Converti 0 a NaN per permettere ffill corretto, poi ripristina
            df[col] = df[col].replace(0, np.nan)
            df[col] = df.groupby('RID')[col].ffill()
            df[col] = df[col].fillna(0)
        
        # Warning per RID senza dati MARRY
        if rids_no_data:
            warnings.append({
                'type': 'missing_marry_no_source',
                'n_rids': len(rids_no_data),
                'sample_RIDs': list(rids_no_data)[:5]
            })
        
        # Warning per righe senza MARRY dopo ffill (prime visite senza dato)
        marry_any = df[marry_cols].any(axis=1)
        missing_rows = ~marry_any
        if missing_rows.any():
            rids_with_missing = df.loc[missing_rows, 'RID'].unique()
            # Escludi RID senza dati (già loggati sopra)
            rids_with_partial = [r for r in rids_with_missing if r not in rids_no_data]
            if rids_with_partial:
                warnings.append({
                    'type': 'missing_marry_early_visits',
                    'n_rows': int(missing_rows.sum()),
                    'n_rids': len(rids_with_partial),
                    'sample_RIDs': rids_with_partial[:5],
                    'note': 'Visits before first MARRY value in source df'
                })
        
        return df, conflicts, warnings
    
    # ==========================================================================
    # HELPER PER DUMMY IMMUTABILI
    # ==========================================================================
    
    def _get_dummy_pattern(self, df_rid: pd.DataFrame, source_cols: List[str], group_cols: List[str]) -> List[Any]:
        """
        Estrae il pattern dummy per un RID: quale colonna è True per ogni riga.
        """
        pattern = []
        
        for idx in df_rid.index:
            value = None
            for src_col, orig_col in zip(source_cols, group_cols):
                if src_col in df_rid.columns:
                    cell = df_rid.loc[idx, src_col]
                    if pd.notna(cell) and (cell == 1 or cell == True):
                        value = orig_col
                        break
            pattern.append(value)
        
        return pattern
    
    def _check_consistency(self, pattern: List[Any]) -> Tuple[bool, Any]:
        """Verifica se un pattern è coerente (tutti i valori non-null sono uguali)."""
        non_null = [v for v in pattern if v is not None]
        
        if not non_null:
            return True, None
        
        unique_values = set(non_null)
        
        if len(unique_values) == 1:
            return True, non_null[0]
        else:
            return False, None
    
    def _get_mode(self, pattern: List[Any]) -> Any:
        """Ritorna il valore più frequente (moda) del pattern."""
        non_null = [v for v in pattern if v is not None]
        
        if not non_null:
            return None
        
        from collections import Counter
        counts = Counter(non_null)
        return counts.most_common(1)[0][0]
    
    # ==========================================================================
    # RESOLVER CUSTOM
    # ==========================================================================
    
    def _resolve_cohort(self, df: pd.DataFrame, var: str, col_1: str, col_2: str) -> Tuple[pd.DataFrame, List[Dict]]:
        """
        COHORT: quando entrambi presenti, usa il più alto in gerarchia.
        """
        conflicts = []
        df[var] = df[col_1].combine_first(df[col_2])
        
        mask_both = df[col_1].notna() & df[col_2].notna()
        mask_differ = mask_both & (df[col_1] != df[col_2])
        
        if mask_differ.any():
            hierarchy_map = {v: i for i, v in enumerate(self.COHORT_GERARCHIA)}
            
            v1_rank = df.loc[mask_differ, col_1].map(lambda x: hierarchy_map.get(x, -1))
            v2_rank = df.loc[mask_differ, col_2].map(lambda x: hierarchy_map.get(x, -1))
            
            use_v2 = (v2_rank > v1_rank)
            
            mask_use_v2 = mask_differ.copy()
            mask_use_v2.loc[mask_differ] = use_v2
            df.loc[mask_use_v2, var] = df.loc[mask_use_v2, col_2]
        
        return df, conflicts
    
    def _resolve_age(self, df: pd.DataFrame, var: str, col_1: str, col_2: str) -> Tuple[pd.DataFrame, List[Dict]]:
        """AGE: se differenza <= 0.5 anni → media, altrimenti df1."""
        conflicts = []
        df[var] = df[col_1].combine_first(df[col_2])
        
        mask_both = df[col_1].notna() & df[col_2].notna()
        
        if mask_both.any():
            v1 = df.loc[mask_both, col_1]
            v2 = df.loc[mask_both, col_2]
            diff = (v1 - v2).abs()
            
            avg_mask = mask_both.copy()
            avg_mask.loc[mask_both] = diff <= 0.5
            df.loc[avg_mask, var] = ((df.loc[avg_mask, col_1] + df.loc[avg_mask, col_2]) / 2).round(1)
            
            conflict_mask = mask_both & ~avg_mask
            if conflict_mask.any():
                conflicts.append({
                    'variable': var,
                    'type': 'large_diff_used_df1',
                    'threshold': 0.5,
                    'n_conflicts': int(conflict_mask.sum()),
                    'sample_RIDs': df.loc[conflict_mask, 'RID'].head(5).tolist()
                })
        
        return df, conflicts
    
    # ==========================================================================
    # RESOLVER STANDARD
    # ==========================================================================
    
    def _resolve_immutable(self, df: pd.DataFrame, var: str, col_1: str, col_2: str) -> Tuple[pd.DataFrame, List[Dict]]:
        """Immutabile: valore unico per RID."""
        conflicts = []
        
        def resolve_group(g: pd.DataFrame) -> pd.Series:
            v1 = g[col_1].dropna()
            v2 = g[col_2].dropna()
            
            u1, u2 = v1.unique(), v2.unique()
            cons1, cons2 = len(u1) <= 1, len(u2) <= 1
            
            if cons1 and len(u1) == 1:
                value = u1[0]
            elif cons2 and len(u2) == 1:
                value = u2[0]
            elif len(v1) > 0:
                value = v1.mode().iloc[0]
            elif len(v2) > 0:
                value = v2.mode().iloc[0] if not cons2 else u2[0]
            else:
                value = np.nan
            
            return pd.Series(value, index=g.index)
        
        df[var] = df.groupby('RID', group_keys=False).apply(resolve_group)
        
        rid_stats = df.groupby('RID').agg({
            col_1: lambda x: x.dropna().nunique(),
            col_2: lambda x: x.dropna().nunique()
        }).rename(columns={col_1: 'n_unique_1', col_2: 'n_unique_2'})
        
        inconsistent_rids = rid_stats[(rid_stats['n_unique_1'] > 1) | (rid_stats['n_unique_2'] > 1)]
        if len(inconsistent_rids) > 0:
            conflicts.append({
                'variable': var,
                'type': 'within_rid_inconsistency',
                'n_rids_affected': len(inconsistent_rids),
                'sample_RIDs': inconsistent_rids.index[:5].tolist()
            })
        
        return df, conflicts
    
    def _resolve_default(self, df: pd.DataFrame, var: str, col_1: str, col_2: str) -> Tuple[pd.DataFrame, List[Dict]]:
        """Default: coalesce (df1 prioritario) con logging conflitti."""
        conflicts = []
        df[var] = df[col_1].combine_first(df[col_2])
        
        mask = df[col_1].notna() & df[col_2].notna() & (df[col_1] != df[col_2])
        if mask.any():
            conflicts.append({
                'variable': var,
                'type': 'value_mismatch_used_df1',
                'n_conflicts': int(mask.sum()),
                'sample_RIDs': df.loc[mask, 'RID'].head(5).tolist()
            })
        
        return df, conflicts
    
    # ==========================================================================
    # POST-PROCESSING
    # ==========================================================================
    
    def _forward_fill_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Forward-fill per colonne specifiche (es. EDUCAT)."""
        cols = [c for c in self.COFACTOR_CONFIG.get('forward_fill_cols', []) if c in df.columns]
        
        if cols:
            df = df.sort_values(['RID', 'EXAMDATE'])
            for col in cols:
                df[col] = df.groupby('RID')[col].ffill()
        
        return df


    def _propagate_all_immutables(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Propaga TUTTE le variabili immutabili a tutte le righe del RID.
        Include: scalari immutabili, immutabili single_df, e dummy immutabili.
        
        Idempotente: se già propagate, non cambia nulla.
        """
        # Scalari immutabili
        scalar_immutables = (
            self.COFACTOR_CONFIG['immutable'] + 
            self.COFACTOR_CONFIG['immutable_single_df']
        )
        
        for col in scalar_immutables:
            if col in df.columns:
                df[col] = df.groupby('RID')[col].transform(
                    lambda x: x.dropna().iloc[0] if x.notna().any() else np.nan
                )
        
        # Dummy immutabili
        for group_cols in self.COFACTOR_CONFIG['dummy_groups_immutable'].values():
            for col in group_cols:
                if col in df.columns:
                    df[col] = df.groupby('RID')[col].transform(
                        lambda x: x.dropna().iloc[0] if x.notna().any() else np.nan
                    )
        
        return df
    
    def _validate_dummies(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict]]:
        """Valida che ogni gruppo dummy abbia max 1 True per riga."""
        warnings = []
        
        for group_name, cols in self._all_dummy_groups.items():
            existing = [c for c in cols if c in df.columns]
            if len(existing) < 2:
                continue
            
            true_count = df[existing].sum(axis=1)
            violations = true_count > 1
            
            if violations.any():
                warnings.append({
                    'type': 'dummy_violation',
                    'group': group_name,
                    'n_violations': int(violations.sum()),
                    'fixed': self.fix_dummy_violations,
                    'sample_RIDs': df.loc[violations, 'RID'].unique()[:5].tolist()
                })
                
                if self.fix_dummy_violations:
                    for idx in df.index[violations]:
                        first_true = None
                        for col in existing:
                            if df.loc[idx, col] == 1 or df.loc[idx, col] == True:
                                if first_true is None:
                                    first_true = col
                                else:
                                    df.loc[idx, col] = 0
        
        return df, warnings
    
    # ==========================================================================
    # UTILITIES
    # ==========================================================================
    
    def _save_log(self, log: Dict[str, Any], log_path: str) -> None:
        """Salva log con summary."""
        summary = {
            'n_conflicts': len(log.get('conflicts', [])),
            'n_warnings': len(log.get('warnings', [])),
            'total_issues': sum(
                item.get('n_conflicts', 0) or item.get('n_violations', 0) or item.get('n_rows', 0) or item.get('n_rids', 0) or 0
                for item in log.get('conflicts', []) + log.get('warnings', [])
            )
        }
        
        output = {'summary': summary, 'details': log}
        
        with open(log_path, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        
        print(f'    Log: {summary["n_conflicts"]} conflict types, {summary["n_warnings"]} warning types')
    
    @staticmethod
    def _to_serializable(value: Any) -> Any:
        """Converte valori numpy/pandas in tipi Python nativi per JSON."""
        if pd.isna(value):
            return None
        if isinstance(value, (np.integer, np.floating)):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        return value



################################################# OLD FUNCTIONS #################################################
    def get_merged_col_reference(self, col1, col2, diff_idx, col_name, subject_id):
        '''
        Funzione per ottenere la colonna matchata per le colonne di riferimento.
        Focus su un solo soggetto che ha righe incomune tra due dataframe.
        Args:
            col1: colonna del primo dataframe
            col2: colonna del secondo dataframe
            diff_idx: indici delle righe che differiscono
            col_name: nome della colonna
            subject_id: id del soggetto
        Returns:
            colonna matchata
        '''

        if col_name == 'EXAMDATE':
            print(f'{subject_id} \n### EXAMDATE_1 != EXAMDATE_2 ---> should be handled before')
            return col1.copy().fillna(col2)

        elif col_name in ['VISCODE', 'VISIT_MONTH']:
            # visit code andrà poi eliminato mentre visit_month verrà normalizzato/ricalcolato alla fine del merge
            return col1.copy().fillna(col2)

        elif col_name == 'COHORT':
            merged_col = col1.copy(deep=True)
            for x in diff_idx:
                merged_col.loc[x] = self.select_cohort(col1.loc[x], col2.loc[x], new_cohort=True)
            return merged_col
        elif col_name == 'STATUS':
            def choose(v1, v2):
                # Regola 1: se uno è "complete", vince "complete"
                if v1 == "complete" or v2 == "complete":
                    return "complete"
                # Regola 2: se uno è "partial", vince "partial"
                if v1 == "partial" or v2 == "partial":
                    return "partial"
                # Regola 3: altrimenti prendi il primo non-None / non-NaN
                return v1 if v1 not in (None, float('nan')) else v2

            # combine applica la funzione solo agli elementi che differiscono o contengono NaN
            merged_col = col1.combine(col2, choose)
            return merged_col
        elif col_name == 'METHOD':
            print(f'{subject_id} \n### METHOD_1 != METHOD_2 ---> should be handled before')
            return col1.copy().fillna(col2)
        else:
            print(f'### {col_name} is not a reference column studied, it will be merged with the simple method')
            return col1.copy().fillna(col2)


    def check_trend_and_range(self, col1, col2, mean_col, trend, min_val, max_val):
        '''
        Funzione per verificare se il trend e il range sono corretti.
        '''
        if trend == 'increasing':
            in_trend = col1.dropna().is_monotonic_increasing and col2.dropna().is_monotonic_increasing and mean_col.dropna().is_monotonic_increasing
        elif trend == 'inverse':
            in_trend = col1.dropna().is_monotonic_decreasing and col2.dropna().is_monotonic_decreasing and mean_col.dropna().is_monotonic_decreasing
        else:
            raise ValueError(f'### {trend} is not a valid trend')
        
        in_range = min_val <= col1.min() and col1.max() <= max_val and min_val <= col2.min() and col2.max() <= max_val and min_val <= mean_col.min() and mean_col.max() <= max_val
       
        return in_trend, in_range


    def get_merged_float_range_and_trend(self, col1, col2, mean_col, merged_col, in_trend, trend, min_val, max_val, idx, col_name):
        '''
        Funzione per ottenere il valore mergeato per una singola riga.
        '''
        # FUNZIONE INTERNA per verificare se il valore rispetta il trend rispetto al precedente valore e ai possibili valori successivi.
        def value_respects_trend(value:float, trend:str, prev_value:float=-100, next_values:list=[]):
            trend_check = []
            if trend == 'increasing':
                if value >= prev_value or pd.isna(prev_value):
                    trend_check.append(True)
                    if next_values:
                        for next_value in next_values:
                            if value <= next_value:
                                trend_check.append(True)
                            else:
                                trend_check.append(False)
                    return trend_check
                else:
                    trend_check.append(False)
                    return trend_check
            elif trend == 'inverse':
                if value <= prev_value or pd.isna(prev_value):
                    trend_check.append(True)
                    if next_values:
                        for next_value in next_values:
                            if value >= next_value:
                                trend_check.append(True)
                            else:
                                trend_check.append(False)
                    return trend_check
                else:
                    trend_check.append(False)
                    return trend_check
            else:
                raise ValueError(f'### {trend} is not a valid trend')
            return trend_check

        def get_closest_value(value1, value2, mean_val, prev_value, next_values):
            '''
            Funzione per ottenere il valore più vicino tra due valori.
            '''
            if not pd.isna(prev_value) and next_values:
                diff1 = abs(value1 - prev_value) + min(abs(value1 - next_values[0]), abs(value1 - next_values[1]), abs(value1 - next_values[2]))
                diff2 = abs(value2 - prev_value) + min(abs(value2 - next_values[0]), abs(value2 - next_values[1]), abs(value2 - next_values[2]))
                diff_mean = abs(mean_val - prev_value) + min(abs(mean_val - next_values[0]), abs(mean_val - next_values[1]), abs(mean_val - next_values[2]))
                closest = value1 if diff1 <= diff2 and diff1 <= diff_mean else value2 if diff2 <= diff1 and diff2 <= diff_mean else mean_val
                return closest
            elif not pd.isna(prev_value):
                diff1 = abs(value1 - prev_value)
                diff2 = abs(value2 - prev_value)
                diff_mean = abs(mean_val - prev_value)
                closest = value1 if diff1 <= diff2 and diff1 <= diff_mean else value2 if diff2 <= diff1 and diff2 <= diff_mean else mean_val
                return closest
            elif next_values:
                diff1 = min(abs(value1 - next_values[0]), abs(value1 - next_values[1]), abs(value1 - next_values[2]))
                diff2 = min(abs(value2 - next_values[0]), abs(value2 - next_values[1]), abs(value2 - next_values[2]))
                diff_mean = min(abs(mean_val - next_values[0]), abs(mean_val - next_values[1]), abs(mean_val - next_values[2]))
                closest = value1 if diff1 <= diff2 and diff1 <= diff_mean else value2 if diff2 <= diff1 and diff2 <= diff_mean else mean_val
                return closest
            else:
                return mean_val

        # VERIFICA RANGE
        
        in_range_1 = min_val <= col1.loc[idx] <= max_val if not pd.isna(col1.loc[idx]) else False
        in_range_2 = min_val <= col2.loc[idx] <= max_val if not pd.isna(col2.loc[idx]) else False
        # valutazioni basate sul range
        if in_range_1 == in_range_2 and not pd.isna(col1.loc[idx]) and not pd.isna(col2.loc[idx]):
            # se i valori sono entrambi dentro o fuori dal range allora fa la media tra i due valori
            # e poi scelgo tra i 2 valori e la loro media quale segue meglio il trend degli altri valori
            mean_val = mean_col.loc[idx]
            #if abs(col1.loc[idx]-col2.loc[idx]) > 0.1*mean_val:
            #    print(f'ATTENZIONE: {idx} --> differnza MAGGIORE del 10% della media')
            
            if in_trend:
                # se il trend è corretto allora prende la media
                selected_value = mean_val
                #print(f'{idx} --> scelta in trend e range')
            else:
                # altrimenti scelgo il valore che segue il trend sia rispetto al precedente valore già fissato in merged_col che rispetto ai possibili valori successivi (col1, col2, mean_col)
                # imposto valori precedenti e successivi basati sull'indice corrente
                prev_value = merged_col.loc[idx-1] if idx > 0 else np.nan
                next_values = [col1.loc[idx+1], col2.loc[idx+1], mean_col.loc[idx+1]] if idx < len(col1) - 1 else []
                # verifico se i valori rispettano il trend rispetto al precedente valore e ai possibili valori successivi
                trend_check1 = value_respects_trend(col1.loc[idx], trend, prev_value=prev_value, next_values=next_values)
                trend_check2 = value_respects_trend(col2.loc[idx], trend, prev_value=prev_value, next_values=next_values)
                trend_check_mean = value_respects_trend(mean_val, trend, prev_value=prev_value, next_values=next_values)
                # conta quanti True ci sono in trend_check1, trend_check2 e trend_check_mean
                tot_ok1 = sum(trend_check1)
                tot_ok2 = sum(trend_check2)
                tot_ok_mean = sum(trend_check_mean)

                # prendo la media se:
                # 1) tutti i valori della trend_check_mean sono True
                # 2) se il primo valore della trend_check_mean è True (rispetta il trend con il precedente valore già fissato in merged_col) 
                #    e la somma dei True della trend_check_mean è maggiore o uguale alla somma dei True della trend_check1 e della trend_check2
                if all(trend_check_mean) or (trend_check_mean[0] and tot_ok_mean >= tot_ok1 and tot_ok_mean >= tot_ok2 and tot_ok_mean > 1):
                    selected_value = mean_val
                    #print(f'{idx} --> analisi trend --> MEDIA')
                # prendo il valore di col1 se:
                # 1) tutti i valori della trend_check1 sono True
                # 2) se il primo valore della trend_check1 è True (rispetta il trend con il precedente valore già fissato in merged_col) 
                #    e la somma dei True della trend_check1 è maggiore o uguale alla somma dei True della trend_check2 e della trend_check_mean
                elif all(trend_check1) or (trend_check1[0] and tot_ok1 >= tot_ok2 and tot_ok1 >= tot_ok_mean and tot_ok1 > 1):
                    selected_value = col1.loc[idx]
                    #print(f'{idx} --> analisi trend --> COL1')
                # prendo il valore di col2 se:
                # 1) tutti i valori della trend_check2 sono True
                # 2) se il primo valore della trend_check2 è True (rispetta il trend con il precedente valore già fissato in merged_col) 
                #    e la somma dei True della trend_check2 è maggiore o uguale alla somma dei True della trend_check1 e della trend_check_mean
                elif all(trend_check2) or (trend_check2[0] and tot_ok2 >= tot_ok1 and tot_ok2 >= tot_ok_mean and tot_ok2 > 1):
                    selected_value = col2.loc[idx]
                    #print(f'{idx} --> analisi trend --> COL2')
                else:
                    # altrimenti non prende nessuno dei due valori --> verifico quale è il problema e valuto come comportarmi
                    closest = get_closest_value(col1.loc[idx], col2.loc[idx], mean_val, prev_value, next_values)
                    #print(f'{idx} --> analisi trend --> CLOSEST VALUE')
                    selected_value = closest
        
        elif (pd.isna(col1.loc[idx]) or not in_range_1) and in_range_2:
            selected_value = col2.loc[idx]
        elif (pd.isna(col2.loc[idx]) or not in_range_2) and in_range_1:
            selected_value = col1.loc[idx]
        else:
            selected_value = np.nan

        return selected_value

    def get_merged_columns_float(self, col1, col2, diff_idx, col_name, param_type, param_ref, method=None, status1=None, status2=None, rid=None):
        '''
        Funzione per ottenere la colonna matchata per le colonne di in comune tra due dataframe per VOLUMI e BIOMARCATORI.
        Focus su un solo soggetto che ha righe incomune tra due dataframe.
        Args:
            col1: colonna del primo dataframe
            col2: colonna del secondo dataframe
            diff_idx: indici delle righe che differiscono
            col_name: nome della colonna
            param_ref: dizionario dei valori di riferimento per i parametri
            method: metodo di merge
            status1: status del primo dataframe
            status2: status del secondo dataframe
            rid: RID del soggetto
        Returns:
            colonna matchata
        '''

        #print('\n',col_name, '-------------------------------------------------------------')
        param_list = ['volume', 'biomark']
        if param_type not in param_list:
            raise ValueError(f'### {param_type} is not a valid parameter type, it must be one of {param_list}')
        
        ### Valori di RIFERIMENTO
        if param_type == 'volume':
            min_val = param_ref[col_name][0]
            max_val = param_ref[col_name][1]
            trend = param_ref[col_name][2]
        elif param_type == 'biomark':
            if method is None:
                raise ValueError(f'### {col_name} is a biomark, but method is not provided, call the function with the method')
            min_val = param_ref[col_name][method][0]
            max_val = param_ref[col_name][method][1]
            trend = param_ref[col_name][method][2]
        
        # ottengo la colonna media, trattando adeguatamente i nan
        mean_col = col1.combine(col2, lambda x, y: (x + y) / 2 if not pd.isna(x) and not pd.isna(y) else x if pd.isna(x) else y, fill_value=np.nan)
        # verifico se la differenza tra i due valori è maggiore del 10% della media
        diff_col = col1.combine(col2, lambda x, y: abs(x - y) if not pd.isna(x) and not pd.isna(y) else np.nan, fill_value=np.nan)
        diff_in_col = np.nanmax([(col1.shift(0) - col1.shift(-1)).abs().iloc[:-1].max(), (col2.shift(0) - col2.shift(-1)).abs().iloc[:-1].max()])
        diff_check_mean = (diff_col > 0.2*(mean_col.median())).fillna(False)
        diff_check_in_col = (diff_col > diff_in_col).fillna(False)
        
        if not all(~diff_check_mean) and rid is not None:           
            print(f'------> ATTENZIONE: {col_name} diff >>> 20% media tra visite ({round(0.2*(mean_col.median()), 4)})')
            # Aggiorna il file JSON se rid è disponibile
            json_file_path = 'diff_mean_tracking.json'
            # Carica il file JSON se esiste, altrimenti crea un dizionario vuoto
            if os.path.exists(json_file_path):
                with open(json_file_path, 'r', encoding='utf-8') as f:
                    diff_dict = json.load(f)
            else:
                diff_dict = {}
            # Converte diff_check in lista di booleani
            diff_check_list_mean = diff_check_mean.tolist()
            
            # Converte diff_col.values in lista e converte NaN in None per JSON
            diff_col_values = list(diff_col.values)
            diff_col_values_clean = self._convert_nan_to_none(diff_col_values)
            
            # Se col_name è già nelle chiavi, appende alla lista esistente
            if col_name in diff_dict:
                diff_dict[col_name].append([rid, 0.2*(mean_col.median()), diff_check_list_mean, diff_col_values_clean])
            else:
                # Altrimenti aggiunge la nuova chiave con una lista contenente [rid, lista_booleani]
                diff_dict[col_name] = [[rid, 0.2*(mean_col.median()), diff_check_list_mean, diff_col_values_clean]]
            
            # Converte NaN in None prima di salvare
            diff_dict_clean = self._convert_nan_to_none(diff_dict)
            
            # Salva il dizionario aggiornato nel file JSON
            with open(json_file_path, 'w', encoding='utf-8') as f:
                json.dump(diff_dict_clean, f, indent=4, ensure_ascii=False)
        
        if not all(~diff_check_in_col) and rid is not None:           
            print(f'------> ATTENZIONE: {col_name} diff >>> diff in col tra visite ({round(diff_in_col, 4)})')
            # Aggiorna il file JSON se rid è disponibile
            json_file_path = 'diff_in_col_tracking.json'
            # Carica il file JSON se esiste, altrimenti crea un dizionario vuoto
            if os.path.exists(json_file_path):
                with open(json_file_path, 'r', encoding='utf-8') as f:
                    diff_dict = json.load(f)
            else:
                diff_dict = {}
            # Converte diff_check in lista di booleani
            diff_check_list_in_col = diff_check_in_col.tolist()
            
            # Converte diff_col.values in lista e converte NaN in None per JSON
            diff_col_values = list(diff_col.values)
            diff_col_values_clean = self._convert_nan_to_none(diff_col_values)
            
            # Se col_name è già nelle chiavi, appende alla lista esistente
            if col_name in diff_dict:
                diff_dict[col_name].append([rid, diff_in_col, diff_check_list_in_col, diff_col_values_clean])
            else:
                # Altrimenti aggiunge la nuova chiave con una lista contenente [rid, lista_booleani]
                diff_dict[col_name] = [[rid, diff_in_col, diff_check_list_in_col, diff_col_values_clean]]
            
            # Converte NaN in None prima di salvare
            diff_dict_clean = self._convert_nan_to_none(diff_dict)
            
            # Salva il dizionario aggiornato nel file JSON
            with open(json_file_path, 'w', encoding='utf-8') as f:
                json.dump(diff_dict_clean, f, indent=4, ensure_ascii=False)

        # verifico se tutte le colonne sono dentro il range e seguono il trend
        in_trend, in_range = self.check_trend_and_range(col1, col2, mean_col, trend, min_val, max_val)

        ### inizializzazione colonna MERGED
        if in_trend and in_range:
            #print('### In trend and in range --> EASY')
            # se tutte le colonne sono dentro il range e seguono il trend allora prende la media e lo salva nella colonna MERGED
            merged_col = mean_col.copy()
            return merged_col
        else:
            # altrimenti inizzializzo con colonna 1 e  i valori della 2 al posto dei nan
            merged_col = col1.copy().fillna(col2)

        ### quindi sistemo il MERGE per le righe che differiscono
        for idx in diff_idx:
            # Verifico uno dei due valori non sia nan e lo salvo nella colonna merged
            if pd.isna(col1.loc[idx]) and not pd.isna(col2.loc[idx]):
                merged_col.loc[idx] = col2.loc[idx]
                #print(idx, '--> uno dei due valori è NAN, prendo quello non NAN')
                continue
            elif not pd.isna(col1.loc[idx]) and pd.isna(col2.loc[idx]):
                merged_col.loc[idx] = col1.loc[idx]
                #print(idx, '--> uno dei due valori è NAN, prendo quello non NAN')
                continue

            # CRITERIO a PRIORI (se presente)
            if param_type == 'volume':
                stat1 = status1.loc[idx] if status1 is not None else None
                stat2 = status2.loc[idx] if status2 is not None else None
                # se i due status sono diversi allora prende quello che ha lo status 'complete'
                if stat1 is not None or stat2 is not None and stat1 != stat2:
                    if (stat1 == 'complete' and stat2 != 'complete') or (stat1 == 'partial' and stat2 is None):
                        merged_col.loc[idx] = col1.loc[idx]
                        #print(idx, f'--> scelta per STATUS1 (1){stat1} vs (2){stat2}')
                    elif (stat2 == 'complete' and stat1 != 'complete') or (stat2 == 'partial' and stat1 is None):
                        merged_col.loc[idx] = col2.loc[idx]
                        #print(idx, f'--> scelta per STATUS2 (1){stat1} vs (2){stat2}')
                    else:
                        merged_col.loc[idx] = np.nan
                    continue
            elif param_type == 'biomark':
                pass

            # se i dati non sono nan e fallisce il criterio a PRIORI, procedo con il merge pasato su trend e range
            merged_col.loc[idx] = self.get_merged_float_range_and_trend(col1, col2, mean_col, merged_col, in_trend, trend, min_val, max_val, idx, col_name)

        return merged_col




    def merge_paired_rows_rid_specific(self, df1, df2, index1, index2, ref_col, subject_id):
        ### RIFERIMENTI
        # keys dei VOLUMI, rifereimenti valori e andamento patologico (trend)
        volume_ref = self.get_json_file('volume_values_settings.json')
        volume_keys = list(volume_ref.keys())+['IMAGEUID', 'FSVERSION', 'FLDSTRENG', 'STATUS']
        
        # keys delle SCALE e BIOMARCATORI, rifereimenti valori e andamento patologico (trend) 
        norm_ref = self.get_json_file('normalization_settings.json')
        scale_keys = ['MMSE', 'RAVLT_immediate', 'FAQ', 'MOCA', 'CRSB', 'CRSGLOB', 'ADAS11', 'ADAS13']
        scale_ref = {k: norm_ref[k] for k in scale_keys if k in norm_ref}
        biomark_ref = {k: v for k, v in norm_ref.items() if k not in scale_keys}        
        
        # keys dei COFATTORI, rifereimenti valori e andamento patologico (trend)
        cofattori_ref = self.get_json_file('cofattori_values_settings.json')

        ### CREAZIONE TEMPORANEO MERGE 
        # specifico per soggetto
        #col_list = [x for x in df1.columns if x in df2.columns and x != 'RID']
        common_cols = list(df1.columns.intersection(df2.columns))
        col_list = list(df1.columns) + [c for c in df2.columns if c not in df1.columns]
        df_compare = self.create_temp_merge(df1, df2, index1, index2, rid=subject_id, col_list=col_list)
        ### CREAZIONE MERGE per il soggetto con righe accoppiate
        base_cols = [c for c in ['RID', 'DX_1', 'DX_2'] if c in df_compare.columns]
        df_merge = df_compare[base_cols].copy()
        
        col_exact_match = []
        col_equal_with_Nan = []
        col_diff = []
        row_diff = {}

        
        col_to_merge = [c for c in col_list if c not in base_cols]
        #print(f'col_to_merge {col_to_merge}')
        for col in col_to_merge:
            col1 = col + '_1'
            col2 = col + '_2'
            # se la colonna non è in comune tra i due df, prendo la colonna tra le due che non è tutta nan
            if col not in common_cols:
                temp_df = df_compare[[col1, col2]].dropna(axis=1, how="all")
                if temp_df.shape[1] > 0:
                    valid_col = temp_df.iloc[:, 0]
                    df_merge[col] = valid_col
                else:
                    # Se entrambe le colonne sono completamente NaN, crea una colonna di NaN
                    df_merge[col] = pd.Series([None] * len(df_merge), index=df_merge.index)
                continue

            if df_compare[col1].equals(df_compare[col2]):
                col_exact_match.append(col)
                df_merge[col] = df_compare[col1]
                continue
            else:
                # Crea un DataFrame temporaneo con le due colonne e rimuove le righe dove almeno una ha NaN
                temp_df = df_compare[[col1, col2]].dropna()
                if temp_df[col1].equals(temp_df[col2]):
                    col_equal_with_Nan.append(col)
                    df_merge[col] = df_compare[col1].copy().fillna(df_compare[col2])
                    continue
            
            # Se arriviamo qui, le colonne differiscono
            col_diff.append(col)
            diff_mask = ~df_compare[col1].eq(df_compare[col2])
            diff_idx = df_compare.index[diff_mask].tolist()
            if diff_idx:
                row_diff[col] = diff_idx
            
            ### COLONNE DI RIFERIMENTO
            if col in ref_col+['STATUS', 'METHOD']:
                df_merge[col] = self.get_merged_col_reference(df_compare[col1], df_compare[col2], diff_idx, col_name=col, subject_id=subject_id)
            ### VOLUMI
            elif col in volume_ref.keys():
                if 'STATUS' in col_list:
                    status1 = df_compare['STATUS_1'] if 'STATUS' in df1.columns else None
                    status2 = df_compare['STATUS_2'] if 'STATUS' in df2.columns else None
                    df_merge[col] = self.get_merged_columns_float( df_compare[col1], df_compare[col2], diff_idx, col_name=col, param_type='volume', param_ref=volume_ref, status1=status1, status2=status2, rid=subject_id)
                else:
                    df_merge[col] = self.get_merged_columns_float( df_compare[col1], df_compare[col2], diff_idx, col_name=col, param_type='volume', param_ref=volume_ref, rid=subject_id)
            
            ### SCALE
            elif col in scale_ref.keys():
                df_merge[col] = self.get_merged_col_scale(df_compare[col1], df_compare[col2], diff_idx, col_name=col, cofattori_ref=cofattori_ref)
            
            ### BIOMARCATORI
            elif col in biomark_ref.keys():
                if 'METHOD' in col_list and df_compare['METHOD_1'].unique().tolist() == df_compare['METHOD_2'].unique().tolist() and len(df_compare['METHOD_1'].unique()) == 1:
                    method = df_compare['METHOD_1']
                    df_merge[col] = self.get_merged_columns_float(df_compare[col1], df_compare[col2], diff_idx, col_name=col, param_type='biomark', param_ref=scale_ref, method=method, rid=subject_id)
                else:
                    raise ValueError(f'### {col} is a biomark, but there are problems with the METHOD column, it is not among the column list or is not unique for all the rows or equal among dfs')
            
            ### COFATTORI
            elif col in cofattori_ref.keys():
                df_merge[col] = self.get_merged_col_cofattori(df_compare[col1], df_compare[col2], diff_idx, col_name=col, cofattori_ref=cofattori_ref)
            
            ### ALTRIMENTI
            else:
                print(f'### {col} is not a reference column studied, it will be merged with the simple method')
                df_merge[col] = df_compare[col1].copy().fillna(df_compare[col2])
            continue
        return df_merge