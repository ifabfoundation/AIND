"""
Automated Merge Pipeline
=========================
Genera automaticamente tutti i merge finali per ogni combinazione di filtri.

Uso:
    python automated_merge.py

Output:
    ./merged_outputs/merge_FSVERSION-X_method_CSF-Y_....csv
"""

from dl_client import DatalakeClient
from data_model.MergerTools import MergerTools
import pandas as pd
import numpy as np
import json
from itertools import product
from pathlib import Path
from datetime import datetime


# =============================================================================
# CONFIGURAZIONE
# =============================================================================

# Finestra temporale per buffer matches
BUFFER_DAYS = 80

# Colonne chiave
RID_COL = 'RID'
DATE_COL = 'EXAMDATE'

# Mapping dataset -> colonna filtro
# Le chiavi devono matchare i nomi dei file (es. VOLUMES_merged.csv → 'VOLUMES')
FILTER_COLUMNS = {
    'VOLUMES': 'FSVERSION',
    'SCALE': None,
    'CSF': 'METHOD_CSF',
    'PLASMA': 'METHOD_PLASMA',
    'PET': 'METHOD_PET',
    'COFACTOR': None
}

# Ordine di merge incrementale (usa le stesse chiavi di FILTER_COLUMNS)
MERGE_ORDER = ['VOLUMES', 'SCALE', 'CSF', 'PLASMA', 'PET', 'COFACTOR']

# =============================================================================
# FILTRO VALORI - Specifica quali valori includere per ogni filtro
# =============================================================================
# - None o chiave assente = usa TUTTI i valori disponibili nel dataset
# - Lista vuota [] = SALTA questo filtro (nessuna combinazione)
# - Lista di valori = usa SOLO questi valori
#
# Esempio:
#   FILTER_VALUES_WHITELIST = {
#       'FSVERSION': ['6.0', '7.0'],      # solo queste versioni
#       'METHOD_CSF': ['UPLC'],           # solo UPLC
#       'METHOD_PLASMA': None,            # tutti i valori disponibili
#       # 'METHOD_PET' non specificato    # tutti i valori disponibili
#   }
# =============================================================================
FILTER_VALUES_WHITELIST = {
    'FSVERSION': None,      # None = tutti i valori
    'METHOD_CSF': ['elecsys', 'lumipulse'],
    'METHOD_PLASMA': ['lumipulse'],
    'METHOD_PET': ['FBB','FBP'],
}

# Colonne volumetriche per valutare qualità
VOLUME_COLS = [
    'Ventricles%ICV', 'Hippocampus%ICV', 'Entorhinal%ICV',
    'Fusiform%ICV', 'MidTemp%ICV', 'ICV%ICV'
]

# Prefix per upload sul datalake
DATALAKE_PREFIX = 'cleaned/merged/combination'

# File locale per statistiche
STATS_OUTPUT_FILE = Path('./merge_statistics.csv')

# =============================================================================
# CONFIGURAZIONE METADATA ARRICCHITI
# =============================================================================

# Path ai file di settings per metadata
VOLUME_SETTINGS_FILE = Path('./volume_values_settings.json')
NORMALIZATION_SETTINGS_FILE = Path('./normalization_settings.json')
COFATTORI_SETTINGS_FILE = Path('./cofattori_values_settings.json')

# Definizione predittori per categoria
PREDITTORI_COLUMNS = {
    'volumes': ['Brain%ICV', 'Ventricles%ICV', 'Hippocampus%ICV',
                'Entorhinal%ICV', 'Fusiform%ICV', 'MidTemp%ICV'],
    'scale': ['MMSE', 'RAVLT_immediate', 'FAQ', 'MOCA', 'CDRSB', 'CDRGLOB', 'ADAS11', 'ADAS13'],
    'csf': ['CSF_DATE', 'METHOD_CSF', 'AB40_CSF', 'AB42_CSF', 'AB4240_CSF',
            'PT181_CSF', 'TTAU_CSF', 'PT181_AB42_CSF'],
    'plasma': ['AB40_PL', 'AB42_PL', 'AB4240_PL', 'PT181_PL', 'PT217_AB42_PL',
               'TTAU_PL', 'nPT217_PL', 'PT217_nPT217_PL', 'ALPHASYN', 'NFL_PL', 'GFAP'],
    'pet': ['AMY_CENTILOIDS', 'SUMMARY_SUVR', 'PRECUNEUS_SUVR', 'TAU_METAROI',
            'CTX_INFERIORPARIETAL_SUVR', 'CTX_PARAHIPPOCAMPAL_SUVR', 'CTX_LATERALOCCIPITAL_SUVR',
            'CTX_MIDDLETEMPORAL_SUVR', 'INFERIOR_TEMPORAL_SUVR', 'CTX_ENTORHINAL_SUVR',
            'CTX_FUSIFORM_SUVR', 'CSF_SUVR']
}

# Definizione cofattori
COFATTORI_COLUMNS = [
    'GENDER/female', 'GENDER/male', 'DX/CN', 'DX/Dementia', 'DX/MCI', 'EDUCAT',
    'MARRY/divorced', 'MARRY/married', 'ETHNICITY/latino', 'ETHNICITY/not_latino',
    'MARRY/single', 'MARRY/widowed', 'RACE/Asian', 'RACE/Black', 'RACE/Mixed',
    'RACE/Native_american', 'RACE/White', 'APOE', 'APOE_4', 'DIAN_MUTATION',
    'Aprofile', 'Tprofile', 'Nprofile'
]

# Mapping colonna -> filtro per selezione metodo in normalization_settings
COLUMN_METHOD_MAPPING = {
    '_CSF': 'METHOD_CSF',       # colonne che contengono _CSF usano METHOD_CSF
    '_PL': 'METHOD_PLASMA',     # colonne che contengono _PL usano METHOD_PLASMA
    '_SUVR': 'METHOD_PET',      # colonne SUVR usano METHOD_PET
    '_CENTILOIDS': 'METHOD_PET', # AMY_CENTILOIDS usa METHOD_PET
}


# =============================================================================
# ECCEZIONI CUSTOM
# =============================================================================

class MergeError(Exception):
    """Errore durante il merge che richiede intervento manuale."""
    pass


# =============================================================================
# FUNZIONI DI IDENTIFICAZIONE DATASET
# =============================================================================

def identify_dataset_type(df_name: str) -> str | None:
    """
    Identifica il tipo di dataset dal nome del file.

    Args:
        df_name: nome del file (es. "VOLUMES_merged.csv")

    Returns:
        Codice del tipo dataset (es. "VOLUMES") o None
    """
    for code in FILTER_COLUMNS.keys():
        if code in df_name.upper():
            return code
    return None


def get_filter_column(df_name: str) -> str | None:
    """
    Restituisce la colonna filtro per il dataset.

    Args:
        df_name: nome del file

    Returns:
        Nome colonna filtro o None se dataset senza filtro
    """
    ds_type = identify_dataset_type(df_name)
    if ds_type:
        return FILTER_COLUMNS[ds_type]
    return None


# =============================================================================
# FUNZIONI PER METADATA ARRICCHITI
# =============================================================================

def load_json_settings(file_path: Path) -> dict:
    """Carica un file JSON di settings."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_method_for_column(col_name: str, combination: dict) -> str | None:
    """
    Determina quale metodo usare per una colonna basandosi sulla combinazione.
    Es: AB40_CSF -> usa combination['METHOD_CSF']
    """
    for suffix, filter_key in COLUMN_METHOD_MAPPING.items():
        if suffix in col_name:
            return combination.get(filter_key)
    return None


def filter_volume_settings(settings: dict, df_columns: list) -> tuple:
    """
    Filtra volume_values_settings per colonne esistenti nel DataFrame.

    Returns:
        (filtered_settings, list_of_columns)
    """
    filtered = {}
    for col, values in settings.items():
        if col in df_columns:
            filtered[col] = values
    return filtered, list(filtered.keys())


def filter_normalization_settings(settings: dict, df_columns: list,
                                   combination: dict) -> tuple:
    """
    Filtra normalization_settings per colonne esistenti, selezionando il metodo corretto.

    Logica:
    - Se il valore e' una lista [min, max, direction] -> usa direttamente
    - Se il valore e' un dict con metodi -> seleziona il metodo dalla combination

    Returns:
        (filtered_settings, list_of_columns)
    """
    filtered = {}
    for col, values in settings.items():
        if col not in df_columns:
            continue

        if isinstance(values, list):
            # Formato semplice: [min, max, direction]
            filtered[col] = values
        elif isinstance(values, dict):
            # Formato con metodi: {"elecsys": [...], "lumipulse": [...]}
            method = get_method_for_column(col, combination)
            if method and method.lower() in values:
                filtered[col] = values[method.lower()]
            elif 'unknown' in values:
                # Fallback a unknown se disponibile
                filtered[col] = values['unknown']

    return filtered, list(filtered.keys())


def filter_predittori(df_columns: list) -> list:
    """
    Filtra PREDITTORI_COLUMNS per colonne esistenti nel DataFrame.

    Returns:
        lista di tutte le colonne predittori esistenti
    """
    all_predittori = []
    for category, cols in PREDITTORI_COLUMNS.items():
        existing = [c for c in cols if c in df_columns]
        all_predittori.extend(existing)
    return all_predittori


def filter_cofattori(df_columns: list, exclude: list = None) -> list:
    """
    Filtra COFATTORI_COLUMNS per colonne esistenti, con possibilita' di esclusione.

    Args:
        df_columns: lista colonne del DataFrame
        exclude: lista di cofattori da escludere

    Returns:
        lista cofattori filtrati
    """
    exclude = exclude or []
    return [c for c in COFATTORI_COLUMNS if c in df_columns and c not in exclude]


def filter_cofattori_metadata(settings: dict, df_columns: list) -> dict:
    """Filtra cofattori_values_settings per colonne esistenti."""
    return {col: values for col, values in settings.items() if col in df_columns}


def build_enriched_metadata(df: pd.DataFrame, combination: dict,
                            file_code: str, exclude_cofattori: list = None) -> dict:
    """
    Costruisce i metadata arricchiti per l'upload sul datalake.

    Args:
        df: DataFrame del merge
        combination: dict con i filtri usati (FSVERSION, METHOD_CSF, etc.)
        file_code: codice identificativo del file
        exclude_cofattori: lista di cofattori da escludere

    Returns:
        dict con tutti i metadata
    """
    df_columns = list(df.columns)

    # Carica settings
    volume_settings = load_json_settings(VOLUME_SETTINGS_FILE)
    norm_settings = load_json_settings(NORMALIZATION_SETTINGS_FILE)
    cofattori_settings = load_json_settings(COFATTORI_SETTINGS_FILE)

    # Filtra settings per colonne esistenti
    volume_norm_values, norm_volume = filter_volume_settings(volume_settings, df_columns)
    norm_scale_value, norm_scala = filter_normalization_settings(norm_settings, df_columns, combination)
    predittori = filter_predittori(df_columns)
    cofattori = filter_cofattori(df_columns, exclude_cofattori)
    cofattori_metadata = filter_cofattori_metadata(cofattori_settings, df_columns)

    # Costruisci metadata
    metadata = {
        'file_code': file_code,
        'level': 'merged_comb',

        # Settings normalizzazione
        'volume_norm_values': volume_norm_values,
        'norm_volume': norm_volume,
        'norm_scale_value': norm_scale_value,
        'norm_scala': norm_scala,

        # Predittori e cofattori
        'predittori': predittori,
        'cofattori': cofattori,
        'cofattori_metadata': cofattori_metadata,
    }

    # Aggiungi filtri per dataset
    if 'FSVERSION' in combination:
        metadata['VOLUMES_filter'] = combination['FSVERSION']
    if 'METHOD_CSF' in combination:
        metadata['CSF_filter'] = combination['METHOD_CSF']
    if 'METHOD_PLASMA' in combination:
        metadata['PLASMA_filter'] = combination['METHOD_PLASMA']
    if 'METHOD_PET' in combination:
        metadata['PET_filter'] = combination['METHOD_PET']

    return metadata


# =============================================================================
# FUNZIONI PER GENERARE COMBINAZIONI
# =============================================================================

def get_all_filter_values(dfs: dict, df_names: list) -> dict:
    """
    Raccoglie tutti i valori possibili per ogni filtro dai dataset caricati.

    Args:
        dfs: dizionario {nome_file: dataframe}
        df_names: lista dei nomi file

    Returns:
        Dizionario {nome_filtro: [valori]} o {nome_filtro: None} se non presente
    """
    filter_values = {}

    for col_name in set(FILTER_COLUMNS.values()):
        if col_name is None:
            continue

        # Cerca in tutti i dataset
        values = set()
        for name in df_names:
            df = dfs.get(name)
            if df is not None and col_name in df.columns:
                vals = df[col_name].dropna().unique().tolist()
                values.update(vals)

        if values:
            filter_values[col_name] = sorted(list(values), key=str)
        else:
            filter_values[col_name] = None

    return filter_values


def generate_all_combinations(filter_values: dict) -> list:
    """
    Genera il prodotto cartesiano di tutti i filtri.

    Args:
        filter_values: dizionario da get_all_filter_values()

    Returns:
        Lista di dizionari, ognuno rappresenta una combinazione
    """
    # Rimuovi filtri senza valori
    active_filters = {k: v for k, v in filter_values.items() if v is not None}

    if not active_filters:
        return [{}]  # Nessun filtro attivo, una sola combinazione vuota

    # Genera prodotto cartesiano
    keys = list(active_filters.keys())
    values_lists = [active_filters[k] for k in keys]

    combinations = []
    for combo in product(*values_lists):
        combinations.append(dict(zip(keys, combo)))

    return combinations


# =============================================================================
# FUNZIONI DI SELEZIONE RIGA MIGLIORE
# =============================================================================

def count_nan_in_volume_cols(row: pd.Series) -> int:
    """Conta NaN nelle colonne volumetriche."""
    cols = [c for c in VOLUME_COLS if c in row.index]
    return row[cols].isna().sum()


def compare_rows_for_selection(row1: pd.Series, row2: pd.Series) -> int:
    """
    Confronta due righe e restituisce l'indice della migliore.

    Criteri in ordine:
    1. Meno NaN nelle colonne volumetriche
    2. STATUS complete > partial > nan
    3. FLDSTRENG maggiore
    4. Default: prima riga
    """
    # Criterio 1: Meno NaN
    nan1 = count_nan_in_volume_cols(row1)
    nan2 = count_nan_in_volume_cols(row2)

    if nan1 < nan2:
        return row1.name
    if nan2 < nan1:
        return row2.name

    # Criterio 2: STATUS
    status_priority = {'complete': 0, 'partial': 1}
    s1 = row1.get('STATUS', np.nan)
    s2 = row2.get('STATUS', np.nan)
    p1 = status_priority.get(s1, 2)
    p2 = status_priority.get(s2, 2)

    if p1 < p2:
        return row1.name
    if p2 < p1:
        return row2.name

    # Criterio 3: FLDSTRENG
    f1 = pd.to_numeric(row1.get('FLDSTRENG', np.nan), errors='coerce')
    f2 = pd.to_numeric(row2.get('FLDSTRENG', np.nan), errors='coerce')

    if pd.notna(f1) and pd.notna(f2):
        if f1 > f2:
            return row1.name
        if f2 > f1:
            return row2.name
    elif pd.notna(f1):
        return row1.name
    elif pd.notna(f2):
        return row2.name

    # Default
    return row1.name


def select_best_from_group(df_group: pd.DataFrame) -> int:
    """Seleziona la riga migliore da un gruppo di duplicati."""
    if len(df_group) == 1:
        return df_group.index[0]

    best_idx = df_group.index[0]
    for idx in df_group.index[1:]:
        best_idx = compare_rows_for_selection(
            df_group.loc[best_idx],
            df_group.loc[idx]
        )
    return best_idx


# =============================================================================
# FUNZIONI DI RIMOZIONE DUPLICATI E CARDINALITA'
# =============================================================================

def remove_duplicates_volumes(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Rimuove duplicati (stesso RID + EXAMDATE) dal dataset VOLUMES.
    """
    df = df.copy()

    duplicates = df.groupby([RID_COL, DATE_COL]).filter(lambda x: len(x) > 1)

    if duplicates.empty:
        if verbose:
            print("  Nessun duplicato trovato")
        return df

    n_groups = duplicates.groupby([RID_COL, DATE_COL]).ngroups

    if verbose:
        print(f"  Trovati {len(duplicates)} righe duplicate in {n_groups} gruppi")

    indices_to_drop = []

    for (rid, date), group in duplicates.groupby([RID_COL, DATE_COL]):
        best_idx = select_best_from_group(group)
        dropped = [i for i in group.index if i != best_idx]
        indices_to_drop.extend(dropped)

    df_clean = df.drop(indices_to_drop)

    if verbose:
        print(f"  Rimosse {len(indices_to_drop)} righe, rimaste {len(df_clean)}")

    return df_clean


def check_match_cardinality(matches: dict, verbose: bool = False) -> tuple:
    """
    Analizza la cardinalità dei match.

    Returns:
        stats: {tipo: count}
        problematic: {tipo: lista_match}
    """
    stats = {'1:1': 0, '1:N': 0, 'N:1': 0, 'N:N': 0}
    problematic = {'1:N': [], 'N:1': [], 'N:N': []}

    for key, (idx1, idx2) in matches.items():
        n1, n2 = len(idx1), len(idx2)

        if n1 == 1 and n2 == 1:
            stats['1:1'] += 1
        elif n1 == 1 and n2 > 1:
            stats['1:N'] += 1
            problematic['1:N'].append((key, idx1, idx2))
        elif n1 > 1 and n2 == 1:
            stats['N:1'] += 1
            problematic['N:1'].append((key, idx1, idx2))
        else:
            stats['N:N'] += 1
            problematic['N:N'].append((key, idx1, idx2))

    if verbose:
        print(f"    1:1: {stats['1:1']}, 1:N: {stats['1:N']}, N:1: {stats['N:1']}, N:N: {stats['N:N']}")

    return stats, problematic


def reduce_matches_to_one_to_one(df: pd.DataFrame, matches: dict,
                                  problematic: dict, side: str = 'df1') -> dict:
    """
    Riduce match N:1 o 1:N a 1:1 selezionando la riga migliore.
    """
    reduced_matches = matches.copy()
    key_list = 'N:1' if side == 'df1' else '1:N'

    for key, idx1, idx2 in problematic[key_list]:
        if side == 'df1':
            group = df.loc[idx1]
            best_idx = select_best_from_group(group)
            reduced_matches[key] = [[best_idx], idx2]
        else:
            group = df.loc[idx2]
            best_idx = select_best_from_group(group)
            reduced_matches[key] = [idx1, [best_idx]]

    return reduced_matches


# =============================================================================
# FUNZIONI DI FILTRAGGIO BUFFER MATCHES
# =============================================================================

def filter_buffer_matches_to_merge(df_base: pd.DataFrame, df_add: pd.DataFrame,
                                    buffer_matches: dict) -> tuple:
    """
    Filtra buffer matches decidendo quali mergiare.

    Returns:
        to_merge, to_skip, manual_check
    """
    to_merge = {}
    to_skip = {}
    manual_check = []

    for key, (idx1, idx2) in buffer_matches.items():
        x1 = idx1[0] if isinstance(idx1, list) else idx1
        x2 = idx2[0] if isinstance(idx2, list) else idx2

        viscode1 = df_base.loc[x1, 'VISCODE'] if 'VISCODE' in df_base.columns else None
        viscode2 = df_add.loc[x2, 'VISCODE'] if 'VISCODE' in df_add.columns else None

        date1 = pd.to_datetime(df_base.loc[x1, DATE_COL])
        date2 = pd.to_datetime(df_add.loc[x2, DATE_COL])

        if viscode1 != viscode2:
            if viscode1 and viscode2 and 'm' in str(viscode1) and 'm' in str(viscode2):
                manual_check.append((key, idx1, idx2))
                to_skip[key] = [idx1, idx2]
                continue

        if viscode1 == viscode2 or abs(date1 - date2) < pd.Timedelta(days=30):
            to_merge[key] = [idx1, idx2]
        else:
            to_skip[key] = [idx1, idx2]

    return to_merge, to_skip, manual_check


def align_dates_for_buffer(df_base: pd.DataFrame, df_add: pd.DataFrame,
                           buffer_to_merge: dict, df_add_name: str) -> tuple:
    """Allinea le date dei buffer matches."""
    df_base = df_base.copy()
    df_add = df_add.copy()

    for key, (idx1, idx2) in buffer_to_merge.items():
        x1 = idx1[0] if isinstance(idx1, list) else idx1
        x2 = idx2[0] if isinstance(idx2, list) else idx2

        df_add.loc[x2, DATE_COL] = df_base.loc[x1, DATE_COL]

    return df_base, df_add


# =============================================================================
# FUNZIONE DI MERGE DATASETS
# =============================================================================

def merge_datasets(df_base: pd.DataFrame, df_add: pd.DataFrame,
                   exact_matches: dict, buffer_to_merge: dict) -> pd.DataFrame:
    """
    Esegue il merge effettivo tra due dataset.
    """
    df_base = df_base.copy()
    df_add = df_add.copy()

    cols_only_add = list(df_add.columns.difference(df_base.columns))

    all_matches = {**exact_matches, **buffer_to_merge}
    matched_idx2 = set()

    for key, (idx1, idx2) in all_matches.items():
        if isinstance(idx2, list):
            matched_idx2.update(idx2)
        else:
            matched_idx2.add(idx2)

    for col in cols_only_add:
        df_base[col] = np.nan

    for key, (idx1, idx2) in all_matches.items():
        i1 = idx1[0] if isinstance(idx1, list) else idx1
        i2 = idx2[0] if isinstance(idx2, list) else idx2

        for col in cols_only_add:
            df_base.loc[i1, col] = df_add.loc[i2, col]

    unmatched_idx2 = df_add.index.difference(matched_idx2)
    df_add_unmatched = df_add.loc[unmatched_idx2].copy()

    for col in df_base.columns:
        if col not in df_add_unmatched.columns:
            df_add_unmatched[col] = np.nan

    df_add_unmatched = df_add_unmatched[df_base.columns]

    result = pd.concat([df_base, df_add_unmatched], ignore_index=True)

    return result


# =============================================================================
# FUNZIONE PRINCIPALE DI MERGE TRA DUE DATASET
# =============================================================================

def merge_single_pair(df_base: pd.DataFrame, df_add: pd.DataFrame,
                      df1_name: str, df2_name: str,
                      merge_tools: MergerTools,
                      verbose: bool = True) -> pd.DataFrame:
    """
    Esegue il merge completo tra due dataset.

    Args:
        df_base: primo dataset
        df_add: secondo dataset da aggiungere
        df1_name: nome del primo dataset
        df2_name: nome del secondo dataset
        merge_tools: istanza di MergerTools
        verbose: se True, stampa info

    Returns:
        DataFrame mergiato

    Raises:
        MergeError: se ci sono match 1:N o N:N non gestibili
    """
    df_base = df_base.copy()
    df_add = df_add.copy()

    if verbose:
        print(f"  Merging: {df1_name} ({len(df_base)} rows) + {df2_name} ({len(df_add)} rows)")

    # Rimuovi duplicati se VOLUMES
    if 'VOLUMES' in df1_name.upper():
        df_base = remove_duplicates_volumes(df_base, verbose=verbose)
    if 'VOLUMES' in df2_name.upper():
        df_add = remove_duplicates_volumes(df_add, verbose=verbose)

    # Trova matches
    exact_matches, buffer_matches = merge_tools.find_visit_matches(
        df_base, df_add, buffer_days=BUFFER_DAYS
    )

    if verbose:
        print(f"    Exact matches: {len(exact_matches)}, Buffer matches: {len(buffer_matches)}")

    # Verifica cardinalità exact matches
    stats_exact, prob_exact = check_match_cardinality(exact_matches, verbose=False)

    # Controllo errori bloccanti
    if prob_exact['1:N'] or prob_exact['N:N']:
        raise MergeError(
            f"Match 1:N ({len(prob_exact['1:N'])}) o N:N ({len(prob_exact['N:N'])}) "
            f"trovati in exact matches tra {df1_name} e {df2_name}. "
            f"Richiesto intervento manuale."
        )

    # Riduci N:1 automaticamente
    if prob_exact['N:1']:
        exact_matches = reduce_matches_to_one_to_one(
            df_base, exact_matches, prob_exact, side='df1'
        )
        if verbose:
            print(f"    Ridotti {len(prob_exact['N:1'])} match exact N:1 a 1:1")

    # Verifica cardinalità buffer matches
    stats_buffer, prob_buffer = check_match_cardinality(buffer_matches, verbose=False)

    if prob_buffer['1:N'] or prob_buffer['N:N']:
        raise MergeError(
            f"Match 1:N ({len(prob_buffer['1:N'])}) o N:N ({len(prob_buffer['N:N'])}) "
            f"trovati in buffer matches tra {df1_name} e {df2_name}. "
            f"Richiesto intervento manuale."
        )

    if prob_buffer['N:1']:
        buffer_matches = reduce_matches_to_one_to_one(
            df_base, buffer_matches, prob_buffer, side='df1'
        )
        if verbose:
            print(f"    Ridotti {len(prob_buffer['N:1'])} match buffer N:1 a 1:1")

    # Filtra buffer matches
    buffer_to_merge, buffer_to_skip, manual_check = filter_buffer_matches_to_merge(
        df_base, df_add, buffer_matches
    )

    if verbose:
        print(f"    Buffer to merge: {len(buffer_to_merge)}, skipped: {len(buffer_to_skip)}")

    # Controllo finale cardinalità buffer_to_merge
    if buffer_to_merge:
        stats_merge, prob_merge = check_match_cardinality(buffer_to_merge, verbose=False)
        if prob_merge['N:1']:
            buffer_to_merge = reduce_matches_to_one_to_one(
                df_base, buffer_to_merge, prob_merge, side='df1'
            )

    # Allinea date
    if buffer_to_merge:
        df_base, df_add = align_dates_for_buffer(df_base, df_add, buffer_to_merge, df2_name)

    # Merge finale
    df_merged = merge_datasets(df_base, df_add, exact_matches, buffer_to_merge)

    if verbose:
        print(f"    Result: {len(df_merged)} rows")

    return df_merged


# =============================================================================
# FUNZIONI DI ORCHESTRAZIONE
# =============================================================================

def apply_filter_for_combination(df: pd.DataFrame, ds_name: str,
                                  combination: dict) -> pd.DataFrame:
    """
    Applica il filtro corretto al dataset basato sulla combinazione.

    Args:
        df: dataframe da filtrare
        ds_name: nome/tipo del dataset (es. 'VOLUMES')
        combination: dizionario della combinazione corrente

    Returns:
        DataFrame filtrato
    """
    filter_col = FILTER_COLUMNS.get(ds_name)

    if filter_col is None:
        return df.copy()

    filter_val = combination.get(filter_col)

    if filter_val is None:
        return df.copy()

    return df[df[filter_col] == filter_val].copy()


def run_incremental_merge(dfs: dict, combination: dict,
                          merge_tools: MergerTools,
                          verbose: bool = True) -> pd.DataFrame:
    """
    Esegue il merge incrementale per una combinazione specifica.

    Args:
        dfs: dizionario {dataset_type: dataframe}
        combination: combinazione di filtri da applicare
        merge_tools: istanza di MergerTools
        verbose: se True, stampa info

    Returns:
        DataFrame finale con tutti i dataset mergiati
    """
    df_result = None
    result_name = "merged"

    for ds_name in MERGE_ORDER:
        if ds_name not in dfs:
            if verbose:
                print(f"  Skipping {ds_name} (non presente)")
            continue

        df_filtered = apply_filter_for_combination(dfs[ds_name], ds_name, combination)

        if df_filtered.empty:
            if verbose:
                print(f"  Skipping {ds_name} (vuoto dopo filtro)")
            continue

        if df_result is None:
            df_result = df_filtered.copy()
            result_name = ds_name
            if verbose:
                print(f"  Base dataset: {ds_name} ({len(df_result)} rows)")
        else:
            df_result = merge_single_pair(
                df_result, df_filtered,
                result_name, ds_name,
                merge_tools,
                verbose=verbose
            )
            result_name = f"{result_name}+{ds_name}"

    return df_result


# =============================================================================
# FUNZIONI DI STATISTICHE
# =============================================================================

def compute_merge_statistics(df: pd.DataFrame, file_code: str,
                              combination: dict) -> dict:
    """
    Calcola statistiche essenziali su un dataset mergiato.

    Args:
        df: dataframe da analizzare
        file_code: codice file per identificazione
        combination: combinazione di filtri usata

    Returns:
        Dizionario con tutte le statistiche
    """
    stats = {
        'file_code': file_code,
        'filename': combination_to_filename(combination),
    }

    # Aggiungi valori della combinazione come colonne separate
    for key, val in combination.items():
        stats[key] = val

    # Dimensioni
    stats['n_righe'] = len(df)
    stats['n_colonne'] = len(df.columns)

    # Soggetti e visite
    stats['n_soggetti'] = df[RID_COL].nunique()

    visite_per_soggetto = df.groupby(RID_COL).size()
    stats['visite_min'] = visite_per_soggetto.min()
    stats['visite_max'] = visite_per_soggetto.max()
    stats['visite_media'] = round(visite_per_soggetto.mean(), 2)
    stats['visite_mediana'] = visite_per_soggetto.median()
    stats['visite_std'] = round(visite_per_soggetto.std(), 2)
    stats['visite_q25'] = visite_per_soggetto.quantile(0.25)
    stats['visite_q75'] = visite_per_soggetto.quantile(0.75)

    # Range temporale
    if DATE_COL in df.columns:
        dates = pd.to_datetime(df[DATE_COL])
        stats['date_min'] = dates.min().strftime('%Y-%m-%d')
        stats['date_max'] = dates.max().strftime('%Y-%m-%d')
    else:
        stats['date_min'] = None
        stats['date_max'] = None

    # NaN statistiche aggregate
    nan_pct = (df.isna().sum() / len(df) * 100)
    stats['nan_medio_pct'] = round(nan_pct.mean(), 2)
    stats['nan_max_pct'] = round(nan_pct.max(), 2)
    stats['colonne_complete'] = int((nan_pct == 0).sum())
    stats['colonne_con_nan'] = int((nan_pct > 0).sum())

    return stats


def save_statistics_csv(all_stats: list, output_file: Path = STATS_OUTPUT_FILE) -> Path:
    """
    Salva tutte le statistiche in un unico CSV di riepilogo.

    Args:
        all_stats: lista di dizionari con statistiche per ogni merge
        output_file: percorso file output

    Returns:
        Path del file salvato
    """
    df_stats = pd.DataFrame(all_stats)

    # Riordina colonne: prima file_code e combinazione, poi stats
    first_cols = ['file_code', 'filename']
    filter_cols = [c for c in df_stats.columns if c in FILTER_COLUMNS.values() and c is not None]
    stat_cols = [c for c in df_stats.columns if c not in first_cols + filter_cols]

    ordered_cols = first_cols + sorted(filter_cols) + stat_cols
    df_stats = df_stats[[c for c in ordered_cols if c in df_stats.columns]]

    df_stats.to_csv(output_file, index=False)

    return output_file


def combination_to_filename(combination: dict) -> str:
    """Genera nome file dalla combinazione."""
    if not combination:
        return "merge_all.csv"

    parts = []
    for key, val in sorted(combination.items()):
        safe_val = str(val).replace('.', '-').replace('/', '-')
        parts.append(f"{key}_{safe_val}")

    return f"merge_{'_'.join(parts)}.csv"


def combination_to_file_code(combination: dict) -> str:
    """Genera file_code dalla combinazione per il datalake."""
    if not combination:
        return "FINALMERGE_ALL"

    parts = []
    for key, val in sorted(combination.items()):
        safe_val = str(val).replace('.', '-').replace('/', '-')
        parts.append(f"{key}_{safe_val}")

    return f"FINALMERGE_{'_'.join(parts)}"


def save_merged_result(df: pd.DataFrame, combination: dict,
                       client: DatalakeClient,
                       exclude_cofattori: list = None) -> dict:
    """
    Carica il risultato del merge sul datalake con metadata arricchiti.

    Args:
        df: DataFrame da caricare
        combination: combinazione usata per il merge
        client: istanza di DatalakeClient
        exclude_cofattori: lista di cofattori da escludere dai metadata

    Returns:
        Risultato dell'upload dal client
    """
    filename = combination_to_filename(combination)
    file_code = combination_to_file_code(combination)

    # Costruisci metadata arricchiti
    metadata = build_enriched_metadata(df, combination, file_code, exclude_cofattori)

    result = client.upload_dataframe(
        df=df,
        object_name=filename,
        prefix=DATALAKE_PREFIX,
        metadata=metadata
    )

    return result


def run_all_combinations(dfs: dict, client: DatalakeClient,
                         verbose: bool = True,
                         exclude_cofattori: list = None) -> tuple:
    """
    Entry point principale: esegue tutti i merge per tutte le combinazioni.

    Args:
        dfs: dizionario {dataset_type: dataframe}
        client: istanza di DatalakeClient per upload risultati
        verbose: se True, stampa info
        exclude_cofattori: lista di cofattori da escludere dai metadata

    Returns:
        Tuple (results, stats_file):
            - results: Dizionario {file_code: risultato_upload}
            - stats_file: Path del CSV con statistiche
    """
    merge_tools = MergerTools()

    # Estrai valori filtro dai dataset
    filter_values = {}
    for ds_name, df in dfs.items():
        filter_col = FILTER_COLUMNS.get(ds_name)
        if filter_col and filter_col in df.columns:
            vals = df[filter_col].dropna().unique().tolist()
            if vals:
                if filter_col not in filter_values:
                    filter_values[filter_col] = set()
                filter_values[filter_col].update(vals)

    # Converti set in liste ordinate
    filter_values = {k: sorted(list(v), key=str) for k, v in filter_values.items()}

    # Applica whitelist per filtrare i valori
    for filter_col, whitelist in FILTER_VALUES_WHITELIST.items():
        if filter_col not in filter_values:
            continue

        if whitelist is None:
            # None = usa tutti i valori (non filtra)
            continue
        elif len(whitelist) == 0:
            # Lista vuota = rimuovi questo filtro
            del filter_values[filter_col]
        else:
            # Lista di valori = usa solo questi
            available = set(filter_values[filter_col])
            selected = [v for v in whitelist if v in available]
            if selected:
                filter_values[filter_col] = selected
            else:
                # Nessun valore della whitelist è disponibile
                print(f"  WARNING: nessun valore di {filter_col} nella whitelist trovato nei dati")
                print(f"    Whitelist: {whitelist}")
                print(f"    Disponibili: {available}")

    # Genera combinazioni
    combinations = generate_all_combinations(filter_values)

    print(f"\n{'='*60}")
    print(f"AUTOMATED MERGE PIPELINE")
    print(f"{'='*60}")
    print(f"Dataset disponibili: {list(dfs.keys())}")
    print(f"Filtri trovati: {filter_values}")
    print(f"Combinazioni totali: {len(combinations)}")
    print(f"Datalake prefix: {DATALAKE_PREFIX}")
    print(f"Stats output: {STATS_OUTPUT_FILE}")
    print(f"{'='*60}\n")

    results = {}
    all_stats = []  # Raccoglie statistiche per ogni merge

    for i, comb in enumerate(combinations):
        print(f"\n[{i+1}/{len(combinations)}] Combinazione: {comb or 'nessun filtro'}")
        print("-" * 50)

        try:
            df_merged = run_incremental_merge(dfs, comb, merge_tools, verbose=verbose)

            if df_merged is not None:
                # Upload sul datalake con metadata arricchiti
                upload_result = save_merged_result(df_merged, comb, client, exclude_cofattori)
                file_code = combination_to_file_code(comb)
                results[file_code] = upload_result

                # Calcola e raccogli statistiche
                stats = compute_merge_statistics(df_merged, file_code, comb)
                all_stats.append(stats)

                print(f"  CARICATO: {DATALAKE_PREFIX}/{combination_to_filename(comb)}")
                print(f"  File code: {file_code}")
                print(f"  Righe: {len(df_merged)}, Colonne: {len(df_merged.columns)}")
                print(f"  Soggetti: {stats['n_soggetti']}, NaN medio: {stats['nan_medio_pct']}%")
            else:
                print(f"  SKIP: nessun dato dopo i filtri")

        except MergeError as e:
            print(f"\n{'!'*60}")
            print(f"ERRORE: {e}")
            print(f"{'!'*60}")
            raise

    # Salva CSV statistiche in locale
    stats_file = None
    if all_stats:
        stats_file = save_statistics_csv(all_stats)
        print(f"\n{'='*60}")
        print(f"STATISTICHE SALVATE: {stats_file}")

    print(f"\n{'='*60}")
    print(f"COMPLETATO: {len(results)} merge caricati sul datalake")
    print(f"{'='*60}")

    return results, stats_file


# =============================================================================
# FUNZIONE DI CARICAMENTO DATASET
# =============================================================================

def load_datasets_from_datalake(client: DatalakeClient,
                                 file_codes: list = None) -> dict:
    """
    Carica i dataset dal datalake.

    Args:
        client: istanza di DatalakeClient
        file_codes: lista codici file da caricare. Se None, carica tutti.

    Returns:
        Dizionario {dataset_type: dataframe}
    """
    if file_codes is None:
        file_codes = ['VOLMERGE', 'SCALEMERGE', 'CSFMERGE', 'PETMERGE', 'COFMERGE'] #['VOLMERGE', 'SCALEMERGE', 'CSFMERGE', 'PLMERGE', 'PETMERGE', 'COFMERGE']

    print(f"Caricamento dataset: {file_codes}")

    search = client.query_files(query={'custom.file_code': file_codes})
    zip_files = client.download_file(search['object_name'], extract_zip=True)

    print(f"Scaricati {len(zip_files)} file")

    dfs = {}
    for file_name, df_raw in zip_files.items():
        df = df_raw.copy()
        df['EXAMDATE'] = pd.to_datetime(df['EXAMDATE'])
        if 'FSVERSION' in df.columns:
            df['FSVERSION'] = df['FSVERSION'].astype(str)

        # Identifica tipo dataset
        ds_type = identify_dataset_type(file_name)
        if ds_type:
            dfs[ds_type] = df
            print(f"  {ds_type}: {len(df)} righe")

    return dfs


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Entry point principale."""
    print(f"\n{'#'*60}")
    print(f"# AUTOMATED MERGE PIPELINE")
    print(f"# {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*60}\n")

    # Inizializza client datalake
    client = DatalakeClient()

    # Carica dataset
    dfs = load_datasets_from_datalake(client)

    # Esegui tutti i merge e carica sul datalake
    results, stats_file = run_all_combinations(dfs, client)

    # Riepilogo finale
    print(f"\n\nRiepilogo file caricati sul datalake:")
    for file_code, result in results.items():
        print(f"  {file_code}")

    if stats_file:
        print(f"\nStatistiche salvate in: {stats_file}")

    return results, stats_file


if __name__ == "__main__":
    main()
