"""
Script to inject missing values into synthetic data based on real data patterns.

This script:
1. Loads real and synthetic data from datalake using DatalakeClient
2. Calculates missing data rates from real data for each variable
3. Randomly injects missing values into synthetic data matching the real patterns
4. Uploads results back to datalake

Usage:
    python inject_missing_values.py --file-code-real ADNIMERGE_cleaned_03 --file-code-gen ADNIMERGE_synthetic --prefix synthetic/missing_injected
"""

import pandas as pd
import numpy as np
import argparse
import json
from pathlib import Path

# Import functions from reverse_from_datalake
from reverse_from_datalake import load_from_datalake, get_file_metadata, upload_to_datalake
from dl_client import DatalakeClient


def inject_missing_longitudinal(df_synth, df_real, target_columns=None, id_col='ID', time_col='TIME'):
    """
    Replica il pattern di missing data dai dati reali ai dati sintetici.

    Per ogni variabile:
    1. Calcola il missing rate dai dati reali
    2. Inietta randomicamente lo stesso numero di missing nei dati sintetici

    Parameters:
    -----------
    df_synth : pd.DataFrame
        Dati sintetici (saranno modificati in-place)
    df_real : pd.DataFrame
        Dati reali (usati per calcolare i missing rates)
    target_columns : list of str, optional
        Lista di colonne su cui applicare l'iniezione di missing.
        Se None (default), applica su tutte le colonne eccetto id_col e time_col
    id_col : str, default='ID'
        Nome della colonna ID paziente
    time_col : str, default='TIME'
        Nome della colonna temporale

    Returns:
    --------
    pd.DataFrame
        Dati sintetici con missing data iniettato
    """
    print("\n" + "="*80)
    print("INJECTING MISSING VALUES")
    print("="*80)

    # Copia il dataframe per non modificare l'originale
    df_result = df_synth.copy()

    # Determina le colonne su cui lavorare
    if target_columns is None:
        # Usa tutte le colonne eccetto ID e TIME
        columns_to_process = [c for c in df_real.columns if c not in [id_col, time_col]]
    else:
        # Usa solo le colonne specificate (verifica che esistano)
        columns_to_process = []
        for col in target_columns:
            if col in df_real.columns and col in df_synth.columns:
                columns_to_process.append(col)
            else:
                print(f"   ⚠ Warning: Column '{col}' not found in both datasets, skipping")

    print(f"\nProcessing {len(columns_to_process)} columns...")
    print(f"Real data shape: {df_real.shape}")
    print(f"Synthetic data shape: {df_synth.shape}")

    # Statistiche
    stats = {
        'processed_columns': [],
        'missing_injected': {}
    }

    # Per ogni colonna target
    for col in columns_to_process:
        # Calcola missing rate dai dati reali
        missing_rate = df_real[col].isna().mean()

        if missing_rate > 0:
            print(f"\n  Processing '{col}':")
            print(f"    Real missing rate: {missing_rate:.2%}")

            # Conta valori non-missing già presenti nel sintetico
            current_missing_rate = df_result[col].isna().mean()
            print(f"    Current synthetic missing rate: {current_missing_rate:.2%}")

            # NUOVO APPROCCIO: calcola il numero totale di missing da iniettare
            # a livello globale invece che per paziente

            # Identifica tutte le righe non-missing nel dataset sintetico
            non_missing_mask = df_result[col].notna()
            non_missing_indices = df_result[non_missing_mask].index.tolist()

            # Calcola il numero totale di missing da iniettare
            # Usa round() per una percentuale più precisa
            total_non_missing = len(non_missing_indices)
            n_missing_to_inject = round(len(df_result) * missing_rate - (len(df_result) - total_non_missing))

            # Assicurati che non cerchiamo di iniettare più missing di quanti valori non-missing abbiamo
            n_missing_to_inject = max(0, min(n_missing_to_inject, total_non_missing))

            print(f"    Total non-missing values available: {total_non_missing}")
            print(f"    Target missing to inject: {n_missing_to_inject}")

            if n_missing_to_inject > 0:
                # Seleziona randomicamente quali valori rendere missing
                missing_idx = np.random.choice(
                    non_missing_indices,
                    size=n_missing_to_inject,
                    replace=False
                )
                df_result.loc[missing_idx, col] = np.nan

            final_missing_rate = df_result[col].isna().mean()
            print(f"    Final synthetic missing rate: {final_missing_rate:.2%}")
            print(f"    Total values set to missing: {n_missing_to_inject}")

            stats['processed_columns'].append(col)
            stats['missing_injected'][col] = {
                'target_rate': missing_rate,
                'initial_rate': current_missing_rate,
                'final_rate': final_missing_rate,
                'values_injected': n_missing_to_inject
            }
        else:
            print(f"\n  Skipping '{col}': no missing values in real data")

    print("\n" + "="*80)
    print("INJECTION SUMMARY")
    print("="*80)
    print(f"Total columns processed: {len(stats['processed_columns'])}")
    print(f"Columns with missing injected: {len(stats['missing_injected'])}")

    return df_result, stats


def inject_missing_pipeline(file_code_real,
                            file_code_gen,
                            target_columns=None,
                            upload_to_dl=True,
                            prefix='synthetic/missing_injected',
                            save_local=False,
                            output_path=None,
                            query_params_real=None,
                            query_params_gen=None,
                            id_col='ID',
                            time_col='TIME'):
    """
    Pipeline completa: carica dati reali e sintetici, inietta missing, upload.

    Parameters:
    -----------
    file_code_real : str
        File code dei dati reali nel datalake
    file_code_gen : str
        File code dei dati sintetici/generati nel datalake
    target_columns : list of str, optional
        Lista di colonne su cui applicare l'iniezione.
        Se None, applica su tutte le colonne eccetto ID e TIME
    upload_to_dl : bool, default=True
        Se True, carica il risultato sul datalake
    prefix : str, default='synthetic/missing_injected'
        Prefix per l'upload sul datalake
    save_local : bool, default=False
        Se True, salva una copia locale
    output_path : str, optional
        Path per salvare il file CSV locale
    query_params_real : dict, optional
        Parametri query aggiuntivi per i dati reali
    query_params_gen : dict, optional
        Parametri query aggiuntivi per i dati sintetici
    id_col : str, default='ID'
        Nome colonna ID
    time_col : str, default='TIME'
        Nome colonna temporale

    Returns:
    --------
    pandas.DataFrame
        Dati sintetici con missing iniettati
    dict
        Statistiche sull'iniezione
    """
    print("="*80)
    print("MISSING VALUES INJECTION PIPELINE")
    print("="*80)

    # [STEP 1] Carica dati reali dal datalake
    print("\n[STEP 1] Loading REAL data from datalake...")
    df_real, _, _ = load_from_datalake(file_code_real, query_params_real)

    # [STEP 2] Carica dati sintetici dal datalake
    print("\n[STEP 2] Loading SYNTHETIC data from datalake...")
    df_synth, object_name_synth, original_file_name = load_from_datalake(file_code_gen, query_params_gen)

    # [STEP 3] Inietta missing values
    print("\n[STEP 3] Injecting missing values...")
    df_result, stats = inject_missing_longitudinal(
        df_synth=df_synth,
        df_real=df_real,
        target_columns=target_columns,
        id_col=id_col,
        time_col=time_col
    )

    # [STEP 4] Upload al datalake
    if upload_to_dl:
        print("\n[STEP 4] Uploading to datalake...")

        # Ottieni metadata originali dei dati sintetici
        client = DatalakeClient()
        query = {'custom.file_code': file_code_gen}
        if query_params_gen:
            query.update(query_params_gen)

        search_result = client.search_files(query=query)
        if not search_result or 'files' not in search_result or len(search_result['files']) == 0:
            raise ValueError(f"No file found with code '{file_code_gen}' for metadata retrieval")

        metadata_object_name = search_result['files'][0]['object_name']
        original_metadata = get_file_metadata(metadata_object_name)

        # Crea nuovi metadata
        new_metadata = original_metadata.copy()
        original_file_code = new_metadata.get('file_code', file_code_gen)
        new_metadata['file_code'] = f"{original_file_code}_missing_injected"
        new_metadata['missing_injection'] = {
            'source_real': file_code_real,
            'source_synthetic': file_code_gen,
            'columns_processed': stats['processed_columns'],
            'n_columns_with_missing': len(stats['missing_injected'])
        }

        # Crea nuovo file name
        file_name_parts = original_file_name.rsplit('.', 1)
        if len(file_name_parts) == 2:
            new_file_name = f"{file_name_parts[0]}_missing_injected.{file_name_parts[1]}"
        else:
            new_file_name = f"{original_file_name}_missing_injected"

        # Upload
        upload_success = upload_to_datalake(
            df=df_result,
            file_name=new_file_name,
            prefix=prefix,
            metadata=new_metadata
        )

        if not upload_success:
            print("   ⚠ Warning: Upload to datalake failed!")

    # [STEP 5] Salva copia locale se richiesto
    if save_local and output_path:
        print(f"\n[STEP 5] Saving local copy to: {output_path}")
        df_result.to_csv(output_path, index=False)
        print(f"   ✓ Saved successfully")

    print("\n" + "="*80)
    print("PIPELINE COMPLETED")
    print("="*80)
    print(f"Final dataset shape: {df_result.shape}")

    return df_result, stats


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description='Inject missing values into synthetic data based on real data patterns'
    )

    parser.add_argument(
        '--file-code-real',
        type=str,
        required=True,
        help='File code for REAL data in datalake (e.g., ADNIMERGE_cleaned_03)'
    )

    parser.add_argument(
        '--file-code-gen',
        type=str,
        required=True,
        help='File code for SYNTHETIC/GENERATED data in datalake (e.g., ADNIMERGE_synthetic)'
    )

    parser.add_argument(
        '--target-columns',
        type=str,
        nargs='+',
        default=None,
        help='Specific columns to inject missing values into (default: all columns except ID and TIME)'
    )

    parser.add_argument(
        '--prefix',
        type=str,
        default='synthetic/missing_injected',
        help='Datalake prefix for upload (default: synthetic/missing_injected)'
    )

    parser.add_argument(
        '--no-upload',
        action='store_true',
        help='Do not upload to datalake'
    )

    parser.add_argument(
        '--save-local',
        action='store_true',
        help='Save a local copy of the result'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output CSV file path for local copy (requires --save-local)'
    )

    parser.add_argument(
        '--query-param-real',
        type=str,
        action='append',
        help='Additional query parameters for REAL data in format key=value'
    )

    parser.add_argument(
        '--query-param-gen',
        type=str,
        action='append',
        help='Additional query parameters for SYNTHETIC data in format key=value'
    )

    parser.add_argument(
        '--id-col',
        type=str,
        default='ID',
        help='Name of the ID column (default: ID)'
    )

    parser.add_argument(
        '--time-col',
        type=str,
        default='TIME',
        help='Name of the TIME column (default: TIME)'
    )

    args = parser.parse_args()

    # Parse query parameters per dati reali
    query_params_real = {}
    if args.query_param_real:
        for param in args.query_param_real:
            key, value = param.split('=', 1)
            query_params_real[key] = value

    # Parse query parameters per dati sintetici
    query_params_gen = {}
    if args.query_param_gen:
        for param in args.query_param_gen:
            key, value = param.split('=', 1)
            query_params_gen[key] = value

    # Run pipeline
    df_result, stats = inject_missing_pipeline(
        file_code_real=args.file_code_real,
        file_code_gen=args.file_code_gen,
        target_columns=args.target_columns,
        upload_to_dl=not args.no_upload,
        prefix=args.prefix,
        save_local=args.save_local,
        output_path=args.output,
        query_params_real=query_params_real if query_params_real else None,
        query_params_gen=query_params_gen if query_params_gen else None,
        id_col=args.id_col,
        time_col=args.time_col
    )

    # Stampa statistiche finali
    print("\nDETAILED STATISTICS:")
    print("="*80)
    for col, col_stats in stats['missing_injected'].items():
        print(f"\n{col}:")
        print(f"  Target rate: {col_stats['target_rate']:.2%}")
        print(f"  Initial rate: {col_stats['initial_rate']:.2%}")
        print(f"  Final rate: {col_stats['final_rate']:.2%}")
        print(f"  Values injected: {col_stats['values_injected']}")

    return df_result, stats


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 1:
        # No arguments provided, show examples
        print("="*80)
        print("MISSING VALUES INJECTION - EXAMPLE USAGE")
        print("="*80)
        print("\nBasic usage (inject missing on all columns):")
        print("  python inject_missing_values.py --file-code-real ADNIMERGE_cleaned_03 --file-code-gen ADNIMERGE_synthetic")

        print("\nWith specific columns:")
        print("  python inject_missing_values.py --file-code-real ADNIMERGE_cleaned_03 --file-code-gen ADNIMERGE_synthetic --target-columns ADAS11 ADAS13 MMSE")

        print("\nWith custom prefix:")
        print("  python inject_missing_values.py --file-code-real ADNIMERGE_cleaned_03 --file-code-gen ADNIMERGE_validation --prefix validation/missing_injected")

        print("\nSave local copy without upload:")
        print("  python inject_missing_values.py --file-code-real ADNIMERGE_cleaned_03 --file-code-gen ADNIMERGE_synthetic --no-upload --save-local --output result.csv")

        print("\nWith query parameters:")
        print("  python inject_missing_values.py --file-code-real ADNIMERGE_cleaned_03 --file-code-gen ADNIMERGE_synthetic --query-param-real custom.level=cleaned_03")

        print("\n" + "="*80)
        print("PROGRAMMATIC USAGE")
        print("="*80)
        print("""
from inject_missing_values import inject_missing_pipeline

# Basic usage - inject on all columns
df_result, stats = inject_missing_pipeline(
    file_code_real='ADNIMERGE_cleaned_03',
    file_code_gen='ADNIMERGE_synthetic',
    upload_to_dl=True,
    prefix='synthetic/missing_injected'
)

# With specific columns
df_result, stats = inject_missing_pipeline(
    file_code_real='ADNIMERGE_cleaned_03',
    file_code_gen='ADNIMERGE_synthetic',
    target_columns=['ADAS11', 'ADAS13', 'MMSE', 'FAQ'],
    upload_to_dl=True,
    prefix='synthetic/missing_injected',
    save_local=True,
    output_path='result_with_missing.csv'
)
        """)
        print("="*80)
    else:
        # Run with command-line arguments
        main()
