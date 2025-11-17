"""
Method-Aware ATN Profile System for ADNI Data (Updated Version)
================================================================

This module provides method-aware ATN (Amyloid-Tau-Neurodegeneration) profiling
for ADNI biomarker data, with support for all CSF, plasma, and PET biomarkers.

Updated to work with actual ADNI_variables_cleanedBB.xlsx structure.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union


class ATNConfig:
    """
    Configuration class for ATN biomarker thresholds organized by method.
    Based on literature cutoffs for different analytical platforms.
    """
    
    def __init__(self):
        """Initialize ATN configuration with literature-based cutoffs."""
        
        # ====================
        # AMYLOID (A) CUTOFFS
        # ====================
        self.amyloid_cutoffs = {
            # CSF Amyloid Beta 42 (pg/mL)
            'AB42_CSF': {
                'elecsys': 1000,           # Roche Elecsys
                'lumipulse': 500,          # Fujirebio Lumipulse
                'elisa': 192,              # INNOTEST (ELISA)
                'AlzBio3': 192,            # Luminex AlzBio3
                'massospectometry': 600,   # Mass spectrometry
                '2DUPLC_massospectometry': 600,
                'MesoScaleDiscovery': 450,
                'MesoScaleDiscovery, Rockville MD': 450,
                'saladax': 500,
                'unknown': 192             # Default/conservative
            },
            
            # CSF Amyloid Beta 40 (pg/mL) - higher is abnormal
            'AB40_CSF': {
                'elecsys': 15000,
                'lumipulse': 10000,
                'elisa': 8000,
                'massospectometry': 10000,
                '2DUPLC_massospectometry': 10000,
                'MesoScaleDiscovery': 10000,
                'unknown': 10000
            },
            
            # CSF AB42/AB40 ratio - lower is worse
            'AB4240_CSF': {
                'elecsys': 0.059,
                'lumipulse': 0.069,
                'elisa': 0.073,
                'massospectometry': 0.067,
                '2DUPLC_massospectometry': 0.067,
                'unknown': 0.073
            },
            'A4240_CSF': {  # Alternative naming
                'elecsys': 0.059,
                'lumipulse': 0.069,
                'elisa': 0.073,
                'unknown': 0.073
            },
            
            # Plasma Amyloid (pg/mL)
            'AB42_PL': {
                'unknown': 13.5,   # C2N MS platform typical
                'c2n': 13.5,
                'simoa': 192
            },
            'AB40_PL': {
                'unknown': 200,    # Plasma levels much lower than CSF
                'c2n': 200,
                'simoa': 300
            },
            
            # Plasma AB42/AB40 ratio - lower is worse
            'AB4240_PL': {
                'unknown': 0.092,  # C2N MS
                'c2n': 0.092,
                'simoa': 0.103
            },
            
            # PET Amyloid
            'SUMMARY_SUVR': {
                'unknown': 1.11,   # Composite SUVR
                'florbetapir': 1.11,
                'florbetaben': 1.08,
                'flutemetamol': 0.60,
                'pib': 1.5
            },
            'AMY_CENTILOIDS': {
                'unknown': 20,     # Centiloid scale
            },
            'AMYLOID_STATUS': {
                'unknown': 1,      # Binary: 1 = positive, 0 = negative
            },
            'ENTORHINAL_SUVR': {
                'unknown': 1.11
            },
            'INFERIOR_TEMPORAL_SUVR': {
                'unknown': 1.11
            }
        }
        
        # ====================
        # TAU (T) CUTOFFS
        # ====================
        self.tau_cutoffs = {
            # CSF pTau181 (pg/mL)
            'PT181_CSF': {
                'elecsys': 24,             # Roche Elecsys
                'lumipulse': 26,           # Fujirebio Lumipulse
                'elisa': 23,               # INNOTEST
                'AlzBio3': 23,             # Luminex AlzBio3
                'MesoScaleDiscovery': 20,
                'unknown': 23
            },
            
            # CSF Total Tau (pg/mL)
            'TTAU_CSF': {
                'elecsys': 300,
                'lumipulse': 400,
                'elisa': 300,
                'AlzBio3': 300,
                'saladax': 350,
                'MesoScaleDiscovery': 280,
                'unknown': 300
            },
            
            # Plasma pTau181 (pg/mL)
            'PT181_PL': {
                'unknown': 2.0,    # Simoa typical
                'simoa': 2.0,
                'quanterix': 2.0,
                'c2n': 2.5
            },
            
            # Plasma pTau217 (pg/mL or ng/mL depending on assay)
            'PT217_PL': {
                'unknown': 0.020,  # C2N MS (pg/mL)
                'c2n': 0.020,
                'alzpath': 0.030,  # ng/mL (different units!)
                'janssen': 0.018
            },
            
            # Plasma Total Tau (pg/mL)
            'TTAU_PL': {
                'unknown': 3.5,    # Simoa
                'simoa': 3.5,
                'quanterix': 3.5
            },
            
            # Non-phosphorylated Tau 217
            'nPT217_PL': {
                'unknown': 0.150,  # pg/mL
                'c2n': 0.150
            },
            
            # pTau217/npTau217 ratio
            'PT217_nPT217_PL': {
                'unknown': 0.15,
                'c2n': 0.15
            },
            
            # Tau PET SUVR
            'ENTORHINAL_SUVR': {
                'unknown': 1.25,
                'flortaucipir': 1.25,
                'mk6240': 1.20
            },
            'FUSIFORM_SUVR': {
                'unknown': 1.30,
                'flortaucipir': 1.30
            },
            'INFERIOR_TEMPORAL_SUVR': {
                'unknown': 1.30,
                'flortaucipir': 1.30
            },
            'MIDDLETEMPORAL_SUVR': {
                'unknown': 1.30,
                'flortaucipir': 1.30
            },
            'PARAHIPPOCAMPAL_SUVR': {
                'unknown': 1.28,
                'flortaucipir': 1.28
            },
            'PRECUNEUS_SUVR': {
                'unknown': 1.25,
                'flortaucipir': 1.25
            }
        }
        
        # ====================
        # NEURODEGENERATION (N) CUTOFFS
        # ====================
        self.neurodegeneration_cutoffs = {
            # Plasma Neurofilament Light (pg/mL)
            'NFL_PL': {
                'unknown': 30,
                'simoa': 30,
                'quanterix': 30,
                'ella': 35
            },
            
            # CSF volumes and structures (from UCSF FreeSurfer)
            'Hippocampus': {
                'unknown': 6000,  # mm³ - threshold for significant atrophy
            },
            'Ventricles': {
                'unknown': 50000,  # mm³ - threshold for enlargement
            },
            'Entorhinal': {
                'unknown': 3000,   # mm³
            },
            'MidTemp': {
                'unknown': 20000,  # mm³
            },
            'WholeBrain': {
                'unknown': 1000000,  # mm³
            },
            
            # FDG-PET (if available)
            'FDG_SUVR': {
                'unknown': 1.21,   # Angular gyrus meta-ROI
            }
        }
    
    def get_cutoff(self, biomarker: str, method: Optional[str] = None, 
                   category: str = 'amyloid') -> Optional[float]:
        """
        Get the appropriate cutoff for a biomarker based on method.
        
        Args:
            biomarker: Name of the biomarker
            method: Analytical method
            category: 'amyloid', 'tau', or 'neurodegeneration'
            
        Returns:
            Cutoff threshold value, or None if not found
        """
        # Select appropriate cutoffs dictionary
        if category == 'amyloid':
            cutoffs_dict = self.amyloid_cutoffs
        elif category == 'tau':
            cutoffs_dict = self.tau_cutoffs
        elif category == 'neurodegeneration':
            cutoffs_dict = self.neurodegeneration_cutoffs
        else:
            return None
        
        # Check if biomarker exists in configuration
        if biomarker not in cutoffs_dict:
            return None
        
        biomarker_cutoffs = cutoffs_dict[biomarker]
        
        # If method is None, NaN, or not found, use 'unknown' default
        if method is None or pd.isna(method) or method not in biomarker_cutoffs:
            method = 'unknown'
        
        return biomarker_cutoffs.get(method)


class ATNProfileCalculator:
    """Calculator for ATN profiles with method-aware thresholds."""
    
    def __init__(self, config: Optional[ATNConfig] = None):
        """Initialize calculator with configuration."""
        self.config = config if config is not None else ATNConfig()
    
    def add_method_column(self, df: pd.DataFrame, 
                          variable_mapping: Dict[str, str]) -> pd.DataFrame:
        """Add method columns to the dataframe based on variable metadata."""
        df = df.copy()
        
        for var, method in variable_mapping.items():
            if var in df.columns:
                method_col = f'method_{var}'
                df[method_col] = method
        
        return df
    
    def calculate_Apositive(self, row: pd.Series, 
                           method_cols: Optional[Dict[str, str]] = None) -> int:
        """Calculate A-positive status using method-aware cutoffs."""
        if method_cols is None:
            method_cols = {}
        
        # Priority order for amyloid biomarkers
        biomarker_priority = [
            # First check ratios (most specific)
            ('AB4240_CSF', 'ratio', 'lower_worse'),
            ('A4240_CSF', 'ratio', 'lower_worse'),
            ('AB4240_PL', 'ratio', 'lower_worse'),
            # Then concentrations
            ('AB42_CSF', 'concentration', 'lower_worse'),
            ('AB42_PL', 'concentration', 'lower_worse'),
            # Then PET
            ('AMYLOID_STATUS', 'binary', 'higher_worse'),
            ('AMY_CENTILOIDS', 'centiloid', 'higher_worse'),
            ('SUMMARY_SUVR', 'suvr', 'higher_worse'),
            ('ENTORHINAL_SUVR', 'suvr', 'higher_worse'),
            ('INFERIOR_TEMPORAL_SUVR', 'suvr', 'higher_worse')
        ]
        
        for biomarker, bio_type, direction in biomarker_priority:
            if biomarker in row and pd.notna(row[biomarker]):
                # Get method for this biomarker
                method_col = method_cols.get(biomarker, f'method_{biomarker}')
                method = row.get(method_col, 'unknown')
                if pd.isna(method):
                    method = 'unknown'
                
                # Get cutoff
                cutoff = self.config.get_cutoff(biomarker, method, 'amyloid')
                if cutoff is None:
                    continue
                
                value = row[biomarker]
                
                # Handle binary biomarkers
                if bio_type == 'binary':
                    return int(value) if pd.notna(value) else pd.NA
                
                # Apply logic based on direction
                if direction == 'lower_worse':
                    return 1 if value < cutoff else 0
                elif direction == 'higher_worse':
                    return 1 if value >= cutoff else 0
        
        return pd.NA
    
    def calculate_Tpositive(self, row: pd.Series,
                           method_cols: Optional[Dict[str, str]] = None) -> int:
        """Calculate T-positive status using method-aware cutoffs."""
        if method_cols is None:
            method_cols = {}
        
        # Priority order for tau biomarkers
        biomarker_priority = [
            # CSF first (most established)
            'PT181_CSF',
            'TTAU_CSF',
            # Plasma next
            'PT217_PL',
            'PT181_PL',
            'TTAU_PL',
            'nPT217_PL',
            'PT217_nPT217_PL',
            # PET last
            'ENTORHINAL_SUVR',
            'FUSIFORM_SUVR',
            'INFERIOR_TEMPORAL_SUVR',
            'MIDDLETEMPORAL_SUVR',
            'PARAHIPPOCAMPAL_SUVR',
            'PRECUNEUS_SUVR'
        ]
        
        for biomarker in biomarker_priority:
            if biomarker in row and pd.notna(row[biomarker]):
                # Get method
                method_col = method_cols.get(biomarker, f'method_{biomarker}')
                method = row.get(method_col, 'unknown')
                if pd.isna(method):
                    method = 'unknown'
                
                # Get cutoff
                cutoff = self.config.get_cutoff(biomarker, method, 'tau')
                if cutoff is None:
                    continue
                
                value = row[biomarker]
                
                # For all tau markers, higher is worse
                return 1 if value >= cutoff else 0
        
        return pd.NA
    
    def calculate_Npositive(self, row: pd.Series,
                           method_cols: Optional[Dict[str, str]] = None) -> int:
        """Calculate N-positive status using method-aware cutoffs."""
        if method_cols is None:
            method_cols = {}
        
        # Priority order for neurodegeneration biomarkers
        biomarker_priority = [
            ('NFL_PL', 'higher_worse'),
            ('Hippocampus', 'lower_worse'),
            ('Entorhinal', 'lower_worse'),
            ('Ventricles', 'higher_worse'),
            ('WholeBrain', 'lower_worse'),
            ('FDG_SUVR', 'lower_worse')
        ]
        
        for biomarker, direction in biomarker_priority:
            if biomarker in row and pd.notna(row[biomarker]):
                # Get method
                method_col = method_cols.get(biomarker, f'method_{biomarker}')
                method = row.get(method_col, 'unknown')
                if pd.isna(method):
                    method = 'unknown'
                
                # Get cutoff
                cutoff = self.config.get_cutoff(biomarker, method, 'neurodegeneration')
                if cutoff is None:
                    continue
                
                value = row[biomarker]
                
                # Apply logic based on direction
                if direction == 'higher_worse':
                    return 1 if value >= cutoff else 0
                elif direction == 'lower_worse':
                    return 1 if value <= cutoff else 0
        
        return pd.NA
    
    def get_ATN_profile(self, df: pd.DataFrame, 
                       method_cols: Optional[Dict[str, str]] = None) -> Tuple[pd.DataFrame, List[str]]:
        """Calculate ATN profiles for all rows in the dataframe."""
        df = df.copy()
        ATN_vars = []
        
        # Calculate A-positive
        df['Apositive'] = df.apply(
            lambda row: self.calculate_Apositive(row, method_cols), 
            axis=1
        ).astype("Int64")
        ATN_vars.append('Apositive')
        
        # Calculate T-positive
        df['Tpositive'] = df.apply(
            lambda row: self.calculate_Tpositive(row, method_cols),
            axis=1
        ).astype("Int64")
        ATN_vars.append('Tpositive')
        
        # Calculate N-positive
        df['Npositive'] = df.apply(
            lambda row: self.calculate_Npositive(row, method_cols),
            axis=1
        ).astype("Int64")
        ATN_vars.append('Npositive')
        
        # Create combined ATN profile string
        df['ATN_profile'] = df.apply(
            lambda row: self._format_ATN_profile(row), axis=1
        )
        ATN_vars.append('ATN_profile')
        
        return df, ATN_vars
    
    def _format_ATN_profile(self, row: pd.Series) -> str:
        """Format ATN profile as string (e.g., 'A+T+N-')."""
        a = row.get('Apositive', pd.NA)
        t = row.get('Tpositive', pd.NA)
        n = row.get('Npositive', pd.NA)
        
        # Use pd.isna() to handle NA values properly
        a_str = '+' if (not pd.isna(a) and a == 1) else ('-' if (not pd.isna(a) and a == 0) else '?')
        t_str = '+' if (not pd.isna(t) and t == 1) else ('-' if (not pd.isna(t) and t == 0) else '?')
        n_str = '+' if (not pd.isna(n) and n == 1) else ('-' if (not pd.isna(n) and n == 0) else '?')
        
        return f'A{a_str}T{t_str}N{n_str}'


def create_method_mapping_from_file(variables_df: pd.DataFrame, 
                                     file_code: str) -> Dict[str, str]:
    """
    Create a mapping of variables to methods from the ADNI variables file.
    
    Args:
        variables_df: DataFrame from ADNI_variables_cleanedBB.xlsx
        file_code: File code to filter for
    
    Returns:
        Dictionary mapping variable names to methods
    """
    # Filter for specific file_code
    file_vars = variables_df[variables_df['file_code'] == file_code].copy()
    
    # Create mapping
    mapping = {}
    for _, row in file_vars.iterrows():
        var = row['variable_code']
        method = row['Unit']
        
        # If method is NaN, use 'unknown'
        if pd.isna(method):
            method = 'unknown'
        
        mapping[var] = method
    
    return mapping


def apply_ATN_profile_to_dataset(dataset_df: pd.DataFrame,
                                  variables_df: pd.DataFrame,
                                  file_code: str,
                                  config: Optional[ATNConfig] = None) -> Tuple[pd.DataFrame, List[str]]:
    """
    Complete workflow to apply ATN profiling to a dataset.
    
    Args:
        dataset_df: The actual measurement data
        variables_df: DataFrame from ADNI_variables_cleanedBB.xlsx with metadata
        file_code: File code identifier
        config: Optional custom ATNConfig
    
    Returns:
        Tuple of (dataset with ATN profiles, list of ATN variable names)
    """
    # Create method mapping from variables file
    method_mapping = create_method_mapping_from_file(variables_df, file_code)
    
    # Initialize calculator
    calculator = ATNProfileCalculator(config)
    
    # Add method columns to dataset
    dataset_df = calculator.add_method_column(dataset_df, method_mapping)
    
    # Calculate ATN profiles
    result_df, atn_vars = calculator.get_ATN_profile(dataset_df)
    
    return result_df, atn_vars


if __name__ == "__main__":
    print("ATN Profile System (Updated for ADNI_variables_cleanedBB.xlsx)")
    print("\nReady to use!")