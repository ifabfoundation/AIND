"""
Pre-Validation Check Script
============================

This script performs comprehensive pre-validation checks before running
the full synthetic data validation pipeline.

It verifies:
1. Schema compatibility (columns, types)
2. Sample size appropriateness
3. Longitudinal structure consistency (if applicable)
4. Target distribution similarity
5. Missing data patterns
6. Value range plausibility

Usage:
------
python pre_validation_check.py --real real_data.csv --synthetic synthetic_data.csv --id ID --target DIAGNOSIS

Or import and use programmatically:
from pre_validation_check import PreValidationChecker

checker = PreValidationChecker(real_data, synthetic_data)
checker.run_all_checks()

Author: IFAB
Date: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import argparse
import sys

# Import datalake client (optional - will warn if not available)
try:
    from dl_client import DatalakeClient
    DATALAKE_AVAILABLE = True
except ImportError:
    DATALAKE_AVAILABLE = False
    print("⚠️  Warning: dl_client not found. Datalake loading disabled.")


# ============================================================================
# DATALAKE HELPER FUNCTIONS
# ============================================================================

def load_from_datalake(file_code: str, level: str = 'cleaned_03') -> pd.DataFrame:
    """
    Load data from datalake using dl_client.

    Parameters
    ----------
    file_code : str
        File code to search for (e.g., 'ADNIMERGE' or 'ADNIMERGEsynthetic')
    level : str, default='cleaned_03'
        Data level in the datalake

    Returns
    -------
    pd.DataFrame
        Loaded dataset

    Raises
    ------
    ImportError
        If dl_client is not available
    RuntimeError
        If file not found or loading fails
    """
    if not DATALAKE_AVAILABLE:
        raise ImportError(
            "dl_client is not available. Please install it or use CSV loading instead."
        )

    print(f"  📡 Connecting to datalake...")

    # Initialize client
    client = DatalakeClient()

    # Query for file
    print(f"  🔍 Searching for file_code='{file_code}', level='{level}'...")
    search = client.query_files(
        query={
            'custom.level': level,
            'custom.file_code': file_code
        }
    )

    if not search or 'object_name' not in search:
        raise RuntimeError(
            f"File not found in datalake: file_code='{file_code}', level='{level}'"
        )

    # Download and extract
    print(f"  ⬇️  Downloading {search['object_name']}...")
    zip_files = client.download_file(
        search['object_name'],
        extract_zip=True
    )

    if not zip_files:
        raise RuntimeError("Downloaded file is empty or could not be extracted")

    # Get first file from zip
    file_name = list(zip_files.keys())[0]
    dataset = zip_files[file_name]

    print(f"  ✅ Loaded {file_name} from datalake ({dataset.shape})")

    return dataset


def load_real_and_synthetic_from_datalake(real_file_code: str,
                                          level: str = 'cleaned_03') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load both real and synthetic datasets from datalake.

    Assumes synthetic data has file_code = real_file_code + 'synthetic'
    (e.g., 'ADNIMERGE' → 'ADNIMERGEsynthetic')

    Parameters
    ----------
    real_file_code : str
        File code for real data (e.g., 'ADNIMERGE')
    level : str, default='cleaned_03'
        Data level in the datalake

    Returns
    -------
    tuple of (pd.DataFrame, pd.DataFrame)
        (real_data, synthetic_data)
    """
    print("\n" + "=" * 80)
    print("LOADING DATA FROM DATALAKE")
    print("=" * 80)

    # Load real data
    print("\n📊 Loading REAL data:")
    reald_data_source_file_code = real_file_code + '_for_validation_reversed'
    real_data = load_from_datalake(reald_data_source_file_code, level)

    # Load synthetic data (convention: real_file_code + 'synthetic')
    synthetic_file_code = real_file_code + '_synthetic_reversed_missing_injected'
    print("\n🤖 Loading SYNTHETIC data:")
    synthetic_data = load_from_datalake(synthetic_file_code, level)

    print("\n" + "=" * 80)
    print(f"✅ Successfully loaded both datasets")
    print(f"   Real: {real_data.shape}")
    print(f"   Synthetic: {synthetic_data.shape}")
    print("=" * 80)

    return real_data, synthetic_data


class PreValidationChecker:
    """
    Comprehensive pre-validation checker for synthetic data.

    Performs all necessary checks before running full validation pipeline
    to ensure data quality and compatibility.
    """

    def __init__(self,
                 real_data: pd.DataFrame,
                 synthetic_data: pd.DataFrame,
                 id_var: Optional[str] = None,
                 target_var: Optional[str] = None,
                 time_var: Optional[str] = None):
        """
        Initialize pre-validation checker.

        Parameters
        ----------
        real_data : pd.DataFrame
            Real dataset
        synthetic_data : pd.DataFrame
            Synthetic dataset
        id_var : str, optional
            Name of ID column (for longitudinal data)
        target_var : str, optional
            Name of target variable (for utility checks)
        time_var : str, optional
            Name of time variable (for longitudinal data)
        """
        self.real_data = real_data.copy()
        self.synthetic_data = synthetic_data.copy()
        self.id_var = id_var
        self.target_var = target_var
        self.time_var = time_var

        self.issues = []  # Critical issues (must fix)
        self.warnings = []  # Warnings (recommended to fix)
        self.passed_checks = []  # List of passed checks

    def run_all_checks(self) -> bool:
        """
        Run all pre-validation checks.

        Returns
        -------
        bool
            True if all critical checks passed (can proceed to validation)
            False if critical issues found (must fix before validation)
        """
        print("=" * 80)
        print("PRE-VALIDATION CHECKS FOR SYNTHETIC DATA")
        print("=" * 80)
        print(f"\nReal data shape: {self.real_data.shape}")
        print(f"Synthetic data shape: {self.synthetic_data.shape}")

        # Run all checks
        self._check_schema()
        self._check_sample_size()
        self._check_longitudinal_structure()
        self._check_target_distribution()
        self._check_missing_data()
        self._check_value_ranges()
        self._check_data_types()

        # Print summary
        self._print_summary()

        # Return True if no critical issues
        return len(self.issues) == 0

    def _check_schema(self):
        """
        Check 1: Schema compatibility.

        Verifies that both datasets have:
        - Same columns
        - Same column order (recommended but not critical)
        """
        print("\n" + "=" * 80)
        print("CHECK 1: SCHEMA COMPATIBILITY")
        print("=" * 80)

        real_cols = set(self.real_data.columns)
        synth_cols = set(self.synthetic_data.columns)

        # Check for missing/extra columns
        missing_in_synth = real_cols - synth_cols
        extra_in_synth = synth_cols - real_cols

        if missing_in_synth:
            self.issues.append(
                f"Missing columns in synthetic data: {missing_in_synth}"
            )
            print(f"  ❌ Missing in synthetic: {missing_in_synth}")

        if extra_in_synth:
            self.warnings.append(
                f"Extra columns in synthetic data: {extra_in_synth}"
            )
            print(f"  ⚠️  Extra in synthetic: {extra_in_synth}")

        if not missing_in_synth and not extra_in_synth:
            print("  ✅ All columns present in both datasets")
            self.passed_checks.append("Schema: columns match")

            # Check column order (warning only)
            if list(self.real_data.columns) != list(self.synthetic_data.columns):
                self.warnings.append("Column order differs (not critical)")
                print("  ⚠️  Column order differs (not critical)")
            else:
                print("  ✅ Column order matches")
                self.passed_checks.append("Schema: column order matches")

    def _check_sample_size(self):
        """
        Check 2: Sample size appropriateness.

        Verifies that synthetic dataset size is reasonable:
        - Recommended: 0.5x to 2.0x real data size
        - Acceptable: 0.3x to 3.0x
        - Problematic: <0.3x or >3.0x
        """
        print("\n" + "=" * 80)
        print("CHECK 2: SAMPLE SIZE")
        print("=" * 80)

        n_real = len(self.real_data)
        n_synth = len(self.synthetic_data)
        ratio = n_synth / n_real if n_real > 0 else 0

        print(f"  Real data: {n_real:,} rows")
        print(f"  Synthetic data: {n_synth:,} rows")
        print(f"  Ratio (synth/real): {ratio:.2f}")

        # Categorize ratio
        if ratio < 0.3:
            self.issues.append(
                f"Synthetic dataset too small (ratio={ratio:.2f}, need >0.3)"
            )
            print(f"  ❌ Synthetic dataset too small for reliable validation")
        elif ratio < 0.5:
            self.warnings.append(
                f"Synthetic dataset small (ratio={ratio:.2f}, recommended >0.5)"
            )
            print(f"  ⚠️  Synthetic dataset on the small side")
        elif ratio > 3.0:
            self.warnings.append(
                f"Synthetic dataset very large (ratio={ratio:.2f})"
            )
            print(f"  ⚠️  Synthetic dataset unusually large")
        elif ratio > 2.0:
            self.warnings.append(
                f"Synthetic dataset large (ratio={ratio:.2f}, recommended <2.0)"
            )
            print(f"  ⚠️  Synthetic dataset larger than recommended")
        else:
            print(f"  ✅ Sample size ratio is good ({ratio:.2f})")
            self.passed_checks.append(f"Sample size: ratio {ratio:.2f} is appropriate")

    def _check_longitudinal_structure(self):
        """
        Check 3: Longitudinal structure consistency.

        Only applicable if id_var is provided.
        Verifies:
        - Number of unique patients/subjects
        - Average visits per patient
        - Distribution of visit counts
        """
        if not self.id_var or self.id_var not in self.real_data.columns:
            print("\n" + "=" * 80)
            print("CHECK 3: LONGITUDINAL STRUCTURE")
            print("=" * 80)
            print("  ⊘ Skipped (no ID variable specified)")
            return

        print("\n" + "=" * 80)
        print("CHECK 3: LONGITUDINAL STRUCTURE")
        print("=" * 80)

        # Count unique patients
        n_patients_real = self.real_data[self.id_var].nunique()
        n_patients_synth = self.synthetic_data[self.id_var].nunique()

        # Calculate visits per patient
        visits_real = self.real_data.groupby(self.id_var).size()
        visits_synth = self.synthetic_data.groupby(self.id_var).size()

        mean_visits_real = visits_real.mean()
        mean_visits_synth = visits_synth.mean()

        print(f"\n  Real data:")
        print(f"    - Patients: {n_patients_real:,}")
        print(f"    - Avg visits/patient: {mean_visits_real:.1f}")
        print(f"    - Min-Max visits: {visits_real.min()}-{visits_real.max()}")

        print(f"\n  Synthetic data:")
        print(f"    - Patients: {n_patients_synth:,}")
        print(f"    - Avg visits/patient: {mean_visits_synth:.1f}")
        print(f"    - Min-Max visits: {visits_synth.min()}-{visits_synth.max()}")

        # Check patient count difference
        patient_diff_ratio = abs(n_patients_real - n_patients_synth) / n_patients_real
        if patient_diff_ratio > 0.3:
            self.warnings.append(
                f"Patient count differs by {patient_diff_ratio:.1%} (>30%)"
            )
            print(f"\n  ⚠️  Patient count differs by {patient_diff_ratio:.1%}")
        else:
            print(f"\n  ✅ Patient count is similar (diff: {patient_diff_ratio:.1%})")
            self.passed_checks.append("Longitudinal: similar patient count")

        # Check visits per patient difference
        visits_diff_ratio = abs(mean_visits_real - mean_visits_synth) / mean_visits_real
        if visits_diff_ratio > 0.4:
            self.warnings.append(
                f"Avg visits/patient differs by {visits_diff_ratio:.1%} (>40%)"
            )
            print(f"  ⚠️  Avg visits/patient differs by {visits_diff_ratio:.1%}")
        else:
            print(f"  ✅ Avg visits/patient is similar (diff: {visits_diff_ratio:.1%})")
            self.passed_checks.append("Longitudinal: similar visits/patient")

        # Check if time variable exists
        if self.time_var and self.time_var in self.real_data.columns:
            print(f"\n  Time variable '{self.time_var}' found")

            time_range_real = (self.real_data[self.time_var].min(),
                              self.real_data[self.time_var].max())
            time_range_synth = (self.synthetic_data[self.time_var].min(),
                               self.synthetic_data[self.time_var].max())

            print(f"    Real time range: {time_range_real[0]:.1f} to {time_range_real[1]:.1f}")
            print(f"    Synth time range: {time_range_synth[0]:.1f} to {time_range_synth[1]:.1f}")

    def _check_target_distribution(self):
        """
        Check 4: Target distribution similarity.

        Only applicable if target_var is provided.
        Verifies that class distributions are similar (important for TSTR).
        """
        if not self.target_var or self.target_var not in self.real_data.columns:
            print("\n" + "=" * 80)
            print("CHECK 4: TARGET DISTRIBUTION")
            print("=" * 80)
            print("  ⊘ Skipped (no target variable specified)")
            return

        print("\n" + "=" * 80)
        print("CHECK 4: TARGET DISTRIBUTION")
        print("=" * 80)

        # Calculate distributions
        real_dist = self.real_data[self.target_var].value_counts(normalize=True).sort_index()
        synth_dist = self.synthetic_data[self.target_var].value_counts(normalize=True).sort_index()

        print(f"\n  Class distribution comparison:")
        print(f"  {'Class':<15} {'Real':>10} {'Synthetic':>10} {'Diff':>10} {'Status':>10}")
        print("  " + "-" * 60)

        max_diff = 0
        for class_label in real_dist.index:
            real_prop = real_dist[class_label]
            synth_prop = synth_dist.get(class_label, 0)
            diff = abs(real_prop - synth_prop)
            max_diff = max(max_diff, diff)

            if diff > 0.15:
                status = "⚠️  HIGH"
            elif diff > 0.1:
                status = "⚠️  MED"
            else:
                status = "✅ OK"

            print(f"  {str(class_label):<15} {real_prop:>9.1%} {synth_prop:>10.1%} "
                  f"{diff:>9.1%} {status:>10}")

        # Check for missing classes in synthetic
        missing_classes = set(real_dist.index) - set(synth_dist.index)
        extra_classes = set(synth_dist.index) - set(real_dist.index)

        if missing_classes:
            self.issues.append(
                f"Missing classes in synthetic: {missing_classes}"
            )
            print(f"\n  ❌ Missing classes in synthetic: {missing_classes}")

        if extra_classes:
            self.warnings.append(
                f"Extra classes in synthetic: {extra_classes}"
            )
            print(f"\n  ⚠️  Extra classes in synthetic: {extra_classes}")

        # Overall assessment
        if max_diff > 0.2:
            self.warnings.append(
                f"Target distribution differs significantly (max diff: {max_diff:.1%})"
            )
            print(f"\n  ⚠️  Target distribution differs significantly")
        elif max_diff > 0.1:
            self.warnings.append(
                f"Target distribution differs moderately (max diff: {max_diff:.1%})"
            )
            print(f"\n  ⚠️  Target distribution differs moderately")
        else:
            print(f"\n  ✅ Target distribution is similar (max diff: {max_diff:.1%})")
            self.passed_checks.append(f"Target: distribution similar (max diff {max_diff:.1%})")

    def _check_missing_data(self):
        """
        Check 5: Missing data patterns.

        Verifies that missing data patterns are similar between real and synthetic.
        Large differences may indicate model issues.
        """
        print("\n" + "=" * 80)
        print("CHECK 5: MISSING DATA PATTERNS")
        print("=" * 80)

        # Calculate missing percentages
        real_missing = self.real_data.isnull().mean() * 100
        synth_missing = self.synthetic_data.isnull().mean() * 100

        # Find columns with significant differences
        diff = (real_missing - synth_missing).abs()
        problematic_cols = diff[diff > 15].sort_values(ascending=False)

        if len(problematic_cols) > 0:
            print(f"\n  ⚠️  Columns with large missing data differences (>15%):")
            print(f"  {'Column':<25} {'Real':>10} {'Synthetic':>10} {'Diff':>10}")
            print("  " + "-" * 60)

            for col in problematic_cols.index[:10]:  # Show top 10
                print(f"  {col:<25} {real_missing[col]:>9.1f}% {synth_missing[col]:>9.1f}% "
                      f"{diff[col]:>9.1f}%")

            self.warnings.append(
                f"{len(problematic_cols)} columns have large missing data differences"
            )
        else:
            print("  ✅ Missing data patterns are similar across all columns")
            self.passed_checks.append("Missing data: patterns are similar")

        # Overall missing data summary
        print(f"\n  Overall missing data:")
        print(f"    Real: {real_missing.mean():.1f}% (avg across columns)")
        print(f"    Synthetic: {synth_missing.mean():.1f}% (avg across columns)")

    def _check_value_ranges(self):
        """
        Check 6: Value range plausibility.

        Verifies that synthetic values are within plausible ranges
        (not too far outside real data ranges).
        """
        print("\n" + "=" * 80)
        print("CHECK 6: VALUE RANGES")
        print("=" * 80)

        # Get numeric columns from real data
        real_numeric_cols = set(self.real_data.select_dtypes(include=[np.number]).columns)

        # Get numeric columns from synthetic data
        synth_numeric_cols = set(self.synthetic_data.select_dtypes(include=[np.number]).columns)

        # Use only COMMON numeric columns (present in both datasets)
        numeric_cols = list(real_numeric_cols & synth_numeric_cols)

        # Exclude ID and time variables from this check
        exclude = [c for c in [self.id_var, self.time_var] if c is not None]
        numeric_cols = [c for c in numeric_cols if c not in exclude]

        if len(numeric_cols) == 0:
            print("  ⊘ No common numeric columns to check")
            return

        print(f"\n  Checking {len(numeric_cols)} common numeric columns...")

        issues_found = []

        for col in numeric_cols[:15]:  # Check first 15 columns
            real_min = self.real_data[col].min()
            real_max = self.real_data[col].max()
            synth_min = self.synthetic_data[col].min()
            synth_max = self.synthetic_data[col].max()

            real_range = real_max - real_min
            tolerance = 0.2 * real_range  # Allow 20% extension beyond real range

            # Check if synthetic values exceed real range significantly
            issues = []
            if synth_min < real_min - tolerance:
                issues.append(f"min too low ({synth_min:.2f} vs {real_min:.2f})")

            if synth_max > real_max + tolerance:
                issues.append(f"max too high ({synth_max:.2f} vs {real_max:.2f})")

            if issues:
                issue_str = f"{col}: {', '.join(issues)}"
                issues_found.append(issue_str)
                self.warnings.append(f"Value range: {issue_str}")

        if issues_found:
            print(f"\n  ⚠️  Columns with values outside expected range:")
            for issue in issues_found[:10]:  # Show top 10
                print(f"    {issue}")
        else:
            print("  ✅ All numeric values are within plausible ranges")
            self.passed_checks.append("Value ranges: all plausible")

    def _check_data_types(self):
        """
        Check 7: Data type consistency.

        Verifies that columns have the same data types in both datasets.
        """
        print("\n" + "=" * 80)
        print("CHECK 7: DATA TYPE CONSISTENCY")
        print("=" * 80)

        # Get common columns
        common_cols = set(self.real_data.columns) & set(self.synthetic_data.columns)

        type_mismatches = []

        for col in common_cols:
            real_type = self.real_data[col].dtype
            synth_type = self.synthetic_data[col].dtype

            # Check if types are compatible (not necessarily identical)
            if not self._types_compatible(real_type, synth_type):
                type_mismatches.append(
                    f"{col}: Real={real_type}, Synth={synth_type}"
                )

        if type_mismatches:
            print(f"\n  ⚠️  Type mismatches found in {len(type_mismatches)} columns:")
            for mismatch in type_mismatches[:10]:  # Show top 10
                print(f"    {mismatch}")

            self.warnings.append(
                f"{len(type_mismatches)} columns have type mismatches"
            )
        else:
            print("  ✅ All column types are consistent")
            self.passed_checks.append("Data types: all consistent")

    @staticmethod
    def _types_compatible(type1, type2) -> bool:
        """Check if two pandas dtypes are compatible."""
        # Convert to string for comparison
        str1 = str(type1)
        str2 = str(type2)

        # Numeric types are compatible with each other
        numeric_types = ['int', 'float', 'uint']
        if any(t in str1 for t in numeric_types) and any(t in str2 for t in numeric_types):
            return True

        # Object and string types are compatible
        if ('object' in str1 or 'string' in str1) and ('object' in str2 or 'string' in str2):
            return True

        # Otherwise, must be exact match
        return str1 == str2

    def _print_summary(self):
        """Print comprehensive summary of all checks."""
        print("\n" + "=" * 80)
        print("PRE-VALIDATION CHECK SUMMARY")
        print("=" * 80)

        # Count checks
        total_checks = len(self.passed_checks) + len(self.warnings) + len(self.issues)

        print(f"\n  Total checks performed: {total_checks}")
        print(f"  ✅ Passed: {len(self.passed_checks)}")
        print(f"  ⚠️  Warnings: {len(self.warnings)}")
        print(f"  ❌ Critical issues: {len(self.issues)}")

        # Show passed checks
        if self.passed_checks:
            print(f"\n  ✅ PASSED CHECKS ({len(self.passed_checks)}):")
            for check in self.passed_checks:
                print(f"     • {check}")

        # Show warnings
        if self.warnings:
            print(f"\n  ⚠️  WARNINGS ({len(self.warnings)}):")
            print("  These are not critical but should be reviewed:")
            for warning in self.warnings:
                print(f"     • {warning}")

        # Show critical issues
        if self.issues:
            print(f"\n  ❌ CRITICAL ISSUES ({len(self.issues)}):")
            print("  These MUST be fixed before validation:")
            for issue in self.issues:
                print(f"     • {issue}")

        # Final recommendation
        print("\n" + "=" * 80)
        if len(self.issues) == 0:
            if len(self.warnings) == 0:
                print("✅ ALL CHECKS PASSED!")
                print("   Your data is ready for full validation.")
            else:
                print("✅ READY FOR VALIDATION (with warnings)")
                print("   You can proceed, but review warnings to improve quality.")
        else:
            print("❌ NOT READY FOR VALIDATION")
            print("   Please fix critical issues before proceeding.")
        print("=" * 80)

    def save_report(self, filename: str = 'pre_validation_report.txt'):
        """
        Save pre-validation report to text file.

        Parameters
        ----------
        filename : str
            Output filename
        """
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("PRE-VALIDATION CHECK REPORT\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Real data shape: {self.real_data.shape}\n")
            f.write(f"Synthetic data shape: {self.synthetic_data.shape}\n\n")

            # Passed checks
            f.write(f"PASSED CHECKS ({len(self.passed_checks)}):\n")
            f.write("-" * 80 + "\n")
            for check in self.passed_checks:
                f.write(f"  ✅ {check}\n")
            f.write("\n")

            # Warnings
            f.write(f"WARNINGS ({len(self.warnings)}):\n")
            f.write("-" * 80 + "\n")
            if self.warnings:
                for warning in self.warnings:
                    f.write(f"  ⚠️  {warning}\n")
            else:
                f.write("  None\n")
            f.write("\n")

            # Issues
            f.write(f"CRITICAL ISSUES ({len(self.issues)}):\n")
            f.write("-" * 80 + "\n")
            if self.issues:
                for issue in self.issues:
                    f.write(f"  ❌ {issue}\n")
            else:
                f.write("  None\n")
            f.write("\n")

            # Conclusion
            f.write("=" * 80 + "\n")
            f.write("CONCLUSION\n")
            f.write("=" * 80 + "\n")
            if len(self.issues) == 0:
                f.write("✅ Data is ready for full validation\n")
            else:
                f.write("❌ Please fix critical issues before validation\n")

        print(f"\n✅ Pre-validation report saved to: {filename}")


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description='Pre-validation check for synthetic data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Load from CSV files
  python pre_validation_check.py --real real.csv --synthetic synth.csv

  # Load from datalake (using file_code)
  python pre_validation_check.py --datalake --file-code ADNIMERGE --level cleaned_03

  # With ID and target variables
  python pre_validation_check.py --real real.csv --synthetic synth.csv --id PTID --target DIAGNOSIS

  # Datalake with all variables
  python pre_validation_check.py --datalake --file-code ADNIMERGE --id PTID --target DIAGNOSIS --time TIME
        """
    )

    # Data source: either CSV or datalake
    parser.add_argument('--datalake', action='store_true',
                       help='Load data from datalake instead of CSV files')
    parser.add_argument('--file-code', default=None,
                       help='File code for datalake loading (e.g., ADNIMERGE). Synthetic will be {file_code}synthetic')
    parser.add_argument('--level', default='cleaned_03',
                       help='Datalake level (default: cleaned_03)')

    # CSV file paths (used if --datalake is not set)
    parser.add_argument('--real', '-r', default=None,
                       help='Path to real data CSV file (ignored if --datalake is used)')
    parser.add_argument('--synthetic', '-s', default=None,
                       help='Path to synthetic data CSV file (ignored if --datalake is used)')

    # Column specifications
    parser.add_argument('--id', '-i', default=None,
                       help='Name of ID column (for longitudinal data)')
    parser.add_argument('--target', '-t', default=None,
                       help='Name of target variable column (for TSTR utility test)')
    parser.add_argument('--time', default=None,
                       help='Name of time variable column (for longitudinal data)')

    # Output
    parser.add_argument('--output', '-o', default='pre_validation_report.txt',
                       help='Output report filename (default: pre_validation_report.txt)')

    args = parser.parse_args()

    # Validate arguments
    if args.datalake:
        if not args.file_code:
            print("❌ Error: --file-code is required when using --datalake")
            sys.exit(1)
        if not DATALAKE_AVAILABLE:
            print("❌ Error: dl_client is not available. Cannot use --datalake option.")
            print("   Install dl_client or use CSV loading instead.")
            sys.exit(1)
    else:
        if not args.real or not args.synthetic:
            print("❌ Error: --real and --synthetic are required when not using --datalake")
            sys.exit(1)

    # Load data
    try:
        if args.datalake:
            # Load from datalake
            real_data, synthetic_data = load_real_and_synthetic_from_datalake(
                args.file_code,
                args.level
            )
        else:
            # Load from CSV files
            print("\nLoading data from CSV files...")
            real_data = pd.read_csv(args.real)
            print(f"  ✅ Loaded real data: {args.real} ({real_data.shape})")

            synthetic_data = pd.read_csv(args.synthetic)
            print(f"  ✅ Loaded synthetic data: {args.synthetic} ({synthetic_data.shape})")

    except Exception as e:
        print(f"\n❌ Error loading data: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Run checks
    checker = PreValidationChecker(
        real_data=real_data,
        synthetic_data=synthetic_data,
        id_var=args.id,
        target_var=args.target,
        time_var=args.time
    )

    passed = checker.run_all_checks()

    # Save report
    checker.save_report(args.output)

    # Exit with appropriate code
    sys.exit(0 if passed else 1)


if __name__ == '__main__':
    main()
