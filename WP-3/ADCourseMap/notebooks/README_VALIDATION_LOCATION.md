# Synthetic Data Validation Scripts

All validation scripts and documentation have been moved to:

📁 **`validation/`**

## Quick Links

- 📚 **Main README**: [`validation/README_VALIDATION.md`](validation/README_VALIDATION.md)
- 📖 **Usage Guide**: [`validation/USAGE_EXAMPLES.md`](validation/USAGE_EXAMPLES.md)
- 🚀 **Quick Start Script**: [`validation/validate_with_preprocessing.py`](validation/validate_with_preprocessing.py)

## Quick Start

```bash
cd validation
python validate_with_preprocessing.py
```

## Files in `validation/` Folder

### Main Scripts
- `validate_with_preprocessing.py` - **START HERE**: All-in-one validation with preprocessing
- `pre_validation_check.py` - Fast pre-validation checks
- `validate_data.py` - Comprehensive validation
- `preprocess_synthetic.py` - Data alignment utilities

### Examples
- `example_datalake_usage.py` - Basic datalake example
- `example_with_preprocessing.py` - Complete preprocessing example

### Documentation
- `README_VALIDATION.md` - Main documentation
- `USAGE_EXAMPLES.md` - Detailed usage guide

## Most Common Use Case

For AIND project with generated cofactors:

```bash
cd validation
python validate_with_preprocessing.py
```

This handles everything automatically:
1. ✅ Loads from datalake
2. ✅ Removes 'generated_' prefix
3. ✅ Aligns datasets
4. ✅ Runs all validations
5. ✅ Generates reports

See [`validation/README_VALIDATION.md`](validation/README_VALIDATION.md) for full documentation.
