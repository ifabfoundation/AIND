"""
Post-generation package for reversing data transformations and noise injection.

This package provides tools to:
1. Reverse normalization and encoding transformations applied during data preparation
2. Add differential privacy noise to synthetic data for privacy enhancement
"""

from .denormalize import (
    denormalize_fixed_scale,
    denormalize_dataset_minmax,
    denormalize_dataset_volumes,
    load_normalization_settings
)

from .dummy_to_categorical import (
    dummies_to_categorical,
    multiple_dummies_to_categorical,
    auto_detect_dummies,
    label_decode,
    binary_to_categorical
)

from .reverse_transformations import reverse_all_transformations

# Import noise injection modules
from .noise_injection import (
    PostGenerationNoiseInjector,
    LongitudinalNoiseInjector,
    add_post_generation_noise
)

# Try to import datalake functions (optional dependency)
try:
    from .reverse_from_datalake import (
        load_from_datalake,
        reverse_datalake_data
    )
    _has_datalake = True
except ImportError:
    _has_datalake = False
    load_from_datalake = None
    reverse_datalake_data = None

# Try to import noise injection datalake functions (optional dependency)
try:
    from .noise_injection_from_datalake import inject_noise_datalake_data
    _has_noise_datalake = True
except ImportError:
    _has_noise_datalake = False
    inject_noise_datalake_data = None

__all__ = [
    # Denormalization functions
    'denormalize_fixed_scale',
    'denormalize_dataset_minmax',
    'denormalize_dataset_volumes',
    'load_normalization_settings',
    # Dummy to categorical functions
    'dummies_to_categorical',
    'multiple_dummies_to_categorical',
    'auto_detect_dummies',
    'label_decode',
    'binary_to_categorical',
    # Reverse transformations
    'reverse_all_transformations',
    # Noise injection classes and functions
    'PostGenerationNoiseInjector',
    'LongitudinalNoiseInjector',
    'add_post_generation_noise'
]

# Add datalake functions if available
if _has_datalake:
    __all__.extend(['load_from_datalake', 'reverse_datalake_data'])

if _has_noise_datalake:
    __all__.extend(['inject_noise_datalake_data'])

__version__ = '1.1.0'
