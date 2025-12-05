@echo off
::python post-generation\noise_injection_from_datalake.py --real-code ADNIMERGE_for_validation --synthetic-code ADNIMERGE_synthetic --epsilon 0.5
::python post-generation\reverse_from_datalake.py --file-code ADNIMERGE_synthetic --prefix synthetic/reversed
::python post-generation\reverse_from_datalake.py --file-code ADNIMERGE_for_validation --prefix validation/reversed
python post-generation\reverse_from_datalake.py --file-code ADNIMERGE_synthetic_utility --prefix synthetic/utility/reversed
python post-generation\reverse_from_datalake.py --file-code ADNIMERGE_to_test_utility --prefix validation/utility/reversed
::python post-generation\residual_par_correction.py --file-code-real ADNIMERGE_for_validation_reversed --file-code-leaspy ADNIMERGE_synthetic_reversed --baseline-match-feats MMSE
::python post-generation\inject_missing_values.py --file-code-real ADNIMERGE_for_validation_reversed --file-code-gen ADNIMERGE_synthetic_reversed --prefix synthetic/missing_injected
pause