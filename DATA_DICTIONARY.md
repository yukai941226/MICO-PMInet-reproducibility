# Data dictionary

## Raw spectra

`data/raw/all_data_acquire.csv.gz` contains 6,336 spectra (704 organ samples x 9 technical spectra). Metadata columns are followed by wavenumber columns in cm⁻¹.

| Column | Meaning |
|---|---|
| `label_all` | Original hyphen-delimited spectrum identifier |
| `label_1` | PMI label (`20min` or hours) |
| `label_2` | Organ code |
| `label_3` | Sex code |
| `label_4` | Animal number within PMI |
| `label_5`, `label_6` | Technical acquisition identifiers |
| numeric columns | ATR-FTIR absorbance at the named wavenumber |

## Processed data

The released processed tables contain one row per animal-organ pair after averaging nine spectra. `train_dataset` contains 64 animals at modeled PMIs. `test_modeled` contains 16 held-out animals at modeled PMIs. `test_unseen` contains eight animals at non-modeled PMIs. `test_dataset` is the union of the two test subsets and is retained for exact comparison with the reported analysis code.

The organ order is Brain, Heart, Kidney, Liver, Lung, Muscle, Spleen, and VH. Spectral features comprise 467 columns from approximately 900.59 to 1799.26 cm⁻¹.

## Cohort terminology

`train_dataset` contains the 64-rat model-development cohort. Internal validation rows are created only from this cohort during repeated stratified splitting. `test_modeled` contains 16 held-out rats at modeled PMI values. `test_unseen` contains eight rats at non-modeled PMI values. These held-out cohorts are reported separately by `python run.py evaluate`.

The archived `Test_*` fields aggregate predictions across retention levels 1-8 on the combined held-out cohort. They are not complete-input metrics. Complete-input metrics use the `reserveorgan-8-*` prefix.

## Batch metadata

The released spectral tables do not encode acquisition batch. `data/metadata/batch_metadata_template.csv` is provided for author completion from laboratory or instrument records; no batch value should be inferred from the sample identifier.
