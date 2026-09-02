# MICO-PMInet reproducibility package

Core code and data required to reproduce the computational workflow in:

> MICO-PMInet: An Interpretable Multi-organ Incomplete-input Cooperative Learning Framework Based on ATR-FTIR Spectra for Robust Postmortem Interval Estimation

This repository intentionally excludes figure-generation code. It covers raw spectral preprocessing, animal-level partitioning, single-organ PLS baselines, the 16-model MICO-PMInet comparison, validation-based beta selection, cohort-separated organ-retention testing, statistical analysis, checkpoint inference, AWA-weight extraction, and SHAP aggregation.

The manuscript model is **OSB-WMHA-AWA-MORM with beta 0.2**. Training defaults to the article-proof protocol: 48/16 stratified development splits, the two-term loss in Equation 17, at most 1,000 epochs, and early-stopping patience 200. The supplied checkpoint, archived metrics, AWA outputs, and SHAP cache remain published reference artifacts; see `ARTICLE_ALIGNMENT.md` for their provenance.

## Quick start

Python 3.13.5 was reported in the manuscript. Python 3.12 or 3.13 is supported here.

On a new Windows computer, install Anaconda or Miniconda, open **Anaconda Prompt**, enter the downloaded repository directory, and run:

```bat
setup_conda_windows.bat
```

On a new macOS or Linux computer:

```bash
bash setup_conda_unix.sh
```

Both scripts run the equivalent manual workflow:

```bash
conda env create -f environment.yml
conda activate mico-pminet
python -m pip install --no-deps -e .
python -m pytest -q
python run.py verify
```

The scripts locate the repository directory automatically and perform installation plus release verification. The Conda environment contains its own Python and packages. On Windows, no standalone Python installation, PowerShell execution-policy change, `.venv`, or `Activate.ps1` command is required. If the environment already exists, reuse it with `conda activate mico-pminet` or update it with `conda env update -n mico-pminet -f environment.yml`.

## Reproduction commands

Fast structural and numerical verification using the released processed data, final checkpoint, and archived SHAP values:

```bash
python run.py verify
```

Reproduce validation-based beta selection for the manuscript OSB-WMHA-AWA-MORM model from the complete 16-model, eight-beta summary table:

```bash
python run.py select
```

Evaluate the supplied OSB-WMHA-AWA-MORM fold checkpoint separately on modeled-time, unseen-time, and combined held-out cohorts:

```bash
python run.py evaluate
```

Run the split-correlated component analysis:

```bash
python run.py stats
```

Rebuild all processed datasets from the 6,336 raw spectra:

```bash
python run.py preprocess
```

Run all eight single-organ PLS baselines:

```bash
python run.py pls
```

Run a short end-to-end training test (not for paper numbers):

```bash
python run.py train --profile smoke
```

Recompute the manuscript Type II ANOVA:

```bash
python run.py anova --fold-results reference/published_fold_metrics.csv
```

Export the actual learned AWA aggregation weights for the manuscript checkpoint. These are distinct from SHAP values:

```bash
python run.py awa-weights
```

Recompute the full 16-model, eight-beta, ten-split article experiment:

```bash
python run.py train --profile paper --protocol manuscript_protocol
python run.py select --fold-results results/deep/fold_metrics_all_betas.csv
python run.py stats --fold-results results/deep/fold_metrics.csv
```

Equivalently, the complete sequence can be launched with `python run.py all --profile paper --protocol manuscript_protocol`. This includes the full training workload.

The full experiment trains 1,280 neural networks. Wall-clock time and memory use depend on the hardware and are not estimated by this package. The implementation is CPU/GPU portable, but small floating-point differences across hardware are expected. Compare results with `reference/published_model_metrics.csv` using the supplied verifier and stated tolerances rather than requiring bitwise equality.

## Repository layout

```text
configs/                  Executed and manuscript-stated protocols
data/raw/                 Compressed 6,336-spectrum input table
data/processed/           Released 64/24-rat model inputs
reference/                Archived beta grid, fold metrics, checkpoints, and fixed outputs
src/mico_pminet/          Platform-independent core implementation
tests/                    Fast invariants and model smoke tests
results/                  Generated outputs (ignored by Git)
```

`ARTICLE_ALIGNMENT.md` maps the proof methods to the executable code and distinguishes the article protocol from archived numerical artifacts.
`VALIDATION.md` records the release checks actually executed. A `Dockerfile` and GitHub Actions workflow provide an additional platform-independent verification path.
The GitHub Actions matrix creates the Conda environment from scratch and runs tests plus release verification on Windows, Ubuntu, and macOS.
`CHECKSUMS.sha256` records the released data, checkpoint, and reference-result hashes.
`MODEL_ARCHITECTURE.md` gives the exact layer-by-layer implementation and parameter counts.
`reference/README.md` identifies the released metrics, model checkpoint, and fixed numerical outputs.

## Exact analysis definitions

- The split unit is an individual rat, never an individual organ or repeated scan.
- The fixed modeling partition contains 64 training rats and 16 held-out rats at 0.5, 1, 2, 3, 6, 12, 18, and 24 h.
- Eight additional rats at 20 min, 4 h, 11 h, and 20 h form the unseen-time subset.
- To match the supplied reported results, `Test_*` aggregates predictions across all organ-retention levels (1-8 organs) and across both held-out subsets. `reserveorgan-8-*` is the complete eight-organ result.
- Each stratified repeated development split contains 48 training and 16 validation rats, with two validation rats at every modeled PMI.
- MORM creates one complete input plus two random realizations for each missing level 1-7 during training (720 inputs), and ten fixed realizations per missing level for validation (1,136 inputs).
- Beta is selected separately for each architecture using validation RMSE across ten stratified repeated holdouts; the released manuscript model is OSB-WMHA-AWA-MORM with beta 0.2.
- `python run.py evaluate` reports the 16 modeled-time test rats and eight unseen-time test rats separately.

## Training protocol files

`configs/manuscript_protocol.json` is the default and follows the proof: up to 1,000 epochs, patience 200, 48/16 development splits, and the two losses shown in Equation 17 (`gamma=0`). `configs/reported_results.json` is non-default and retained only to document the provenance of supplied archived artifacts. It must not be presented as the article methods protocol.

## Data and ethical scope

The included tables contain animal ATR-FTIR spectra and experimental labels only. No human participant or personally identifying data are included. Animal procedures are described in the article under approval IACUC-CQMU-2025-0995.

## Citation and license

Use `CITATION.cff` when citing this repository. Source code is released under the MIT License; that license does not automatically cover the spectral data. Confirm redistribution authorization and add an explicit data license before making the data public.

For a detailed Chinese walkthrough, see [QUICKSTART_ZH.md](QUICKSTART_ZH.md).
