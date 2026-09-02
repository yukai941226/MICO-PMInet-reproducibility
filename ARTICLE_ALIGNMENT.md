# Article-to-code alignment

This release uses the June 2026 article proof as the authoritative description of the executable workflow. The default `manuscript_protocol` implements Sections 2.3 and 2.5.6-2.5.8 and Equations 16-17 as follows.

| Article procedure | Executable implementation |
|---|---|
| Nine technical spectra form one representative spectrum | Following the author's clarification of the executed workflow, each technical spectrum is preprocessed independently and the nine corrected spectra are then averaged |
| 1800-900 cm-1, then SNV and airPLS | `preprocess` applies those operations to each of the 6,336 technical spectra before sample-level averaging |
| Reproducible released tables | Processed CSV gzip timestamps are fixed so identical inputs produce byte-identical files |
| 64 modeling rats split into 48 training and 16 validation rats | Each of ten stratified repeated splits retains two rats per modeled PMI for validation |
| Two MORM training rounds | One complete input plus two random realizations at each missing level 1-7: 48 x 15 = 720 inputs |
| Ten fixed MORM validation rounds | One complete input plus ten fixed realizations at each missing level 1-7: 16 x 71 = 1,136 inputs |
| Joint loss in Equation 17 | `alpha * main_MSE + beta * organ_MSE`; `alpha=1` and beta is searched over the eight stated values |
| AdamW, learning rate 0.001, weight decay 1e-4 | Implemented directly |
| ReduceLROnPlateau | Implemented with factor 0.5 and patience 100 as explicit operational settings not numerically specified in the article |
| Batch size 32, gradient norm 1.0 | Implemented directly |
| Up to 1,000 epochs, early-stopping patience 200 | Implemented directly |
| Ten-fold cross-validation with stratified random sampling | Implemented as ten reproducible stratified repeated 48/16 holdouts, consistent with the explicit sample counts in Section 2.5.6 |
| Beta chosen by minimum validation cross-validation RMSE | Implemented separately for each of the 16 model configurations without using test metrics in the ranking |

## Archived numerical files

The files under `reference/` are the numerical tables, checkpoint, AWA outputs, and SHAP values associated with the submitted results. They are retained as published reference artifacts. The former execution settings are preserved only as `configs/reported_results.json`; that non-default protocol differs from the article proof in validation fraction, maximum epochs, early stopping, and an additional consistency term.

The default code no longer uses that archival protocol for training. Because the full 1,280 training jobs were not rerun while preparing this release, the archived numerical files must not be described as a fresh execution of the corrected article protocol. A complete new execution is launched with:

```bash
python run.py train --profile paper --protocol manuscript_protocol --processed-dir results/processed
```

This distinction preserves both methodological fidelity and provenance of the already reported numerical artifacts.

## Author clarification of preprocessing order

The proof states that nine spectra were used to obtain a representative average spectrum and then describes fingerprint selection, SNV, and airPLS. The author has clarified that the actual executed order was: apply 1800-900 cm-1 selection, SNV, and airPLS independently to all nine technical spectra, then average the nine preprocessed spectra. The code follows this confirmed order because it reproduces the submitted processed data and single-organ PLS results. This clarification should also be stated explicitly in the revised Methods text to remove ambiguity.
