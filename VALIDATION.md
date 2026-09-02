# Validation record

Validation date: 2026-09-02. Platform: Windows, Conda Python 3.13.5, PyTorch 2.9.1, CPU.

The release package was checked as follows:

- `python -m pytest -q --basetemp <workspace-temp>`: 9 tests passed, including explicit article checks for 48/16 development splits, 720/1,136 MORM expansions, the two-term loss, protocol settings, beta-selection isolation, model shapes, and normalized masked AWA weights.
- `python run.py preprocess`: rebuilt all processed tables from 6,336 raw spectra by applying 1800-900 cm-1 selection, SNV, and airPLS to each technical spectrum and then averaging nine corrected spectra per sample, following the author's clarification.
- Two independent preprocessing runs produced byte-identical deterministic gzip files.
- `python run.py verify --processed-dir data/processed`: passed article-protocol, data-split, tensor-shape, checkpoint-inference, published-reference-table, beta-grid, and SHAP-order checks.
- `python run.py pls`: completed for all eight organs using the author-confirmed preprocessing order and reproduced the submitted single-organ reference values within normal numerical tolerance.
- `python run.py anova --fold-results reference/published_fold_metrics.csv`: completed on the archived 160 selected fold records.
- `python run.py shap`: reproduced the archived organ ranking and feature tables.
- `python run.py train --profile smoke --protocol manuscript_protocol`: completed one OSB-WMHA-AWA-MORM model, beta 0.2, one 48/16 split, and two epochs on CPU.
- `python run.py select`: reproduced beta 0.2 for OSB-WMHA-AWA-MORM from the archived validation summaries.
- `python run.py stats`: completed the split-correlated component analysis; singular random-intercept fits fell back to split-clustered robust covariance.
- `python run.py evaluate`: evaluated the supplied OSB checkpoint on the author-confirmed processed data, separately for modeled-time, unseen-time, and combined held-out cohorts.
- `python run.py awa-weights`: exported actual AWA weights across all ten archived OSB folds and kept them distinct from SHAP attribution.

At the author's request, the full 1,280-run hyperparameter experiment was not rerun during packaging. The proof-aligned implementation, fixed inputs, archived submitted results, published summaries, final checkpoint, and deterministic resume support are included. Archived numerical artifacts are provenance-labeled and are not represented as a fresh run of the corrected default protocol.
