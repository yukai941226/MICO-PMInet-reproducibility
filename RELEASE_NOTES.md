# Release notes

## v1.3.0 — 2026-09-02

This release makes the article proof, rather than the archived execution settings, authoritative for the default executable workflow.

- Applies 1800-900 cm-1 selection, SNV, and airPLS to each technical spectrum before averaging the nine corrected spectra, following the author's clarification of the executed workflow.
- Uses ten stratified repeated 48/16 development splits, producing the stated 720 MORM training inputs and 1,136 validation inputs.
- Uses Equation 17's two-term loss with gamma zero.
- Defaults to 1,000 epochs and early-stopping patience 200.
- Makes resumed jobs retain the same run seeds as uninterrupted jobs.
- Adds article-alignment tests and an explicit provenance document for archived numerical artifacts.

The full 1,280-job experiment was not rerun for this packaging update.

## v1.2.1 — 2026-09-01

This packaging update makes Conda the default first-run environment workflow.

- Adds path-independent setup scripts for Windows and macOS/Linux.
- Creates the pinned Python 3.13.5 `mico-pminet` environment from `environment.yml`.
- Runs installation tests and release verification automatically after setup.
- Adds a GitHub Actions Conda matrix for Windows, Ubuntu, and macOS.
- Removes `.venv` and PowerShell activation steps from the recommended workflow.

The scientific code, released data, checkpoint, archived results, and manuscript model are unchanged from v1.2.0.

## v1.2.0 — 2026-08-30

This release reproduces the manuscript workflow with OSB-WMHA-AWA-MORM and beta 0.2 as the default model.

- Includes raw and processed ATR-FTIR tables and fixed animal-level splits.
- Reproduces all eight PLS baselines and the complete 16-model, eight-beta, ten-split experiment.
- Supplies archived beta summaries, fold metrics, manuscript model summaries, and one executable OSB checkpoint.
- Separates modeled-time, unseen-time, and combined held-out cohort evaluation.
- Exports model-internal AWA weights separately from SHAP attribution.
- Provides the exact model architecture, training configurations, automated tests, checksums, Docker support, and Windows/macOS/Linux instructions.
- Provides first-run Conda setup scripts for Windows and macOS/Linux; the scripts install the pinned environment and run tests plus release verification automatically.

Figure-generation code is intentionally excluded; the package contains the core numerical workflow only.
