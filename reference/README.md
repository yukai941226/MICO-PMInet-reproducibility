# Reference-artifact index

These files reproduce the numerical workflow reported for OSB-WMHA-AWA-MORM with beta 0.2.

They are retained as submitted/published numerical reference artifacts. They were not regenerated during the article-protocol alignment in v1.3.0. The executable default now follows the proof's 48/16 splits, 1,000 epochs, patience 200, and two-term loss. See `../ARTICLE_ALIGNMENT.md` before making provenance claims.

- `published_beta_summary.csv`: validation columns for all 16 models × eight beta values, each summarized over ten internal splits. `python run.py select` uses it to reproduce beta 0.2 for the manuscript model.
- `published_fold_metrics.csv`: ten selected-beta fold records for each of 16 models (160 rows). This reproduces manuscript summaries and supplies records for `python run.py anova` and `python run.py stats`.
- `published_model_metrics.csv`: manuscript-level model summaries.
- `mico_pminet_fold5.pt`: one OSB-WMHA-AWA-MORM beta-0.2 checkpoint used for portable inference checks and SHAP recomputation.
- `mico_pminet_fold5_evaluation/`: cohort-separated descriptive evaluation of the supplied checkpoint. It is not a ten-fold performance estimate.
- `published_shap_values.npz` and `published_organ_importance.csv`: manuscript-model SHAP cache and its organ summary.
- `reported_osb_fold5_awa_weights/`: actual AWA weights from the supplied fold-5 checkpoint.
- `reported_osb_all_folds_awa_weights/`: checkpoint-level AWA summaries computed from all ten archived OSB checkpoints. The complete experiment command can regenerate those checkpoints.

The fold-5 checkpoint provides an executable example and numerical integrity check. Paper-level performance estimates remain the ten-split summaries.
