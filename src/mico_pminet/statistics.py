from __future__ import annotations

import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import shapiro
from statsmodels.formula.api import ols
from statsmodels.formula.api import mixedlm
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.multitest import multipletests


FACTOR_COLUMNS = {
    "branch": "Branch type",
    "attention": "Inter-organ interaction",
    "aggregation": "Aggregation strategy",
    "training": "Training strategy",
}


def add_components(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    parts = output["model_name"].str.split("-", expand=True)
    if parts.shape[1] < 4:
        raise ValueError("model_name must encode four components")
    output["branch"] = parts[0]
    output["attention"] = parts[1]
    output["aggregation"] = parts[2]
    output["training"] = parts[3]
    return output


def run_anova(fold_results: Path, output_dir: Path) -> None:
    """Legacy Type II ANOVA retained only to reproduce the submitted analysis."""
    output_dir.mkdir(parents=True, exist_ok=True)
    data = add_components(pd.read_csv(fold_results))
    metric_columns = [
        column
        for column in data.columns
        if (column.endswith("RMSE") or column.endswith("R2"))
        and column.startswith(("Train_", "Val_", "Test_", "reserveorgan-"))
    ]
    factors = list(FACTOR_COLUMNS)
    main_terms = [f"C({factor})" for factor in factors]
    interaction_terms = [
        f"C({left}):C({right})" for left, right in combinations(factors, 2)
    ]
    formula_terms = main_terms + interaction_terms
    rows: list[dict[str, float | str]] = []

    for metric in metric_columns:
        subset = data[[metric, *factors]].dropna().copy()
        formula = f"Q('{metric}') ~ " + " + ".join(formula_terms)
        model = ols(formula, data=subset).fit()
        table = anova_lm(model, typ=2)
        total_sum = table["sum_sq"].sum()
        residual = table.loc["Residual", "sum_sq"]
        for term, values in table.drop(index="Residual").iterrows():
            sum_sq = float(values["sum_sq"])
            rows.append(
                {
                    "metric": metric,
                    "term": term,
                    "F": float(values["F"]),
                    "p": float(values["PR(>F)"]),
                    "eta_squared": sum_sq / total_sum,
                    "partial_eta_squared": sum_sq / (sum_sq + residual),
                }
            )

    results = pd.DataFrame(rows)
    rejected, q_values, _, _ = multipletests(results["p"], alpha=0.05, method="fdr_bh")
    results["q"] = q_values
    results["significant_q_0.05"] = rejected
    results.to_csv(output_dir / "anova_effects.csv", index=False)
    print(results.sort_values("q").head(30).to_string(index=False))


def run_correlated_analysis(
    fold_results: Path,
    output_dir: Path,
    metrics: tuple[str, ...] = ("Val_RMSE", "Val_MAE", "Val_R2"),
) -> None:
    """Analyze paired split-level results without treating runs as independent."""
    output_dir.mkdir(parents=True, exist_ok=True)
    data = add_components(pd.read_csv(fold_results))
    missing = {"fold", *metrics} - set(data.columns)
    if missing:
        raise ValueError(f"Missing correlated-analysis columns: {sorted(missing)}")

    factors = list(FACTOR_COLUMNS)
    main_terms = [f"C({factor})" for factor in factors]
    interaction_terms = [
        f"C({left}):C({right})" for left, right in combinations(factors, 2)
    ]
    formula_terms = main_terms + interaction_terms
    effect_rows: list[dict[str, float | str]] = []
    diagnostic_rows: list[dict[str, float | int | str]] = []

    for metric in metrics:
        subset = data[[metric, "fold", *factors]].dropna().copy()
        formula = f"Q('{metric}') ~ " + " + ".join(formula_terms)
        method = "linear mixed-effects model; split random intercept"
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = mixedlm(formula, subset, groups=subset["fold"]).fit(
                    reml=True, method="lbfgs", maxiter=2000, disp=False
                )
            fixed_names = list(result.fe_params.index)
            estimates = result.fe_params
            standard_errors = result.bse_fe
            p_values = result.pvalues.loc[fixed_names]
            intervals = result.conf_int().loc[fixed_names]
            residuals = np.asarray(result.resid)
            exog = np.asarray(result.model.exog)
            random_intercept_variance = float(result.cov_re.iloc[0, 0])
        except Exception:
            # A zero random-effect variance can make MixedLM singular. Cluster-robust
            # covariance still respects the pairing of configurations within a split.
            method = "OLS with split-clustered robust covariance"
            result = ols(formula, subset).fit(
                cov_type="cluster",
                cov_kwds={"groups": subset["fold"], "use_correction": True},
            )
            fixed_names = list(result.params.index)
            estimates = result.params
            standard_errors = result.bse
            p_values = result.pvalues
            intervals = result.conf_int()
            residuals = np.asarray(result.resid)
            exog = np.asarray(result.model.exog)
            random_intercept_variance = float("nan")

        for term in fixed_names:
            if term == "Intercept":
                continue
            effect_rows.append(
                {
                    "metric": metric,
                    "method": method,
                    "term": term,
                    "estimate": float(estimates[term]),
                    "standard_error": float(standard_errors[term]),
                    "ci95_low": float(intervals.loc[term, 0]),
                    "ci95_high": float(intervals.loc[term, 1]),
                    "p": float(p_values[term]),
                }
            )

        shapiro_stat, shapiro_p = shapiro(residuals)
        bp_stat, bp_p, _, _ = het_breuschpagan(residuals, exog)
        diagnostic_rows.append(
            {
                "metric": metric,
                "method": method,
                "n_observations": len(subset),
                "n_splits": int(subset["fold"].nunique()),
                "random_intercept_variance": random_intercept_variance,
                "residual_shapiro_W": float(shapiro_stat),
                "residual_shapiro_p": float(shapiro_p),
                "breusch_pagan_statistic": float(bp_stat),
                "breusch_pagan_p": float(bp_p),
            }
        )

    effects = pd.DataFrame(effect_rows)
    rejected, q_values, _, _ = multipletests(effects["p"], alpha=0.05, method="fdr_bh")
    effects["q"] = q_values
    effects["significant_q_0.05"] = rejected
    effects.to_csv(output_dir / "correlated_effects.csv", index=False)
    pd.DataFrame(diagnostic_rows).to_csv(
        output_dir / "residual_diagnostics.csv", index=False
    )
    print(effects.sort_values("q").head(30).to_string(index=False))
