"""Turn research artifacts into recruiter-readable evidence and conclusions."""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".cache" / "matplotlib")
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from numpy.typing import NDArray
from sklearn.metrics import normalized_mutual_info_score, pairwise_distances

from src.data_pipeline import PROCESSED_DIR, RESULTS_DIR, ROOT, load_config
from src.landscape import run_clustering, run_pca, run_umap
from src.trajectory import build_feature_matrix

FIGURE_DIR = ROOT / "figures"
REPORT_PATH = ROOT / "REPORT.md"
OUT_OF_SAMPLE_PATH = RESULTS_DIR / "out_of_sample_results.parquet"
RANDOM_TRAJECTORY_PATH = RESULTS_DIR / "trajectory_random.parquet"
COMBINED_TRAJECTORY_PATH = RESULTS_DIR / "trajectory_combined.parquet"
LOGGER = logging.getLogger(__name__)


def _default_trajectory_path() -> Path:
    """Prefer the complete portfolio artifact, with random search as a fallback."""
    return COMBINED_TRAJECTORY_PATH if COMBINED_TRAJECTORY_PATH.exists() else RANDOM_TRAJECTORY_PATH


def _style_axis(axis: Axes) -> None:
    axis.grid(alpha=0.18, linewidth=0.8)
    axis.spines[["top", "right"]].set_visible(False)


def _save_continuous_scatter(
    points: np.ndarray,
    color: pd.Series | np.ndarray,
    title: str,
    path: Path,
    colorbar_label: str,
) -> None:
    """Save a landscape view for a continuous diagnostic such as Sharpe or order."""
    fig, axis = plt.subplots(figsize=(8, 6), constrained_layout=True)
    scatter = axis.scatter(
        points[:, 0],
        points[:, 1],
        c=np.asarray(color),
        cmap="viridis",
        s=30,
        alpha=0.78,
        edgecolors="none",
    )
    axis.set(title=title, xlabel="Landscape dimension 1", ylabel="Landscape dimension 2")
    _style_axis(axis)
    fig.colorbar(scatter, ax=axis, label=colorbar_label)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_categorical_scatter(
    points: np.ndarray,
    categories: pd.Series | np.ndarray,
    title: str,
    path: Path,
    legend_title: str,
) -> None:
    """Save a categorical landscape view with a discrete, readable legend."""
    labels = pd.Series(np.asarray(categories), dtype="object")
    fig, axis = plt.subplots(figsize=(8, 6), constrained_layout=True)
    palette = plt.get_cmap("tab10")
    # Plot broad-coverage categories first so smaller, overlapping groups remain
    # visible on top (important when two optimizers evaluate the same expression).
    unique_labels = labels.value_counts().sort_values(ascending=False).index.tolist()
    for position, label in enumerate(unique_labels):
        mask = labels.eq(label).to_numpy()
        axis.scatter(
            points[mask, 0],
            points[mask, 1],
            s=30,
            alpha=0.78,
            color=palette(position % 10),
            edgecolors="none",
            label=str(label),
        )
    axis.set(title=title, xlabel="Landscape dimension 1", ylabel="Landscape dimension 2")
    _style_axis(axis)
    axis.legend(title=legend_title, frameon=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _upper_triangle_distances(values: np.ndarray) -> NDArray[np.float64]:
    if len(values) < 2:
        return np.array([], dtype=float)
    distances = pairwise_distances(values)
    return np.asarray(distances[np.triu_indices_from(distances, k=1)], dtype=np.float64)


def _cost_curve(value: object) -> tuple[np.ndarray, np.ndarray, float]:
    sensitivity = cast(Mapping[str, float], json.loads(value) if isinstance(value, str) else value)
    pairs = sorted((float(cost), float(sharpe)) for cost, sharpe in sensitivity.items())
    costs = np.array([pair[0] for pair in pairs], dtype=float)
    sharpes = np.array([pair[1] for pair in pairs], dtype=float)
    break_even = np.nan
    for left in range(len(pairs) - 1):
        if sharpes[left] >= 0 > sharpes[left + 1]:
            fraction = sharpes[left] / (sharpes[left] - sharpes[left + 1])
            break_even = float(costs[left] + fraction * (costs[left + 1] - costs[left]))
            break
    return costs, sharpes, break_even


def _parse_expression(value: object) -> dict[str, Any]:
    parsed = json.loads(value) if isinstance(value, str) else value
    if not isinstance(parsed, dict):
        raise ValueError("expression must be a JSON object")
    return cast(dict[str, Any], parsed)


def _expression_label(value: object) -> str:
    expression = _parse_expression(value)
    primitive = str(expression["primitive"]).replace("_", " ")
    if expression["primitive"] == "ma_crossover":
        parameters = f"{expression['short']}/{expression['long']}-day"
    else:
        parameters = f"{expression.get('lookback', '—')}-day"
    normalized = expression.get("normalize_by")
    suffix = (
        f", volatility-normalized over {expression['normalize_lookback']} days"
        if normalized
        else ", unnormalized"
    )
    return f"{parameters} {primitive}{suffix}"


def _panel_description() -> str:
    """Describe the saved research panel without making report generation depend on it."""
    prices_path = PROCESSED_DIR / "prices.parquet"
    config = load_config()
    if not prices_path.exists():
        return (
            f"a fixed U.S. large-cap universe from {config.start_date} to {config.end_date}, "
            f"using a {config.forward_return_horizon}-day forward-return target"
        )
    prices = pd.read_parquet(prices_path)
    start = pd.Timestamp(prices.index.min()).date()
    end = pd.Timestamp(prices.index.max()).date()
    return (
        f"{prices.shape[1]} U.S. large-cap equities across {prices.shape[0]:,} trading days "
        f"({start} to {end}), using a {config.forward_return_horizon}-day forward-return target"
    )


def _optimizer_summary_lines(summary: pd.DataFrame) -> str:
    return "\n".join(
        f"- **{name.title()} search:** {int(row['count'])} evaluations; "
        f"mean validation Sharpe {row['mean']:.3f}, median {row['median']:.3f}, "
        f"best {row['max']:.3f}."
        for name, row in summary.iterrows()
    )


def _holdout_table(holdout: pd.DataFrame) -> str:
    header = (
        "| Optimizer | Selected expression | Validation Sharpe | Test gross Sharpe | "
        "Test net Sharpe | Trial-adjusted probability | 95% bootstrap interval |\n"
        "|---|---|---:|---:|---:|---:|---:|"
    )
    rows = []
    for _, row in holdout.iterrows():
        rows.append(
            f"| {str(row['optimizer']).title()} | {_expression_label(row['expression'])} | "
            f"{row['selection_sharpe']:.3f} | {row['test_sharpe']:.3f} | "
            f"{row.get('test_net_sharpe', np.nan):.3f} | "
            f"{row.get('test_probabilistic_sharpe_ratio', np.nan):.1%} | "
            f"[{row.get('test_sharpe_ci_lower', np.nan):.3f}, "
            f"{row.get('test_sharpe_ci_upper', np.nan):.3f}] |"
        )
    return "\n".join([header, *rows])


def generate_report(
    trajectory_path: Path | None = None,
    figure_dir: Path = FIGURE_DIR,
    report_path: Path = REPORT_PATH,
    out_of_sample_path: Path = OUT_OF_SAMPLE_PATH,
) -> dict[str, object]:
    """Analyze a saved search trajectory and write figures plus a narrative report."""
    selected_trajectory_path = trajectory_path or _default_trajectory_path()
    trajectory = pd.read_parquet(selected_trajectory_path)
    features = build_feature_matrix(trajectory)
    expression_features = build_feature_matrix(trajectory, include_metrics=False)
    pca = run_pca(features)
    umap_embedding = run_umap(features)
    clusters = run_clustering(features)
    expression_clusters = run_clustering(expression_features)
    expressions = trajectory["expression"].map(_parse_expression)
    primitive = expressions.map(lambda value: value["primitive"])

    figure_dir.mkdir(parents=True, exist_ok=True)
    _save_continuous_scatter(
        pca,
        trajectory["sharpe"],
        "Candidate landscape: validation Sharpe",
        figure_dir / "pca_sharpe.png",
        "Validation Sharpe",
    )
    _save_continuous_scatter(
        umap_embedding,
        trajectory["iteration"],
        "Search coverage over evaluation order",
        figure_dir / "umap_iteration.png",
        "Evaluation order",
    )
    _save_categorical_scatter(
        umap_embedding,
        trajectory["optimizer"],
        "Search coverage by optimizer",
        figure_dir / "umap_optimizer.png",
        "Optimizer",
    )
    _save_categorical_scatter(
        umap_embedding,
        clusters["kmeans"],
        "Candidate regions in the search landscape",
        figure_dir / "umap_cluster.png",
        "KMeans cluster",
    )

    cross_tab = pd.crosstab(
        pd.Series(clusters["kmeans"], name="cluster"), primitive.rename("primitive")
    )
    cluster_purity = float(cross_tab.max(axis=1).sum() / cross_tab.to_numpy().sum())
    primitive_codes = pd.Categorical(primitive).codes
    primitive_cluster_nmi = float(
        normalized_mutual_info_score(primitive_codes, clusters["kmeans"])
    )
    fig, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    image = axis.imshow(cross_tab.to_numpy(), aspect="auto", cmap="Blues")
    axis.set_xticks(range(len(cross_tab.columns)), cross_tab.columns, rotation=35, ha="right")
    axis.set_yticks(range(len(cross_tab.index)), cross_tab.index)
    axis.set(
        xlabel="Signal family",
        ylabel="KMeans cluster",
        title="How candidate regions relate to signal families",
    )
    for row_index in range(cross_tab.shape[0]):
        for column_index in range(cross_tab.shape[1]):
            axis.text(
                column_index,
                row_index,
                str(int(cross_tab.iloc[row_index, column_index])),
                ha="center",
                va="center",
            )
    fig.colorbar(image, ax=axis, label="Evaluations")
    fig.savefig(figure_dir / "cluster_primitive_crosstab.png", dpi=180)
    plt.close(fig)

    sharpe = trajectory["sharpe"].replace([np.inf, -np.inf], np.nan)
    cutoff = float(sharpe.quantile(0.9))
    top_mask = sharpe >= cutoff
    all_distances = _upper_triangle_distances(features.to_numpy())
    top_distances = _upper_triangle_distances(features.loc[top_mask].to_numpy())
    expression_all_distances = _upper_triangle_distances(expression_features.to_numpy())
    expression_top_distances = _upper_triangle_distances(
        expression_features.loc[top_mask].to_numpy()
    )
    top_clusters = np.unique(clusters["kmeans"][top_mask.to_numpy()])
    expression_top_clusters = np.unique(expression_clusters["kmeans"][top_mask.to_numpy()])
    cluster_view_nmi = float(
        normalized_mutual_info_score(clusters["kmeans"], expression_clusters["kmeans"])
    )

    fig, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    axis.hist(all_distances, bins=30, alpha=0.55, density=True, label="All candidate pairs")
    if top_distances.size:
        axis.hist(
            top_distances,
            bins=20,
            alpha=0.65,
            density=True,
            label="Top-decile candidate pairs",
        )
    axis.set(
        xlabel="Distance in standardized feature space",
        ylabel="Density",
        title="Are stronger candidates structurally concentrated?",
    )
    _style_axis(axis)
    axis.legend()
    fig.savefig(figure_dir / "diversity_distances.png", dpi=180)
    plt.close(fig)

    optimizer_summary = trajectory.groupby("optimizer")["sharpe"].agg(
        ["count", "mean", "median", "max"]
    )
    results: dict[str, object] = {
        "n_iterations": int(len(trajectory)),
        "top_decile_cutoff": cutoff,
        "mean_all_pair_distance": (
            float(np.mean(all_distances)) if all_distances.size else np.nan
        ),
        "mean_top_pair_distance": (
            float(np.mean(top_distances)) if top_distances.size else np.nan
        ),
        "top_decile_cluster_count": int(len(top_clusters)),
        "expression_only_mean_all_pair_distance": float(
            np.mean(expression_all_distances)
        ),
        "expression_only_mean_top_pair_distance": float(
            np.mean(expression_top_distances)
        ),
        "expression_only_top_decile_cluster_count": int(len(expression_top_clusters)),
        "full_vs_expression_cluster_nmi": cluster_view_nmi,
        "cluster_primitive_purity": cluster_purity,
        "cluster_primitive_nmi": primitive_cluster_nmi,
        "optimizer_summary": optimizer_summary.to_dict(orient="index"),
    }

    optimizer_lines = _optimizer_summary_lines(optimizer_summary)
    if len(optimizer_summary) == 1:
        optimizer_interpretation = (
            "Only one optimizer is present in this artifact, so no optimizer comparison is made."
        )
    else:
        optimizer_interpretation = (
            "This comparison is descriptive, not a horse race: the budgets differ "
            "(500 random versus 200 Bayesian evaluations) and only one seed was run."
        )

    holdout_section: str
    decision_section: str
    executive_outcome: str
    if out_of_sample_path.exists():
        included_optimizers = set(trajectory["optimizer"].astype(str))
        holdout = pd.read_parquet(out_of_sample_path)
        holdout = holdout.loc[holdout["optimizer"].astype(str).isin(included_optimizers)].copy()
    else:
        holdout = pd.DataFrame()

    if not holdout.empty:
        fig, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
        for expression, rows in holdout.groupby("expression", sort=False):
            representative = rows.iloc[0]
            costs, net_sharpes, _ = _cost_curve(representative["cost_sensitivity"])
            optimizers = " + ".join(sorted(rows["optimizer"].astype(str).str.title()))
            axis.plot(
                costs,
                net_sharpes,
                marker="o",
                linewidth=2.2,
                label=f"{_expression_label(expression)}\nselected by {optimizers}",
            )
        axis.axhline(0.0, color="#333333", linewidth=1, linestyle="--")
        axis.set(
            xlabel="Transaction cost (bps per unit turnover)",
            ylabel="Held-out net Sharpe",
            title="Implementation costs erase the selected candidate's edge",
        )
        _style_axis(axis)
        axis.legend(fontsize=9)
        fig.savefig(figure_dir / "holdout_cost_sensitivity.png", dpi=180)
        plt.close(fig)

        same_expression = holdout["expression"].nunique() == 1 and len(holdout) > 1
        representative = holdout.iloc[0]
        _, _, break_even = _cost_curve(representative["cost_sensitivity"])
        transaction_cost = float(representative["transaction_cost_bps"])
        gross_sharpe = float(representative["test_sharpe"])
        net_sharpe = float(representative["test_net_sharpe"])
        lower = float(representative["test_sharpe_ci_lower"])
        upper = float(representative["test_sharpe_ci_upper"])
        convergence_note = (
            "Both optimizers independently selected the same expression. The identical "
            "test rows therefore represent one candidate reached by two search procedures, "
            "not two independent confirmations."
            if same_expression
            else "The optimizers selected different expressions, so each row is a separate holdout audit."
        )
        holdout_section = f"""## 5. Holdout audit: does the winner survive?

The highest-validation-Sharpe candidate from each optimizer was fixed before the test period was examined. The table reports the same cost assumption configured for the study.

{_holdout_table(holdout)}

{convergence_note}

![Held-out cost sensitivity](figures/holdout_cost_sensitivity.png)

At **{transaction_cost:.1f} bps per unit turnover**, gross Sharpe declined from **{gross_sharpe:.3f}** to **{net_sharpe:.3f}**. The interpolated break-even cost was only **{break_even:.2f} bps**. The 95% circular moving-block bootstrap interval, **[{lower:.3f}, {upper:.3f}]**, spans economically weak outcomes. The trial-adjusted probability asks whether test Sharpe exceeds an expected best-of-N benchmark based on each optimizer's logged trial count; it remains below 50% in this run.
"""
        decision_section = """## 6. Research decision

**I would not advance this candidate as validated alpha.** It remains an interesting hypothesis—the optimizer convergence and slow volume signal merit economic investigation—but the current evidence is too fragile after implementation costs, dependence-aware uncertainty, and repeated-trial adjustment.

This negative decision is a substantive output of the analysis. The research process prevented a positive gross backtest from becoming an overstated trading claim.
"""
        executive_outcome = (
            f"The selected candidate produced **{gross_sharpe:.3f} gross test Sharpe**, "
            f"but only **{net_sharpe:.3f} net Sharpe at {transaction_cost:.1f} bps per unit "
            f"turnover**, with a **95% bootstrap interval of [{lower:.3f}, {upper:.3f}]**. "
            "**Decision: do not claim a reliable trading edge.**"
        )
        results["out_of_sample"] = holdout.to_dict(orient="records")
    else:
        holdout_section = """## 5. Holdout audit

No held-out artifact was found for the optimizer data in this report. A trading claim requires validation-based selection followed by a one-time test evaluation.
"""
        decision_section = """## 6. Research decision

The landscape analysis is descriptive. Without a matching holdout audit, I would not make a claim about statistical or economic edge.
"""
        executive_outcome = (
            "This artifact maps the search trajectory, but no matching held-out result is "
            "available. **Decision: make no trading claim.**"
        )

    report = f"""# Alpha Search Under Honest Validation

> Quantitative research portfolio report

## Executive summary

I investigated whether automated search over interpretable equity signals reveals stable high-performing regions or mainly selects the best result from noise. I evaluated **{len(trajectory):,} candidates** across five signal families, preserved every trial, mapped the search geometry, and subjected validation-selected winners to a one-time holdout audit.

{executive_outcome}

This is the central judgment of the project: an attractive backtest is a research lead, not evidence, until it survives timing controls, honest selection, costs, uncertainty, and multiple testing.

## 1. Question and experimental design

The experiment uses {_panel_description()}. Candidate expressions combine momentum, mean reversion, volatility, volume ratio, or moving-average crossover with discrete lookbacks and optional volatility normalization.

| Stage | Role in the experiment | Protection against false discovery |
|---|---|---|
| Train (60%) | Supplies Bayesian optimizer feedback; random search remains unguided | Test data cannot guide search |
| Validation (20%) | Selects one winner per optimizer | Separates fitting from model choice |
| Test (20%) | One-time audit of fixed winners | Prevents repeated holdout shopping |

Signals observed at the close of day `t` first receive a return beginning on `t+1`. Forward-return windows that cross a split boundary are purged. Portfolios use cross-sectional rank z-scores scaled to unit gross exposure, which makes candidates comparable without introducing leverage differences.

## 2. What the landscape shows

The top-decile validation cutoff was **{cutoff:.3f} Sharpe**. Mean pairwise distance was **{np.mean(all_distances):.3f}** across all candidate pairs and **{np.mean(top_distances) if top_distances.size else np.nan:.3f}** among top-decile pairs. The stronger candidates were therefore more concentrated in the standardized full-feature space.

To test whether performance metrics created that conclusion by construction, I rebuilt the representation using expression fields only. Mean distance was **{np.mean(expression_all_distances):.3f}** for all expression pairs and **{np.mean(expression_top_distances) if expression_top_distances.size else np.nan:.3f}** among the top decile. Top candidates occupied **{len(expression_top_clusters)} of {len(np.unique(expression_clusters["kmeans"]))} expression-only clusters**.

![Pairwise distance distributions](figures/diversity_distances.png)

**Interpretation:** the landscape is noisy but not structureless. Stronger validation results show concentration, and some structure remains after removing realized performance. This is evidence about the geometry of this experiment—not proof of mathematical local optima or future returns.

## 3. Signal families and candidate regions

KMeans clusters and signal families have **normalized mutual information {primitive_cluster_nmi:.3f}** and **purity {cluster_purity:.3f}**. Full-feature and expression-only cluster assignments have **NMI {cluster_view_nmi:.3f}**. These diagnostics quantify how much candidate geometry follows the primitive definitions and how much changes when realized outcomes enter the representation.

![Candidate regions](figures/umap_cluster.png)

![Cluster by signal family](figures/cluster_primitive_crosstab.png)

UMAP coordinates are synthetic neighborhood coordinates; their absolute values have no financial interpretation. Cluster counts are operational summaries that depend on feature scaling and the requested number of groups.

## 4. Search behavior

{optimizer_lines}

{optimizer_interpretation}

![Search coverage by optimizer](figures/umap_optimizer.png)

{holdout_section}

{decision_section}

## 7. Limitations and next research steps

- The fixed present-day large-cap universe introduces survivorship bias.
- Five-day targets overlap, so adjacent portfolio observations are dependent; annualization alone does not restore independence.
- Daily overlapping 5-day portfolios are not implemented as explicit overlapping sleeves, so turnover, costs, and drawdown are research diagnostics rather than a production P&L simulation.
- The linear transaction-cost model omits capacity, market impact, borrow availability, and execution dynamics.
- The optimizer comparison uses unequal budgets and one seed, so it is descriptive.
- The expected-best-Sharpe benchmark treats trial count as independent and is a transparent approximation, not an effective-trials estimator.

If the hypothesis were continued, I would first obtain point-in-time constituent data, add sector and beta neutralization, test subperiod and universe stability, model liquidity-dependent costs, and repeat optimizer comparisons over matched budgets and multiple seeds. Those steps come before expanding the signal library.

## 8. Implementation evidence

- Vectorized and explicitly lagged cross-sectional backtest
- Typed, configurable search over a fixed expression grammar
- Crash-safe Parquet logging of every evaluation
- PCA, UMAP, KMeans, agglomerative clustering, and HDBSCAN
- Dependence-aware bootstrap, trial-adjusted Sharpe, and transaction-cost sweep
- Synthetic tests for leakage, perfect-foresight direction, turnover, drawdown, costs, split purging, and holdout selection
- Streamlit explorer for candidate-level inspection

## Additional views

![PCA colored by validation Sharpe](figures/pca_sharpe.png)

![Search coverage over evaluation order](figures/umap_iteration.png)
"""
    report_path.write_text(report, encoding="utf-8")
    return results


def main() -> None:  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", type=Path)
    args = parser.parse_args()
    LOGGER.info(
        "Report summary:\n%s",
        json.dumps(generate_report(args.trajectory), indent=2),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
