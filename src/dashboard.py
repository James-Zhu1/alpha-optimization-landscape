"""Interactive portfolio view of the alpha-search trajectory and diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.alpha_lib import AlphaExpression
from src.data_pipeline import RESULTS_DIR
from src.landscape import run_clustering, run_pca, run_umap
from src.trajectory import build_feature_matrix

DATASETS = {
    "Combined (random + Bayesian)": RESULTS_DIR / "trajectory_combined.parquet",
    "Random search": RESULTS_DIR / "trajectory_random.parquet",
    "Bayesian search": RESULTS_DIR / "trajectory_bayesian.parquet",
}
HOLDOUT_PATH = RESULTS_DIR / "out_of_sample_results.parquet"
METRIC_LABELS = {
    "sharpe": "Validation Sharpe",
    "sortino": "Sortino",
    "ic": "Information coefficient",
    "turnover": "Turnover",
    "annualized_volatility": "Annualized volatility",
    "max_drawdown": "Maximum drawdown",
    "iteration": "Evaluation order",
}


@st.cache_data(show_spinner=False)
def load_trajectory(path_string: str, modified_ns: int) -> pd.DataFrame:
    """Load a trajectory and expose expression fields as filterable columns."""
    del modified_ns  # Included in the cache key so changed parquet files reload.
    frame = pd.read_parquet(path_string).copy()
    expressions = frame["expression"].map(
        lambda value: json.loads(value) if isinstance(value, str) else value
    )
    frame["primitive"] = expressions.map(lambda value: value["primitive"])
    frame["normalizer"] = expressions.map(lambda value: value.get("normalize_by") or "none")
    frame["expression_label"] = expressions.map(_expression_label)
    frame["row_id"] = frame["optimizer"].astype(str) + ":" + frame["iteration"].astype(str)
    if "source_iteration" not in frame:
        frame["source_iteration"] = frame["iteration"]
    return frame


def _expression_label(expression: AlphaExpression) -> str:
    primitive = str(expression["primitive"])
    if primitive == "ma_crossover":
        parameters = f"{expression['short']}/{expression['long']}"
    else:
        parameters = str(expression.get("lookback", "—"))
    normalizer = expression.get("normalize_by")
    if normalizer:
        return f"{primitive} ({parameters}), ÷ vol {expression['normalize_lookback']}"
    return f"{primitive} ({parameters})"


@st.cache_data(show_spinner="Computing PCA, UMAP, and clusters…")
def analyze_trajectory(
    path_string: str,
    modified_ns: int,
    n_clusters: int,
) -> pd.DataFrame:
    """Attach stable embeddings and cluster labels to every evaluation."""
    frame = load_trajectory(path_string, modified_ns)
    features = build_feature_matrix(frame)
    pca = run_pca(features, n_components=3)
    umap_embedding = run_umap(features)
    clusters = run_clustering(features, n_clusters=n_clusters)
    analyzed = frame.copy()
    analyzed[["pca_1", "pca_2"]] = pca[:, :2]
    analyzed[["umap_1", "umap_2"]] = umap_embedding
    for method, labels in clusters.items():
        analyzed[f"cluster_{method}"] = labels.astype(str)
    return analyzed


def _available_datasets() -> dict[str, Path]:
    return {label: path for label, path in DATASETS.items() if path.exists()}


def _scatter(
    frame: pd.DataFrame,
    embedding: str,
    color_choice: str,
    cluster_method: str,
) -> go.Figure:
    if embedding == "UMAP":
        x, y = "umap_1", "umap_2"
    else:
        x, y = "pca_1", "pca_2"
    color_column = {
        "Sharpe": "sharpe",
        "Optimizer": "optimizer",
        "Primitive": "primitive",
        "Cluster": f"cluster_{cluster_method}",
        "Evaluation order": "iteration",
    }[color_choice]
    figure = px.scatter(
        frame,
        x=x,
        y=y,
        color=color_column,
        custom_data=["row_id"],
        hover_name="expression_label",
        hover_data={
            "optimizer": True,
            "sharpe": ":.3f",
            "ic": ":.3f",
            "turnover": ":.3f",
            "iteration": True,
            x: False,
            y: False,
            "row_id": False,
        },
        labels={
            x: f"{embedding} 1",
            y: f"{embedding} 2",
            color_column: color_choice,
        },
        color_continuous_scale="Viridis",
        render_mode="webgl",
    )
    figure.update_traces(marker={"size": 8, "opacity": 0.78})
    figure.update_layout(
        height=620,
        margin={"l": 10, "r": 10, "t": 20, "b": 10},
        legend_title_text=color_choice,
    )
    return figure


def _selected_row_id(chart_event: object) -> str | None:
    try:
        points = chart_event.selection.points  # type: ignore[attr-defined]
        if points and points[0].get("customdata"):
            return str(points[0]["customdata"][0])
    except (AttributeError, IndexError, KeyError, TypeError):
        return None
    return None


def _render_selected_alpha(frame: pd.DataFrame, selected_id: str) -> None:
    selected = frame.loc[frame["row_id"] == selected_id]
    if selected.empty:
        return
    row = selected.iloc[0]
    st.subheader("Selected evaluation")
    st.caption(
        f"{row['expression_label']} · {row['optimizer']} evaluation {int(row['source_iteration'])}"
    )
    columns = st.columns(6)
    values = (
        ("Validation Sharpe", row["sharpe"]),
        ("Sortino", row["sortino"]),
        ("IC", row["ic"]),
        ("Turnover", row["turnover"]),
        ("Ann. volatility", row["annualized_volatility"]),
        ("Max drawdown", row["max_drawdown"]),
    )
    for column, (label, value) in zip(columns, values, strict=True):
        column.metric(label, f"{float(value):.3f}")
    with st.expander("Expression definition"):
        expression = row["expression"]
        st.json(json.loads(expression) if isinstance(expression, str) else expression)


def _render_research_outcome() -> None:
    """Lead with the decision supported by the saved holdout audit."""
    if not HOLDOUT_PATH.exists():
        return
    holdout = pd.read_parquet(HOLDOUT_PATH)
    if holdout.empty:
        return
    representative = holdout.iloc[0]
    expression = json.loads(representative["expression"])
    optimizers = " + ".join(sorted(holdout["optimizer"].astype(str).str.title().unique()))
    columns = st.columns(4)
    columns[0].metric("Gross test Sharpe", f"{float(representative['test_sharpe']):.3f}")
    columns[1].metric(
        f"Net test Sharpe @ {float(representative['transaction_cost_bps']):g} bps",
        f"{float(representative['test_net_sharpe']):.3f}",
    )
    columns[2].metric(
        "95% bootstrap interval",
        (
            f"[{float(representative['test_sharpe_ci_lower']):.3f}, "
            f"{float(representative['test_sharpe_ci_upper']):.3f}]"
        ),
    )
    columns[3].metric("Selected by", optimizers)
    st.caption(f"Selected expression: {_expression_label(expression)}")
    st.warning(
        "Research decision: do not claim a reliable trading edge. The positive gross "
        "result is fragile to implementation costs, uncertainty, and repeated-trial adjustment."
    )


def _render_search_progress(frame: pd.DataFrame) -> None:
    progress = frame.sort_values(["optimizer", "source_iteration"]).copy()
    progress["optimizer_evaluation"] = progress.groupby("optimizer").cumcount() + 1
    progress["best_sharpe"] = progress.groupby("optimizer")["sharpe"].cummax()
    figure = px.line(
        progress,
        x="optimizer_evaluation",
        y="best_sharpe",
        color="optimizer",
        labels={
            "optimizer_evaluation": "Evaluations completed",
            "best_sharpe": "Best Sharpe so far",
            "optimizer": "Optimizer",
        },
    )
    figure.update_layout(height=390, margin={"l": 10, "r": 10, "t": 20, "b": 10})
    st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})


def _render_diagnostics(frame: pd.DataFrame, cluster_method: str) -> None:
    left, right = st.columns(2)
    with left:
        distribution = px.box(
            frame,
            x="optimizer",
            y="sharpe",
            color="optimizer",
            points="outliers",
            labels={"optimizer": "Optimizer", "sharpe": "Sharpe"},
        )
        distribution.update_layout(
            height=410,
            showlegend=False,
            margin={"l": 10, "r": 10, "t": 20, "b": 10},
        )
        st.plotly_chart(distribution, width="stretch", config={"displayModeBar": False})
    with right:
        cluster_column = f"cluster_{cluster_method}"
        cross_tab = pd.crosstab(frame[cluster_column], frame["primitive"])
        heatmap = px.imshow(
            cross_tab,
            text_auto=True,
            aspect="auto",
            labels={
                "x": "Primitive",
                "y": f"{cluster_method.title()} cluster",
                "color": "Evaluations",
            },
            color_continuous_scale="Blues",
        )
        heatmap.update_layout(height=410, margin={"l": 10, "r": 10, "t": 20, "b": 10})
        st.plotly_chart(heatmap, width="stretch", config={"displayModeBar": False})


def main() -> None:
    st.set_page_config(page_title="Alpha Search Under Honest Validation", page_icon="◫", layout="wide")
    st.title("Alpha Search Under Honest Validation")
    st.caption(
        "A portfolio case study of what survives after search, selection, costs, and uncertainty."
    )
    st.markdown(
        "I generated interpretable cross-sectional equity signals, preserved every evaluation, "
        "and mapped the resulting search landscape. This explorer lets you inspect the evidence "
        "behind the final research decision."
    )
    _render_research_outcome()

    with st.expander("How to read this explorer"):
        st.markdown(
            "- **Landscape:** nearby points have similar standardized expression fields and "
            "backtest diagnostics; axis values have no direct financial meaning.\n"
            "- **Search coverage:** cumulative best validation Sharpe shows when each optimizer "
            "found stronger candidates.\n"
            "- **Cluster diagnostics:** compare algorithmic regions with economically named "
            "signal families.\n\n"
            "All Sharpe values below are validation-window values unless explicitly labeled test."
        )

    datasets = _available_datasets()
    if not datasets:
        st.error("No trajectory parquet files were found. Run the search pipeline first.")
        st.stop()

    with st.sidebar:
        st.header("View")
        dataset_label = st.selectbox("Trajectory", list(datasets))
        embedding = st.radio("Embedding", ["UMAP", "PCA"], horizontal=True)
        color_choice = st.selectbox(
            "Color points by",
            ["Sharpe", "Optimizer", "Primitive", "Cluster", "Evaluation order"],
        )
        cluster_method = st.selectbox("Cluster method", ["kmeans", "hdbscan", "agglomerative"])
        n_clusters = st.slider("KMeans/agglomerative clusters", 2, 12, 5)

    path = datasets[dataset_label]
    frame = analyze_trajectory(str(path), path.stat().st_mtime_ns, n_clusters)

    with st.sidebar:
        st.header("Filters")
        optimizer_filter = st.multiselect(
            "Optimizer",
            sorted(frame["optimizer"].unique()),
            default=sorted(frame["optimizer"].unique()),
        )
        primitive_filter = st.multiselect(
            "Primitive",
            sorted(frame["primitive"].unique()),
            default=sorted(frame["primitive"].unique()),
        )
        minimum, maximum = float(frame["sharpe"].min()), float(frame["sharpe"].max())
        sharpe_range = st.slider(
            "Sharpe range",
            min_value=minimum,
            max_value=maximum,
            value=(minimum, maximum),
        )

    visible = frame.loc[
        frame["optimizer"].isin(optimizer_filter)
        & frame["primitive"].isin(primitive_filter)
        & frame["sharpe"].between(*sharpe_range)
    ].copy()
    if visible.empty:
        st.warning("No evaluations match the current filters.")
        st.stop()

    summary = st.columns(4)
    summary[0].metric("Visible evaluations", f"{len(visible):,}")
    summary[1].metric("Best validation Sharpe", f"{visible['sharpe'].max():.3f}")
    summary[2].metric("Median validation Sharpe", f"{visible['sharpe'].median():.3f}")
    summary[3].metric("Top-decile cutoff", f"{visible['sharpe'].quantile(0.9):.3f}")

    landscape_tab, progress_tab, diagnostics_tab = st.tabs(
        ["Candidate landscape", "Search coverage", "Cluster diagnostics"]
    )
    with landscape_tab:
        event = st.plotly_chart(
            _scatter(visible, embedding, color_choice, cluster_method),
            width="stretch",
            key="landscape-scatter",
            on_select="rerun",
            selection_mode="points",
        )
        clicked_id = _selected_row_id(event)
        if clicked_id:
            st.session_state["selected_alpha_id"] = clicked_id
        available_ids = visible["row_id"].tolist()
        selected_id = st.session_state.get("selected_alpha_id", available_ids[0])
        if selected_id not in available_ids:
            selected_id = available_ids[0]
        # Built once rather than per option: format_func runs for every entry.
        labels = visible.set_index("row_id")["expression_label"]
        selected_id = str(
            st.selectbox(
                "Inspect evaluation",
                available_ids,
                index=available_ids.index(selected_id),
                format_func=lambda value: str(labels.at[value]),
            )
        )
        st.session_state["selected_alpha_id"] = selected_id
        _render_selected_alpha(visible, selected_id)
    with progress_tab:
        _render_search_progress(visible)
        st.dataframe(
            visible[
                [
                    "iteration",
                    "optimizer",
                    "primitive",
                    "expression_label",
                    "sharpe",
                    "sortino",
                    "ic",
                    "turnover",
                    "max_drawdown",
                ]
            ].sort_values("sharpe", ascending=False),
            width="stretch",
            hide_index=True,
        )
    with diagnostics_tab:
        _render_diagnostics(visible, cluster_method)


if __name__ == "__main__":
    main()
