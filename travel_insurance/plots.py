"""
plots.py
--------
All Matplotlib / Seaborn visualisation functions.

CCDS role: code to create visualizations.

Design rules:
  - Every function returns a Figure without calling plt.show().
  - apply_global_style() sets project-wide aesthetics; call once at notebook start.
  - No hardcoded colours: all palettes come from travel_insurance.config.
  - Axes always carry title, x-label, and y-label.
  - Percentage annotations used wherever a relative comparison is more informative
    than raw counts (e.g., purchase rates, class balance, CV scores).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from sklearn.metrics import (
    auc,
    confusion_matrix as sk_confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

from travel_insurance.config import CMAP_DIV, CMAP_SEQ, PALETTE, PALETTE_LIST, TARGET


# ── Global style ──────────────────────────────────────────────────────────────

def apply_global_style() -> None:
    """Apply project-wide Matplotlib/Seaborn styling. Call once at notebook start."""
    sns.set_theme(style="whitegrid", palette=PALETTE_LIST, font_scale=1.05)
    plt.rcParams.update({
        "figure.dpi":        120,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         True,
        "grid.color":        "#E8E8E8",
        "grid.linewidth":    0.7,
        "legend.frameon":    False,
        "font.size":         11,
    })


# ── Target distribution ───────────────────────────────────────────────────────

def plot_target_distribution(df: pd.DataFrame) -> Figure:
    """
    Horizontal bar chart of target class counts with percentage annotations.
    Shows both count and share so class imbalance is immediately readable.
    """
    counts = df[TARGET].value_counts().sort_index()
    total  = counts.sum()
    labels = {0: "No Insurance (0)", 1: "Purchased (1)"}
    pcts   = counts / total * 100

    fig, ax = plt.subplots(figsize=(7, 4))
    bar_data = pd.DataFrame({
        "Class": [labels[k] for k in counts.index],
        "Count": counts.values,
        "Pct":   pcts.values,
    })
    palette = {labels[k]: PALETTE[k] for k in counts.index}
    sns.barplot(data=bar_data, x="Class", y="Count", hue="Class",
                palette=palette, legend=False, ax=ax, width=0.5)

    for i, (count, pct) in enumerate(zip(counts.values, pcts.values)):
        ax.text(i, count + total * 0.01, f"{pct:.1f}%",
                ha="center", va="bottom", fontsize=11, fontweight="bold")
        ax.text(i, count / 2, f"n={count:,}",
                ha="center", va="center", fontsize=10, color="white", fontweight="bold")

    ax.set_title("Target Distribution: Insurance Purchase Rate", fontsize=13, pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("Count", fontsize=11)
    ax.set_ylim(0, counts.max() * 1.20)
    fig.tight_layout()
    return fig


# ── Stratification check ──────────────────────────────────────────────────────

def plot_stratification_check(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Figure:
    """
    Side-by-side bars confirming the stratified split preserved class proportions.
    Annotations show percentage only; count displayed in the title.
    """
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=False)
    for ax, split_df, title in zip(axes, [train_df, test_df], ["Train Set", "Test Set"]):
        counts = split_df[TARGET].value_counts().sort_index()
        total  = counts.sum()
        pcts   = counts / total * 100
        bar_df = pd.DataFrame({
            "Class": ["No (0)", "Yes (1)"],
            "Count": counts.values,
        })
        bars = sns.barplot(
            data=bar_df, x="Class", y="Count", hue="Class",
            palette={"No (0)": PALETTE[0], "Yes (1)": PALETTE[1]},
            ax=ax, width=0.45, legend=False,
        )
        for i, (count, pct) in enumerate(zip(counts.values, pcts.values)):
            ax.text(i, count + total * 0.015, f"{pct:.1f}%",
                    ha="center", va="bottom", fontsize=11, fontweight="bold")
        ax.set_title(f"{title}  (n={total:,})", fontsize=11)
        ax.set_ylabel("Count")
        ax.set_ylim(0, counts.max() * 1.22)
    fig.suptitle("Stratified Split: Class Balance Preserved", fontsize=12, y=1.02)
    fig.tight_layout()
    return fig


# ── Univariate distributions ──────────────────────────────────────────────────

def plot_numerical_distributions(df: pd.DataFrame, features: list[str]) -> Figure:
    """
    Overlapping seaborn histplots with KDE per target class.
    Vertical dashed lines mark the group medians with text annotations.
    """
    n     = len(features)
    nrows = (n + 1) // 2
    fig, axes = plt.subplots(nrows, 2, figsize=(13, 4.5 * nrows))
    axes = axes.flatten()

    cls_labels = {0: "No Insurance", 1: "Purchased"}
    for ax, feat in zip(axes, features):
        for cls in [0, 1]:
            subset = df[df[TARGET] == cls][feat].dropna()
            sns.histplot(
                subset, kde=True, ax=ax,
                color=PALETTE[cls], alpha=0.35,
                line_kws={"linewidth": 2},
                label=cls_labels[cls],
                bins=28,
            )
            median = subset.median()
            ax.axvline(median, color=PALETTE[cls], linestyle="--", linewidth=1.4)
            ax.text(
                median, ax.get_ylim()[1] * 0.88,
                f"med={median:,.0f}",
                color=PALETTE[cls], fontsize=8, ha="center",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, ec="none"),
            )
        ax.set_title(feat, fontsize=11)
        ax.set_xlabel(feat)
        ax.set_ylabel("Count")
        ax.legend(fontsize=9)

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle("Numerical Feature Distributions by Insurance Status", fontsize=13, y=1.01)
    fig.tight_layout()
    return fig


def plot_categorical_distributions(df: pd.DataFrame, features: list[str]) -> Figure:
    """
    Grouped seaborn barplots showing insurance purchase RATE (%) per category level.
    Each bar is annotated with its exact percentage so differences are precise.
    """
    n     = len(features)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.5 * nrows))
    axes_flat = np.array(axes).flatten() if n > 1 else [axes]

    for ax, feat in zip(axes_flat, features):
        rate_df = (
            df.groupby(feat)[TARGET]
            .agg(["mean", "count"])
            .rename(columns={"mean": "purchase_rate", "count": "n"})
            .reset_index()
        )
        rate_df["pct"] = rate_df["purchase_rate"] * 100

        palette_map = {
            cat: PALETTE_LIST[i % len(PALETTE_LIST)]
            for i, cat in enumerate(rate_df[feat])
        }
        sns.barplot(
            data=rate_df, x=feat, y="pct",
            hue=feat, palette=palette_map, legend=False,
            ax=ax, width=0.5, order=rate_df[feat].tolist(),
        )
        # Annotate each bar: percentage on top, n= inside
        for i, row in rate_df.iterrows():
            ax.text(
                i, row["pct"] + 1.2,
                f"{row['pct']:.1f}%",
                ha="center", va="bottom", fontsize=10, fontweight="bold",
            )
            ax.text(
                i, row["pct"] / 2,
                f"n={row['n']:,}",
                ha="center", va="center", fontsize=9, color="white", fontweight="bold",
            )
        ax.set_title(feat, fontsize=11)
        ax.set_xlabel("")
        ax.set_ylabel("Purchase Rate (%)")
        ax.set_ylim(0, rate_df["pct"].max() * 1.30)
        ax.tick_params(axis="x", rotation=15)

        # Overall purchase rate reference line
        overall = df[TARGET].mean() * 100
        ax.axhline(overall, color="grey", linestyle=":", linewidth=1.2)
        ax.text(
            len(rate_df) - 0.5, overall + 1,
            f"Overall {overall:.1f}%", ha="right", fontsize=8, color="grey",
        )

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.suptitle("Insurance Purchase Rate by Categorical Feature", fontsize=13, y=1.03)
    fig.tight_layout()
    return fig


# ── Bivariate ─────────────────────────────────────────────────────────────────

def plot_boxplots_by_target(df: pd.DataFrame, features: list[str]) -> Figure:
    """
    Seaborn violinplot + boxplot overlay per feature, split by target class.
    Median and IQR range annotated on each violin for precise reading.
    """
    n     = len(features)
    nrows = (n + 1) // 2
    fig, axes = plt.subplots(nrows, 2, figsize=(13, 4.5 * nrows))
    axes = axes.flatten()

    hue_order = [0, 1]
    hue_labels = {0: "No Insurance", 1: "Purchased"}

    for ax, feat in zip(axes, features):
        plot_df = df[[feat, TARGET]].copy()
        plot_df["Status"] = plot_df[TARGET].map(hue_labels)

        sns.violinplot(
            data=plot_df, x="Status", y=feat, hue="Status",
            palette={"No Insurance": PALETTE[0], "Purchased": PALETTE[1]},
            inner=None, linewidth=0.8, alpha=0.55, ax=ax,
            order=["No Insurance", "Purchased"], legend=False,
        )
        sns.boxplot(
            data=plot_df, x="Status", y=feat, hue="Status",
            palette={"No Insurance": PALETTE[0], "Purchased": PALETTE[1]},
            width=0.18, flierprops=dict(marker=".", markersize=3, alpha=0.4),
            linewidth=1.2, ax=ax,
            order=["No Insurance", "Purchased"], legend=False,
        )
        # Annotate medians
        for i, (cls, label) in enumerate(hue_labels.items()):
            med = df[df[TARGET] == cls][feat].median()
            ax.text(
                i, med, f" {med:,.0f}",
                va="center", ha="left", fontsize=9, fontweight="bold",
                color="black",
            )
        ax.set_title(feat, fontsize=11)
        ax.set_xlabel("")
        ax.set_ylabel(feat)

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle("Numerical Features vs. Insurance Status", fontsize=13, y=1.01)
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(df: pd.DataFrame, features: list[str]) -> Figure:
    """
    Side-by-side lower-triangle heatmaps: Pearson (left) and Spearman (right).
    The target column/row is bolded in both panels for quick reading.
    """
    cols = features + [TARGET]
    corr_p = df[cols].corr(method="pearson")
    corr_s = df[cols].corr(method="spearman")
    mask   = np.triu(np.ones_like(corr_p, dtype=bool), k=1)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    for ax, corr, label, cbar_label in zip(
        axes,
        [corr_p, corr_s],
        ["Pearson Correlation", "Spearman Correlation"],
        ["Pearson r", "Spearman ρ"],
    ):
        sns.heatmap(
            corr, mask=mask, annot=True, fmt=".2f",
            cmap=CMAP_DIV, center=0, vmin=-1, vmax=1, ax=ax,
            linewidths=0.6, linecolor="white",
            annot_kws={"size": 10, "weight": "bold"},
            cbar_kws={"shrink": 0.8, "label": cbar_label},
        )
        tick_labels = ax.get_xticklabels()
        for lbl in tick_labels:
            if lbl.get_text() == TARGET:
                lbl.set_fontweight("bold")
        ax.set_xticklabels(tick_labels, rotation=30, ha="right")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        ax.set_title(label, fontsize=12, pad=12)

    fig.tight_layout()
    return fig


def plot_income_age_scatter(df: pd.DataFrame) -> Figure:
    """
    Scatter of age vs. annual_income coloured by target class.
    KDE density contours overlaid per class to reveal distributional overlap.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    for cls in [0, 1]:
        sub   = df[df[TARGET] == cls]
        label = "No Insurance" if cls == 0 else "Purchased"
        sns.scatterplot(
            data=sub, x="age", y="annual_income",
            color=PALETTE[cls], alpha=0.35, s=18,
            label=label, ax=ax,
        )
        sns.kdeplot(
            data=sub, x="age", y="annual_income",
            color=PALETTE[cls], levels=4, linewidths=1.2, alpha=0.7, ax=ax,
        )

    ax.set_xlabel("age", fontsize=11)
    ax.set_ylabel("Annual Income (INR)", fontsize=11)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"Rs {x/1e5:.0f}L"))
    ax.set_title("Age vs. Annual Income by Insurance Status", fontsize=12)
    ax.legend(title="", fontsize=10)
    fig.tight_layout()
    return fig


# ── Model evaluation ──────────────────────────────────────────────────────────

def plot_confusion_matrix(cm: np.ndarray, model_name: str) -> Figure:
    """
    Seaborn heatmap confusion matrix with three annotation layers:
      - Raw count (large, centre)
      - Row percentage, i.e. recall per class (smaller, below count)
      - Column percentage, i.e. precision proxy (lighter text)
    """
    row_sum = cm.sum(axis=1, keepdims=True)
    col_sum = cm.sum(axis=0, keepdims=True)
    row_pct = cm / np.where(row_sum == 0, 1, row_sum) * 100
    col_pct = cm / np.where(col_sum == 0, 1, col_sum) * 100

    # Build annotation strings: count + row% + col%
    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = (
                f"{cm[i, j]:,}\n"
                f"{row_pct[i, j]:.1f}% of actual\n"
                f"{col_pct[i, j]:.1f}% of predicted"
            )

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=annot, fmt="", cmap=CMAP_SEQ,
        xticklabels=["No Insurance", "Purchased"],
        yticklabels=["No Insurance", "Purchased"],
        linewidths=0.5, linecolor="white",
        ax=ax, cbar=False,
        annot_kws={"size": 10},
    )
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_title(f"Confusion Matrix: {model_name}", fontsize=11, pad=10)
    fig.tight_layout()
    return fig


def plot_roc_curves(models: dict, X_test: pd.DataFrame, y_test: pd.Series) -> Figure:
    """ROC curves for all final models with AUC in legend. Diagonal baseline included."""
    fig, ax = plt.subplots(figsize=(7, 5))
    colors  = sns.color_palette("tab10", n_colors=len(models))
    for (name, model), color in zip(models.items(), colors):
        y_score = (
            model.predict_proba(X_test)[:, 1]
            if hasattr(model, "predict_proba")
            else model.decision_function(X_test)
        )
        fpr, tpr, _ = roc_curve(y_test, y_score)
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{name}  (AUC = {auc(fpr, tpr):.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random classifier")
    ax.fill_between([0, 1], [0, 1], alpha=0.04, color="grey")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC Curves: Model Comparison", fontsize=12)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    return fig


def plot_pr_curves(models: dict, X_test: pd.DataFrame, y_test: pd.Series) -> Figure:
    """Precision-Recall curves with no-skill baseline and AP scores in legend."""
    baseline = y_test.mean()
    fig, ax  = plt.subplots(figsize=(7, 5))
    colors   = sns.color_palette("tab10", n_colors=len(models))
    for (name, model), color in zip(models.items(), colors):
        y_score = (
            model.predict_proba(X_test)[:, 1]
            if hasattr(model, "predict_proba")
            else model.decision_function(X_test)
        )
        prec, rec, _ = precision_recall_curve(y_test, y_score)
        ap = auc(rec, prec)
        ax.plot(rec, prec, color=color, lw=2, label=f"{name}  (AP = {ap:.3f})")

    ax.axhline(baseline, color="grey", ls="--", lw=1.2,
               label=f"No-skill baseline ({baseline:.2f})")
    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title("Precision-Recall Curves: Model Comparison", fontsize=12)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    return fig


def plot_cv_comparison(cv_df: pd.DataFrame, metric: str = "roc_auc") -> Figure:
    """
    Horizontal seaborn barplot of mean CV scores with std error bars.
    Bars are sorted best-to-worst (cv_df is already sorted).
    Exact score annotated to the right of each bar.
    """
    col_mean = f"cv_{metric}_mean"
    col_std  = f"cv_{metric}_std"

    plot_df = cv_df.reset_index()
    plot_df = plot_df.iloc[::-1].reset_index(drop=True)   # reverse so best is on top

    plot_df["_color"] = [
        PALETTE_LIST[0] if i >= len(plot_df) - 2 else "#AAAAAA"
        for i in range(len(plot_df))
    ]
    palette_map = dict(zip(plot_df["model"], plot_df["_color"]))

    fig, ax = plt.subplots(figsize=(9, max(3, len(plot_df) * 0.75)))
    sns.barplot(
        data=plot_df, y="model", x=col_mean, hue="model",
        palette=palette_map, ax=ax, orient="h", legend=False,
    )
    # Error bars manually (seaborn 0.13+ estimator= approach)
    for i, (_, row) in enumerate(plot_df.iterrows()):
        ax.errorbar(
            row[col_mean], i, xerr=row[col_std],
            fmt="none", color="black", capsize=4, linewidth=1.5,
        )
        ax.text(
            row[col_mean] + row[col_std] + 0.003, i,
            f"{row[col_mean]:.4f} ± {row[col_std]:.4f}",
            va="center", fontsize=9,
        )

    ax.set_xlabel(f"CV {metric.replace('_', ' ').upper()} (mean)", fontsize=11)
    ax.set_ylabel("")
    ax.set_title(f"Cross-Validation Comparison: {metric.upper()}", fontsize=12)
    ax.set_xlim(0, min(1.0, plot_df[col_mean].max() + 0.12))
    fig.tight_layout()
    return fig


def plot_feature_importance(
    feature_names: list[str],
    importances: np.ndarray,
    model_name: str,
    top_n: int = 15,
) -> Figure:
    """
    Horizontal seaborn barplot of feature importances.
    Bars are annotated with exact importance values as percentages.
    """
    idx   = np.argsort(importances)[-top_n:]
    names = np.array(feature_names)[idx]
    vals  = importances[idx]
    total = vals.sum()

    plot_df = pd.DataFrame({"feature": names, "importance": vals})

    fig, ax = plt.subplots(figsize=(9, max(4, top_n * 0.45)))
    plot_df["_color"] = [
        PALETTE_LIST[0] if v >= np.median(vals) else "#AAAAAA" for v in vals
    ]
    palette_map = dict(zip(plot_df["feature"], plot_df["_color"]))
    sns.barplot(data=plot_df, y="feature", x="importance", hue="feature",
                palette=palette_map, ax=ax, orient="h", legend=False)

    for i, (val, name) in enumerate(zip(vals, names)):
        ax.text(val + total * 0.005, i,
                f"{val/total*100:.1f}%",
                va="center", fontsize=9)

    ax.set_xlabel("Importance (mean decrease in impurity)", fontsize=11)
    ax.set_ylabel("")
    ax.set_title(f"Feature Importances: {model_name} (top {top_n})", fontsize=12)
    fig.tight_layout()
    return fig


def plot_learning_curve(
    estimator,
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    cv: int = 5,
    scoring: str = "roc_auc",
) -> Figure:
    """
    Learning curve with shaded std band.
    Annotates the final train and CV validation score for quick reading.
    """
    from sklearn.model_selection import learning_curve

    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y,
        cv=cv, scoring=scoring,
        train_sizes=np.linspace(0.1, 1.0, 10),
        n_jobs=-1, random_state=42,
    )
    t_mean = train_scores.mean(axis=1)
    t_std  = train_scores.std(axis=1)
    v_mean = val_scores.mean(axis=1)
    v_std  = val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(train_sizes, t_mean, "o-", color=PALETTE[0], lw=2, label="Train")
    ax.fill_between(train_sizes, t_mean - t_std, t_mean + t_std,
                    alpha=0.15, color=PALETTE[0])
    ax.plot(train_sizes, v_mean, "s-", color=PALETTE[1], lw=2, label="CV Validation")
    ax.fill_between(train_sizes, v_mean - v_std, v_mean + v_std,
                    alpha=0.15, color=PALETTE[1])

    # Annotate final scores
    ax.annotate(
        f"Train {t_mean[-1]:.3f}",
        xy=(train_sizes[-1], t_mean[-1]),
        xytext=(train_sizes[-1] * 0.82, t_mean[-1] + 0.01),
        fontsize=9, color=PALETTE[0], fontweight="bold",
    )
    ax.annotate(
        f"CV {v_mean[-1]:.3f}",
        xy=(train_sizes[-1], v_mean[-1]),
        xytext=(train_sizes[-1] * 0.82, v_mean[-1] - 0.025),
        fontsize=9, color=PALETTE[1], fontweight="bold",
    )

    ax.set_xlabel("Training Set Size", fontsize=11)
    ax.set_ylabel(scoring.upper(), fontsize=11)
    ax.set_title(f"Learning Curve: {model_name}", fontsize=12)
    ax.legend(fontsize=10)
    fig.tight_layout()
    return fig


# ── Export ────────────────────────────────────────────────────────────────────

def save_figure(fig: Figure, name: str, figures_dir) -> None:
    """Save a figure to figures_dir as PNG at 150 dpi."""
    from pathlib import Path
    Path(figures_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(Path(figures_dir) / f"{name}.png", dpi=150, bbox_inches="tight")
