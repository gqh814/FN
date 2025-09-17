import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.patches as mpatches
import statsmodels.api as sm
import matplotlib as mpl

def plot_growth_vs_initial(summary, goal="sdgi_s", highlight_ids=None,
                           start_year=2000, end_year=2024,region_to_color=None,
                           color_by_region=False,
                           region_to_code=None,
                           region_categories=None,
                           scale_by_population=True,
                           show_weighted=False):
    """
    Scatter plot of growth vs initial score for a given goal,
    bubble size = population in end_year,
    with optional region coloring and weighted regression.

    Parameters
    ----------
    summary : pd.DataFrame
        Dataframe with *_start, *_rel_change, population, and region columns.
    goal : str, default "sdgi_s"
        Which goal/indicator to plot (e.g. "sdgi_s", "goal1", "goal5", etc.).
    highlight_ids : list of str, optional
        List of country IDs (ISO codes) to label.
    start_year : int
        The starting year used in summary.
    end_year : int
        The ending year used in summary.
    color_by_region : bool, default False
        If True, color points by region.
    region_to_code : dict, optional
        Mapping from region → numeric code (for stable coloring).
    region_categories : list, optional
        List of all region categories (for consistent legend order).
    scale_by_population : bool, default True
        If True, scale bubble sizes by population.
    show_weighted : bool, default False
        If True, also plot a population-weighted regression line.
    """

    # Ensure population is numeric
    summary["population"] = pd.to_numeric(summary["population"], errors="coerce")

    # Drop rows with missing values
    valid = summary[[f"{goal}_start", f"{goal}_rel_change", "population", "region"]].dropna()
    obs = len(valid)

    # --- Select variables ---
    x = valid[f"{goal}_start"].to_numpy()
    y = valid[f"{goal}_rel_change"].to_numpy()
    w = valid["population"].to_numpy()

    if scale_by_population:
        sizes = (w / w.max()) * 800
    else:
        sizes = np.full_like(x, 50)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(10, 6))

    # Scatter with optional region coloring
    # Scatter with region coloring (pastel colors)
    if color_by_region and region_to_color is not None:
        colors = valid["region"].map(region_to_color)
        ax.scatter(x, y, s=sizes, c=colors, alpha=0.6, edgecolor="k")

        # Legend: always use the same region→color mapping
        region_patches = [
            mpatches.Patch(color=region_to_color[r], label=r) for r in region_categories
        ]
        region_legend = ax.legend(handles=region_patches, title="Region",
                                loc="upper right", frameon=True)
        ax.add_artist(region_legend)
    else:
        ax.scatter(x, y, s=sizes, alpha=0.6, edgecolor="k")

    # Highlight selected countries
    if highlight_ids:
        for cid in highlight_ids:
            if cid in summary.index:
                ax.text(summary.loc[cid, f"{goal}_start"],
                        summary.loc[cid, f"{goal}_rel_change"],
                        cid, fontsize=10, fontweight="bold")

    # --- Unweighted fit ---
    m, b = np.polyfit(x, y, 1)
    xx = np.linspace(x.min(), x.max(), 200)
    fit_line, = ax.plot(xx, m*xx + b, color="grey", linestyle="--", linewidth=2,
                        label=f"OLS: y={m:.3f}x+{b:.3f}")

    # --- Weighted fit (optional) ---
    fit_handles = [fit_line]
    if show_weighted:
        X = sm.add_constant(x)
        wls_model = sm.WLS(y, X, weights=w).fit()
        m_w, b_w = wls_model.params[1], wls_model.params[0]
        fit_weighted, = ax.plot(xx, m_w*xx + b_w, "c-.", linewidth=2,
                                label=f"Weighted OLS: y={m_w:.3f}x+{b_w:.3f}")
        fit_handles.append(fit_weighted)

    # --- Legends ---
    if color_by_region and region_to_color is not None:
        region_patches = [
            mpatches.Patch(color=region_to_color[r], label=r)
            for r in region_categories
        ]
        region_legend = ax.legend(handles=region_patches, title="Region",
                                loc="upper right", frameon=True)
        ax.add_artist(region_legend)


    # Fit legend (always shown)
    ax.legend(handles=fit_handles, loc="lower right")

    # Labels
    ax.set_xlabel(f"{goal.upper()} Initial Score ({start_year})")
    ax.set_ylabel(f"{goal.upper()} Growth ({start_year}–{end_year}) (percentage points)")
    ax.set_title(f"{goal.upper()}: Growth vs Initial Score\n"
                 f"Bubble size ∼ Population in {end_year}\n"
                 f"Observations: {obs}")

    ax.grid(True)
    plt.show()



def build_summary(df, df_pop, start_year, end_year):
    """
    Build a summary dataframe for a chosen start and end year.
    Matches the notebook procedure exactly.
    """

    # Remove aggregates (ids starting with "_")
    df = df[~df["id"].astype(str).str.startswith("_")].copy()

    # --- Clean numeric columns (sdgi_s + goals) ---
    goal_cols = [c for c in df.columns if c.startswith("goal")]
    all_cols = ["sdgi_s"] + goal_cols
    for col in all_cols:
        df[col] = df[col].astype(str).str.replace(",", ".").astype(float)

    # --- Get start and end year data ---
    df_start = df[df["year"] == start_year].set_index("id")[all_cols + ["indexreg_"]]
    df_end   = df[df["year"] == end_year].set_index("id")[all_cols]

    # --- Build structured DataFrame ---
    summary = pd.DataFrame(index=df_start.index)
    for col in all_cols:
        summary[f"{col}_start"] = df_start[col]
        summary[f"{col}_rel_change"] = (df_end[col] - df_start[col])  # absolute change

    summary["region"] = df_start["indexreg_"]

    # --- Add population (fixed at 2024) ---
    df_pop_long = df_pop.melt(
        id_vars=["Country Name", "Country Code"],
        var_name="year",
        value_name="population"
    )
    df_pop_long["year"] = pd.to_numeric(df_pop_long["year"], errors="coerce")
    pop_2024 = df_pop_long[df_pop_long["year"] == end_year][["Country Code", "population"]]

    summary = summary.reset_index().merge(
        pop_2024, left_on="id", right_on="Country Code", how="left"
    ).drop(columns=["Country Code"])

    # Force population to float immediately
    summary["population"] = pd.to_numeric(summary["population"], errors="coerce").astype(float)

    summary = summary.set_index("id")

    return summary

def compute_fit(summary, goal="sdgi_s"):
    """
    Compute slope of growth vs initial score.
    Returns unweighted and weighted slope estimates (WLS with population weights).
    """
    # Drop missing
    valid = summary[[f"{goal}_start", f"{goal}_rel_change", "population"]].dropna()

    # Ensure numeric
    valid["population"] = pd.to_numeric(valid["population"], errors="coerce")

    x = valid[f"{goal}_start"].to_numpy(dtype=float)
    y = valid[f"{goal}_rel_change"].to_numpy(dtype=float)
    w = valid["population"].to_numpy(dtype=float)

    # --- Unweighted fit (plain OLS with polyfit) ---
    m_unw, _ = np.polyfit(x, y, 1)

    # --- Weighted fit (WLS with population as weights) ---
    X = sm.add_constant(x)
    wls_model = sm.WLS(y, X, weights=w).fit()
    m_w = wls_model.params[1]  # slope

    return m_unw, m_w


def build_summaries(df, df_pop, periods):
    """
    Build summary DataFrames for a list of periods.
    periods = [(2000, 2012), (2012, 2024), ...]
    Returns a dict of summaries.
    """
    summaries = {}
    for start, end in periods:
        key = f"{start}-{end}"
        summaries[key] = build_summary(df, df_pop, start, end)
    return summaries

def compare_periods(summaries, goal="sdgi_s"):
    """
    Compute unweighted and weighted betas for all periods.
    Returns a DataFrame with results.
    """
    rows = []
    for period, summary in summaries.items():
        b_unw, b_w = compute_fit(summary, goal=goal)
        rows.append({"Period": period, "Unweighted": b_unw, "Weighted": b_w})
    return pd.DataFrame(rows).set_index("Period")

def plot_beta_comparison(results, goal="sdgi_s"):
    results.plot(
        kind="bar", figsize=(8,5), color=["grey", "c"], edgecolor="black"
    )
    plt.axhline(0, color="black", linewidth=1)
    plt.ylabel("Slope (β)")
    plt.title(f"Convergence Slopes: {goal.upper()}")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()
