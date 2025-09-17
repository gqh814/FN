import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.patches as mpatches
import statsmodels.api as sm
import matplotlib as mpl

def plot_growth_vs_initial(summary, goal="sdgi_s", highlight_ids=None,
                           start_year=2000, end_year=2024,
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
    if color_by_region and region_to_code is not None and region_categories is not None:
        codes = valid["region"].map(region_to_code).to_numpy()
        norm = mpl.colors.Normalize(vmin=0, vmax=len(region_categories)-1)
        ax.scatter(x, y, s=sizes, c=codes, cmap=plt.cm.tab10, norm=norm,
                   alpha=0.6, edgecolor="k")
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
    fit_line, = ax.plot(xx, m*xx + b, "k--", linewidth=2,
                        label=f"Unweighted: y={m:.3f}x+{b:.3f}")

    # --- Weighted fit (optional) ---
    fit_handles = [fit_line]
    if show_weighted:
        X = sm.add_constant(x)
        wls_model = sm.WLS(y, X, weights=w).fit()
        m_w, b_w = wls_model.params[1], wls_model.params[0]
        fit_weighted, = ax.plot(xx, m_w*xx + b_w, "c-.", linewidth=2,
                                label=f"Weighted: y={m_w:.3f}x+{b_w:.3f}")
        fit_handles.append(fit_weighted)

    # --- Legends ---
    # Region legend (if enabled)
    if color_by_region and region_to_code is not None and region_categories is not None:
        region_patches = [
            mpatches.Patch(color=plt.cm.tab10(norm(region_to_code[r])), label=r)
            for r in region_categories
        ]
        region_legend = ax.legend(handles=region_patches, title="Region",
                                  loc="upper right", frameon=True)
        ax.add_artist(region_legend)

    # Fit legend (always shown)
    ax.legend(handles=fit_handles, loc="lower right")

    # Labels
    ax.set_xlabel(f"{goal.upper()} Initial Score ({start_year})")
    ax.set_ylabel(f"{goal.upper()} Relative Growth ({start_year}–{end_year})")
    ax.set_title(f"{goal.upper()}: Growth vs Initial Score\n"
                 f"Bubble size ∼ Population in {end_year}\n"
                 f"Observations: {obs}")

    ax.grid(True)
    plt.show()
