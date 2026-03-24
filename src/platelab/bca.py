import numpy as np
import matplotlib.pyplot as plt

from .fitting import fit_lin, lin_func, convert_to_conc


def process_bca(df, std_conc_col, sample_col=None, ax=None):
    """Fit a BCA standard curve and back-calculate sample protein concentrations.

    Typical workflow::

        df = read_plate("BCA.xlsx")
        df = label(df, well_dict={
            "std_conc": {0: ["A1","A2"], 125: ["B1","B2"], 250: ["C1","C2"],
                         500: ["D1","D2"], 750: ["E1","E2"], 1000: ["F1","F2"],
                         1500: ["G1","G2"], 2000: ["H1","H2"]},
            "sample":   {"eluate_1": ["A3","A4"], "eluate_2": ["B3","B4"]},
        })
        params, results, fig = process_bca(df, std_conc_col="std_conc", sample_col="sample")

    Parameters
    ----------
    df : pd.DataFrame
        Labeled plate DataFrame containing a ``'value'`` column plus the
        label columns added by :func:`label`.
    std_conc_col : str
        Name of the column holding standard concentrations (µg/mL).
        Wells that are *not* standards should have ``NaN`` in this column.
    sample_col : str, optional
        Name of the column holding sample identifiers. When provided,
        the returned DataFrame is grouped and averaged by sample name.
        If ``None``, all non-standard wells are returned individually with
        their back-calculated concentrations.
    ax : matplotlib.axes.Axes, optional
        Axes to draw the standard curve on. If ``None`` a new figure is
        created and returned as the third element of the tuple.

    Returns
    -------
    params : tuple
        ``(m, b, r_squared)`` from the linear fit
        ``absorbance = m * conc_ug_mL + b``.
    results : pd.DataFrame
        When *sample_col* is given: columns are ``[sample_col,
        'absorbance', 'conc_ug_mL', 'n']`` (replicates averaged).
        Otherwise: the sample rows from *df* with a new ``'conc_ug_mL'``
        column appended.
    fig : matplotlib.figure.Figure or None
        The newly created figure when *ax* was ``None``; ``None`` otherwise.
    """
    # ------------------------------------------------------------------ #
    # Split standards vs samples
    # ------------------------------------------------------------------ #
    std_mask = df[std_conc_col].notna()
    std_df = df[std_mask].copy()

    if sample_col is not None:
        sample_mask = df[sample_col].notna()
    else:
        sample_mask = ~std_mask
    sample_df = df[sample_mask].copy()

    # ------------------------------------------------------------------ #
    # Average standard replicates, then fit
    # ------------------------------------------------------------------ #
    std_avg = (
        std_df.groupby(std_conc_col)["value"]
        .mean()
        .reset_index()
        .rename(columns={std_conc_col: "conc_ug_mL", "value": "absorbance"})
    )

    x = std_avg["conc_ug_mL"].values.astype(float)
    y = std_avg["absorbance"].values.astype(float)

    m, b, r_squared = fit_lin(x, y)
    params = (m, b, r_squared)

    # ------------------------------------------------------------------ #
    # Plot
    # ------------------------------------------------------------------ #
    new_fig = ax is None
    if new_fig:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = None

    # Individual replicate scatter
    ax.scatter(
        std_df[std_conc_col], std_df["value"],
        color="steelblue", alpha=0.5, s=30, zorder=3, label="replicates",
    )
    # Per-concentration means
    ax.scatter(
        x, y,
        color="steelblue", s=80, edgecolors="navy", linewidths=0.8,
        zorder=4, label="mean",
    )
    # Fit line
    x_fit = np.linspace(x.min(), x.max(), 300)
    ax.plot(
        x_fit, lin_func(x_fit, m, b),
        color="tomato", linewidth=1.5,
        label=f"y = {m:.5f}x + {b:.4f}\n$R^2$ = {r_squared:.4f}",
    )

    ax.set_xlabel("Protein concentration (µg/mL)")
    ax.set_ylabel("Absorbance (562 nm)")
    ax.set_title("BCA Standard Curve")
    ax.legend(fontsize=8)

    if new_fig:
        fig.tight_layout()

    # ------------------------------------------------------------------ #
    # Back-calculate sample concentrations
    # ------------------------------------------------------------------ #
    sample_df["conc_ug_mL"] = convert_to_conc(sample_df["value"], (m, b))

    if sample_col is not None:
        results = (
            sample_df.groupby(sample_col)
            .agg(
                absorbance=("value", "mean"),
                conc_ug_mL=("conc_ug_mL", "mean"),
                n=("value", "count"),
            )
            .reset_index()
        )
    else:
        results = sample_df

    return params, results, fig
