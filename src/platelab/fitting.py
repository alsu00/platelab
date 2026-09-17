import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit


# ---------------------------------------------------------------------------
# Model functions
# ---------------------------------------------------------------------------

def exp_func(x, a, b, c):
    """Exponential model: a * exp(b * x) + c."""
    return a * np.exp(b * x) + c


def lin_func(x, m, b):
    """Linear model: m * x + b."""
    return m * x + b


def quad_func(x, a, b, c):
    """Quadratic model: a * x² + b * x + c."""
    return a * x**2 + b * x + c


def sat_exp_func(x, a, b, c):
    """Saturating exponential model: a * (1 - exp(-b * x)) + c."""
    return a * (1 - np.exp(-b * x)) + c


def mm_func(x, v_m, K_m):
    """Michaelis-Menten model: (v_m * x) / (K_m + x)."""
    return (v_m * x) / (K_m + x)


def sigmoid_5pl_func(x, a, b, c, d, g):
    """5-parameter logistic: y = d + (a - d) / (1 + (x/c)^b)^g.

    Parameters
    ----------
    a : float
        Lower asymptote (y as x → 0).
    b : float
        Slope / Hill coefficient (must be > 0).
    c : float
        Midpoint (x at half-max response, EC50).
    d : float
        Upper asymptote (y as x → ∞).
    g : float
        Asymmetry parameter (g = 1 gives symmetric 4PL).
    """
    return d + (a - d) / (1 + (x / c) ** b) ** g


# ---------------------------------------------------------------------------
# Auto p0 estimators
# ---------------------------------------------------------------------------

def _auto_p0_exp(x, y):
    """Estimate (a, b, c) for y = a*exp(b*x) + c."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    c0 = y.min() - 1e-3 * (y.max() - y.min() + 1)
    a0 = y.max() - c0
    with np.errstate(invalid="ignore", divide="ignore"):
        log_y = np.log(y - c0)
    valid = np.isfinite(log_y)
    if valid.sum() >= 2:
        b0 = np.polyfit(x[valid], log_y[valid], 1)[0]
    else:
        b0 = 1.0 if y[-1] > y[0] else -1.0
    return (a0, b0, c0)


def _auto_p0_sat_exp(x, y):
    """Estimate (a, b, c) for y = a*(1-exp(-b*x)) + c."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    idx = np.argsort(x)
    xs, ys = x[idx], y[idx]

    # c: extrapolate the initial slope back to x=0 rather than taking ys[0]
    # directly, which is biased if the curve has already risen at the first point
    n_init = min(4, len(xs))
    if n_init >= 2:
        coeffs = np.polyfit(xs[:n_init], ys[:n_init], 1)
        c0 = float(np.polyval(coeffs, 0))
    else:
        c0 = float(ys[0])

    # a: average the plateau region for robustness against end-point noise
    n_plat = min(4, len(xs))
    plateau = float(np.mean(ys[-n_plat:]))
    a0 = plateau - c0
    if a0 <= 0:
        a0 = 1.0

    # b: x at 63% saturation satisfies b = 1/x_63
    target = c0 + 0.632 * a0
    above = xs[ys >= target]
    if len(above) > 0 and above[0] > 0:
        b0 = 1.0 / above[0]
    else:
        slope = (ys[1] - ys[0]) / (xs[1] - xs[0]) if len(xs) > 1 else 1.0
        b0 = abs(slope / a0) if a0 != 0 else 1.0

    return (a0, b0, c0)


# ---------------------------------------------------------------------------
# Fitting helpers
# ---------------------------------------------------------------------------

def fit_exp(x, y, p0=None):
    """Fit an exponential curve and return (a, b, c, r_squared).

    Model: ``y = a * exp(b * x) + c``

    Parameters
    ----------
    x, y : array-like
        Data to fit.
    p0 : tuple, optional
        Initial parameter guess ``(a, b, c)``. Auto-estimated from the data
        if not provided.

    Returns
    -------
    tuple
        ``(a, b, c, r_squared)``
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if p0 is None:
        p0 = _auto_p0_exp(x, y)
    popt, _ = curve_fit(exp_func, x, y, p0=p0, maxfev=10000)
    a, b, c = popt
    residuals = y - exp_func(x, a, b, c)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    return a, b, c, r_squared


def fit_lin(x, y):
    """Fit a linear curve and return (m, b, r_squared).

    Parameters
    ----------
    x, y : array-like
        Data to fit.

    Returns
    -------
    tuple
        ``(m, b, r_squared)``
    """
    popt, pcov = curve_fit(lin_func, x, y)
    m, b = popt
    residuals = y - lin_func(x, m, b)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    return m, b, r_squared


def fit_sat_exp(x, y, p0=None):
    """Fit a saturating exponential and return (a, b, c, r_squared).

    Model: ``y = a * (1 - exp(-b * x)) + c``

    Parameters
    ----------
    x, y : array-like
        Data to fit.
    p0 : tuple, optional
        Initial parameter guess ``(a, b, c)``. Auto-estimated from the data
        if not provided.

    Returns
    -------
    tuple
        ``(a, b, c, r_squared)``
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if p0 is None:
        p0 = _auto_p0_sat_exp(x, y)
    popt, _ = curve_fit(sat_exp_func, x, y, p0=p0, maxfev=10000)
    a, b, c = popt
    residuals = y - sat_exp_func(x, a, b, c)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    return a, b, c, r_squared


def _auto_p0_sigmoid(x, y):
    """Estimate (a, b, c, d, g) for the 5PL model."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    idx = np.argsort(x)
    xs, ys = x[idx], y[idx]

    # Direction: increasing or decreasing
    increasing = ys[-1] > ys[0]
    a0 = float(ys.min()) if increasing else float(ys.max())
    d0 = float(ys.max()) if increasing else float(ys.min())

    # c: x where response crosses the halfway point between a and d
    mid = (a0 + d0) / 2.0
    crossings = np.where(np.diff(np.sign(ys - mid)))[0]
    if len(crossings) > 0:
        i = crossings[0]
        # linear interpolation between the two bounding points
        x0, x1 = xs[i], xs[i + 1]
        y0, y1 = ys[i], ys[i + 1]
        c0 = x0 + (mid - y0) / (y1 - y0) * (x1 - x0) if y1 != y0 else (x0 + x1) / 2
    else:
        c0 = float(np.median(xs))
    c0 = max(c0, 1e-9)

    return (a0, 2.0, c0, d0, 1.0)


def fit_sigmoid(x, y, p0=None):
    """Fit a 5-parameter logistic curve and return (a, b, c, d, g, r_squared).

    Model: ``y = d + (a - d) / (1 + (x/c)^b)^g``

    Parameters
    ----------
    x, y : array-like
        Data to fit.
    p0 : tuple, optional
        Initial guess ``(a, b, c, d, g)``. Auto-estimated if not provided.

    Returns
    -------
    tuple
        ``(a, b, c, d, g, r_squared)``
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if p0 is None:
        p0 = _auto_p0_sigmoid(x, y)
    bounds = (
        [-np.inf, 1e-6,  1e-9, -np.inf, 1e-6],
        [ np.inf, np.inf, np.inf,  np.inf, np.inf],
    )
    popt, _ = curve_fit(sigmoid_5pl_func, x, y, p0=p0, bounds=bounds, maxfev=20000)
    a, b, c, d, g = popt
    residuals = y - sigmoid_5pl_func(x, a, b, c, d, g)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    return a, b, c, d, g, r2


def fit_quad(x, y):
    """Fit a quadratic curve and return (a, b, c, r_squared).

    Model: ``y = a * x² + b * x + c``

    Parameters
    ----------
    x, y : array-like
        Data to fit.

    Returns
    -------
    tuple
        ``(a, b, c, r_squared)``
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    coeffs = np.polyfit(x, y, 2)
    a, b, c = coeffs
    residuals = y - quad_func(x, a, b, c)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else float('nan')
    return a, b, c, r_squared


def fit_mm(x, y):
    """Fit a Michaelis-Menten curve and return (v_m, K_m, r_squared).

    Parameters
    ----------
    x, y : array-like
        Data to fit.

    Returns
    -------
    tuple
        ``(v_m, K_m, r_squared)``
    """
    popt, pcov = curve_fit(mm_func, x, y)
    v_m, K_m = popt
    residuals = y - mm_func(x, v_m, K_m)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    return v_m, K_m, r_squared


# ---------------------------------------------------------------------------
# Standard curve fitting (DataFrame-level, with plotting)
# ---------------------------------------------------------------------------

def fit_lin_stdcurve(
    df: pd.DataFrame,
    conc_col: str = "concentration",
    value_col: str = "value",
    label_col: str = None,
    plot: bool = True,
    xlabel: str = None,
    ylabel: str = None,
):
    """Fit a linear standard curve (or one per label group) to plate reader data.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at least *value_col* and *conc_col*.
    conc_col : str, optional
        Column name for the x-axis (concentration). Default ``'concentration'``.
    value_col : str, optional
        Column name for the y-axis (signal/value). Default ``'value'``.
    label_col : str, optional
        If provided, the DataFrame is split by unique values in this column and
        a separate fit is performed for each group.  All groups are overlaid on
        the same plot in different colours.
    plot : bool, optional
        Whether to display the plot. Default ``True``.

    Returns
    -------
    tuple or dict
        * **Single curve** (``label_col=None``): ``(m, b)``
        * **Multiple curves** (``label_col`` provided):
          ``{label_value: (m, b), ...}``
    """
    def _fit_group(x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        m, b, r2 = fit_lin(x, y)
        return m, b, r2

    def _make_fit_line(x, m, b):
        x = np.asarray(x, dtype=float)
        x_line = np.linspace(x.min(), x.max(), 200)
        return x_line, lin_func(x_line, m, b)

    def _eq_str(m, b, r2, prefix=""):
        sign = "+" if b >= 0 else "-"
        eq = f"y = {m:.4g}x {sign} {abs(b):.4g}  R² = {r2:.4f}"
        return f"{prefix}  {eq}" if prefix else eq

    fig, ax = plt.subplots(figsize=(6, 4))

    if label_col is None:
        m, b, r2 = _fit_group(df[conc_col], df[value_col])
        x_line, y_line = _make_fit_line(df[conc_col], m, b)
        fit_df = pd.DataFrame({conc_col: x_line, value_col: y_line})

        sns.scatterplot(data=df, x=conc_col, y=value_col, ax=ax,
                        color="steelblue", s=40, zorder=3)
        sns.lineplot(data=fit_df, x=conc_col, y=value_col, ax=ax,
                     color="steelblue", linewidth=1.5,
                     label=_eq_str(m, b, r2))

        ax.legend(bbox_to_anchor=(0.5, -0.18), loc="upper center",
                  borderaxespad=0, fontsize=8, frameon=False)
        ax.set_title("Standard curve")
        ax.set_xlabel(xlabel if xlabel is not None else conc_col)
        ax.set_ylabel(ylabel if ylabel is not None else value_col)
        plt.tight_layout()
        if plot:
            plt.show()
        return (m, b)

    else:
        groups = df[label_col].unique()
        palette = sns.color_palette("tab10", len(groups))
        results = {}

        for color, grp in zip(palette, groups):
            sub = df[df[label_col] == grp]
            m, b, r2 = _fit_group(sub[conc_col], sub[value_col])
            x_line, y_line = _make_fit_line(sub[conc_col], m, b)
            fit_df = pd.DataFrame({conc_col: x_line, value_col: y_line})

            sns.scatterplot(data=sub, x=conc_col, y=value_col, ax=ax,
                            color=color, s=40, zorder=3)
            sns.lineplot(data=fit_df, x=conc_col, y=value_col, ax=ax,
                         color=color, linewidth=1.5,
                         label=_eq_str(m, b, r2, prefix=str(grp)))

            results[grp] = (m, b)

        ax.legend(bbox_to_anchor=(0.5, -0.18), loc="upper center",
                  borderaxespad=0, fontsize=8, frameon=False)
        ax.set_title("Standard curves")
        ax.set_xlabel(xlabel if xlabel is not None else conc_col)
        ax.set_ylabel(ylabel if ylabel is not None else value_col)
        plt.tight_layout()
        if plot:
            plt.show()
        return results


def fit_exp_stdcurve(
    df: pd.DataFrame,
    conc_col: str = "concentration",
    value_col: str = "value",
    label_col: str = None,
    p0: tuple = None,
    plot: bool = True,
    xlabel: str = None,
    ylabel: str = None,
):
    """Fit an exponential standard curve (or one per label group) to plate reader data.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at least *value_col* and *conc_col*.
    conc_col : str, optional
        Column name for the x-axis (concentration). Default ``'concentration'``.
    value_col : str, optional
        Column name for the y-axis (signal/value). Default ``'value'``.
    label_col : str, optional
        If provided, the DataFrame is split by unique values in this column and
        a separate fit is performed for each group.  All groups are overlaid on
        the same plot in different colours.
    p0 : tuple, optional
        Initial parameter guess ``(a, b, c)`` passed to ``fit_exp``.
    plot : bool, optional
        Whether to display the plot. Default ``True``.

    Returns
    -------
    tuple or dict
        * **Single curve** (``label_col=None``): ``(a, b, c)``
        * **Multiple curves** (``label_col`` provided):
          ``{label_value: (a, b, c), ...}``
    """
    def _fit_group(x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        a, b, c, r2 = fit_exp(x, y, p0=p0)
        return a, b, c, r2

    def _make_fit_line(x, a, b, c):
        x = np.asarray(x, dtype=float)
        x_line = np.linspace(x.min(), x.max(), 200)
        return x_line, exp_func(x_line, a, b, c)

    def _eq_str(a, b, c, r2, prefix=""):
        sign = "+" if c >= 0 else "-"
        eq = f"y = {a:.4g}·e^({b:.4g}x) {sign} {abs(c):.4g}  R² = {r2:.4f}"
        return f"{prefix}  {eq}" if prefix else eq

    fig, ax = plt.subplots(figsize=(6, 4))

    if label_col is None:
        a, b, c, r2 = _fit_group(df[conc_col], df[value_col])
        x_line, y_line = _make_fit_line(df[conc_col], a, b, c)
        fit_df = pd.DataFrame({conc_col: x_line, value_col: y_line})

        sns.scatterplot(data=df, x=conc_col, y=value_col, ax=ax,
                        color="steelblue", s=40, zorder=3)
        sns.lineplot(data=fit_df, x=conc_col, y=value_col, ax=ax,
                     color="steelblue", linewidth=1.5,
                     label=_eq_str(a, b, c, r2))

        ax.legend(bbox_to_anchor=(0.5, -0.18), loc="upper center",
                  borderaxespad=0, fontsize=8, frameon=False)
        ax.set_title("Standard curve (exponential)")
        ax.set_xlabel(xlabel if xlabel is not None else conc_col)
        ax.set_ylabel(ylabel if ylabel is not None else value_col)
        plt.tight_layout()
        if plot:
            plt.show()
        return (a, b, c)

    else:
        groups = df[label_col].unique()
        palette = sns.color_palette("tab10", len(groups))
        results = {}

        for color, grp in zip(palette, groups):
            sub = df[df[label_col] == grp]
            a, b, c, r2 = _fit_group(sub[conc_col], sub[value_col])
            x_line, y_line = _make_fit_line(sub[conc_col], a, b, c)
            fit_df = pd.DataFrame({conc_col: x_line, value_col: y_line})

            sns.scatterplot(data=sub, x=conc_col, y=value_col, ax=ax,
                            color=color, s=40, zorder=3)
            sns.lineplot(data=fit_df, x=conc_col, y=value_col, ax=ax,
                         color=color, linewidth=1.5,
                         label=_eq_str(a, b, c, r2, prefix=str(grp)))

            results[grp] = (a, b, c)

        ax.legend(bbox_to_anchor=(0.5, -0.18), loc="upper center",
                  borderaxespad=0, fontsize=8, frameon=False)
        ax.set_title("Standard curves (exponential)")
        ax.set_xlabel(xlabel if xlabel is not None else conc_col)
        ax.set_ylabel(ylabel if ylabel is not None else value_col)
        plt.tight_layout()
        if plot:
            plt.show()
        return results


def fit_sat_exp_stdcurve(
    df: pd.DataFrame,
    conc_col: str = "concentration",
    value_col: str = "value",
    label_col: str = None,
    p0: tuple = None,
    plot: bool = True,
    xlabel: str = None,
    ylabel: str = None,
):
    """Fit a saturating exponential standard curve (or one per label group).

    Model: ``y = a * (1 - exp(-b * x)) + c``

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at least *value_col* and *conc_col*.
    conc_col : str, optional
        Column name for the x-axis. Default ``'concentration'``.
    value_col : str, optional
        Column name for the y-axis. Default ``'value'``.
    label_col : str, optional
        If provided, the DataFrame is split by unique values in this column
        and a separate fit is performed for each group.
    p0 : tuple, optional
        Initial parameter guess ``(a, b, c)``. Auto-estimated if omitted.
    plot : bool, optional
        Whether to display the plot. Default ``True``.

    Returns
    -------
    tuple or dict
        * **Single curve** (``label_col=None``): ``(a, b, c)``
        * **Multiple curves**: ``{label_value: (a, b, c), ...}``
    """
    def _fit_group(x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        a, b, c, r2 = fit_sat_exp(x, y, p0=p0)
        return a, b, c, r2

    def _make_fit_line(x, a, b, c):
        x = np.asarray(x, dtype=float)
        x_line = np.linspace(x.min(), x.max(), 200)
        return x_line, sat_exp_func(x_line, a, b, c)

    def _eq_str(a, b, c, r2, prefix=""):
        sign = "+" if c >= 0 else "-"
        eq = f"y = {a:.4g}·(1−e^(−{b:.4g}x)) {sign} {abs(c):.4g}  R² = {r2:.4f}"
        return f"{prefix}  {eq}" if prefix else eq

    fig, ax = plt.subplots(figsize=(6, 4))

    if label_col is None:
        a, b, c, r2 = _fit_group(df[conc_col], df[value_col])
        x_line, y_line = _make_fit_line(df[conc_col], a, b, c)
        fit_df = pd.DataFrame({conc_col: x_line, value_col: y_line})

        sns.scatterplot(data=df, x=conc_col, y=value_col, ax=ax,
                        color="steelblue", s=40, zorder=3)
        sns.lineplot(data=fit_df, x=conc_col, y=value_col, ax=ax,
                     color="steelblue", linewidth=1.5,
                     label=_eq_str(a, b, c, r2))

        ax.legend(bbox_to_anchor=(0.5, -0.18), loc="upper center",
                  borderaxespad=0, fontsize=8, frameon=False)
        ax.set_title("Standard curve (saturating exponential)")
        ax.set_xlabel(xlabel if xlabel is not None else conc_col)
        ax.set_ylabel(ylabel if ylabel is not None else value_col)
        plt.tight_layout()
        if plot:
            plt.show()
        return (a, b, c)

    else:
        groups = df[label_col].unique()
        palette = sns.color_palette("tab10", len(groups))
        results = {}

        for color, grp in zip(palette, groups):
            sub = df[df[label_col] == grp]
            a, b, c, r2 = _fit_group(sub[conc_col], sub[value_col])
            x_line, y_line = _make_fit_line(sub[conc_col], a, b, c)
            fit_df = pd.DataFrame({conc_col: x_line, value_col: y_line})

            sns.scatterplot(data=sub, x=conc_col, y=value_col, ax=ax,
                            color=color, s=40, zorder=3)
            sns.lineplot(data=fit_df, x=conc_col, y=value_col, ax=ax,
                         color=color, linewidth=1.5,
                         label=_eq_str(a, b, c, r2, prefix=str(grp)))

            results[grp] = (a, b, c)

        ax.legend(bbox_to_anchor=(0.5, -0.18), loc="upper center",
                  borderaxespad=0, fontsize=8, frameon=False)
        ax.set_title("Standard curves (saturating exponential)")
        ax.set_xlabel(xlabel if xlabel is not None else conc_col)
        ax.set_ylabel(ylabel if ylabel is not None else value_col)
        plt.tight_layout()
        if plot:
            plt.show()
        return results


def fit_quad_stdcurve(
    df: pd.DataFrame,
    conc_col: str = "concentration",
    value_col: str = "value",
    label_col: str = None,
    plot: bool = True,
    xlabel: str = None,
    ylabel: str = None,
):
    """Fit a quadratic standard curve (or one per label group) to plate reader data.

    Model: ``y = a * x² + b * x + c``

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at least *value_col* and *conc_col*.
    conc_col : str, optional
        Column name for the x-axis (concentration). Default ``'concentration'``.
    value_col : str, optional
        Column name for the y-axis (signal/value). Default ``'value'``.
    label_col : str, optional
        If provided, the DataFrame is split by unique values in this column and
        a separate fit is performed for each group.
    plot : bool, optional
        Whether to display the plot. Default ``True``.

    Returns
    -------
    tuple or dict
        * **Single curve** (``label_col=None``): ``(a, b, c)``
        * **Multiple curves** (``label_col`` provided):
          ``{label_value: (a, b, c), ...}``
    """
    def _fit_group(x, y):
        a, b, c, r2 = fit_quad(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        return a, b, c, r2

    def _make_fit_line(x, a, b, c):
        x = np.asarray(x, dtype=float)
        x_line = np.linspace(x.min(), x.max(), 200)
        return x_line, quad_func(x_line, a, b, c)

    def _eq_str(a, b, c, r2, prefix=""):
        b_sign = "+" if b >= 0 else "-"
        c_sign = "+" if c >= 0 else "-"
        eq = (f"y = {a:.4g}x² {b_sign} {abs(b):.4g}x "
              f"{c_sign} {abs(c):.4g}  R² = {r2:.4f}")
        return f"{prefix}  {eq}" if prefix else eq

    fig, ax = plt.subplots(figsize=(6, 4))

    if label_col is None:
        a, b, c, r2 = _fit_group(df[conc_col], df[value_col])
        x_line, y_line = _make_fit_line(df[conc_col], a, b, c)
        fit_df = pd.DataFrame({conc_col: x_line, value_col: y_line})

        sns.scatterplot(data=df, x=conc_col, y=value_col, ax=ax,
                        color="steelblue", s=40, zorder=3)
        sns.lineplot(data=fit_df, x=conc_col, y=value_col, ax=ax,
                     color="steelblue", linewidth=1.5,
                     label=_eq_str(a, b, c, r2))

        ax.legend(bbox_to_anchor=(0.5, -0.18), loc="upper center",
                  borderaxespad=0, fontsize=8, frameon=False)
        ax.set_title("Standard curve (quadratic)")
        ax.set_xlabel(xlabel if xlabel is not None else conc_col)
        ax.set_ylabel(ylabel if ylabel is not None else value_col)
        plt.tight_layout()
        if plot:
            plt.show()
        return (a, b, c)

    else:
        groups = df[label_col].unique()
        palette = sns.color_palette("tab10", len(groups))
        results = {}

        for color, grp in zip(palette, groups):
            sub = df[df[label_col] == grp]
            a, b, c, r2 = _fit_group(sub[conc_col], sub[value_col])
            x_line, y_line = _make_fit_line(sub[conc_col], a, b, c)
            fit_df = pd.DataFrame({conc_col: x_line, value_col: y_line})

            sns.scatterplot(data=sub, x=conc_col, y=value_col, ax=ax,
                            color=color, s=40, zorder=3)
            sns.lineplot(data=fit_df, x=conc_col, y=value_col, ax=ax,
                         color=color, linewidth=1.5,
                         label=_eq_str(a, b, c, r2, prefix=str(grp)))

            results[grp] = (a, b, c)

        ax.legend(bbox_to_anchor=(0.5, -0.18), loc="upper center",
                  borderaxespad=0, fontsize=8, frameon=False)
        ax.set_title("Standard curves (quadratic)")
        ax.set_xlabel(xlabel if xlabel is not None else conc_col)
        ax.set_ylabel(ylabel if ylabel is not None else value_col)
        plt.tight_layout()
        if plot:
            plt.show()
        return results

