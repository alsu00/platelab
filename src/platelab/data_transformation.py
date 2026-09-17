import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Concentration conversions (inverse of standard curves)
# ---------------------------------------------------------------------------

def convert_to_conc(rlu, params):
    """Convert a reading to concentration using a linear standard curve.

    Parameters
    ----------
    rlu : float or array-like
        Raw reading (RLU / RFU / OD).
    params : tuple
        ``(m, b)`` from a linear fit where ``reading = m * conc + b``.

    Returns
    -------
    float or ndarray
        Concentration.
    """
    m, b = params
    return (rlu - b) / m


def convert_to_conc_exp(rlu, params):
    """Convert a reading to concentration using an exponential standard curve.

    Parameters
    ----------
    rlu : float or array-like
        Raw reading.
    params : tuple
        ``(a, b, c)`` from an exponential fit where
        ``reading = a * exp(b * conc) + c``.

    Returns
    -------
    float or ndarray
        Concentration.
    """
    a, b, c = params
    return np.log((rlu - c) / a) / b


def convert_to_conc_quad(rlu, params):
    """Convert a reading to concentration using a quadratic standard curve.

    Inverts the model ``reading = a * conc² + b * conc + c`` via the
    quadratic formula, selecting the non-negative root.

    Parameters
    ----------
    rlu : float or array-like
        Raw reading (RLU / RFU / OD).
    params : tuple
        ``(a, b, c)`` from a quadratic fit where
        ``reading = a * conc² + b * conc + c``.

    Returns
    -------
    float or ndarray
        Concentration.
    """
    a, b, c = params
    rlu = np.asarray(rlu, dtype=float)
    discriminant = b**2 - 4 * a * (c - rlu)
    discriminant = np.maximum(discriminant, 0.0)
    x1 = (-b + np.sqrt(discriminant)) / (2 * a)
    x2 = (-b - np.sqrt(discriminant)) / (2 * a)
    return np.where(x1 >= 0, x1, x2)


def convert_to_conc_sat_exp(rlu, params):
    """Convert a reading to concentration using a saturating exponential standard curve.

    Inverts the model ``reading = a * (1 - exp(-b * conc)) + c``:

    .. math::

        conc = -\\frac{\\ln\\!\\left(1 - \\frac{rlu - c}{a}\\right)}{b}

    Parameters
    ----------
    rlu : float or array-like
        Raw reading.
    params : tuple
        ``(a, b, c)`` from a saturating exponential fit.

    Returns
    -------
    float or ndarray
        Concentration.
    """
    a, b, c = params
    return -np.log(1 - (rlu - c) / a) / b


# ---------------------------------------------------------------------------
# DataFrame-level concentration getters
# ---------------------------------------------------------------------------

def get_conc_lin(df: pd.DataFrame, params: tuple, value_col: str = 'value', conc_col: str = 'conc') -> pd.DataFrame:
    """Apply a linear standard curve to a DataFrame column to get concentrations.

    Parameters
    ----------
    df : pd.DataFrame
    params : tuple
        ``(m, b)`` from a linear fit.
    value_col : str, optional
        Column containing raw readings. Default ``'value'``.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with a new ``'concentration'`` column.
    """
    df = df.copy()
    df[conc_col] = convert_to_conc(df[value_col], params)
    return df


def get_conc_exp(df: pd.DataFrame, params: tuple, value_col: str = 'value', conc_col: str = 'conc') -> pd.DataFrame:
    """Apply an exponential standard curve to a DataFrame column to get concentrations.

    Parameters
    ----------
    df : pd.DataFrame
    params : tuple
        ``(a, b, c)`` from an exponential fit.
    value_col : str, optional
        Column containing raw readings. Default ``'value'``.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with a new ``'concentration'`` column.
    """
    df = df.copy()
    df[conc_col] = convert_to_conc_exp(df[value_col], params)
    return df


def get_conc_quad(df: pd.DataFrame, params: tuple, value_col: str = 'value', conc_col: str = 'conc') -> pd.DataFrame:
    """Apply a quadratic standard curve to a DataFrame column to get concentrations.

    Parameters
    ----------
    df : pd.DataFrame
    params : tuple
        ``(a, b, c)`` from a quadratic fit.
    value_col : str, optional
        Column containing raw readings. Default ``'value'``.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with a new ``'concentration'`` column.
    """
    df = df.copy()
    df[conc_col] = convert_to_conc_quad(df[value_col], params)
    return df


def get_conc_sat_exp(df: pd.DataFrame, params: tuple, value_col: str = 'value', conc_col: str = 'conc') -> pd.DataFrame:
    """Apply a saturating exponential standard curve to a DataFrame column to get concentrations.

    Parameters
    ----------
    df : pd.DataFrame
    params : tuple
        ``(a, b, c)`` from a saturating exponential fit.
    value_col : str, optional
        Column containing raw readings. Default ``'value'``.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with a new ``'concentration'`` column.
    """
    df = df.copy()
    df[conc_col] = convert_to_conc_sat_exp(df[value_col], params)
    return df


# ---------------------------------------------------------------------------
# Kinetic data transformations
# ---------------------------------------------------------------------------

def subtract_baseline(df: pd.DataFrame, timepoint: float = None) -> pd.DataFrame:
    """Subtract the value at a given timepoint from all timepoints for each well.

    Parameters
    ----------
    df : pd.DataFrame
        Kinetic dataframe with columns 'well', 'time_s', and 'value'.
    timepoint : float, optional
        The timepoint to use as the baseline. Defaults to the earliest
        timepoint in the dataframe.

    Returns
    -------
    pd.DataFrame
        DataFrame with a new 'delta_value' column containing baseline-subtracted values.
    """
    if timepoint is None:
        ref_time = df['time_s'].min()
    else:
        ref_time = timepoint

    baseline = (
        df[df['time_s'] == ref_time]
        .set_index('well')['value']
        .rename('t0_value')
    )
    df = df.join(baseline, on='well')
    df['delta_value'] = df['value'] - df['t0_value']
    return df
