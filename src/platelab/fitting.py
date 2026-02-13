import numpy as np
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


def mm_func(x, v_m, K_m):
    """Michaelis-Menten model: (v_m * x) / (K_m + x)."""
    return (v_m * x) / (K_m + x)


# ---------------------------------------------------------------------------
# Fitting helpers
# ---------------------------------------------------------------------------

def fit_exp(x, y, p0=None):
    """Fit an exponential curve and return (a, b, c, r_squared).

    Parameters
    ----------
    x, y : array-like
        Data to fit.
    p0 : tuple, optional
        Initial parameter guess passed to ``scipy.optimize.curve_fit``.

    Returns
    -------
    tuple
        ``(a, b, c, r_squared)``
    """
    popt, pcov = curve_fit(exp_func, x, y, p0=p0)
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
