"""Candidate scoring and ranking for cross-match results."""

import logging

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord, Galactic
import astropy.units as u

from .config import SCORING_WEIGHTS

logger = logging.getLogger(__name__)


def _best_band_stats(df):
    """Compute per-source best-band variability statistics.

    For each source, pick the band with the most observations and return:
    amplitude (rms_mag), nobs, chi2_reduced, and number of filters observed.
    """
    bands = ["g", "r", "i"]
    nobs_cols = [f"nobs_{b}" for b in bands]
    rms_cols = [f"rms_mag_{b}" for b in bands]
    chi2_cols = [f"chi2_{b}" for b in bands]

    # Fill NaN with 0 for nobs and pick best band per row
    nobs_arr = df[nobs_cols].fillna(0).values  # (N, 3)
    best = np.argmax(nobs_arr, axis=1)
    rows = np.arange(len(df))

    rms_arr = df[rms_cols].values
    chi2_arr = df[chi2_cols].values

    amplitude = rms_arr[rows, best]
    nobs = nobs_arr[rows, best]
    chi2 = chi2_arr[rows, best]
    n_filters = (nobs_arr > 0).sum(axis=1)

    # Reduced chi-squared: chi2 / (nobs - 1), guarding against div-by-zero
    chi2_red = np.where(nobs > 1, chi2 / (nobs - 1), 0.0)

    return amplitude, nobs, chi2_red, n_filters


def _normalise(arr):
    """Min-max normalise to [0, 1], handling constant arrays."""
    arr = np.asarray(arr, dtype=float)
    lo, hi = np.nanmin(arr), np.nanmax(arr)
    if hi == lo:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def score_candidates(cat_a, cat_b):
    """Score and rank cross-match candidates.

    Parameters
    ----------
    cat_a : pd.DataFrame
        Category A candidates (no Gaia match).
    cat_b : pd.DataFrame
        Category B candidates (Gaia match, not known variable).

    Returns
    -------
    pd.DataFrame
        Combined, scored, and sorted candidates.
    """
    df = pd.concat([cat_a, cat_b], ignore_index=True)
    if len(df) == 0:
        logger.warning("No candidates to score")
        df["score"] = []
        return df

    # Variability statistics
    amplitude, nobs, chi2_red, n_filters = _best_band_stats(df)
    df["amplitude"] = amplitude
    df["nobs"] = nobs
    df["chi2_red"] = chi2_red
    df["n_filters"] = n_filters

    # Galactic latitude
    coords = SkyCoord(ra=df["ra"].values * u.deg, dec=df["dec"].values * u.deg)
    gal = coords.galactic
    df["gal_lat_abs"] = np.abs(gal.b.deg)

    # Normalise each component to [0, 1]
    w = SCORING_WEIGHTS
    score = np.zeros(len(df))
    score += w["amplitude"] * _normalise(amplitude)
    score += w["nobs"] * _normalise(nobs)
    score += w["chi2_red"] * _normalise(chi2_red)
    score += w["n_filters"] * _normalise(n_filters)
    score += w["gal_lat"] * _normalise(df["gal_lat_abs"].values)

    # Gaia G-mag bonus for Category B (brighter = higher score)
    is_b = df["category"] == "B"
    if is_b.any():
        g_mag = df.loc[is_b, "gaia_g_mag"].values
        # Invert: brighter (lower mag) → higher score
        g_score = _normalise(-g_mag)
        score[is_b.values] += w["gaia_g_mag"] * g_score

    df["score"] = score
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    logger.info(
        "Scored %d candidates (Cat A: %d, Cat B: %d). "
        "Top score: %.3f, Median: %.3f",
        len(df),
        (df["category"] == "A").sum(),
        (df["category"] == "B").sum(),
        df["score"].iloc[0] if len(df) > 0 else 0,
        df["score"].median(),
    )
    return df
