"""Candidate scoring and ranking for cross-match results."""

import logging

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u

from .config import SCORING_WEIGHTS

logger = logging.getLogger(__name__)


def _normalise(arr):
    """Min-max normalise to [0, 1], handling constant arrays."""
    arr = np.asarray(arr, dtype=float)
    lo, hi = np.nanmin(arr), np.nanmax(arr)
    if hi == lo:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def score_candidates(cat_a, cat_b):
    """Score and rank cross-match candidates.

    Uses pre-computed best-band stats from the crossmatch pivot step:
    best_amplitude, best_nobs, best_chiSQ, n_filters, plus galactic
    latitude and Gaia G-mag for Category B.

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

    amplitude = df["best_amplitude"].fillna(0).values
    nobs = df["best_nobs"].fillna(0).values
    chi2 = df["best_chisq"].fillna(0).values
    n_filters = df["n_filters"].fillna(0).values

    # Reduced chi-squared
    chi2_red = np.where(nobs > 1, chi2 / (nobs - 1), 0.0)
    df["chi2_red"] = chi2_red

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
