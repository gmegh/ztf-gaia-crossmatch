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


def score_candidates(cat_a, cat_b, cat_c):
    """Score and rank cross-match candidates.

    Categories A and B are scored on ZTF variability metrics.
    Category C (Gaia-only, no ZTF) is scored on Gaia photometry
    and galactic latitude.

    Parameters
    ----------
    cat_a : pd.DataFrame
        Category A candidates (ZTF source, no Gaia match).
    cat_b : pd.DataFrame
        Category B candidates (ZTF+Gaia match, not known variable).
    cat_c : pd.DataFrame
        Category C candidates (Gaia source, no ZTF match).

    Returns
    -------
    pd.DataFrame
        Combined, scored, and sorted candidates.
    """
    df = pd.concat([cat_a, cat_b, cat_c], ignore_index=True)
    if len(df) == 0:
        logger.warning("No candidates to score")
        df["score"] = []
        return df

    is_ztf = df["category"].isin(["A", "B"])
    is_c = df["category"] == "C"

    # Galactic latitude (applies to all categories)
    coords = SkyCoord(ra=df["ra"].values * u.deg, dec=df["dec"].values * u.deg)
    gal = coords.galactic
    df["gal_lat_abs"] = np.abs(gal.b.deg)

    score = np.zeros(len(df))

    # ── Score Categories A & B (ZTF variability metrics) ───────────────
    if is_ztf.any():
        amplitude = df.loc[is_ztf, "best_amplitude"].fillna(0).values
        nobs = df.loc[is_ztf, "best_nobs"].fillna(0).values
        chi2 = df.loc[is_ztf, "best_chisq"].fillna(0).values
        n_filters = df.loc[is_ztf, "n_filters"].fillna(0).values

        chi2_red = np.where(nobs > 1, chi2 / (nobs - 1), 0.0)
        df.loc[is_ztf, "chi2_red"] = chi2_red

        w = SCORING_WEIGHTS
        s = np.zeros(is_ztf.sum())
        s += w["amplitude"] * _normalise(amplitude)
        s += w["nobs"] * _normalise(nobs)
        s += w["chi2_red"] * _normalise(chi2_red)
        s += w["n_filters"] * _normalise(n_filters)
        s += w["gal_lat"] * _normalise(df.loc[is_ztf, "gal_lat_abs"].values)
        score[is_ztf.values] = s

    # Gaia G-mag bonus for Category B (brighter = higher score)
    is_b = df["category"] == "B"
    if is_b.any():
        g_mag = df.loc[is_b, "gaia_g_mag"].values
        g_score = _normalise(-g_mag)
        w = SCORING_WEIGHTS
        score[is_b.values] += w["gaia_g_mag"] * g_score

    # ── Score Category C (Gaia-only: brightness + galactic latitude) ───
    if is_c.any():
        g_mag_c = df.loc[is_c, "gaia_g_mag"].fillna(21).values
        gal_lat_c = df.loc[is_c, "gal_lat_abs"].values

        # Brighter Gaia sources that ZTF misses are more interesting
        s_c = 0.6 * _normalise(-g_mag_c) + 0.4 * _normalise(gal_lat_c)
        score[is_c.values] = s_c
        df.loc[is_c, "chi2_red"] = np.nan

    df["score"] = score
    df = df.sort_values("score", ascending=False).reset_index(drop=True)

    n_a = (df["category"] == "A").sum()
    n_b = (df["category"] == "B").sum()
    n_c = (df["category"] == "C").sum()
    logger.info(
        "Scored %d candidates (Cat A: %d, Cat B: %d, Cat C: %d). "
        "Top score: %.3f, Median: %.3f",
        len(df), n_a, n_b, n_c,
        df["score"].iloc[0] if len(df) > 0 else 0,
        df["score"].median(),
    )
    return df
