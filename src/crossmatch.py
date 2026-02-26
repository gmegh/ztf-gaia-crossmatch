"""Positional cross-match between ZTF and Gaia catalogs."""

import logging

import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
import pandas as pd

from .config import XMATCH_RADIUS_ARCSEC

logger = logging.getLogger(__name__)


def _pivot_ztf_by_filter(ztf_df):
    """Pivot ZTF per-(oid, filter) rows into one row per oid.

    The ZTF objects table has one row per (oid, filter). We aggregate
    across filters so each physical source is a single row with
    per-filter columns and best-band summary stats.
    """
    # Gator returns lowercase column names; normalise for safety
    ztf_df.columns = [c.lower() for c in ztf_df.columns]

    # Keep one positional entry per oid (they're the same across filters)
    pos = ztf_df.groupby("oid")[["ra", "dec"]].first()

    # Pivot variability stats by filter
    filter_map = {1: "g", 2: "r", 3: "i"}
    ztf_df = ztf_df.copy()
    ztf_df["band"] = ztf_df["fid"].map(filter_map)

    stat_cols = [
        "nobs", "ngoodobs", "meanmag", "medianmag", "minmag", "maxmag",
        "magrms", "weightedmagrms", "weightedmeanmag",
        "chisq", "medmagerr",
        "stetsonj", "stetsonk", "vonneumannratio",
        "skewness", "smallkurtosis",
    ]

    pivoted = pos.copy()
    pivoted["n_filters"] = ztf_df.groupby("oid")["fid"].nunique()
    pivoted["transient"] = ztf_df.groupby("oid")["transient"].max()

    for col in stat_cols:
        if col not in ztf_df.columns:
            continue
        piv = ztf_df.pivot_table(index="oid", columns="band", values=col)
        piv.columns = [f"{col}_{b}" for b in piv.columns]
        pivoted = pivoted.join(piv, how="left")

    # Best-band summary: pick band with most observations
    for band in ["g", "r", "i"]:
        col = f"nobs_{band}"
        if col not in pivoted.columns:
            pivoted[col] = 0
    nobs_cols = [f"nobs_{b}" for b in ["g", "r", "i"]]
    nobs_arr = pivoted[nobs_cols].fillna(0).values
    best_idx = np.argmax(nobs_arr, axis=1)
    bands = ["g", "r", "i"]
    rows = np.arange(len(pivoted))

    def _pick_best(col_base):
        cols = [f"{col_base}_{b}" for b in bands]
        for c in cols:
            if c not in pivoted.columns:
                pivoted[c] = np.nan
        arr = pivoted[cols].values
        return arr[rows, best_idx]

    pivoted["best_nobs"] = _pick_best("nobs")
    pivoted["best_magrms"] = _pick_best("magrms")
    pivoted["best_chisq"] = _pick_best("chisq")
    pivoted["best_meanmag"] = _pick_best("meanmag")
    pivoted["best_minmag"] = _pick_best("minmag")
    pivoted["best_maxmag"] = _pick_best("maxmag")
    pivoted["best_amplitude"] = pivoted["best_maxmag"] - pivoted["best_minmag"]
    pivoted["best_stetsonj"] = _pick_best("stetsonj")
    pivoted["best_vonneumannratio"] = _pick_best("vonneumannratio")

    return pivoted.reset_index()


def crossmatch(ztf_table, gaia_sources_table, gaia_variables_table):
    """Positional cross-match ZTF objects against Gaia DR3.

    Parameters
    ----------
    ztf_table : astropy.table.Table
        ZTF DR23 objects (one row per oid+filter).
    gaia_sources_table : astropy.table.Table
        Gaia DR3 source catalog.
    gaia_variables_table : astropy.table.Table
        Gaia DR3 variable summary (source_id, ra, dec).

    Returns
    -------
    cat_a : pd.DataFrame
        Category A — ZTF sources with no Gaia counterpart within radius.
    cat_b : pd.DataFrame
        Category B — ZTF sources matched to Gaia but not in vari_summary.
    cat_c : pd.DataFrame
        Category C — Gaia sources with no ZTF counterpart within radius.
    """
    # Pivot ZTF to one row per source
    ztf_df = _pivot_ztf_by_filter(ztf_table.to_pandas())
    gaia_src_df = gaia_sources_table.to_pandas()
    # Normalise Gaia column names to lowercase
    gaia_src_df.columns = [c.lower() for c in gaia_src_df.columns]

    gaia_var_ids = set(gaia_variables_table["source_id"])
    logger.info(
        "Cross-matching %d ZTF sources against %d Gaia sources "
        "(%d known variables)",
        len(ztf_df), len(gaia_src_df), len(gaia_var_ids),
    )

    ztf_coords = SkyCoord(
        ra=ztf_df["ra"].values * u.deg,
        dec=ztf_df["dec"].values * u.deg,
    )
    gaia_coords = SkyCoord(
        ra=gaia_src_df["ra"].values * u.deg,
        dec=gaia_src_df["dec"].values * u.deg,
    )

    # ── Forward match: ZTF → Gaia ──────────────────────────────────────
    idx, sep2d, _ = ztf_coords.match_to_catalog_sky(gaia_coords)
    matched = sep2d.arcsec <= XMATCH_RADIUS_ARCSEC

    # Category A: ZTF source with no Gaia match
    cat_a = ztf_df[~matched].copy()
    cat_a["category"] = "A"
    cat_a["gaia_source_id"] = np.nan
    cat_a["gaia_g_mag"] = np.nan
    logger.info("Category A (ZTF, no Gaia match): %d sources", len(cat_a))

    # Category B: ZTF matched to Gaia but NOT a known variable
    matched_ztf = ztf_df[matched].copy()
    matched_gaia_idx = idx[matched]
    matched_ztf["gaia_source_id"] = gaia_src_df.iloc[matched_gaia_idx][
        "source_id"
    ].values
    matched_ztf["gaia_g_mag"] = gaia_src_df.iloc[matched_gaia_idx][
        "phot_g_mean_mag"
    ].values

    is_known_var = matched_ztf["gaia_source_id"].isin(gaia_var_ids)
    cat_b = matched_ztf[~is_known_var].copy()
    cat_b["category"] = "B"
    logger.info(
        "Category B (ZTF+Gaia, not variable): %d sources "
        "(discarded %d known variables)",
        len(cat_b), is_known_var.sum(),
    )

    # ── Reverse match: Gaia → ZTF ─────────────────────────────────────
    gaia_idx, gaia_sep2d, _ = gaia_coords.match_to_catalog_sky(ztf_coords)
    gaia_has_ztf = gaia_sep2d.arcsec <= XMATCH_RADIUS_ARCSEC

    unmatched_gaia = gaia_src_df[~gaia_has_ztf].copy()
    # Exclude known Gaia variables — we want sources Gaia sees but
    # does NOT flag as variable and ZTF doesn't detect at all
    is_gaia_var = unmatched_gaia["source_id"].isin(gaia_var_ids)
    cat_c = unmatched_gaia[~is_gaia_var].copy()
    cat_c["category"] = "C"
    cat_c["oid"] = np.nan  # no ZTF counterpart
    cat_c["gaia_source_id"] = cat_c["source_id"]
    cat_c["gaia_g_mag"] = cat_c["phot_g_mean_mag"]
    logger.info(
        "Category C (Gaia, no ZTF match): %d sources "
        "(discarded %d known variables)",
        len(cat_c), is_gaia_var.sum(),
    )

    return cat_a, cat_b, cat_c
