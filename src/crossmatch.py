"""Positional cross-match between ZTF and Gaia catalogs."""

import logging

import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
import pandas as pd

from .config import XMATCH_RADIUS_ARCSEC

logger = logging.getLogger(__name__)


def crossmatch(ztf_table, gaia_sources_table, gaia_variables_table):
    """Positional cross-match ZTF objects against Gaia DR3.

    Parameters
    ----------
    ztf_table : astropy.table.Table
        ZTF DR23 objects.
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
    """
    ztf_df = ztf_table.to_pandas()
    gaia_src_df = gaia_sources_table.to_pandas()

    # Build set of known Gaia variable source_ids
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

    # Match each ZTF source to its nearest Gaia neighbour
    idx, sep2d, _ = ztf_coords.match_to_catalog_sky(gaia_coords)
    matched = sep2d.arcsec <= XMATCH_RADIUS_ARCSEC

    # ── Category A: no Gaia match ──────────────────────────────────────
    cat_a = ztf_df[~matched].copy()
    cat_a["category"] = "A"
    cat_a["gaia_source_id"] = np.nan
    cat_a["gaia_g_mag"] = np.nan
    logger.info("Category A (no Gaia match): %d sources", len(cat_a))

    # ── Category B: Gaia match but NOT a known variable ────────────────
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
        "Category B (Gaia match, not variable): %d sources "
        "(discarded %d known variables)",
        len(cat_b), is_known_var.sum(),
    )

    return cat_a, cat_b
