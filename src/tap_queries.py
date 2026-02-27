"""Query functions for ZTF DR23 (IRSA Gator) and Gaia DR3 (ESA TAP)."""

import base64
import configparser
import io
import math
import os
import time
import logging

import numpy as np
import requests
from astroquery.utils.tap import TapPlus
from astropy.table import Table, vstack

from .config import (
    IRSA_TAP_URL,
    GAIA_TAP_URL,
    TAP_MAX_RETRIES,
    TAP_RETRY_DELAY,
    TAP_QUERY_TIMEOUT,
)

logger = logging.getLogger(__name__)

IRSA_LOGIN_URL = "https://irsa.ipac.caltech.edu/account/signon/login.do"
IRSA_GATOR_URL = "https://irsa.ipac.caltech.edu/cgi-bin/Gator/nph-query"

# Lightweight columns for Cat C-only processing (positions + brightness)
ZTF_POSITION_COLUMNS = ["oid", "ra", "dec", "fid", "meanMag"]

# Columns we need from ZTF objects catalog
ZTF_COLUMNS = [
    "oid", "ra", "dec", "fid", "filtercode", "transient",
    "nobs", "ngoodobs",
    "meanMag", "medianMag", "minMag", "maxMag",
    "magRMS", "weightedMagRMS", "weightedMeanMag",
    "chiSQ", "medMagErr",
    "stetsonJ", "stetsonK", "vonNeumannRatio",
    "skewness", "smallKurtosis",
    "astrometricRMS",
]


def _load_irsa_credentials():
    """Load IRSA credentials from ~/.ztfquery (ztfquery format)."""
    path = os.path.expanduser("~/.ztfquery")
    if not os.path.exists(path):
        raise FileNotFoundError(
            "No ~/.ztfquery file found. Create one with IRSA credentials."
        )
    config = configparser.ConfigParser()
    config.read(path)
    username = config["irsa"]["username"]
    password_raw = config["irsa"]["password"]
    password = base64.b64decode(password_raw[2:-1]).decode("utf-8")
    return username, password


def _get_irsa_session():
    """Create an authenticated requests.Session for IRSA."""
    username, password = _load_irsa_credentials()
    session = requests.Session()
    login_url = (
        f"{IRSA_LOGIN_URL}?josso_cmd=login"
        f"&josso_username={username}&josso_password={password}"
    )
    session.get(login_url)
    logger.info("Authenticated with IRSA as %s", username)
    return session


GATOR_MAX_BOX_ARCSEC = 1200  # Gator limit for box searches


def _gator_box_query(session, ra_cen, dec_cen, box_arcsec, description="",
                     columns=None):
    """Single Gator box query with retry."""
    params = {
        "catalog": "ztf_objects_dr23",
        "spatial": "box",
        "objstr": f"{ra_cen} {dec_cen}",
        "size": str(int(box_arcsec)),
        "outfmt": "3",  # VOTable
        "selcols": ",".join(columns or ZTF_COLUMNS),
    }
    for attempt in range(1, TAP_MAX_RETRIES + 1):
        try:
            resp = session.get(
                IRSA_GATOR_URL, params=params, timeout=TAP_QUERY_TIMEOUT
            )
            resp.raise_for_status()
            table = Table.read(io.BytesIO(resp.content), format="votable")
            return table
        except Exception as exc:
            logger.warning(
                "%s attempt %d failed: %s", description, attempt, exc
            )
            if attempt < TAP_MAX_RETRIES:
                time.sleep(TAP_RETRY_DELAY)
            else:
                raise


def query_ztf_objects(ra_min, ra_max, dec_min, dec_max):
    """Query ZTF DR23 objects in a bounding box via IRSA Gator.

    Automatically chunks large areas into sub-boxes of ≤1200 arcsec
    (the Gator limit) and deduplicates results by (oid, fid).
    """
    ra_size_arcsec = (ra_max - ra_min) * 3600
    dec_size_arcsec = (dec_max - dec_min) * 3600

    session = _get_irsa_session()

    # If small enough, single query
    if max(ra_size_arcsec, dec_size_arcsec) <= GATOR_MAX_BOX_ARCSEC:
        ra_cen = (ra_min + ra_max) / 2
        dec_cen = (dec_min + dec_max) / 2
        box = max(ra_size_arcsec, dec_size_arcsec)
        logger.info(
            "ZTF Gator query: single box %.0f arcsec at (%.3f, %.3f)",
            box, ra_cen, dec_cen,
        )
        return _gator_box_query(session, ra_cen, dec_cen, box, "ZTF Gator")

    # Chunk into sub-boxes
    step_deg = GATOR_MAX_BOX_ARCSEC / 3600  # 0.333 deg
    ra_centers = np.arange(
        ra_min + step_deg / 2, ra_max, step_deg
    )
    dec_centers = np.arange(
        dec_min + step_deg / 2, dec_max, step_deg
    )
    n_chunks = len(ra_centers) * len(dec_centers)
    logger.info(
        "ZTF Gator query: area %.2f x %.2f deg → %d sub-boxes of %d arcsec",
        ra_max - ra_min, dec_max - dec_min, n_chunks, GATOR_MAX_BOX_ARCSEC,
    )

    tables = []
    for i, ra_c in enumerate(ra_centers):
        for j, dec_c in enumerate(dec_centers):
            chunk_id = i * len(dec_centers) + j + 1
            desc = f"ZTF chunk {chunk_id}/{n_chunks}"
            try:
                t = _gator_box_query(
                    session, ra_c, dec_c, GATOR_MAX_BOX_ARCSEC, desc
                )
                if len(t) > 0:
                    tables.append(t)
                    logger.info("  %s: %d rows", desc, len(t))
            except Exception as exc:
                logger.warning("  %s failed, skipping: %s", desc, exc)

    if not tables:
        return Table()

    combined = vstack(tables)
    # Deduplicate by (oid, fid)
    df = combined.to_pandas()
    before = len(df)
    df = df.drop_duplicates(subset=["oid", "fid"])
    logger.info(
        "ZTF Gator: %d total rows, %d after dedup (removed %d duplicates)",
        before, len(df), before - len(df),
    )
    return Table.from_pandas(df)


def query_ztf_positions(ra_min, ra_max, dec_min, dec_max):
    """Query ZTF DR23 positions and mean magnitudes via IRSA Gator.

    Lightweight variant of query_ztf_objects() that fetches only
    oid, ra, dec, fid, meanMag — sufficient for Cat C cross-matching.
    """
    ra_size_arcsec = (ra_max - ra_min) * 3600
    dec_size_arcsec = (dec_max - dec_min) * 3600

    session = _get_irsa_session()

    if max(ra_size_arcsec, dec_size_arcsec) <= GATOR_MAX_BOX_ARCSEC:
        ra_cen = (ra_min + ra_max) / 2
        dec_cen = (dec_min + dec_max) / 2
        box = max(ra_size_arcsec, dec_size_arcsec)
        logger.info(
            "ZTF positions Gator query: single box %.0f arcsec at (%.3f, %.3f)",
            box, ra_cen, dec_cen,
        )
        return _gator_box_query(
            session, ra_cen, dec_cen, box, "ZTF positions",
            columns=ZTF_POSITION_COLUMNS,
        )

    step_deg = GATOR_MAX_BOX_ARCSEC / 3600
    ra_centers = np.arange(ra_min + step_deg / 2, ra_max, step_deg)
    dec_centers = np.arange(dec_min + step_deg / 2, dec_max, step_deg)
    n_chunks = len(ra_centers) * len(dec_centers)
    logger.info(
        "ZTF positions Gator query: area %.2f x %.2f deg → %d sub-boxes",
        ra_max - ra_min, dec_max - dec_min, n_chunks,
    )

    tables = []
    for i, ra_c in enumerate(ra_centers):
        for j, dec_c in enumerate(dec_centers):
            chunk_id = i * len(dec_centers) + j + 1
            desc = f"ZTF pos chunk {chunk_id}/{n_chunks}"
            try:
                t = _gator_box_query(
                    session, ra_c, dec_c, GATOR_MAX_BOX_ARCSEC, desc,
                    columns=ZTF_POSITION_COLUMNS,
                )
                if len(t) > 0:
                    tables.append(t)
                    logger.info("  %s: %d rows", desc, len(t))
            except Exception as exc:
                logger.warning("  %s failed, skipping: %s", desc, exc)

    if not tables:
        return Table()

    combined = vstack(tables)
    df = combined.to_pandas()
    before = len(df)
    df = df.drop_duplicates(subset=["oid", "fid"])
    logger.info(
        "ZTF positions: %d total rows, %d after dedup (removed %d duplicates)",
        before, len(df), before - len(df),
    )
    return Table.from_pandas(df)


def _run_gaia_tap_query(adql, description="Gaia TAP query"):
    """Execute an async TAP query against Gaia ESA (no row limit)."""
    tap = TapPlus(url=GAIA_TAP_URL)
    for attempt in range(1, TAP_MAX_RETRIES + 1):
        try:
            logger.info(
                "%s (attempt %d/%d)", description, attempt, TAP_MAX_RETRIES
            )
            job = tap.launch_job_async(adql, verbose=False)
            table = job.get_results()
            # Normalise column names to lowercase
            for col in table.colnames:
                if col != col.lower():
                    table.rename_column(col, col.lower())
            logger.info(
                "%s returned %d rows", description, len(table)
            )
            return table
        except Exception as exc:
            logger.warning(
                "%s attempt %d failed: %s", description, attempt, exc
            )
            if attempt < TAP_MAX_RETRIES:
                time.sleep(TAP_RETRY_DELAY)
            else:
                raise RuntimeError(
                    f"{description} failed after {TAP_MAX_RETRIES} attempts"
                ) from exc


def query_gaia_sources(ra_min, ra_max, dec_min, dec_max):
    """Query Gaia DR3 sources in a bounding box."""
    adql = f"""
    SELECT
        source_id, ra, dec, phot_g_mean_mag,
        phot_bp_mean_mag, phot_rp_mean_mag, parallax, pmra, pmdec
    FROM gaiadr3.gaia_source
    WHERE ra BETWEEN {ra_min} AND {ra_max}
      AND dec BETWEEN {dec_min} AND {dec_max}
    """
    return _run_gaia_tap_query(adql, "Gaia sources query")


def query_gaia_variables(ra_min, ra_max, dec_min, dec_max):
    """Query Gaia DR3 variable summary in a bounding box."""
    adql = f"""
    SELECT
        vs.source_id, gs.ra, gs.dec
    FROM gaiadr3.vari_summary AS vs
    JOIN gaiadr3.gaia_source AS gs USING (source_id)
    WHERE gs.ra BETWEEN {ra_min} AND {ra_max}
      AND gs.dec BETWEEN {dec_min} AND {dec_max}
    """
    return _run_gaia_tap_query(adql, "Gaia variables query")
