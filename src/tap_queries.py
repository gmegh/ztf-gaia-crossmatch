"""TAP service query functions for ZTF DR23 and Gaia DR3."""

import time
import logging

from astroquery.utils.tap import TapPlus
import astropy.table

from .config import (
    IRSA_TAP_URL,
    GAIA_TAP_URL,
    TAP_MAX_RETRIES,
    TAP_RETRY_DELAY,
    TAP_QUERY_TIMEOUT,
)

logger = logging.getLogger(__name__)


def _run_tap_query(tap_url, adql, description="TAP query"):
    """Execute a synchronous TAP query with retry logic.

    Parameters
    ----------
    tap_url : str
        TAP service base URL.
    adql : str
        ADQL query string.
    description : str
        Human-readable label for logging.

    Returns
    -------
    astropy.table.Table
    """
    tap = TapPlus(url=tap_url)
    for attempt in range(1, TAP_MAX_RETRIES + 1):
        try:
            logger.info(
                "%s (attempt %d/%d)", description, attempt, TAP_MAX_RETRIES
            )
            job = tap.launch_job(adql, verbose=False)
            table = job.get_results()
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


def query_ztf_objects(ra_min, ra_max, dec_min, dec_max):
    """Query ZTF DR23 objects in a bounding box.

    Returns columns needed for cross-match and scoring:
    oid, ra, dec, nobs_g/r/i, mean_mag_g/r/i, chi2_g/r/i, etc.
    """
    adql = f"""
    SELECT
        oid, ra, dec,
        nobs_g, nobs_r, nobs_i,
        mean_mag_g, mean_mag_r, mean_mag_i,
        rms_mag_g, rms_mag_r, rms_mag_i,
        chi2_g, chi2_r, chi2_i
    FROM ztf_objects_dr23
    WHERE ra BETWEEN {ra_min} AND {ra_max}
      AND dec BETWEEN {dec_min} AND {dec_max}
    """
    return _run_tap_query(IRSA_TAP_URL, adql, "ZTF objects query")


def query_gaia_sources(ra_min, ra_max, dec_min, dec_max):
    """Query Gaia DR3 sources in a bounding box.

    Returns positional and photometric columns.
    """
    adql = f"""
    SELECT
        source_id, ra, dec, phot_g_mean_mag,
        phot_bp_mean_mag, phot_rp_mean_mag, parallax, pmra, pmdec
    FROM gaiadr3.gaia_source
    WHERE ra BETWEEN {ra_min} AND {ra_max}
      AND dec BETWEEN {dec_min} AND {dec_max}
    """
    return _run_tap_query(GAIA_TAP_URL, adql, "Gaia sources query")


def query_gaia_variables(ra_min, ra_max, dec_min, dec_max):
    """Query Gaia DR3 variable summary in a bounding box.

    Returns source_ids of known Gaia variables.
    """
    adql = f"""
    SELECT
        vs.source_id, gs.ra, gs.dec
    FROM gaiadr3.vari_summary AS vs
    JOIN gaiadr3.gaia_source AS gs USING (source_id)
    WHERE gs.ra BETWEEN {ra_min} AND {ra_max}
      AND gs.dec BETWEEN {dec_min} AND {dec_max}
    """
    return _run_tap_query(GAIA_TAP_URL, adql, "Gaia variables query")
