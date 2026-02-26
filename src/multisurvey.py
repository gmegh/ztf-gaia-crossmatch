"""Multi-survey cross-checks: 2MASS, WISE, PS1, SDSS, TNS."""

import logging

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.vizier import Vizier
import requests

logger = logging.getLogger(__name__)

# VizieR catalog IDs for cone searches
VIZIER_CATALOGS = {
    "2MASS": {
        "catalog": "II/246/out",
        "columns": ["RAJ2000", "DEJ2000", "Jmag", "Hmag", "Kmag"],
        "label": "2MASS",
    },
    "AllWISE": {
        "catalog": "II/328/allwise",
        "columns": ["RAJ2000", "DEJ2000", "W1mag", "W2mag", "W3mag", "W4mag"],
        "label": "AllWISE",
    },
    "PS1": {
        "catalog": "II/349/ps1",
        "columns": ["RAJ2000", "DEJ2000", "gmag", "rmag", "imag", "zmag", "ymag"],
        "label": "Pan-STARRS1",
    },
    "SDSS": {
        "catalog": "V/147/sdss12",
        "columns": ["RA_ICRS", "DE_ICRS", "umag", "gmag", "rmag", "imag", "zmag", "class"],
        "label": "SDSS DR12",
    },
}

TNS_API_URL = "https://www.wis-tns.org/api/get/search"


def _vizier_cone_search(ra, dec, catalog_key, radius_arcsec=3.0):
    """Query a VizieR catalog via cone search.

    Returns
    -------
    dict or None
        Dictionary of matched columns, or None if no match.
    """
    cat_info = VIZIER_CATALOGS[catalog_key]
    v = Vizier(columns=cat_info["columns"], row_limit=1)

    coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
    try:
        result = v.query_region(coord, radius=radius_arcsec * u.arcsec,
                                catalog=cat_info["catalog"])
        if result and len(result) > 0 and len(result[0]) > 0:
            row = result[0][0]
            return {col: float(row[col]) if hasattr(row[col], '__float__')
                    else str(row[col])
                    for col in row.colnames}
    except Exception as exc:
        logger.debug("VizieR %s query failed for (%.5f, %.5f): %s",
                     catalog_key, ra, dec, exc)
    return None


def _query_sdss_region(candidates):
    """Query SDSS DR12 for the coordinate region covered by candidates.

    Returns
    -------
    pd.DataFrame or None
        SDSS sources in the region, or None on failure.
    """
    ra_min = candidates["ra"].min() - 0.01
    ra_max = candidates["ra"].max() + 0.01
    dec_min = candidates["dec"].min() - 0.01
    dec_max = candidates["dec"].max() + 0.01

    v = Vizier(columns=["RA_ICRS", "DE_ICRS", "gmag"], row_limit=-1)
    try:
        center = SkyCoord(
            ra=(ra_min + ra_max) / 2 * u.deg,
            dec=(dec_min + dec_max) / 2 * u.deg,
        )
        width = max(ra_max - ra_min, dec_max - dec_min) * u.deg
        result = v.query_region(
            center, width=width, height=width,
            catalog="V/147/sdss12",
        )
        if not result or len(result) == 0 or len(result[0]) == 0:
            return None
        return result[0].to_pandas()
    except Exception as exc:
        logger.warning("SDSS bulk query failed: %s", exc)
        return None


def filter_sdss(candidates, cat_a_radius=1.0, cat_c_radius=3.0):
    """Remove Cat A and Cat C candidates that have an SDSS counterpart.

    Cat C: if a Gaia source exists in SDSS but not in ZTF, it's likely
    a ZTF coverage/processing issue — remove it.
    Cat A: if a ZTF source has no Gaia match but does have an SDSS match,
    the source is known and likely not a novel transient.

    Parameters
    ----------
    candidates : pd.DataFrame
        Scored candidates (all categories).
    cat_a_radius : float
        Match radius (arcsec) for Cat A SDSS cross-match.
    cat_c_radius : float
        Match radius (arcsec) for Cat C SDSS cross-match.

    Returns
    -------
    pd.DataFrame
        Candidates with SDSS-matched Cat A/C entries removed.
    """
    sdss = _query_sdss_region(candidates)
    if sdss is None:
        logger.info("SDSS filter: no SDSS sources in region, skipping")
        return candidates

    logger.info("SDSS filter: %d SDSS sources in region", len(sdss))

    sdss_coords = SkyCoord(
        ra=sdss["RA_ICRS"].values * u.deg,
        dec=sdss["DE_ICRS"].values * u.deg,
    )

    total_removed = 0

    # ── Filter Cat C ────────────────────────────────────────────────
    cat_c = candidates[candidates["category"] == "C"]
    if len(cat_c) > 0:
        n_before_c = len(cat_c)
        coords_c = SkyCoord(
            ra=cat_c["ra"].values * u.deg,
            dec=cat_c["dec"].values * u.deg,
        )
        _, sep_c, _ = coords_c.match_to_catalog_sky(sdss_coords)
        has_sdss_c = sep_c.arcsec <= cat_c_radius

        remove_ids_c = set(
            cat_c[has_sdss_c]["gaia_source_id"].apply(
                lambda x: str(int(x)) if pd.notna(x) else ""
            )
        )
        if remove_ids_c:
            mask_c = candidates["gaia_source_id"].apply(
                lambda x: str(int(x)) if pd.notna(x) else ""
            ).isin(remove_ids_c)
            candidates = candidates[~mask_c].reset_index(drop=True)

        n_removed_c = int(has_sdss_c.sum())
        total_removed += n_removed_c
        logger.info(
            "SDSS filter: removed %d / %d Cat C (SDSS match within %.1f\")",
            n_removed_c, n_before_c, cat_c_radius,
        )

    # ── Filter Cat A ────────────────────────────────────────────────
    cat_a = candidates[candidates["category"] == "A"]
    if len(cat_a) > 0:
        n_before_a = len(cat_a)
        coords_a = SkyCoord(
            ra=cat_a["ra"].values * u.deg,
            dec=cat_a["dec"].values * u.deg,
        )
        _, sep_a, _ = coords_a.match_to_catalog_sky(sdss_coords)
        has_sdss_a = sep_a.arcsec <= cat_a_radius

        remove_idx_a = cat_a.index[has_sdss_a]
        candidates = candidates.drop(remove_idx_a).reset_index(drop=True)

        n_removed_a = int(has_sdss_a.sum())
        total_removed += n_removed_a
        logger.info(
            "SDSS filter: removed %d / %d Cat A (SDSS match within %.1f\")",
            n_removed_a, n_before_a, cat_a_radius,
        )

    # ── Filter Cat B ────────────────────────────────────────────────
    cat_b = candidates[candidates["category"] == "B"]
    if len(cat_b) > 0:
        n_before_b = len(cat_b)
        coords_b = SkyCoord(
            ra=cat_b["ra"].values * u.deg,
            dec=cat_b["dec"].values * u.deg,
        )
        _, sep_b, _ = coords_b.match_to_catalog_sky(sdss_coords)
        has_sdss_b = sep_b.arcsec <= cat_c_radius

        remove_idx_b = cat_b.index[has_sdss_b]
        candidates = candidates.drop(remove_idx_b).reset_index(drop=True)

        n_removed_b = int(has_sdss_b.sum())
        total_removed += n_removed_b
        logger.info(
            "SDSS filter: removed %d / %d Cat B (SDSS match within %.1f\")",
            n_removed_b, n_before_b, cat_c_radius,
        )

    logger.info("SDSS filter: %d total candidates removed", total_removed)
    return candidates


def crosscheck_source(ra, dec):
    """Cross-check a single source against all external surveys.

    Parameters
    ----------
    ra, dec : float
        Position in degrees.

    Returns
    -------
    dict
        Keys are survey names, values are match dicts or None.
    """
    results = {}
    for key in VIZIER_CATALOGS:
        results[key] = _vizier_cone_search(ra, dec, key)
    return results


def crosscheck_candidates(candidates, n_top=50):
    """Cross-check top candidates against external surveys.

    Parameters
    ----------
    candidates : pd.DataFrame
        Scored candidates.
    n_top : int
        Number of top candidates to cross-check.

    Returns
    -------
    dict
        Mapping of identifier → survey results dict.
    """
    top = candidates.head(n_top)
    logger.info("Multi-survey cross-check for top %d candidates", len(top))

    all_results = {}
    for i, (_, row) in enumerate(top.iterrows()):
        cat = row["category"]
        if cat == "C":
            ident = str(int(row["gaia_source_id"]))
        else:
            ident = str(int(row["oid"])) if pd.notna(row["oid"]) else str(int(row["gaia_source_id"]))

        ra, dec = row["ra"], row["dec"]
        results = crosscheck_source(ra, dec)

        matched_surveys = [k for k, v in results.items() if v is not None]
        logger.info(
            "  [%d/%d] %s (Cat %s): matched %s",
            i + 1, len(top), ident, cat,
            ", ".join(matched_surveys) if matched_surveys else "none",
        )
        all_results[ident] = results

    return all_results
