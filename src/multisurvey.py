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
