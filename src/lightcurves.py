"""Light curve retrieval and plotting for ZTF candidates."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import requests

from .config import RESULTS_DIR
from .tap_queries import _get_irsa_session

logger = logging.getLogger(__name__)

IRSA_LIGHTCURVE_URL = (
    "https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves"
)
BAND_COLORS = {"zg": "#28a745", "zr": "#dc3545", "zi": "#fd7e14"}
BAND_LABELS = {"zg": "ZTF g", "zr": "ZTF r", "zi": "ZTF i"}

PLOTS_DIR = RESULTS_DIR / "lightcurves"


def fetch_lightcurve(oid=None, ra=None, dec=None, radius_arcsec=2.0, session=None):
    """Fetch ZTF light curve data via IRSA.

    Queries by OID if available (faster, exact match), otherwise by position.

    Parameters
    ----------
    oid : int or str, optional
        ZTF object identifier.
    ra, dec : float, optional
        Position in degrees (used if oid is None).
    radius_arcsec : float
        Cone search radius (only for positional queries).
    session : requests.Session, optional
        Authenticated IRSA session. Created if not provided.

    Returns
    -------
    pd.DataFrame or None
        Light curve with columns: mjd, mag, magerr, filtercode, catflags.
        None if no data found.
    """
    if session is None:
        session = _get_irsa_session()

    if oid is not None:
        params = {
            "ID": str(int(oid)),
            "BAD_CATFLAGS_MASK": "32768",
            "FORMAT": "CSV",
        }
    else:
        params = {
            "POS": f"CIRCLE {ra} {dec} {radius_arcsec / 3600}",
            "BAD_CATFLAGS_MASK": "32768",
            "FORMAT": "CSV",
        }

    try:
        resp = session.get(IRSA_LIGHTCURVE_URL, params=params, timeout=120)
        resp.raise_for_status()
        text = resp.text.strip()
        if len(text) == 0 or "No data" in text or "\n" not in text:
            return None

        from io import StringIO
        df = pd.read_csv(StringIO(text), comment="#")
        if len(df) == 0:
            return None

        # Normalise column names
        df.columns = [c.strip().lower() for c in df.columns]
        return df
    except Exception as exc:
        identifier = oid if oid is not None else f"({ra:.5f}, {dec:.5f})"
        logger.warning("Light curve fetch failed for %s: %s", identifier, exc)
        return None


def plot_lightcurve(lc_df, oid, output_path):
    """Plot a multi-band ZTF light curve.

    Parameters
    ----------
    lc_df : pd.DataFrame
        Light curve data with mjd, mag, magerr, filtercode columns.
    oid : str
        Object identifier for the title.
    output_path : Path
        Where to save the PNG.
    """
    fig, ax = plt.subplots(figsize=(10, 4))

    for fcode, group in lc_df.groupby("filtercode"):
        fcode = fcode.strip() if isinstance(fcode, str) else fcode
        color = BAND_COLORS.get(fcode, "#666666")
        label = BAND_LABELS.get(fcode, fcode)
        ax.errorbar(
            group["mjd"], group["mag"], yerr=group["magerr"],
            fmt=".", ms=3, alpha=0.6, color=color, label=label, elinewidth=0.5,
        )

    ax.invert_yaxis()
    ax.set_xlabel("MJD")
    ax.set_ylabel("Magnitude")
    ax.set_title(f"ZTF Light Curve — {oid}")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def fetch_and_plot_top_candidates(candidates, n_top=50):
    """Fetch light curves and create plots for top N candidates.

    Only applies to Categories A and B (ZTF sources).
    Category C sources (Gaia-only) have no ZTF light curves.

    Parameters
    ----------
    candidates : pd.DataFrame
        Scored candidate DataFrame.
    n_top : int
        Number of top candidates to fetch light curves for.

    Returns
    -------
    dict
        Mapping of oid → light curve PNG path (relative to RESULTS_DIR).
    """
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Only fetch for ZTF sources (Cat A and B)
    ztf_candidates = candidates[candidates["category"].isin(["A", "B"])].head(n_top)
    logger.info(
        "Fetching light curves for top %d ZTF candidates", len(ztf_candidates)
    )

    session = _get_irsa_session()
    lc_paths = {}

    for i, (_, row) in enumerate(ztf_candidates.iterrows()):
        oid_val = int(row["oid"]) if pd.notna(row["oid"]) else None
        oid_str = str(oid_val) if oid_val else "unknown"
        png_path = PLOTS_DIR / f"{oid_str}.png"

        if png_path.exists():
            lc_paths[oid_str] = f"lightcurves/{oid_str}.png"
            continue

        lc_df = fetch_lightcurve(oid=oid_val, ra=row["ra"], dec=row["dec"],
                                 session=session)
        if lc_df is not None and len(lc_df) > 0:
            plot_lightcurve(lc_df, oid_str, png_path)
            lc_paths[oid_str] = f"lightcurves/{oid_str}.png"
            logger.info(
                "  [%d/%d] %s: %d points, saved %s",
                i + 1, len(ztf_candidates), oid_str, len(lc_df), png_path.name,
            )
        else:
            logger.info("  [%d/%d] %s: no light curve data", i + 1, len(ztf_candidates), oid_str)

    logger.info("Light curves: %d/%d successful", len(lc_paths), len(ztf_candidates))
    return lc_paths
