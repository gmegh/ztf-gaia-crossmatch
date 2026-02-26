"""Configuration constants for the ZTF-Gaia cross-match pipeline."""

from pathlib import Path
import healpy as hp
import numpy as np

# ── Project paths ──────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
TILES_DIR = DATA_DIR / "tiles"
RESULTS_DIR = ROOT_DIR / "results"
WEBSITE_DIR = ROOT_DIR / "website"
TEMPLATES_DIR = ROOT_DIR / "templates"
PROGRESS_FILE = DATA_DIR / "progress.json"

# Ensure directories exist
for d in [DATA_DIR, TILES_DIR, RESULTS_DIR, WEBSITE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── TAP endpoints ──────────────────────────────────────────────────────
IRSA_TAP_URL = "https://irsa.ipac.caltech.edu/TAP"
GAIA_TAP_URL = "https://gea.esac.esa.int/tap-server/tap"

# ── HEALPix parameters ────────────────────────────────────────────────
NSIDE = 32
PILOT_RA = 180.0   # deg
PILOT_DEC = 30.0   # deg
PILOT_TILE = hp.ang2pix(NSIDE, PILOT_RA, PILOT_DEC, lonlat=True, nest=True)

# ── Cross-match ────────────────────────────────────────────────────────
XMATCH_RADIUS_ARCSEC = 1.5

# ── TAP query settings ────────────────────────────────────────────────
TAP_MAX_RETRIES = 3
TAP_RETRY_DELAY = 10  # seconds
TAP_QUERY_TIMEOUT = 300  # seconds

# ── Scoring weights (relative importance) ─────────────────────────────
SCORING_WEIGHTS = {
    "amplitude": 0.25,
    "nobs": 0.15,
    "chi2_red": 0.20,
    "n_filters": 0.10,
    "gal_lat": 0.10,
    "gaia_g_mag": 0.20,  # Category B only
}


def tile_bounds(tile_index):
    """Return (ra_min, ra_max, dec_min, dec_max) for a HEALPix tile.

    Uses the tile corner vertices to compute a bounding box.
    """
    boundaries = hp.boundaries(NSIDE, tile_index, step=1, nest=True)
    # boundaries shape: (3, 4) in Cartesian; convert to lon/lat
    lon, lat = hp.vec2ang(boundaries.T, lonlat=True)
    ra_min, ra_max = float(np.min(lon)), float(np.max(lon))
    dec_min, dec_max = float(np.min(lat)), float(np.max(lat))
    # Handle RA wrap-around (tile straddling 0/360)
    if ra_max - ra_min > 180:
        ra_min, ra_max = float(np.min(lon[lon > 180])), float(np.max(lon[lon < 180]))
    return ra_min, ra_max, dec_min, dec_max
