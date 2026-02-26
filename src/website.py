"""Jinja2 static site generator for the ZTF-Gaia cross-match results."""

import json
import logging
import shutil

import healpy as hp
import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader

from .config import (
    WEBSITE_DIR, TEMPLATES_DIR, RESULTS_DIR, ROOT_DIR,
    NSIDE, XMATCH_RADIUS_ARCSEC, PROGRESS_FILE,
)
from .scoring import BRIGHT_MAG_LIMIT

logger = logging.getLogger(__name__)

CATEGORY_DESCRIPTIONS = {
    "A": "ZTF source with no Gaia counterpart within 1.5\u2033",
    "B": "ZTF source matched to Gaia, not a known Gaia variable",
    "C": "Gaia source with no ZTF counterpart within 1.5\u2033",
}

FULL_SKY_SQ_DEG = 41_252.96
TOTAL_TILES = 12 * NSIDE ** 2
TILE_AREA_SQ_DEG = FULL_SKY_SQ_DEG / TOTAL_TILES


def _safe_float(val):
    """Convert to float, returning None for NaN/None."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _source_id(row):
    """Get the display identifier for a candidate row."""
    if row["category"] == "C":
        return str(int(row["gaia_source_id"]))
    return str(int(row["oid"])) if pd.notna(row.get("oid")) else str(int(row["gaia_source_id"]))


def _sky_urls(ra, dec):
    """Build sky image and external viewer URLs for a position."""
    # ZTF cutout via CDS hips2fits service (g-band)
    ztf_hips_url = (
        f"https://alasky.cds.unistra.fr/hips-image-services/hips2fits"
        f"?hips=CDS/P/ZTF/DR7/g&width=256&height=256"
        f"&fov=0.03&projection=TAN&coordsys=icrs"
        f"&ra={ra:.6f}&dec={dec:.6f}&format=jpg"
    )

    # Aladin Lite interactive view (shows DSS2 by default; user can switch to Gaia overlay)
    aladin_url = (
        f"https://aladin.cds.unistra.fr/AladinLite/"
        f"?target={ra:.6f}+{dec:.6f}&fov=0.03&survey=CDS%2FP%2FDSS2%2Fcolor"
    )

    # Legacy Survey cutout (useful as external link)
    legacy_survey_url = (
        f"https://www.legacysurvey.org/viewer/cutout.jpg"
        f"?ra={ra:.6f}&dec={dec:.6f}&size=256&layer=ls-dr10&pixscale=0.262"
    )

    esasky_url = (
        f"https://sky.esa.int/esasky/?target={ra:.6f}+{dec:.6f}"
        f"&hips=DSS2+color&fov=0.05&cooframe=J2000&sci=true"
    )
    simbad_url = (
        f"https://simbad.u-strasbg.fr/simbad/sim-coo"
        f"?Coord={ra:.6f}+{dec:.6f}&CooFrame=FK5&CooEpoch=2000"
        f"&CooEqui=2000&CooDefinedFrames=none&Radius=10&Radius.unit=arcsec"
    )
    return {
        "ztf_hips_url": ztf_hips_url,
        "aladin_url": aladin_url,
        "legacy_survey_url": legacy_survey_url,
        "esasky_url": esasky_url,
        "simbad_url": simbad_url,
    }


def _coverage_stats():
    """Compute sky coverage statistics from progress file."""
    tiles_processed = 0
    if PROGRESS_FILE.exists():
        tiles_processed = len(json.loads(PROGRESS_FILE.read_text()))
    if tiles_processed == 0:
        tiles_processed = 1  # At least the pilot tile

    area_sq_deg = tiles_processed * TILE_AREA_SQ_DEG
    sky_fraction = area_sq_deg / FULL_SKY_SQ_DEG
    return {
        "tiles_processed": tiles_processed,
        "area_sq_deg": area_sq_deg,
        "sky_fraction": sky_fraction,
        "nside": NSIDE,
        "total_tiles": TOTAL_TILES,
        "xmatch_radius": XMATCH_RADIUS_ARCSEC,
        "bright_mag_limit": BRIGHT_MAG_LIMIT,
    }


def generate_website(candidates, lc_paths=None, survey_results=None, n_top=100):
    """Generate the static HTML website from candidate results.

    Parameters
    ----------
    candidates : pd.DataFrame
        Scored candidates DataFrame.
    lc_paths : dict, optional
        Mapping of identifier -> light curve PNG path (relative).
    survey_results : dict, optional
        Mapping of identifier -> {survey: match_dict} from multisurvey.
    n_top : int
        Number of top candidates to generate source pages for.
    """
    lc_paths = lc_paths or {}
    survey_results = survey_results or {}

    # Set up Jinja2
    env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)))
    env.globals["url_prefix"] = ""

    # Prepare output directories
    WEBSITE_DIR.mkdir(parents=True, exist_ok=True)
    sources_dir = WEBSITE_DIR / "sources"
    sources_dir.mkdir(exist_ok=True)

    # Copy static assets
    static_src = ROOT_DIR / "static"
    static_dst = WEBSITE_DIR / "static"
    if static_dst.exists():
        shutil.rmtree(static_dst)
    shutil.copytree(static_src, static_dst)

    # Copy light curve images
    lc_src = RESULTS_DIR / "lightcurves"
    lc_dst = WEBSITE_DIR / "lightcurves"
    if lc_src.exists():
        if lc_dst.exists():
            shutil.rmtree(lc_dst)
        shutil.copytree(lc_src, lc_dst)

    # Coverage stats
    coverage = _coverage_stats()

    # How many candidates had survey cross-checks
    n_surveyed = len(survey_results)

    # ── Build candidate data for index page ──────────────────────────
    # Select top candidates per category to ensure balanced representation
    # (scores are not directly comparable across categories)
    per_cat = max(1, n_top // 3)
    top_a = candidates[candidates["category"] == "A"].head(per_cat)
    top_b = candidates[candidates["category"] == "B"].head(per_cat)
    top_c = candidates[candidates["category"] == "C"].head(per_cat)
    top = pd.concat([top_a, top_b, top_c]).sort_values("score", ascending=False).reset_index(drop=True)
    index_candidates = []
    for rank, (_, row) in enumerate(top.iterrows(), 1):
        ident = _source_id(row)
        surveys = survey_results.get(ident, {})
        matched = [k for k, v in surveys.items() if v is not None]
        index_candidates.append({
            "rank": rank,
            "id": ident,
            "oid": str(int(row["oid"])) if pd.notna(row.get("oid")) else None,
            "ra": round(float(row["ra"]), 6),
            "dec": round(float(row["dec"]), 6),
            "category": row["category"],
            "score": round(float(row["score"]), 4),
            "amplitude": round(float(row["best_amplitude"]), 4) if pd.notna(row.get("best_amplitude")) else None,
            "nobs": int(row["best_nobs"]) if pd.notna(row.get("best_nobs")) else None,
            "gaia_g_mag": round(float(row["gaia_g_mag"]), 2) if pd.notna(row.get("gaia_g_mag")) else None,
            "survey_matches": ", ".join(matched) if matched else "",
        })

    # Serialize to JSON for embedding in template
    candidates_json = json.dumps(index_candidates, separators=(",", ":"))

    index_tmpl = env.get_template("index.html")
    index_html = index_tmpl.render(
        url_prefix="",
        total_candidates=len(candidates),
        cat_a_count=int((candidates["category"] == "A").sum()),
        cat_b_count=int((candidates["category"] == "B").sum()),
        cat_c_count=int((candidates["category"] == "C").sum()),
        candidates_json=candidates_json,
        n_surveyed=n_surveyed,
        **coverage,
    )
    (WEBSITE_DIR / "index.html").write_text(index_html)
    logger.info("Generated index.html with %d candidates", len(index_candidates))

    # ── Source pages ───────────────────────────────────────────────────
    source_tmpl = env.get_template("source.html")
    for _, row in top.iterrows():
        ident = _source_id(row)
        surveys = survey_results.get(ident, {})
        has_any_survey = any(v is not None for v in surveys.values())

        gaia_bp = _safe_float(row.get("phot_bp_mean_mag"))
        gaia_rp = _safe_float(row.get("phot_rp_mean_mag"))

        # Sky image URLs
        sky_urls = _sky_urls(float(row["ra"]), float(row["dec"]))

        source_data = {
            "id": ident,
            "oid": str(int(row["oid"])) if pd.notna(row.get("oid")) else None,
            "gaia_source_id": str(int(row["gaia_source_id"])) if pd.notna(row.get("gaia_source_id")) else None,
            "ra": row["ra"],
            "dec": row["dec"],
            "category": row["category"],
            "category_desc": CATEGORY_DESCRIPTIONS.get(row["category"], ""),
            "score": row["score"],
            "amplitude": _safe_float(row.get("best_amplitude")),
            "nobs": _safe_float(row.get("best_nobs")),
            "n_filters": _safe_float(row.get("n_filters")),
            "chi2_red": _safe_float(row.get("chi2_red")),
            "gaia_g_mag": _safe_float(row.get("gaia_g_mag")),
            "gaia_bp_mag": gaia_bp,
            "gaia_rp_mag": gaia_rp,
            "gal_lat_abs": row.get("gal_lat_abs", 0),
            "lc_path": lc_paths.get(ident),
            "surveys": surveys,
            "has_any_survey": has_any_survey,
            **sky_urls,
        }

        html = source_tmpl.render(url_prefix="../", source=source_data)
        (sources_dir / f"{ident}.html").write_text(html)

    logger.info("Generated %d source pages in %s", len(top), sources_dir)
    logger.info("Website output: %s", WEBSITE_DIR)
