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
    NSIDE, XMATCH_RADIUS_ARCSEC, PROGRESS_FILE, PROGRESS_CAT_C_FILE,
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
    ztf_hips_url = (
        f"https://alasky.cds.unistra.fr/hips-image-services/hips2fits"
        f"?hips=CDS/P/ZTF/DR7/g&width=256&height=256"
        f"&fov=0.03&projection=TAN&coordsys=icrs"
        f"&ra={ra:.6f}&dec={dec:.6f}&format=jpg"
    )
    aladin_url = (
        f"https://aladin.cds.unistra.fr/AladinLite/"
        f"?target={ra:.6f}+{dec:.6f}&fov=0.03&survey=CDS%2FP%2FDSS2%2Fcolor"
    )
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


def _coverage_stats(progress_file=None):
    """Compute sky coverage statistics from a progress file."""
    progress_file = progress_file or PROGRESS_FILE
    tiles_processed = 0
    if progress_file.exists():
        tiles_processed = len(json.loads(progress_file.read_text()))
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


def _build_source_page(source_tmpl, sources_dir, row, lc_paths, survey_results,
                       back_page="index.html"):
    """Generate a single source detail page."""
    ident = _source_id(row)
    surveys = survey_results.get(ident, {})
    has_any_survey = any(v is not None for v in surveys.values())

    gaia_bp = _safe_float(row.get("phot_bp_mean_mag"))
    gaia_rp = _safe_float(row.get("phot_rp_mean_mag"))
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
        "nearest_ztf_sep": _safe_float(row.get("nearest_ztf_sep")),
        "nearest_ztf_mag": _safe_float(row.get("nearest_ztf_mag")),
        "nearest_ztf_mag_diff": _safe_float(row.get("nearest_ztf_mag_diff")),
        "lc_path": lc_paths.get(ident),
        "surveys": surveys,
        "has_any_survey": has_any_survey,
        **sky_urls,
    }

    html = source_tmpl.render(
        url_prefix="../", source=source_data, back_page=back_page,
    )
    (sources_dir / f"{ident}.html").write_text(html)
    return ident


def generate_website(candidates, lc_paths=None, survey_results=None, n_top=100,
                     cat_c_survey=None):
    """Generate the static HTML website from candidate results.

    Parameters
    ----------
    candidates : pd.DataFrame
        Scored pilot candidates DataFrame (all categories).
    lc_paths : dict, optional
        Mapping of identifier -> light curve PNG path (relative).
    survey_results : dict, optional
        Mapping of identifier -> {survey: match_dict} from multisurvey.
    n_top : int
        Number of top candidates to generate source pages for.
    cat_c_survey : pd.DataFrame, optional
        Cat C survey candidates from wide-sky processing.
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

    # Pilot coverage stats
    coverage = _coverage_stats(PROGRESS_FILE)
    n_surveyed = len(survey_results)

    # -- Build pilot index page (all categories) --
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
            "nearest_ztf_sep": round(float(row["nearest_ztf_sep"]), 2) if pd.notna(row.get("nearest_ztf_sep")) else None,
            "nearest_ztf_mag": round(float(row["nearest_ztf_mag"]), 2) if pd.notna(row.get("nearest_ztf_mag")) else None,
            "nearest_ztf_mag_diff": round(float(row["nearest_ztf_mag_diff"]), 2) if pd.notna(row.get("nearest_ztf_mag_diff")) else None,
            "survey_matches": ", ".join(matched) if matched else "",
        })

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

    # -- Pilot source pages --
    source_tmpl = env.get_template("source.html")
    for _, row in top.iterrows():
        _build_source_page(source_tmpl, sources_dir, row, lc_paths,
                           survey_results, back_page="index.html")

    logger.info("Generated %d pilot source pages", len(top))

    # -- Cat C survey page --
    if cat_c_survey is not None and len(cat_c_survey) > 0:
        survey_coverage = _coverage_stats(PROGRESS_CAT_C_FILE)

        survey_top = cat_c_survey.head(n_top)
        survey_candidates_list = []
        for rank, (_, row) in enumerate(survey_top.iterrows(), 1):
            ident = str(int(row["gaia_source_id"]))
            survey_candidates_list.append({
                "rank": rank,
                "id": ident,
                "ra": round(float(row["ra"]), 6),
                "dec": round(float(row["dec"]), 6),
                "score": round(float(row["score"]), 4),
                "gaia_g_mag": round(float(row["gaia_g_mag"]), 2) if pd.notna(row.get("gaia_g_mag")) else None,
                "nearest_ztf_sep": round(float(row["nearest_ztf_sep"]), 2) if pd.notna(row.get("nearest_ztf_sep")) else None,
                "nearest_ztf_mag": round(float(row["nearest_ztf_mag"]), 2) if pd.notna(row.get("nearest_ztf_mag")) else None,
                "nearest_ztf_mag_diff": round(float(row["nearest_ztf_mag_diff"]), 2) if pd.notna(row.get("nearest_ztf_mag_diff")) else None,
            })

        survey_json = json.dumps(survey_candidates_list, separators=(",", ":"))

        survey_tmpl = env.get_template("cat_c_survey.html")
        survey_html = survey_tmpl.render(
            url_prefix="",
            total_candidates=len(cat_c_survey),
            candidates_json=survey_json,
            **survey_coverage,
        )
        (WEBSITE_DIR / "cat_c_survey.html").write_text(survey_html)
        logger.info(
            "Generated cat_c_survey.html with %d candidates",
            len(survey_candidates_list),
        )

        # Source pages for Cat C survey candidates (shared directory)
        for _, row in survey_top.iterrows():
            _build_source_page(source_tmpl, sources_dir, row, {},
                               {}, back_page="cat_c_survey.html")

        logger.info("Generated %d Cat C survey source pages", len(survey_top))
    else:
        # Generate empty Cat C survey page
        survey_tmpl = env.get_template("cat_c_survey.html")
        survey_html = survey_tmpl.render(
            url_prefix="",
            total_candidates=0,
            candidates_json="[]",
            **_coverage_stats(PROGRESS_CAT_C_FILE),
        )
        (WEBSITE_DIR / "cat_c_survey.html").write_text(survey_html)
        logger.info("Generated empty cat_c_survey.html (no survey data)")

    logger.info("Website output: %s", WEBSITE_DIR)
