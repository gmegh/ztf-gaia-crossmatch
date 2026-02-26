"""Jinja2 static site generator for the ZTF-Gaia cross-match results."""

import logging
import shutil

import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader

from .config import WEBSITE_DIR, TEMPLATES_DIR, RESULTS_DIR, ROOT_DIR

logger = logging.getLogger(__name__)

CATEGORY_DESCRIPTIONS = {
    "A": "ZTF source with no Gaia counterpart within 1.5″",
    "B": "ZTF source matched to Gaia, not a known Gaia variable",
    "C": "Gaia source with no ZTF counterpart within 1.5″",
}


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


def generate_website(candidates, lc_paths=None, survey_results=None, n_top=100):
    """Generate the static HTML website from candidate results.

    Parameters
    ----------
    candidates : pd.DataFrame
        Scored candidates DataFrame.
    lc_paths : dict, optional
        Mapping of identifier → light curve PNG path (relative).
    survey_results : dict, optional
        Mapping of identifier → {survey: match_dict} from multisurvey.
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

    # ── Index page ─────────────────────────────────────────────────────
    top = candidates.head(n_top)
    index_candidates = []
    for _, row in top.iterrows():
        ident = _source_id(row)
        surveys = survey_results.get(ident, {})
        matched = [k for k, v in surveys.items() if v is not None]
        index_candidates.append({
            "id": ident,
            "oid": str(int(row["oid"])) if pd.notna(row.get("oid")) else None,
            "ra": row["ra"],
            "dec": row["dec"],
            "category": row["category"],
            "score": row["score"],
            "amplitude": _safe_float(row.get("best_amplitude")),
            "nobs": _safe_float(row.get("best_nobs")),
            "gaia_g_mag": _safe_float(row.get("gaia_g_mag")),
            "survey_matches": ", ".join(matched) if matched else "",
        })

    index_tmpl = env.get_template("index.html")
    index_html = index_tmpl.render(
        url_prefix="",
        total_candidates=len(candidates),
        cat_a_count=int((candidates["category"] == "A").sum()),
        cat_b_count=int((candidates["category"] == "B").sum()),
        cat_c_count=int((candidates["category"] == "C").sum()),
        candidates=index_candidates,
    )
    (WEBSITE_DIR / "index.html").write_text(index_html)
    logger.info("Generated index.html with %d candidates", len(index_candidates))

    # ── Source pages ───────────────────────────────────────────────────
    source_tmpl = env.get_template("source.html")
    for _, row in top.iterrows():
        ident = _source_id(row)
        surveys = survey_results.get(ident, {})
        has_any_survey = any(v is not None for v in surveys.values())

        # Get Gaia BP/RP if available
        gaia_bp = _safe_float(row.get("phot_bp_mean_mag"))
        gaia_rp = _safe_float(row.get("phot_rp_mean_mag"))

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
        }

        html = source_tmpl.render(url_prefix="../", source=source_data)
        (sources_dir / f"{ident}.html").write_text(html)

    logger.info("Generated %d source pages in %s", len(top), sources_dir)
    logger.info("Website output: %s", WEBSITE_DIR)
