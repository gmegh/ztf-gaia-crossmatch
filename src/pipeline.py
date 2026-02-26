"""Main orchestrator for the ZTF-Gaia cross-match pipeline."""

import json
import logging
import sys
from pathlib import Path

import pandas as pd
from astropy.table import Table

from .config import (
    DATA_DIR,
    PILOT_TILE,
    TILES_DIR,
    RESULTS_DIR,
    PROGRESS_FILE,
    tile_bounds,
)
from .tap_queries import query_ztf_objects, query_gaia_sources, query_gaia_variables
from .crossmatch import crossmatch
from .scoring import score_candidates
from .lightcurves import fetch_and_plot_top_candidates
from .multisurvey import crosscheck_candidates
from .website import generate_website

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _load_progress():
    """Load the set of completed tile indices."""
    if PROGRESS_FILE.exists():
        return set(json.loads(PROGRESS_FILE.read_text()))
    return set()


def _save_progress(completed):
    """Persist the set of completed tile indices."""
    PROGRESS_FILE.write_text(json.dumps(sorted(int(x) for x in completed)))


def process_tile(tile_index):
    """Run the full pipeline for a single HEALPix tile.

    Steps:
    1. Compute tile bounding box
    2. Query ZTF, Gaia sources, and Gaia variables
    3. Positional cross-match (Categories A, B, C)
    4. Score and rank candidates
    5. Save per-tile results as Parquet

    Parameters
    ----------
    tile_index : int
        HEALPix NESTED tile index (nside=32).

    Returns
    -------
    pd.DataFrame
        Scored candidates for this tile.
    """
    ra_min, ra_max, dec_min, dec_max = tile_bounds(tile_index)
    logger.info(
        "Processing tile %d  (RA %.2f–%.2f, Dec %.2f–%.2f)",
        tile_index, ra_min, ra_max, dec_min, dec_max,
    )

    # ── Queries (with local caching) ──────────────────────────────────
    cache_dir = DATA_DIR / f"cache_{tile_index:05d}"
    cache_dir.mkdir(exist_ok=True)
    ztf_cache = cache_dir / "ztf_objects.parquet"
    gaia_src_cache = cache_dir / "gaia_sources.parquet"
    gaia_var_cache = cache_dir / "gaia_variables.parquet"

    if ztf_cache.exists():
        logger.info("Loading cached ZTF objects from %s", ztf_cache)
        ztf_objects = Table.from_pandas(pd.read_parquet(ztf_cache))
    else:
        ztf_objects = query_ztf_objects(ra_min, ra_max, dec_min, dec_max)
        ztf_objects.to_pandas().to_parquet(ztf_cache, index=False)

    if gaia_src_cache.exists():
        logger.info("Loading cached Gaia sources from %s", gaia_src_cache)
        gaia_sources = Table.from_pandas(pd.read_parquet(gaia_src_cache))
    else:
        gaia_sources = query_gaia_sources(ra_min, ra_max, dec_min, dec_max)
        gaia_sources.to_pandas().to_parquet(gaia_src_cache, index=False)

    if gaia_var_cache.exists():
        logger.info("Loading cached Gaia variables from %s", gaia_var_cache)
        gaia_variables = Table.from_pandas(pd.read_parquet(gaia_var_cache))
    else:
        gaia_variables = query_gaia_variables(ra_min, ra_max, dec_min, dec_max)
        gaia_variables.to_pandas().to_parquet(gaia_var_cache, index=False)

    if len(ztf_objects) == 0:
        logger.warning("Tile %d: no ZTF objects returned", tile_index)
        return pd.DataFrame()

    if len(gaia_sources) == 0:
        logger.warning("Tile %d: no Gaia sources returned", tile_index)
        return pd.DataFrame()

    # ── Cross-match ────────────────────────────────────────────────────
    cat_a, cat_b, cat_c = crossmatch(ztf_objects, gaia_sources, gaia_variables)

    # ── Scoring ────────────────────────────────────────────────────────
    candidates = score_candidates(cat_a, cat_b, cat_c)

    # ── Save checkpoint ────────────────────────────────────────────────
    tile_path = TILES_DIR / f"tile_{tile_index:05d}.parquet"
    candidates.to_parquet(tile_path, index=False)
    logger.info("Saved %d candidates to %s", len(candidates), tile_path)

    # Update progress
    completed = _load_progress()
    completed.add(tile_index)
    _save_progress(completed)

    return candidates


def run_pilot(n_lightcurves=50, n_multisurvey=50, n_website=100):
    """Run the full pipeline on the pilot tile, including Phase 3.

    Parameters
    ----------
    n_lightcurves : int
        Number of top candidates to fetch light curves for.
    n_multisurvey : int
        Number of top candidates for multi-survey cross-checks.
    n_website : int
        Number of top candidates to include on the website.
    """
    logger.info("=== Pilot run: tile %d ===", PILOT_TILE)
    candidates = process_tile(PILOT_TILE)

    if len(candidates) == 0:
        logger.warning("Pilot produced no candidates")
        return candidates

    # Save full results
    output_path = RESULTS_DIR / "pilot_candidates.parquet"
    candidates.to_parquet(output_path, index=False)
    logger.info("Pilot results saved to %s (%d candidates)", output_path, len(candidates))

    # Print summary
    logger.info("Top 10 candidates:")
    cols = ["oid", "gaia_source_id", "ra", "dec", "category", "score"]
    top = candidates.head(10)[cols]
    logger.info("\n%s", top.to_string())

    # ── Phase 3: Light curves ──────────────────────────────────────────
    logger.info("=== Phase 3: Light curves ===")
    lc_paths = fetch_and_plot_top_candidates(candidates, n_top=n_lightcurves)

    # ── Phase 3: Multi-survey cross-checks ─────────────────────────────
    logger.info("=== Phase 3: Multi-survey cross-checks ===")
    survey_results = crosscheck_candidates(candidates, n_top=n_multisurvey)

    # ── Phase 3: Website generation ────────────────────────────────────
    logger.info("=== Phase 3: Website generation ===")
    generate_website(
        candidates,
        lc_paths=lc_paths,
        survey_results=survey_results,
        n_top=n_website,
    )

    return candidates


def main():
    """Entry point for `python -m src.pipeline`."""
    run_pilot()


if __name__ == "__main__":
    main()
