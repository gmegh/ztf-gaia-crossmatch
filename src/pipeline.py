"""Main orchestrator for the ZTF-Gaia cross-match pipeline."""

import json
import logging
import sys
from pathlib import Path

import healpy as hp
import numpy as np
import pandas as pd
from astropy.table import Table

from .config import (
    DATA_DIR,
    NSIDE,
    PILOT_TILE,
    TILES_DIR,
    TILES_CAT_C_DIR,
    RESULTS_DIR,
    PROGRESS_FILE,
    PROGRESS_CAT_C_FILE,
    tile_bounds,
)
from .tap_queries import (
    query_ztf_objects, query_ztf_positions,
    query_gaia_sources, query_gaia_variables,
)
from .crossmatch import crossmatch, crossmatch_cat_c_only
from .scoring import score_candidates, score_cat_c
from .lightcurves import fetch_and_plot_top_candidates
from .multisurvey import crosscheck_candidates, filter_sdss
from .website import generate_website

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _load_progress(path=None):
    """Load the set of completed tile indices."""
    path = path or PROGRESS_FILE
    if path.exists():
        return set(json.loads(path.read_text()))
    return set()


def _save_progress(completed, path=None):
    """Persist the set of completed tile indices."""
    path = path or PROGRESS_FILE
    path.write_text(json.dumps(sorted(int(x) for x in completed)))


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
        "Processing tile %d  (RA %.2f-%.2f, Dec %.2f-%.2f)",
        tile_index, ra_min, ra_max, dec_min, dec_max,
    )

    # -- Queries (with local caching) --
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

    # -- Cross-match --
    cat_a, cat_b, cat_c = crossmatch(ztf_objects, gaia_sources, gaia_variables)

    # -- Scoring --
    candidates = score_candidates(cat_a, cat_b, cat_c)

    # -- Save checkpoint --
    tile_path = TILES_DIR / f"tile_{tile_index:05d}.parquet"
    candidates.to_parquet(tile_path, index=False)
    logger.info("Saved %d candidates to %s", len(candidates), tile_path)

    # Update progress
    completed = _load_progress()
    completed.add(tile_index)
    _save_progress(completed)

    return candidates


def process_tile_cat_c(tile_index):
    """Run Cat C-only pipeline for a single HEALPix tile.

    Uses lightweight ZTF positions instead of the full objects table.

    Parameters
    ----------
    tile_index : int
        HEALPix NESTED tile index (nside=32).

    Returns
    -------
    pd.DataFrame
        Scored Cat C candidates for this tile.
    """
    ra_min, ra_max, dec_min, dec_max = tile_bounds(tile_index)
    logger.info(
        "Processing tile %d (Cat C only)  (RA %.2f-%.2f, Dec %.2f-%.2f)",
        tile_index, ra_min, ra_max, dec_min, dec_max,
    )

    cache_dir = DATA_DIR / f"cache_{tile_index:05d}"
    cache_dir.mkdir(exist_ok=True)
    ztf_pos_cache = cache_dir / "ztf_positions.parquet"
    gaia_src_cache = cache_dir / "gaia_sources.parquet"
    gaia_var_cache = cache_dir / "gaia_variables.parquet"

    if ztf_pos_cache.exists():
        logger.info("Loading cached ZTF positions from %s", ztf_pos_cache)
        ztf_positions = Table.from_pandas(pd.read_parquet(ztf_pos_cache))
    else:
        ztf_positions = query_ztf_positions(ra_min, ra_max, dec_min, dec_max)
        ztf_positions.to_pandas().to_parquet(ztf_pos_cache, index=False)

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

    if len(ztf_positions) == 0:
        logger.warning("Tile %d: no ZTF positions returned", tile_index)
        return pd.DataFrame()

    if len(gaia_sources) == 0:
        logger.warning("Tile %d: no Gaia sources returned", tile_index)
        return pd.DataFrame()

    cat_c = crossmatch_cat_c_only(ztf_positions, gaia_sources, gaia_variables)
    candidates = score_cat_c(cat_c)

    tile_path = TILES_CAT_C_DIR / f"tile_{tile_index:05d}.parquet"
    candidates.to_parquet(tile_path, index=False)
    logger.info("Saved %d Cat C candidates to %s", len(candidates), tile_path)

    completed = _load_progress(PROGRESS_CAT_C_FILE)
    completed.add(tile_index)
    _save_progress(completed, PROGRESS_CAT_C_FILE)

    return candidates


def _select_survey_tiles(n_tiles=123):
    """Select uniformly-spaced tiles in the northern sky (Dec > 0).

    Uses NESTED pixel ordering -- sorting by pixel index naturally
    spreads tiles across the sky. We take every ~Nth tile for
    uniform coverage.
    """
    total = 12 * NSIDE ** 2
    all_pixels = np.arange(total)
    # Get center coordinates for each pixel
    ra, dec = hp.pix2ang(NSIDE, all_pixels, nest=True, lonlat=True)
    northern = all_pixels[dec > 0]

    # Exclude the pilot tile (already fully processed)
    northern = northern[northern != PILOT_TILE]

    # Uniform sampling: take every Nth tile
    step = max(1, len(northern) // n_tiles)
    selected = northern[::step][:n_tiles]

    logger.info(
        "Selected %d survey tiles from %d northern-sky tiles (step=%d)",
        len(selected), len(northern), step,
    )
    return selected.tolist()


def run_cat_c_survey(n_tiles=123):
    """Run Cat C-only processing across multiple tiles.

    Parameters
    ----------
    n_tiles : int
        Number of tiles to process (default: 123 ~ 1% of sky).

    Returns
    -------
    pd.DataFrame
        Combined Cat C candidates from all tiles.
    """
    logger.info("=== Cat C Wide-Sky Survey (%d tiles) ===", n_tiles)

    tiles = _select_survey_tiles(n_tiles)
    completed = _load_progress(PROGRESS_CAT_C_FILE)
    pending = [t for t in tiles if t not in completed]
    logger.info(
        "%d tiles total, %d already completed, %d pending",
        len(tiles), len(tiles) - len(pending), len(pending),
    )

    for i, tile_idx in enumerate(pending):
        logger.info("=== Survey tile %d/%d (pixel %d) ===", i + 1, len(pending), tile_idx)
        try:
            process_tile_cat_c(tile_idx)
        except Exception as exc:
            logger.error("Tile %d failed: %s", tile_idx, exc)

    # Concatenate all tile results
    all_dfs = []
    for tile_idx in tiles:
        tile_path = TILES_CAT_C_DIR / f"tile_{tile_idx:05d}.parquet"
        if tile_path.exists():
            df = pd.read_parquet(tile_path)
            if len(df) > 0:
                df["tile_index"] = tile_idx
                all_dfs.append(df)

    if not all_dfs:
        logger.warning("No Cat C results from survey")
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)
    logger.info("Combined %d Cat C candidates from %d tiles", len(combined), len(all_dfs))

    # SDSS filter on combined Cat C
    logger.info("=== SDSS filter for Cat C survey ===")
    combined = filter_sdss(combined, cat_a_radius=1.0, cat_c_radius=3.0)

    # Re-sort by score
    combined = combined.sort_values("score", ascending=False).reset_index(drop=True)

    output_path = RESULTS_DIR / "cat_c_survey_candidates.parquet"
    combined.to_parquet(output_path, index=False)
    logger.info(
        "Cat C survey results saved to %s (%d candidates)",
        output_path, len(combined),
    )

    return combined


def run_pilot(n_lightcurves=500, n_multisurvey=50, n_website=500):
    """Run the full pipeline on the pilot tile, including Phase 3.

    Parameters
    ----------
    n_lightcurves : int
        Number of top candidates to fetch light curves for.
    n_multisurvey : int
        Number of top candidates for multi-survey cross-checks.
    n_website : int
        Number of top candidates to include on the website.

    Returns
    -------
    tuple
        (candidates, lc_paths, survey_results) for website generation.
    """
    logger.info("=== Pilot run: tile %d ===", PILOT_TILE)
    candidates = process_tile(PILOT_TILE)

    if len(candidates) == 0:
        logger.warning("Pilot produced no candidates")
        return candidates, {}, {}

    # -- SDSS filter: remove Cat A/C sources with SDSS counterpart --
    logger.info("=== SDSS filter for Cat A & C ===")
    candidates = filter_sdss(candidates, cat_a_radius=1.0, cat_c_radius=3.0)

    # Save full results
    output_path = RESULTS_DIR / "pilot_candidates.parquet"
    candidates.to_parquet(output_path, index=False)
    logger.info("Pilot results saved to %s (%d candidates)", output_path, len(candidates))

    # Print summary
    logger.info("Top 10 candidates:")
    cols = ["oid", "gaia_source_id", "ra", "dec", "category", "score"]
    top = candidates.head(10)[cols]
    logger.info("\n%s", top.to_string())

    # -- Phase 3: Light curves --
    logger.info("=== Phase 3: Light curves ===")
    lc_paths = fetch_and_plot_top_candidates(candidates, n_top=n_lightcurves)

    # -- Phase 3: Multi-survey cross-checks --
    logger.info("=== Phase 3: Multi-survey cross-checks ===")
    survey_results = crosscheck_candidates(candidates, n_top=n_multisurvey)

    return candidates, lc_paths, survey_results


def main():
    """Entry point for `python -m src.pipeline`."""
    pilot_candidates, lc_paths, survey_results = run_pilot()
    cat_c_survey = run_cat_c_survey(n_tiles=123)

    # -- Website generation (both tabs) --
    logger.info("=== Website generation ===")
    generate_website(
        pilot_candidates,
        lc_paths=lc_paths,
        survey_results=survey_results,
        n_top=500,
        cat_c_survey=cat_c_survey,
    )


if __name__ == "__main__":
    main()
