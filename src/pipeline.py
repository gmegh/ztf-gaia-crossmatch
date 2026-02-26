"""Main orchestrator for the ZTF-Gaia cross-match pipeline."""

import json
import logging
import sys

import pandas as pd

from .config import (
    PILOT_TILE,
    TILES_DIR,
    RESULTS_DIR,
    PROGRESS_FILE,
    tile_bounds,
)
from .tap_queries import query_ztf_objects, query_gaia_sources, query_gaia_variables
from .crossmatch import crossmatch
from .scoring import score_candidates

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
    PROGRESS_FILE.write_text(json.dumps(sorted(completed)))


def process_tile(tile_index):
    """Run the full pipeline for a single HEALPix tile.

    Steps:
    1. Compute tile bounding box
    2. Query ZTF, Gaia sources, and Gaia variables via TAP
    3. Positional cross-match
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

    # ── TAP queries ────────────────────────────────────────────────────
    ztf_objects = query_ztf_objects(ra_min, ra_max, dec_min, dec_max)
    gaia_sources = query_gaia_sources(ra_min, ra_max, dec_min, dec_max)
    gaia_variables = query_gaia_variables(ra_min, ra_max, dec_min, dec_max)

    if len(ztf_objects) == 0:
        logger.warning("Tile %d: no ZTF objects returned", tile_index)
        return pd.DataFrame()

    if len(gaia_sources) == 0:
        logger.warning("Tile %d: no Gaia sources returned", tile_index)
        return pd.DataFrame()

    # ── Cross-match ────────────────────────────────────────────────────
    cat_a, cat_b = crossmatch(ztf_objects, gaia_sources, gaia_variables)

    # ── Scoring ────────────────────────────────────────────────────────
    candidates = score_candidates(cat_a, cat_b)

    # ── Save checkpoint ────────────────────────────────────────────────
    tile_path = TILES_DIR / f"tile_{tile_index:05d}.parquet"
    candidates.to_parquet(tile_path, index=False)
    logger.info("Saved %d candidates to %s", len(candidates), tile_path)

    # Update progress
    completed = _load_progress()
    completed.add(tile_index)
    _save_progress(completed)

    return candidates


def run_pilot():
    """Run the pipeline on the single pilot tile (RA=180, Dec=+30)."""
    logger.info("=== Pilot run: tile %d ===", PILOT_TILE)
    candidates = process_tile(PILOT_TILE)

    if len(candidates) > 0:
        output_path = RESULTS_DIR / "pilot_candidates.parquet"
        candidates.to_parquet(output_path, index=False)
        logger.info("Pilot results saved to %s", output_path)
        logger.info("Top 10 candidates:")
        top = candidates.head(10)[
            ["oid", "ra", "dec", "category", "score", "amplitude", "nobs"]
        ]
        logger.info("\n%s", top.to_string())
    else:
        logger.warning("Pilot produced no candidates")

    return candidates


def main():
    """Entry point for `python -m src.pipeline`."""
    run_pilot()


if __name__ == "__main__":
    main()
