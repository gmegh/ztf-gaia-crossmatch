"""ZTF Forced Photometry Service (ZFPS) integration.

Queries the IRSA ZFPS at Gaia source positions to check whether ZTF
actually detects flux there.  If significant detections are found at
a brightness consistent with the Gaia G magnitude, the Cat C candidate
is flagged as a false positive.

Prerequisites
-------------
You must register for the ZFPS by emailing ztf@ipac.caltech.edu.
Once approved you will receive credentials (email + password) that
are used to authenticate with the service.
"""

import logging
import time
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .config import RESULTS_DIR

logger = logging.getLogger(__name__)

ZFPS_SUBMIT_URL = (
    "https://ztfweb.ipac.caltech.edu/cgi-bin/requestForcedPhotometry.cgi"
)
# Service-level HTTP Basic Auth (public, from ZFPS docs)
ZFPS_HTTP_USER = "ztffps"
ZFPS_HTTP_PASS = "dontgocrazy!"

# Default JD range: ZTF operations (2018-03-17 to present)
DEFAULT_JD_START = 2458194.5  # 2018-03-17
DEFAULT_JD_END = 2460800.5    # ~2025-06

# If median forced-phot magnitude is within this many mags of Gaia G,
# consider it a detection consistent with the Gaia source.
MAG_TOLERANCE = 2.5
# Minimum number of forced-phot detections to consider it real
MIN_DETECTIONS = 5


def submit_forced_phot(ra, dec, email, password,
                       jd_start=None, jd_end=None):
    """Submit a single forced photometry request to the ZFPS.

    Parameters
    ----------
    ra, dec : float
        Position in degrees.
    email : str
        Registered ZFPS email.
    password : str
        ZFPS password.
    jd_start, jd_end : float, optional
        Julian Date range.

    Returns
    -------
    str or None
        Response text from ZFPS (contains job info), or None on failure.
    """
    params = {
        "ra": f"{ra:.6f}",
        "dec": f"{dec:.6f}",
        "jdstart": str(jd_start or DEFAULT_JD_START),
        "jdend": str(jd_end or DEFAULT_JD_END),
        "email": email,
        "userpass": password,
    }
    try:
        resp = requests.get(
            ZFPS_SUBMIT_URL,
            params=params,
            auth=(ZFPS_HTTP_USER, ZFPS_HTTP_PASS),
            timeout=60,
        )
        resp.raise_for_status()
        return resp.text
    except Exception as exc:
        logger.warning("ZFPS submit failed for (%.5f, %.5f): %s", ra, dec, exc)
        return None


def submit_batch(candidates, email, password, max_concurrent=100,
                 delay_between=1.0):
    """Submit forced photometry requests for Cat C candidates.

    The ZFPS allows max 100 concurrent jobs.  This function submits
    in batches and logs progress.  Results arrive via email.

    Parameters
    ----------
    candidates : pd.DataFrame
        Cat C candidates to check (must have ra, dec, gaia_source_id).
    email, password : str
        ZFPS credentials.
    max_concurrent : int
        Maximum simultaneous submissions.
    delay_between : float
        Seconds to wait between submissions to avoid rate-limiting.

    Returns
    -------
    list[dict]
        Submission results with source_id, ra, dec, response.
    """
    results = []
    total = len(candidates)
    logger.info("Submitting %d forced photometry requests to ZFPS", total)

    for i, (_, row) in enumerate(candidates.iterrows()):
        sid = str(int(row["gaia_source_id"]))
        ra, dec = float(row["ra"]), float(row["dec"])

        if i > 0 and i % max_concurrent == 0:
            logger.info(
                "  Submitted %d/%d, pausing for queue refresh...", i, total
            )
            time.sleep(60)  # wait for queue slots to free

        resp = submit_forced_phot(ra, dec, email, password)
        results.append({
            "gaia_source_id": sid,
            "ra": ra,
            "dec": dec,
            "response": resp[:200] if resp else None,
        })

        if delay_between > 0:
            time.sleep(delay_between)

        if (i + 1) % 10 == 0:
            logger.info("  Submitted %d/%d", i + 1, total)

    logger.info("All %d requests submitted. Results will arrive via email.", total)

    # Save submission log
    log_path = RESULTS_DIR / "zfps_submissions.csv"
    pd.DataFrame(results).to_csv(log_path, index=False)
    logger.info("Submission log saved to %s", log_path)

    return results


def parse_forced_phot_file(filepath):
    """Parse a downloaded ZFPS result file.

    Parameters
    ----------
    filepath : str or Path
        Path to the ZFPS output file (IPAC table or CSV).

    Returns
    -------
    pd.DataFrame or None
        Parsed forced photometry table, or None if empty/invalid.
    """
    path = Path(filepath)
    text = path.read_text()

    # ZFPS outputs IPAC table format; skip comment lines
    lines = [l for l in text.split("\n") if not l.startswith("\\")]
    if len(lines) < 3:
        return None

    try:
        df = pd.read_csv(StringIO("\n".join(lines)), sep=r"\s+")
        return df if len(df) > 0 else None
    except Exception:
        return None


def check_detection(fp_df, gaia_g_mag):
    """Check if forced photometry shows a detection consistent with Gaia.

    Parameters
    ----------
    fp_df : pd.DataFrame
        Forced photometry table from ZFPS.
    gaia_g_mag : float
        Gaia G-band magnitude of the candidate.

    Returns
    -------
    dict
        Detection summary: is_detected, n_detections, median_mag, mag_diff.
    """
    # ZFPS columns: forcediffimflux, forcediffimfluxunc, zpdiff, etc.
    # Convert flux to magnitude: mag = zpdiff - 2.5 * log10(flux)
    if "forcediffimflux" not in fp_df.columns:
        return {"is_detected": False, "n_detections": 0}

    flux = fp_df["forcediffimflux"].values
    flux_unc = fp_df["forcediffimfluxunc"].values
    zp = fp_df["zpdiff"].values if "zpdiff" in fp_df.columns else 25.0

    # Significant detections: flux > 3 * uncertainty
    sig = (flux > 0) & (flux_unc > 0) & (flux / flux_unc > 3)
    n_det = int(sig.sum())

    if n_det < MIN_DETECTIONS:
        return {"is_detected": False, "n_detections": n_det}

    # Convert to magnitudes
    mags = zp[sig] - 2.5 * np.log10(flux[sig])
    median_mag = float(np.median(mags))
    mag_diff = abs(median_mag - gaia_g_mag)

    return {
        "is_detected": mag_diff < MAG_TOLERANCE,
        "n_detections": n_det,
        "median_mag": round(median_mag, 2),
        "mag_diff": round(mag_diff, 2),
    }


def filter_cat_c_with_forced_phot(candidates, fp_results_dir):
    """Filter Cat C candidates using downloaded ZFPS results.

    Parameters
    ----------
    candidates : pd.DataFrame
        Full scored candidates DataFrame.
    fp_results_dir : str or Path
        Directory containing downloaded ZFPS result files,
        named by gaia_source_id (e.g., "4013150925224821760.txt").

    Returns
    -------
    pd.DataFrame
        Candidates with false-positive Cat C entries removed.
    """
    fp_dir = Path(fp_results_dir)
    if not fp_dir.exists():
        logger.warning("Forced phot results dir not found: %s", fp_dir)
        return candidates

    cat_c = candidates[candidates["category"] == "C"]
    false_positives = set()

    for _, row in cat_c.iterrows():
        sid = str(int(row["gaia_source_id"]))
        fp_file = fp_dir / f"{sid}.txt"
        if not fp_file.exists():
            continue

        fp_df = parse_forced_phot_file(fp_file)
        if fp_df is None:
            continue

        result = check_detection(fp_df, float(row["gaia_g_mag"]))
        if result["is_detected"]:
            false_positives.add(sid)
            logger.info(
                "  %s: DETECTED in ZTF forced phot "
                "(n=%d, median_mag=%.1f, Gaia_G=%.1f, Δ=%.1f)",
                sid, result["n_detections"],
                result.get("median_mag", 0), row["gaia_g_mag"],
                result.get("mag_diff", 0),
            )

    if false_positives:
        mask = candidates["gaia_source_id"].apply(
            lambda x: str(int(x)) if pd.notna(x) else ""
        ).isin(false_positives)
        candidates = candidates[~mask].reset_index(drop=True)
        logger.info(
            "Forced photometry: removed %d Cat C false positives",
            len(false_positives),
        )
    else:
        logger.info("Forced photometry: no false positives found (or no results)")

    return candidates
