"""ZTF Forced Photometry Service (ZFPS) integration.

Submits batch forced-photometry requests at Gaia source positions to
check whether ZTF actually detects flux there.  If significant
detections are found at a brightness consistent with the Gaia G
magnitude, the Cat C candidate is flagged as a false positive.

Uses the ZFPS batch API:
  Submit:  POST https://ztfweb.ipac.caltech.edu/cgi-bin/batchfp.py/submit
  Status:  GET  https://ztfweb.ipac.caltech.edu/cgi-bin/getBatchForcedPhotometryRequests.cgi
  Download: wget with HTTP Basic Auth (ztffps / dontgocrazy!)

Max 1500 positions per submission, 15000 total in the system.
"""

import json
import logging
import re
import time
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .config import RESULTS_DIR

logger = logging.getLogger(__name__)

# ── ZFPS endpoints and auth ────────────────────────────────────────────
ZFPS_SUBMIT_URL = (
    "https://ztfweb.ipac.caltech.edu/cgi-bin/batchfp.py/submit"
)
ZFPS_REGULAR_URL = (
    "https://ztfweb.ipac.caltech.edu/cgi-bin/requestForcedPhotometry.cgi"
)
ZFPS_STATUS_URL = (
    "https://ztfweb.ipac.caltech.edu/cgi-bin/"
    "getBatchForcedPhotometryRequests.cgi"
)
# Service-level HTTP Basic Auth (public, from ZFPS docs)
ZFPS_HTTP_USER = "ztffps"
ZFPS_HTTP_PASS = "dontgocrazy!"

# Default JD range: ZTF operations (2018-03-17 to ~2025-06)
DEFAULT_JD_START = 2458194.5  # 2018-03-17
DEFAULT_JD_END = 2460800.5    # ~2025-06

# If median forced-phot magnitude is within this many mags of Gaia G,
# consider it a detection consistent with the Gaia source.
MAG_TOLERANCE = 2.5
# Minimum number of significant forced-phot epochs to consider it real
MIN_DETECTIONS = 5


# ── Submit ──────────────────────────────────────────────────────────────

def submit_batch(ra_list, dec_list, email, password,
                 jd_start=None, jd_end=None):
    """Submit a batch of positions to the ZFPS.

    Parameters
    ----------
    ra_list, dec_list : list[float]
        Positions in decimal degrees (max 1500 per call).
    email : str
        Registered ZFPS email.
    password : str
        ZFPS password.
    jd_start, jd_end : float, optional
        Julian Date range.

    Returns
    -------
    str
        Response text from ZFPS.
    """
    if len(ra_list) > 1500:
        raise ValueError(
            f"ZFPS allows max 1500 positions per request, got {len(ra_list)}"
        )

    payload = {
        "ra": str(ra_list),
        "dec": str(dec_list),
        "jdstart": str(jd_start or DEFAULT_JD_START),
        "jdend": str(jd_end or DEFAULT_JD_END),
        "email": email,
        "userpass": password,
    }
    logger.info(
        "Submitting %d positions to ZFPS batch API", len(ra_list)
    )
    # The ZFPS server has intermittent auth issues; retry up to 10 times
    for attempt in range(1, 11):
        resp = requests.post(
            ZFPS_SUBMIT_URL,
            auth=(ZFPS_HTTP_USER, ZFPS_HTTP_PASS),
            data=payload,
            timeout=120,
        )
        if resp.status_code == 200:
            logger.info("ZFPS response (attempt %d): %s", attempt, resp.text[:500])
            return resp.text
        logger.warning(
            "ZFPS submit attempt %d failed (HTTP %d), retrying...",
            attempt, resp.status_code,
        )
        time.sleep(2)

    resp.raise_for_status()  # raise on final failure
    return resp.text


def submit_single(ra, dec, email, password,
                  jd_start=None, jd_end=None):
    """Submit a single position to the regular ZFPS (non-batch).

    Parameters
    ----------
    ra, dec : float
        Position in decimal degrees.
    email, password : str
        ZFPS credentials.

    Returns
    -------
    str
        Response text.
    """
    payload = {
        "ra": str(ra),
        "dec": str(dec),
        "jdstart": str(jd_start or DEFAULT_JD_START),
        "jdend": str(jd_end or DEFAULT_JD_END),
        "email": email,
        "userpass": password,
    }
    for attempt in range(1, 6):
        resp = requests.post(
            ZFPS_REGULAR_URL,
            auth=(ZFPS_HTTP_USER, ZFPS_HTTP_PASS),
            data=payload,
            timeout=120,
        )
        if resp.status_code == 200:
            return resp.text
        logger.warning(
            "Regular ZFPS attempt %d failed (HTTP %d)",
            attempt, resp.status_code,
        )
        time.sleep(2)
    resp.raise_for_status()
    return resp.text


def submit_cat_c(candidates, email, password,
                 jd_start=None, jd_end=None):
    """Submit all Cat C candidates for forced photometry.

    Splits into batches of 1500 if needed.

    Parameters
    ----------
    candidates : pd.DataFrame
        Scored candidates (all categories).
    email, password : str
        ZFPS credentials.

    Returns
    -------
    list[str]
        Response texts from each batch submission.
    """
    cat_c = candidates[candidates["category"] == "C"].copy()
    if len(cat_c) == 0:
        logger.info("No Cat C candidates to submit")
        return []

    ra_all = cat_c["ra"].tolist()
    dec_all = cat_c["dec"].tolist()
    responses = []

    # For small batches, use the regular (non-batch) endpoint
    # which is more reliable
    if len(ra_all) <= 20:
        logger.info(
            "Using regular ZFPS endpoint for %d positions", len(ra_all)
        )
        for i, (ra, dec) in enumerate(zip(ra_all, dec_all)):
            resp = submit_single(
                ra, dec, email, password,
                jd_start=jd_start, jd_end=jd_end,
            )
            responses.append(resp)
            logger.info("  [%d/%d] RA=%.6f, Dec=%.6f: submitted",
                        i + 1, len(ra_all), ra, dec)
            time.sleep(1)
    else:
        # Split into chunks of 1500 for batch endpoint
        for i in range(0, len(ra_all), 1500):
            ra_chunk = ra_all[i:i + 1500]
            dec_chunk = dec_all[i:i + 1500]
            resp = submit_batch(
                ra_chunk, dec_chunk, email, password,
                jd_start=jd_start, jd_end=jd_end,
            )
            responses.append(resp)
            if i + 1500 < len(ra_all):
                logger.info("Waiting 5s before next batch...")
                time.sleep(5)

    logger.info(
        "Submitted %d Cat C positions in %d batch(es). "
        "Check status with check_status().",
        len(ra_all), len(responses),
    )
    return responses


# ── Status & Download ───────────────────────────────────────────────────

def check_status(email, password, option="All recent jobs"):
    """Check the status of submitted ZFPS jobs.

    Parameters
    ----------
    email, password : str
        ZFPS credentials.
    option : str
        "All recent jobs" or "Pending jobs".

    Returns
    -------
    str
        Response text containing wget download commands.
    """
    params = {
        "email": email,
        "userpass": password,
        "option": option,
        "action": "Query Database",
    }
    resp = requests.get(
        ZFPS_STATUS_URL,
        auth=(ZFPS_HTTP_USER, ZFPS_HTTP_PASS),
        params=params,
        timeout=60,
    )
    resp.raise_for_status()
    return resp.text


def download_results(email, password, output_dir=None):
    """Check status and download all completed ZFPS results.

    Parameters
    ----------
    email, password : str
        ZFPS credentials.
    output_dir : str or Path, optional
        Directory to save result files (default: RESULTS_DIR/forced_phot).

    Returns
    -------
    list[Path]
        Paths to downloaded result files.
    """
    output_dir = Path(output_dir or RESULTS_DIR / "forced_phot")
    output_dir.mkdir(parents=True, exist_ok=True)

    status_text = check_status(email, password)

    # Extract download URLs from wget commands in the response
    urls = re.findall(
        r'https://ztfweb\.ipac\.caltech\.edu/[^\s"<>]+\.txt', status_text
    )
    if not urls:
        logger.info("No completed ZFPS results found (or none ready yet)")
        logger.debug("Status response: %s", status_text[:1000])
        return []

    downloaded = []
    for url in urls:
        filename = url.split("/")[-1]
        outpath = output_dir / filename
        if outpath.exists():
            logger.debug("Already downloaded: %s", filename)
            downloaded.append(outpath)
            continue

        try:
            resp = requests.get(
                url,
                auth=(ZFPS_HTTP_USER, ZFPS_HTTP_PASS),
                timeout=120,
            )
            resp.raise_for_status()
            outpath.write_text(resp.text)
            downloaded.append(outpath)
            logger.info("Downloaded: %s", filename)
        except Exception as exc:
            logger.warning("Failed to download %s: %s", url, exc)

    logger.info("Downloaded %d result files to %s", len(downloaded), output_dir)
    return downloaded


# ── Parse & Analyse ─────────────────────────────────────────────────────

def parse_forced_phot_file(filepath):
    """Parse a downloaded ZFPS result file.

    The output is a space-delimited ASCII table with # comment lines.

    Parameters
    ----------
    filepath : str or Path
        Path to the ZFPS output file.

    Returns
    -------
    pd.DataFrame or None
        Parsed forced photometry table, or None if empty/invalid.
    """
    path = Path(filepath)
    text = path.read_text()

    # Skip comment/header lines starting with #
    lines = [line for line in text.split("\n") if not line.startswith("#")]
    clean = "\n".join(lines).strip()
    if not clean:
        return None

    try:
        df = pd.read_csv(StringIO(clean), sep=r"\s+")
        return df if len(df) > 0 else None
    except Exception:
        return None


def analyse_lightcurve(fp_df, ra, dec, gaia_g_mag):
    """Analyse forced photometry lightcurve at a single position.

    The batch ZFPS returns one file with all positions interleaved.
    This function works on a pre-filtered subset for one (ra, dec).

    Parameters
    ----------
    fp_df : pd.DataFrame
        Forced photometry rows for this position.
    ra, dec : float
        Position (for logging).
    gaia_g_mag : float
        Gaia G-band magnitude to compare against.

    Returns
    -------
    dict
        Detection summary: is_detected, n_detections, median_mag, mag_diff.
    """
    if "forcediffimflux" not in fp_df.columns:
        return {"is_detected": False, "n_detections": 0}

    # Only use successfully processed epochs
    if "procstatus" in fp_df.columns:
        fp_df = fp_df[fp_df["procstatus"] == 0]

    flux = fp_df["forcediffimflux"].values.astype(float)
    flux_unc = fp_df["forcediffimfluxunc"].values.astype(float)
    zp = fp_df["zpdiff"].values.astype(float) if "zpdiff" in fp_df.columns else 25.0

    # Significant detections: positive flux > 3 sigma
    sig = (flux > 0) & (flux_unc > 0) & (flux / flux_unc > 3)
    n_det = int(sig.sum())

    if n_det < MIN_DETECTIONS:
        return {"is_detected": False, "n_detections": n_det}

    # Convert to magnitudes: mag = zpdiff - 2.5 * log10(flux)
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

    Looks for result files in fp_results_dir and analyses each one.
    The batch ZFPS produces files like batchfp_req<ID>_lc.txt containing
    all positions from that batch.  We match rows back to candidates by
    rounding (ra, dec) to 5 decimal places.

    Parameters
    ----------
    candidates : pd.DataFrame
        Full scored candidates DataFrame.
    fp_results_dir : str or Path
        Directory containing downloaded ZFPS result files.

    Returns
    -------
    pd.DataFrame
        Candidates with false-positive Cat C entries removed.
    """
    fp_dir = Path(fp_results_dir)
    if not fp_dir.exists():
        logger.warning("Forced phot results dir not found: %s", fp_dir)
        return candidates

    # Load and concatenate all result files
    all_dfs = []
    for fp_file in sorted(fp_dir.glob("*.txt")):
        df = parse_forced_phot_file(fp_file)
        if df is not None:
            all_dfs.append(df)

    if not all_dfs:
        logger.info("No valid forced photometry result files found")
        return candidates

    fp_all = pd.concat(all_dfs, ignore_index=True)
    logger.info(
        "Loaded %d forced-phot epochs from %d files",
        len(fp_all), len(all_dfs),
    )

    # Build lookup: round coordinates to match positions
    cat_c = candidates[candidates["category"] == "C"].copy()
    false_positives = set()
    fp_results_log = []

    for _, row in cat_c.iterrows():
        sid = str(int(row["gaia_source_id"]))
        ra, dec = float(row["ra"]), float(row["dec"])
        gaia_g = float(row["gaia_g_mag"])

        # Match by proximity (within 1" of requested position)
        if "ra" in fp_all.columns and "dec" in fp_all.columns:
            dist = np.sqrt(
                ((fp_all["ra"] - ra) * np.cos(np.radians(dec))) ** 2
                + (fp_all["dec"] - dec) ** 2
            ) * 3600  # to arcsec
            mask = dist < 1.0
        else:
            continue

        subset = fp_all[mask]
        if len(subset) == 0:
            continue

        result = analyse_lightcurve(subset, ra, dec, gaia_g)
        result["gaia_source_id"] = sid
        result["ra"] = ra
        result["dec"] = dec
        result["gaia_g_mag"] = gaia_g
        fp_results_log.append(result)

        if result["is_detected"]:
            false_positives.add(sid)
            logger.info(
                "  %s: DETECTED in ZTF forced phot "
                "(n=%d, median_mag=%.1f, Gaia_G=%.1f, delta=%.1f)",
                sid, result["n_detections"],
                result.get("median_mag", 0), gaia_g,
                result.get("mag_diff", 0),
            )

    # Save analysis log
    if fp_results_log:
        log_path = RESULTS_DIR / "forced_phot_analysis.csv"
        pd.DataFrame(fp_results_log).to_csv(log_path, index=False)
        logger.info("Forced phot analysis log: %s", log_path)

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
        logger.info(
            "Forced photometry: no false positives found (or no results yet)"
        )

    return candidates
