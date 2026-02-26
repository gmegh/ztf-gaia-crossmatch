# ZTF–Gaia Cross-Match Pipeline

Cross-match ZTF DR23 and Gaia DR3 catalogs to find sources in ZTF that are missing or uncharacterized as variable in Gaia.

## Setup

```bash
python3 -m venv ztf-gaia
ztf-gaia/bin/pip install -r requirements.txt
```

## Usage

Run the pilot tile (single HEALPix tile at RA=180, Dec=+30):

```bash
ztf-gaia/bin/python -m src.pipeline
```

## Pipeline Overview

1. **TAP Queries** — Download ZTF DR23 objects and Gaia DR3 sources/variables for a HEALPix tile
2. **Cross-match** — Positional match (1.5 arcsec) to identify:
   - **Category A**: ZTF sources with no Gaia counterpart
   - **Category B**: ZTF sources with a Gaia match but not flagged as variable
3. **Scoring** — Rank candidates by variability amplitude, number of epochs, chi-squared, etc.
4. **Output** — Scored candidate catalogs saved as Parquet files in `results/`

## Project Structure

- `src/` — Pipeline source code
- `templates/` — Jinja2 HTML templates (future website)
- `static/` — CSS/JS for website
- `results/` — Output catalogs
- `notebooks/` — Exploratory analysis
