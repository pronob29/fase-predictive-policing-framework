"""
fase/data/build_tensors.py
==========================
Phase 1 — Data Engineering & Tensor Formatting

Converts raw Baltimore point-level incident records into a spatiotemporal
graph tensor X of shape (num_nodes, num_timesteps, num_features) ready for
the FASE STGNN + Hawkes model.

Graph nodes : ZCTA polygons clipped to the Baltimore study area
Time axis   : Discretised into `freq` bins (default '1H'; supports '8H', etc.)

Feature layout (F = 13 per node per timestep)
----------------------------------------------
Idx  Name               Type       Notes
---  -----------------  ---------  ----------------------------------------
 0   lag_1              dynamic    incident count at t-1
 1   lag_2              dynamic    incident count at t-2
 2   lag_3              dynamic    incident count at t-3
 3   roll_mean_24h      dynamic    mean of counts in past 24 × freq bins
 4   roll_mean_7d       dynamic    mean of counts in past 7 d of bins
 5   sin_tod            dynamic    sin(2π · hour_of_day / 24)
 6   cos_tod            dynamic    cos(2π · hour_of_day / 24)
 7   sin_dow            dynamic    sin(2π · day_of_week / 7)
 8   cos_dow            dynamic    cos(2π · day_of_week / 7)
 9   pct_minority       static     ACS % minority pop  (broadcast over T)
10   median_income_norm static     ACS median income, min-max scaled [0,1]
11   poverty_rate       static     ACS poverty rate    (broadcast over T)
12   patrol_exposure    dynamic    historical patrol exposure (zeros; updated
                                   by the feedback simulator in Phase 5)

Rolling features at timestep t use only counts at t-1, t-2, …  (no leakage).

Output artefacts (written to data/tensors/baltimore/)
------------------------------------------------------
  X_{train,val,test}.pt   float32 tensor (N, T_split, F)
  y_{train,val,test}.pt   float32 tensor (N, T_split)     raw counts
  meta.pt                 dict   { node_ids, feat_names, time_*, acs, zcta_gdf }

Usage
-----
  python -m fase.data.build_tensors [--freq 1H] [--split 0.70 0.15 0.15]
                                    [--years 2017 2018 2019]
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)

# ── Project path roots ────────────────────────────────────────────────────────
ROOT           = Path(__file__).resolve().parents[2]
DATA_RAW       = ROOT / "data" / "raw"
DATA_TENSORS   = ROOT / "data" / "tensors"
SHAPEFILE_DIR  = DATA_RAW / "shapefiles" / "zcta"
ACS_CSV        = DATA_RAW / "acs_data.csv"
INCIDENT_CSV   = DATA_RAW / "baltimore_part1_crime.csv"

# Baltimore bounding box (WGS-84 decimal degrees)
BALT_BBOX: Dict[str, float] = dict(
    lat_min=39.197, lat_max=39.372,
    lon_min=-76.713, lon_max=-76.529,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Incident loading
# ─────────────────────────────────────────────────────────────────────────────

def load_incidents(
    path: Path = INCIDENT_CSV,
    years: Optional[List[int]] = None,
) -> gpd.GeoDataFrame:
    """
    Load raw Baltimore Part-1 crime records and return a GeoDataFrame.

    Handles the BOM-prefixed header and the city's mixed AM/PM datetime format.
    Points outside the study bounding box and rows with missing coordinates or
    timestamps are silently dropped.

    Parameters
    ----------
    path  : path to the raw CSV (default: data/raw/baltimore_part1_crime.csv)
    years : optional list of years to keep, e.g. [2017, 2018, 2019].
            If None, all available years are used.

    Returns
    -------
    GeoDataFrame with CRS EPSG:4326 and at minimum these columns:
        datetime (Timestamp), lat (float), lon (float), geometry (Point)
    All original columns are preserved.
    """
    df = pd.read_csv(path, low_memory=False, encoding="utf-8-sig")  # utf-8-sig strips BOM

    # Normalise the key column names; the rest are kept as-is
    df = df.rename(columns={
        "CrimeDateTime": "datetime",
        "Latitude":      "lat",
        "Longitude":     "lon",
    })

    # Parse datetime — raw file uses format "8/29/2014 12:00:00 PM"
    df["datetime"] = pd.to_datetime(
        df["datetime"],
        format="%m/%d/%Y %I:%M:%S %p",
        errors="coerce",
    )
    df = df.dropna(subset=["lat", "lon", "datetime"])
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df = df.dropna(subset=["lat", "lon"])

    # Clip to study bounding box
    bb = BALT_BBOX
    df = df[
        (df["lat"] >= bb["lat_min"]) & (df["lat"] <= bb["lat_max"]) &
        (df["lon"] >= bb["lon_min"]) & (df["lon"] <= bb["lon_max"])
    ].copy()

    # Optional year filter
    if years is not None:
        df["_year"] = df["datetime"].dt.year
        df = df[df["_year"].isin(years)].drop(columns=["_year"])

    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["lon"], df["lat"]),
        crs="EPSG:4326",
    )

    logger.info(
        "Loaded %d incidents from %s  (years: %s)",
        len(gdf),
        path.name,
        sorted(gdf["datetime"].dt.year.unique()) if len(gdf) else "—",
    )
    return gdf


# ─────────────────────────────────────────────────────────────────────────────
# 2. ZCTA shapefile loading
# ─────────────────────────────────────────────────────────────────────────────

def load_zcta_shapefile(shapefile_dir: Path = SHAPEFILE_DIR) -> gpd.GeoDataFrame:
    """
    Load Census ZCTA polygons and return those whose centroids fall within
    the Baltimore study bounding box.

    Resolution order
    ----------------
    1. First *.shp file found in `shapefile_dir`  (recommended for
       reproducibility — place tl_2020_us_zcta520.shp there).
       Download from:
         https://www2.census.gov/geo/tiger/TIGER2020/ZCTA5/
    2. Dynamic download via ``pygris.zctas(year=2020, cb=True)``
       (requires an internet connection and ``pip install pygris``).

    Returned GeoDataFrame has exactly two columns: ZCTA5 (str), geometry.
    CRS is normalised to EPSG:4326.
    """
    shp_files = sorted(shapefile_dir.glob("*.shp")) if shapefile_dir.exists() else []

    if shp_files:
        zcta = gpd.read_file(shp_files[0])
        logger.info(
            "Loaded ZCTA shapefile from %s  (%d polygons before clip)",
            shp_files[0].name, len(zcta),
        )
    else:
        logger.info(
            "No local shapefile in %s — trying pygris …", shapefile_dir
        )
        try:
            import pygris  # type: ignore
            zcta = pygris.zctas(year=2020, cb=True)
            logger.info(
                "Downloaded ZCTA polygons via pygris  (%d total)", len(zcta)
            )
        except ImportError as exc:
            raise FileNotFoundError(
                f"No shapefile found in {shapefile_dir} and pygris is not installed.\n"
                "Options:\n"
                "  a) Download the national ZCTA file from Census TIGER/Line:\n"
                "       https://www2.census.gov/geo/tiger/TIGER2020/ZCTA5/\n"
                "     and unzip it into:  data/raw/shapefiles/zcta/\n"
                "  b) Install pygris:  pip install pygris\n"
            ) from exc

    zcta = zcta.to_crs("EPSG:4326")

    # Identify the ZCTA identifier column (varies by Census year/product)
    id_col = next(
        (
            c for c in zcta.columns
            if c.upper().startswith("ZCTA5CE") or c.upper() in ("GEOID", "GEOID20", "ZCTA5")
        ),
        None,
    )
    if id_col is None:
        raise KeyError(
            f"Cannot find a ZCTA identifier column in {list(zcta.columns)}.\n"
            "Expected one of: ZCTA5CE10, ZCTA5CE20, GEOID, GEOID20, ZCTA5."
        )
    zcta = zcta.rename(columns={id_col: "ZCTA5"})
    zcta["ZCTA5"] = zcta["ZCTA5"].astype(str).str.zfill(5)

    # Clip to Baltimore bbox using Census internal-point columns when available
    # (avoids computing centroids in a geographic CRS, which raises a warning).
    bb = BALT_BBOX
    lat_col = next((c for c in zcta.columns if "INTPTLAT" in c.upper()), None)
    lon_col = next((c for c in zcta.columns if "INTPTLON" in c.upper()), None)
    if lat_col and lon_col:
        pt_lat = zcta[lat_col].astype(float)
        pt_lon = zcta[lon_col].astype(float)
    else:
        # Fallback: reproject to UTM 18N for accurate centroid, then back
        projected = zcta.to_crs("EPSG:32618")
        ctr = projected.geometry.centroid.to_crs("EPSG:4326")
        pt_lat, pt_lon = ctr.y, ctr.x
    mask = (
        (pt_lat >= bb["lat_min"]) & (pt_lat <= bb["lat_max"]) &
        (pt_lon >= bb["lon_min"]) & (pt_lon <= bb["lon_max"])
    )
    zcta = zcta[mask][["ZCTA5", "geometry"]].reset_index(drop=True)

    if len(zcta) == 0:
        raise ValueError(
            "Zero ZCTAs retained after clipping to the Baltimore bbox.\n"
            "Make sure you are using the national (not state-level) ZCTA file,\n"
            "or a file that covers Maryland (FIPS 24)."
        )

    logger.info("Retained %d ZCTAs within Baltimore bbox.", len(zcta))
    return zcta


# ─────────────────────────────────────────────────────────────────────────────
# 3. Spatial join: incidents → ZCTAs
# ─────────────────────────────────────────────────────────────────────────────

def join_incidents_to_zcta(
    incidents: gpd.GeoDataFrame,
    zcta: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Spatially join incident points to ZCTA polygons (inner join).

    Points that do not fall inside any ZCTA polygon are dropped and a warning
    is logged.  This is normal for a small fraction of events that lie on
    polygon boundaries or just outside the clipped region.
    """
    joined = gpd.sjoin(
        incidents,
        zcta[["ZCTA5", "geometry"]],
        how="inner",
        predicate="within",
    )
    joined = joined.drop(columns=["index_right"])

    dropped = len(incidents) - len(joined)
    if dropped > 0:
        logger.warning(
            "Spatial join: %d / %d incidents fell outside all ZCTA polygons "
            "and were dropped (%.1f%%).",
            dropped, len(incidents), 100 * dropped / max(len(incidents), 1),
        )
    logger.info(
        "Spatial join complete: %d incidents assigned to %d distinct ZCTAs.",
        len(joined), joined["ZCTA5"].nunique(),
    )
    return joined


# ─────────────────────────────────────────────────────────────────────────────
# 4. ACS data loading
# ─────────────────────────────────────────────────────────────────────────────

# Maps Census API variable names → our canonical column names.
# Also handles alternative pre-processed column names a user might supply.
_ACS_COL_MAP = {
    # Census API raw variables (ACS 5-year)
    "B02001_001E": "pop_total",
    "B02001_003E": "pop_black",
    "B19013_001E": "median_income",
    "B17001_002E": "pop_poverty",
    "B17001_001E": "poverty_denom",
    # Common pre-computed column names
    "pct_black":               "pct_minority",
    "pct_nonwhite":            "pct_minority",
    "percent_minority":        "pct_minority",
    "median_household_income": "median_income",
    "MedInc":                  "median_income",
    "PovRate":                 "poverty_rate",
    "percent_poverty":         "poverty_rate",
}


def load_acs_data(
    acs_csv: Path = ACS_CSV,
    zcta_ids: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Load ACS 5-year ZCTA-level demographic estimates.

    Resolution order
    ----------------
    1. ``acs_csv`` if the file exists  (strongly recommended for
       reproducibility).  Provide a CSV with at minimum:
         ZCTA5 (or GEOID), plus any of the columns listed in _ACS_COL_MAP.
       Typical minimal schema:
         ZCTA5 | pct_minority | median_income | poverty_rate
       or the raw Census API variables:
         ZCTA5 | B02001_001E | B02001_003E | B19013_001E | B17001_001E | B17001_002E
    2. Dynamic fetch via the ``censusdata`` package
       (requires internet + ``pip install censusdata``).

    Returns a DataFrame with columns:
        ZCTA5 (str), pct_minority (float), median_income_norm (float [0,1]),
        poverty_rate (float)

    ZCTAs not present in the ACS data get 0-filled features with a warning.
    """
    if acs_csv.exists():
        df = pd.read_csv(acs_csv, dtype=str)
        df.columns = df.columns.str.strip()

        # Unify GEOID → ZCTA5
        if "GEOID" in df.columns and "ZCTA5" not in df.columns:
            df = df.rename(columns={"GEOID": "ZCTA5"})
        df["ZCTA5"] = df["ZCTA5"].astype(str).str.zfill(5)

        # Rename any recognised columns to canonical names
        df = df.rename(
            columns={k: v for k, v in _ACS_COL_MAP.items() if k in df.columns}
        )

        # Convert numeric columns from string
        num_cols = [
            "pop_total", "pop_black", "median_income",
            "pop_poverty", "poverty_denom",
            "pct_minority", "poverty_rate",
        ]
        for col in num_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        logger.info("Loaded ACS data from %s  (%d rows).", acs_csv.name, len(df))
    else:
        logger.info(
            "acs_data.csv not found at %s — attempting censusdata API …", acs_csv
        )
        df = _fetch_acs_api(zcta_ids)

    df = _derive_acs_features(df)

    if zcta_ids is not None:
        present = set(df["ZCTA5"])
        missing = sorted(set(zcta_ids) - present)
        if missing:
            logger.warning(
                "%d ZCTAs have no ACS data and will receive zero-filled "
                "demographic features: %s%s",
                len(missing),
                missing[:5],
                " …" if len(missing) > 5 else "",
            )
        df = df[df["ZCTA5"].isin(zcta_ids)].copy()

    return df[["ZCTA5", "pct_minority", "median_income_norm", "poverty_rate"]]


def _fetch_acs_api(zcta_ids: Optional[List[str]] = None) -> pd.DataFrame:
    """Pull ACS 5-year (2019) table B02001/B19013/B17001 via censusdata."""
    try:
        import censusdata  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "The censusdata package is required for API-based ACS fetching.\n"
            "Install it with:  pip install censusdata\n"
            "Or provide:       data/raw/acs_data.csv"
        ) from exc

    variables = [
        "B02001_001E",  # total population
        "B02001_003E",  # Black or African-American alone
        "B19013_001E",  # median household income
        "B17001_001E",  # poverty status denominator
        "B17001_002E",  # below poverty level
    ]
    logger.info("Fetching ACS 5-year (2019) data for all US ZCTAs …")
    geo = censusdata.censusgeo([("zip code tabulation area", "*")])
    raw = censusdata.download("acs5", 2019, geo, variables, tabletype="detail")
    raw = raw.reset_index()

    # Extract the 5-digit ZCTA from the geo description string
    raw["ZCTA5"] = (
        raw["index"]
        .astype(str)
        .str.extract(r"(\d{5})", expand=False)
        .str.zfill(5)
    )
    raw = raw.rename(columns={
        "B02001_001E": "pop_total",
        "B02001_003E": "pop_black",
        "B19013_001E": "median_income",
        "B17001_001E": "poverty_denom",
        "B17001_002E": "pop_poverty",
    })
    logger.info("API fetch complete: %d ZCTA rows returned.", len(raw))
    return raw


def _derive_acs_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute pct_minority, poverty_rate, and median_income_norm from whatever
    columns are available.  Missing inputs produce 0.0 with a warning.
    """
    # ── pct_minority ──
    if "pct_minority" not in df.columns:
        if {"pop_black", "pop_total"}.issubset(df.columns):
            df["pct_minority"] = (
                df["pop_black"].clip(lower=0)
                / df["pop_total"].replace(0, np.nan)
            ).fillna(0.0)
        else:
            logger.warning(
                "Cannot derive pct_minority (need pop_black + pop_total "
                "or a pre-computed pct_minority column).  Defaulting to 0."
            )
            df["pct_minority"] = 0.0
    df["pct_minority"] = df["pct_minority"].clip(0.0, 1.0)

    # ── poverty_rate ──
    if "poverty_rate" not in df.columns:
        denom_col = next(
            (c for c in ("poverty_denom", "pop_total") if c in df.columns), None
        )
        if "pop_poverty" in df.columns and denom_col is not None:
            df["poverty_rate"] = (
                df["pop_poverty"].clip(lower=0)
                / df[denom_col].replace(0, np.nan)
            ).fillna(0.0)
        else:
            logger.warning(
                "Cannot derive poverty_rate.  Defaulting to 0."
            )
            df["poverty_rate"] = 0.0
    df["poverty_rate"] = df["poverty_rate"].clip(0.0, 1.0)

    # ── median_income_norm (min-max scaled across nodes) ──
    # Census suppresses low-count ZCTAs with -666666666; treat as missing.
    inc = pd.to_numeric(
        df.get("median_income", pd.Series(0.0, index=df.index)),
        errors="coerce",
    ).replace(-666666666, np.nan).fillna(0.0).clip(lower=0)
    inc_min, inc_max = float(inc.min()), float(inc.max())
    df["median_income_norm"] = (
        (inc - inc_min) / max(inc_max - inc_min, 1.0)
    ).astype(np.float32)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 5. Temporal binning
# ─────────────────────────────────────────────────────────────────────────────

def _normalise_freq(freq: str) -> str:
    """Translate deprecated uppercase pandas frequency aliases (e.g. '1H' → '1h')."""
    _map = {"H": "h", "T": "min", "S": "s", "M": "ME", "A": "YE"}
    import re
    return re.sub(r"([A-Z]+)", lambda m: _map.get(m.group(), m.group()), freq)


def bin_incidents(
    joined: gpd.GeoDataFrame,
    freq: str = "1h",
) -> pd.DataFrame:
    """
    Aggregate incident points into a (ZCTA5 × time_bin) count table.

    Every (ZCTA5, time_bin) pair in the full Cartesian product of observed
    ZCTAs and all bins in the study window is represented — zero-count cells
    are explicitly filled rather than omitted.

    Parameters
    ----------
    joined : GeoDataFrame produced by join_incidents_to_zcta()
    freq   : pandas offset alias, e.g. '1H', '8H', '1D'

    Returns
    -------
    DataFrame with columns: ZCTA5 (str), time_bin (Timestamp), count (int)
    Shape: (num_zctas × num_time_bins, 3)
    """
    freq = _normalise_freq(freq)
    df = joined.copy()
    df["time_bin"] = df["datetime"].dt.floor(freq)

    counts = (
        df.groupby(["ZCTA5", "time_bin"])
        .size()
        .rename("count")
        .reset_index()
    )

    # Full Cartesian grid so downstream rolling never sees gaps
    all_zctas = sorted(counts["ZCTA5"].unique())
    all_bins = pd.date_range(
        start=counts["time_bin"].min(),
        end=counts["time_bin"].max(),
        freq=freq,
    )
    grid = pd.MultiIndex.from_product(
        [all_zctas, all_bins], names=["ZCTA5", "time_bin"]
    )
    counts = (
        counts.set_index(["ZCTA5", "time_bin"])
        .reindex(grid, fill_value=0)
        .reset_index()
    )
    counts["count"] = counts["count"].astype(np.int32)

    nonzero_pct = 100.0 * (counts["count"] > 0).mean()
    logger.info(
        "Binned into %d ZCTAs × %d time bins (%s).  "
        "Non-zero cells: %.2f%%  (sparsity: %.2f%%).",
        len(all_zctas), len(all_bins), freq,
        nonzero_pct, 100 - nonzero_pct,
    )
    return counts


# ─────────────────────────────────────────────────────────────────────────────
# 6. Feature engineering
# ─────────────────────────────────────────────────────────────────────────────

def _freq_to_hours(freq: str) -> float:
    """Return the number of hours represented by one freq bin."""
    offset = pd.tseries.frequencies.to_offset(_normalise_freq(freq))
    return offset.nanos / 3_600_000_000_000.0


# Public constant: canonical feature names in index order
FEATURE_NAMES: List[str] = [
    "lag_1",
    "lag_2",
    "lag_3",
    "roll_mean_24h",
    "roll_mean_7d",
    "sin_tod",
    "cos_tod",
    "sin_dow",
    "cos_dow",
    "pct_minority",
    "median_income_norm",
    "poverty_rate",
    "patrol_exposure",
]


def engineer_features(
    counts: pd.DataFrame,
    acs: pd.DataFrame,
    freq: str = "1H",
) -> Tuple[np.ndarray, np.ndarray, List[str], pd.DatetimeIndex, List[str]]:
    """
    Construct the spatiotemporal feature tensor X and target tensor y.

    Parameters
    ----------
    counts : output of bin_incidents() — columns [ZCTA5, time_bin, count]
    acs    : output of load_acs_data() — columns [ZCTA5, pct_minority,
             median_income_norm, poverty_rate]
    freq   : temporal resolution (must match what was passed to bin_incidents)

    Returns
    -------
    X          : float32 ndarray, shape (N, T, F)   feature tensor
    y          : float32 ndarray, shape (N, T)       raw count targets
    node_ids   : list[str] of ZCTA5 codes, length N
    time_index : pd.DatetimeIndex of length T
    feat_names : list[str] of feature names, length F  (= FEATURE_NAMES)

    Design notes
    ------------
    * Lagged counts and rolling means are computed from the SHIFTED count
      series so that features at time t only use data from t-1, t-2, …
      This prevents any target-leakage into the feature matrix.
    * Static ACS features are broadcast (tiled) across all T timesteps.
    * The patrol_exposure column is initialised to 0 and updated in-place
      by the Phase-5 feedback simulator.
    """
    hours_per_bin = _freq_to_hours(freq)
    bins_24h = max(1, int(round(24.0  / hours_per_bin)))
    bins_7d  = max(1, int(round(168.0 / hours_per_bin)))

    # Pivot: rows = time_bin (T), columns = ZCTA5 (N)
    pivot = (
        counts.pivot(index="time_bin", columns="ZCTA5", values="count")
        .fillna(0.0)
        .astype(np.float32)
    )
    time_index: pd.DatetimeIndex = pivot.index
    node_ids: List[str] = list(pivot.columns)
    T, N = pivot.shape

    count_mat = pivot.values  # (T, N)

    # ── Lagged counts (shift by 1/2/3 steps, zero-pad head) ──────────────────
    lag1 = np.concatenate([np.zeros((1, N), dtype=np.float32), count_mat[:-1]], axis=0)
    lag2 = np.concatenate([np.zeros((2, N), dtype=np.float32), count_mat[:-2]], axis=0)
    lag3 = np.concatenate([np.zeros((3, N), dtype=np.float32), count_mat[:-3]], axis=0)

    # ── Rolling means over the PAST (shift first, then roll) ─────────────────
    # shift(1) moves the series forward by one step so the window at t
    # covers [t-1, t-2, …] — strictly no look-ahead.
    shifted = pivot.shift(1).fillna(0.0)
    roll_24 = (
        shifted.rolling(window=bins_24h, min_periods=1).mean().values.astype(np.float32)
    )
    roll_7d = (
        shifted.rolling(window=bins_7d, min_periods=1).mean().values.astype(np.float32)
    )

    # ── Time-of-day / day-of-week sinusoidal embeddings (shape T, → T × N) ───
    hours_of_day = (
        np.array(time_index.hour, dtype=np.float32)
        + np.array(time_index.minute, dtype=np.float32) / 60.0
    )
    days_of_week = np.array(time_index.dayofweek, dtype=np.float32)

    # Tile across nodes so every feature array is (T, N)
    sin_tod = np.tile(np.sin(2 * np.pi * hours_of_day / 24.0)[:, None], (1, N))
    cos_tod = np.tile(np.cos(2 * np.pi * hours_of_day / 24.0)[:, None], (1, N))
    sin_dow = np.tile(np.sin(2 * np.pi * days_of_week  /  7.0)[:, None], (1, N))
    cos_dow = np.tile(np.cos(2 * np.pi * days_of_week  /  7.0)[:, None], (1, N))

    # ── Static ACS features (shape N, → T × N via tiling) ───────────────────
    acs_idx = acs.set_index("ZCTA5").reindex(node_ids)

    def _static(col: str) -> np.ndarray:
        vals = acs_idx[col].fillna(0.0).values.astype(np.float32)  # (N,)
        return np.tile(vals[None, :], (T, 1))                        # (T, N)

    pct_min  = _static("pct_minority")
    med_inc  = _static("median_income_norm")
    pov_rate = _static("poverty_rate")

    # ── Patrol exposure placeholder ───────────────────────────────────────────
    patrol_exp = np.zeros((T, N), dtype=np.float32)

    # ── Stack → (T, N, F) then transpose to (N, T, F) ───────────────────────
    X_tnf = np.stack(
        [
            lag1, lag2, lag3,
            roll_24, roll_7d,
            sin_tod, cos_tod, sin_dow, cos_dow,
            pct_min, med_inc, pov_rate,
            patrol_exp,
        ],
        axis=-1,
    ).astype(np.float32)  # (T, N, F)

    X = X_tnf.transpose(1, 0, 2)  # (N, T, F)
    y = count_mat.T.astype(np.float32)  # (N, T)

    logger.info(
        "Feature tensor built:  X=%s  y=%s  F=%d features.",
        X.shape, y.shape, len(FEATURE_NAMES),
    )
    return X, y, node_ids, time_index, FEATURE_NAMES


# ─────────────────────────────────────────────────────────────────────────────
# 7. Sequential train / val / test split
# ─────────────────────────────────────────────────────────────────────────────

def sequential_split(
    X: np.ndarray,
    y: np.ndarray,
    time_index: pd.DatetimeIndex,
    ratios: Tuple[float, float, float] = (0.70, 0.15, 0.15),
) -> Dict[str, Dict]:
    """
    Split tensors along the *time* axis (axis=1) preserving chronological order.

    Random splitting is intentionally NOT used: it would cause the model to
    'see the future' during training and invalidate out-of-sample evaluation.

    Parameters
    ----------
    X, y       : tensors of shape (N, T, F) and (N, T)
    time_index : DatetimeIndex of length T
    ratios     : (train, val, test) fractions that must sum to 1

    Returns
    -------
    dict with keys 'train', 'val', 'test'.  Each value is a dict:
        { 'X': ndarray (N, T_split, F),
          'y': ndarray (N, T_split),
          'time_index': DatetimeIndex of length T_split }
    """
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {sum(ratios):.4f}")

    T = X.shape[1]
    t1 = int(T * ratios[0])
    t2 = t1 + int(T * ratios[1])

    splits: Dict[str, Dict] = {
        "train": {
            "X": X[:, :t1, :],
            "y": y[:, :t1],
            "time_index": time_index[:t1],
        },
        "val": {
            "X": X[:, t1:t2, :],
            "y": y[:, t1:t2],
            "time_index": time_index[t1:t2],
        },
        "test": {
            "X": X[:, t2:, :],
            "y": y[:, t2:],
            "time_index": time_index[t2:],
        },
    }

    for name, d in splits.items():
        logger.info(
            "%-5s  X=%-20s  y=%-16s  [%s  →  %s]",
            name.upper(), str(d["X"].shape), str(d["y"].shape),
            d["time_index"][0].date(), d["time_index"][-1].date(),
        )
    return splits


# ─────────────────────────────────────────────────────────────────────────────
# 8. Persist artefacts
# ─────────────────────────────────────────────────────────────────────────────

def save_tensors(
    splits: Dict[str, Dict],
    node_ids: List[str],
    feat_names: List[str],
    acs: pd.DataFrame,
    zcta: gpd.GeoDataFrame,
    out_dir: Path = DATA_TENSORS,
) -> Path:
    """
    Save all tensors and metadata to ``out_dir/baltimore/``.

    Written files
    -------------
    X_train.pt, X_val.pt, X_test.pt   float32 Tensor (N, T_split, F)
    y_train.pt, y_val.pt, y_test.pt   float32 Tensor (N, T_split)
    meta.pkl                           Python pickle dict with keys:
        node_ids    — list[str] of ZCTA5 codes (length N)
        feat_names  — list[str] of feature names (length F)
        time_train  — DatetimeIndex for train split
        time_val    — DatetimeIndex for val split
        time_test   — DatetimeIndex for test split
        acs         — DataFrame with ACS features indexed by ZCTA5
        zcta_gdf    — GeoDataFrame with ZCTA5 + geometry columns

    Note: X/y tensors use torch.save (safe for float32 tensors).
    meta uses stdlib pickle because it carries pandas/geopandas objects
    that are incompatible with torch's weights_only loader.
    """
    out = out_dir / "baltimore"
    out.mkdir(parents=True, exist_ok=True)

    for split, d in splits.items():
        torch.save(torch.from_numpy(d["X"]), out / f"X_{split}.pt")
        torch.save(torch.from_numpy(d["y"]), out / f"y_{split}.pt")
        logger.info("Saved X_%s.pt  y_%s.pt  →  %s", split, split, out)

    meta = {
        "node_ids":   node_ids,
        "feat_names": feat_names,
        "time_train": splits["train"]["time_index"],
        "time_val":   splits["val"]["time_index"],
        "time_test":  splits["test"]["time_index"],
        "acs":        acs,
        "zcta_gdf":   zcta,
    }
    with open(out / "meta.pkl", "wb") as fh:
        pickle.dump(meta, fh, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Saved meta.pkl  →  %s", out)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 9. Top-level pipeline
# ─────────────────────────────────────────────────────────────────────────────

def build(
    freq: str = "1H",
    split_ratios: Tuple[float, float, float] = (0.70, 0.15, 0.15),
    years: Optional[List[int]] = None,
    incident_csv: Path = INCIDENT_CSV,
    shapefile_dir: Path = SHAPEFILE_DIR,
    acs_csv: Path = ACS_CSV,
    out_dir: Path = DATA_TENSORS,
) -> Dict[str, Dict]:
    """
    Run the full Phase-1 pipeline end-to-end.

    Parameters
    ----------
    freq         : temporal bin frequency ('1H', '8H', …)
    split_ratios : (train, val, test) fractions summing to 1
    years        : year filter, e.g. [2017, 2018, 2019].  None = all years.
    incident_csv : path to raw Baltimore Part-1 crime CSV
    shapefile_dir: directory containing Census ZCTA .shp file
    acs_csv      : path to ACS demographic CSV (or path that will be skipped
                   if absent, triggering API fallback)
    out_dir      : root output directory; tensors saved under out_dir/baltimore/

    Returns
    -------
    splits dict (same as sequential_split output) for immediate downstream use.
    """
    logger.info("=" * 60)
    logger.info("FASE Phase 1 — building spatiotemporal tensors")
    logger.info("  freq=%s  years=%s  split=%s", freq, years, split_ratios)
    logger.info("=" * 60)

    incidents = load_incidents(incident_csv, years=years)
    zcta      = load_zcta_shapefile(shapefile_dir)
    joined    = join_incidents_to_zcta(incidents, zcta)
    acs       = load_acs_data(acs_csv, zcta_ids=list(zcta["ZCTA5"].unique()))
    counts    = bin_incidents(joined, freq=freq)
    X, y, node_ids, time_index, feat_names = engineer_features(
        counts, acs, freq=freq
    )
    splits    = sequential_split(X, y, time_index, ratios=split_ratios)
    out       = save_tensors(splits, node_ids, feat_names, acs, zcta, out_dir=out_dir)

    logger.info("Phase 1 complete.  Artefacts in: %s", out)
    return splits


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="FASE Phase 1: build spatiotemporal graph tensors from "
                    "Baltimore crime incident data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--freq", default="1h",
        help="Temporal bin width.  '1h' = 1 hour, '8h' = 8-hour dayparts.",
    )
    parser.add_argument(
        "--split", nargs=3, type=float, default=[0.70, 0.15, 0.15],
        metavar=("TRAIN", "VAL", "TEST"),
        help="Sequential train/val/test split ratios (must sum to 1.0).",
    )
    parser.add_argument(
        "--years", nargs="+", type=int, default=None,
        metavar="YEAR",
        help="Years to include, e.g. --years 2017 2018 2019.  "
             "Default: all available years in the raw file.",
    )
    parser.add_argument(
        "--incident-csv", type=Path, default=INCIDENT_CSV,
        help="Path to raw Baltimore Part-1 crime CSV.",
    )
    parser.add_argument(
        "--shapefile-dir", type=Path, default=SHAPEFILE_DIR,
        help="Directory containing Census ZCTA *.shp file.",
    )
    parser.add_argument(
        "--acs-csv", type=Path, default=ACS_CSV,
        help="Path to ACS demographic CSV.  "
             "Falls back to censusdata API if absent.",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=DATA_TENSORS,
        help="Root output directory.  Tensors written to out-dir/baltimore/.",
    )
    args = parser.parse_args()

    build(
        freq=args.freq,
        split_ratios=tuple(args.split),
        years=args.years,
        incident_csv=args.incident_csv,
        shapefile_dir=args.shapefile_dir,
        acs_csv=args.acs_csv,
        out_dir=args.out_dir,
    )
