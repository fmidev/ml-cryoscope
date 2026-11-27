#!/usr/bin/env python3
"""
cryo-ecxens-performance.py

Validate ECXENS SWE forecasts against SNOWCAP SWE (Scandinavia 500 m), CryoSCOPE layout.

usage: python cryo-ecxens-performance.py [BASE] [MODEL] [redo | redo-adjusted | redo-era5l]      e.g. python cryo-ecxens-performance.py 20191210 swe-ml-2

The first run computes everything. A later run computes only what is missing: every prediction keeps in the state file
(.state-raw.json / .state-adjusted.json in the run folder) the inputs it was made from (ECXENS, ECENS, ECENS remapped to
500 m, SNOWCAP and CLMS files with their modification times). A prediction whose inputs are the same is skipped; one with a
new input (e.g. the 500 m ECENS file made after the first run, or a state file from before this check) is validated again
with its existing pngs kept, so only the missing figures and scores are made; one whose earlier input changed (e.g. a new
SNOWCAP file) is validated again with every figure redrawn. The member ranking and time series are reused, and a comparison (adjusted-vs-raw/, adjusted-era5l-vs-raw/) that is finished
(.state-cmp-<tag>.json) is skipped unless raw ECXENS or that product was done again. So by default only what is missing
is run, e.g. just the ERA5-Land adjusted product and its comparison when that is new. 'redo' as the third argument does everything again; 'redo-adjusted' does both adjusted
products and their comparisons again and keeps the raw results; 'redo-era5l' only the ERA5-Land adjusted one. A product done again (also when its
ECXENS files changed) first has its old folders removed (<pred>-adjusted/ for adjusted, the plain <pred>/ for raw),
and adjusted-vs-raw/ is removed and made again whenever either product was done again.

One run validates both ECXENS products, each with the full set of results below, into the same run folder
OUT_ROOT/<MODEL>/<BASE>/ (one page in SnowLook):
  raw ECXENS        -> <pred>/            ensmean, member01, driest-memberNN, wettest-memberNN, members, all
  ECXENS adjusted   -> <pred>-adjusted/   ensmean-adjusted, member01-adjusted, ..., members-adjusted, all-adjusted
                       (SNOWCAP adjusted: ECXENS_<base>T120000_swe-adjusted-<m>.grib)
  ECXENS adjusted to ERA5-Land -> <pred>-adjusted-era5l/   (ECXENS_<base>T180000_swe-adjusted-era5l-<m>.grib)
                       (adjust-snow-ecxens.py, ECXENS_<base>T120000_swe-adjusted-<m>.grib, split by prep-ecxens-validation.sh;
                       skipped with a message when not there), so every adjusted file has 'adjusted' in its name,
                       e.g. swe-ml-2_20191210_ensmean-adjusted_ecxens-swe_20191211.png
  and each adjusted product against raw in adjusted-vs-raw/ and adjusted-era5l-vs-raw/ (see the end of this text).
  MODEL may be given with or without -adjusted. RUN_RAW / RUN_ADJUSTED switch either off (the other's results are kept).

Inputs, as prep-ecxens-validation.sh writes them, in ECX_DIR (/home/ubuntu/data/cryoscope/snowmodel/ECXENS/):
  ECXENS_<BT>_swe-<m>.grib, ECXENS_<BT>_swe-ensmean.grib        ECXENS SWE (mm) per member and ens mean, 500 m
  ECENS_<BT>_sd-<m>-fix.grib, ECENS_<BT>_sd-ensmean.grib        raw ECENS sd (m w.e.) per member and ens mean, 00 UTC
  ECENS_<BT>_sd-500m-ensmean.grib                               raw ECENS ens mean remapped to the SNOWCAP 500 m grid
  HKL_<first day>-<last day>_SWE.grib or HKL_<YYYYMM>_SWE.grib  SNOWCAP (HKL) SWE (m w.e.) of the forecast days
  Older layouts (one file with every member, other names) still work, see the *_TEMPLATES below.
  SNOWCAP is optional: days without SNOWCAP (or no SNOWCAP at all) still get the forecast maps, scores only where SNOWCAP exists.

Outputs, every figure in its own png, all in
  OUT_ROOT/<MODEL>/<BASE>/<pred>/          (OUT_ROOT = /home/ubuntu/data/cryoscope/snowmodel/ecxens_validation/)
  file names: <MODEL>_<BASE>_<pred>_<kind>[_<valid date>].png, e.g. swe-ml-2_20261121_ensmean_ecxens-swe_20261124.png
  <MODEL>_<BASE>_manifest.csv lists every output with model, base, pred, kind, valid date and lead day (for the dashboard).

Ens mean (pred = ensmean), every day:
  ecens-swe_D           raw ECENS SWE, raw ECENS grid (9 km)
  ecens500-swe_D        raw ECENS SWE remapped to the SNOWCAP grid (500 m)
  ecxens-swe_D          ECXENS SWE (500 m)
  snowcap-swe_D         SNOWCAP SWE (500 m)
  ecxens-bias_D         ECXENS minus SNOWCAP (500 m)
  ecens500-bias_D       ECENS remapped minus SNOWCAP (500 m)
  ecxens-vs-ecens-gain_D   |ECXENS bias| minus |ECENS bias| on the raw ECENS cells, negative = ECXENS closer to SNOWCAP
  over the period: ecxens-mean-bias, ecxens-rmse, ecens500-mean-bias, ecens500-rmse, ecxens-vs-ecens500-mae-gain,
  ecens-mean-bias, ecxens-vs-ecens-mae-gain, ecxens-density-scatter, domain-mean-timeseries (png and csv)
Members (member01, driest-memberNN, wettest-memberNN), every day: only the maps in MEMBER_KINDS
  (ecxens-swe, ecens-swe on the raw ECENS grid, ecxens-bias); the period figures and scores as for the ens mean.
Scores per lead day: ecxens-score-bias-mae, -rmse, -kge, -csi, -pod-far, -volume (ECENS remapped on the same plots),
  ecxens-vs-ecens500-skill, ecxens-vs-ecens-skill, ecxens-metrics.csv, info.json
Regions (REGIONS: Finnish, Swedish and Norwegian Lapland, central and southern parts; country borders from Natural Earth):
  ensmean: domain-mean-timeseries-regions (png, csv), regions-metrics.csv (daily scores per region), regions-score-summary.png
  members: ecxens-timeseries-regions.png, ecens-timeseries-regions.png
Over all members (pred = members): ecxens-timeseries.png, ecens-timeseries.png, timeseries.csv, ranking.csv
Over all predictions (pred = all): ecxens-metrics.csv
CLMS SWE (satellite, CLMS_20000101T000000_<day>T000000_swe_scandi.grib 5 km and ..._swe-500m_scandi.grib remapped to the
  SNOWCAP grid, written by get-clms-swe.sh, looked for in CLMS_DIRS), raw ECXENS ens mean only:
  clms-swe_D (5 km), clms500-swe_D (500 m) maps, and in domain-mean-timeseries (png, csv) and domain-mean-timeseries-regions
  the columns clms (CLMS 500 m) and snowcap_clms (SNOWCAP on the same pixels: CLMS leaves mountains and glaciers out)
Adjusted against raw (pred = adjusted-vs-raw), ens mean unless said:
  adjusted-vs-raw-score-rmse, -bias, -mae, -kge, -skill (per lead day; ECENS remapped on the same plots),
  adjusted-vs-raw-metrics.csv, adjusted-vs-raw-domain-mean-timeseries (png, csv), adjusted-vs-raw-timeseries-regions,
  adjusted-vs-raw-regions-scores, adjusted-vs-raw-members-timeseries (member spread of both),
  adjusted-vs-raw-mae-gain (|adjusted bias| minus |raw bias| over the SNOWCAP days, negative = adjusting helped),
  adjustment-mean (adjusted minus raw, mean over the forecast days)

Bias: forecast minus SNOWCAP (positive = too much snow). Two tier bias legend: soft greens / yellows up to
MILD_BIAS (50 mm), strong blues / reds beyond, so the bad areas stand out.
"""

import os
import sys
import shutil
import glob
import json
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm, LogNorm

# ---------------------------------------------------------------- config
BASE = sys.argv[1] if len(sys.argv) > 1 else '20261121'       # forecast base date, YYYYMMDD
MODEL = sys.argv[2] if len(sys.argv) > 2 else 'swe-ml-2'      # model edition that produced ECXENS (goes into every file name)
REDO = 'redo' in sys.argv[3:]   # True: everything again; False: finished predictions skipped, existing pngs kept
REDO_ADJ = REDO or 'redo-adjusted' in sys.argv[3:]   # every adjusted product and its comparison again, raw kept
REDO_ERA5L = REDO_ADJ or 'redo-era5l' in sys.argv[3:]  # the ERA5-Land adjusted product and its comparison again
SKIP_PNG = not REDO             # per product: False when its ECXENS files changed since the last run
ADJ_SUFFIX = '-adjusted'
BASE_MODEL = MODEL[:-len(ADJ_SUFFIX)] if MODEL.endswith(ADJ_SUFFIX) else MODEL
RUN_RAW = True            # validate raw ECXENS        -> <MODEL>/<BASE>/<pred>/
RUN_ADJUSTED = True       # validate ECXENS adjusted   -> <MODEL>/<BASE>/<pred>-adjusted/
COMPARE_ADJUSTED = True   # adjusted against raw       -> <MODEL>/<BASE>/adjusted-vs-raw/
MODEL = BASE_MODEL
PRED_TAG = ''             # '' for raw, the product's suffix ('-adjusted', '-adjusted-era5l') while it runs (set_product)
MODEL_LABEL = MODEL       # model name in the figure titles ('swe-ml-2 adjusted' for the adjusted product)
BASE_TIME = pd.Timestamp(BASE)
BT = f'{BASE_TIME:%Y%m%dT%H%M%S}'
BASE_TAG = f'{BASE_TIME:%Y%m%d}'
_fill = dict(BT=BT, BT12=f'{BASE_TIME:%Y%m%d}T120000', BT18=f'{BASE_TIME:%Y%m%d}T180000', MODEL=MODEL, Y=f'{BASE_TIME:%Y}')

SNOW_ROOT = '/home/ubuntu/data/cryoscope/snowmodel/'
ECX_DIR = SNOW_ROOT + 'ECXENS/'
ECENS_DIRS = [SNOW_ROOT + 'ECENS/', ECX_DIR, SNOW_ROOT]      # first directory holding a matching file wins
SNOWCAP_DIRS = [SNOW_ROOT + 'SNOWCAP/', ECX_DIR, SNOW_ROOT]
OUT_ROOT = SNOW_ROOT + 'ecxens_validation/'
OUTDIR = f'{OUT_ROOT}{MODEL}/{BASE_TAG}'
PREFIX = f'{MODEL}_{BASE_TAG}'


def set_product(tag=''):
    """Outputs of the following steps: raw ECXENS (tag '', <pred>/) or an adjusted product (tag in ADJ_PRODUCTS, <pred><tag>/)."""
    global PRED_TAG, MODEL_LABEL
    PRED_TAG = tag
    MODEL_LABEL = f"{MODEL} {ADJ_PRODUCTS[tag]['label']}" if tag else MODEL


def adj_tag_of(name):
    """Adjusted product suffix a folder or prediction name ends with ('' for raw); the longest suffix first."""
    return next((t for t in sorted(ADJ_PRODUCTS, key=len, reverse=True) if name.endswith(t)), '')


def tagged(pred):
    """Prediction folder of the current product: ensmean -> ensmean-adjusted while the adjusted one runs."""
    return pred + PRED_TAG if PRED_TAG and not pred.endswith(PRED_TAG) else pred


def prev_path(pred, kind, ext="png"):
    """An output already written (same naming as out_path, pred as given, not registered)."""
    d = os.path.join(OUTDIR, pred) if PRED_SUBFOLDERS else OUTDIR
    return os.path.join(d, f"{PREFIX}_{pred}_{kind}.{ext}")
PRED_SUBFOLDERS = True      # True: OUTDIR/<pred>/ folders; False: everything flat in OUTDIR (names are unique either way)

# file names, tried in order ({m} = member 1..50 or ensmean); a template without {m} is one file holding all members
# names with the model first, so a post-processed product (swe-adjust-hkl.py writes ECXENS_<BT>_swe-500-nd-<tag>-<m>.grib)
# is found when the tag is given as the model, also when the plain members are in the same folder
# ECXENS adjusted to HKL (adjust-snow-ecxens.py, joined as T120000 so SmartMet keeps it apart, split by prep-ecxens-validation.sh),
# m = ensmean gives the ens mean file ECXENS_<base>T120000_swe-adjusted-ensmean.grib
ADJ_TEMPLATES = ['ECXENS_{BT12}_swe-adjusted-{{m}}.grib', 'ECXENS_{BT18}_swe-adjusted-{{m}}.grib']
ADJ_MEAN_TEMPLATES = []
# adjusted products: prediction suffix -> files (adjust-snow-ecxens.py, split by prep-ecxens-validation.sh), names in titles
# and the comparison folder
ADJ_PRODUCTS = {
    '-adjusted': dict(templates=ADJ_TEMPLATES, label='adjusted', name='ECXENS adjusted', cmp='adjusted-vs-raw'),
    '-adjusted-era5l': dict(templates=['ECXENS_{BT18}_swe-adjusted-era5l-{{m}}.grib', 'ECXENS_{BT12}_swe-adjusted-era5l-{{m}}.grib'],
                            label='adjusted ERA5-Land', name='ECXENS adjusted ERA5-Land', cmp='adjusted-era5l-vs-raw'),
}
CMP_DIRS = {v['cmp'] for v in ADJ_PRODUCTS.values()}
ECX_TEMPLATES = [
                 'ECXENS_{BT}_swe-500-nd-{MODEL}-{{m}}.grib', 'ECXENS_{BT}_swe-500-nd_{MODEL}-{{m}}.grib',
                 'ECXENS_{BT}_swe-{{m}}.grib',
                 'ECXENS_{BT}_swe-500-nd.grib',
                 'ECXENS_{BT}_swe-500-nd-{{m}}.grib']
ECENS_TEMPLATES = ['ECENS_{BT}_sd-{{m}}-fix.grib',
                   'ECENS_{BT}_sd.grib',
                   'ECENS_{BT}_era5l_sfc+sde-nd-{{m}}-sd.grib', 'ECENS_{BT}_sd-{{m}}.grib']
# ens mean files whose name does not follow the member naming (e.g. members sd-1-fix.grib, mean sd-ensmean.grib),
# looked for in the members' directory when the template with m = ensmean has no file
ECX_MEAN_TEMPLATES = ['ECXENS_{BT}_swe-ensmean.grib']
ECENS_MEAN_TEMPLATES = ['ECENS_{BT}_sd-ensmean.grib', 'ECENS_{BT}_sd-ensmean-fix.grib']
# raw ECENS ens mean remapped to the SNOWCAP 500 m grid (prep-ecxens-validation.sh, cdo remap); looked for in ECENS_DIRS
ECENS500_TEMPLATES = ['ECENS_{BT}_sd-500m-ensmean.grib']
# CLMS SWE (get-clms-swe.sh): raw 5 km Nordic cut and the remapdis to the SNOWCAP 500 m grid, mm, one file per day
CLMS = True
CLMS_DIRS = [ECX_DIR, SNOW_ROOT + 'CLMS/', SNOW_ROOT]
CLMS_RAW_TEMPLATE = 'CLMS_20000101T000000_{D}T000000_swe_scandi.grib'
CLMS_500_TEMPLATE = 'CLMS_20000101T000000_{D}T000000_swe-500m_scandi.grib'
CLMS_SCALE = 1.0           # get-clms-swe.sh with MULC=1 writes mm
CLMS_GRID = '5 km'
ECENS_GRID = '9 km'      # raw ECENS grid, shown in the titles (the resolution in degrees is added from the file)
FINE_GRID = '500 m'      # ECXENS / SNOWCAP grid, shown in the titles
# daily maps of member01, driest and wettest (the ens mean gets every map)
MEMBER_KINDS = {'ecxens-swe', 'ecens-swe', 'ecxens-bias'}
SAVE_ECENS01_BIAS = False   # daily raw ECENS minus SNOWCAP map on the raw ECENS cells (the 500 m remapped one replaces it)
SNOWCAP_TEMPLATES = ['SNOWCAP_{Y}_SWE.grib', 'SNOWCAP_20150101T000000_{Y}_SWE.grib',
                     'HKL_{Y}_SWE.grib', 'HKL_20150101T000000_{Y}_SWE.grib']   # old HKL file names as fallback
SNOWCAP_MONTH_TEMPLATES = ['SNOWCAP_{YM}_swe.grib', 'SNOWCAP_{YM}_SWE.grib', 'HKL_{YM}_swe.grib', 'HKL_{YM}_SWE.grib']
FC_DAYS = 15        # forecast length in days (months of SNOWCAP to look for)
REGION = 'Scandinavia'
# sub-regions for the time series and scores: key -> (shown name, country (Natural Earth ADM0_A3), latitude from, latitude to)
# the latitude limits are rough lines between the parts of each country; edit freely
REGIONS = {
    'fin_lapland': ('Finnish Lapland', 'FIN', 66.0, 90.0),
    'fin_central': ('Central Finland', 'FIN', 62.5, 66.0),
    'fin_south':   ('Southern Finland', 'FIN', 0.0, 62.5),
    'swe_lapland': ('Swedish Lapland', 'SWE', 65.5, 90.0),
    'swe_central': ('Central Sweden', 'SWE', 60.5, 65.5),
    'swe_south':   ('Southern Sweden', 'SWE', 0.0, 60.5),
    'nor_north':   ('Northern Norway (Lapland)', 'NOR', 65.0, 90.0),
    'nor_central': ('Central Norway', 'NOR', 62.5, 65.0),
    'nor_south':   ('Southern Norway', 'NOR', 0.0, 62.5),
}
REGION_RESULTS = True      # regional time series and scores (needs cartopy for the country borders)
REGION_MASK_STEP = 10      # country test on every 10th 500 m pixel (5 km), copied to the pixels around it: fast, borders within 5 km

FIRST_MEMBER = 1           # the single member validated next to the ens mean (lowest available member if missing)
MEMBERS = range(0, 51)     # members looked for (0 = control, used only if the file has it)
ADD_EXTREME_MEMBERS = True # validate the driest and the wettest member too
RANK_BY = "mean"           # mean: area weighted domain mean SWE over all lead days; max; p99; mean_last_day
RERANK = REDO              # False reuses the ranking csv if it exists (also redone when the ECXENS files changed)
MEMBER_TIMESERIES = True   # domain mean SWE of every member against SNOWCAP
RETIMESERIES = REDO        # False reuses the time series csv if it exists (also redone when the ECXENS files changed)
ECENS_SCALE = 1000.0       # ECMWF sd is m w.e., x1000 gives mm
ECX_SCALE = 1.0            # ECXENS is mm w.e.
SNOWCAP_SCALE = 1000.0         # Scandinavian SNOWCAP SWE is m w.e., always x1000
MASK_ECENS_TO_SNOWCAP = False  # True hides ECENS cells where SNOWCAP has no data
ECENS_OWN_EXTENT = True    # combined figure only: ECENS panel shows its whole grid
PANEL_ECENS_VIEW = "ecxens"  # single ECENS map area: "ecxens" lines up with the ECXENS / SNOWCAP maps, "own" = whole ECENS grid
MIN_COVER = 0.5            # 0.1 degree comparison: share of valid 500 m pixels a cell needs
SAVE_PANELS = True         # ECENS, ECXENS and SNOWCAP SWE maps each in their own png
SAVE_COMBINED = False      # old three panel figure (ecens / ecxens / snowcap side by side), off: every figure separately
PANEL_HEIGHT = 8.0
SNOW_THRESHOLDS = [1, 10, 50]
GRID_TOL = 0.1
DPI = 150
BG_COLOR = "#d9d9d9"

# WMS legend (mm)
SWE_BOUNDS = [0, 1, 10, 25, 50, 75, 100, 150, 200, 300, 400, 600, 800, 1000, 1500]
SWE_COLORS = ["#ffffff", "#e8eef4", "#d4e4f0", "#bcd6ec", "#a2c6e6", "#86b4de", "#6aa0d4", "#4f88c6",
              "#3a6fb6", "#2a56a0", "#1f4088", "#142c6c", "#b48cd2", "#9040b0"]
SWE_UNDER = "#808080"
SWE_OVER = "#7a1060"

# Bias legend (mm), two tier
MILD_BIAS = 50
BIAS_BOUNDS = [-500, -300, -200, -100, -50, -25, -10, -1, 1, 10, 25, 50, 100, 200, 300, 500]
BIAS_COLORS = ["#06164d", "#0c2a8a", "#1747c9", "#2b7bff",      # strong, too little snow
               "#3fae49", "#8fd18a", "#d4efce",                  # mild, too little snow
               "#ffffff",                                        # no bias
               "#fff5c4", "#ffe270", "#f5b800",                  # mild, too much snow
               "#ff4d2e", "#d7191c", "#99000d", "#5a0010"]       # strong, too much snow
BIAS_UNDER = "#6a00ff"
BIAS_OVER = "#ff1493"
assert len(BIAS_COLORS) == len(BIAS_BOUNDS) - 1 and MILD_BIAS in BIAS_BOUNDS and -MILD_BIAS in BIAS_BOUNDS

BIAS_COMPARE = "abs"   # "abs": |ECXENS bias| minus |ECENS bias|; "diff": ECXENS bias minus ECENS bias
GAIN_LABEL = ("|ECXENS bias| minus |ECENS bias| (mm), negative = ECXENS closer to SNOWCAP" if BIAS_COMPARE == "abs"
              else "ECXENS bias minus ECENS bias (mm)")

# ---------------------------------------------------------------- colour maps
swe_cmap = ListedColormap(SWE_COLORS)
swe_cmap.set_under(SWE_UNDER)
swe_cmap.set_over(SWE_OVER)
swe_cmap.set_bad(alpha=0.0)
swe_norm = BoundaryNorm(SWE_BOUNDS, swe_cmap.N)

bias_cmap = ListedColormap(BIAS_COLORS)
bias_cmap.set_under(BIAS_UNDER)
bias_cmap.set_over(BIAS_OVER)
bias_cmap.set_bad(alpha=0.0)
bias_norm = BoundaryNorm(BIAS_BOUNDS, bias_cmap.N)
BIAS_LABEL = f"SWE bias (mm), soft colours |bias| ≤ {MILD_BIAS}, strong colours beyond"

rmse_bounds = [0, 1, 10, 25, 50, 75, 100, 150, 200, 300, 500]
rmse_cmap = ListedColormap(plt.get_cmap("magma_r", len(rmse_bounds) - 1)(np.arange(len(rmse_bounds) - 1)))
rmse_cmap.set_over("black")
rmse_cmap.set_bad(alpha=0.0)
rmse_norm = BoundaryNorm(rmse_bounds, rmse_cmap.N)


# ---------------------------------------------------------------- output names
MANIFEST = {}

def out_path(pred, kind, day=None, ext="png"):
    """<OUTDIR>/<pred>/<MODEL>_<BASE>_<pred>_<kind>[_<YYYYMMDD>].<ext>, registered in the manifest.
    pred gets '-adjusted' while the adjusted product runs."""
    pred = tagged(pred)
    d = os.path.join(OUTDIR, pred) if PRED_SUBFOLDERS else OUTDIR
    os.makedirs(d, exist_ok=True)
    name = f"{PREFIX}_{pred}_{kind}" + (f"_{day:%Y%m%d}" if day is not None else "") + f".{ext}"
    p = os.path.join(d, name)
    MANIFEST[p] = {"model": MODEL, "base": BASE_TAG, "pred": pred, "kind": kind,
                   "valid_date": f"{day:%Y%m%d}" if day is not None else "",
                   "lead_day": (day - BASE_TIME).days if day is not None else "",
                   "ext": ext, "file": os.path.relpath(p, OUT_ROOT)}
    return p


def todo(path):
    """False when the png exists and may be kept (SKIP_PNG): the figure is not drawn again."""
    return not (SKIP_PNG and os.path.exists(path))


def write_manifest():
    rows = [r for p, r in MANIFEST.items() if os.path.exists(p)]
    if not rows:
        return
    p = os.path.join(OUTDIR, f"{PREFIX}_manifest.csv")
    new = pd.DataFrame(rows)
    if os.path.exists(p):   # keep the rows of the product not run this time (RUN_RAW or RUN_ADJUSTED off)
        old = pd.read_csv(p, dtype=str)
        old = old[~old["file"].isin(new["file"]) & old["file"].map(lambda f: os.path.exists(os.path.join(OUT_ROOT, f)))]
        new = pd.concat([old, new.astype(str)], ignore_index=True)
    new.sort_values(["pred", "kind", "valid_date"]).to_csv(p, index=False)
    print(f"Manifest of {len(rows)} files -> {p}")


# ---------------------------------------------------------------- io
def _open_parts(path):
    """Open the main variable of a GRIB file lazily. Returns a list of DataArrays: one normally, two when cfgrib
    cannot put the control (cf) and perturbed (pf) members into one dataset."""
    kw = {"indexpath": ""}
    try:
        dss = [xr.open_dataset(path, engine="cfgrib", backend_kwargs=kw)]
    except Exception as e:
        print(f"{os.path.basename(path)}: not one dataset for cfgrib ({e.__class__.__name__}), opening cf and pf separately")
        dss = []
        for dt in ("cf", "pf"):
            try:
                dss.append(xr.open_dataset(path, engine="cfgrib", backend_kwargs={**kw, "filter_by_keys": {"dataType": dt}}))
            except Exception:
                pass
        if not dss:
            raise
    parts = []
    for ds in dss:
        if not ds.data_vars:
            continue
        name = "sd" if "sd" in ds.data_vars else list(ds.data_vars)[0]
        parts.append(ds[name])
    if not parts:
        raise ValueError(f"{os.path.basename(path)}: cfgrib finds no field (grid description wrong? see the ECCODES errors above)")
    return parts


def _numbers(da):
    if "number" in da.dims:
        return [int(n) for n in da["number"].values]
    if "number" in da.coords:
        return [int(da["number"].values)]
    return []


def normalize(da, label):
    """DataArray of one member to (valid_time, latitude, longitude), lat ascending, float32."""
    if "number" in da.dims:
        if da.sizes["number"] != 1:
            raise ValueError(f"{label}: several members left in the field, select one first")
        da = da.isel(number=0)
    if "step" in da.dims and "time" in da.dims:
        # messages with different reference time / step pairs: cfgrib builds a time x step grid with holes;
        # flatten to valid_time, keep the fields with data, the first one for a valid time seen twice
        da = da.transpose("time", "step", "latitude", "longitude")
        vt = np.asarray(da["valid_time"].values).reshape(-1)
        vals = da.values.reshape(-1, da.sizes["latitude"], da.sizes["longitude"])
        ok = ~pd.isnull(vt) & np.isfinite(vals).any(axis=(1, 2))
        vt, vals = vt[ok], vals[ok]
        uniq, first = np.unique(vt, return_index=True)
        print(f"{label}: time x step grid flattened to {len(uniq)} valid times")
        da = xr.DataArray(vals[first], dims=("valid_time", "latitude", "longitude"), name=da.name,
                          coords={"valid_time": uniq, "latitude": da["latitude"].values, "longitude": da["longitude"].values})
    elif "step" in da.dims:
        da = da.swap_dims({"step": "valid_time"})
    elif "time" in da.dims:
        if "valid_time" in da.coords and da["valid_time"].dims == ("time",):
            da = da.swap_dims({"time": "valid_time"})
        else:
            da = da.rename({"time": "valid_time"})
    else:
        da = da.expand_dims(valid_time=[pd.Timestamp(da["valid_time"].values)])
    keep = {"valid_time", "latitude", "longitude"}
    da = da.drop_vars([c for c in da.coords if c not in keep])
    da = da.sortby("latitude")
    return da.astype("float32").rename("swe")


def load_field(path, scale=1.0):
    """Single field file (SNOWCAP, or one member per file) -> DataArray in mm."""
    parts = _open_parts(path)
    if len(parts) > 1 or len(_numbers(parts[0])) > 1:
        raise ValueError(f"{path}: file holds several members, expected one")
    return normalize(parts[0], os.path.basename(path)) * np.float32(scale)


class Source:
    """ECXENS or ECENS forecast. Split layout: one file per member ({m} in the path), ensmean file optional.
    Single file layout: every member in one file ('number' dimension), the ens mean is computed from the members;
    a single file without members is taken as the ens mean."""

    def __init__(self, kind, path, scale, mean_templates=()):
        self.kind, self.path, self.scale = kind, path, scale
        self.split = "{m}" in path
        self.mean_path = None
        if self.split:
            self.numbers = [m for m in MEMBERS if os.path.exists(path.format(m=m))]
            d = os.path.dirname(path) + "/"
            for p in [path.format(m="ensmean")] + [d + t.format(**_fill) for t in mean_templates]:
                if os.path.exists(p):
                    self.mean_path = p
                    break
            self.has_mean_file = self.mean_path is not None
            self.parts = None
        else:
            self.parts = _open_parts(path)
            self.numbers = sorted({n for p in self.parts for n in _numbers(p) if n in MEMBERS})
            self.has_mean_file = all(len(_numbers(p)) == 0 for p in self.parts)
        print(f"{kind}: {os.path.basename(path)}, {len(self.numbers)} members"
              + (f" ({self.numbers[0]} to {self.numbers[-1]})" if self.numbers else "")
              + (f", ens mean {os.path.basename(self.mean_path)}" if self.mean_path
                 else ", ens mean in file" if self.has_mean_file else ", ens mean computed from the members"))

    def members(self):
        return list(self.numbers)

    def mtime(self):
        """Newest modification time of the files: a changed forecast makes its results be done again."""
        if self.split:
            ps = [self.path.format(m=m) for m in self.numbers] + ([self.mean_path] if self.mean_path else [])
        else:
            ps = [self.path]
        return max((os.path.getmtime(p) for p in ps if os.path.exists(p)), default=0.0)

    def load(self, m):
        if self.split:
            if m == "ensmean":
                if not self.mean_path:
                    raise ValueError(f"{self.kind}: no ens mean file, it is computed in scan_members")
                return load_field(self.mean_path, self.scale)
            return load_field(self.path.format(m=m), self.scale)
        if m == "ensmean":
            if not self.has_mean_file:
                raise ValueError(f"{self.kind}: no ens mean in {self.path}, it is computed in scan_members")
            return normalize(self.parts[0], f"{self.kind} ensmean") * np.float32(self.scale)
        for p in self.parts:
            if m in _numbers(p):
                da = p.sel(number=m) if "number" in p.dims else p
                return normalize(da, f"{self.kind} member {m}") * np.float32(self.scale)
        raise KeyError(f"{self.kind}: member {m} not in {self.path}")


def find_source(kind, dirs, templates, scale, mean_templates=()):
    """First directory and naming that has the file (single file layout) or the member 1 / ensmean file (split)."""
    tried = []
    for d in dirs:
        for t in templates:
            p = d + t.format(**_fill)
            if "{m}" in p:
                probes = [p.format(m=m) for m in (1, FIRST_MEMBER)] + [p.format(m="ensmean")]
            else:
                probes = [p]
            tried.append(os.path.basename(probes[0]))
            if any(os.path.exists(q) for q in probes):
                return Source(kind, p, scale, mean_templates)
    print(f"WARNING: no {kind} file found in {', '.join(dirs)}, tried {', '.join(sorted(set(tried)))}")
    return None


def pick_snowcap():
    """SNOWCAP files for the forecast (list, may be empty): a file holding the first forecast day in its name,
    else the monthly files of every month the forecast reaches, else the whole year files."""
    day1 = f'{BASE_TIME + pd.Timedelta(days=1):%Y%m%d}'
    # the base month too: the prep script used to name the file by it, also when the days run into the next month
    months = sorted({f'{BASE_TIME:%Y%m}'} | {f'{BASE_TIME + pd.Timedelta(days=d):%Y%m}' for d in range(1, FC_DAYS + 1)})
    for d in SNOWCAP_DIRS:
        hits = sorted(glob.glob(f'{d}SNOWCAP_*{day1}*_SWE.grib') + glob.glob(f'{d}HKL_*{day1}*_SWE.grib'))
        if hits:
            return hits[:1]
        monthly = []
        for ym in months:
            p = next((d + t.format(YM=ym) for t in SNOWCAP_MONTH_TEMPLATES if os.path.exists(d + t.format(YM=ym))), None)
            if p:
                monthly.append(p)
        if monthly:
            return monthly
        for t in SNOWCAP_TEMPLATES:
            p = d + t.format(**_fill)
            if os.path.exists(p):
                return [p]
    print(f"WARNING: no SNOWCAP file for {BASE_TIME:%Y-%m-%d} in {', '.join(SNOWCAP_DIRS)}: forecast maps only, no scores")
    return []


def load_snowcap():
    files = pick_snowcap()
    if not files:
        return None, []
    parts = []
    for p in files:
        print(f"Loading SNOWCAP: {p}")
        parts.append(load_field(p, 1.0))
    snowcap = xr.concat(parts, dim="valid_time") if len(parts) > 1 else parts[0]
    snowcap = snowcap.isel(valid_time=np.unique(snowcap.valid_time.values, return_index=True)[1]).sortby("valid_time")
    snowcap = snowcap * np.float32(SNOWCAP_SCALE)
    hmax = float(snowcap.max())
    print(f"SNOWCAP SWE after x{SNOWCAP_SCALE:g}: max {hmax:.1f} mm, mean {float(snowcap.mean()):.1f} mm, "
          f"{pd.Timestamp(snowcap.valid_time.values[0]):%Y-%m-%d} to {pd.Timestamp(snowcap.valid_time.values[-1]):%Y-%m-%d}")
    if hmax < 20:
        print("WARNING: SNOWCAP SWE max below 20 mm: early season, or still m w.e. (check SNOWCAP_SCALE)")
    want = pd.date_range(BASE_TIME + pd.Timedelta(days=1), periods=FC_DAYS, freq="D")
    have = pd.DatetimeIndex(snowcap.valid_time.values).normalize()
    miss = [d for d in want if d not in have]
    if miss:
        print(f"WARNING: SNOWCAP has {FC_DAYS - len(miss)} of the {FC_DAYS} forecast days, missing "
              f"{', '.join(f'{d:%m-%d}' for d in miss[:10])}{' ...' if len(miss) > 10 else ''} (wrong file for this start date?)")
    return snowcap, files


def clms_path(template, day):
    name = template.format(D=f"{pd.Timestamp(day):%Y%m%d}")
    return next((d + name for d in CLMS_DIRS if os.path.exists(d + name)), None)


def clms_available():
    """True when CLMS 500 m files exist for some forecast days."""
    if not CLMS:
        return False
    days = [BASE_TIME + pd.Timedelta(days=d) for d in range(1, FC_DAYS + 1)]
    n = sum(clms_path(CLMS_500_TEMPLATE, d) is not None for d in days)
    print(f"CLMS SWE: {n} of {FC_DAYS} forecast days with the 500 m file"
          + ("" if n else f" (looked for {CLMS_500_TEMPLATE.format(D='<day>')} in {', '.join(CLMS_DIRS)})"))
    return n > 0


def load_clms(template, day, label):
    """One day of CLMS SWE (mm, 2D DataArray), None when there is no file; negative flags and fill values masked."""
    p = clms_path(template, day)
    if p is None:
        return None
    try:
        da = load_field(p, CLMS_SCALE).isel(valid_time=0, drop=True)
    except Exception as ex:
        print(f"WARNING: {label} {os.path.basename(p)} not read: {ex}")
        return None
    return da.where((da >= 0) & (da < 5000))


def check_ecens(ens, snowcap, label="ECENS", mask=True):
    emax = float(ens.max())
    print(f"{label} SWE after x{ECENS_SCALE:g}: max {emax:.1f} mm, mean {float(ens.mean()):.1f} mm")
    if emax < 20:
        print(f"WARNING: {label} SWE max below 20 mm, the values look like m w.e., check ECENS_SCALE")
    if emax > 20000:
        print(f"WARNING: {label} SWE max above 20000 mm, the file may already be in mm, check ECENS_SCALE")
    if mask and MASK_ECENS_TO_SNOWCAP and snowcap is not None:
        ens = mask_to_domain(ens, snowcap)
    return ens


def mask_to_domain(da, ref):
    ok = np.isfinite(ref.isel(valid_time=0)).astype("float32").drop_vars("valid_time", errors="ignore")
    ok = ok.interp(latitude=da.latitude, longitude=da.longitude, method="nearest")
    return da.where(ok > 0.5)


def grid_extent(da):
    lat = da.latitude.values
    lon = da.longitude.values
    dlat = abs(lat[1] - lat[0])
    dlon = abs(lon[1] - lon[0])
    return [lon.min() - dlon / 2, lon.max() + dlon / 2, lat.min() - dlat / 2, lat.max() + dlat / 2]


def match_grid(ecx, da, label):
    """da on the ECXENS grid (as is when the grids are the same, else nearest neighbour)."""
    same_size = (ecx.sizes["latitude"] == da.sizes["latitude"] and ecx.sizes["longitude"] == da.sizes["longitude"])
    if same_size:
        off_lat = float(np.abs(ecx.latitude.values - da.latitude.values).max())
        off_lon = float(np.abs(ecx.longitude.values - da.longitude.values).max())
        cell = min(abs(float(ecx.latitude[1] - ecx.latitude[0])), abs(float(ecx.longitude[1] - ecx.longitude[0])))
    if same_size and max(off_lat, off_lon) < GRID_TOL * cell:
        print(f"{label}: same grid as ECXENS ({max(off_lat, off_lon) / cell:.4f} cells off at most), used without interpolation")
        return da.assign_coords(latitude=ecx.latitude, longitude=ecx.longitude)
    print(f"Grids differ: ECXENS {ecx.sizes['latitude']}x{ecx.sizes['longitude']}, "
          f"{label} {da.sizes['latitude']}x{da.sizes['longitude']}, interpolating {label} to the ECXENS grid (nearest)")
    return da.interp_like(ecx, method="nearest")


def match_snowcap(ecx, snowcap_raw):
    return match_grid(ecx, snowcap_raw, "SNOWCAP")


def coarse_index(lat, lon, clat, clon):
    """Flat index of the coarse cell every fine pixel falls in (-1 outside), plus fine pixels per coarse cell."""
    cdlat = abs(clat[1] - clat[0])
    cdlon = abs(clon[1] - clon[0])
    iy = np.floor((lat - (clat.min() - cdlat / 2)) / cdlat).astype(np.int64)
    ix = np.floor((lon - (clon.min() - cdlon / 2)) / cdlon).astype(np.int64)
    oky = (iy >= 0) & (iy < clat.size)
    okx = (ix >= 0) & (ix < clon.size)
    idx = np.where(oky[:, None] & okx[None, :], iy[:, None] * clon.size + ix[None, :], -1).astype(np.int32)
    n_all = np.bincount(idx[idx >= 0], minlength=clat.size * clon.size)
    return idx, n_all


def to_coarse(field, idx, n_all, shape):
    """Mean of the valid fine pixels per coarse cell; NaN where fewer than MIN_COVER of its pixels are valid."""
    v = np.isfinite(field) & (idx >= 0)
    ssum = np.bincount(idx[v], weights=field[v].astype("float64"), minlength=n_all.size)
    cnt = np.bincount(idx[v], minlength=n_all.size)
    with np.errstate(invalid="ignore", divide="ignore"):
        m = np.where((n_all > 0) & (cnt >= MIN_COVER * n_all), ssum / np.maximum(cnt, 1), np.nan)
    return m.reshape(shape)


# ---------------------------------------------------------------- regions
_CPATHS = None
_REG = {}

def _country_paths():
    """Outlines of the countries in REGIONS as matplotlib paths, from Natural Earth (cartopy); {} when not available."""
    global _CPATHS
    if _CPATHS is not None:
        return _CPATHS
    _CPATHS = {}
    try:
        import cartopy.io.shapereader as shpreader
        from matplotlib.path import Path
        codes = {c for _, c, _, _ in REGIONS.values()}
        shp = shpreader.natural_earth(resolution="50m", category="cultural", name="admin_0_countries")
        for rec in shpreader.Reader(shp).records():
            a = rec.attributes
            code = a.get("ADM0_A3") or a.get("adm0_a3")
            if code in codes:
                _CPATHS[code] = [Path(np.asarray(g.exterior.coords)[:, :2]) for g in getattr(rec.geometry, "geoms", [rec.geometry])]
        print(f"country borders for the regions: {', '.join(sorted(_CPATHS))}")
    except Exception as e:
        print(f"WARNING: country borders not available ({e.__class__.__name__}: {e}), no regional results")
        _CPATHS = {}
    return _CPATHS


def region_index(lat, lon, step=1):
    """Flat pixel indices (int32) of every region in REGIONS on a lat/lon grid, {} without country borders.
    The country test runs on every step-th pixel and is copied to the pixels around it; the latitude limits are exact."""
    key = (lat.size, lon.size, float(lat[0]), float(lon[0]), step)
    if key in _REG:
        return _REG[key]
    paths = _country_paths() if REGION_RESULTS else {}
    out = {}
    if paths:
        slat, slon = lat[::step], lon[::step]
        LON, LAT = np.meshgrid(slon, slat)
        pts = np.column_stack([LON.ravel(), LAT.ravel()])
        rows = (np.arange(lat.size) // step)[:, None]
        cols = (np.arange(lon.size) // step)[None, :]
        country = {}
        for code, plist in paths.items():
            m = np.zeros(len(pts), bool)
            for pth in plist:
                x0, y0 = pth.vertices.min(axis=0)
                x1, y1 = pth.vertices.max(axis=0)
                inb = (pts[:, 0] >= x0) & (pts[:, 0] <= x1) & (pts[:, 1] >= y0) & (pts[:, 1] <= y1)
                if inb.any():
                    m[inb] |= pth.contains_points(pts[inb])
            country[code] = m.reshape(slat.size, slon.size)
        for k, (name, code, la0, la1) in REGIONS.items():
            if code not in country or not country[code].any():
                continue
            band = ((lat >= la0) & (lat < la1))[:, None]
            idx = np.flatnonzero(country[code][rows, cols] & band).astype(np.int32)
            if idx.size:
                out[k] = idx
        print(f"regions on the {lat.size}x{lon.size} grid: " + ", ".join(f"{REGIONS[k][0]} {v.size}" for k, v in out.items()))
    _REG[key] = out
    return out


def rmean(field, idx, w_rows, ncol, extra=None):
    """Area weighted mean of a 2D field over the pixels idx (finite values, and extra (2D bool) where given);
    w_rows: weight per grid row."""
    v = field.ravel()[idx]
    ok = np.isfinite(v)
    if extra is not None:
        ok &= extra.ravel()[idx]
    if not ok.any():
        return np.nan
    w = w_rows[idx[ok] // ncol]
    return float(np.sum(v[ok] * w) / np.sum(w))


# ---------------------------------------------------------------- coastlines (optional)
_LINES = {}

def get_lines(extent):
    key = tuple(np.round(extent, 4))
    if key in _LINES:
        return _LINES[key]
    try:
        import cartopy.feature as cfeature
        from shapely.geometry import box
    except ImportError:
        print("cartopy/shapely not available, plotting without coastlines")
        _LINES[key] = []
        return []
    clip = box(extent[0], extent[2], extent[1], extent[3])
    lines = []
    for feat in (cfeature.NaturalEarthFeature("physical", "coastline", "50m"),
                 cfeature.NaturalEarthFeature("cultural", "admin_0_boundary_lines_land", "50m")):
        try:
            for geom in feat.geometries():
                g = geom.intersection(clip)
                if g.is_empty:
                    continue
                for p in getattr(g, "geoms", [g]):
                    if hasattr(p, "coords"):
                        xy = np.asarray(p.coords)
                        if len(xy) > 1:
                            lines.append(xy)
        except Exception as e:
            print(f"Could not load Natural Earth lines: {e}")
            break
    _LINES[key] = lines
    return lines


# ---------------------------------------------------------------- scores
def scores(s, o):
    s = s.astype("float64")
    o = o.astype("float64")
    n = s.size
    if n < 2:
        return dict(n=n, bias=np.nan, mae=np.nan, rmse=np.nan, r=np.nan, kge=np.nan)
    d = s - o
    r = np.corrcoef(s, o)[0, 1] if s.std() > 0 and o.std() > 0 else np.nan
    alpha = s.std() / o.std() if o.std() > 0 else np.nan
    beta = s.mean() / o.mean() if o.mean() != 0 else np.nan
    kge = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    return dict(n=n, bias=d.mean(), mae=np.abs(d).mean(), rmse=np.sqrt((d ** 2).mean()), r=r, kge=kge)


def contingency(s, o, thr):
    fs = s >= thr
    fo = o >= thr
    a = float(np.sum(fs & fo))
    b = float(np.sum(fs & ~fo))
    c = float(np.sum(~fs & fo))
    d = float(np.sum(~fs & ~fo))
    pod = a / (a + c) if a + c else np.nan
    far = b / (a + b) if a + b else np.nan
    csi = a / (a + b + c) if a + b + c else np.nan
    den = (a + c) * (c + d) + (a + b) * (b + d)
    hss = 2 * (a * d - b * c) / den if den else np.nan
    return pod, far, csi, hss


# ---------------------------------------------------------------- plotting
def draw_map(ax, data, cmap, norm, extent, aspect, lines, title, view=None):
    view = view or extent
    ax.set_facecolor(BG_COLOR)
    im = ax.imshow(np.ma.masked_invalid(data), origin="lower", extent=extent,
                   cmap=cmap, norm=norm, interpolation="nearest")
    for xy in lines:
        ax.plot(xy[:, 0], xy[:, 1], color="k", lw=0.4)
    ax.set_xlim(view[0], view[1])
    ax.set_ylim(view[2], view[3])
    ax.set_aspect(aspect)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    return im


def bias_colorbar(fig, im, ax, shrink=0.8, label=BIAS_LABEL):
    cb = fig.colorbar(im, ax=ax, extend="both", ticks=BIAS_BOUNDS, spacing="uniform", shrink=shrink)
    cb.set_label(label)
    for v in (-MILD_BIAS, MILD_BIAS):
        try:
            cb.ax.axhline(v, color="k", lw=2.0)
        except Exception:
            pass
    return cb


def save_bias_map(path, data, extent, aspect, lines, title, view=None, label=BIAS_LABEL):
    if not todo(path):
        return
    fig, ax = plt.subplots(figsize=(7, 8), constrained_layout=True)
    im = draw_map(ax, data, bias_cmap, bias_norm, extent, aspect, lines, title, view=view)
    bias_colorbar(fig, im, ax, label=label)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def save_rmse_map(path, data, extent, aspect, lines, title):
    if not todo(path):
        return
    fig, ax = plt.subplots(figsize=(7, 8), constrained_layout=True)
    im = draw_map(ax, data, rmse_cmap, rmse_norm, extent, aspect, lines, title)
    fig.colorbar(im, ax=ax, extend="max", ticks=rmse_bounds, spacing="uniform", shrink=0.8, label="mm")
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def save_panel(path, data, ext, aspect, lines, title, view):
    """One SWE map alone, no colour bar: maps with the same view get the same figure size, so they line up."""
    if not todo(path):
        return
    w = PANEL_HEIGHT * (view[1] - view[0]) / ((view[3] - view[2]) * aspect)
    fig, ax = plt.subplots(figsize=(max(w, 3.0), PANEL_HEIGHT), constrained_layout=True)
    draw_map(ax, data, swe_cmap, swe_norm, ext, aspect, lines, title, view=view)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def save_swe_legend(path):
    if not todo(path):
        return
    fig, ax = plt.subplots(figsize=(12, 1.1), constrained_layout=True)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=swe_norm, cmap=swe_cmap), cax=ax, orientation="horizontal",
                      extend="both", ticks=SWE_BOUNDS, spacing="uniform")
    cb.set_label("SWE (mm)")
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def lead_plot(path, title, series, ylabel=None, ylim=None, zero=False, legend_kw=None):
    """One score against lead day; series = list of (x, y, style, label)."""
    if not todo(path):
        return
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for x, y, style, lab in series:
        ax.plot(x, y, style, label=lab)
    if zero:
        ax.axhline(0, color="k", lw=0.6)
    if ylim:
        ax.set_ylim(*ylim)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.set_xlabel("Lead day (from base time)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(**(legend_kw or {}))
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------- one validation run
def wmean(v, ok, w):
    """Area weighted mean of v over the pixels ok (w = weights of the same shape)."""
    return float(np.sum(v[ok] * w[ok]) / np.sum(w[ok])) if ok.any() else np.nan


def plot_ensmean_timeseries(tsd, pred, name, ens_name):
    """Domain mean SWE per day: ECXENS, ECENS remapped to 500 m and SNOWCAP, over the same pixels."""
    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    for col, lab, sty in [("ecxens", f"{name} ({MODEL_LABEL}, {FINE_GRID})", dict(color="tab:blue", lw=2, marker="o")),
                          ("ecens500", f"{ens_name} remapped to the SNOWCAP grid ({FINE_GRID})",
                           dict(color="tab:orange", lw=2, ls="--", marker="s")),
                          ("snowcap", f"SNOWCAP ({FINE_GRID})", dict(color="red", lw=2.5, marker="o")),
                          ("clms", "CLMS SWE (satellite, remapped to 500 m), CLMS pixels only",
                           dict(color="tab:purple", lw=2, ls="-.", marker="^")),
                          ]:
        if col in tsd and tsd[col].notna().any():
            d = tsd.dropna(subset=[col])
            ax.plot(d["date"], d[col], label=lab, **sty)
    dates = sorted(tsd["date"].unique())
    ax.set_xticks(dates)
    ax.set_xticklabels([f"{pd.Timestamp(t):%m-%d}\n+{(pd.Timestamp(t) - BASE_TIME).days} d" for t in dates], fontsize=8)
    ax.set_xlabel("date and lead day")
    ax.set_ylabel("domain mean SWE (mm)")
    ax.set_title(f"Domain mean SWE, SNOWCAP domain, same pixels for every line"
                 + (" (CLMS lines: CLMS pixels only)" if "clms" in tsd and tsd["clms"].notna().any() else "")
                 + f"\nbase {BASE_TIME:%Y-%m-%d} ({MODEL_LABEL})")
    ax.grid(alpha=0.3)
    ax.legend()
    p = out_path(pred, "domain-mean-timeseries")
    fig.savefig(p, dpi=DPI)
    plt.close(fig)
    tsd.to_csv(out_path(pred, "domain-mean-timeseries", ext="csv"), index=False, float_format="%.4f")
    print(f"domain mean time series -> {p}")


def region_grid(keys):
    """Subplot grid for the regions, 3 columns."""
    ncol = 3
    nrow = -(-len(keys) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 3.6 * nrow), constrained_layout=True, sharex=True, squeeze=False)
    for ax in axes.ravel()[len(keys):]:
        ax.set_visible(False)
    return fig, axes.ravel()


def date_ticks(ax, dates):
    ax.set_xticks(dates[::2])
    ax.set_xticklabels([f"{pd.Timestamp(t):%m-%d}\n+{(pd.Timestamp(t) - BASE_TIME).days} d" for t in dates[::2]], fontsize=7)


def plot_region_timeseries(tsd, pred, name, ens_name):
    """Domain mean SWE per day in every region: ECXENS, ECENS remapped and SNOWCAP over the same pixels."""
    keys = [k for k in REGIONS if k in set(tsd["region"])]
    if not keys:
        return
    fig, axes = region_grid(keys)
    dates = sorted(tsd["date"].unique())
    for ax, k in zip(axes, keys):
        d = tsd[tsd["region"] == k]
        for col, lab, sty in [("ecxens", f"{name} ({MODEL_LABEL})", dict(color="tab:blue", lw=1.8, marker="o", ms=3)),
                              ("ecens500", f"{ens_name} remapped", dict(color="tab:orange", lw=1.8, ls="--", marker="s", ms=3)),
                              ("snowcap", "SNOWCAP", dict(color="red", lw=2.2, marker="o", ms=3)),
                              ("clms", "CLMS (CLMS pixels)", dict(color="tab:purple", lw=1.6, ls="-.", marker="^", ms=3))]:
            if col in d and d[col].notna().any():
                ax.plot(d["date"], d[col], label=lab, **sty)
        ax.set_title(REGIONS[k][0], fontsize=10)
        ax.grid(alpha=0.3)
        date_ticks(ax, dates)
    axes[0].set_ylabel("domain mean SWE (mm)")
    axes[0].legend(fontsize=7)
    fig.suptitle(f"Domain mean SWE per region ({FINE_GRID}, same pixels for every line), base {BASE_TIME:%Y-%m-%d} ({MODEL_LABEL})")
    p = out_path(pred, "domain-mean-timeseries-regions")
    fig.savefig(p, dpi=DPI)
    plt.close(fig)
    tsd.assign(region_name=tsd["region"].map(lambda k: REGIONS[k][0])).to_csv(
        out_path(pred, "domain-mean-timeseries-regions", ext="csv"), index=False, float_format="%.4f")
    print(f"regional domain mean time series -> {p}")


def plot_region_scores(rdf, pred, name, ens_name):
    """Daily scores per region to csv, and the means over the lead days as bars: ECXENS against ECENS remapped."""
    rdf.assign(region_name=rdf["region"].map(lambda k: REGIONS[k][0])).to_csv(
        out_path(pred, "regions-metrics", ext="csv"), index=False, float_format="%.4f")
    keys = [k for k in REGIONS if k in set(rdf["region"])]
    m = rdf.groupby("region").mean(numeric_only=True).reindex(keys)
    has5 = "ecens500_rmse" in m and m["ecens500_rmse"].notna().any()
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), constrained_layout=True)
    x = np.arange(len(keys))
    wbar = 0.38 if has5 else 0.7
    names = [REGIONS[k][0] for k in keys]
    for ax, (sc, title) in zip(axes[:2], [("rmse", "RMSE (mm)"), ("bias", "bias, forecast minus SNOWCAP (mm)")]):
        ax.bar(x - (wbar / 2 if has5 else 0), m[sc], wbar, label=f"{name} ({MODEL_LABEL})", color="tab:blue")
        if has5:
            ax.bar(x + wbar / 2, m[f"ecens500_{sc}"], wbar, label=f"{ens_name} remapped", color="tab:orange")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
        ax.set_title(title)
        ax.axhline(0, color="k", lw=0.6)
        ax.grid(alpha=0.3, axis="y")
    axes[0].legend(fontsize=8)
    ax = axes[2]
    if "skill_mse_vs_ecens500" in m and m["skill_mse_vs_ecens500"].notna().any():
        cols = ["tab:green" if v > 0 else "tab:red" for v in m["skill_mse_vs_ecens500"]]
        ax.bar(x, m["skill_mse_vs_ecens500"], 0.7, color=cols)
        ax.set_title("MSE skill of ECXENS vs ECENS remapped\n(above 0 = ECXENS closer to SNOWCAP)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
    ax.axhline(0, color="k", lw=0.6)
    ax.grid(alpha=0.3, axis="y")
    fig.suptitle(f"Scores per region, means over the lead days, SWE {FINE_GRID}, base {BASE_TIME:%Y-%m-%d} ({MODEL_LABEL})")
    p = out_path(pred, "regions-score-summary")
    fig.savefig(p, dpi=DPI)
    plt.close(fig)
    print(f"regional scores -> {p}")
    cols = [c for c in ["rmse", "ecens500_rmse", "bias", "ecens500_bias", "skill_mse_vs_ecens500"] if c in m]
    print(m[cols].rename(index=lambda k: REGIONS[k][0]).round(2).to_string())


def validate(ecx, snowcap_raw, pred, name, ens=None, ens_name=None, ens500=None, full=True, clms=False):
    """Validation of one prediction. Forecast maps for every forecast day, bias maps and scores on the days
    SNOWCAP has data. ens: raw ECENS of the same member (its own grid); ens500: ECENS remapped to the SNOWCAP
    500 m grid (ens mean only). full: every map (ens mean); False: only the maps in MEMBER_KINDS (member01,
    driest, wettest). clms: CLMS SWE maps and time series lines (raw ECXENS ens mean).
    Returns the daily scores DataFrame (None without SNOWCAP days)."""
    def want(kind):
        return full or kind in MEMBER_KINDS

    with open(out_path(pred, "info", ext="json"), "w") as f:
        json.dump({"model": MODEL, "base": BASE_TAG, "pred": tagged(pred), "name": name, "adjusted": bool(PRED_TAG), "product": PRED_TAG.lstrip("-") or "raw",
                   "ecens_name": ens_name if ens is not None else None, "ecens500": ens500 is not None}, f)
    if SAVE_PANELS:
        save_swe_legend(out_path(pred, "swe-legend"))

    snowcap = match_snowcap(ecx, snowcap_raw) if snowcap_raw is not None else None
    snowcap_times = snowcap.valid_time.values if snowcap is not None else np.array([], dtype="datetime64[ns]")
    e5da = match_grid(ecx, ens500, "ECENS remapped") if ens500 is not None else None
    e5_times = e5da.valid_time.values if e5da is not None else np.array([], dtype="datetime64[ns]")
    times = ecx.valid_time.values
    print(f"{name}: {len(times)} forecast days, {int(np.isin(times, snowcap_times).sum())} with SNOWCAP"
          + (f", {int(np.isin(times, e5_times).sum())} with ECENS remapped to {FINE_GRID}" if e5da is not None else ""))

    lat = ecx.latitude.values
    lon = ecx.longitude.values
    dlat = abs(lat[1] - lat[0])
    dlon = abs(lon[1] - lon[0])
    extent = grid_extent(ecx)
    aspect = 1.0 / np.cos(np.deg2rad(lat.mean()))
    lines = get_lines(extent)
    tag = f"{MODEL_LABEL}, base {BASE_TIME:%Y-%m-%d}"
    if ens is not None:
        ens_extent = grid_extent(ens)
        ens_lines = get_lines(ens_extent) if ECENS_OWN_EXTENT else lines
        ens_res = f"{abs(float(ens.latitude[1] - ens.latitude[0])):g}°"
        ens_grid = f"raw ECENS grid, {ECENS_GRID}, {ens_res}"
        ens_view = ens_extent if ECENS_OWN_EXTENT else extent
        ens_times = ens.valid_time.values
        elat, elon = ens.latitude.values, ens.longitude.values
        cidx, cn_all = coarse_index(lat, lon, elat, elon)
        cshape = (elat.size, elon.size)
        sum_eb = np.zeros(cshape)
        sum_ae = np.zeros(cshape)
        sum_ax = np.zeros(cshape)
        sum_xb = np.zeros(cshape)
        cnt_c = np.zeros(cshape)
        if (cn_all > 0).any():
            print(f"{name}: {int((cn_all > 0).sum())} {ens_res} ECENS cells covered by ECXENS, "
                  f"median {int(np.median(cn_all[cn_all > 0]))} ECXENS pixels per cell")

    R = 6371000.0
    area = (R ** 2 * np.deg2rad(dlat) * np.deg2rad(dlon) * np.cos(np.deg2rad(lat)))[:, None].astype("float64")
    wgt = np.broadcast_to(area, (lat.size, lon.size))
    sum_d = np.zeros((lat.size, lon.size))
    sum_d2 = np.zeros_like(sum_d)
    cnt = np.zeros_like(sum_d)
    sum_d5 = np.zeros_like(sum_d)      # ECENS remapped minus SNOWCAP
    sum_d25 = np.zeros_like(sum_d)
    sum_g5 = np.zeros_like(sum_d)      # |ECXENS bias| minus |ECENS remapped bias|
    cnt5 = np.zeros_like(sum_d)
    dom = np.isfinite(snowcap.isel(valid_time=0).values) if snowcap is not None else None
    ridx = region_index(lat, lon, REGION_MASK_STEP) if full else {}   # regions: ens mean only
    area_rows = area[:, 0]
    reg_ts, reg_rows = [], []
    edges = np.linspace(0, 1500, 301)
    H = np.zeros((edges.size - 1, edges.size - 1))
    rows, ts_rows = [], []

    for t in times:
        ts = pd.Timestamp(t)
        lead = (ts - BASE_TIME).days
        s = ecx.sel(valid_time=t).values
        o = snowcap.sel(valid_time=t).values if np.isin(t, snowcap_times) else None
        e = ens.sel(valid_time=t).values if (ens is not None and np.isin(t, ens_times)) else None
        e5 = e5da.sel(valid_time=t).values if np.isin(t, e5_times) else None
        when = f"{ts:%Y-%m-%d} (lead {lead} d)"
        c5 = None
        if clms:
            c5da = load_clms(CLMS_500_TEMPLATE, ts, "CLMS 500 m")
            if c5da is not None:
                c5 = match_grid(ecx.isel(valid_time=0), c5da, "CLMS 500 m").values
            p_raw = out_path(pred, "clms-swe", ts)
            if todo(p_raw):
                craw = load_clms(CLMS_RAW_TEMPLATE, ts, f"CLMS {CLMS_GRID}")
                if craw is not None:
                    save_panel(p_raw, craw.values, grid_extent(craw), aspect, lines,
                               f"CLMS SWE (satellite, raw grid, {CLMS_GRID})\n{ts:%Y-%m-%d}", extent)
            if c5 is not None:
                save_panel(out_path(pred, "clms500-swe", ts), c5, extent, aspect, lines,
                           f"CLMS SWE (remapped to the SNOWCAP grid, {FINE_GRID})\n{ts:%Y-%m-%d}", extent)

        # ---- SWE maps, each producer in its own file, grid in the title
        if SAVE_PANELS:
            if e is not None and want("ecens-swe"):
                pv = extent if PANEL_ECENS_VIEW == "ecxens" else ens_extent
                save_panel(out_path(pred, "ecens-swe", ts), e, ens_extent, aspect,
                           lines if PANEL_ECENS_VIEW == "ecxens" else ens_lines,
                           f"{ens_name} SWE ({ens_grid})\n{when}", pv)
            if e5 is not None and want("ecens500-swe"):
                save_panel(out_path(pred, "ecens500-swe", ts), e5, extent, aspect, lines,
                           f"{ens_name} SWE (remapped to the SNOWCAP grid, {FINE_GRID})\n{when}", extent)
            if want("ecxens-swe"):
                save_panel(out_path(pred, "ecxens-swe", ts), s, extent, aspect, lines,
                           f"{name} SWE ({MODEL_LABEL}, {FINE_GRID})\n{when}", extent)
            if o is not None and want("snowcap-swe"):
                save_panel(out_path(pred, "snowcap-swe", ts), o, extent, aspect, lines,
                           f"SNOWCAP SWE ({FINE_GRID})\n{ts:%Y-%m-%d}", extent)

        if SAVE_COMBINED:
            panels = []
            if e is not None:
                panels.append((e, ens_extent, ens_view, ens_lines, f"{ens_name} SWE ({ens_grid}) {ts:%Y-%m-%d}"))
            panels.append((s, extent, extent, lines, f"{name} SWE ({FINE_GRID}) {when}"))
            if o is not None:
                panels.append((o, extent, extent, lines, f"SNOWCAP SWE ({FINE_GRID}) {ts:%Y-%m-%d}"))
            fig, axes = plt.subplots(1, len(panels), figsize=(5.7 * len(panels) + 0.6, 8), constrained_layout=True, squeeze=False)
            for ax, (data, ext, view, ln, title) in zip(axes[0], panels):
                im = draw_map(ax, data, swe_cmap, swe_norm, ext, aspect, ln, title, view=view)
            fig.colorbar(im, ax=axes[0], extend="both", ticks=SWE_BOUNDS, spacing="uniform", shrink=0.8, label="SWE (mm)")
            fig.savefig(out_path(pred, "compare-swe", ts), dpi=DPI)
            plt.close(fig)

        # ---- domain mean SWE of ECXENS, ECENS remapped and SNOWCAP over the same pixels (ens mean only)
        if full:
            ok = np.isfinite(s) & (dom if dom is not None else True)
            if e5 is not None:
                ok &= np.isfinite(e5)
            if o is not None:
                ok &= np.isfinite(o)
            # CLMS leaves out mountains, glaciers and water: its line and SNOWCAP over the same pixels (snowcap_clms)
            oc = ok & np.isfinite(c5) if c5 is not None else None
            crow = ({"clms": wmean(c5, oc, wgt), "snowcap_clms": wmean(o, oc, wgt) if o is not None else np.nan}
                    if c5 is not None else ({"clms": np.nan, "snowcap_clms": np.nan} if clms else {}))
            ts_rows.append({"date": ts, "lead_day": lead, "ecxens": wmean(s, ok, wgt),
                            "ecens500": wmean(e5, ok, wgt) if e5 is not None else np.nan,
                            "snowcap": wmean(o, ok, wgt) if o is not None else np.nan, **crow})
            for k, idx in ridx.items():
                rc = ({"clms": rmean(c5, idx, area_rows, lon.size, oc),
                       "snowcap_clms": rmean(o, idx, area_rows, lon.size, oc) if o is not None else np.nan}
                      if c5 is not None else ({"clms": np.nan, "snowcap_clms": np.nan} if clms else {}))
                reg_ts.append({"date": ts, "lead_day": lead, "region": k,
                               "ecxens": rmean(s, idx, area_rows, lon.size, ok),
                               "ecens500": rmean(e5, idx, area_rows, lon.size, ok) if e5 is not None else np.nan,
                               "snowcap": rmean(o, idx, area_rows, lon.size, ok) if o is not None else np.nan, **rc})

        if o is None:
            print(f"{name} {ts:%Y-%m-%d} lead {lead:2d}  no SNOWCAP, maps only  ECX max {np.nanmax(s):.1f}")
            continue

        valid = np.isfinite(s) & np.isfinite(o)
        d = np.where(valid, s - o, np.nan)

        # ---- ECXENS minus SNOWCAP
        if want("ecxens-bias"):
            save_bias_map(out_path(pred, "ecxens-bias", ts), d, extent, aspect, lines,
                          f"{name} minus SNOWCAP SWE ({MODEL_LABEL}, {FINE_GRID})\n{when}")

        # ---- ECENS remapped to 500 m minus SNOWCAP
        valid5 = None
        if e5 is not None:
            valid5 = valid & np.isfinite(e5)
            d5 = np.where(valid5, e5 - o, np.nan)
            if want("ecens500-bias"):
                save_bias_map(out_path(pred, "ecens500-bias", ts), d5, extent, aspect, lines,
                              f"{ens_name} (remapped, {FINE_GRID}) minus SNOWCAP SWE\n{when}")
            dv5 = np.nan_to_num(d5)
            sum_d5 += dv5
            sum_d25 += dv5 ** 2
            sum_g5 += np.where(valid5, np.abs(np.nan_to_num(d)) - np.abs(dv5), 0.0)
            cnt5 += valid5

        # ---- raw ECENS and ECXENS against SNOWCAP on the raw ECENS cells (bias-bias map)
        okc = None
        if e is not None:
            hc = to_coarse(np.where(valid, o, np.nan), cidx, cn_all, cshape)
            xc = to_coarse(np.where(valid, s, np.nan), cidx, cn_all, cshape)
            eb = e - hc
            xb = xc - hc
            gain = np.abs(xb) - np.abs(eb) if BIAS_COMPARE == "abs" else xb - eb
            okc = np.isfinite(eb) & np.isfinite(xb)
            eb = np.where(okc, eb, np.nan)
            gain = np.where(okc, gain, np.nan)
            sum_eb += np.where(okc, eb, 0.0)
            sum_ae += np.where(okc, np.abs(eb), 0.0)
            sum_ax += np.where(okc, np.abs(xb), 0.0)
            sum_xb += np.where(okc, xb, 0.0)
            cnt_c += okc
            if SAVE_ECENS01_BIAS and full:
                save_bias_map(out_path(pred, "ecens-bias", ts), eb, ens_extent, aspect, lines,
                              f"{ens_name} minus SNOWCAP SWE ({ens_grid})\n{when}", view=extent)
            if want("ecxens-vs-ecens-gain"):
                gtitle = f"|{name} bias| minus |{ens_name} bias|" if BIAS_COMPARE == "abs" else f"{name} bias minus {ens_name} bias"
                save_bias_map(out_path(pred, "ecxens-vs-ecens-gain", ts), gain, ens_extent, aspect, lines,
                              f"{gtitle}\nSWE on the {ens_grid} cells, {when}", view=extent, label=GAIN_LABEL)

        # ---- accumulate
        dv = np.nan_to_num(d)
        sum_d += dv
        sum_d2 += dv ** 2
        cnt += valid
        sv, ov = s[valid], o[valid]
        H += np.histogram2d(ov, sv, bins=[edges, edges])[0]

        # ---- scores
        row = {"date": ts.date(), "lead_day": lead}
        row.update(scores(sv, ov))
        snow = (sv >= 1) | (ov >= 1)
        for k, v in scores(sv[snow], ov[snow]).items():
            row[f"snow_{k}"] = v
        for thr in SNOW_THRESHOLDS:
            pod, far, csi, hss = contingency(sv, ov, thr)
            row.update({f"pod_{thr}": pod, f"far_{thr}": far, f"csi_{thr}": csi, f"hss_{thr}": hss})
        a2 = wgt[valid]
        row["vol_ecx_km3"] = np.sum(sv.astype("float64") / 1000.0 * a2) / 1e9
        row["vol_snowcap_km3"] = np.sum(ov.astype("float64") / 1000.0 * a2) / 1e9
        if valid5 is not None and valid5.sum() > 1:
            sv5, ov5, ev5 = s[valid5].astype("float64"), o[valid5].astype("float64"), e5[valid5].astype("float64")
            for k, v in scores(ev5, ov5).items():
                row[f"ecens500_{k}"] = v
            snow5 = (ev5 >= 1) | (ov5 >= 1)
            for k, v in scores(ev5[snow5], ov5[snow5]).items():
                row[f"ecens500_snow_{k}"] = v
            for thr in SNOW_THRESHOLDS:
                pod, far, csi, hss = contingency(ev5, ov5, thr)
                row.update({f"ecens500_pod_{thr}": pod, f"ecens500_far_{thr}": far, f"ecens500_csi_{thr}": csi,
                            f"ecens500_hss_{thr}": hss})
            row["vol_ecens500_km3"] = np.sum(ev5 / 1000.0 * wgt[valid5]) / 1e9
            mse5 = float(np.mean((ev5 - ov5) ** 2))
            row["skill_mse_vs_ecens500"] = 1 - float(np.mean((sv5 - ov5) ** 2)) / mse5 if mse5 > 0 else np.nan
            de = np.abs(sv5 - ov5) - np.abs(ev5 - ov5)
            nz = de != 0
            row["frac_px_improved500"] = float(np.mean(de[nz] < 0)) if nz.any() else np.nan   # pixels where the errors differ
        if okc is not None and okc.sum() > 1:
            for pre, fld in (("ecens", e), ("ecx01", xc)):
                for k, v in scores(fld[okc], hc[okc]).items():
                    if k != "n":
                        row[f"{pre}_{k}"] = v
            row["n_cells01"] = int(okc.sum())
            mse_e = float(np.mean((e[okc] - hc[okc]) ** 2))
            row["skill_mse_vs_ecens"] = 1 - float(np.mean((xc[okc] - hc[okc]) ** 2)) / mse_e if mse_e > 0 else np.nan
            row["frac_cells_improved"] = float(np.mean(gain[okc] < 0))
        rows.append(row)
        for k, idx in ridx.items():
            sr, orr = s.ravel()[idx], o.ravel()[idx]
            vr = np.isfinite(sr) & np.isfinite(orr)
            if vr.sum() < 2:
                continue
            rr = {"date": ts.date(), "lead_day": lead, "region": k}
            rr.update(scores(sr[vr], orr[vr]))
            if e5 is not None:
                er = e5.ravel()[idx]
                v5 = vr & np.isfinite(er)
                if v5.sum() > 1:
                    for kk, vv in scores(er[v5], orr[v5]).items():
                        rr[f"ecens500_{kk}"] = vv
                    m5 = float(np.mean((er[v5].astype("float64") - orr[v5]) ** 2))
                    rr["skill_mse_vs_ecens500"] = 1 - float(np.mean((sr[v5].astype("float64") - orr[v5]) ** 2)) / m5 if m5 > 0 else np.nan
            reg_rows.append(rr)
        e5_txt = (f"  | 500m RMSE ECENS {row['ecens500_rmse']:.1f} skill {row['skill_mse_vs_ecens500']:+.2f}"
                  if "ecens500_rmse" in row else "")
        ens_txt = (f"  | 0.1deg RMSE ECENS {row['ecens_rmse']:.1f} ECXENS {row['ecx01_rmse']:.1f} skill {row['skill_mse_vs_ecens']:+.2f}"
                   if "ecens_rmse" in row else "")
        print(f"{name} {ts:%Y-%m-%d} lead {lead:2d}  bias {row['bias']:7.2f}  rmse {row['rmse']:7.2f}  "
              f"kge {row['kge']:5.2f}  csi_1 {row['csi_1']:.3f}  ECX max {np.nanmax(s):.1f}  SNOWCAP max {np.nanmax(o):.1f}{e5_txt}{ens_txt}")

    if full and ts_rows:
        plot_ensmean_timeseries(pd.DataFrame(ts_rows), pred, name, ens_name or "ECENS")
    if reg_ts:
        plot_region_timeseries(pd.DataFrame(reg_ts), pred, name, ens_name or "ECENS")
    if reg_rows:
        plot_region_scores(pd.DataFrame(reg_rows), pred, name, ens_name or "ECENS")

    if not rows:
        print(f"{name}: no days with SNOWCAP, forecast maps only")
        return None

    df = pd.DataFrame(rows)
    df.to_csv(out_path(pred, "ecxens-metrics", ext="csv"), index=False, float_format="%.4f")
    obs_times = [pd.Timestamp(r) for r in df["date"]]
    period = f"{obs_times[0]:%Y-%m-%d} to {obs_times[-1]:%Y-%m-%d}"

    # ---- mean bias and RMSE maps over the period, separately
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_bias = np.where(cnt > 0, sum_d / cnt, np.nan)
        rmse_map = np.where(cnt > 0, np.sqrt(sum_d2 / cnt), np.nan)
    save_bias_map(out_path(pred, "ecxens-mean-bias"), mean_bias, extent, aspect, lines,
                  f"Mean bias {name} minus SNOWCAP SWE ({MODEL_LABEL}, {FINE_GRID})\n{period}", label="mm")
    save_rmse_map(out_path(pred, "ecxens-rmse"), rmse_map, extent, aspect, lines,
                  f"RMSE {name} SWE ({MODEL_LABEL}, {FINE_GRID})\n{period}")

    # ---- the same for ECENS remapped to 500 m, and the MAE change ECXENS against it
    if cnt5.any():
        with np.errstate(invalid="ignore", divide="ignore"):
            mean_b5 = np.where(cnt5 > 0, sum_d5 / cnt5, np.nan)
            rmse5 = np.where(cnt5 > 0, np.sqrt(sum_d25 / cnt5), np.nan)
            gain5 = np.where(cnt5 > 0, sum_g5 / cnt5, np.nan)
        save_bias_map(out_path(pred, "ecens500-mean-bias"), mean_b5, extent, aspect, lines,
                      f"Mean bias {ens_name} (remapped, {FINE_GRID}) minus SNOWCAP SWE\n{period}", label="mm")
        save_rmse_map(out_path(pred, "ecens500-rmse"), rmse5, extent, aspect, lines,
                      f"RMSE {ens_name} SWE (remapped, {FINE_GRID})\n{period}")
        save_bias_map(out_path(pred, "ecxens-vs-ecens500-mae-gain"), gain5, extent, aspect, lines,
                      f"MAE {name} minus MAE {ens_name} (remapped), SWE {FINE_GRID}\n{period}",
                      label="mm, negative = ECXENS closer to SNOWCAP")

    # ---- mean raw ECENS bias and MAE change over the period, on the raw ECENS cells
    if ens is not None and cnt_c.any():
        with np.errstate(invalid="ignore", divide="ignore"):
            mean_eb = np.where(cnt_c > 0, sum_eb / cnt_c, np.nan)
            if BIAS_COMPARE == "abs":
                mae_gain = np.where(cnt_c > 0, (sum_ax - sum_ae) / cnt_c, np.nan)
            else:
                mae_gain = np.where(cnt_c > 0, (sum_xb - sum_eb) / cnt_c, np.nan)
        save_bias_map(out_path(pred, "ecens-mean-bias"), mean_eb, ens_extent, aspect, lines,
                      f"Mean bias {ens_name} minus SNOWCAP SWE ({ens_grid})\n{period}", view=extent, label="mm")
        mtitle = f"MAE {name} minus MAE {ens_name}" if BIAS_COMPARE == "abs" else f"Mean bias {name} minus mean bias {ens_name}"
        save_bias_map(out_path(pred, "ecxens-vs-ecens-mae-gain"), mae_gain, ens_extent, aspect, lines,
                      f"{mtitle}\nSWE on the {ens_grid} cells, {period}", view=extent,
                      label="mm, negative = ECXENS closer to SNOWCAP" if BIAS_COMPARE == "abs" else "mm")

    # ---- density scatter
    fig, ax = plt.subplots(figsize=(7, 6.5), constrained_layout=True)
    Hm = np.ma.masked_equal(H.T, 0)
    if Hm.count():
        pc = ax.pcolormesh(edges, edges, Hm, norm=LogNorm(vmin=1, vmax=Hm.max()), cmap="viridis")
        fig.colorbar(pc, ax=ax, label="pixel count")
    ax.plot([0, 1500], [0, 1500], "k--", lw=0.8)
    ax.set_xlabel(f"SNOWCAP SWE (mm, {FINE_GRID})")
    ax.set_ylabel(f"{name} SWE (mm, {FINE_GRID})")
    ax.set_title(f"Pixel density, all days pooled ({MODEL_LABEL})\n{period} (0 to 1500 mm)")
    ax.set_aspect("equal")
    fig.savefig(out_path(pred, "ecxens-density-scatter"), dpi=DPI)
    plt.close(fig)

    # ---- scores against lead day, one figure each; ECENS remapped to 500 m on the same plots when it is there
    L = df["lead_day"]
    has5 = "ecens500_rmse" in df and df["ecens500_rmse"].notna().any()
    e5n = f"{ens_name} remapped"
    head = f"{name} vs SNOWCAP SWE ({FINE_GRID}), {tag}"
    lead_plot(out_path(pred, "ecxens-score-bias-mae"), f"Bias and MAE\n{head}",
              [(L, df["bias"], "o-", "ECXENS bias all"), (L, df["snow_bias"], "o--", "ECXENS bias snow px"),
               (L, df["mae"], "s-", "ECXENS MAE all")]
              + ([(L, df["ecens500_bias"], "^:", f"{e5n} bias all"), (L, df["ecens500_mae"], "v:", f"{e5n} MAE all")] if has5 else []),
              ylabel="mm", zero=True)
    lead_plot(out_path(pred, "ecxens-score-rmse"), f"RMSE\n{head}",
              [(L, df["rmse"], "o-", "ECXENS all"), (L, df["snow_rmse"], "o--", "ECXENS snow px")]
              + ([(L, df["ecens500_rmse"], "^:", f"{e5n} all"), (L, df["ecens500_snow_rmse"], "v:", f"{e5n} snow px")] if has5 else []),
              ylabel="mm")
    lead_plot(out_path(pred, "ecxens-score-kge"), f"Correlation and KGE\n{head}",
              [(L, df["r"], "o-", "ECXENS r"), (L, df["kge"], "s-", "ECXENS KGE"), (L, df["snow_kge"], "s--", "ECXENS KGE snow px")]
              + ([(L, df["ecens500_r"], "^:", f"{e5n} r"), (L, df["ecens500_kge"], "v:", f"{e5n} KGE")] if has5 else []))
    lead_plot(out_path(pred, "ecxens-score-csi"), f"Critical success index\n{head}",
              [(L, df[f"csi_{thr}"], "o-", f"ECXENS CSI {thr} mm") for thr in SNOW_THRESHOLDS]
              + ([(L, df[f"ecens500_csi_{thr}"], "^:", f"{e5n} CSI {thr} mm") for thr in SNOW_THRESHOLDS] if has5 else []),
              ylim=(0, 1), legend_kw=dict(fontsize=8, ncol=2))
    lead_plot(out_path(pred, "ecxens-score-pod-far"), f"POD and FAR\n{head}",
              [s for thr in SNOW_THRESHOLDS for s in ((L, df[f"pod_{thr}"], "o-", f"POD {thr} mm"),
                                                      (L, df[f"far_{thr}"], "x--", f"FAR {thr} mm"))],
              ylim=(0, 1), legend_kw=dict(fontsize=8, ncol=2))
    lead_plot(out_path(pred, "ecxens-score-volume"), f"Total SWE volume (common valid pixels, {FINE_GRID})\n{head}",
              [(L, df["vol_ecx_km3"], "o-", name), (L, df["vol_snowcap_km3"], "s-", "SNOWCAP")]
              + ([(L, df["vol_ecens500_km3"], "^:", f"{e5n} (its own valid pixels)")] if has5 else []), ylabel="km3 water")
    if has5:
        lead_plot(out_path(pred, "ecxens-vs-ecens500-skill"), f"{name} against {e5n}, SWE {FINE_GRID}\n{tag}",
                  [(L, df["skill_mse_vs_ecens500"], "o-", "MSE skill vs ECENS remapped (1 = perfect, 0 = as ECENS)"),
                   (L, df["frac_px_improved500"], "s--", "share of pixels closer to SNOWCAP (where the errors differ)")], zero=True)
    if "skill_mse_vs_ecens" in df:
        lead_plot(out_path(pred, "ecxens-vs-ecens-skill"), f"{name} against {ens_name}, SWE on the {ens_grid} cells\n{tag}",
                  [(L, df["skill_mse_vs_ecens"], "o-", "MSE skill vs ECENS (1 = perfect, 0 = as ECENS)"),
                   (L, df["frac_cells_improved"], "s--", "share of cells closer to SNOWCAP")], zero=True)

    print(f"{name} done, outputs in {os.path.dirname(out_path(pred, 'info', ext='json'))}/")
    return df


# ---------------------------------------------------------------- one pass over all members
def scan_members(ecx_src, ens_src, snowcap_raw, want_ts, want_ecx_mean, want_ens_mean):
    """Reads every member once: ranking statistics, domain mean time series and the ens mean sums.
    Returns (ranking DataFrame, time series DataFrame or None, ECXENS ens mean or None, ECENS ens mean or None)."""
    ecx_m = set(ecx_src.members())
    ens_m = set(ens_src.members()) if ens_src is not None else set()
    todo = sorted(ecx_m | (ens_m if want_ens_mean else set()))
    rank_rows, ts_rows = [], []
    acc = {}
    snowcap = snowcap_mask = ci = cell_static = None
    rx = re_ = None   # regions on the ECXENS and the ECENS grid
    snowcap_times = np.array([], dtype="datetime64[ns]")
    hx_done, he_done = set(), set()

    def accumulate(key, da):
        v = da.values
        if key not in acc:
            acc[key] = dict(sum=np.zeros(v.shape, "float32"), n=np.zeros(v.shape, "uint8"),
                            coords=dict(valid_time=da.valid_time.values, latitude=da.latitude.values, longitude=da.longitude.values))
        a = acc[key]
        if v.shape != a["sum"].shape or not np.array_equal(da.valid_time.values, a["coords"]["valid_time"]):
            print(f"WARNING: {key} member has other times or grid, left out of the ens mean")
            return
        ok = np.isfinite(v)
        a["sum"] += np.where(ok, v, 0.0)
        a["n"] += ok.astype("uint8")
        a["members"] = a.get("members", 0) + 1

    for m in todo:
        ecx = ecx_src.load(m) if m in ecx_m else None
        ens = ens_src.load(m) if (m in ens_m and (want_ens_mean or want_ts)) else None
        if ecx is not None:
            w = np.cos(np.deg2rad(ecx.latitude))
            v = ecx.values[np.isfinite(ecx.values)]
            r = {"member": m, "mean": float(ecx.weighted(w).mean()),
                 "mean_last_day": float(ecx.isel(valid_time=-1).weighted(w).mean()),
                 "p99": float(np.percentile(v, 99)) if v.size else np.nan, "max": float(v.max()) if v.size else np.nan}
            rank_rows.append(r)
            del v
            if want_ecx_mean:
                accumulate("ECXENS", ecx)
        if ens is not None and want_ens_mean:
            accumulate("ECENS", ens)

        if want_ts and ecx is not None:
            lat, lon = ecx.latitude.values, ecx.longitude.values
            wx = np.broadcast_to(np.cos(np.deg2rad(lat))[:, None], (lat.size, lon.size))
            if snowcap_raw is not None and snowcap is None:
                snowcap = match_snowcap(ecx, snowcap_raw)
                snowcap_times = snowcap.valid_time.values
                snowcap_mask = np.isfinite(snowcap.isel(valid_time=0).values)
            if ens is not None and ci is None:
                elat, elon = ens.latitude.values, ens.longitude.values
                cidx, cn_all = coarse_index(lat, lon, elat, elon)
                cshape = (elat.size, elon.size)
                we = np.broadcast_to(np.cos(np.deg2rad(elat))[:, None], cshape)
                ci = (cidx, cn_all, cshape)
                if snowcap_mask is not None:
                    cell_static = np.isfinite(to_coarse(np.where(snowcap_mask, 1.0, np.nan), *ci))
                re_ = region_index(elat, elon, 1)
                we_rows = np.cos(np.deg2rad(elat))
            if rx is None:
                rx = region_index(lat, lon, REGION_MASK_STEP)
                wx_rows = np.cos(np.deg2rad(lat))
            for t in ecx.valid_time.values:
                ts = pd.Timestamp(t)
                s = ecx.sel(valid_time=t).values
                ok = np.isfinite(s) & (snowcap_mask if snowcap_mask is not None else True)
                ts_rows.append({"date": ts, "source": "ECXENS", "member": m, "region": "all", "swe_mm": wmean(s, ok, wx)})
                for k, idx in rx.items():
                    ts_rows.append({"date": ts, "source": "ECXENS", "member": m, "region": k,
                                    "swe_mm": rmean(s, idx, wx_rows, lon.size, snowcap_mask)})
                o = snowcap.sel(valid_time=t).values if np.isin(t, snowcap_times) else None
                if o is not None and ts not in hx_done:
                    ts_rows.append({"date": ts, "source": "SNOWCAP_ECXENS", "member": -1, "region": "all",
                                    "swe_mm": wmean(o, ok & np.isfinite(o), wx)})
                    for k, idx in rx.items():
                        ts_rows.append({"date": ts, "source": "SNOWCAP_ECXENS", "member": -1, "region": k,
                                        "swe_mm": rmean(o, idx, wx_rows, lon.size, snowcap_mask)})
                    hx_done.add(ts)
                if ens is not None and np.isin(t, ens.valid_time.values):
                    e = ens.sel(valid_time=t).values
                    cell_ok = cell_static if cell_static is not None else np.isfinite(to_coarse(s, *ci))
                    okc = np.isfinite(e) & cell_ok
                    ts_rows.append({"date": ts, "source": "ECENS", "member": m, "region": "all", "swe_mm": wmean(e, okc, we)})
                    for k, idx in (re_ or {}).items():
                        ts_rows.append({"date": ts, "source": "ECENS", "member": m, "region": k,
                                        "swe_mm": rmean(e, idx, we_rows, e.shape[1], cell_ok)})
                    if o is not None and ts not in he_done:
                        hc = to_coarse(o, *ci)
                        ts_rows.append({"date": ts, "source": "SNOWCAP_ECENS", "member": -1, "region": "all",
                                        "swe_mm": wmean(hc, okc & np.isfinite(hc), we)})
                        for k, idx in (re_ or {}).items():
                            ts_rows.append({"date": ts, "source": "SNOWCAP_ECENS", "member": -1, "region": k,
                                            "swe_mm": rmean(hc, idx, we_rows, hc.shape[1], cell_ok)})
                        he_done.add(ts)
        txt = f"member {m:2d}"
        if ecx is not None:
            txt += f"  mean {r['mean']:7.2f}  last day mean {r['mean_last_day']:7.2f}  p99 {r['p99']:7.1f}  max {r['max']:7.1f} mm"
        if ens is None and m in ecx_m and ens_src is not None:
            txt += "  (no ECENS member)"
        print(txt)
        del ecx, ens

    means = {}
    for key, a in acc.items():
        with np.errstate(invalid="ignore", divide="ignore"):
            mv = np.where(a["n"] > 0, a["sum"] / np.maximum(a["n"], 1), np.nan).astype("float32")
        means[key] = xr.DataArray(mv, dims=("valid_time", "latitude", "longitude"), coords=a["coords"], name="swe")
        print(f"{key} ens mean computed from {a.get('members', 0)} members")
    rk = pd.DataFrame(rank_rows) if rank_rows else None
    tsd = pd.DataFrame(ts_rows) if ts_rows else None
    return rk, tsd, means.get("ECXENS"), means.get("ECENS")


def plot_member_timeseries(df):
    """One plot per producer: members thin blue, ens mean of the members black, SNOWCAP red; whole domain, then per region."""
    if "region" not in df:
        df = df.assign(region="all")
    plot_member_timeseries_regions(df)
    df = df[df["region"] == "all"]
    for src, hsrc, kind, what in [("ECXENS", "SNOWCAP_ECXENS", "ecxens-timeseries", "ECXENS (500 m, SNOWCAP domain)"),
                                  ("ECENS", "SNOWCAP_ECENS", "ecens-timeseries", "raw ECENS (0.1° cells, SNOWCAP averaged into the cells)")]:
        d = df[df["source"] == src]
        if d.empty:
            print(f"no {src} members for the time series plot")
            continue
        fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
        for i, (m, g) in enumerate(d.groupby("member")):
            g = g.sort_values("date")
            ax.plot(g["date"], g["swe_mm"], color="tab:blue", lw=0.8, alpha=0.6,
                    label=f"{src} members ({d['member'].nunique()})" if i == 0 else None)
        mean = d.groupby("date")["swe_mm"].mean()
        ax.plot(mean.index, mean.values, color="k", lw=1.8, label=f"{src} member mean")
        h = df[df["source"] == hsrc].sort_values("date")
        if not h.empty:
            ax.plot(h["date"], h["swe_mm"], color="red", lw=2.5, marker="o", label="SNOWCAP")
        dates = sorted(d["date"].unique())
        ax.set_xticks(dates)
        ax.set_xticklabels([f"{pd.Timestamp(t):%m-%d}\n+{(pd.Timestamp(t) - BASE_TIME).days} d" for t in dates], fontsize=8)
        ax.set_xlabel("date and lead day")
        ax.set_ylabel("domain mean SWE (mm)")
        ax.set_title(f"{what}, all members vs SNOWCAP, base {BASE_TIME:%Y-%m-%d} ({MODEL_LABEL})")
        ax.grid(alpha=0.3)
        ax.legend()
        p = out_path("members", kind)
        fig.savefig(p, dpi=DPI)
        plt.close(fig)
        print(f"time series plot -> {p}")


def plot_member_timeseries_regions(df):
    """Every member, their mean and SNOWCAP per region, one figure per producer with a panel per region."""
    keys = [k for k in REGIONS if k in set(df["region"])]
    if not keys:
        return
    for src, hsrc, kind, what in [("ECXENS", "SNOWCAP_ECXENS", "ecxens-timeseries-regions", f"ECXENS ({FINE_GRID})"),
                                  ("ECENS", "SNOWCAP_ECENS", "ecens-timeseries-regions", f"raw ECENS ({ECENS_GRID} cells)")]:
        if df[df["source"] == src].empty:
            continue
        fig, axes = region_grid(keys)
        dates = sorted(df.loc[df["source"] == src, "date"].unique())
        for ax, k in zip(axes, keys):
            d = df[(df["source"] == src) & (df["region"] == k)]
            for i, (m, g) in enumerate(d.groupby("member")):
                g = g.sort_values("date")
                ax.plot(g["date"], g["swe_mm"], color="tab:blue", lw=0.6, alpha=0.5, label=f"members ({d['member'].nunique()})" if i == 0 else None)
            mean = d.groupby("date")["swe_mm"].mean()
            ax.plot(mean.index, mean.values, color="k", lw=1.6, label="member mean")
            h = df[(df["source"] == hsrc) & (df["region"] == k)].sort_values("date")
            if not h.empty:
                ax.plot(h["date"], h["swe_mm"], color="red", lw=2.2, marker="o", ms=3, label="SNOWCAP")
            ax.set_title(REGIONS[k][0], fontsize=10)
            ax.grid(alpha=0.3)
            date_ticks(ax, dates)
        axes[0].set_ylabel("domain mean SWE (mm)")
        axes[0].legend(fontsize=7)
        fig.suptitle(f"{what} members vs SNOWCAP per region, base {BASE_TIME:%Y-%m-%d} ({MODEL_LABEL})")
        p = out_path("members", kind)
        fig.savefig(p, dpi=DPI)
        plt.close(fig)
        print(f"regional member time series -> {p}")


# ---------------------------------------------------------------- one product (raw or adjusted ECXENS)
def state_path():
    return os.path.join(OUTDIR, f".state{PRED_TAG or '-raw'}.json")


def read_state():
    try:
        with open(state_path()) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def write_state(st):
    with open(state_path(), "w") as f:
        json.dump(st, f, indent=2)


def _ftime(p):
    return int(round(os.path.getmtime(p))) if p and os.path.exists(p) else None


def input_sig(ecx_src, ens_src, e5_path, snowcap_files, label, use_clms):
    """Inputs one prediction is made from, with their modification times: a change in any of them makes it be done again."""
    sig = {"ecxens": int(round(ecx_src.mtime())),
           "ecens": int(round(ens_src.mtime())) if ens_src is not None else None,
           "snowcap": sorted([os.path.basename(f), _ftime(f)] for f in snowcap_files) or None,
           "clms": bool(use_clms)}
    if label == "ensmean":      # the 500 m ECENS file is used by the ens mean only
        sig["ecens500"] = [os.path.basename(e5_path), _ftime(e5_path)] if e5_path else None
    return sig


def sig_change(prev, sig):
    """'same': nothing to do; 'new': inputs added (or no record of the inputs), keep the pngs and make the missing ones;
    'changed': an input used before changed, redraw everything of this prediction."""
    old = (prev or {}).get("sig")
    if old == sig:
        return "same"
    if old is None:
        return "new"
    for k, v in sig.items():
        if old.get(k) not in (None, False) and old.get(k) != v:
            return "changed"
    return "new"


def clear_product():
    """Removes the result folders of the current product before it is done again, so nothing of the earlier run is left
    (e.g. a driest or wettest member folder of another member number): adjusted: every <pred>-adjusted/; raw: every plain
    <pred>/ (adjusted-vs-raw/ is handled in main)."""
    if not os.path.isdir(OUTDIR):
        return
    gone = []
    for d in sorted(os.listdir(OUTDIR)):
        full = os.path.join(OUTDIR, d)
        if not os.path.isdir(full) or d.startswith('.') or d in CMP_DIRS:
            continue
        if adj_tag_of(d) == PRED_TAG:
            shutil.rmtree(full)
            gone.append(d)
    if gone:
        print(f"removed the earlier {MODEL_LABEL} results: {', '.join(gone)}")


def run_model(snowcap, snowcap_files, ecx_src, clms_on=False):
    """Every result for one ECXENS product into OUTDIR (set_product first). Returns (ens mean loader, anything redone).
    Predictions finished in an earlier run with the same ECXENS files are skipped (REDO does them again)."""
    global SKIP_PNG
    os.makedirs(OUTDIR, exist_ok=True)
    st = read_state()
    mtime = ecx_src.mtime()
    forced = REDO or (REDO_ADJ and bool(PRED_TAG)) or (REDO_ERA5L and PRED_TAG == '-adjusted-era5l')
    changed = forced or (st.get("mtime") is not None and abs(st["mtime"] - mtime) > 1)
    SKIP_PNG = not changed      # first run with a state file: existing pngs are kept
    if changed:
        print(f"{MODEL_LABEL}: " + ("done again (redo)" if forced else "ECXENS files changed since the last run, done again"))
        clear_product()
    if changed or "mtime" not in st:
        st = {"ecxens_file": os.path.basename(ecx_src.path), "mtime": mtime, "preds": {}}
        write_state(st)
    print(f"\n################ {MODEL_LABEL}, base {BASE_TIME:%Y-%m-%d}, ECXENS {os.path.basename(ecx_src.path)}, "
          f"outputs in {OUTDIR}/<pred>{PRED_TAG}/")
    ens_src = find_source("ECENS", ECENS_DIRS, ECENS_TEMPLATES, ECENS_SCALE, ECENS_MEAN_TEMPLATES)
    e5_path = next((d + t.format(**_fill) for d in ECENS_DIRS for t in ECENS500_TEMPLATES
                    if os.path.exists(d + t.format(**_fill))), None)
    print(f"ECENS ens mean remapped to {FINE_GRID}: " + (os.path.basename(e5_path) if e5_path else
          f"not found ({', '.join(t.format(**_fill) for t in ECENS500_TEMPLATES)}), ens mean without the {FINE_GRID} ECENS maps"))

    # ---- one pass over the members if anything needs it
    rank_csv = out_path("members", "ranking", ext="csv")
    ts_csv = out_path("members", "timeseries", ext="csv")
    has_members = len(ecx_src.members()) > 0
    want_rank = has_members and ADD_EXTREME_MEMBERS and (RERANK or changed or not os.path.exists(rank_csv))
    want_ts = has_members and MEMBER_TIMESERIES and (RETIMESERIES or changed or not os.path.exists(ts_csv))
    prev_mean = st["preds"].get("ensmean")
    mean_sig = input_sig(ecx_src, ens_src, e5_path, snowcap_files, "ensmean", clms_on and not PRED_TAG)
    mean_done = sig_change(prev_mean, mean_sig) == "same" and os.path.exists(prev_path(tagged("ensmean"), "info", "json"))
    want_ecx_mean = not ecx_src.has_mean_file and not mean_done   # done: computed later only if the comparison needs it
    want_ens_mean = ens_src is not None and not ens_src.has_mean_file and not mean_done
    rk = ts_df = ecx_mean = ens_mean = None
    # raw ECENS member time series are the same for raw and adjusted ECXENS: kept in .ecens-members-timeseries.csv of the
    # run folder and read from there while the ECENS files are unchanged, so the ECENS members are read only once
    ens_cache = os.path.join(OUTDIR, ".ecens-members-timeseries.csv")
    ens_meta = ens_cache[:-4] + ".json"
    ens_ts = None
    if want_ts and ens_src is not None and not REDO and os.path.exists(ens_cache):
        try:
            with open(ens_meta) as f:
                ok_cache = abs(json.load(f).get("mtime", -1) - ens_src.mtime()) <= 1
        except (OSError, ValueError):
            ok_cache = False
        if ok_cache:
            ens_ts = pd.read_csv(ens_cache, parse_dates=["date"])
            print(f"raw ECENS member time series from {os.path.basename(ens_cache)} (ECENS members not read again)")
    if want_rank or want_ts or want_ecx_mean or want_ens_mean:
        scan_ens = None if (ens_ts is not None and not want_ens_mean) else ens_src
        print(f"\n=== reading all members (ranking {want_rank}, time series {want_ts}, "
              f"ECXENS ens mean {want_ecx_mean}, ECENS ens mean {want_ens_mean}, ECENS members {scan_ens is not None}) ===")
        rk, ts_df, ecx_mean, ens_mean = scan_members(ecx_src, scan_ens, snowcap, want_ts, want_ecx_mean, want_ens_mean)
        if ts_df is not None:
            ens_rows = ts_df["source"].isin(["ECENS", "SNOWCAP_ECENS"])
            if ens_ts is not None:
                ts_df = pd.concat([ts_df[~ens_rows], ens_ts], ignore_index=True)
            elif ens_src is not None and ens_rows.any():
                ts_df[ens_rows].to_csv(ens_cache, index=False, float_format="%.4f")
                with open(ens_meta, "w") as f:
                    json.dump({"mtime": ens_src.mtime(), "ecens_file": os.path.basename(ens_src.path)}, f)
    if has_members and ADD_EXTREME_MEMBERS:
        if rk is None and os.path.exists(rank_csv):
            print(f"Reusing member ranking: {rank_csv}")
            rk = pd.read_csv(rank_csv)
        if rk is not None:
            rk = rk[[c for c in rk.columns if not c.startswith("rank_")]].sort_values(RANK_BY).reset_index(drop=True)
            rk["rank_" + RANK_BY] = np.arange(1, len(rk) + 1)
            rk.to_csv(rank_csv, index=False, float_format="%.4f")
            print(f"Member ranking by {RANK_BY} -> {rank_csv}")
            print(rk.head(3).to_string(index=False))
            print("...")
            print(rk.tail(3).to_string(index=False))
    if MEMBER_TIMESERIES and has_members:
        if ts_df is None and os.path.exists(ts_csv):
            print(f"Reusing member time series: {ts_csv}")
            ts_df = pd.read_csv(ts_csv, parse_dates=["date"])
            if not todo(out_path("members", "ecxens-timeseries")) and not todo(out_path("members", "ecxens-timeseries-regions")):
                ts_df = None    # csv and plots (whole domain and regions) there already
        elif ts_df is not None:
            ts_df.to_csv(ts_csv, index=False, float_format="%.4f")
        if ts_df is not None:
            plot_member_timeseries(ts_df)

    # ---- predictions to validate: (label, name, ECXENS loader, ECENS loader, ECENS name, ECENS 500 m loader)
    ens_members = set(ens_src.members()) if ens_src is not None else set()

    def ens_loader(m):
        if ens_src is None:
            return lambda: None
        if m == "ensmean":
            return lambda: ens_mean if ens_mean is not None else (ens_src.load("ensmean") if ens_src.has_mean_file else None)
        return lambda: ens_src.load(m) if m in ens_members else None

    def ecx_mean_loader():
        # ens mean of this product: from the member pass, the ens mean file, or the members read now (only when needed)
        if ecx_mean is not None:
            return ecx_mean
        if ecx_src.has_mean_file or not has_members:
            return ecx_src.load("ensmean")
        print(f"{MODEL_LABEL}: computing the ECXENS ens mean from the members")
        return scan_members(ecx_src, None, snowcap, False, True, False)[2]

    preds = [("ensmean", "ECXENS ens mean" if has_members or ecx_src.split else "ECXENS", ecx_mean_loader,
              ens_loader("ensmean"), "ECENS ens mean",
              (lambda: load_field(e5_path, ECENS_SCALE)) if e5_path else (lambda: None))]
    members = ecx_src.members()
    if members:
        m1 = FIRST_MEMBER if FIRST_MEMBER in members else min(x for x in members if x > 0) if any(x > 0 for x in members) else members[0]
        if m1 != FIRST_MEMBER:
            print(f"ECXENS member {FIRST_MEMBER} missing, member {m1} validated instead")
        preds.append((f"member{m1:02d}", f"ECXENS member {m1}", (lambda m=m1: ecx_src.load(m)), ens_loader(m1), f"ECENS member {m1}",
                      lambda: None))
    if rk is not None and len(rk):
        for label, m in [("driest", int(rk["member"].iloc[0])), ("wettest", int(rk["member"].iloc[-1]))]:
            preds.append((f"{label}-member{m:02d}", f"ECXENS member {m} ({label})", (lambda m=m: ecx_src.load(m)),
                          ens_loader(m), f"ECENS member {m}", lambda: None))
            print(f"{label} member by {RANK_BY}: {m}")

    if PRED_TAG:   # adjusted product: names say so (shown in SnowLook and in the figure titles)
        preds = [(lab, nm.replace("ECXENS", ADJ_PRODUCTS[PRED_TAG]['name'], 1)) + tuple(rest) for lab, nm, *rest in preds]
    # one run-info for both products: the keys of the product run now are replaced, the other's kept
    ri_path = os.path.join(OUTDIR, f"{PREFIX}_run-info.json")
    try:
        with open(ri_path) as f:
            info = json.load(f)
    except (OSError, ValueError):
        info = {}
    key = "ecxens" + PRED_TAG.replace("-", "_")
    info.update({"model": MODEL, "base_time": f"{BASE_TIME:%Y-%m-%d %H:%M}", "region": REGION,
                 f"{key}_file": os.path.basename(ecx_src.path), f"{key}_members": ecx_src.members(),
                 "ecens_file": os.path.basename(ens_src.path) if ens_src is not None else None,
                 "ecens500_file": os.path.basename(e5_path) if e5_path else None,
                 "snowcap_file": [os.path.basename(p) for p in snowcap_files],
                 "regions": {k: {"name": v[0], "country": v[1], "lat_from": v[2], "lat_to": v[3]} for k, v in REGIONS.items()}})
    other = [x for x in info.get("preds", []) if adj_tag_of(x) != PRED_TAG]
    info["preds"] = other + [tagged(p[0]) for p in preds]
    with open(ri_path, "w") as f:
        json.dump(info, f, indent=2)

    all_dfs = []
    redone = False
    skip_png_product = SKIP_PNG
    for label, name, load_ecx, load_ens, ens_name, load_e5 in preds:
        use_clms = clms_on and label == "ensmean" and not PRED_TAG
        prev = st["preds"].get(label)
        sig = input_sig(ecx_src, ens_src, e5_path, snowcap_files, label, use_clms)
        how = sig_change(prev, sig)
        if how == "same" and os.path.exists(prev_path(tagged(label), "info", "json")):
            print(f"\n=== {name}: done in an earlier run with the same inputs, skipped ('redo' as the third argument does it again) ===")
            df = _read(prev_path(tagged(label), "ecxens-metrics", "csv"))
            if df is not None:
                all_dfs.append(df.assign(pred=label))
            continue
        redone = True
        # new inputs only: existing pngs kept, the missing ones made; an input used before changed: every png redrawn
        SKIP_PNG = skip_png_product and how != "changed"
        print(f"\n=== {name} ===" + ("" if prev is None else
              "  (an input changed since the last run, done again)" if how == "changed" else
              "  (inputs added since the last run, missing results made)"))
        ecx = load_ecx()
        ens = load_ens()
        if ens is not None:
            ens = check_ecens(ens, snowcap)
        else:
            print(f"{ens_name}: not available, maps without it")
        try:
            e5 = load_e5()
        except Exception as ex:   # a broken remapped file must not stop the rest of the validation
            print(f"WARNING: {ens_name} remapped to {FINE_GRID} not read, left out: {ex}")
            e5 = None
        if e5 is not None:
            e5 = check_ecens(e5, snowcap, label=f"{ens_name} remapped to {FINE_GRID}", mask=False)
        df = validate(ecx, snowcap, label, name, ens=ens, ens_name=ens_name, ens500=e5, full=(label == "ensmean"), clms=use_clms)
        if df is not None:
            all_dfs.append(df.assign(pred=label))
        del ecx, ens, e5
        st["preds"][label] = {"clms": use_clms, "sig": sig}
        write_state(st)     # after every prediction, so a crash continues from here
    SKIP_PNG = skip_png_product

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        cols = ["pred"] + [c for c in combined.columns if c != "pred"]
        p = out_path("all", "ecxens-metrics", ext="csv")
        combined[cols].to_csv(p, index=False, float_format="%.4f")
        print(f"\nCombined scores: {p}")
    return preds[0][2], redone


# ---------------------------------------------------------------- adjusted against raw
def _read(path, **kw):
    return pd.read_csv(path, **kw) if os.path.exists(path) else None


def cmp_state(atag):
    """Marker of a finished comparison of one adjusted product (written at the end of compare_adjusted)."""
    return os.path.join(OUTDIR, f".state-cmp{atag}.json")


def compare_adjusted(snowcap, load_raw, load_adj, atag="-adjusted"):
    """An adjusted product against raw ECXENS (their csv files in <pred>/ and <pred><atag>/) into its comparison folder
    (adjusted-vs-raw/, adjusted-era5l-vs-raw/). The file kinds are the same for every product, the folder tells them apart."""
    P = ADJ_PRODUCTS[atag]
    pred = P["cmp"]
    rn, an = f"ECXENS raw ({MODEL})", f"{P['name']} ({MODEL})"
    tag = f"ens mean, SWE {FINE_GRID}, base {BASE_TIME:%Y-%m-%d}"
    print(f"\n=== {an} against {rn} ===")

    # ---- scores per lead day
    mr = _read(prev_path("ensmean", "ecxens-metrics", "csv"))
    ma = _read(prev_path("ensmean" + atag, "ecxens-metrics", "csv"))
    if mr is not None and ma is not None:
        m = mr.merge(ma, on=["date", "lead_day"], suffixes=("_raw", "_adj"))
        m["skill_mse_adj_vs_raw"] = 1 - m["rmse_adj"] ** 2 / m["rmse_raw"] ** 2
        m["skill_mse_adj_vs_ecens500"] = (1 - m["rmse_adj"] ** 2 / m["ecens500_rmse_raw"] ** 2) if "ecens500_rmse_raw" in m else np.nan
        keep = ["date", "lead_day"] + [f"{c}_{v}" for c in ["bias", "mae", "rmse", "r", "kge", "snow_bias", "snow_rmse", "snow_kge",
                                                             "ecens500_bias", "ecens500_mae", "ecens500_rmse", "ecens500_kge"]
                                       for v in ("raw", "adj") if f"{c}_{v}" in m and not (c.startswith("ecens500") and v == "adj")]
        keep += ["skill_mse_adj_vs_raw", "skill_mse_adj_vs_ecens500"]
        m[keep].to_csv(out_path(pred, "adjusted-vs-raw-metrics", ext="csv"), index=False, float_format="%.4f")
        L = m["lead_day"]
        has5 = "ecens500_rmse_raw" in m and m["ecens500_rmse_raw"].notna().any()
        e5 = "ECENS ens mean remapped"
        for sc, title, zero in [("rmse", "RMSE (mm)", False), ("bias", "Bias, forecast minus SNOWCAP (mm)", True),
                                ("mae", "MAE (mm)", False), ("kge", "KGE", False)]:
            series = [(L, m[f"{sc}_raw"], "o-", rn), (L, m[f"{sc}_adj"], "s-", an)]
            if f"snow_{sc}_raw" in m:
                series += [(L, m[f"snow_{sc}_raw"], "o--", f"{rn}, snow px"), (L, m[f"snow_{sc}_adj"], "s--", f"{an}, snow px")]
            if has5 and f"ecens500_{sc}_raw" in m:
                series.append((L, m[f"ecens500_{sc}_raw"], "^:", e5))
            lead_plot(out_path(pred, f"adjusted-vs-raw-score-{sc}"), f"{title}\n{tag}", series, zero=zero,
                      legend_kw=dict(fontsize=8))
        lead_plot(out_path(pred, "adjusted-vs-raw-score-skill"), f"MSE skill of {P['name']} (1 = perfect, 0 = as the reference)\n{tag}",
                  [(L, m["skill_mse_adj_vs_raw"], "o-", f"against {rn}")]
                  + ([(L, m["skill_mse_adj_vs_ecens500"], "s--", f"against {e5}")] if has5 else []), zero=True)
        cols = [c for c in ["lead_day", "bias_raw", "bias_adj", "rmse_raw", "rmse_adj", "ecens500_rmse_raw", "skill_mse_adj_vs_raw"] if c in m]
        print(m[cols].round(2).to_string(index=False))
    else:
        print("ens mean scores of raw or adjusted missing (no SNOWCAP days?), no lead day comparison")

    # ---- domain mean time series
    tr = _read(prev_path("ensmean", "domain-mean-timeseries", "csv"), parse_dates=["date"])
    ta = _read(prev_path("ensmean" + atag, "domain-mean-timeseries", "csv"), parse_dates=["date"])
    if tr is not None and ta is not None:
        t = tr.merge(ta[["date", "ecxens"]].rename(columns={"ecxens": "ecxens_adjusted"}), on="date", how="outer").sort_values("date")
        t.to_csv(out_path(pred, "adjusted-vs-raw-domain-mean-timeseries", ext="csv"), index=False, float_format="%.4f")
        fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
        for col, lab, sty in [("ecxens", rn, dict(color="tab:blue", lw=2, marker="o")),
                              ("ecxens_adjusted", an, dict(color="tab:green", lw=2, marker="D")),
                              ("ecens500", "ECENS ens mean remapped", dict(color="tab:orange", lw=2, ls="--", marker="s")),
                              ("snowcap", "SNOWCAP", dict(color="red", lw=2.5, marker="o")),
                              ("clms", "CLMS (satellite, CLMS pixels only)", dict(color="tab:purple", lw=2, ls="-.", marker="^"))]:
            if col in t and t[col].notna().any():
                d = t.dropna(subset=[col])
                ax.plot(d["date"], d[col], label=lab, **sty)
        dates = sorted(t["date"].unique())
        ax.set_xticks(dates)
        ax.set_xticklabels([f"{pd.Timestamp(x):%m-%d}\n+{(pd.Timestamp(x) - BASE_TIME).days} d" for x in dates], fontsize=8)
        ax.set_ylabel("domain mean SWE (mm)")
        ax.set_title(f"Domain mean SWE, SNOWCAP domain\n{tag}")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.savefig(out_path(pred, "adjusted-vs-raw-domain-mean-timeseries"), dpi=DPI)
        plt.close(fig)

    # ---- regions: time series and scores
    rr = _read(prev_path("ensmean", "domain-mean-timeseries-regions", "csv"), parse_dates=["date"])
    ra = _read(prev_path("ensmean" + atag, "domain-mean-timeseries-regions", "csv"), parse_dates=["date"])
    if rr is not None and ra is not None:
        r = rr.merge(ra[["date", "region", "ecxens"]].rename(columns={"ecxens": "ecxens_adjusted"}), on=["date", "region"], how="outer")
        keys = [k for k in REGIONS if k in set(r["region"])]
        fig, axes = region_grid(keys)
        dates = sorted(r["date"].unique())
        for ax, k in zip(axes, keys):
            d = r[r["region"] == k].sort_values("date")
            for col, lab, sty in [("ecxens", "raw", dict(color="tab:blue", lw=1.8, marker="o", ms=3)),
                                  ("ecxens_adjusted", "adjusted", dict(color="tab:green", lw=1.8, marker="D", ms=3)),
                                  ("ecens500", "ECENS remapped", dict(color="tab:orange", lw=1.8, ls="--", marker="s", ms=3)),
                                  ("snowcap", "SNOWCAP", dict(color="red", lw=2.2, marker="o", ms=3))]:
                if col in d and d[col].notna().any():
                    ax.plot(d["date"], d[col], label=lab, **sty)
            ax.set_title(REGIONS[k][0], fontsize=10)
            ax.grid(alpha=0.3)
            date_ticks(ax, dates)
        axes[0].set_ylabel("domain mean SWE (mm)")
        axes[0].legend(fontsize=7)
        fig.suptitle(f"Domain mean SWE per region, ECXENS raw and {P['label']}, {tag}")
        fig.savefig(out_path(pred, "adjusted-vs-raw-timeseries-regions"), dpi=DPI)
        plt.close(fig)
    sr = _read(prev_path("ensmean", "regions-metrics", "csv"))
    sa = _read(prev_path("ensmean" + atag, "regions-metrics", "csv"))
    if sr is not None and sa is not None:
        keys = [k for k in REGIONS if k in set(sr["region"]) | set(sa["region"])]
        gr = sr.groupby("region").mean(numeric_only=True).reindex(keys)
        ga = sa.groupby("region").mean(numeric_only=True).reindex(keys)
        has5 = "ecens500_rmse" in gr and gr["ecens500_rmse"].notna().any()
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)
        x = np.arange(len(keys))
        nb = 3 if has5 else 2
        wbar = 0.8 / nb
        names = [REGIONS[k][0] for k in keys]
        for ax, (sc, title) in zip(axes[:2], [("rmse", "RMSE (mm)"), ("bias", "bias, forecast minus SNOWCAP (mm)")]):
            ax.bar(x - wbar * (nb - 1) / 2, gr[sc], wbar, label="ECXENS raw", color="tab:blue")
            ax.bar(x - wbar * (nb - 1) / 2 + wbar, ga[sc], wbar, label=P["name"], color="tab:green")
            if has5:
                ax.bar(x + wbar, gr[f"ecens500_{sc}"], wbar, label="ECENS remapped", color="tab:orange")
            ax.set_xticks(x)
            ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
            ax.set_title(title)
            ax.axhline(0, color="k", lw=0.6)
            ax.grid(alpha=0.3, axis="y")
        axes[0].legend(fontsize=8)
        sk = 1 - ga["rmse"] ** 2 / gr["rmse"] ** 2      # from the mean RMSE over the lead days
        ax = axes[2]
        ax.bar(x, sk, 0.7, color=["tab:green" if v > 0 else "tab:red" for v in sk])
        ax.set_title("MSE skill of adjusted vs raw (from the mean RMSE)\n(above 0 = adjusting helped)")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
        ax.axhline(0, color="k", lw=0.6)
        ax.grid(alpha=0.3, axis="y")
        fig.suptitle(f"Scores per region, means over the lead days, {tag}")
        fig.savefig(out_path(pred, "adjusted-vs-raw-regions-scores"), dpi=DPI)
        plt.close(fig)
        tab = pd.DataFrame({"rmse_raw": gr["rmse"], "rmse_adj": ga["rmse"], "bias_raw": gr["bias"], "bias_adj": ga["bias"],
                            "skill_adj_vs_raw": sk}).rename(index=lambda k: REGIONS[k][0])
        tab.to_csv(out_path(pred, "adjusted-vs-raw-regions-scores", ext="csv"), float_format="%.4f")
        print(tab.round(2).to_string())

    # ---- member spread of both against SNOWCAP (whole domain)
    mr = _read(prev_path("members", "timeseries", "csv"), parse_dates=["date"])
    ma = _read(prev_path("members" + atag, "timeseries", "csv"), parse_dates=["date"])
    if mr is not None and ma is not None:
        def sel(df, src):
            d = df[df["source"] == src]
            return d[d["region"] == "all"] if "region" in d else d
        fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
        for df, lab, col in [(mr, "ECXENS raw", "tab:blue"), (ma, P["name"], "tab:green")]:
            d = sel(df, "ECXENS")
            if d.empty:
                continue
            q = d.groupby("date")["swe_mm"].quantile([0.05, 0.5, 0.95]).unstack()
            ax.fill_between(q.index, q[0.05], q[0.95], color=col, alpha=0.2, label=f"{lab} members 5 to 95 %")
            ax.plot(d.groupby("date")["swe_mm"].mean(), color=col, lw=2, label=f"{lab} member mean")
        h = sel(mr, "SNOWCAP_ECXENS").sort_values("date")
        if not h.empty:
            ax.plot(h["date"], h["swe_mm"], color="red", lw=2.5, marker="o", label="SNOWCAP")
        ax.set_ylabel("domain mean SWE (mm)")
        ax.set_title(f"ECXENS members raw and {P['label']} vs SNOWCAP, SNOWCAP domain, base {BASE_TIME:%Y-%m-%d}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.savefig(out_path(pred, "adjusted-vs-raw-members-timeseries"), dpi=DPI)
        plt.close(fig)

    # ---- maps: where adjusting helped, and the mean adjustment
    if not todo(out_path(pred, "adjustment-mean")) and not todo(out_path(pred, "adjusted-vs-raw-mae-gain")):
        print(f"comparison maps there already, outputs in {os.path.join(OUTDIR, pred)}/")
        return
    raw, adj = load_raw(), load_adj()
    if raw is None or adj is None:
        return
    adj = match_grid(raw, adj, P["name"])
    sc = match_snowcap(raw, snowcap) if snowcap is not None else None
    sc_times = sc.valid_time.values if sc is not None else np.array([], dtype="datetime64[ns]")
    shape = (raw.sizes["latitude"], raw.sizes["longitude"])
    sum_a, cnt_a, sum_g, cnt_g = np.zeros(shape), np.zeros(shape), np.zeros(shape), np.zeros(shape)
    times = [t for t in raw.valid_time.values if t in set(adj.valid_time.values)]
    for t in times:
        a = adj.sel(valid_time=t).values.astype("float64")
        x = raw.sel(valid_time=t).values.astype("float64")
        ok = np.isfinite(a) & np.isfinite(x)
        sum_a[ok] += (a - x)[ok]
        cnt_a[ok] += 1
        if t in sc_times:
            o = sc.sel(valid_time=t).values.astype("float64")
            v = ok & np.isfinite(o)
            sum_g[v] += (np.abs(a - o) - np.abs(x - o))[v]
            cnt_g[v] += 1
    extent = grid_extent(raw)
    aspect = 1.0 / np.cos(np.deg2rad(raw.latitude.values.mean()))
    lines = get_lines(extent)
    period = f"{pd.Timestamp(times[0]):%Y-%m-%d} to {pd.Timestamp(times[-1]):%Y-%m-%d}" if times else ""
    with np.errstate(invalid="ignore", divide="ignore"):
        save_bias_map(out_path(pred, "adjustment-mean"), np.where(cnt_a > 0, sum_a / cnt_a, np.nan), extent, aspect, lines,
                      f"Mean adjustment, {P['name']} minus raw SWE ({FINE_GRID})\n{period}", label="mm")
        if cnt_g.any():
            save_bias_map(out_path(pred, "adjusted-vs-raw-mae-gain"), np.where(cnt_g > 0, sum_g / cnt_g, np.nan), extent, aspect, lines,
                          f"|{P['label']} bias| minus |raw bias| vs SNOWCAP SWE ({FINE_GRID})\n{period}, {int(cnt_g.max())} SNOWCAP days",
                          label="mm, negative = adjusting helped")
    print(f"comparison done, outputs in {os.path.join(OUTDIR, pred)}/")
    with open(cmp_state(atag), "w") as f:
        json.dump({"finished": f"{pd.Timestamp.now():%Y-%m-%d %H:%M}"}, f)


# ---------------------------------------------------------------- main
def main():
    global SKIP_PNG
    os.makedirs(OUTDIR, exist_ok=True)
    print(f"model {MODEL}, base {BASE_TIME:%Y-%m-%d}, outputs in {OUTDIR}" + ("  (redo: everything again)" if REDO else ""))
    snowcap, snowcap_files = load_snowcap()
    clms_on = clms_available()
    loaders = {}
    redone = False
    raw_src = find_source("ECXENS", [ECX_DIR], ECX_TEMPLATES, ECX_SCALE, ECX_MEAN_TEMPLATES)
    if RUN_RAW:
        set_product()
        if raw_src is None:
            print(f"no raw ECXENS forecast for {BASE_TIME:%Y-%m-%d} in {ECX_DIR}, raw left out")
        else:
            loaders["raw"], r = run_model(snowcap, snowcap_files, raw_src, clms_on)
            redone |= r
    redone_tag = {}
    srcs = {}
    if RUN_ADJUSTED:
        for atag, P in ADJ_PRODUCTS.items():
            set_product(atag)
            srcs[atag] = find_source("ECXENS", [ECX_DIR], P["templates"], ECX_SCALE, ADJ_MEAN_TEMPLATES)
            if srcs[atag] is None:
                print(f"no {P['name']} ({P['templates'][0].format(**_fill).format(m='<n>')}) in {ECX_DIR}, left out; "
                      f"run adjust-snow-ecxens.py and prep-ecxens-validation.sh {BASE_TAG} first")
                continue
            loaders[atag], r = run_model(snowcap, snowcap_files, srcs[atag])
            redone_tag[atag] = r
    set_product()

    def load_from(src):
        return (lambda: src.load("ensmean")) if src is not None and src.has_mean_file else (lambda: None)
    for atag, P in ADJ_PRODUCTS.items():
        # new results of raw or of this product: its comparison is drawn again, from an empty folder
        SKIP_PNG = not (REDO or REDO_ADJ or (REDO_ERA5L and atag == '-adjusted-era5l') or redone or redone_tag.get(atag))
        cmp_dir = os.path.join(OUTDIR, P["cmp"])
        if not SKIP_PNG:
            if os.path.isdir(cmp_dir):
                shutil.rmtree(cmp_dir)
                print(f"removed the earlier {P['cmp']} results")
            if os.path.exists(cmp_state(atag)):
                os.remove(cmp_state(atag))
        elif os.path.exists(cmp_state(atag)) or os.path.exists(prev_path(P["cmp"], "adjusted-vs-raw-metrics", "csv")):
            # finished (a comparison from before the marker existed counts as finished when its scores are there)
            print(f"\n=== {P['cmp']}: done in an earlier run, skipped ('redo-adjusted' or 'redo-era5l' does it again) ===")
            continue
        if COMPARE_ADJUSTED and os.path.exists(prev_path("ensmean", "info", "json")) and os.path.exists(prev_path("ensmean" + atag, "info", "json")):
            if atag not in loaders:
                loaders[atag] = load_from(srcs.get(atag) or find_source("ECXENS", [ECX_DIR], P["templates"], ECX_SCALE, ADJ_MEAN_TEMPLATES))
            compare_adjusted(snowcap, loaders.get("raw") or load_from(raw_src), loaders[atag], atag)
    if MANIFEST:
        write_manifest()
    if not loaders:
        raise SystemExit(f"no ECXENS forecast for {BASE_TIME:%Y-%m-%d} in {ECX_DIR}")


if __name__ == "__main__":
    main()