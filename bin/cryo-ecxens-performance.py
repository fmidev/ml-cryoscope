#!/usr/bin/env python3
"""
cryo-ecxens-performance.py

ecxens performance

"""

import os
import sys
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
BASE_TIME = pd.Timestamp(BASE)
BT = f'{BASE_TIME:%Y%m%dT%H%M%S}'
BASE_TAG = f'{BASE_TIME:%Y%m%d}'
_fill = dict(BT=BT, MODEL=MODEL, Y=f'{BASE_TIME:%Y}')

SNOW_ROOT = '/home/ubuntu/data/cryoscope/snowmodel/'
ECX_DIR = SNOW_ROOT + 'ECXENS/'
ECENS_DIRS = [SNOW_ROOT + 'ECENS/', ECX_DIR, SNOW_ROOT]      # first directory holding a matching file wins
HKL_DIRS = [SNOW_ROOT + 'HKL/', ECX_DIR, SNOW_ROOT]
OUT_ROOT = SNOW_ROOT + 'ecxens_validation/'
OUTDIR = f'{OUT_ROOT}{MODEL}/{BASE_TAG}'
PREFIX = f'{MODEL}_{BASE_TAG}'
PRED_SUBFOLDERS = True      # True: OUTDIR/<pred>/ folders; False: everything flat in OUTDIR (names are unique either way)

# file names, tried in order ({m} = member 1..50 or ensmean); a template without {m} is one file holding all members
ECX_TEMPLATES = ['ECXENS_{BT}_swe-500-nd.grib',
                 'ECXENS_{BT}_swe-500-nd-{MODEL}-{{m}}.grib', 'ECXENS_{BT}_swe-500-nd_{MODEL}-{{m}}.grib',
                 'ECXENS_{BT}_swe-500-nd-{{m}}.grib']
ECENS_TEMPLATES = ['ECENS_{BT}_sd.grib',
                   'ECENS_{BT}_sd-{{m}}-fix.grib', 'ECENS_{BT}_era5l_sfc+sde-nd-{{m}}-sd.grib', 'ECENS_{BT}_sd-{{m}}.grib']
HKL_TEMPLATES = ['HKL_{Y}_SWE.grib', 'HKL_20150101T000000_{Y}_SWE.grib']
HKL_MONTH_TEMPLATES = ['HKL_{YM}_swe.grib', 'HKL_{YM}_SWE.grib']
FC_DAYS = 15        # forecast length in days (months of HKL to look for)
REGION = 'Scandinavia'

FIRST_MEMBER = 1           # the single member validated next to the ens mean (lowest available member if missing)
MEMBERS = range(0, 51)     # members looked for (0 = control, used only if the file has it)
ADD_EXTREME_MEMBERS = True # validate the driest and the wettest member too
RANK_BY = "mean"           # mean: area weighted domain mean SWE over all lead days; max; p99; mean_last_day
RERANK = True              # False reuses the ranking csv if it exists
MEMBER_TIMESERIES = True   # domain mean SWE of every member against HKL
RETIMESERIES = True        # False reuses the time series csv if it exists
ECENS_SCALE = 1000.0       # ECMWF sd is m w.e., x1000 gives mm
ECX_SCALE = 1.0            # ECXENS is mm w.e.
HKL_SCALE = 1000.0         # Scandinavian HKL SWE is m w.e., always x1000
MASK_ECENS_TO_HKL = False  # True hides ECENS cells where HKL has no data
ECENS_OWN_EXTENT = True    # combined figure only: ECENS panel shows its whole grid
PANEL_ECENS_VIEW = "ecxens"  # single ECENS map area: "ecxens" lines up with the ECXENS / HKL maps, "own" = whole ECENS grid
MIN_COVER = 0.5            # 0.1 degree comparison: share of valid 500 m pixels a cell needs
SAVE_PANELS = True         # ECENS, ECXENS and HKL SWE maps each in their own png
SAVE_COMBINED = False      # old three panel figure (ecens / ecxens / hkl side by side), off: every figure separately
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
GAIN_LABEL = ("|ECXENS bias| minus |ECENS bias| (mm), negative = ECXENS closer to HKL" if BIAS_COMPARE == "abs"
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
    """<OUTDIR>/<pred>/<MODEL>_<BASE>_<pred>_<kind>[_<YYYYMMDD>].<ext>, registered in the manifest."""
    d = os.path.join(OUTDIR, pred) if PRED_SUBFOLDERS else OUTDIR
    os.makedirs(d, exist_ok=True)
    name = f"{PREFIX}_{pred}_{kind}" + (f"_{day:%Y%m%d}" if day is not None else "") + f".{ext}"
    p = os.path.join(d, name)
    MANIFEST[p] = {"model": MODEL, "base": BASE_TAG, "pred": pred, "kind": kind,
                   "valid_date": f"{day:%Y%m%d}" if day is not None else "",
                   "lead_day": (day - BASE_TIME).days if day is not None else "",
                   "ext": ext, "file": os.path.relpath(p, OUT_ROOT)}
    return p


def write_manifest():
    rows = [r for p, r in MANIFEST.items() if os.path.exists(p)]
    if not rows:
        return
    p = os.path.join(OUTDIR, f"{PREFIX}_manifest.csv")
    pd.DataFrame(rows).sort_values(["pred", "kind", "valid_date"]).to_csv(p, index=False)
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
        name = "sd" if "sd" in ds.data_vars else list(ds.data_vars)[0]
        parts.append(ds[name])
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
    """Single field file (HKL, or one member per file) -> DataArray in mm."""
    parts = _open_parts(path)
    if len(parts) > 1 or len(_numbers(parts[0])) > 1:
        raise ValueError(f"{path}: file holds several members, expected one")
    return normalize(parts[0], os.path.basename(path)) * np.float32(scale)


class Source:
    """ECXENS or ECENS forecast. Split layout: one file per member ({m} in the path), ensmean file optional.
    Single file layout: every member in one file ('number' dimension), the ens mean is computed from the members;
    a single file without members is taken as the ens mean."""

    def __init__(self, kind, path, scale):
        self.kind, self.path, self.scale = kind, path, scale
        self.split = "{m}" in path
        if self.split:
            self.numbers = [m for m in MEMBERS if os.path.exists(path.format(m=m))]
            self.has_mean_file = os.path.exists(path.format(m="ensmean"))
            self.parts = None
        else:
            self.parts = _open_parts(path)
            self.numbers = sorted({n for p in self.parts for n in _numbers(p) if n in MEMBERS})
            self.has_mean_file = all(len(_numbers(p)) == 0 for p in self.parts)
        print(f"{kind}: {os.path.basename(path)}, {len(self.numbers)} members"
              + (f" ({self.numbers[0]} to {self.numbers[-1]})" if self.numbers else "")
              + (", ens mean in file" if self.has_mean_file else ", ens mean computed from the members"))

    def members(self):
        return list(self.numbers)

    def load(self, m):
        if self.split:
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


def find_source(kind, dirs, templates, scale):
    """First directory and naming that has the file (single file layout) or the ensmean / member 1 file (split)."""
    tried = []
    for d in dirs:
        for t in templates:
            p = d + t.format(**_fill)
            if "{m}" in p:
                probes = [p.format(m="ensmean")] + [p.format(m=m) for m in (1, FIRST_MEMBER)]
            else:
                probes = [p]
            tried.append(os.path.basename(probes[0]))
            if any(os.path.exists(q) for q in probes):
                return Source(kind, p, scale)
    print(f"WARNING: no {kind} file found in {', '.join(dirs)}, tried {', '.join(sorted(set(tried)))}")
    return None


def pick_hkl():
    """HKL files for the forecast (list, may be empty): a file holding the first forecast day in its name,
    else the monthly files of every month the forecast reaches, else the whole year files."""
    day1 = f'{BASE_TIME + pd.Timedelta(days=1):%Y%m%d}'
    months = sorted({f'{BASE_TIME + pd.Timedelta(days=d):%Y%m}' for d in range(1, FC_DAYS + 1)})
    for d in HKL_DIRS:
        hits = sorted(glob.glob(f'{d}HKL_*{day1}*_SWE.grib'))
        if hits:
            return hits[:1]
        monthly = []
        for ym in months:
            p = next((d + t.format(YM=ym) for t in HKL_MONTH_TEMPLATES if os.path.exists(d + t.format(YM=ym))), None)
            if p:
                monthly.append(p)
        if monthly:
            return monthly
        for t in HKL_TEMPLATES:
            p = d + t.format(**_fill)
            if os.path.exists(p):
                return [p]
    print(f"WARNING: no HKL file for {BASE_TIME:%Y-%m-%d} in {', '.join(HKL_DIRS)}: forecast maps only, no scores")
    return []


def load_hkl():
    files = pick_hkl()
    if not files:
        return None, []
    parts = []
    for p in files:
        print(f"Loading HKL: {p}")
        parts.append(load_field(p, 1.0))
    hkl = xr.concat(parts, dim="valid_time") if len(parts) > 1 else parts[0]
    hkl = hkl.isel(valid_time=np.unique(hkl.valid_time.values, return_index=True)[1]).sortby("valid_time")
    hkl = hkl * np.float32(HKL_SCALE)
    hmax = float(hkl.max())
    print(f"HKL SWE after x{HKL_SCALE:g}: max {hmax:.1f} mm, mean {float(hkl.mean()):.1f} mm, "
          f"{pd.Timestamp(hkl.valid_time.values[0]):%Y-%m-%d} to {pd.Timestamp(hkl.valid_time.values[-1]):%Y-%m-%d}")
    if hmax < 20:
        print("WARNING: HKL SWE max below 20 mm: early season, or still m w.e. (check HKL_SCALE)")
    return hkl, files


def check_ecens(ens, hkl):
    emax = float(ens.max())
    print(f"ECENS SWE after x{ECENS_SCALE:g}: max {emax:.1f} mm, mean {float(ens.mean()):.1f} mm")
    if emax < 20:
        print("WARNING: ECENS SWE max below 20 mm, the values look like m w.e., check ECENS_SCALE")
    if emax > 20000:
        print("WARNING: ECENS SWE max above 20000 mm, the file may already be in mm, check ECENS_SCALE")
    if MASK_ECENS_TO_HKL and hkl is not None:
        ens = mask_to_domain(ens, hkl)
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


def match_hkl(ecx, hkl_raw):
    """HKL on the ECXENS grid (as is when the grids are the same, else nearest neighbour)."""
    same_size = (ecx.sizes["latitude"] == hkl_raw.sizes["latitude"] and ecx.sizes["longitude"] == hkl_raw.sizes["longitude"])
    if same_size:
        off_lat = float(np.abs(ecx.latitude.values - hkl_raw.latitude.values).max())
        off_lon = float(np.abs(ecx.longitude.values - hkl_raw.longitude.values).max())
        cell = min(abs(float(ecx.latitude[1] - ecx.latitude[0])), abs(float(ecx.longitude[1] - ecx.longitude[0])))
    if same_size and max(off_lat, off_lon) < GRID_TOL * cell:
        print(f"Same grid as HKL ({max(off_lat, off_lon) / cell:.4f} cells off at most), HKL used without interpolation")
        return hkl_raw.assign_coords(latitude=ecx.latitude, longitude=ecx.longitude)
    print(f"Grids differ: ECXENS {ecx.sizes['latitude']}x{ecx.sizes['longitude']}, "
          f"HKL {hkl_raw.sizes['latitude']}x{hkl_raw.sizes['longitude']}, interpolating HKL to ECXENS grid (nearest)")
    return hkl_raw.interp_like(ecx, method="nearest")


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
    fig, ax = plt.subplots(figsize=(7, 8), constrained_layout=True)
    im = draw_map(ax, data, bias_cmap, bias_norm, extent, aspect, lines, title, view=view)
    bias_colorbar(fig, im, ax, label=label)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def save_panel(path, data, ext, aspect, lines, title, view):
    """One SWE map alone, no colour bar: maps with the same view get the same figure size, so they line up."""
    w = PANEL_HEIGHT * (view[1] - view[0]) / ((view[3] - view[2]) * aspect)
    fig, ax = plt.subplots(figsize=(max(w, 3.0), PANEL_HEIGHT), constrained_layout=True)
    draw_map(ax, data, swe_cmap, swe_norm, ext, aspect, lines, title, view=view)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def save_swe_legend(path):
    fig, ax = plt.subplots(figsize=(12, 1.1), constrained_layout=True)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=swe_norm, cmap=swe_cmap), cax=ax, orientation="horizontal",
                      extend="both", ticks=SWE_BOUNDS, spacing="uniform")
    cb.set_label("SWE (mm)")
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def lead_plot(path, title, series, ylabel=None, ylim=None, zero=False, legend_kw=None):
    """One score against lead day; series = list of (x, y, style, label)."""
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
def validate(ecx, hkl_raw, pred, name, ens=None, ens_name=None):
    """Full validation of one prediction. Forecast maps for every forecast day, bias maps and scores on the days
    HKL has data. ens: raw ECENS of the same member. Returns the daily scores DataFrame (None without HKL days)."""
    with open(out_path(pred, "info", ext="json"), "w") as f:
        json.dump({"model": MODEL, "base": BASE_TAG, "pred": pred, "name": name,
                   "ecens_name": ens_name if ens is not None else None}, f)
    if SAVE_PANELS:
        save_swe_legend(out_path(pred, "swe-legend"))

    hkl = match_hkl(ecx, hkl_raw) if hkl_raw is not None else None
    hkl_times = hkl.valid_time.values if hkl is not None else np.array([], dtype="datetime64[ns]")
    times = ecx.valid_time.values
    print(f"{name}: {len(times)} forecast days, {int(np.isin(times, hkl_times).sum())} with HKL")

    lat = ecx.latitude.values
    lon = ecx.longitude.values
    dlat = abs(lat[1] - lat[0])
    dlon = abs(lon[1] - lon[0])
    extent = grid_extent(ecx)
    aspect = 1.0 / np.cos(np.deg2rad(lat.mean()))
    lines = get_lines(extent)
    tag = f"{MODEL}, base {BASE_TIME:%Y-%m-%d}"
    if ens is not None:
        ens_extent = grid_extent(ens)
        ens_lines = get_lines(ens_extent) if ECENS_OWN_EXTENT else lines
        ens_res = f"{abs(float(ens.latitude[1] - ens.latitude[0])):g}°"
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
    sum_d = np.zeros((lat.size, lon.size))
    sum_d2 = np.zeros_like(sum_d)
    cnt = np.zeros_like(sum_d)
    edges = np.linspace(0, 1500, 301)
    H = np.zeros((edges.size - 1, edges.size - 1))
    rows = []

    for t in times:
        ts = pd.Timestamp(t)
        lead = (ts - BASE_TIME).days
        s = ecx.sel(valid_time=t).values
        o = hkl.sel(valid_time=t).values if np.isin(t, hkl_times) else None
        e = ens.sel(valid_time=t).values if (ens is not None and np.isin(t, ens_times)) else None

        # ---- SWE maps, each producer in its own file
        if SAVE_PANELS:
            if e is not None:
                pv = extent if PANEL_ECENS_VIEW == "ecxens" else ens_extent
                save_panel(out_path(pred, "ecens-swe", ts), e, ens_extent, aspect,
                           lines if PANEL_ECENS_VIEW == "ecxens" else ens_lines,
                           f"{ens_name} raw SWE ({ens_res})\n{ts:%Y-%m-%d} (lead {lead} d)", pv)
            save_panel(out_path(pred, "ecxens-swe", ts), s, extent, aspect, lines,
                       f"{name} SWE ({MODEL})\n{ts:%Y-%m-%d} (lead {lead} d)", extent)
            if o is not None:
                save_panel(out_path(pred, "hkl-swe", ts), o, extent, aspect, lines, f"HKL SWE\n{ts:%Y-%m-%d}", extent)

        if SAVE_COMBINED:
            panels = []
            if e is not None:
                panels.append((e, ens_extent, ens_view, ens_lines, f"{ens_name} raw SWE ({ens_res}) {ts:%Y-%m-%d}"))
            panels.append((s, extent, extent, lines, f"{name} SWE {ts:%Y-%m-%d} (lead {lead} d)"))
            if o is not None:
                panels.append((o, extent, extent, lines, f"HKL SWE {ts:%Y-%m-%d}"))
            fig, axes = plt.subplots(1, len(panels), figsize=(5.7 * len(panels) + 0.6, 8), constrained_layout=True, squeeze=False)
            for ax, (data, ext, view, ln, title) in zip(axes[0], panels):
                im = draw_map(ax, data, swe_cmap, swe_norm, ext, aspect, ln, title, view=view)
            fig.colorbar(im, ax=axes[0], extend="both", ticks=SWE_BOUNDS, spacing="uniform", shrink=0.8, label="SWE (mm)")
            fig.savefig(out_path(pred, "compare-swe", ts), dpi=DPI)
            plt.close(fig)

        if o is None:
            print(f"{name} {ts:%Y-%m-%d} lead {lead:2d}  no HKL, maps only  ECX max {np.nanmax(s):.1f}")
            continue

        valid = np.isfinite(s) & np.isfinite(o)
        d = np.where(valid, s - o, np.nan)

        # ---- ECXENS bias map
        save_bias_map(out_path(pred, "ecxens-bias", ts), d, extent, aspect, lines,
                      f"{name} minus HKL ({MODEL})\n{ts:%Y-%m-%d} (lead {lead} d)")

        # ---- ECENS and ECXENS against HKL on the ECENS 0.1 degree grid
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
            save_bias_map(out_path(pred, "ecens-bias", ts), eb, ens_extent, aspect, lines,
                          f"{ens_name} minus HKL ({ens_res})\n{ts:%Y-%m-%d} (lead {lead} d)", view=extent)
            gtitle = f"|{name} bias| minus |{ens_name} bias|" if BIAS_COMPARE == "abs" else f"{name} bias minus {ens_name} bias"
            save_bias_map(out_path(pred, "ecxens-vs-ecens-gain", ts), gain, ens_extent, aspect, lines,
                          f"{gtitle} ({ens_res})\n{ts:%Y-%m-%d} (lead {lead} d)", view=extent, label=GAIN_LABEL)

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
        a2 = np.broadcast_to(area, s.shape)[valid]
        row["vol_ecx_km3"] = np.sum(sv.astype("float64") / 1000.0 * a2) / 1e9
        row["vol_hkl_km3"] = np.sum(ov.astype("float64") / 1000.0 * a2) / 1e9
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
        ens_txt = (f"  | 0.1deg RMSE ECENS {row['ecens_rmse']:.1f} ECXENS {row['ecx01_rmse']:.1f} skill {row['skill_mse_vs_ecens']:+.2f}"
                   if "ecens_rmse" in row else "")
        print(f"{name} {ts:%Y-%m-%d} lead {lead:2d}  bias {row['bias']:7.2f}  rmse {row['rmse']:7.2f}  "
              f"kge {row['kge']:5.2f}  csi_1 {row['csi_1']:.3f}  ECX max {np.nanmax(s):.1f}  HKL max {np.nanmax(o):.1f}{ens_txt}")

    if not rows:
        print(f"{name}: no days with HKL, forecast maps only")
        return None

    df = pd.DataFrame(rows)
    df.to_csv(out_path(pred, "ecxens-metrics", ext="csv"), index=False, float_format="%.4f")
    obs_times = [pd.Timestamp(r) for r in df["date"]]
    period = f"{obs_times[0]:%Y-%m-%d} to {obs_times[-1]:%Y-%m-%d}"

    # ---- mean bias and RMSE maps, separately
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_bias = np.where(cnt > 0, sum_d / cnt, np.nan)
        rmse_map = np.where(cnt > 0, np.sqrt(sum_d2 / cnt), np.nan)
    save_bias_map(out_path(pred, "ecxens-mean-bias"), mean_bias, extent, aspect, lines,
                  f"Mean bias {name} minus HKL ({MODEL})\n{period}", label="mm")
    fig, ax = plt.subplots(figsize=(7, 8), constrained_layout=True)
    im = draw_map(ax, rmse_map, rmse_cmap, rmse_norm, extent, aspect, lines, f"RMSE {name} ({MODEL})\n{period}")
    fig.colorbar(im, ax=ax, extend="max", ticks=rmse_bounds, spacing="uniform", shrink=0.8, label="mm")
    fig.savefig(out_path(pred, "ecxens-rmse"), dpi=DPI)
    plt.close(fig)

    # ---- mean ECENS bias and MAE change over the period, separately
    if ens is not None and cnt_c.any():
        with np.errstate(invalid="ignore", divide="ignore"):
            mean_eb = np.where(cnt_c > 0, sum_eb / cnt_c, np.nan)
            if BIAS_COMPARE == "abs":
                mae_gain = np.where(cnt_c > 0, (sum_ax - sum_ae) / cnt_c, np.nan)
            else:
                mae_gain = np.where(cnt_c > 0, (sum_xb - sum_eb) / cnt_c, np.nan)
        save_bias_map(out_path(pred, "ecens-mean-bias"), mean_eb, ens_extent, aspect, lines,
                      f"Mean bias {ens_name} minus HKL ({ens_res})\n{period}", view=extent, label="mm")
        mtitle = f"MAE {name} minus MAE {ens_name}" if BIAS_COMPARE == "abs" else f"Mean bias {name} minus mean bias {ens_name}"
        save_bias_map(out_path(pred, "ecxens-vs-ecens-mae-gain"), mae_gain, ens_extent, aspect, lines,
                      f"{mtitle} ({ens_res})\n{period}", view=extent,
                      label="mm, negative = ECXENS closer to HKL" if BIAS_COMPARE == "abs" else "mm")

    # ---- density scatter
    fig, ax = plt.subplots(figsize=(7, 6.5), constrained_layout=True)
    Hm = np.ma.masked_equal(H.T, 0)
    if Hm.count():
        pc = ax.pcolormesh(edges, edges, Hm, norm=LogNorm(vmin=1, vmax=Hm.max()), cmap="viridis")
        fig.colorbar(pc, ax=ax, label="pixel count")
    ax.plot([0, 1500], [0, 1500], "k--", lw=0.8)
    ax.set_xlabel("HKL SWE (mm)")
    ax.set_ylabel(f"{name} SWE (mm)")
    ax.set_title(f"Pixel density, all days pooled ({MODEL})\n{period} (0 to 1500 mm)")
    ax.set_aspect("equal")
    fig.savefig(out_path(pred, "ecxens-density-scatter"), dpi=DPI)
    plt.close(fig)

    # ---- scores against lead day, one figure each
    L = df["lead_day"]
    head = f"{name} vs HKL, {tag}"
    lead_plot(out_path(pred, "ecxens-score-bias-mae"), f"Bias and MAE\n{head}",
              [(L, df["bias"], "o-", "bias all"), (L, df["snow_bias"], "o--", "bias snow px"), (L, df["mae"], "s-", "MAE all")],
              ylabel="mm", zero=True)
    lead_plot(out_path(pred, "ecxens-score-rmse"), f"RMSE\n{head}",
              [(L, df["rmse"], "o-", "all"), (L, df["snow_rmse"], "o--", "snow px")], ylabel="mm")
    lead_plot(out_path(pred, "ecxens-score-kge"), f"Correlation and KGE\n{head}",
              [(L, df["r"], "o-", "r"), (L, df["kge"], "s-", "KGE"), (L, df["snow_kge"], "s--", "KGE snow px")])
    lead_plot(out_path(pred, "ecxens-score-csi"), f"Critical success index\n{head}",
              [(L, df[f"csi_{thr}"], "o-", f"CSI {thr} mm") for thr in SNOW_THRESHOLDS], ylim=(0, 1))
    lead_plot(out_path(pred, "ecxens-score-pod-far"), f"POD and FAR\n{head}",
              [s for thr in SNOW_THRESHOLDS for s in ((L, df[f"pod_{thr}"], "o-", f"POD {thr} mm"),
                                                      (L, df[f"far_{thr}"], "x--", f"FAR {thr} mm"))],
              ylim=(0, 1), legend_kw=dict(fontsize=8, ncol=2))
    lead_plot(out_path(pred, "ecxens-score-volume"), f"Total SWE volume (common valid pixels)\n{head}",
              [(L, df["vol_ecx_km3"], "o-", name), (L, df["vol_hkl_km3"], "s-", "HKL")], ylabel="km3 water")
    if "skill_mse_vs_ecens" in df:
        lead_plot(out_path(pred, "ecxens-vs-ecens-skill"), f"{name} against {ens_name} on {ens_res} cells\n{tag}",
                  [(L, df["skill_mse_vs_ecens"], "o-", "MSE skill vs ECENS (1 = perfect, 0 = as ECENS)"),
                   (L, df["frac_cells_improved"], "s--", "share of cells closer to HKL")], zero=True)

    print(f"{name} done, outputs in {os.path.dirname(out_path(pred, 'info', ext='json'))}/")
    return df


# ---------------------------------------------------------------- one pass over all members
def scan_members(ecx_src, ens_src, hkl_raw, want_ts, want_ecx_mean, want_ens_mean):
    """Reads every member once: ranking statistics, domain mean time series and the ens mean sums.
    Returns (ranking DataFrame, time series DataFrame or None, ECXENS ens mean or None, ECENS ens mean or None)."""
    ecx_m = set(ecx_src.members())
    ens_m = set(ens_src.members()) if ens_src is not None else set()
    todo = sorted(ecx_m | (ens_m if want_ens_mean else set()))
    rank_rows, ts_rows = [], []
    acc = {}
    hkl = hkl_mask = ci = cell_static = None
    hkl_times = np.array([], dtype="datetime64[ns]")
    hx_done, he_done = set(), set()

    def wmean(v, ok, w):
        return float(np.sum(v[ok] * w[ok]) / np.sum(w[ok])) if ok.any() else np.nan

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
            if hkl_raw is not None and hkl is None:
                hkl = match_hkl(ecx, hkl_raw)
                hkl_times = hkl.valid_time.values
                hkl_mask = np.isfinite(hkl.isel(valid_time=0).values)
            if ens is not None and ci is None:
                elat, elon = ens.latitude.values, ens.longitude.values
                cidx, cn_all = coarse_index(lat, lon, elat, elon)
                cshape = (elat.size, elon.size)
                we = np.broadcast_to(np.cos(np.deg2rad(elat))[:, None], cshape)
                ci = (cidx, cn_all, cshape)
                if hkl_mask is not None:
                    cell_static = np.isfinite(to_coarse(np.where(hkl_mask, 1.0, np.nan), *ci))
            for t in ecx.valid_time.values:
                ts = pd.Timestamp(t)
                s = ecx.sel(valid_time=t).values
                ok = np.isfinite(s) & (hkl_mask if hkl_mask is not None else True)
                ts_rows.append({"date": ts, "source": "ECXENS", "member": m, "swe_mm": wmean(s, ok, wx)})
                o = hkl.sel(valid_time=t).values if np.isin(t, hkl_times) else None
                if o is not None and ts not in hx_done:
                    ts_rows.append({"date": ts, "source": "HKL_ECXENS", "member": -1, "swe_mm": wmean(o, ok & np.isfinite(o), wx)})
                    hx_done.add(ts)
                if ens is not None and np.isin(t, ens.valid_time.values):
                    e = ens.sel(valid_time=t).values
                    cell_ok = cell_static if cell_static is not None else np.isfinite(to_coarse(s, *ci))
                    okc = np.isfinite(e) & cell_ok
                    ts_rows.append({"date": ts, "source": "ECENS", "member": m, "swe_mm": wmean(e, okc, we)})
                    if o is not None and ts not in he_done:
                        hc = to_coarse(o, *ci)
                        ts_rows.append({"date": ts, "source": "HKL_ECENS", "member": -1,
                                        "swe_mm": wmean(hc, okc & np.isfinite(hc), we)})
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
    """One plot per producer: members thin blue, ens mean of the members black, HKL red."""
    for src, hsrc, kind, what in [("ECXENS", "HKL_ECXENS", "ecxens-timeseries", "ECXENS (500 m, HKL domain)"),
                                  ("ECENS", "HKL_ECENS", "ecens-timeseries", "raw ECENS (0.1° cells, HKL averaged into the cells)")]:
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
            ax.plot(h["date"], h["swe_mm"], color="red", lw=2.5, marker="o", label="HKL")
        dates = sorted(d["date"].unique())
        ax.set_xticks(dates)
        ax.set_xticklabels([f"{pd.Timestamp(t):%m-%d}\n+{(pd.Timestamp(t) - BASE_TIME).days} d" for t in dates], fontsize=8)
        ax.set_xlabel("date and lead day")
        ax.set_ylabel("domain mean SWE (mm)")
        ax.set_title(f"{what}, all members vs HKL, base {BASE_TIME:%Y-%m-%d} ({MODEL})")
        ax.grid(alpha=0.3)
        ax.legend()
        p = out_path("members", kind)
        fig.savefig(p, dpi=DPI)
        plt.close(fig)
        print(f"time series plot -> {p}")


# ---------------------------------------------------------------- main
def main():
    os.makedirs(OUTDIR, exist_ok=True)
    print(f"model {MODEL}, base {BASE_TIME:%Y-%m-%d}, outputs in {OUTDIR}")

    hkl, hkl_files = load_hkl()
    ecx_src = find_source("ECXENS", [ECX_DIR], ECX_TEMPLATES, ECX_SCALE)
    if ecx_src is None:
        raise SystemExit(f"no ECXENS forecast for {BASE_TIME:%Y-%m-%d} in {ECX_DIR}")
    ens_src = find_source("ECENS", ECENS_DIRS, ECENS_TEMPLATES, ECENS_SCALE)

    # ---- one pass over the members if anything needs it
    rank_csv = out_path("members", "ranking", ext="csv")
    ts_csv = out_path("members", "timeseries", ext="csv")
    has_members = len(ecx_src.members()) > 0
    want_rank = has_members and ADD_EXTREME_MEMBERS and (RERANK or not os.path.exists(rank_csv))
    want_ts = has_members and MEMBER_TIMESERIES and (RETIMESERIES or not os.path.exists(ts_csv))
    want_ecx_mean = not ecx_src.has_mean_file
    want_ens_mean = ens_src is not None and not ens_src.has_mean_file
    rk = ts_df = ecx_mean = ens_mean = None
    if want_rank or want_ts or want_ecx_mean or want_ens_mean:
        print(f"\n=== reading all members (ranking {want_rank}, time series {want_ts}, "
              f"ECXENS ens mean {want_ecx_mean}, ECENS ens mean {want_ens_mean}) ===")
        rk, ts_df, ecx_mean, ens_mean = scan_members(ecx_src, ens_src, hkl, want_ts, want_ecx_mean, want_ens_mean)
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
        elif ts_df is not None:
            ts_df.to_csv(ts_csv, index=False, float_format="%.4f")
        if ts_df is not None:
            plot_member_timeseries(ts_df)

    # ---- predictions to validate: (label, name, ECXENS loader, ECENS loader, ECENS name)
    ens_members = set(ens_src.members()) if ens_src is not None else set()

    def ens_loader(m):
        if ens_src is None:
            return lambda: None
        if m == "ensmean":
            return lambda: ens_mean if ens_mean is not None else (ens_src.load("ensmean") if ens_src.has_mean_file else None)
        return lambda: ens_src.load(m) if m in ens_members else None

    preds = [("ensmean", "ECXENS ens mean" if has_members or ecx_src.split else "ECXENS",
              (lambda: ecx_mean) if ecx_mean is not None else (lambda: ecx_src.load("ensmean")),
              ens_loader("ensmean"), "ECENS ens mean")]
    members = ecx_src.members()
    if members:
        m1 = FIRST_MEMBER if FIRST_MEMBER in members else min(x for x in members if x > 0) if any(x > 0 for x in members) else members[0]
        if m1 != FIRST_MEMBER:
            print(f"ECXENS member {FIRST_MEMBER} missing, member {m1} validated instead")
        preds.append((f"member{m1:02d}", f"ECXENS member {m1}", (lambda m=m1: ecx_src.load(m)), ens_loader(m1), f"ECENS member {m1}"))
    if rk is not None and len(rk):
        for label, m in [("driest", int(rk["member"].iloc[0])), ("wettest", int(rk["member"].iloc[-1]))]:
            preds.append((f"{label}-member{m:02d}", f"ECXENS member {m} ({label})", (lambda m=m: ecx_src.load(m)),
                          ens_loader(m), f"ECENS member {m}"))
            print(f"{label} member by {RANK_BY}: {m}")

    with open(os.path.join(OUTDIR, f"{PREFIX}_run-info.json"), "w") as f:
        json.dump({"model": MODEL, "base_time": f"{BASE_TIME:%Y-%m-%d %H:%M}", "region": REGION,
                   "ecxens_file": os.path.basename(ecx_src.path), "ecxens_members": ecx_src.members(),
                   "ecens_file": os.path.basename(ens_src.path) if ens_src is not None else None,
                   "hkl_file": [os.path.basename(p) for p in hkl_files],
                   "preds": [p[0] for p in preds]}, f, indent=2)

    all_dfs = []
    for label, name, load_ecx, load_ens, ens_name in preds:
        print(f"\n=== {name} ===")
        ecx = load_ecx()
        ens = load_ens()
        if ens is not None:
            ens = check_ecens(ens, hkl)
        else:
            print(f"{ens_name}: not available, maps without it")
        df = validate(ecx, hkl, label, name, ens=ens, ens_name=ens_name)
        if df is not None:
            all_dfs.append(df.assign(pred=label))
        del ecx, ens

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        cols = ["pred"] + [c for c in combined.columns if c != "pred"]
        p = out_path("all", "ecxens-metrics", ext="csv")
        combined[cols].to_csv(p, index=False, float_format="%.4f")
        print(f"\nCombined scores: {p}")
    write_manifest()


if __name__ == "__main__":
    main()