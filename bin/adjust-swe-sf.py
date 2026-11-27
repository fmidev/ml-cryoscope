#!/usr/bin/env python3
"""
adjust-snow-ecxsf.py

Post-processing of the seasonal SWE forecast ECXSF (2.8 km, Scandinavia 0.025 deg grid), the seasonal twin of
adjust-snow-ecxens.py: the forecast is moved to the latest observed SNOWCAP (HKL) SWE state of the start day, or to
ERA5-Land snow with --source era5l, and the correction fades out linearly with lead time (DECAY_DAYS = 180, about six
months, so the last weeks of the 215 day forecast are the model's own). When there is no recent enough HKL observation
(--source auto), the HKL daily climatology (2015 to 2023) of the first forecast day is used instead.

    A        = start(base) - forecast(first forecast day)          one field per member, mm
    adjusted = max(0, forecast(t) + w(t) * g(t) * A)
    w(t)     = WEIGHT0 * max(0, 1 - days from the first forecast day / DECAY_DAYS)       (--decay 0: w = WEIGHT0 every day)
    g(t)     = melt guard, only for snow ADDED (A > 0): min(1, forecast(t) / forecast(first day)) where the forecast had
               at least GUARD_MIN mm on the first day, else 1. While the forecast keeps or builds its snow g = 1; when the
               forecast melts its own snow, the added snow melts with it, and at the forecast's melt-out it is gone, so a
               winter start does not leave added snow lying into the summer. --no-melt-guard: g = 1 (plain shift, as ECXENS).

--product clim (Mikko's idea from the xgb-ml-team channel, "current situation has to be the starting point for the
climatology"): no model at all, the HKL daily climatology of every forecast day plus the anomaly of the start day, fading out:
    anomaly  = start(base) - climatology(base day)                     one field, mm (HKL observation of the base date)
    clim_adj = max(0, climatology(t) + w(t) * g(t) * anomaly)          g: the same melt guard, against the climatology's own melt
Only the mean climatology is used (no min or max). One file for the whole run, not per member: the member file given on the
command line is only the template for the grid, the days and the GRIB headers. Without an HKL observation (--source auto
falls back to clim) the anomaly is zero and the product is the plain climatology. It is the "observed start, then the normal
year" reference the model forecasts can be compared with.
  python adjust-snow-ecxsf.py 202411 0 --product clim       -> ens/ECXSF_<YYYYMM>_swe-scandi-clim-adjusted.grib
  join name in grib/: ECXSF_<YYYYMM>01T060000_swe-scandi-clim-adjusted.grib (T060000 keeps it apart in SmartMet)

Start fields on the forecast grid:
  HKL (500 m, m w.e.) is averaged into the 2.8 km cells (mean of the valid 500 m pixels, at least MIN_COVER of a cell),
  ERA5-Land (9 km, m w.e.) and the remapped ERA5-Land file of the monthly run are taken at the nearest point.
  Pixels without a start value (sea, outside HKL) or without forecast keep the forecast as it is.

--source auto (default): HKL observation of the base date or at most MAX_HKL_AGE days before it, else the climatology.
--source obs:   only the observation (stops when there is none).
--source clim:  HKL_20200101T000000_2015-2023_SWE-ydaymean.grib (CLIM_FILES), the field with the month and day of the
                first forecast day.
--source era5l: ERA5-Land sd of the base date (or the newest day up to --lag days before it); by default the remapped file
                the monthly run already fetches for the sd lag (ens/era5l_sdwe_<YYYYMM>_neu-remap.grib), --era5l gives a file.

usage: python adjust-snow-ecxsf.py <YYYYMM> <member> [options]      one ensemble member (0..50) per run, with parallel
options: --product forecast|clim --source auto|obs|clim|era5l --era5l <file> --lag 6 --weight 1 --decay 180 --no-melt-guard --tag adjusted
         --hkl <file> --dir <folder> --in <member file name with {YM} and {m}>

Production (run-xgb-predict-swe-sf.sh, /home/users/smartmet/data, conda env xgb2), after the grib_set step:
  seq 0 50 | parallel -j10 python /home/users/smartmet/cryoscope-smart/bin/adjust-snow-ecxsf.py $year$month {}
  in:  ens/ECXSF_<YYYYMM>_swe-scandi-<m>-fix.grib                 ECXSF members after the GRIB header fix
  out: ens/ECXSF_<YYYYMM>_swe-scandi-<tag>-<m>.grib               every GRIB header of the input kept, only values change
  join: grib_copy ens/ECXSF_<YYYYMM>_swe-scandi-adjusted-{0..50}.grib grib/ECXSF_<YYYYMM>01T120000_swe-scandi-adjusted.grib
        (T120000 keeps it apart from the raw ECXSF in SmartMet; ERA5-Land adjusted: tag adjusted-era5l, T180000)
The start field is computed by the first member and kept in <source>-start-sf-<YYYY-MM-DD>.npz in the folder; the other
members read it.
"""
import os, glob, argparse
import numpy as np
import pandas as pd
import xarray as xr
import eccodes as ec

# ---------------------------------------------------------------- settings
DATA = '/home/ubuntu/data/cryoscope/snowmodel/'      # dev-destine
DIR = DATA + 'ECXSF/'                                 # ECXSF members (copied from firedanger ens/) and the outputs
IN_TEMPLATE = 'ECXSF_{YM}01T000000_swe-scandi-{m}.grib'          # daily members (not the -monmean files)
OUT_TEMPLATE = 'ECXSF_{YM}01T000000_swe-scandi-{tag}-{m}.grib'
OUT_CLIM_TEMPLATE = 'ECXSF_{YM}01T000000_swe-scandi-{tag}.grib'     # --product clim: one file for the run
# HKL (SNOWCAP) files, m w.e., 500 m; CHECK the folders on firedanger (copy the files there if they are only on smart)
HKL_DIRS = [DATA + 'SNOWCAP/', DATA + 'ECXSF/', DATA + 'ECXENS/', DATA, DATA + 'grib/']
HKL_TEMPLATES = ['HKL_20150101T000000_{Y}_SWE.grib', 'SNOWCAP_20150101T000000_{Y}_SWE.grib', 'HKL_{Y}_SWE.grib', 'SNOWCAP_{Y}_SWE.grib']
HKL_MONTH_TEMPLATES = ['HKL_{YM}_SWE.grib', 'SNOWCAP_{YM}_SWE.grib', 'HKL_{YM}_swe.grib', 'SNOWCAP_{YM}_swe.grib']
HKL_DAY_GLOBS = ['HKL_*{D}*SWE*.grib', 'SNOWCAP_*{D}*SWE*.grib', 'HKL_*{D}*swe*.grib', 'SNOWCAP_*{D}*swe*.grib']
CLIM_FILES = ['HKL_20200101T000000_2015-2023_SWE-ydaymean.grib']   # daily climatology, one field per day of 2020
HKL_SCALE = 1000.0       # Scandinavian HKL SWE is m w.e., x1000 gives mm like the forecast
MAX_HKL_AGE = 5          # days: newest HKL day at most this many days before the base date
MIN_COVER = 0.5          # share of valid 500 m HKL pixels a 2.8 km cell needs for a start value
# ERA5-Land sd (m w.e.) for --source era5l: the remapped file of the monthly run first, then the raw one
ERA5L_TEMPLATES = ['era5l_sdwe_{YM}_neu-remap.grib', 'era5l_sdwe_{YM}_neu.grib']
ERA5L_DIRS = [DATA + 'ECXSF/', DATA + 'ERA5L/', DATA]
ERA5L_LAG = 6            # days: newest ERA5-Land day accepted at most this many days before the base date
ERA5L_SCALE = 1000.0     # m w.e. -> mm
WEIGHT0 = 1.0            # share of the offset applied on the first forecast day: 1 = all the way to the start field
DECAY_DAYS = 180         # days until the correction is gone (about 6 months); 0: no fading
GUARD_MIN = 1.0          # mm: the melt guard needs at least this much forecast snow on the first day

ap = argparse.ArgumentParser(description='move the seasonal SWE forecast (ECXSF 2.8 km) to the observed state of the start day')
ap.add_argument('month', help='forecast start month YYYYMM (base date the 1st, 00 UTC)')
ap.add_argument('member', type=int, help='ensemble member number (0..50)')
ap.add_argument('--product', choices=['forecast', 'clim'], default='forecast',
                help='forecast: adjust the ECXSF member (default); clim: HKL climatology plus the fading start anomaly (one file)')
ap.add_argument('--source', choices=['auto', 'obs', 'clim', 'era5l'], default='auto',
                help='auto: latest HKL observation, climatology when there is none (default); obs: observation only; '
                     'clim: climatology only; era5l: ERA5-Land sd of the base date')
ap.add_argument('--era5l', default=None, help='ERA5-Land file holding sd of the start day (default: ERA5L_TEMPLATES)')
ap.add_argument('--lag', type=int, default=ERA5L_LAG, help='ERA5-Land: oldest accepted day, days before the base date')
ap.add_argument('--tag', default=None, help='name in the output files (default adjusted, adjusted-era5l for --source era5l)')
ap.add_argument('--weight', type=float, default=WEIGHT0, help='share of the offset on the first forecast day (0 to 1)')
ap.add_argument('--decay', type=float, default=DECAY_DAYS, help='days until the correction is zero, linear; 0: no fading')
ap.add_argument('--no-melt-guard', dest='guard', action='store_false', help='plain shift, added snow not tied to the forecast melt')
ap.add_argument('--dir', default=DIR)
ap.add_argument('--in', dest='intmpl', default=IN_TEMPLATE, help='member file name with {YM} and {m}')
ap.add_argument('--hkl', default=None, help='HKL file holding the start day (or the climatology file with --source clim)')
args = ap.parse_args()
if not 0.0 <= args.weight <= 1.0:
    raise SystemExit(f'--weight {args.weight}: give a share from 0 to 1')
tag = args.tag or ('clim-adjusted' if args.product == 'clim' else 'adjusted-era5l' if args.source == 'era5l' else 'adjusted')

YM = args.month
BASE = pd.Timestamp(f'{YM[:4]}-{YM[4:6]}-01')
d = os.path.abspath(args.dir) + '/'


# ---------------------------------------------------------------- helpers
def first_field(path):
    """Field of the first valid time of a GRIB file, in the file's own grid order, and that valid time."""
    ds = xr.open_dataset(path, engine='cfgrib', backend_kwargs={'indexpath': ''})
    da = ds[list(ds.data_vars)[0]]
    tdims = [x for x in da.dims if x not in ('latitude', 'longitude')]
    vt = np.asarray(da['valid_time'].values) if 'valid_time' in da.coords else None
    if tdims:
        if vt is not None and vt.ndim == len(tdims):
            vi = vt.astype('datetime64[ns]').astype('int64')
            vi = np.where(vi == np.iinfo(np.int64).min, np.iinfo(np.int64).max, vi)   # NaT last
            i = np.unravel_index(int(np.argmin(vi)), vt.shape)
            da = da.isel({t: int(k) for t, k in zip(tdims, i)})
        else:
            da = da.isel({t: 0 for t in tdims})
    da = da.squeeze([x for x in da.dims if x not in ('latitude', 'longitude') and da.sizes[x] == 1])
    t0 = pd.Timestamp(np.asarray(da['valid_time'].values).ravel()[0]).normalize()
    return da.load(), t0


def pick_day(path, days, scale, short_name=None):
    """Field (mm) of the first of `days` found in a GRIB file (any time layout), that day; None when none is there."""
    kw = {'indexpath': ''}
    if short_name:
        kw['filter_by_keys'] = {'shortName': short_name}
    try:
        ds = xr.open_dataset(path, engine='cfgrib', backend_kwargs=kw)
    except Exception as ex:
        print(f'{os.path.basename(path)}: not read ({ex.__class__.__name__})')
        return None, None
    if not ds.data_vars:
        return None, None
    da = ds[short_name] if short_name in ds.data_vars else ds[list(ds.data_vars)[0]]
    if 'valid_time' not in da.coords:
        return None, None
    v = da['valid_time']
    vt = pd.DatetimeIndex(np.asarray(v.values).ravel()).normalize()
    for day in days:
        hit = np.where(vt == day)[0]
        if hit.size:
            pos = np.unravel_index(int(hit[0]), v.shape) if v.ndim else ()
            h = da.isel({dim: int(i) for dim, i in zip(v.dims, pos)})
            h = h.squeeze([x for x in h.dims if x not in ('latitude', 'longitude') and h.sizes[x] == 1])
            return h.sortby('latitude').load() * scale, day
    return None, None


def hkl_start(base):
    """HKL SWE (mm, 500 m) of the base date or the newest day before it (at most MAX_HKL_AGE days), that day and the file."""
    days = [base - pd.Timedelta(days=k) for k in range(MAX_HKL_AGE + 1)]
    files = [args.hkl] if args.hkl else []
    for day in days:   # files named by the day, newest day first
        files += sorted(f for h in HKL_DIRS for g in HKL_DAY_GLOBS for f in glob.glob(h + g.format(D=f'{day:%Y%m%d}')))
    for ym in dict.fromkeys(f'{x:%Y%m}' for x in days):
        files += [h + t.format(YM=ym) for h in HKL_DIRS for t in HKL_MONTH_TEMPLATES]
    for y in sorted({x.year for x in days}, reverse=True):
        files += [h + t.format(Y=y) for h in HKL_DIRS for t in HKL_TEMPLATES]
    tried = [f for f in dict.fromkeys(files) if f and os.path.exists(f)]
    for f in tried:
        h, day = pick_day(f, days, HKL_SCALE)
        if h is not None:
            return h, day, f
    raise SystemExit(f'no HKL SWE for {base:%Y-%m-%d} or the {MAX_HKL_AGE} days before it in {", ".join(HKL_DIRS)}; '
                     f'files with other days: {", ".join(os.path.basename(f) for f in tried) or "none"}')


def hkl_clim(day):
    """HKL SWE climatology (mm, 500 m) of the month and day of `day`, the date of that field and the file.
    Read with eccodes by each message's own validityDate (the ydaymean file is not one time axis for cfgrib)."""
    files = [args.hkl] if args.hkl else [h + c for h in HKL_DIRS for c in CLIM_FILES]
    files = [f for f in dict.fromkeys(files) if f and os.path.exists(f)]
    if not files:
        raise SystemExit(f'no HKL climatology file ({", ".join(CLIM_FILES)}) in {", ".join(HKL_DIRS)}')
    want = [(day.month, day.day)] + ([(2, 28)] if (day.month, day.day) == (2, 29) else [])
    for f in files:
        with open(f, 'rb') as fh:
            while True:
                gid = ec.codes_grib_new_from_file(fh)
                if gid is None:
                    break
                try:
                    vd = str(ec.codes_get(gid, 'validityDate'))
                    if (int(vd[4:6]), int(vd[6:8])) not in want:
                        continue
                    ni, nj = ec.codes_get(gid, 'Ni'), ec.codes_get(gid, 'Nj')
                    vals = ec.codes_get_values(gid).astype('float64')
                    if ec.codes_get(gid, 'bitmapPresent'):
                        vals[vals == ec.codes_get(gid, 'missingValue')] = np.nan
                    lat = np.linspace(ec.codes_get(gid, 'latitudeOfFirstGridPointInDegrees'),
                                      ec.codes_get(gid, 'latitudeOfLastGridPointInDegrees'), nj)
                    lon = np.linspace(ec.codes_get(gid, 'longitudeOfFirstGridPointInDegrees'),
                                      ec.codes_get(gid, 'longitudeOfLastGridPointInDegrees'), ni)
                    h = xr.DataArray(vals.reshape(nj, ni), dims=('latitude', 'longitude'),
                                     coords={'latitude': lat, 'longitude': lon}).sortby('latitude')
                    return h * HKL_SCALE, pd.Timestamp(vd), f
                finally:
                    ec.codes_release(gid)
    raise SystemExit(f'no HKL climatology for {day:%m-%d} in {", ".join(os.path.basename(f) for f in files)}')


def clim_days(keys, lat, lon):
    """HKL climatology (mm) aggregated to the forecast grid for every (month, day) in keys: {(month, day): field}.
    The file is read once with eccodes, every message by its own validityDate; 29 Feb falls back to 28 Feb."""
    files = [args.hkl] if (args.hkl and args.source == 'clim') else [h + c for h in HKL_DIRS for c in CLIM_FILES]
    files = [f for f in dict.fromkeys(files) if f and os.path.exists(f)]
    if not files:
        raise SystemExit(f'no HKL climatology file ({", ".join(CLIM_FILES)}) in {", ".join(HKL_DIRS)}')
    want = set(keys) | ({(2, 28)} if (2, 29) in keys else set())
    out, cells = {}, None
    with open(files[0], 'rb') as fh:
        while True:
            gid = ec.codes_grib_new_from_file(fh)
            if gid is None:
                break
            try:
                vd = str(ec.codes_get(gid, 'validityDate'))
                k = (int(vd[4:6]), int(vd[6:8]))
                if k not in want or k in out:
                    continue
                ni, nj = ec.codes_get(gid, 'Ni'), ec.codes_get(gid, 'Nj')
                vals = ec.codes_get_values(gid).astype('float64')
                if ec.codes_get(gid, 'bitmapPresent'):
                    vals[vals == ec.codes_get(gid, 'missingValue')] = np.nan
                la = np.linspace(ec.codes_get(gid, 'latitudeOfFirstGridPointInDegrees'),
                                 ec.codes_get(gid, 'latitudeOfLastGridPointInDegrees'), nj)
                lo = np.linspace(ec.codes_get(gid, 'longitudeOfFirstGridPointInDegrees'),
                                 ec.codes_get(gid, 'longitudeOfLastGridPointInDegrees'), ni)
                if cells is None:
                    cells = cell_index(la, lo, lat, lon)
                out[k] = cells(vals.reshape(nj, ni) * HKL_SCALE).astype('float32')
            finally:
                ec.codes_release(gid)
    if (2, 29) in keys and (2, 29) not in out and (2, 28) in out:
        out[(2, 29)] = out[(2, 28)]
    miss = sorted(set(keys) - set(out))
    if miss:
        raise SystemExit(f'{os.path.basename(files[0])}: no climatology for {", ".join(f"{a:02d}-{b:02d}" for a, b in miss[:5])}'
                         + (' ...' if len(miss) > 5 else ''))
    print(f'HKL climatology of {len(out)} days from {os.path.basename(files[0])} on the forecast grid')
    return out


def era5l_start(base):
    """ERA5-Land sd (mm) of the base date or the newest day up to args.lag days before it, that day and the file."""
    days = [base - pd.Timedelta(days=k) for k in range(args.lag + 1)]
    files = [args.era5l] if args.era5l else [h + t.format(YM=YM) for h in ERA5L_DIRS for t in ERA5L_TEMPLATES]
    files = [f for f in dict.fromkeys(files) if f and os.path.exists(f)]
    for f in files:
        h, day = pick_day(f, days, ERA5L_SCALE, short_name='sd')
        if h is None:
            h, day = pick_day(f, days, ERA5L_SCALE)    # cdo remapped files may lose the shortName
        if h is not None:
            return h, day, f
    raise SystemExit(f'no ERA5-Land sd for {base:%Y-%m-%d} or the {args.lag} days before it in '
                     f'{", ".join(os.path.basename(f) for f in files) or ", ".join(t.format(YM=YM) for t in ERA5L_TEMPLATES)}; give it with --era5l')


def cell_index(slat, slon, lat, lon):
    """Function taking a field on the (finer) src grid slat x slon and returning the mean of its valid pixels in every cell
    of the forecast grid (lat, lon: cell centres, any order); NaN where fewer than MIN_COVER of a cell's pixels are valid.
    The pixel to cell index is made once, so many fields on the same grid are fast."""
    dlat, dlon = float(lat[1] - lat[0]), float(lon[1] - lon[0])
    iy = np.rint((np.asarray(slat) - lat[0]) / dlat).astype(np.int64)
    ix = np.rint((np.asarray(slon) - lon[0]) / dlon).astype(np.int64)
    oky, okx = (iy >= 0) & (iy < lat.size), (ix >= 0) & (ix < lon.size)
    idx = np.where(oky[:, None] & okx[None, :], iy[:, None] * lon.size + ix[None, :], -1)
    inside = idx >= 0
    n_all = np.bincount(idx[inside], minlength=lat.size * lon.size)

    def cells(v):
        ok = np.isfinite(v) & inside
        ssum = np.bincount(idx[ok], weights=v[ok].astype('float64'), minlength=n_all.size)
        cnt = np.bincount(idx[ok], minlength=n_all.size)
        with np.errstate(invalid='ignore', divide='ignore'):
            m = np.where((n_all > 0) & (cnt >= MIN_COVER * n_all), ssum / np.maximum(cnt, 1), np.nan)
        return m.reshape(lat.size, lon.size)
    return cells


def to_cells(src, lat, lon):
    """A field (DataArray) on the forecast grid: cell means of a finer src (see cell_index), nearest point of a coarser one."""
    slat, slon = src['latitude'].values, src['longitude'].values
    if abs(float(slat[1] - slat[0])) >= 0.9 * abs(float(lat[1] - lat[0])):   # not finer than the forecast grid
        return src.sel(latitude=xr.DataArray(lat, dims='y'), longitude=xr.DataArray(lon, dims='x'), method='nearest').values
    return cell_index(slat, slon, lat, lon)(src.values)


def weight(days_from_first):
    if args.decay <= 0:
        return args.weight
    return args.weight * max(0.0, 1.0 - days_from_first / args.decay)


def write_clim(src, dst, A, clim, t0):
    """--product clim: every GRIB message of src (only its headers and valid days are used) replaced by
    max(0, climatology(day) + w(t) * g(t) * A), g the melt guard against the climatology's own melt; the climatology of
    the first forecast day is the guard's reference."""
    n = 0
    c1 = clim[(t0.month, t0.day)].ravel().astype('float64')
    guard = args.guard & (A > 0) & np.isfinite(c1) & (c1 >= GUARD_MIN)
    with open(src, 'rb') as fin, open(dst, 'wb') as fout:
        while True:
            gid = ec.codes_grib_new_from_file(fin)
            if gid is None:
                break
            try:
                vday = pd.Timestamp(str(ec.codes_get(gid, 'validityDate')))
                c = clim[(vday.month, vday.day)].ravel().astype('float64')
                if c.size != A.size:
                    raise SystemExit(f'{os.path.basename(src)}: {ec.codes_get(gid, "numberOfValues")} values in a message, '
                                     f'the climatology on the forecast grid has {c.size}')
                w = weight((vday - t0).days)
                g = np.ones_like(A)
                if guard.any():
                    with np.errstate(invalid='ignore', divide='ignore'):
                        g[guard] = np.clip(c[guard] / c1[guard], 0.0, 1.0)
                out = np.maximum(c + w * g * A, 0.0)
                miss = ec.codes_get(gid, 'missingValue')
                bad = ~np.isfinite(out)
                if bad.any():
                    ec.codes_set(gid, 'bitmapPresent', 1)
                    ec.codes_set(gid, 'missingValue', miss)
                    out = np.where(bad, miss, out)
                ec.codes_set_values(gid, out)
                ec.codes_write(gid, fout)
                n += 1
            finally:
                ec.codes_release(gid)
    return n


def adjust_member(src, dst, A, f1, t0):
    """Every GRIB message of src (one per day) plus w(t) * g(t) * A, below zero set to zero, missing values kept."""
    n = 0
    add = A > 0
    guard = args.guard & add & np.isfinite(f1) & (f1 >= GUARD_MIN)
    with open(src, 'rb') as fin, open(dst, 'wb') as fout:
        while True:
            gid = ec.codes_grib_new_from_file(fin)
            if gid is None:
                break
            try:
                vals = ec.codes_get_values(gid).astype('float64')
                if vals.size != A.size:
                    raise SystemExit(f'{os.path.basename(src)}: {vals.size} values in a message, the start field has {A.size}')
                miss = ec.codes_get(gid, 'missingValue')
                bitmap = ec.codes_get(gid, 'bitmapPresent')
                bad = (vals == miss) if bitmap else ~np.isfinite(vals)
                vday = pd.Timestamp(str(ec.codes_get(gid, 'validityDate')))
                w = weight((vday - t0).days)
                g = np.ones_like(A)
                if guard.any():
                    with np.errstate(invalid='ignore', divide='ignore'):
                        g[guard] = np.clip(vals[guard] / f1[guard], 0.0, 1.0)
                out = np.where(bad, vals, np.maximum(vals + w * g * A, 0.0))
                if bitmap:
                    ec.codes_set(gid, 'missingValue', miss)
                ec.codes_set_values(gid, out)
                ec.codes_write(gid, fout)
                n += 1
            finally:
                ec.codes_release(gid)
    return n


# ---------------------------------------------------------------- main
m = args.member
src = d + args.intmpl.format(YM=YM, m=m)
if not os.path.exists(src):
    raise SystemExit(f'no member file {src}')
dst = d + (OUT_CLIM_TEMPLATE.format(YM=YM, tag=tag) if args.product == 'clim' else OUT_TEMPLATE.format(YM=YM, tag=tag, m=m))
f1da, t0 = first_field(src)
lat, lon = f1da['latitude'].values, f1da['longitude'].values

# start field on the forecast grid, made once and shared by the members (temporary name, then renamed, so a member
# running at the same time never reads half a file)
cache = d + (f'era5l-start-sf-{BASE:%Y-%m-%d}.npz' if args.source == 'era5l' else f'hkl-{args.source}-start-sf-{BASE:%Y-%m-%d}.npz')
if os.path.exists(cache) and not args.hkl and not args.era5l:
    z = np.load(cache)
    start, sday, sfile, src_used = z['start'], pd.Timestamp(str(z['day'])), str(z['file_used']), str(z['source'])
    if start.shape != f1da.shape:
        raise SystemExit(f'{cache}: grid {start.shape}, member {m} has {f1da.shape}; delete the cache file')
else:
    src_used = args.source
    if args.source in ('auto', 'obs'):
        try:
            s, sday, sfile = hkl_start(BASE)
            src_used = 'obs'
        except SystemExit as ex:
            if args.source == 'obs':
                raise
            print(f'no HKL observation for the start ({ex}); using the HKL climatology')
            src_used = 'clim'
    if src_used == 'clim':
        s, sday, sfile = hkl_clim(t0)
    if src_used == 'era5l':
        s, sday, sfile = era5l_start(BASE)
    start = to_cells(s, lat, lon).astype('float32')
    tmp = cache + f'.{os.getpid()}.npz'
    np.savez(tmp, start=start, day=f'{sday:%Y%m%d}', file_used=sfile, source=src_used)
    os.replace(tmp, cache)
    what = {'clim': f'HKL SWE climatology of {sday:%m-%d}', 'era5l': f'ERA5-Land sd of {sday:%Y-%m-%d}'}.get(
        src_used, f'HKL SWE observation of {sday:%Y-%m-%d}')
    print(f'{what} ({os.path.basename(sfile)}) on the forecast grid: mean {np.nanmean(start):.1f} mm, max {np.nanmax(start):.0f} mm, '
          f'{100 * np.isfinite(start).mean():.1f} % of the grid -> {os.path.basename(cache)}')
    if np.nanmax(start) < 20:
        print('WARNING: start SWE max below 20 mm: early season, or not m w.e. (check HKL_SCALE / ERA5L_SCALE)')
if src_used in ('obs', 'era5l') and sday >= t0:
    raise SystemExit(f'start day {sday:%Y-%m-%d} is not before the first forecast day {t0:%Y-%m-%d}: '
                     'that would use observations the forecast can not have')

if args.product == 'clim':
    # anomaly of the start day against the climatology of the same day; the climatology of every forecast day
    vdays = []
    with open(src, 'rb') as fh:
        while True:
            gid = ec.codes_grib_new_from_file(fh)
            if gid is None:
                break
            vdays.append(pd.Timestamp(str(ec.codes_get(gid, 'validityDate'))))
            ec.codes_release(gid)
    keys = sorted({(x.month, x.day) for x in vdays + [sday]})
    clim = clim_days(keys, lat, lon)
    if src_used == 'clim':
        A = np.zeros(f1da.size)
        print('no HKL observation for the start: anomaly 0, the product is the plain climatology')
    else:
        A = start - clim[(sday.month, sday.day)]
        A = np.where(np.isfinite(A), A, 0.0).ravel()
    n = write_clim(src, dst, A, clim, t0)
    nz = A != 0
    print(f'climatology adjusted to the start {sday:%Y-%m-%d} ({src_used}): anomaly mean '
          + (f'{A[nz].mean():+.1f} mm, 5 to 95 %: {np.percentile(A[nz], 5):+.0f} to {np.percentile(A[nz], 95):+.0f} mm' if nz.any() else '0')
          + f', weights {weight(0):.3f} to {weight(n - 1):.3f}, melt guard {"on" if args.guard else "off"}, {n} days -> {os.path.basename(dst)}')
    raise SystemExit(0)

f1 = f1da.values.astype('float64')
A = start - f1
A = np.where(np.isfinite(A), A, 0.0).ravel()          # no start value or no forecast: no correction
f1 = f1.ravel()
n = adjust_member(src, dst, A, f1, t0)
nz = A != 0
label = {'era5l': 'ERA5-Land', 'clim': 'HKL climatology', 'obs': 'HKL obs'}.get(src_used, src_used)
if nz.any():
    print(f'member {m:2d}: start {label} {sday:%m-%d} minus forecast {t0:%m-%d}, offset mean {A[nz].mean():+.1f} mm, '
          f'|offset| {np.abs(A[nz]).mean():.1f} mm (5 to 95 %: {np.percentile(A[nz], 5):+.0f} to {np.percentile(A[nz], 95):+.0f}), '
          f'weights {weight(0):.3f} to {weight(n - 1):.3f}, melt guard {"on" if args.guard else "off"}, {n} days -> {os.path.basename(dst)}')
else:
    print(f'member {m:2d}: no pixel with both a start value and forecast, written unchanged -> {os.path.basename(dst)}')