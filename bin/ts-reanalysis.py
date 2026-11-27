#!/usr/bin/env python3
"""
Fetches the reanalysis time series of the SWE model locations from SmartMet servers (timeseries plugin), the sources and
features chosen in a yml file
"""
import os, sys, json
import requests
import pandas as pd
import numpy as np
from urllib.parse import quote_plus
try:
    import yaml
except ImportError:
    raise SystemExit('PyYAML is missing: conda install pyyaml (or pip install pyyaml)')

# ---------------------------------------------------------------- settings
TAGS = ('training', 'validation')
path = '/home/users/smartmet/data/snowmodel/'
REGION = 'scandi_alps'
BATCH = 5000             # locations per request, unless the source or the yml (batch: 2000) gives its own
DEFAULT_HOURS = '0'      # hour= of the request: 00 UTC
DEFAULT_KIND = 'daily'   # last part of the output file names
EMPTY_MARKER = False     # True: a year without data is saved as an empty checkpoint, not fetched again

# period: covers both domains (Scandinavia 2015-09 to 2023-09, Alps 2016-09 to 2024-09)
start = '20150830T000000'
end = '20240901T000000'

# sources: url, product name in the output files and the FMI keys of their features (an empty key = not known yet)
SOURCES = {
    'era5l': dict(url='http://ml.harvesterseasons.com:8080', product='ERA5L', batch=5000, keys={
        'fal': 'ALBEDOSLR-0TO1:ERA5L:5099:1:0:1:0',   # forecast albedo
        'asn': 'ASN-0TO1:ERA5L:5099:1:0:1:0',         # snow albedo
        'es': 'ES-M:ERA5L:5099:1:0:1:0',              # snow evaporation
        'e': 'EVAP-M:ERA5L:5099:1:0:1:0',             # evaporation
        'pev': 'EVAPP-M:ERA5L:5099:1:0:1:0',          # potential evaporation
        'slhf': 'FLLAT-JM2:ERA5L:5099:1:0:1:0',       # surface latent heat flux
        'sshf': 'FLSEN-JM2:ERA5L:5099:1:0:1:0',       # surface sensible heat flux
        'sd': 'HSNOW-M:ERA5L:5099:1:0:1:0',           # snow depth water equivalent (m)
        'laihv': 'LAI_HV-M2M2:ERA5L:5099:1:0:1:0',    # leaf area index, high vegetation
        'lailv': 'LAI_LV-M2M2:ERA5L:5099:1:0:1:0',    # leaf area index, low vegetation
        'licd': 'LICD-M:ERA5L:5099:1:0:1:0',          # lake ice total depth
        'lict': 'LICT-K:ERA5L:5099:1:0:1:0',          # lake ice surface temperature
        'sp': 'PGR-PA:ERA5L:5099:1:0:1:0',            # surface pressure
        'ssrd': 'RADGLOA-JM2:ERA5L:5099:1:0:1:0',
        'strd': 'RADLWA-JM2:ERA5L:5099:1:0:1:0',
        'str': 'RNETLWA-JM2:ERA5L:5099:1:0:1:0',
        'ssr': 'RNETSWA-JM2:ERA5L:5099:1:0:1:0',
        'ro': 'RO-M:ERA5L:5099:1:0:1:0',              # runoff
        'skt': 'SKT-K:ERA5L:5099:1:0:1:0',            # skin temperature
        'smlt': 'SMLT-M:ERA5L:5099:1:0:1:0',          # snowmelt
        'sf': 'SNACC-KGM2:ERA5L:5099:1:0:1:0',        # snowfall
        'rsn': 'SND-KGM3:ERA5L:5099:1:0:1:0',         # snow density
        'snowc': 'SNOWC-PRCNT:ERA5L:5099:1:0:0',      # snow cover
        'swvl1': 'SOILWET-M3M3:ERA5L:5099:9:7:1:0',   # volumetric soil water, layer 1
        'swvl2': 'SWVL2-M3M3:ERA5L:5099:9:1820:1:0',
        'swvl3': 'SWVL3-M3M3:ERA5L:5099:9:7268:1:0',
        'swvl4': 'SWVL4-M3M3:ERA5L:5099:9:25855:1:0',
        'src': 'SRC-M:ERA5L:5099:1:0:1:0',            # skin reservoir content
        'sro': 'SRO-M:ERA5L:5099:1:0:1:0',
        'ssro': 'SSRO-M:ERA5L:5099:1:0:1:0',
        'stl1': 'STL1-K:ERA5L:5099:9:7:1:0',          # soil temperature, layer 1
        't2': 'T2-K:ERA5L:5099:1:0:1:0',              # 2 m temperature
        'td2': 'TD2-K:ERA5L:5099:1:0:1:0',            # 2 m dewpoint temperature
        'tsn': 'TSN-K:ERA5L:5099:1:0:1:0',            # temperature of the snow layer
        'u10': 'U10-MS:ERA5L:5099:1:0:1:0',
        'v10': 'V10-MS:ERA5L:5099:1:0:1:0',
        'tp': 'RR-M:ERA5L:5099:1:0:1:0',              # total precipitation
    }),
    'era5ld': dict(url='http://sm.cryo-scope.eu:8080', product='ERA5LD', batch=2000, keys={
        'tmax': 'T2-K:ERA5LD:5093:1:0:1',             # daily 2 m temperature statistics
    }),
    'era5+l': dict(url='http://sm.cryo-scope.eu:8080', product='ERA5+L', batch=2000, keys={
        'fal': 'ALBEDOSLR-0TO1:ERA5+L:5109:1:0:1',
        'asn': 'ASN-0TO1:ERA5+L:5109:1:0:1',
        'laihv': 'LAI_HV-M2M2:ERA5+L:5109:1:0:1',
        'lailv': 'LAI_LV-M2M2:ERA5+L:5109:1:0:1',
        'sp': 'PGR-PA:ERA5+L:5109:1:0:1',
        'skt': 'SKT-K:ERA5+L:5109:1:0:1',
        'swvl1': 'SOILWET-M3M3:ERA5+L:5109:9:7:1',
        'swvl2': 'SWVL2-M3M3:ERA5+L:5109:9:1820:1',
        'swvl3': 'SWVL3-M3M3:ERA5+L:5109:9:7268:1',
        'swvl4': 'SWVL4-M3M3:ERA5+L:5109:9:25600:1',
        'src': 'SRC-M:ERA5+L:5109:1:0:1',
        'stl2': 'TSOIL-K:ERA5+L:5109:9:1820:1',
        'stl3': 'STL3-K:ERA5+L:5109:9:7268:1',
        'stl4': 'STL4-K:ERA5+L:5109:9:25600:1',
        'tsn': 'TSN-K:ERA5+L:5109:1:0:1',
        'u10': 'U10-MS:ERA5+L:5109:1:0:1',
        'v10': 'V10-MS:ERA5+L:5109:1:0:1',
        'sd': 'SD-M:ERA5+L:5109:1:0:1',
        'rsn': 'SND-KGM3:ERA5+L:5109:1:0:1',
        'stl1': 'STL1-K:ERA5+L:5109:9:7:1',
        't2': 'T2-K:ERA5+L:5109:1:0:1',
        # keys not known yet: give them here or in the yml (feature: key)
        'e': '', 'slhf': '', 'sshf': '', 'str': '', 'ssr': '', 'ro': '', 'sf': '', 'tp': '', 'tmax': '',
    }),
    'era5': dict(url='http://smartmet.xyz:8080', product='ERA5', batch=2000, keys={
        **{f'{v}{p}': f'{k}:ERA5:5021:2:{p}:1:0' for v, k in
           [('t', 'T-K'), ('z', 'Z-M2S2'), ('u', 'U-MS'), ('v', 'V-MS'), ('q', 'Q-KGKG')] for p in (500, 700, 850)},
    }),
}

# ---------------------------------------------------------------- command line and yml
args = sys.argv[1:]
TAG = args[0] if args else ''
if TAG not in TAGS:
    raise SystemExit(f'usage: python {os.path.basename(sys.argv[0])} <{"|".join(TAGS)}> [config.yml] [source ...]')
rest = args[1:]
CONFIG = rest.pop(0) if rest and rest[0].endswith(('.yml', '.yaml')) else \
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'snowmodel-features.yml')
ONLY = rest

lonlats_file = f'{path}snowcap_scandi_alps_{TAG}_lonlats.txt'
output_dir = f'{path}{TAG}_inputs/reanalysis/'
yearly_dir = os.path.join(output_dir, 'yearly')


def parse_features(v):
    """features of a source in the yml -> {name: key or None}. Accepts a list (- e), a mapping (e: <key>), a list of
    mappings, or names on separate lines without dashes (yml reads those as one string)."""
    if v is None:
        return {}
    if isinstance(v, str):
        return {n: None for n in v.replace(',', ' ').split()}
    if isinstance(v, dict):
        return {str(k): (str(x) if x else None) for k, x in v.items()}
    out = {}
    for item in v:
        if isinstance(item, dict):
            out.update({str(k): (str(x) if x else None) for k, x in item.items()})
        else:
            out.update({n: None for n in str(item).replace(',', ' ').split()})
    return out


def read_config(p):
    """{source: dict(url, product, origintime, hours, kind, keys={name: FMI key})}; stops on unknown sources,
    features without a key yet are left out with a warning"""
    if not os.path.exists(p):
        raise SystemExit(f'config file not found: {p}')
    with open(p) as f:
        cfg = yaml.safe_load(f) or {}
    srcs = cfg.get('sources') or {}
    if not srcs:
        raise SystemExit(f'{p}: no sources')
    unknown = [s for s in srcs if s not in SOURCES and not (srcs[s] or {}).get('url')]
    if unknown:
        raise SystemExit(f'{p}: unknown source(s) {", ".join(unknown)}; known: {", ".join(SOURCES)} (or give url and product)')
    out, missing = {}, []
    for s, S in srcs.items():
        S = S or {}
        if S.get('enabled', True) is False:
            continue
        base = SOURCES.get(s, {})
        feats = parse_features(S.get('features'))
        keys = {}
        for n, k in feats.items():
            key = k or base.get('keys', {}).get(n) or ''
            if not key:
                missing.append(f'{s}: {n}')
                continue
            keys[n] = key
        ot = S.get('origintime')
        out[s] = dict(url=S.get('url', base.get('url')), product=S.get('product', base.get('product', s.upper())),
                      origintime=str(ot) if ot is not None else None, hours=str(S.get('hours', DEFAULT_HOURS)),
                      kind=S.get('kind', DEFAULT_KIND), batch=int(S.get('batch', base.get('batch', BATCH))), keys=keys)
    if missing:
        print(f'WARNING: no FMI key yet for {", ".join(missing)}, left out of this run '
              f'(give them in {p} as feature: <key> or in SOURCES)')
    return out


# ---------------------------------------------------------------- helpers
def parse_array_string(s):
    if not isinstance(s, str):
        return s
    s = s.strip()
    if s.startswith('['): s = s[1:]
    if s.endswith(']'): s = s[:-1]
    out = []
    for tok in s.split():
        try:
            out.append(float(tok))
        except ValueError:
            out.append(np.nan)
    return out


def explode_response(df1, feat):
    for col in ('latitude', 'longitude', feat):
        df1[col] = df1[col].apply(parse_array_string)
    rows = []
    for _, r in df1.iterrows():
        lats = r['latitude'] if isinstance(r['latitude'], list) else [r['latitude']]
        lons = r['longitude'] if isinstance(r['longitude'], list) else [r['longitude']]
        vals = r[feat] if isinstance(r[feat], list) else [r[feat]]
        n = max(len(lats), len(lons), len(vals))
        for i in range(n):
            rows.append({'utctime': r['utctime'],
                         'latitude': lats[i] if i < len(lats) else np.nan,
                         'longitude': lons[i] if i < len(lons) else np.nan,
                         feat: vals[i] if i < len(vals) else np.nan})
    out = pd.DataFrame(rows, columns=['utctime', 'latitude', 'longitude', feat])
    out['latitude'] = out['latitude'].astype(str)
    out['longitude'] = out['longitude'].astype(str)
    return out


def year_window(y, gstart, gend):
    # clamp the calendar year to the global [start, end] window
    return max(f'{y}0101T000000', gstart), min(f'{y}1231T230000', gend)


def get(url):
    """GET; raises on an error, so a failed request never becomes a checkpoint."""
    r = requests.get(url=url)
    if r.status_code != 200:
        raise RuntimeError(f'HTTP {r.status_code}: {r.text[:200]} | {url[:300]}')
    return r


def read_lonlats(p):
    """Locations as text: columns named lon/longitude and lat/latitude if the file has a header,
    else the first column is longitude and the second latitude (lonlats)."""
    df = pd.read_csv(p, sep=None, engine='python', dtype=str, header=None, comment='#')
    first = [str(v).strip().lower() for v in df.iloc[0]]
    names = {'lon': 'longitude', 'longitude': 'longitude', 'lat': 'latitude', 'latitude': 'latitude'}
    if any(v in names for v in first):   # header row
        df.columns = [names.get(v, v) for v in first]
        df = df.iloc[1:]
    else:
        df = df.iloc[:, :2]
        df.columns = ['longitude', 'latitude']
    df = df[['latitude', 'longitude']].apply(lambda c: c.str.strip())
    # sanity check: the order of the columns, Scandinavia and the Alps are both north of 44N and east of 4W
    lat, lon = df['latitude'].astype(float), df['longitude'].astype(float)
    if lat.min() < 40 or lat.max() > 75 or lon.min() < -15 or lon.max() > 40:
        raise SystemExit(f'{p}: latitudes {lat.min():.2f} to {lat.max():.2f}, longitudes {lon.min():.2f} to {lon.max():.2f}: '
                         'columns in the wrong order? (expected longitude latitude)')
    dup = int(df.duplicated().sum())
    if dup:
        print(f'WARNING: {dup} duplicate locations in {p}, dropped')
        df = df.drop_duplicates()
    return df.reset_index(drop=True)


def wkt_of(locs):
    return 'MULTIPOINT(' + ','.join(f'{lon} {lat}' for lat, lon in zip(locs['latitude'], locs['longitude'])) + ')'


def fetch_feature(feat, fmikey, S, locs, Y, X):
    """One feature for one batch of locations, year by year with checkpoints, then one file for the whole period."""
    col = feat
    stem = f"{S['product']}_{feat}_{REGION}_{Y}-{X}"
    out = os.path.join(output_dir, f"{stem}_{start[:8]}-{end[:8]}_{S['kind']}.csv")
    if os.path.exists(out):
        print(f'skipping (exists): {os.path.basename(out)}')
        return
    wkt_param = quote_plus(wkt_of(locs))
    wkt_debug = quote_plus(wkt_of(locs.iloc[:3]))
    yearly = []
    for y in range(int(start[:4]), int(end[:4]) + 1):
        ys, ye = year_window(y, start, end)
        if ys > ye:
            continue
        year_file = os.path.join(yearly_dir, f"{stem}_{y}_{S['kind']}.csv")
        if os.path.exists(year_file):
            print(f'loading cached year {feat} {Y}-{X} {y}')
            df_year = pd.read_csv(year_file, dtype={'latitude': str, 'longitude': str})
            if not df_year.dropna().empty:
                yearly.append(df_year)
            continue

        print(f'fetching {feat} {Y}-{X} {ys[:8]}-{ye[:8]}')
        q1 = (f"{S['url']}/timeseries"
              f"?wkt={wkt_param}"
              f"&param=utctime,latitude,longitude,{fmikey} as {col}"
              f"&starttime={ys}Z&endtime={ye}Z&hour={S['hours']}"
              f"&format=json&precision=full&timeformat=sql&tz=utc"
              + (f"&origintime={S['origintime']}" if S['origintime'] else ''))
        if y == int(start[:4]):
            print(f'  debug query (first 3 locs): {q1.replace(wkt_param, wkt_debug)}')

        response = get(q1)
        df1 = pd.DataFrame(json.loads(response.content))
        if df1.empty:
            df_year = pd.DataFrame(columns=['utctime', 'latitude', 'longitude', col])
        else:
            df1.columns = ['utctime', 'latitude', 'longitude', col]
            df_year = explode_response(df1, col)
        df_year['latitude'] = df_year['latitude'].astype(str)
        df_year['longitude'] = df_year['longitude'].astype(str)
        has_data = not df_year.dropna().empty
        print(f'  {len(df_year.dropna())} rows with data')

        if has_data or EMPTY_MARKER:
            df_year.to_csv(year_file, index=False)
            print(f'  saved yearly backup: {os.path.basename(year_file)}' + ('' if has_data else ' (empty marker)'))
        else:
            print(f'  no valid data for {feat} {y}, no backup written, fetched again on the next run')
        if has_data:
            yearly.append(df_year)

    if not yearly:
        print(f'  no valid data for {feat} {Y}-{X} in any year, nothing saved')
        return
    df_all = pd.concat(yearly, ignore_index=True)
    df_all['latitude'] = df_all['latitude'].astype(str)
    df_all['longitude'] = df_all['longitude'].astype(str)
    df_all.to_csv(out, index=False)
    print(f'  {len(df_all)} rows -> {out}')


# ---------------------------------------------------------------- main
def main():
    srcs = read_config(CONFIG)
    unknown = [s for s in ONLY if s not in srcs]
    if unknown:
        raise SystemExit(f'source(s) {", ".join(unknown)} not in {CONFIG}; there: {", ".join(srcs)}')
    run = ONLY or list(srcs)
    os.makedirs(yearly_dir, exist_ok=True)

    locs = read_lonlats(lonlats_file)
    print(f'{TAG}: {len(locs)} locations from {lonlats_file}')
    print(locs.head(3).to_string(index=False))
    print(f'config {CONFIG} | sources: {", ".join(run)} | period {start[:8]} to {end[:8]} | outputs in {output_dir}')

    for s in run:
        S = srcs[s]
        if not S['keys']:
            print(f'\n===== {s}: no feature with an FMI key, nothing to fetch =====')
            continue
        B = S['batch']
        batches = [(Y, min(Y + B, len(locs))) for Y in range(0, len(locs), B)]
        print(f'\n===== {s} ({S["url"]}, origintime {S["origintime"] or "not sent"}, {len(batches)} batches of up to {B}): '
              f'{", ".join(S["keys"])} =====')
        for feat, fmikey in S['keys'].items():
            for Y, X in batches:
                fetch_feature(feat, fmikey, S, locs.iloc[Y:X].reset_index(drop=True), Y, X)


if __name__ == '__main__':
    main()