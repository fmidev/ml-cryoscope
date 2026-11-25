#!/usr/bin/env python3
import os, sys, json, time
import requests
import pandas as pd
import numpy as np
from urllib.parse import quote_plus

TAGS = ('training', 'validation')
TAG = sys.argv[1] if len(sys.argv) > 1 else ''
if TAG not in TAGS:
    raise SystemExit(f'usage: python {os.path.basename(sys.argv[0])} <{"|".join(TAGS)}> [group ...]')
ONLY = sys.argv[2:]   # groups given on the command line; empty = every group with enabled=True

path = '/home/users/smartmet/data/snowmodel/'
lonlats_file = f'{path}snowcap_scandi_alps_{TAG}_lonlats.txt'
output_dir = f'{path}{TAG}_inputs/reanalysis/'
yearly_dir = os.path.join(output_dir, 'yearly')
REGION = 'scandi_alps'
BATCH = 5000

# period: covers both domains (Scandinavia 2015-09 to 2023-09, Alps 2016-09 to 2024-09)
start = '20150830T000000'
end = '20240901T000000'

source_era5l = 'http://ml.harvesterseasons.com:8080'
era5l={
    'fal': 'ALBEDOSLR-0TO1:ERA5L:5099:1:0:1:0', # forecast albedo
    'asn': 'ASN-0TO1:ERA5L:5099:1:0:1:0', # snow albedo
    'es': 'ES-M:ERA5L:5099:1:0:1:0', # snow evaporation 
    'e': 'EVAP-M:ERA5L:5099:1:0:1:0', # evaporation
    'pev': 'EVAPP-M:ERA5L:5099:1:0:1:0', # potential evaporation
    'slhf': 'FLLAT-JM2:ERA5L:5099:1:0:1:0', # surface latent heat flux
    'sshf': 'FLSEN-JM2:ERA5L:5099:1:0:1:0', # surface sensible heat flux
    'sd': 'HSNOW-M:ERA5L:5099:1:0:1:0', # snow depth (m)
    'laihv': 'LAI_HV-M2M2:ERA5L:5099:1:0:1:0', # leaf area index high vegetation
    'lailv': 'LAI_LV-M2M2:ERA5L:5099:1:0:1:0', # leaf area index low vegetation
    'licd': 'LICD-M:ERA5L:5099:1:0:1:0', # lake ice total depth
    'lict': 'LICT-K:ERA5L:5099:1:0:1:0', # lake ice surface temperature
    'sp': 'PGR-PA:ERA5L:5099:1:0:1:0', # surface pressure
    'ssrd': 'RADGLOA-JM2:ERA5L:5099:1:0:1:0', 
    'strd': 'RADLWA-JM2:ERA5L:5099:1:0:1:0', 
    'str': 'RNETLWA-JM2:ERA5L:5099:1:0:1:0', 
    'ssr': 'RNETSWA-JM2:ERA5L:5099:1:0:1:0', 
    'ro': 'RO-M:ERA5L:5099:1:0:1:0', # runoff
    'skt': 'SKT-K:ERA5L:5099:1:0:1:0', # skin temperature (new)
    'smlt': 'SMLT-M:ERA5L:5099:1:0:1:0', # snowmelt
    'sf': 'SNACC-KGM2:ERA5L:5099:1:0:1:0', # snow fall
    'rsn': 'SND-KGM3:ERA5L:5099:1:0:1:0', # snow density
    'snowc': 'SNOWC-PRCNT:ERA5L:5099:1:0:0', # snow cover
    'swvl1': 'SOILWET-M3M3:ERA5L:5099:9:7:1:0', # volumetric soil water layer 1
    'swvl2': 'SWVL2-M3M3:ERA5L:5099:9:1820:1:0', # volumetric soil water layer 2
    'swvl3': 'SWVL3-M3M3:ERA5L:5099:9:7268:1:0', # volumetric soil water layer 3
    'swvl4': 'SWVL4-M3M3:ERA5L:5099:9:25855:1:0', # volumetric soil water layer 4
    'src': 'SRC-M:ERA5L:5099:1:0:1:0', # skin reservoir content
    'sro': 'SRO-M:ERA5L:5099:1:0:1:0', 
    'ssro': 'SSRO-M:ERA5L:5099:1:0:1:0', 
    'stl1': 'STL1-K:ERA5L:5099:9:7:1:0', # soil temperature layer 1
    't2': 'T2-K:ERA5L:5099:1:0:1:0', # 2m temperature
    'td2': 'TD2-K:ERA5L:5099:1:0:1:0', # 2m dewpoint temperature
    'tsn': 'TSN-K:ERA5L:5099:1:0:1:0', # temperature of snow layer
    'u10': 'U10-MS:ERA5L:5099:1:0:1:0', # 10m u wind component
    'v10': 'V10-MS:ERA5L:5099:1:0:1:0', # 10m v wind component
    'tp': 'RR-M:ERA5L:5099:1:0:1:0' # total precipitation
  }

source_era5ltmax='http://sm.cryo-scope.eu:8080'
era5l_tmax={
    't': 'T2-K:ERA5LD:5093:1:0:1', # daily 2m temp stats
}

source_era5+l='http://sm.cryo-scope.eu:8080'
era5+l={
    'sd':   'SD-M:ERA5+L:5109:1:0:1',
    'rsn':  'SND-KGM3:ERA5+L:5109:1:0:1',
    'stl1': 'STL1-K:ERA5+L:5109:9:7:1',
    't2':   'T2-K:ERA5+L:5109:1:0:1'
}

source_era5='http://smartmet.xyz:8080'
era5 ={
    't500_era5': 'T-K:ERA5:5021:2:500:1:0', # 500hPa temperature
    't700_era5': 'T-K:ERA5:5021:2:700:1:0', # 700hPa temperature
    't850_era5': 'T-K:ERA5:5021:2:850:1:0', # 850hPa temperature
    'z500_era5': 'Z-M2S2:ERA5:5021:2:500:1:0', # 500hPa geopotential
    'z700_era5': 'Z-M2S2:ERA5:5021:2:700:1:0', # 700hPa geopotential
    'z850_era5': 'Z-M2S2:ERA5:5021:2:850:1:0', # 850hPa geopotential
    'u500_era5': 'U-MS:ERA5:5021:2:500:1:0', # 500hPa u wind component
    'u700_era5': 'U-MS:ERA5:5021:2:700:1:0', # 700hPa u wind component
    'u850_era5': 'U-MS:ERA5:5021:2:850:1:0', # 850hPa u wind component
    'v500_era5': 'V-MS:ERA5:5021:2:500:1:0', # 500hPa v wind component
    'v700_era5': 'V-MS:ERA5:5021:2:700:1:0', # 700hPa v wind component
    'v850_era5': 'V-MS:ERA5:5021:2:850:1:0', # 850hPa v wind component
    'q500_era5': 'Q-KGKG:ERA5:5021:2:500:1:0', # 500hPa specific humidity
    'q700_era5': 'Q-KGKG:ERA5:5021:2:700:1:0', # 700hPa specific humidity
    'q850_era5': 'Q-KGKG:ERA5:5021:2:850:1:0', # 850hPa specific humidity
}

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


def get_with_retry(url):
    """GET with retries; raises after the last try, so a failed request never becomes a checkpoint."""
    for i in range(1, RETRIES + 1):
        try:
            r = requests.get(url=url, timeout=TIMEOUT)
            if r.status_code == 200:
                return r
            msg = f'HTTP {r.status_code}: {r.text[:200]}'
        except requests.RequestException as e:
            msg = str(e)
        print(f'  request failed (try {i}/{RETRIES}): {msg}')
        if i < RETRIES:
            time.sleep(30 * i)
    raise RuntimeError(f'request failed {RETRIES} times, stopping: {url[:300]}')


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


def fetch_feature(feat, fmikey, G, locs, Y, X):
    """One feature for one batch of locations, year by year with checkpoints, then one file for the whole period."""
    col = G['col'](feat)
    stem = f"{G['product']}_{feat}_{REGION}_{Y}-{X}"
    out = os.path.join(output_dir, f"{stem}_{start[:8]}-{end[:8]}_{G['kind']}.csv")
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
        year_file = os.path.join(yearly_dir, f"{stem}_{y}_{G['kind']}.csv")
        if os.path.exists(year_file):
            print(f'loading cached year {feat} {Y}-{X} {y}')
            df_year = pd.read_csv(year_file, dtype={'latitude': str, 'longitude': str})
            if not df_year.dropna().empty:
                yearly.append(df_year)
            continue

        print(f'fetching {feat} {Y}-{X} {ys[:8]}-{ye[:8]}')
        q1 = (f"{G['source']}/timeseries"
              f"?wkt={wkt_param}"
              f"&param=utctime,latitude,longitude,{fmikey} as {col}"
              f"&starttime={ys}Z&endtime={ye}Z&hour={G['hours']}"
              f"&format=json&precision=full&timeformat=sql&tz=utc&origintime={G['origintime']}")
        if y == int(start[:4]):
            print(f'  debug query (first 3 locs): {q1.replace(wkt_param, wkt_debug)}')

        response = get_with_retry(q1)
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

        if has_data or G['empty_marker']:
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
    os.makedirs(yearly_dir, exist_ok=True)
    unknown = [g for g in ONLY if g not in GROUPS]
    if unknown:
        raise SystemExit(f'unknown group(s) {", ".join(unknown)}; groups: {", ".join(GROUPS)}')
    groups = ONLY or [g for g, G in GROUPS.items() if G['enabled']]

    locs = read_lonlats(lonlats_file)
    batches = [(Y, min(Y + BATCH, len(locs))) for Y in range(0, len(locs), BATCH)]
    print(f'{TAG}: {len(locs)} locations from {lonlats_file}, {len(batches)} batches of up to {BATCH}')
    print(locs.head(3).to_string(index=False))
    print(f'groups: {", ".join(groups)} | period {start[:8]} to {end[:8]} | outputs in {output_dir}')

    for g in groups:
        G = GROUPS[g]
        print(f'\n===== {g} ({G["source"]}) =====')
        for feat, fmikey in G['keys'].items():
            for Y, X in batches:
                fetch_feature(feat, fmikey, G, locs.iloc[Y:X].reset_index(drop=True), Y, X)


if __name__ == '__main__':
    main()