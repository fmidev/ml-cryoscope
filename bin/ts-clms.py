#!/usr/bin/env python3
"""
Fetches the CLMS snow water equivalent (HSNOW-M:CLMS:5051, 5 km Northern Hemisphere, daily, 2006-2024) from the SmartMet
timeseries plugin for the ECA&D/E-OBS snow depth station locations (CLMS+EOBS/stations_lonlats.txt).

Period: the SNOWCAP target dates, 2015-09-01 to 2024-09-01 (covers Scandinavia 2015-2023 and the Alps 2016-2024).
Units: the server gives the value in m (w.e.); the query multiplies it by 1000 (MUL), so the outputs are in mm like the
SNOWCAP target. Negative values (mask/flag codes: water, mountains, glaciers, missing) are set to NaN and counted.

usage: python ts-clms-swe.py
"""
import os, json
import requests
import pandas as pd
import numpy as np
from urllib.parse import quote_plus, quote

# ---------------------------------------------------------------- settings
path = '/home/ubuntu/data/cryoscope/snowmodel/CLMS+EOBS/'
lonlats_file = f'{path}stations_lonlats.txt'
output_dir = path
yearly_dir = os.path.join(output_dir, 'yearly')

URL = 'http://ml.harvesterseasons.com:8080'
FMIKEY = 'HSNOW-M:CLMS:5051:1:0:1'
PARAM = f'MUL{{{FMIKEY};1000}}'   # m -> mm in the query
ORIGINTIME = '20000101T000000'   # generation CLMS:20000101T000000
FEAT = 'swe'             # column name in the outputs
PRODUCT = 'CLMS'
DATASET = 'clms'         # dataset tag in every output and checkpoint file name
REGION = 'stations'
BATCH = 5000             # locations per request
HOURS = '0'              # hour= of the request: 00 UTC (CLMS is daily at T000000)
KIND = 'daily'
EMPTY_MARKER = False     # True: a year without data is saved as an empty checkpoint, not fetched again

start = '20150901T000000'
end = '20240901T000000'


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
    lat, lon = df['latitude'].astype(float), df['longitude'].astype(float)
    # same station selection as ts-reanalysis-clms.py: keep those in the box, stop only if most fall outside (swapped)
    inside = lat.between(40, 75) & lon.between(-15, 40)
    if inside.mean() < 0.5:
        raise SystemExit(f'{p}: only {int(inside.sum())} of {len(df)} locations inside 40-75N, 15W-40E: '
                         'columns in the wrong order? (expected longitude latitude)')
    if not inside.all():
        print(f'WARNING: {int((~inside).sum())} locations outside 40-75N, 15W-40E in {p}, dropped')
        df = df[inside]
    dup = int(df.duplicated().sum())
    if dup:
        print(f'WARNING: {dup} duplicate locations in {p}, dropped')
        df = df.drop_duplicates()
    return df.reset_index(drop=True)


def wkt_of(locs):
    return 'MULTIPOINT(' + ','.join(f'{lon} {lat}' for lat, lon in zip(locs['latitude'], locs['longitude'])) + ')'


def fetch_batch(locs, Y, X):
    """CLMS SWE for one batch of locations, year by year with checkpoints, then one file for the whole period."""
    stem = f'{PRODUCT}_{FEAT}_{DATASET}_{REGION}_{Y}-{X}'
    out = os.path.join(output_dir, f'{stem}_{start[:8]}-{end[:8]}_{KIND}.csv')
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
        year_file = os.path.join(yearly_dir, f'{stem}_{y}_{KIND}.csv')
        if os.path.exists(year_file):
            print(f'loading cached year {FEAT} {Y}-{X} {y}')
            df_year = pd.read_csv(year_file, dtype={'latitude': str, 'longitude': str})
            if not df_year.dropna().empty:
                yearly.append(df_year)
            continue

        print(f'fetching {FEAT} {Y}-{X} {ys[:8]}-{ye[:8]}')
        q1 = (f'{URL}/timeseries'
              f'?wkt={wkt_param}'
              f"&param=utctime,latitude,longitude,{quote(PARAM, safe=':')}%20as%20{FEAT}"
              f'&starttime={ys}Z&endtime={ye}Z&hour={HOURS}'
              f'&format=json&precision=full&timeformat=sql&tz=utc'
              + (f'&origintime={ORIGINTIME}' if ORIGINTIME else ''))
        if y == int(start[:4]):
            print(f'  debug query (first 3 locs): {q1.replace(wkt_param, wkt_debug)}')

        response = get(q1)
        df1 = pd.DataFrame(json.loads(response.content))
        if df1.empty:
            df_year = pd.DataFrame(columns=['utctime', 'latitude', 'longitude', FEAT])
        else:
            df1.columns = ['utctime', 'latitude', 'longitude', FEAT]
            df_year = explode_response(df1, FEAT)
        df_year['latitude'] = df_year['latitude'].astype(str)
        df_year['longitude'] = df_year['longitude'].astype(str)

        # flag codes (negative) -> NaN
        v = pd.to_numeric(df_year[FEAT], errors='coerce')
        neg = int((v < 0).sum())
        if neg:
            codes = ', '.join(f'{c:g}: {n}' for c, n in v[v < 0].value_counts().sort_index().items())
            print(f'  {neg} negative values set to NaN (code: count {codes})')
        df_year[FEAT] = v.where(v >= 0)

        has_data = not df_year.dropna().empty
        print(f'  {len(df_year.dropna())} rows with data of {len(df_year)}')
        print(df_year)

        if has_data or EMPTY_MARKER:
            df_year.to_csv(year_file, index=False)
            print(f'  saved yearly backup: {os.path.basename(year_file)}' + ('' if has_data else ' (empty marker)'))
        else:
            print(f'  no valid data for {FEAT} {y}, no backup written, fetched again on the next run')
        if has_data:
            yearly.append(df_year)

    if not yearly:
        print(f'  no valid data for {FEAT} {Y}-{X} in any year, nothing saved')
        return
    df_all = pd.concat(yearly, ignore_index=True)
    df_all['latitude'] = df_all['latitude'].astype(str)
    df_all['longitude'] = df_all['longitude'].astype(str)
    df_all.to_csv(out, index=False)
    print(f'  {len(df_all)} rows -> {out}')


# ---------------------------------------------------------------- main
def main():
    os.makedirs(yearly_dir, exist_ok=True)
    locs = read_lonlats(lonlats_file)
    print(f'{DATASET}: {len(locs)} locations from {lonlats_file}')
    print(locs.head(3).to_string(index=False))
    batches = [(Y, min(Y + BATCH, len(locs))) for Y in range(0, len(locs), BATCH)]
    print(f'{PARAM} from {URL} (origintime {ORIGINTIME or "not sent"}) | period {start[:8]} to {end[:8]} | '
          f'{len(batches)} batches of up to {BATCH} | outputs in {output_dir}')
    for Y, X in batches:
        fetch_batch(locs.iloc[Y:X].reset_index(drop=True), Y, X)


if __name__ == '__main__':
    main()