#!/usr/bin/env python3
"""
Plots the ECA&D/E-OBS snow depth stations (stations.txt from eobs_stations_to_lonlat.py) on a map.

Figure 1: all stations, coloured by height like the SNOWCAP training location maps (turbo); stations outside the fetch box (40-75N, 15W-40E) in grey.
Figure 2 (with --clms): stations coloured by the share of days with a valid CLMS SWE value (masked = 0 %),
          read from the ts-clms-swe.py outputs.

usage:
  python plot-eobs-stations.py
  python plot-eobs-stations.py --clms
  python plot-eobs-stations.py --stations /path/stations.txt --outdir /path/figs --clms
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAVE_CARTOPY = True
except ImportError:
    HAVE_CARTOPY = False

path = '/home/ubuntu/data/cryoscope/snowmodel/CLMS+EOBS/'
BOX = dict(lat0=40, lat1=75, lon0=-15, lon1=40)   # same box as the fetch scripts
EXTENT = [-25, 45, 34, 76]                        # lon0, lon1, lat0, lat1 of the map
GREY = '#b0b0b0'
HEIGHT_CMAP = 'turbo'                             # as in the SNOWCAP training location maps


def new_map(title):
    """Map axes: cartopy with coastlines if available, else plain lon/lat axes."""
    fig = plt.figure(figsize=(11, 9))
    if HAVE_CARTOPY:
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.LambertConformal(central_longitude=10, central_latitude=52))
        ax.set_extent(EXTENT, crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, facecolor='#f4f4f2', zorder=0)
        ax.add_feature(cfeature.OCEAN, facecolor='#e8eef3', zorder=0)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.4, edgecolor='#606060')
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor='#909090')
        gl = ax.gridlines(draw_labels=True, linewidth=0.3, color='#c8c8c8', x_inline=False, y_inline=False)
        gl.top_labels = gl.right_labels = False
        tr = dict(transform=ccrs.PlateCarree())
    else:
        print('cartopy not found: plotting on plain lon/lat axes without coastlines')
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlim(EXTENT[0], EXTENT[1])
        ax.set_ylim(EXTENT[2], EXTENT[3])
        ax.set_aspect(1 / np.cos(np.deg2rad(55)))
        ax.grid(linewidth=0.3, color='#c8c8c8')
        ax.set_xlabel('longitude')
        ax.set_ylabel('latitude')
        tr = {}
    # fetch box
    bx = [BOX['lon0'], BOX['lon1'], BOX['lon1'], BOX['lon0'], BOX['lon0']]
    by = [BOX['lat0'], BOX['lat0'], BOX['lat1'], BOX['lat1'], BOX['lat0']]
    if HAVE_CARTOPY:
        ax.plot(bx, by, color='#404040', linewidth=1, linestyle='--', transform=ccrs.Geodetic(), label='fetch box')
    else:
        ax.plot(bx, by, color='#404040', linewidth=1, linestyle='--', label='fetch box')
    ax.set_title(title, fontsize=12, loc='left')
    return fig, ax, tr


def read_stations(p):
    st = pd.read_csv(p)
    st['inside'] = st['latitude'].between(BOX['lat0'], BOX['lat1']) & st['longitude'].between(BOX['lon0'], BOX['lon1'])
    st['height'] = st['height'].where(st['height'] > -9999)
    return st


def plot_height(st, out):
    ins, outs = st[st['inside']], st[~st['inside']]
    # same colouring as the SNOWCAP training location maps: turbo, 0 to the highest location, colour bar 'height'
    fig, ax, tr = new_map(f'ECA&D snow depth stations, coloured by height: {len(ins)} inside the fetch box, {len(st)} in total')
    ax.scatter(outs['longitude'], outs['latitude'], s=8, c=GREY, linewidths=0, label=f'outside box ({len(outs)})', **tr)
    sc = ax.scatter(ins['longitude'], ins['latitude'], s=8, c=ins['height'], cmap=HEIGHT_CMAP, vmin=0,
                    vmax=float(np.nanmax(ins['height'])), linewidths=0, label=f'inside box ({len(ins)})', **tr)
    cb = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
    cb.set_label('height')
    ax.legend(loc='lower left', fontsize=9, frameon=True)
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'saved {out}')


def read_clms_validity(clms_dir):
    """Share of days with a valid (non-NaN) CLMS SWE per location, from the ts-clms-swe.py outputs."""
    files = sorted(glob.glob(os.path.join(clms_dir, 'CLMS_swe_clms_stations_*_daily.csv')))
    files = [f for f in files if os.path.dirname(f) == clms_dir.rstrip('/')]   # whole-period files, not yearly/
    if not files:
        raise SystemExit(f'no CLMS_swe_clms_stations_*_daily.csv in {clms_dir}')
    parts = []
    for f in files:
        print(f'reading {os.path.basename(f)}')
        d = pd.read_csv(f, usecols=['latitude', 'longitude', 'swe'])
        parts.append(d.groupby(['latitude', 'longitude'])['swe'].agg(valid=lambda s: s.notna().mean()).reset_index())
    v = pd.concat(parts, ignore_index=True)
    v['lat_r'] = v['latitude'].round(4)
    v['lon_r'] = v['longitude'].round(4)
    return v[['lat_r', 'lon_r', 'valid']]


def plot_clms(st, v, out):
    st = st[st['inside']].copy()
    st['lat_r'] = st['latitude'].round(4)
    st['lon_r'] = st['longitude'].round(4)
    m = st.merge(v, on=['lat_r', 'lon_r'], how='left')
    nofetch = m['valid'].isna()
    masked = m['valid'].eq(0)
    ok = ~nofetch & ~masked
    fig, ax, tr = new_map(f'CLMS SWE at the stations: {int(ok.sum())} with data, {int(masked.sum())} always masked, '
                          f'{int(nofetch.sum())} not fetched')
    ax.scatter(m.loc[masked, 'longitude'], m.loc[masked, 'latitude'], s=12, marker='x', c='#c0392b', linewidths=0.8,
               label=f'always masked ({int(masked.sum())})', **tr)
    if nofetch.any():
        ax.scatter(m.loc[nofetch, 'longitude'], m.loc[nofetch, 'latitude'], s=8, c=GREY, linewidths=0,
                   label=f'not fetched ({int(nofetch.sum())})', **tr)
    sc = ax.scatter(m.loc[ok, 'longitude'], m.loc[ok, 'latitude'], s=10, c=100 * m.loc[ok, 'valid'], cmap='Blues',
                    vmin=0, vmax=100, edgecolors='#606060', linewidths=0.2, label=f'with data ({int(ok.sum())})', **tr)
    cb = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
    cb.set_label('days with a valid CLMS SWE value (%)')
    ax.legend(loc='lower left', fontsize=9, frameon=True)
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'saved {out}')
    # per country summary
    summ = m.assign(masked=masked, with_data=ok).groupby('cn')[['masked', 'with_data']].sum().astype(int)
    summ['total'] = summ.sum(axis=1)
    print(summ.sort_values('total', ascending=False).to_string())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stations', default=f'{path}stations.txt', help='stations.txt (latitude,longitude,staid,...)')
    ap.add_argument('--outdir', default=path, help='where the figures go')
    ap.add_argument('--clms', action='store_true', help='also plot the CLMS SWE validity per station')
    ap.add_argument('--clmsdir', default=path, help='folder of the ts-clms-swe.py outputs')
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    st = read_stations(args.stations)
    plot_height(st, os.path.join(args.outdir, 'eobs_stations_map.png'))
    if args.clms:
        v = read_clms_validity(args.clmsdir)
        plot_clms(st, v, os.path.join(args.outdir, 'eobs_stations_clms_validity_map.png'))


if __name__ == '__main__':
    main()