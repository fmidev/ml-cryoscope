# CryoSCOPE SWE model dashboard: training runs (Optuna, SHAP, EDA), independent validation, validation explorer,
# past SWE forecast predictions validated against HKL, and the latest SWE predictions on a web map
# pages, in sidebar sections (NAV at the end of the file):
#   Overview
#   Final model:                Training and model performance, Independent validation   (only the final model, USED_MODEL, swe-ml-2)
#   All model versions:         Training explorer, Validation explorer                   (every model edition)
#   SWE predictions (past):     10 and 15 days (500 m), seasonal 215 days (2.8 km), seasonal 215 days (9 km)   (final model, divided by start date)
#   Compare model predictions:  10 and 15 days, seasonal                                  (every model edition)
#   Latest SWE predictions:     web map with SmartMet WMS layers; click a spot for its ECXENS and ECXSF member time series
# reads the outputs of the tuning script (models/, results/), the validation script (validationresults_<optuna_tag>/, one folder per model)
# the 10 day ECXENS vs HKL forecast validation script (validation_ecens/<model>/<start date>/<prediction>/, copied from the forecast server)
# and the seasonal ECXSF vs HKL validation script xgb-validate-swe-sf.py
# (validation_sf/<model>/<start date>/<prediction>/ or validation_sf/<model>/<prediction>/, run_info.json in the folder above <prediction>/)
# seasonal resolution (2.8 km or 9 km) of a run: 'resolution' or 'grid' in run_info.json (the validation script writes grid, e.g. 0.025°),
#   else a _9km / _2.8km suffix on the model folder (FIN5_9km, FIN5_Alps_9km), else the folder it is in: SF_DIR 2.8 km, SF9_DIR 9 km
# the Latest SWE predictions page reads the SmartMet Server directly: WMS for the map, the timeseries plugin for the clicked spot
# run: in docker, see snowlook_setup.md (locally: streamlit run snowlook.py --server.port 8503)
# needs: pip install streamlit plotly pandas numpy requests folium streamlit-folium
import os,glob,json,contextlib
from urllib.parse import parse_qsl,urlencode
import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
try:
    import folium
except ImportError:
    folium=None
try:
    from streamlit_folium import st_folium
except ImportError:
    st_folium=None

# logos: snowlook.png (app wordmark), snowlook_icon.png (app icon, browser tab), snowlook_dark.png (wordmark for a dark theme),
# cryo.png (CryoSCOPE logo), cryoscope.png (banner with the funding statement), snowcap.png (square), fmi.png (wide),
# optional trafficability.png; looked up in LOGO_DIR first, then next to the app
APP_NAME='SnowLook'
LOGO_DIR=os.environ.get('LOGO_DIR','/home/ubuntu/data/cryoscope')
APP_DIR=os.path.dirname(os.path.abspath(__file__))
def logo(name):
    for d in (LOGO_DIR,APP_DIR):
        p=os.path.join(d,name)
        if os.path.exists(p): return p
    return None

st.set_page_config(page_title=APP_NAME,page_icon=logo('snowlook_icon.png') or '❄️',layout='wide')

# CryoSCOPE logo at the top left of the sidebar, above the CryoSCOPE SWE title
if logo('cryo.png'):
    if hasattr(st,'logo'):
        try: st.logo(logo('cryo.png'),size='large')
        except TypeError: st.logo(logo('cryo.png'))   # streamlit < 1.39 has no size
    else:
        st.sidebar.image(logo('cryo.png'),width=180)
SHOW_FOOTER=True   # the overview page has the logos in its own text, so it turns the footer off


# hidden for now: the EDA tab of the training pages, and the All figures tabs (training, past seasonal, compare 10 / 15 days)
SHOW_EDA=False
SHOW_ALL_FIGURES=False
SEASON_MONTHS=[9,10,11,12,1,2,3,4,5,6,7,8]
MONTH_NAMES={1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
PRED_STYLE={'SWE_observed':('observed','black','solid'),'SWE_xgboost':('XGBoost','#1f77b4','solid'),
            'SWE_climatology':('climatology','#2ca02c','dot'),'SWE_era5l':('ERA5-Land','#ff7f0e','dash')}
PEAK_MIN=10.0  # mm, same as peak_min in the validation script
ERR_MAX=150.0  # mm, fixed colour scale for RMSE, ubRMSE, MAE maps (all predictors): 0..ERR_MAX, values above shown red
ME_MAX=75.0    # mm, fixed symmetric colour scale for ME and difference maps: -ME_MAX..+ME_MAX

# SWE predictions page: prediction folders written by the forecast validation script, shown in this order (any others after them)
FC_ORDER=['ensmean','member01','wettest','driest']
# layout: validation_ecens/<model>/<start date>/<prediction>/, e.g. validation_ecens/FIN5/20230210/wettest/swe_metrics.csv
#   (the folder is set by SWE_PRED_DIR or in the sidebar, default <snowmodel directory>/validation_ecens)
#   <model>      model edition (optuna_tag) that made the ECXENS forecast, any name works (also models not in EDITION_NAMES);
#                'model' in <start date>/run_info.json overrides the folder name
#   <start date> forecast base date, YYYYMMDD (also YYYYMMDDTHHMMSS, YYYYMMDDHH or YYYY-MM-DD); 'base_time' in run_info.json
#                overrides it, and a folder name that is no date falls back to the most common (date minus lead day) in the scores
#   <prediction> ensmean, member01, wettest, driest, ... each with swe_metrics.csv and the png figures
# older layouts still work: validation/<model>/<prediction>/ (start date from the scores) and validation/<prediction>/ (model FC_LEGACY_MODEL,
# the model those old results came from)
# region: 'region' in run_info.json, else a region suffix on the model folder (FIN5noclim_Alps, FIN5noclim_alps, FIN5noclim-Alps),
#   else FC_DEFAULT_REGION; e.g. validation/FIN5noclim_Alps/20230210/ is model FIN5noclim, region Alps
FC_LEGACY_MODEL='FINmulti'
# simpler SWE predictions (past) 10 and 15 days page (final model): no 'scores per start date, chart' expander, and below the maps only
# Ensemble members (domain mean plots of every member) and Period summary; the compare page keeps everything
FC_SIMPLE=True
FC_DEFAULT_REGION='Scandinavia'
FC_REGIONS={'alps':'Alps','scandi':'Scandinavia','scandinavia':'Scandinavia'}   # folder suffix / run_info region (lower case) -> shown name
# the final model edition: the Final model and SWE predictions (past) pages show only this one,
# the All model versions and Compare model predictions pages show every iteration. If this exact tag has no results, the first edition
# whose name starts with USED_PREFIX is used (so a retrained swe-ml-2... edition shows up without editing this line)
USED_MODEL='swe-ml-2'
USED_PREFIX='swe-ml-2'
# shown first and picked by default on the explorer and compare pages
PROD_MODEL=USED_MODEL
# model folder name -> model edition, 10 / 15 days and seasonal (validation_ecens/FIN5, validation_ecens/FIN5_Alps, validation_sf/FIN2, validation_sf/FIN5)
MODEL_ALIASES={'FIN2':'FIN2multiNoMax','FIN5':'FIN5noclim'}
def prod_first(models,first=PROD_MODEL): return sorted(models,key=lambda m:(m!=first,m))
def used_model(models):
    # USED_MODEL if it is among the models, else the first one starting with USED_PREFIX, else None
    models=list(models)
    if USED_MODEL in models: return USED_MODEL
    return next((m for m in models if str(m).lower().startswith(USED_PREFIX.lower())),None)
def default_model(models,first=PROD_MODEL):
    # given model (production by default) if it has results, else the edition picked on the Training explorer page, else the first
    if first in models: return first
    ed=st.session_state.get('edition_keep')
    return ed if ed in models else models[0]
FC_SKIP_DIRS={'app_cache','tmp','__pycache__'}   # folders under validation/ that are never models, start dates or predictions
# names shown when a folder has no pred_info.json; member1, member01, ... are named 'ECXENS member N' automatically
FC_NAMES={'ensmean':'ECXENS ens mean','wettest':'ECXENS wettest member','driest':'ECXENS driest member'}
MILD_BIAS=50   # mm, same as MILD_BIAS in the forecast validation script (soft bias colours up to this, strong beyond)
# member_ranking.csv columns -> label
MEMBER_STATS={'mean':'domain mean SWE, all lead days','mean_last_day':'domain mean SWE, last lead day','p99':'99th percentile SWE','max':'max SWE'}
# scores in swe_metrics.csv -> (label, unit); 'snow_' versions use only pixels where forecast or HKL has at least 1 mm
FC_METRICS={'bias':('bias (forecast minus HKL)','mm'),'mae':('MAE','mm'),'rmse':('RMSE','mm'),'r':('correlation r',''),'kge':('KGE','')}

# start dates (cases) of the final model on the SWE predictions (past) pages: region -> {start date: (what the case tests, out of training seasons)}
# out of training = True is marked with ★. Results for start dates not listed here still show up, without a description.
# 10 / 15 day ECXENS starts, 00 UTC, 1 to 3 days before the event
FC_CASES={
    'Alps':{
        '2018-01-19':('level 5 snowfall, rain at lower elevations',False),
        '2018-04-10':('rapid April melt',False),
        '2019-01-01':('continuous northern snowfall',False),
        '2019-01-08':('continuous northern snowfall, second start',False),
        '2020-12-03':('southern extreme snowfall',False),
        '2024-05-28':('late high elevation snow',False),
        '2024-11-17':('low altitude November dump',True),
        '2025-01-25':('snowfall up to the level 4 avalanche danger of 28 Jan',True),
        '2025-03-08':('southern stau',True),
        '2025-11-22':('late November burst',True),
        '2025-12-15':('dry, snow poor persistence (nothing happens control)',True)},
    'Scandinavia':{
        '2018-05-05':('record warm May melt',False),
        '2019-12-10':('record December snow in Lapland',False),
        '2020-04-08':('record April snow depth',False),
        '2020-05-18':('melt flood onset',False),
        '2023-01-04':('heavy snowfall in Ångermanland, normal year',False),
        '2024-01-01':('extreme cold and snowstorm in Skåne',True),
        '2024-01-15':('nationwide snowfall',True),
        '2025-01-10':('föhn warm spell',True),
        '2026-01-15':('dry, cold, thin snow, fire danger in Rogaland',True),
        '2026-03-01':('March partial melt in a deficit year',True)}}
# seasonal ECXSF starts (215 days): Nov 1 covers accumulation, peak and most melt; Feb 1 or Mar 1 melt timing; Oct 1 onset
SF_CASES={
    'Alps':{
        '2016-11-01':('very little snow, late start',False),
        '2017-11-01':('very much snow, then early melt',False),
        '2018-03-01':('very much snow, then early melt (melt start)',False),
        '2018-12-01':('north extreme, south dry',False),
        '2021-11-01':('snow drought',False),
        '2022-02-01':('snow drought (melt start)',False),
        '2023-11-01':('elevation split, long high altitude season',False),
        '2024-03-01':('elevation split, long high altitude season (melt start)',False),
        '2024-11-01':('poor north and east, normal south',True),
        '2025-02-01':('poor north and east, normal south (melt start)',True),
        '2025-11-01':('record dry south, early burst',True),
        '2026-02-01':('record dry south, early burst (melt start)',True)},
    'Scandinavia':{
        '2017-11-01':('snowy, early melt, wildfire summer',False),
        '2018-03-01':('snowy, early melt, wildfire summer (melt start)',False),
        '2019-10-01':('long season north, snowless south (onset)',False),
        '2019-12-01':('long season north, snowless south, Kemijoki melt',False),
        '2022-11-01':('near normal',False),
        '2023-11-01':('cold, snowy, long season',True),
        '2024-03-01':('cold, snowy, long season (melt start)',True),
        '2024-11-01':('mild, little snow',True),
        '2025-11-01':('snow drought south Norway, winter fire risk',True),
        '2026-01-01':('snow drought south Norway, winter fire risk (second start)',True)}}
# the 9 km runs cover all of Europe: their start dates are the Scandinavian and Alpine ones together
def _merge_cases(*regions):
    out={}
    for reg in regions:
        for k,(desc,o) in SF_CASES[reg].items():
            d,o0=out.get(k,('',False))
            out[k]=((d+'; ' if d else '')+f'{reg}: {desc}',o0 or o)
    return dict(sorted(out.items()))
SF_CASES['Europe']=_merge_cases('Scandinavia','Alps')
SF_CASE_REGIONS={'2.8 km':['Scandinavia','Alps'],'9 km':['Europe']}   # regions with planned start dates on each seasonal page

# start dates left out of the SWE prediction (past) and compare pages, even when results exist
# entries match by prefix: '2023-02' hides every start date in February 2023, '2023-02-10' only that day
CASE_HIDE={'2023-02'}
def case_hidden(k): return any(str(k).startswith(x) for x in CASE_HIDE)
# source (reference) for the snow conditions of each case: region -> {start date: url}; shown as a link in the start date table.
# From the case selection research (Nov 2026): SLF winter reviews, GeoSphere/MeteoSwiss/DWD Alpine climate reviews, ECMWF case study,
# Colombo et al. 2023 (ERL), FMI and SMHI winter summaries, NASA Earth Observatory, NVE reports; the 9 km Europe region uses both
_SLF1617='https://www.slf.ch/en/news/2017/05/short-winter-marked-by-little-snowfall-and-few-avalanche-victims/'
_SLF1718='https://www.slf.ch/en/news/2018/05/extraordinary-snowfall-and-very-high-avalanche-danger/'
_JAN2018='https://www.preventionweb.net/quick/21725'
_ECMWF2019='https://confluence.ecmwf.int/x/FnayBw'
_DEC2020='https://weathertoski.co.uk/weather-snow/archive/snow-report-11-december-2020'
_COLOMBO='https://doi.org/10.1088/1748-9326/acdb88'
_SLF2324='https://idw-online.de/en/news830159'
_SLF2425='https://www.snowindustrynews.com/articles/slf-reports-snow-poor-winter-pronounced-inner-alpine-old-snow-problem-few-fatal-avalanche-accidents'
_ALP2425='https://www.geosphere.at/en/news-and-events/news/alpine-climate-winter-half-year-2024-25-balance-sheet'
_ALP2526='https://geosphere.at/en/news-and-events/news/alpine-climate-review-of-the-2025-26-winter-half-year'
_SE2018='https://en.wikipedia.org/wiki/2018_Sweden_wildfires'
_FMI2020='https://en.ilmatieteenlaitos.fi/tiedote/1260104454'
_NASA2020='https://www.earthobservatory.nasa.gov/images/146821/frozen-finland-thaws'
_SMHI2023='https://www.smhi.se/klimat/klimatet-da-och-nu/arets-och-arstidernas-vader/arets-och-arstidernas-vader-och-vatten/2023-02-27-vintern-2023---ganska-normal-utifran-den-nuvarande-normalperioden'
_SMHI2024='https://www.smhi.se/klimat/klimatet-da-och-nu/arets-och-arstidernas-vader/arets-och-arstidernas-vader-och-vatten/2024-02-27-vintern-2024---klassisk-svensk-vinter'
_FMI2025='https://en.ilmatieteenlaitos.fi/press-release/7oakiFP4OVygRzXzkdXhpD'
_NO2026FIRE='https://www.mojanorwegia.pl/en/such-weather-is-rare-in-january-norway-threatened-by-wildfires-26922.html'
_NVE2026='https://www.mojanorwegia.pl/en/norways-problems-deepen-another-threat-looms-on-the-horizon-28251.html'
FC_CASE_SOURCES={
    'Alps':{'2018-01-19':_JAN2018,'2018-04-10':_SLF1718,'2019-01-01':_ECMWF2019,'2019-01-08':_ECMWF2019,'2020-12-03':_DEC2020,
            '2024-05-28':_SLF2324,'2024-11-17':_SLF2425,'2025-01-25':_SLF2425,'2025-03-08':_SLF2425,'2025-11-22':_ALP2526,'2025-12-15':_ALP2526},
    'Scandinavia':{'2018-05-05':_SE2018,'2019-12-10':_FMI2020,'2020-04-08':_NASA2020,'2020-05-18':_NASA2020,'2023-01-04':_SMHI2023,
                   '2024-01-01':_SMHI2024,'2024-01-15':_SMHI2024,'2025-01-10':_FMI2025,'2026-01-15':_NO2026FIRE,'2026-03-01':_NVE2026}}
SF_CASE_SOURCES={
    'Alps':{'2016-11-01':_SLF1617,'2017-11-01':_SLF1718,'2018-03-01':_SLF1718,'2018-12-01':_ECMWF2019,'2021-11-01':_COLOMBO,'2022-02-01':_COLOMBO,
            '2023-11-01':_SLF2324,'2024-03-01':_SLF2324,'2024-11-01':_ALP2425,'2025-02-01':_ALP2425,'2025-11-01':_ALP2526,'2026-02-01':_ALP2526},
    'Scandinavia':{'2017-11-01':_SE2018,'2018-03-01':_SE2018,'2019-10-01':_FMI2020,'2019-12-01':_FMI2020,'2022-11-01':_SMHI2023,
                   '2023-11-01':_SMHI2024,'2024-03-01':_SMHI2024,'2024-11-01':_FMI2025,'2025-11-01':_NVE2026,'2026-01-01':_NO2026FIRE}}
# source columns of the start date table per region: column name -> {start date: url}; the 9 km Europe page gets one column per region
def case_sources(table,region):
    if region=='Europe': return {f'source {r}':table.get(r,{}) for r in ('Scandinavia','Alps')}
    return {'source':table.get(region,{})}
# predictions whose daily maps are collapsed (opened with a click) on the 10 / 15 day pages
FC_COLLAPSED={'wettest','driest'}

# training locations figures in the snowmodel directory: area -> file name prefix; shown on the Training and Training explorer pages
TRAIN_LOC_FIGS={'Scandinavia':'dem+fch_merged_fch25','Alps':'dem+fch_merged_alps_fch2'}
# figure suffix -> caption, in the order shown (the polar plot is left out on purpose)
TRAIN_LOC_KINDS=[('map_height','elevation'),('map_slope','slope'),('map_aspect','aspect'),
                 ('map_fch_2021','forest canopy height 2021'),('hist','distributions')]

# validation locations maps in the snowmodel directory: area -> file name; shown on the Independent validation page
VAL_LOC_FIGS={'Scandinavia':'validation_dem+fch_merged_fin25_10000_fscs1_scandi_map.png',
              'Alps':'validation_dem+fch_merged_fin2_10000_fscs1_alps_map.png'}

# model editions = optuna_tag of the training run -> name shown in the app; unknown tags are shown as they are
EDITION_NAMES={'FINmulti':'FINmulti','MANYfeatures':'MANYfeatures','FIN2multiNoMax':'FIN2multiNoMax','FIN5noclim':'FIN5noclim','SnowWaterEq':'SnowWaterEq','swe-ml-2':'swe-ml-2 (final model)'}

# Overview page text: what the SWE ML work is and how the pages fit together; edit freely (markdown)
OVERVIEW_INTRO=('The CryoSCOPE SWE model is an XGBoost trained machine learning model that estimates daily snow water equivalent (SWE) '
                'from weather, terrain and land cover data, for two domains: the Alps and Fenno-Scandinavia. '
                'Target "truth" in training is SWE from Snowcap ([snowcap.info](https://snowcap.info)), a daily 500 m product. '
                'The end product is SWE forecasts, driven by ECMWF ensemble forecasts from 10 to 15 days (ECXENS) '
                'up to a full season of 215 days (ECXSF).')
# Overview page, under the text: one row of logos with their links under them (logo file, width in pixels, link, link text),
# then the funding statement banner under the row; a missing logo file is skipped, its link is still shown
OVERVIEW_LOGOS=[('cryo.png',260,'https://cryoscope-project.eu/','cryoscope-project.eu'),
                ('snowcap.png',120,'https://snowcap.info','snowcap.info'),
                ('fmi.png',300,'https://trafficability.xyz','trafficability.xyz')]
OVERVIEW_FUNDING=('cryoscope.png',760)
# source code of the model, the prediction and validation scripts and this app; shown on the Overview page under the intro
REPO_URL='https://github.com/fmidev/ml-cryoscope'
# Overview page: the pages by section, drawn as a grid of links (section, short note, accent colour, [(page, url_path, what it shows)]);
# url_path must match NAV at the end of the file
OVERVIEW_SECTIONS=[
    ('Latest SWE predictions','the newest runs','#c0392b',[
        ('Map and time series','latest','the newest ECXENS and ECXSF runs on a web map; click a spot for its ensemble member time series, '
                                       'with observed SWE of this snow season where available')]),
    ('Final model','the swe-ml-2 model behind the forecasts','#1f5f99',[
        ('Training and model performance','training','training locations, SHAP, model performance metrics'),
        ('Independent validation','validation','validation locations, independent validation performance metrics')]),
    ('All model versions','every training iteration','#5f7387',[
        ('Training explorer','training-explorer','every training iteration (model edition), with the editions compared side by side'),
        ('Validation explorer','validation-explorer','the same validation views for every iteration, and all iterations compared')]),
    ('SWE predictions (past)','final model against HKL (9 km: ERA5-Land), by start date','#2b8a9a',[
        ('10 and 15 days (500 m)','prediction-10-15-days','ECXENS forecasts: the downscaled 10 or 15 day EC ensemble, daily'),
        ('Seasonal 215 days (2.8 km)','prediction-seasonal-2p8km','ECXSF monthly means: the downscaled EC seasonal forecast on the 2.8 km grid'),
        ('Seasonal 215 days (9 km)','prediction-seasonal-9km','ECXSF monthly means over Europe against ERA5-Land, with time series and scores per region')]),
    ('Compare model predictions','model editions side by side','#7a6a9e',[
        ('10 and 15 days','compare-10-15-days','ECXENS of several model editions side by side, or any edition in detail'),
        ('Seasonal','compare-seasonal','ECXSF of several model editions side by side (2.8 km, 9 km or both), or any edition in detail')]),
]
# colours inherit the light or dark theme; only the thin accent bar and the hover tint carry the section colour
OVERVIEW_CSS=(
    '<style>'
    '.ov{margin:1.4rem 0 0.4rem 0}'
    '.ov-sec{display:grid;grid-template-columns:minmax(180px,230px) 1fr;gap:1.2rem 2rem;padding:1.15rem 0;'
    'border-top:1px solid rgba(128,128,128,0.22)}'
    '.ov-sec:last-child{border-bottom:1px solid rgba(128,128,128,0.22)}'
    '.ov-head .n{font-size:1.08rem;font-weight:600;line-height:1.3}'
    '.ov-head .s{font-size:0.86rem;opacity:0.68;margin-top:0.3rem;line-height:1.4}'
    '.ov-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:0.75rem}'
    'a.ov-item{display:block;padding:0.8rem 1rem 0.85rem 1rem;border-radius:6px;border-left:3px solid var(--acc);'
    'background:rgba(128,128,128,0.07);color:inherit !important;text-decoration:none !important;transition:background 0.15s}'
    'a.ov-item:hover{background:color-mix(in srgb,var(--acc) 14%,transparent)}'
    'a.ov-item:focus-visible{outline:2px solid var(--acc);outline-offset:2px}'
    '.ov-item .t{font-weight:600;font-size:0.98rem;line-height:1.3}'
    '.ov-item .d{font-size:0.86rem;opacity:0.75;margin-top:0.3rem;line-height:1.45}'
    '@media (max-width:1100px){.ov-grid{grid-template-columns:repeat(2,minmax(0,1fr))}}'
    '@media (max-width:760px){.ov-sec{grid-template-columns:1fr;gap:0.6rem}.ov-grid{grid-template-columns:1fr}}'
    '@media (prefers-reduced-motion:reduce){a.ov-item{transition:none}}'
    '</style>')
def overview_pages_html():
    # one row per section: name and note on the left, its pages as links on the right; links open in the same tab
    from html import escape as e
    rows=[]
    for sec,note,acc,pages in OVERVIEW_SECTIONS:
        items=''.join(f'<a class="ov-item" href="{e(url)}" target="_self" style="--acc:{acc}">'
                      f'<div class="t">{e(title)}</div><div class="d">{e(desc)}</div></a>' for title,url,desc in pages)
        rows.append(f'<div class="ov-sec"><div class="ov-head"><div class="n">{e(sec)}</div><div class="s">{e(note)}</div></div>'
                    f'<div class="ov-grid">{items}</div></div>')
    return OVERVIEW_CSS+'<div class="ov">'+''.join(rows)+'</div>'

# feature descriptions for the Features tab: name -> (group, description, units)
# units are ECMWF native units unless the fetch converted them; edit freely, unknown features still show up as 'other'
FEATURE_INFO={
    'latitude':('location','latitude','°N'),
    'longitude':('location','longitude','°E'),
    'height':('terrain','elevation','m'),
    'slope':('terrain','terrain slope','°'),
    'aspect':('terrain','terrain aspect','°'),
    'tcf':('land cover','tree cover fraction','0..1'),
    'laihv':('land cover','leaf area index, high vegetation','m² m⁻²'),
    'lailv':('land cover','leaf area index, low vegetation','m² m⁻²'),
    'swe_mean':('SNOWCAP climatology','SWE climatology, mean','mm'),
    'swe_max':('SNOWCAP climatology','SWE climatology, max','mm'),
    'swe_min':('SNOWCAP climatology','SWE climatology, min','mm'),
    'snowdepth_mean':('SNOWCAP climatology','snow depth climatology, mean',''),
    'snowdepth_max':('SNOWCAP climatology','snow depth climatology, max',''),
    'snowdepth_min':('SNOWCAP climatology','snow depth climatology, min',''),
    'tp':('forcing: precipitation','total precipitation','m'),
    'sf':('forcing: precipitation','snowfall','m w.e.'),
    'slhf':('forcing: energy','surface latent heat flux','J m⁻²'),
    'sshf':('forcing: energy','surface sensible heat flux','J m⁻²'),
    'ssr':('forcing: energy','surface net solar radiation','J m⁻²'),
    'str':('forcing: energy','surface net thermal radiation','J m⁻²'),
    't2':('forcing: temperature','2 m temperature','K'),
    'td2':('forcing: temperature','2 m dewpoint temperature','K'),
    'tmax_max':('forcing: temperature','daily maximum 2 m temperature','K'),
    'tmin_min':('forcing: temperature','daily minimum 2 m temperature','K'),
    'skt':('forcing: temperature','skin temperature','K'),
    'stl1':('forcing: temperature','soil temperature, level 1','K'),
    'u10':('forcing: wind & pressure','10 m wind, eastward','m s⁻¹'),
    'v10':('forcing: wind & pressure','10 m wind, northward','m s⁻¹'),
    'sp':('forcing: wind & pressure','surface pressure','Pa'),
    'ro':('reanalysis snow & hydrology','runoff','m'),
    'sro':('reanalysis snow & hydrology','surface runoff','m'),
    'ssro':('reanalysis snow & hydrology','sub-surface runoff','m'),
    'rsn':('reanalysis snow & hydrology','snow density','kg m⁻³'),
    'sd':('reanalysis snow & hydrology','snow depth water equivalent','m w.e.'),
    'smlt':('reanalysis snow & hydrology','snowmelt','m w.e.'),
    'es':('reanalysis snow & hydrology','snow evaporation','m w.e.'),
    'e':('reanalysis snow & hydrology','total evaporation','m w.e.'),
    'pev':('reanalysis snow & hydrology','potential evaporation','m'),
    'asn':('reanalysis snow & hydrology','snow albedo','0..1'),
    'tsn':('reanalysis snow & hydrology','temperature of snow layer','K'),
    'src':('reanalysis snow & hydrology','skin reservoir content','m w.e.'),
    'swvl1':('reanalysis snow & hydrology','volumetric soil water, layer 1','m³ m⁻³'),
    'swvl2':('reanalysis snow & hydrology','volumetric soil water, layer 2','m³ m⁻³'),
    'doy_cos':('time','day of year, cosine',''),
    'doy_sin':('time','day of year, sine',''),
}
# pressure level features q500, t700, z850, ...: one entry per variable instead of 15 dict lines
PLEV_VARS={'q':('specific humidity','kg kg⁻¹'),'t':('temperature','K'),'u':('wind, eastward','m s⁻¹'),
           'v':('wind, northward','m s⁻¹'),'z':('geopotential','m² s⁻²')}
def feature_info(f):
    if f in FEATURE_INFO: return FEATURE_INFO[f]
    if len(f)==4 and f[0] in PLEV_VARS and f[1:] in ('500','700','850'):
        d,u=PLEV_VARS[f[0]]; return 'forcing: upper air',f'{d}, {f[1:]} hPa',u
    if f.endswith('_lag1'):
        g,d,u=feature_info(f[:-5]); return g,f'{d}, previous day (lag 1)',u
    if f.endswith('_00'):
        g,d,u=feature_info(f[:-3]); return g,f'{d}, 00 UTC',u
    return 'other','',''

def edition_name(tag): return EDITION_NAMES.get(tag,tag)
def edition_label(tag,runs):
    n=next((r.get('n_features') for r in runs if r['optuna_tag']==tag),None)
    return f'{edition_name(tag)} ({n} features)' if n else edition_name(tag)

# ---------------------------------------------------------------
# Sidebar: navigation and paths
# ---------------------------------------------------------------
# app wordmark under the CryoSCOPE logo, above the page menu; the plain title when the png is missing
if logo('snowlook.png'): st.sidebar.image(logo('snowlook.png'),width=240)
else: st.sidebar.title(f'❄️ {APP_NAME}')
_NAV_BOX=st.sidebar.container()   # the page menu (NAV, end of the file) is drawn here, between the title and the data paths
AI_NOTICE='Created with help from AI'   # shown at the bottom of the sidebar
# public site (SNOWLOOK_PUBLIC=1, set in the docker image): the paths come from the environment only, no 'data paths' box and
# no 'reload files' button in the sidebar, so visitors can not point the app at other folders of the server
PUBLIC=os.environ.get('SNOWLOOK_PUBLIC','0')=='1'
def path_input(label,value): return value if PUBLIC else st.text_input(label,value)
with (contextlib.nullcontext() if PUBLIC else st.sidebar.expander('data paths',expanded=False)):
    base=path_input('snowmodel directory',os.environ.get('SNOWMODEL_DIR','/home/ubuntu/data/cryoscope/snowmodel/'))
    optuna_db=path_input('optuna sqlite',os.environ.get('OPTUNA_DB','/home/ubuntu/data/ML/MLexperiments.sqlite3'))
    PRED_DIR=path_input('10 day (ECXENS) validation directory',os.environ.get('SWE_PRED_DIR',os.path.join(base,'validation_ecens')))
    SF_DIR=path_input('seasonal 2.8 km (ECXSF) validation directory',os.environ.get('SWE_SF_DIR',os.path.join(base,'validation_sf')))
    # 9 km: validation_sf9km/ or, if only that one exists, validationresults_sf9km/
    _sf9=next((p for p in (os.path.join(base,d) for d in ('validation_sf9km','validationresults_sf9km')) if os.path.isdir(p)),os.path.join(base,'validation_sf9km'))
    SF9_DIR=path_input('seasonal 9 km (ECXSF) validation directory',os.environ.get('SWE_SF9_DIR',_sf9))
    # Latest SWE predictions page: timeseries requests go from this server; the WMS address must be public, the browser loads the tiles itself
    SMARTMET_URL=path_input('SmartMet Server (latest predictions, timeseries)',os.environ.get('SWE_SMARTMET_URL','https://sm.cryo-scope.eu')).rstrip('/')
    WMS_URL=path_input('WMS address (latest predictions, public)',
                          os.environ.get('SWE_WMS_URL',(SMARTMET_URL+'/wms') if SMARTMET_URL else '')).rstrip('/')

def read_json(p):
    try:
        with open(p) as f: return json.load(f)
    except Exception: return {}

# validation folders written by the validation script (validationresults_<optuna_tag>/), newest first
# only validationresults_<tag>/: the plain validationresults/ folder from before the per-model folders is left out
val_dirs=sorted((os.path.basename(p) for p in glob.glob(os.path.join(base,'validationresults_*')) if os.path.isdir(p)),
                key=lambda d:-os.path.getmtime(os.path.join(base,d)))
def val_dir_tag(d):
    # model tag of a validation folder: optuna_tag in run_info.json, else the folder name after validationresults_
    tag=read_json(os.path.join(base,d,'run_info.json')).get('optuna_tag')
    if tag: return tag
    return d[len('validationresults_'):] if d.startswith('validationresults_') else d
def val_complete(d):
    # unfinished runs have complete=False in run_info.json; older folders without the flag count as finished
    return read_json(os.path.join(base,d,'run_info.json')).get('complete',True)
# validated model editions: tag -> folder (newest folder wins if two hold the same tag); folders without scores are left out
VAL_BY_TAG={}
for _d in val_dirs:
    if os.path.exists(os.path.join(base,_d,'validation_metrics.csv')): VAL_BY_TAG.setdefault(val_dir_tag(_d),_d)
def val_dir_for(stem):
    # validation folder whose run_info.json points to this training run
    for d in val_dirs:
        if read_json(os.path.join(base,d,'run_info.json')).get('training_stem')==stem: return d
    return None
MOD_DIR=os.path.join(base,'models'); RES_DIR=os.path.join(base,'results')
VAL_DIR=FIG_DIR=None   # set by use_val() on the Independent validation and Validation explorer pages
if not PUBLIC and st.sidebar.button('reload files'):
    st.cache_data.clear(); st.cache_resource.clear()

# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------
def mtime(p): return os.path.getmtime(p) if os.path.exists(p) else 0.0
def stamp(p): return pd.Timestamp(mtime(p),unit='s').tz_localize('UTC').tz_convert('Europe/Helsinki').strftime('%Y-%m-%d %H:%M')

@st.cache_data(show_spinner=False)
def _read_csv(p,mt): return pd.read_csv(p)
def read_csv(p): return _read_csv(p,mtime(p)) if os.path.exists(p) else None

def stretch(fn,*a,**k):
    # newer streamlit uses width='stretch', older use_container_width=True
    try: return fn(*a,width='stretch',**k)
    except Exception: return fn(*a,use_container_width=True,**k)
def img(p,caption=None):
    if os.path.exists(p): stretch(st.image,p,caption=caption)
    else: st.info(f'not found: {os.path.basename(p)}')
def chart(fig,**k): return stretch(st.plotly_chart,fig,**k)
def table(df,**k): return stretch(st.dataframe,df,**k)
def fmt(v,f):
    try: return f.format(v) if v is not None and pd.notna(v) else 'n/a'
    except (ValueError,TypeError): return str(v)

def quick_scores(o,p):
    o=np.asarray(o,float); p=np.asarray(p,float); ok=np.isfinite(o)&np.isfinite(p); o,p=o[ok],p[ok]
    if len(o)<2: return {'n':len(o)}
    e=p-o; so,sp=o.std(),p.std()
    r=np.corrcoef(o,p)[0,1] if so>0 and sp>0 else np.nan
    a=sp/so if so>0 else np.nan; b=p.mean()/o.mean() if o.mean()>0 else np.nan
    return {'n':len(o),'RMSE':np.sqrt(np.mean(e**2)),'ME':e.mean(),'MAE':np.abs(e).mean(),'r':r,
            'KGE':1-np.sqrt((r-1)**2+(a-1)**2+(b-1)**2),'peak obs':o.max(),'peak pred':p.max()}

def is_err(col): return col=='RMSE' or col.startswith('RMSE_') or col in ('ubRMSE','MAE')

def err_scale(err_max):
    # viridis from 0 to err_max, red above; the range ends 10 % above err_max so the red band shows in the colour bar
    top=err_max*1.1; f=err_max/top
    vir=px.colors.sample_colorscale('Viridis',[i/10 for i in range(11)])
    return [[f*i/10,c] for i,c in enumerate(vir)]+[[f+1e-6,'red'],[1.0,'red']],(0,top)

def color_args(s,col,err_max=ERR_MAX):
    v=pd.Series(s).replace([np.inf,-np.inf],np.nan).dropna()
    if v.empty: return 'Viridis',None
    if is_err(col): return err_scale(err_max)   # same limits for every predictor, domain and run
    if col.startswith('skill'): return 'RdBu',(-1,1)
    if col=='ME' or 'minus' in col: return 'RdBu_r',(-ME_MAX,ME_MAX)
    if col in ('r','pearson_r'): return 'Magma',(max(0.0,float(np.nanpercentile(v,2))),1)
    if col in ('height','dist_to_train_m'): return 'Viridis',(float(v.min()),float(v.max()))
    return 'Viridis',(0,float(np.nanpercentile(v,98)) or 1.0)

def map_fig(d,col,hover=None,height=600,title=None,rng=None,cscale=None,err_max=ERR_MAX):
    cs,cr=color_args(d[col],col,err_max)
    kw=dict(lat='latitude',lon='longitude',color=col,color_continuous_scale=cscale or cs,range_color=rng or cr,
            hover_data=hover,custom_data=['latitude','longitude'],height=height,title=title,
            zoom=3 if d['longitude'].max()-d['longitude'].min()>12 else 4.5,
            center=dict(lat=float(d['latitude'].mean()),lon=float(d['longitude'].mean())))
    try: fig=px.scatter_map(d,map_style='carto-positron',**kw)
    except AttributeError: fig=px.scatter_mapbox(d,mapbox_style='carto-positron',**kw)  # plotly < 5.24
    fig.update_traces(marker=dict(size=5))
    fig.update_layout(margin=dict(l=0,r=0,t=40 if title else 10,b=0))
    if is_err(col) and rng is None and cscale is None:
        tv=list(np.linspace(0,err_max,4))
        fig.update_coloraxes(colorbar=dict(tickvals=tv+[err_max*1.05],ticktext=[f'{x:g}' for x in tv]+[f'>{err_max:g}']))
    return fig

def click_latlon(p):
    # lat/lon of a clicked map point; streamlit returns lat/lon directly for map traces,
    # customdata can be a list, a dict or missing (e.g. the red 'selected' marker), so try both
    lat,lon=p.get('lat'),p.get('lon')
    if lat is not None and lon is not None:
        try: return float(lat),float(lon)
        except (TypeError,ValueError): pass
    cd=p.get('customdata')
    if isinstance(cd,dict): cd=list(cd.values())
    if isinstance(cd,(list,tuple)) and len(cd)>=2:
        try: return float(cd[0]),float(cd[1])
        except (TypeError,ValueError): pass
    return None

def hbar(d,x,title,n):
    d=d.sort_values(x,ascending=False).head(n).iloc[::-1]
    fig=px.bar(d,x=x,y='feature',orientation='h',title=title,height=22*len(d)+120)
    fig.update_layout(margin=dict(l=0,r=10,t=40,b=0),yaxis_title=None)
    return fig

def train_vs_indep(h,vm):
    # tuning validation (held-out seasons at training locations, drove tuning and early stopping) vs independent validation locations
    v=vm[(~vm['model'].astype(str).str.startswith('baseline_'))&(vm['subset']=='all_rows')]
    bydom=h.get('valid_metrics_by_domain') or {}
    rows=[]
    for _,r in v.iterrows():
        dom=r['domain']
        tr=({'rmse':h.get('valid_rmse_all'),'kge':h.get('valid_kge_all')} if dom=='all' else bydom.get(dom,{}))
        rows.append({'domain':dom,'RMSE tuning valid':tr.get('rmse'),'RMSE independent':r.get('RMSE'),
                     'KGE tuning valid':tr.get('kge'),'KGE independent':r.get('KGE'),
                     'R² independent':r.get('R2'),'ME independent':r.get('ME')})
    return pd.DataFrame(rows)

def run_features(h,g=None,s=None):
    # feature list in model column order; older hyperparams files without 'features' fall back to the importance csv
    if h.get('features'): return list(h['features'])
    for d in (g,s):
        if d is not None and 'feature' in d: return d['feature'].tolist()
    return []

def feature_table(feats,g,s):
    t=pd.DataFrame({'feature':feats})
    info=[feature_info(f) for f in feats]
    t['group']=[i[0] for i in info]; t['description']=[i[1] for i in info]; t['units']=[i[2] for i in info]
    for d,col,lab in [(s,'mean_abs_shap','shap'),(g,'gain','gain')]:   # SHAP columns right after units, gain after them
        if d is None or col not in d: continue
        t=t.merge(d[['feature',col]],on='feature',how='left')
        tot=t[col].sum()
        t[f'{lab}_%']=100*t[col]/tot if tot>0 else np.nan
        t[f'{lab}_rank']=t[col].rank(ascending=False,method='min').astype('Int64')
    return t

# ---------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------
@st.cache_data(ttl=60,show_spinner=False)
def training_runs(mod_dir):
    out=[]
    for p in glob.glob(os.path.join(mod_dir,'hyperparams_swe_*.json')):
        try:
            with open(p) as f: h=json.load(f)
        except Exception: continue
        stem=os.path.basename(p)[len('hyperparams_swe_'):-len('.json')]  # variant_domain_optunatag
        parts=stem.split('_')
        h.setdefault('variant',parts[0]); h.setdefault('domain',parts[1] if len(parts)>1 else '')
        h['optuna_tag']=stem[len(h['variant'])+len(h['domain'])+2:]
        h['stem']=stem; h['modified']=stamp(p); h['_mtime']=mtime(p); h['_path']=p
        out.append(h)
    return sorted(out,key=lambda h:-h['_mtime'])

def use_val(d):
    # point the validation pages to one validation folder
    global VAL_DIR,FIG_DIR
    VAL_DIR=os.path.join(base,d); FIG_DIR=os.path.join(VAL_DIR,'figs')

def val_models():
    # training runs, validated model editions (newest training first) and editions not validated yet
    runs=training_runs(MOD_DIR)
    trained=list(dict.fromkeys(r['optuna_tag'] for r in runs))   # newest training run first
    tags=[t for t in trained if t in VAL_BY_TAG]+sorted(t for t in VAL_BY_TAG if t not in trained)
    todo=[t for t in trained if t not in VAL_BY_TAG]
    return runs,tags,todo

def pick_val_model():
    # model edition selection for the Validation explorer page; shares the choice with the Training explorer page
    runs,tags,todo=val_models()
    if not tags:
        st.warning(f'no validationresults_<tag>/ folders with validation_metrics.csv in {base}')
        if todo: st.code('python xgb-validate-cryo-swe.py missing',language='bash')
        return None
    keep=st.session_state.get('edition_keep')
    tag=st.radio('model edition',tags,index=tags.index(keep) if keep in tags else 0,horizontal=True,
                 format_func=lambda t:edition_label(t,runs)+('' if val_complete(VAL_BY_TAG[t]) else ' (running)'))
    st.session_state['edition_keep']=tag
    use_val(VAL_BY_TAG[tag])
    if todo:
        st.caption('not validated yet: '+', '.join(edition_label(t,runs) for t in todo)+
                   '. ')
    return tag

@st.cache_data(show_spinner='reading optuna study')
def load_study(db,name,mt):
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        t=optuna.load_study(study_name=name,storage=f'sqlite:///{db}').trials_dataframe()
        t=t[t['state']=='COMPLETE']
        return t if 'values_0' in t else None
    except Exception:
        return None

@st.cache_resource(show_spinner='loading predictions (first load converts the csv.gz to parquet)')
def load_pred(path,mt):
    # parquet cache next to the predictions, in <validation folder>/app_cache/, so every model has its own
    pq=os.path.join(os.path.dirname(path),'app_cache',os.path.basename(path).replace('.csv.gz','.parquet'))
    if os.path.exists(pq) and mtime(pq)>=mt:
        return pd.read_parquet(pq)
    df=pd.read_csv(path,parse_dates=['utctime'])
    f64=df.select_dtypes('float64').columns.difference(['latitude','longitude'])
    df[f64]=df[f64].astype(np.float32)
    df['tag']=df['tag'].astype('category')
    try:
        os.makedirs(os.path.dirname(pq),exist_ok=True); df.to_parquet(pq,index=False)
    except Exception as e:
        print('parquet cache not written:',e)
    return df

@st.cache_resource(show_spinner=False)
def loc_index(path,mt):
    return load_pred(path,mt).groupby(['latitude','longitude'],sort=False).indices

@st.cache_data(show_spinner='computing per location scores')
def loc_table(path,mt):
    df=load_pred(path,mt)
    e=df['SWE_xgboost']-df['SWE_observed']
    agg=dict(tag=('tag','first'),height=('height','first'),n=('err','size'),obs_mean=('SWE_observed','mean'),
             obs_max=('SWE_observed','max'),pred_mean=('SWE_xgboost','mean'),ME=('err','mean'),MSE=('err2','mean'))
    if 'dist_to_train_m' in df: agg['dist_to_train_m']=('dist_to_train_m','first')
    t=df.assign(err=e,err2=e**2).groupby(['latitude','longitude'],sort=False).agg(**agg).reset_index()
    t['RMSE']=np.sqrt(t.pop('MSE')); t['tag']=t['tag'].astype(str)
    return t

@st.cache_data(show_spinner='computing daily domain means')
def domain_mean(path,mt):
    df=load_pred(path,mt); cols=[c for c in PRED_STYLE if c in df]
    return df.groupby([df['utctime'].dt.normalize().rename('date'),df['tag'].astype(str).rename('tag')])[cols].mean().reset_index()

def nearest(lt,lat,lon):
    d=(lt['latitude']-lat)**2+((lt['longitude']-lon)*np.cos(np.deg2rad(lat)))**2
    r=lt.loc[d.idxmin()]
    return float(r['latitude']),float(r['longitude'])

def fc_sort_key(p):
    # prediction folders in FC_ORDER first (ensmean, member01, wettest, driest), any others after them by name
    lab=os.path.basename(p.rstrip('/'))
    return (FC_ORDER.index(lab) if lab in FC_ORDER else len(FC_ORDER),lab)

def fc_subdirs(p):
    try:
        return sorted(d for d in os.listdir(p) if os.path.isdir(os.path.join(p,d)) and not d.startswith('.') and d not in FC_SKIP_DIRS)
    except OSError:
        return []

def fc_pred_dirs(p):
    # prediction folders directly under p (those with swe_metrics.csv), in FC_ORDER
    return sorted((os.path.join(p,d) for d in fc_subdirs(p) if os.path.exists(os.path.join(p,d,'swe_metrics.csv'))),key=fc_sort_key)

def fc_date_from_name(name):
    for f in ('%Y%m%d','%Y%m%dT%H%M%S','%Y%m%d%H','%Y-%m-%d','%Y%m'):   # %Y%m: seasonal folders like 202601
        try: return pd.to_datetime(name,format=f)
        except (ValueError,TypeError): pass
    return None

def fc_base_from_scores(pred):
    # most common (date minus lead day) in one prediction's scores
    try:
        d=pd.read_csv(os.path.join(pred,'swe_metrics.csv'),usecols=['date','lead_day'])
        m=(pd.to_datetime(d['date'])-pd.to_timedelta(d['lead_day'],unit='D')).mode()
        return m.iloc[0] if len(m) else None
    except Exception:
        return None

def fc_region_name(r):
    # 'alps' / 'Alps' -> 'Alps'; unknown names are shown as they are
    return FC_REGIONS.get(str(r).lower(),str(r)) if r else None

def fc_split_model(folder):
    # model folder name -> (model, region): 'FIN5noclim_Alps' -> ('FIN5noclim','Alps'), 'FIN2multiNoMax' -> ('FIN2multiNoMax',None)
    for sep in ('_','-'):
        if sep in folder:
            head,tail=folder.rsplit(sep,1)
            if tail.lower() in FC_REGIONS: return head,FC_REGIONS[tail.lower()]
    return folder,None

def fc_regions(runs):
    # regions of the runs, FC_DEFAULT_REGION first
    return sorted({r['region'] for r in runs},key=lambda x:(x!=FC_DEFAULT_REGION,x))

def pick_region(runs,key):
    # region radio when the runs hold more than one region; returns the region and the runs of that region
    regions=fc_regions(runs)
    if len(regions)>1:
        if st.session_state.get(key) not in regions: st.session_state[key]=regions[0]
        region=st.radio('region',regions,horizontal=True,key=key)
    else:
        region=regions[0]
    return region,[r for r in runs if r['region']==region]

@st.cache_data(ttl=60,show_spinner=False)
def fc_runs(pdir,known=()):
    # forecast validation runs: validation/<model>/<start date>/<prediction>/swe_metrics.csv; one run = one model, region and start date.
    # newest start date first
    found=[]   # (run folder, model folder name)
    for m in fc_subdirs(pdir):
        mdir=os.path.join(pdir,m)
        if fc_pred_dirs(mdir): found.append((mdir,m))                  # old layout: validation/<model>/<prediction>/
        for d in fc_subdirs(mdir):
            if fc_pred_dirs(os.path.join(mdir,d)): found.append((os.path.join(mdir,d),m))
    if fc_pred_dirs(pdir): found.append((pdir,FC_LEGACY_MODEL))      # oldest layout: validation/<prediction>/
    out=[]
    for run,mfolder in found:
        model,region=fc_split_model(mfolder)
        preds=fc_pred_dirs(run)
        info=read_json(os.path.join(run,'run_info.json'))
        bt=None
        if info.get('base_time'):
            try: bt=pd.Timestamp(info['base_time'])
            except (ValueError,TypeError): bt=None
        if bt is None and run!=pdir: bt=fc_date_from_name(os.path.basename(run))
        if bt is None: bt=fc_base_from_scores(preds[0])
        region=fc_region_name(info.get('region')) or region or FC_DEFAULT_REGION
        out.append({'run':run,'preds':preds,'base_time':bt,'model':MODEL_ALIASES.get(info.get('model') or model,info.get('model') or model),'region':region})
    # start dates in CASE_HIDE are left out here, so they are in no table, chart or menu of any page
    out=[r for r in out if r['base_time'] is None or not case_hidden(f"{r['base_time']:%Y-%m-%d}")]
    return sorted(out,key=lambda r:r['base_time'] if r['base_time'] is not None else pd.Timestamp('1900-01-01'),reverse=True)
def fc_rel(run):
    return os.path.relpath(run,PRED_DIR) if os.path.normpath(run)!=os.path.normpath(PRED_DIR) else os.path.basename(PRED_DIR.rstrip('/'))

def fc_run_label(r):
    bt=r['base_time']
    when=(f"{bt:%Y-%m-%d}"+(f" {bt:%H} UTC" if bt.hour else '')) if bt is not None else 'start date unknown'
    return f"{when}  ({fc_rel(r['run'])}, {len(r['preds'])} predictions)"

def fc_name(pred_dir):
    # shown name: pred_info.json written by the validation script (e.g. 'ECXENS member 23 (wettest)'),
    # then FC_NAMES, then member<N> (member1, member01, ...) -> 'ECXENS member N'
    name=read_json(os.path.join(pred_dir,'pred_info.json')).get('name')
    if name: return name
    lab=os.path.basename(pred_dir.rstrip('/'))
    if lab in FC_NAMES: return FC_NAMES[lab]
    if lab.startswith('member') and lab[6:].isdigit(): return f'ECXENS member {int(lab[6:])}'
    return lab

def fc_metrics(preds):
    # daily scores of every prediction in the run, with 'pred' (folder) and 'name' (shown) columns
    parts=[]
    for p in preds:
        d=read_csv(os.path.join(p,'swe_metrics.csv'))
        if d is not None: parts.append(d.assign(pred=os.path.basename(p),name=fc_name(p)))
    if not parts: return None
    df=pd.concat(parts,ignore_index=True)
    df['date']=pd.to_datetime(df['date'])
    return df

@st.cache_data(show_spinner=False)
def _img_ratio(p,mt):
    try:
        from PIL import Image
        with Image.open(p) as im: w,h=im.size
        return w/h
    except Exception:
        return None
def img_ratio(p,default):
    # width / height of a png, so images side by side in st.columns get the same height
    r=_img_ratio(p,mtime(p)) if os.path.exists(p) else None
    return r or default

# ---------------------------------------------------------------
# Start dates (cases) on the SWE predictions (past) pages of the final model
# ---------------------------------------------------------------
def case_label(cases,region,k):
    # '2018-01-19: level 5 snowfall, rain at lower elevations'
    desc,out=cases.get(region,{}).get(k,('',None))
    return k+(f': {desc}' if desc else '')

def region_choice(runs,cases,key):
    # region radio over the regions with results and the regions with planned start dates; returns the region and its runs
    regions=sorted({r['region'] for r in runs}|set(cases),key=lambda x:(x!=FC_DEFAULT_REGION,x))
    if len(regions)>1:
        if st.session_state.get(key) not in regions: st.session_state[key]=regions[0]
        region=st.radio('region',regions,horizontal=True,key=key)
    else:
        region=regions[0]
    return region,[r for r in runs if r['region']==region]

def case_overview(cases,runs,date_key,scores,key,chart_box=True,sources=None):
    # table of every planned start date and every start date with results, mean scores of the ens mean where run;
    # returns {start date: run} of the start dates with results (CASE_HIDE left out of both)
    # sources: {column name: {start date: url}}, each shown as a link column after the case description
    sources=sources or {}
    by={}
    for r in runs:
        if not case_hidden(date_key(r)): by.setdefault(date_key(r),r)
    rows=[]
    for k in sorted(k for k in set(cases)|set(by) if not case_hidden(k)):
        desc,out=cases.get(k,('',None))
        row={'start date':k,'case':desc}
        row.update({c:d.get(k,'') for c,d in sources.items()})
        row.update({'results':'yes' if k in by else 'not run yet','_out':bool(out)})
        if k in by: row.update(scores(by[k]))
        rows.append(row)
    if not rows:
        return by
    t=pd.DataFrame(rows)
    cfg={c:st.column_config.LinkColumn(c,display_text=r'https?://(?:www\.)?([^/]+)') for c in sources}
    table(t.drop(columns=['_out']).round(3),hide_index=True,column_config=cfg)
    num=[c for c in t.columns if c not in ('start date','case','results','_out') and c not in sources and pd.api.types.is_numeric_dtype(t[c])
         and t[c].notna().any()]
    if num and chart_box:   # chart_box=False: no "scores per start date, chart" expander
        with st.expander('scores per start date, chart',expanded=False):
            m=st.radio('score',num,index=num.index('RMSE (mm)') if 'RMSE (mm)' in num else 0,horizontal=True,key=f'{key}_met')
            d=t.dropna(subset=[m]).assign(seasons=lambda x:np.where(x['_out'],'out of the training seasons','in the training seasons'))
            f=px.bar(d,x='start date',y=m,color='seasons',hover_data=['case'],height=360,
                     color_discrete_map={'in the training seasons':'#1f77b4','out of the training seasons':'#ff7f0e'},labels={'seasons':''})
            if 'bias' in m or 'skill' in m: f.add_hline(y=0,line_color='black',line_width=0.6)
            f.update_layout(xaxis_type='category',legend=dict(orientation='h',y=1.1),margin=dict(t=30))
            chart(f,key=f'{key}_chart')
    return by

# ---------------------------------------------------------------
# Page: overview
# ---------------------------------------------------------------
SNOWFALL=True          # falling snowflakes on the overview page
SNOWFLAKES=40          # number of flakes
SNOW_PILE=True         # snow piles up at the bottom of the window while the overview is open; starts again on every visit or reload
SNOW_PILE_PX=56        # highest the pile gets, in pixels; the page gets the same space at its end, so the pile never covers the last content
SNOW_PILE_S=120        # seconds the pile takes to reach that height
def snowfall_html(n=SNOWFLAKES):
    # CSS only falling snow (Streamlit strips scripts): flakes in a fixed layer over the page, clicks pass through,
    # light blue so they show on both a light and a dark theme; switched off for viewers who ask for reduced motion
    rng=np.random.default_rng(7)   # fixed seed, so the flakes do not jump on every rerun
    flakes=[]
    for i in range(n):
        left=rng.uniform(0,100); size=rng.uniform(0.6,1.5); dur=rng.uniform(9,20); delay=-rng.uniform(0,20)
        sway=rng.uniform(15,45)*(1 if i%2 else -1); op=rng.uniform(0.35,0.85)
        flakes.append(f'<span class="snow-f" style="left:{left:.1f}vw;font-size:{size:.2f}rem;opacity:{op:.2f};'
                      f'animation-duration:{dur:.1f}s,{dur/3:.1f}s;animation-delay:{delay:.1f}s,{delay:.1f}s;--sway:{sway:.0f}px">❄</span>')
    css=('<style>'
         '.snow-layer{position:fixed;inset:0;pointer-events:none;overflow:hidden;z-index:999990}'
         '.snow-f{position:absolute;top:-2rem;color:#8ec5e6;text-shadow:0 0 2px rgba(255,255,255,0.6);'
         'animation-name:snow-fall,snow-sway;animation-timing-function:linear,ease-in-out;'
         'animation-iteration-count:infinite,infinite;animation-direction:normal,alternate}'
         '@keyframes snow-fall{to{transform:translateY(calc(100vh + 3rem))}}'
         '@keyframes snow-sway{to{margin-left:var(--sway)}}'
         '@media (prefers-reduced-motion:reduce){.snow-layer{display:none}}'
         '</style>')
    pile=''
    if SNOW_PILE:
        # two drifts with uneven tops (svg paths stretched to the window width), growing from nothing up to SNOW_PILE_PX;
        # the animation starts when the element is drawn, so opening the overview again or reloading starts from an empty ground
        from urllib.parse import quote
        def drift(path,fill,edge):
            svg=(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 100" preserveAspectRatio="none">'
                 f'<path d="{path}" fill="{fill}" stroke="{edge}" stroke-width="2" vector-effect="non-scaling-stroke"/></svg>')
            return 'url("data:image/svg+xml,'+quote(svg)+'")'
        back=drift('M0 40 C80 8 170 12 250 34 S420 2 530 22 S700 50 810 16 S1000 0 1090 28 S1180 30 1200 18 L1200 100 L0 100 Z','#dcebf5','#b9d6ea')
        front=drift('M0 58 C100 30 190 40 290 52 S460 22 580 46 S750 64 870 36 S1050 24 1140 50 S1190 52 1200 44 L1200 100 L0 100 Z','#f4f9fc','#c6dfee')
        css+=('<style>'
              f'.snow-pile{{position:fixed;left:0;right:0;bottom:0;height:{SNOW_PILE_PX}px;pointer-events:none;z-index:999991;'
              f'transform-origin:bottom;transform:scaleY(0);animation:snow-pile {SNOW_PILE_S}s cubic-bezier(.3,.1,.4,1) forwards}}'
              '.snow-pile i{position:absolute;inset:0;background-size:100% 100%;background-repeat:no-repeat}'
              f'.snow-pile .b{{background-image:{back}}}'
              f'.snow-pile .f{{background-image:{front};filter:drop-shadow(0 -1px 2px rgba(120,170,205,0.25))}}'
              '@keyframes snow-pile{to{transform:scaleY(1)}}'
              f'section.main .block-container,[data-testid="stMainBlockContainer"]{{padding-bottom:{SNOW_PILE_PX+48}px !important}}'
              '@media (prefers-reduced-motion:reduce){.snow-pile{display:none}}'
              '</style>')
        pile='<div class="snow-pile" aria-hidden="true"><i class="b"></i><i class="f"></i></div>'
    return css+'<div class="snow-layer" aria-hidden="true">'+''.join(flakes)+'</div>'+pile

def page_overview():
    # site description and the logos, no results (those are on the other pages)
    global SHOW_FOOTER
    SHOW_FOOTER=False
    if SNOWFALL: st.markdown(snowfall_html(),unsafe_allow_html=True)
    st.title(APP_NAME)
    st.markdown('**CryoSCOPE snow water equivalent (SWE) ML model**')
    st.markdown(OVERVIEW_INTRO)
    if REPO_URL: st.markdown(f'Code: [{REPO_URL.split("://")[-1]}]({REPO_URL})')
    st.markdown(overview_pages_html(),unsafe_allow_html=True)
    st.write('')
    # logos side by side, vertically centred in their row; the links in a row of their own, so they line up
    widths=[w for _,w,_,_ in OVERVIEW_LOGOS]
    try: row=st.columns(widths+[max(1,1400-sum(widths))],vertical_alignment='center',gap='large')
    except TypeError: row=st.columns(widths+[max(1,1400-sum(widths))])   # streamlit < 1.36
    links=st.columns(widths+[max(1,1400-sum(widths))],gap='large')
    for col,lcol,(name,width,url,text) in zip(row,links,OVERVIEW_LOGOS):
        p=logo(name)
        if p: col.image(p,width=width)
        if url: lcol.markdown(f'[{text}]({url})')
    st.write('')
    p=logo(OVERVIEW_FUNDING[0])
    if p: st.image(p,width=OVERVIEW_FUNDING[1])

def indep_val_all(runs,subset):
    # validation_metrics.csv of every validated model edition for one subset (all_rows or snow_rows), XGBoost rows named by edition
    parts=[]
    for tag,d in VAL_BY_TAG.items():
        v=read_csv(os.path.join(base,d,'validation_metrics.csv'))
        if v is None: continue
        v=v[v['subset']==subset].copy()
        xg=~v['model'].astype(str).str.startswith('baseline_')
        v['model']=v['model'].astype(str)
        v.loc[xg,'model']='XGBoost '+edition_label(tag,runs)
        parts.append(v)
    return pd.concat(parts,ignore_index=True).drop_duplicates(['model','domain']) if parts else None

# ---------------------------------------------------------------
# Pages: training and model performance (final model only) and training explorer (every model version)
# ---------------------------------------------------------------
def page_training(): training_view(explorer=False)
def page_training_explorer(): training_view(explorer=True)

def training_view(explorer):
    runs=training_runs(MOD_DIR)
    if not runs:
        st.warning(f'no hyperparams_swe_*.json in {MOD_DIR}'); return

    # model edition (optuna_tag), newest first
    eds=list(dict.fromkeys(r['optuna_tag'] for r in runs))
    if explorer:
        st.title('Training explorer')
        # kept in session_state so it survives page switches
        keep=st.session_state.get('edition_keep')
        ed=st.radio('model edition',eds,index=eds.index(keep) if keep in eds else 0,horizontal=True,
                    format_func=lambda t:edition_label(t,runs))
        st.session_state['edition_keep']=ed
    else:
        ed=used_model(eds)
        if ed is None:
            st.title('Training and model performance')
            st.warning(f'no training run of the final model {USED_MODEL} in {MOD_DIR}. Every run is on the Training explorer page.'); return
    er=[r for r in runs if r['optuna_tag']==ed]
    stem=st.selectbox('training run',[r['stem'] for r in er],key=f'run_{ed}') if len(er)>1 else er[0]['stem']
    h=next(r for r in er if r['stem']==stem)
    P=lambda name,ext='png': os.path.join(RES_DIR,f'{name}_{stem}.{ext}')

    if explorer: st.header(f'Training run: {edition_name(h["optuna_tag"])}')
    else: st.title(f'Training and model performance: {edition_name(h["optuna_tag"])}')
    st.caption(f"variant {h['variant']} | domain {h['domain']} | objective {h.get('objective')} | split by {h.get('split_by','?')} | "
               f"train seasons {h.get('train_seasons','?')} | valid seasons {h.get('valid_seasons','?')} | "
               f"{h.get('n_features','?')} features | written {h['modified']}"
               +('' if explorer else ' | other model versions are on the Training explorer page'))
    cards=[('valid RMSE (mm)','valid_rmse_all','{:.2f}'),('valid KGE','valid_kge_all','{:.3f}'),('valid R²','valid_r2_all','{:.3f}'),
           ('bias (mm)','valid_bias','{:+.2f}'),('RMSE snow-on (mm)','valid_rmse_snowon','{:.2f}'),('trees','n_estimators_earlystop','{}')]
    for col,(lab,k,f) in zip(st.columns(len(cards)),cards): col.metric(lab,fmt(h.get(k),f))
    st.caption('These scores use the validation seasons that also drove tuning and early stopping, so they are optimistic. '
               'See the Independent validation page for the independent locations.')
    vd=val_dir_for(stem)
    vmr=read_csv(os.path.join(base,vd,'validation_metrics.csv')) if vd else None
    if vmr is not None:
        st.markdown(f'**Independent validation of this run** (`{vd}`, validation locations not used in training, all rows)')
        table(train_vs_indep(h,vmr).round(3),hide_index=True)
    else:
        st.info(f"no independent validation for this run yet: python xgb-validate-cryo-swe.py {h['optuna_tag']}")

    names=['Features','Tuning','Fit','Feature importance','SHAP']+(['EDA'] if SHOW_EDA else [])+['Config']+(['All figures'] if SHOW_ALL_FIGURES else [])
    names+=(['Compare editions'] if explorer else [])+['Training locations']
    T=dict(zip(names,st.tabs(names)))

    with T['Features']:
        gi=read_csv(P('gain_importance','csv')); si=read_csv(P('shap_importance','csv'))
        feats=run_features(h,gi,si)
        if not feats:
            st.info('no feature list in the hyperparams json and no importance csv for this run')
        else:
            ft=feature_table(feats,gi,si)
            c=st.columns(4)
            c[0].metric('features',len(ft))
            c[1].metric('feature groups',ft['group'].nunique())
            c[2].metric('target','SWE (mm)')
            c[3].metric('input file',h.get('input_file','n/a'))
            c1,c2=st.columns([3,1])
            q=c1.text_input('filter by name, group or description','')
            sopts=[x for x in ['shap_rank','gain_rank','group','feature'] if x in ft]   # SHAP rank first, so it is the default
            sortby=c2.selectbox('sort by',sopts,format_func=lambda x:{'shap_rank':'SHAP rank','gain_rank':'gain rank'}.get(x,x))
            v=ft
            if q:
                ql=q.lower()
                v=ft[ft[['feature','group','description']].apply(lambda r:' '.join(r).lower(),axis=1).str.contains(ql,regex=False)]
            v=v.sort_values(sortby,kind='stable')
            cfg={'gain':None,'mean_abs_shap':st.column_config.NumberColumn('mean |SHAP|',format='%.4f')}
            if 'gain_%' in v: cfg['gain_%']=st.column_config.ProgressColumn('gain %',format='%.1f',min_value=0,max_value=float(ft['gain_%'].max() or 1))
            if 'shap_%' in v: cfg['shap_%']=st.column_config.ProgressColumn('|SHAP| %',format='%.1f',min_value=0,max_value=float(ft['shap_%'].max() or 1))
            table(v,hide_index=True,column_config=cfg,height=min(36*len(v)+40,900))
            st.caption('Importance is from this run (gain on all trees, mean |SHAP| on a validation sample, log-SWE margin). '
                       'Descriptions come from FEATURE_INFO at the top of the app.')
            st.download_button('download feature table (csv)',ft.to_csv(index=False),file_name=f'features_{stem}.csv',mime='text/csv')
            share=[x for x in ['shap_%','gain_%'] if x in ft]
            if share:
                grp=ft.groupby('group')[share].sum().reset_index().melt(id_vars='group',var_name='measure',value_name='share %')
                fig=px.bar(grp,x='share %',y='group',color='measure',barmode='group',orientation='h',height=60+45*ft['group'].nunique(),
                           title='importance share by feature group')
                fig.update_layout(yaxis={'categoryorder':'total ascending'},yaxis_title=None,margin=dict(l=0,r=10,t=40,b=0))
                chart(fig)

    with T['Tuning']:
        par=read_csv(P('optuna_pareto','csv'))
        allt=load_study(optuna_db,h['study_name'],mtime(optuna_db)) if h.get('study_name') and os.path.exists(optuna_db) else None
        if par is None and allt is None:
            st.info('no Pareto csv or multi-objective optuna study found for this run')
        else:
            fig=go.Figure()
            if allt is not None:
                fig.add_scatter(x=allt['values_0'],y=allt['values_1'],mode='markers',name=f'all trials ({len(allt)})',
                                marker=dict(color='lightgray',size=7),customdata=allt['number'],
                                hovertemplate='trial %{customdata}<br>RMSE %{x:.3f}<br>KGE %{y:.3f}<extra></extra>')
            if par is not None:
                par=par.sort_values('valid_rmse').round(4)
                hov=[c for c in par.columns if c not in ('valid_rmse','valid_kge')]
                fig.add_scatter(x=par['valid_rmse'],y=par['valid_kge'],mode='lines+markers',name='Pareto front',
                                line=dict(color='crimson',width=1),
                                marker=dict(size=10,color=par['std_ratio'],colorscale='Viridis',showscale=True,colorbar=dict(title='std ratio')),
                                customdata=par[hov].values,
                                hovertemplate='<br>'.join(f'{c}: %{{customdata[{i}]}}' for i,c in enumerate(hov))+'<extra></extra>')
                if h.get('rmse_tol') is not None:
                    fig.add_vline(x=par['valid_rmse'].min()*(1+h['rmse_tol']),line_dash='dot',line_color='gray',
                                  annotation_text=f"RMSE_TOL {h['rmse_tol']}")
            if h.get('chosen_trial_rmse') is not None:
                fig.add_scatter(x=[h['chosen_trial_rmse']],y=[h['chosen_trial_kge']],mode='markers',name=f"chosen trial {h.get('chosen_trial')}",
                                marker=dict(symbol='circle-open',size=20,color='black',line=dict(width=3)))
            fig.update_layout(xaxis_title='valid RMSE',yaxis_title='valid KGE',height=560,legend=dict(orientation='h',y=1.08))
            chart(fig)
            if par is not None:
                st.markdown('**Pareto front trials** (std ratio below 1 means too flat, high SWE underestimated)')
                table(par,hide_index=True)
        img(P('optuna_history'),'Optuna Pareto front and optimization history')

    with T['Fit']:
        c1,c2=st.columns([1,1])
        with c1: img(P('valid_scatter'),'observed vs predicted SWE, validation seasons')
        with c2:
            st.markdown('**Scores on validation seasons**')
            keys=[k for k in h if k.startswith('valid_') and not isinstance(h[k],dict)]
            table(pd.DataFrame({'metric':keys,'value':[h[k] for k in keys]}),hide_index=True)
            if isinstance(h.get('valid_metrics_by_domain'),dict):
                st.markdown('**By domain**')
                table(pd.DataFrame(h['valid_metrics_by_domain']).T.round(3))

    with T['Feature importance']:
        g=read_csv(P('gain_importance','csv')); s=read_csv(P('shap_importance','csv'))
        nmax=max(len(g) if g is not None else 0,len(s) if s is not None else 0,1)
        n=st.slider('features shown',5,max(5,nmax),min(25,max(5,nmax)))
        c1,c2=st.columns(2)
        for col,d,x,title in [(c1,g,'gain','XGBoost gain'),(c2,s,'mean_abs_shap','mean |SHAP| (log-SWE margin)')]:
            with col:
                if d is None: st.info(f'no {title} file')
                else: chart(hbar(d,x,title,n))

    with T['SHAP']:
        st.caption('With reg:tweedie the SHAP values are in log(SWE) margin units: contributions multiply on the SWE scale.')
        c1,c2=st.columns(2)
        with c1: img(P('shap_beeswarm'),'SHAP beeswarm')
        with c2: img(P('shap_bar'),'mean |SHAP|')

    if 'EDA' in T:
        with T['EDA']:
            c1,c2=st.columns(2)
            with c1: img(P('eda_swe_hist'),'snow-on SWE distribution, train vs valid')
            with c2: img(P('eda_swe_seasonal'),'mean SWE by month, train vs valid')

    with T['Config']:
        meta={'objective','eval_metric','tree_method','num_parallel_tree','variant','domain','features','n_features',
              'split_by','snow_start_month','train_seasons','valid_seasons','tuning','rmse_tol','input_file','study_name'}
        hp={k:v for k,v in h.items() if k not in meta and not k.startswith(('valid_','chosen_','_')) and k not in ('stem','modified','optuna_tag','pareto_size','n_estimators_earlystop')}
        c1,c2=st.columns(2)
        with c1:
            st.markdown('**Chosen hyperparameters**')
            table(pd.DataFrame({'parameter':list(hp),'value':[str(v) for v in hp.values()]}),hide_index=True)
        with c2:
            st.markdown(f"**Features ({len(h.get('features',[]))})**")
            st.code(', '.join(h.get('features',[])),language=None)
            st.markdown('**Setup**')
            st.json({k:h[k] for k in meta if k in h and k!='features'},expanded=False)
        mp=os.path.join(MOD_DIR,f'mdl_swe_{stem}.json')
        st.caption(f'model file: {mp} ({mtime(mp) and os.path.getsize(mp)/1e6:.1f} MB)' if os.path.exists(mp) else f'model file missing: {mp}')
        st.download_button('download hyperparams json',json.dumps({k:v for k,v in h.items() if not k.startswith('_')},indent=2),
                           file_name=os.path.basename(h['_path']),mime='application/json')

    if 'All figures' in T:
        with T['All figures']:
            # every figure the training script wrote for this run: results/*_<variant>_<domain>_<optuna_tag>.png
            figs=sorted(glob.glob(os.path.join(RES_DIR,f'*_{stem}.png')))
            if not figs:
                st.info(f'no figures for {stem} in {RES_DIR}')
            else:
                fnames=[os.path.basename(p) for p in figs]
                pick=st.multiselect(f'{len(figs)} figures in {RES_DIR}',fnames,default=fnames,key='train_figs')
                ncol=st.radio('columns',[1,2,3],index=1,horizontal=True,key='train_figs_cols')
                cols=st.columns(ncol)
                for i,p in enumerate([p for p in figs if os.path.basename(p) in pick]):
                    with cols[i%ncol]: img(p,os.path.basename(p)[:-len(f'_{stem}.png')])

    if explorer:
        with T['Compare editions']:
            compare_editions(runs,h)

    with T['Training locations']:
        # terrain and forest of the training locations (same for every run): maps of height, slope, aspect, canopy height and histograms
        areas=[a for a in TRAIN_LOC_FIGS if glob.glob(os.path.join(base,TRAIN_LOC_FIGS[a]+'_*.png'))]
        if not areas:
            st.info(f"no training location figures ({', '.join(v+'_*.png' for v in TRAIN_LOC_FIGS.values())}) in {base}")
        else:
            opts=areas+(['both side by side'] if len(areas)>1 else [])
            area=st.radio('area',opts,horizontal=True,key='trainloc_area')
            st.caption('Terrain and forest canopy height at the training locations. These describe the training data set, '
                       'so they are the same for every training run.')
            if area=='both side by side':
                for kind,cap in TRAIN_LOC_KINDS:
                    figs=[os.path.join(base,f'{TRAIN_LOC_FIGS[a]}_{kind}.png') for a in areas]
                    if not any(os.path.exists(f) for f in figs): continue
                    st.markdown(f'**{cap}**')
                    for col,a,f in zip(st.columns(len(areas)),areas,figs):
                        with col: img(f,a)
            else:
                pre=os.path.join(base,TRAIN_LOC_FIGS[area])
                maps=[(f'{pre}_{k}.png',c) for k,c in TRAIN_LOC_KINDS if k.startswith('map_') and os.path.exists(f'{pre}_{k}.png')]
                cols=st.columns(2)
                for i,(f,c) in enumerate(maps):
                    with cols[i%2]: img(f,f'{area}: {c}')
                for k,c in TRAIN_LOC_KINDS:
                    if not k.startswith('map_') and os.path.exists(f'{pre}_{k}.png'): img(f'{pre}_{k}.png',f'{area}: {c}')

def compare_editions(runs,h):
    # newest run of each edition for the same domain as the selected run
    seen={}
    for r in runs:
        if r['domain']==h['domain']: seen.setdefault(r['optuna_tag'],r)
    eds=prod_first(seen)
    pick=st.multiselect('editions to compare',eds,default=eds,format_func=lambda t:edition_label(t,runs),key='cmp_eds')
    cmp=[seen[t] for t in pick]
    if len(cmp)<2:
        st.info('pick at least two editions'); return
    # scores: tuning validation seasons (same seasons for every edition) and independent locations if validated
    rows={}
    for r in cmp:
        d={'features':r.get('n_features'),'trees':r.get('n_estimators_earlystop'),
           'tuning valid RMSE (mm)':r.get('valid_rmse_all'),'tuning valid KGE':r.get('valid_kge_all'),
           'tuning valid R²':r.get('valid_r2_all'),'tuning valid bias (mm)':r.get('valid_bias'),
           'tuning valid RMSE snow-on (mm)':r.get('valid_rmse_snowon')}
        for dom,m in (r.get('valid_metrics_by_domain') or {}).items():
            d[f'tuning valid RMSE {dom}']=m.get('rmse'); d[f'tuning valid KGE {dom}']=m.get('kge')
        rvd=val_dir_for(r['stem'])
        rvm=read_csv(os.path.join(base,rvd,'validation_metrics.csv')) if rvd else None
        if rvm is not None:
            for _,x in train_vs_indep(r,rvm).iterrows():
                d[f"independent RMSE {x['domain']}"]=x['RMSE independent']
                d[f"independent KGE {x['domain']}"]=x['KGE independent']
        rows[edition_name(r['optuna_tag'])]=d
    st.markdown('**Scores**')
    table(pd.DataFrame(rows).apply(pd.to_numeric,errors='coerce').round(3))
    missing=[edition_name(r['optuna_tag']) for r in cmp if val_dir_for(r['stem']) is None]
    if missing:
        st.caption('no independent validation yet for: '+', '.join(missing))

    # Pareto fronts on top of each other, chosen trial circled
    fig=go.Figure()
    for r in cmp:
        par=read_csv(os.path.join(RES_DIR,f"optuna_pareto_{r['stem']}.csv"))
        if par is None: continue
        par=par.sort_values('valid_rmse')
        fig.add_scatter(x=par['valid_rmse'],y=par['valid_kge'],mode='lines+markers',name=edition_name(r['optuna_tag']))
        if r.get('chosen_trial_rmse') is not None:
            fig.add_scatter(x=[r['chosen_trial_rmse']],y=[r['chosen_trial_kge']],mode='markers',showlegend=False,
                            marker=dict(symbol='circle-open',size=18,color='black',line=dict(width=2)))
    if fig.data:
        fig.update_layout(xaxis_title='valid RMSE',yaxis_title='valid KGE',height=480,
                          title='Pareto fronts (circle = chosen trial)',legend=dict(orientation='h',y=1.08))
        chart(fig,key='cmp_pareto')

    # SHAP share per feature and per feature group
    parts=[]
    for r in cmp:
        s=read_csv(os.path.join(RES_DIR,f"shap_importance_{r['stem']}.csv"))
        if s is not None and s['mean_abs_shap'].sum()>0:
            parts.append(s.assign(**{'share %':100*s['mean_abs_shap']/s['mean_abs_shap'].sum(),
                                     'edition':edition_name(r['optuna_tag'])}))
    if parts:
        imp=pd.concat(parts)
        nf=max(6,int(imp['feature'].nunique()))
        n=st.slider('features shown',5,nf,min(25,nf),key='cmp_n')
        top=imp.groupby('feature')['share %'].max().nlargest(n).index
        d=imp[imp['feature'].isin(top)]
        order=d.groupby('feature')['share %'].max().sort_values().index.tolist()
        f=px.bar(d,x='share %',y='feature',color='edition',barmode='group',orientation='h',height=28*len(order)+140,
                 category_orders={'feature':order[::-1]},title='mean |SHAP| share per feature (log-SWE margin)')
        f.update_layout(yaxis_title=None,margin=dict(l=0,r=10,t=40,b=0),legend=dict(orientation='h',y=1.04))
        chart(f,key='cmp_shap')
        imp['group']=imp['feature'].map(lambda x:feature_info(x)[0])
        grp=imp.groupby(['edition','group'])['share %'].sum().reset_index()
        f=px.bar(grp,x='share %',y='group',color='edition',barmode='group',orientation='h',
                 height=60+45*grp['group'].nunique(),title='mean |SHAP| share per feature group')
        f.update_layout(yaxis={'categoryorder':'total ascending'},yaxis_title=None,margin=dict(l=0,r=10,t=40,b=0))
        chart(f,key='cmp_grp')
        st.caption('Shares are within each edition, so MANYfeatures spreads the same total over more features.')

# ---------------------------------------------------------------
# Pages: independent validation (final model only) and validation explorer (every model version)
# ---------------------------------------------------------------
VAL_VIEWS=['Scores and figures','Location time series and maps']
VAL_ALL='All iterations compared'

def page_validation():
    runs,tags,todo=val_models()
    tag=used_model(tags)
    if tag is None:
        st.title('Independent validation')
        st.warning(f'no independent validation of the final model {USED_MODEL} yet (validationresults_<tag>/ with validation_metrics.csv in {base})')
        st.code(f'python xgb-validate-cryo-swe.py {USED_MODEL}',language='bash')
        return
    use_val(VAL_BY_TAG[tag])
    st.title(f'Independent validation: {edition_name(tag)}')
    view=st.radio('view',VAL_VIEWS,horizontal=True,key='val_view')
    st.caption('Other model versions are on the Validation explorer page.')
    if view==VAL_VIEWS[0]: val_results(tag)
    else: val_locations(tag)

def page_validation_explorer():
    st.title('Validation explorer')
    view=st.radio('view',VAL_VIEWS+[VAL_ALL],horizontal=True,key='valx_view')
    if view==VAL_ALL:
        val_iterations(); return
    tag=pick_val_model()
    if tag is None: return
    st.header(edition_name(tag))
    if view==VAL_VIEWS[0]: val_results(tag)
    else: val_locations(tag)

def val_results(tag):
    # scores, maps, monthly, snow season and figures of one validated model edition (VAL_DIR set by use_val)
    vpath=os.path.join(VAL_DIR,'validation_metrics.csv')
    vm=read_csv(vpath)
    if vm is None:
        st.warning(f'no validation_metrics.csv in {VAL_DIR}'); return
    info=read_json(os.path.join(VAL_DIR,'run_info.json'))
    st.caption(f'independent validation, validation locations not used in training | results in {VAL_DIR} | written {stamp(vpath)}'
               + (f" | model from training run {info['training_stem']}" if info.get('training_stem') else '')
               + (' | validation still running or interrupted, results incomplete' if info.get('complete') is False else ''))

    models=[m for m in vm['model'].unique() if not str(m).startswith('baseline_')]
    c1,c2,c3=st.columns([1,1,1])
    mname=models[0] if len(models)==1 else c1.selectbox('model',models)
    dom=c2.selectbox('evaluated on',vm.loc[vm['model']==mname,'domain'].unique().tolist())
    subset=c3.radio('rows',['all_rows','snow_rows'],horizontal=True,
                    format_func=lambda s:'all rows' if s=='all_rows' else 'snow rows (obs or pred > 0)')
    label=f'{mname}model_{dom}'
    bases=[c[len('skill_MSE_vs_'):] for c in vm.columns if c.startswith('skill_MSE_vs_')]
    row=vm[(vm['model']==mname)&(vm['domain']==dom)&(vm['subset']==subset)]
    if row.empty:
        st.warning('no scores for this selection'); return
    r=row.iloc[0]
    cards=[('RMSE (mm)',r.get('RMSE'),'{:.1f}'),('ubRMSE (mm)',r.get('ubRMSE'),'{:.1f}'),('ME (mm)',r.get('ME'),'{:+.1f}'),
           ('R²',r.get('R2'),'{:.3f}'),('KGE',r.get('KGE'),'{:.3f}')]+[(f'MSE skill vs {b}',r.get(f'skill_MSE_vs_{b}'),'{:.2f}') for b in bases]
    for col,(lab,v,f) in zip(st.columns(len(cards)),cards): col.metric(lab,fmt(v,f))

    tabs=st.tabs(['Scores','Map','Monthly & snow year','Snow season','Figures','Training vs validation','Feature importance','Validation locations'])

    with tabs[0]:
        comp=vm[(vm['domain']==dom)&(vm['subset']==subset)&vm['model'].isin([mname]+[f'baseline_{b}' for b in bases])].copy()
        comp['predictor']=comp['model'].str.replace('baseline_','',regex=False).replace({mname:f'XGBoost ({edition_name(tag)})'})
        show=[c for c in ['n','obs_mean','pred_mean','ME','MAE','MedAE','P95_AE','RMSE','ubRMSE','nRMSE','PBIAS_%','R2','pearson_r',
                          'spearman_r','explained_var','std_ratio','KGE','POD','FAR','CSI','accuracy','HSS'] if c in comp]
        show+=[c for c in comp.columns if c.startswith(('skill_','RMSE_','n_')) and c not in show]   # skill against each baseline, on the same rows
        table(comp.set_index('predictor')[show].T.round(3))
        st.caption('Detection scores (POD, FAR, CSI, HSS) use a 1 mm snow threshold. For baselines, snow rows are rows where obs or that baseline is above 0.')
        long=comp.melt(id_vars='predictor',value_vars=[c for c in ['RMSE','ubRMSE','MAE'] if c in comp],var_name='metric')
        c1,c2=st.columns(2)
        with c1: chart(px.bar(long,x='metric',y='value',color='predictor',barmode='group',height=380,title='errors (mm)'))
        with c2: chart(px.bar(comp,x='predictor',y=[c for c in ['R2','KGE','pearson_r'] if c in comp],barmode='group',height=380,title='skill scores'))

    with tabs[1]:
        if dom=='all':
            st.info('per location maps exist for single domains only, pick Scandinavia or Alps')
        else:
            pl=read_csv(os.path.join(VAL_DIR,f'per_location_{label}.csv'))
            if pl is None: st.info(f'no per_location_{label}.csv')
            else:
                opts=[c for c in pl.columns if c not in ('latitude','longitude') and pd.api.types.is_numeric_dtype(pl[c])]
                rmse_base=[c for c in opts if c.startswith('RMSE_')]
                # defaults: XGBoost RMSE on the left, ERA5-Land RMSE (or the first baseline RMSE) on the right
                d1='RMSE' if 'RMSE' in opts else opts[0]
                d2=next((c for c in rmse_base if 'era5' in c.lower()),rmse_base[0] if rmse_base else opts[min(1,len(opts)-1)])
                emax=st.number_input('red above (mm), RMSE/MAE maps',value=ERR_MAX,min_value=10.0,step=25.0)
                hov={c:':.2f' for c in ['RMSE','ME','r','obs_mean','pred_mean','n','dist_to_train_m']+rmse_base if c in pl}
                for i,(pane,default) in enumerate(zip(st.columns(2),[d1,d2])):
                    with pane:
                        col=st.selectbox(f'map {i+1}: color by',opts,index=opts.index(default),key=f'valmap_col{i}_{label}')
                        chart(map_fig(pl,col,hover=hov,height=560,err_max=emax),key=f'valmap{i}')
                        if is_err(col):
                            st.caption(f'{int((pl[col]>emax).sum())} of {len(pl)} locations above {emax:g} mm (red)')
                        chart(px.histogram(pl,x=col,nbins=80,height=260,title=f'{col} across {len(pl)} locations'),key=f'valhist{i}')

    with tabs[2]:
        mt=read_csv(os.path.join(VAL_DIR,'validation_metrics_monthly.csv'))
        if mt is not None:
            mt=mt[(mt['model']==mname)&(mt['domain']==dom)].copy()
            mt['mon']=mt['month'].astype(int).map(MONTH_NAMES)
            order={'mon':[MONTH_NAMES[m] for m in SEASON_MONTHS]}
            c1,c2=st.columns(2)
            with c1:
                chart(px.bar(mt.melt(id_vars='mon',value_vars=['RMSE','ubRMSE','ME']),x='mon',y='value',color='variable',
                             barmode='group',category_orders=order,height=380,title='monthly errors (mm)',labels={'mon':'month','value':'mm'}))
            with c2:
                chart(px.line(mt.melt(id_vars='mon',value_vars=['obs_mean','pred_mean']),x='mon',y='value',color='variable',markers=True,
                              category_orders=order,height=380,title='monthly mean SWE (mm)',labels={'mon':'month','value':'mm'}))
            sk=[c for c in ['KGE','R2','explained_var'] if c in mt]
            if sk:
                chart(px.line(mt.melt(id_vars='mon',value_vars=sk),x='mon',y='value',color='variable',markers=True,
                              category_orders=order,height=340,title='monthly skill',labels={'mon':'month','value':''}))
            with st.expander('monthly scores table'):
                table(mt.drop(columns=['model','domain','mon']).round(3),hide_index=True)
        sy=read_csv(os.path.join(VAL_DIR,'validation_metrics_snowyear.csv'))
        if sy is not None:
            sy=sy[(sy['model']==mname)&(sy['domain']==dom)]
            obs=sy.drop_duplicates('season')[['season','obs_mean']].rename(columns={'obs_mean':'value'}).assign(predictor='observed')
            mean=pd.concat([obs,sy[['season','predictor','pred_mean']].rename(columns={'pred_mean':'value'})])
            c=st.columns(3)
            with c[0]: chart(px.line(sy,x='season',y='RMSE',color='predictor',markers=True,height=360,title='RMSE per snow year (mm)'))
            with c[1]:
                f=px.line(sy,x='season',y='ME',color='predictor',markers=True,height=360,title='ME per snow year (mm)'); f.add_hline(y=0,line_color='black',line_width=0.5); chart(f)
            with c[2]: chart(px.line(mean,x='season',y='value',color='predictor',markers=True,height=360,title='mean SWE per snow year (mm)'))
            with st.expander('snow year scores table'):
                table(sy.drop(columns=[c for c in ['model','domain','period'] if c in sy]).round(3),hide_index=True)

    with tabs[3]:
        ss=read_csv(os.path.join(VAL_DIR,'validation_metrics_snowseason.csv'))
        if ss is None or dom=='all':
            st.info('snow season metrics exist for single domains only' if dom=='all' else 'no validation_metrics_snowseason.csv')
        else:
            s=ss[(ss['model']==mname)&(ss['domain']==dom)].set_index('predictor').drop(columns=['model','domain'])
            c1,c2=st.columns([1,2])
            with c1:
                st.markdown('**Peak SWE, timing and duration**')
                table(s.T.round(2))
                st.caption(f'timing scores (peak, onset, melt-out) only for location seasons with observed peak ≥ {PEAK_MIN:g} mm; onset and melt-out use 1 mm')
            with c2:
                stab=read_csv(os.path.join(VAL_DIR,f'snowseason_{label}.csv'))
                if stab is not None:
                    preds=[c[len('peak_'):] for c in stab.columns if c.startswith('peak_SWE_') and c!='peak_SWE_observed' and not c.startswith('peak_dos')]
                    pc=st.radio('predictor',preds,horizontal=True,format_func=lambda c:PRED_STYLE.get(c,(c,))[0])
                    smp=stab.dropna(subset=[f'peak_{pc}'])
                    smp=smp.sample(min(20000,len(smp)),random_state=1)   # baselines can be NaN at some locations, so sample from the rows left
                    lim=float(np.nanpercentile(np.r_[smp['peak_SWE_observed'],smp[f'peak_{pc}']],99.5))
                    f=px.scatter(smp,x='peak_SWE_observed',y=f'peak_{pc}',opacity=0.3,render_mode='webgl',height=420,
                                 hover_data=['latitude','longitude','season'],title='peak SWE per location season (mm)',
                                 labels={'peak_SWE_observed':'observed peak',f'peak_{pc}':'predicted peak'})
                    f.add_scatter(x=[0,lim],y=[0,lim],mode='lines',line=dict(color='red',dash='dash'),showlegend=False)
                    f.update_layout(xaxis_range=[0,lim],yaxis_range=[0,lim]); chart(f)
                    snowy=stab[stab['peak_SWE_observed']>=PEAK_MIN]
                    errs=pd.concat([pd.DataFrame({'error (days)':(snowy[f'{m}_{c}']-snowy[f'{m}_SWE_observed']).clip(-90,90),
                                                  'metric':m,'predictor':PRED_STYLE.get(c,(c,))[0]}).dropna()
                                    for m in ['peak_dos','onset','meltout'] for c in ['SWE_xgboost']+[p for p in preds if p!='SWE_xgboost'] if f'{m}_{c}' in snowy])
                    chart(px.histogram(errs,x='error (days)',color='predictor',facet_col='metric',barmode='overlay',nbins=60,opacity=0.55,height=330,
                                       title='timing errors, predicted minus observed (clipped at ±90 d)'))

    with tabs[4]:
        figs=[p for p in sorted(glob.glob(os.path.join(FIG_DIR,f'*_{label}.png'))) if not os.path.basename(p).startswith(('distance_','shap_dep'))]
        figs+=[p for p in [os.path.join(FIG_DIR,f'shap_{mname}model.png')] if os.path.exists(p)]
        if not figs: st.info(f'no figures for {label} in {FIG_DIR}')
        pick=st.multiselect('figures',[os.path.basename(p) for p in figs],default=[os.path.basename(p) for p in figs])
        for p in figs:
            if os.path.basename(p) in pick: img(p,os.path.basename(p))

    with tabs[5]:
        h=next((x for x in training_runs(MOD_DIR) if x['stem']==info.get('training_stem')),None)
        if h is None:
            st.info('no training run linked to this folder (run_info.json missing or its hyperparams file not found)')
        else:
            st.markdown(f"**Training run `{h['stem']}`** | train seasons {h.get('train_seasons')} | tuning validation seasons {h.get('valid_seasons')} | "
                        f"chosen trial {h.get('chosen_trial')} of {h.get('pareto_size')} on the Pareto front")
            table(train_vs_indep(h,vm).round(3),hide_index=True)
            st.caption('tuning valid: held-out seasons at the training locations, also used for tuning and early stopping, so optimistic. '
                       'independent: validation locations never used in training.')

    with tabs[6]:
        sv=read_csv(os.path.join(VAL_DIR,f'shap_importance_{mname}model.csv'))
        stn=info.get('training_stem')
        stv=read_csv(os.path.join(RES_DIR,f'shap_importance_{stn}.csv')) if stn else None
        if sv is None:
            st.info(f'no shap_importance_{mname}model.csv in this folder')
        else:
            parts=[sv.assign(**{'share %':100*sv['mean_abs_shap']/sv['mean_abs_shap'].sum(),'data':'validation locations'})]
            if stv is not None:
                parts.append(stv.assign(**{'share %':100*stv['mean_abs_shap']/stv['mean_abs_shap'].sum(),'data':'training run (tuning validation seasons)'}))
            imp=pd.concat(parts)
            order=sv.sort_values('mean_abs_shap')['feature'].tolist()
            fig=px.bar(imp,x='share %',y='feature',color='data',barmode='group',orientation='h',height=28*len(order)+140,
                       category_orders={'feature':order[::-1]},title='mean |SHAP| share per feature (log-SWE margin)')
            fig.update_layout(yaxis_title=None,margin=dict(l=0,r=10,t=40,b=0),legend=dict(orientation='h',y=1.04))
            chart(fig)
            st.caption('A feature that matters much more on the validation locations than in training points to conditions the training data covered poorly.')
        img(os.path.join(FIG_DIR,f'shap_{mname}model.png'),'SHAP beeswarm on validation locations')

    with tabs[7]:
        # maps of the independent validation locations (same for every validation run)
        areas=[a for a in VAL_LOC_FIGS if os.path.exists(os.path.join(base,VAL_LOC_FIGS[a]))]
        if not areas:
            st.info(f"no validation location maps ({', '.join(VAL_LOC_FIGS.values())}) in {base}")
        else:
            opts=areas+(['both side by side'] if len(areas)>1 else [])
            # start with the domain picked in 'evaluated on' above, both for 'all'
            a=st.radio('area',opts,index=opts.index(dom) if dom in opts else len(opts)-1,horizontal=True,key=f'valloc_area_{dom}')
            st.caption('Locations used for the independent validation, never used in training. '
                       'The terrain of the training locations is on the Training and model performance page, tab Training locations.')
            show=areas if a=='both side by side' else [a]
            for col,x in zip(st.columns(len(show)),show):
                with col: img(os.path.join(base,VAL_LOC_FIGS[x]),f'{x}: validation locations')

def val_iterations():
    # every validated model edition side by side: independent scores per domain and monthly scores
    runs=training_runs(MOD_DIR)
    if not VAL_BY_TAG:
        st.info(f'no validationresults_<tag>/ folders with validation_metrics.csv in {base}'); return
    c1,c2=st.columns(2)
    subset=c1.radio('rows',['all_rows','snow_rows'],horizontal=True,key='valx_cmp_rows',
                    format_func=lambda s:'all rows' if s=='all_rows' else 'snow rows (obs or pred > 0)')
    vm=indep_val_all(runs,subset)
    if vm is None:
        st.info('no readable validation_metrics.csv'); return
    mets=[m for m in ['RMSE','ubRMSE','MAE','ME','R2','KGE','pearson_r','std_ratio','POD','FAR','CSI','HSS'] if m in vm]
    met=c2.selectbox('score',mets,index=mets.index('RMSE') if 'RMSE' in mets else 0,key='valx_cmp_met')
    st.markdown(f'**{met}, independent validation locations** (baselines shown once)')
    table(vm.pivot_table(index='model',columns='domain',values=met).round(3))
    f=px.bar(vm,x='domain',y=met,color='model',barmode='group',height=420,labels={'model':''})
    if met=='ME': f.add_hline(y=0,line_color='black',line_width=0.6)
    chart(f.update_layout(legend=dict(orientation='h',y=-0.25)),key='valx_cmp_bar')

    # monthly scores of every edition (XGBoost rows only)
    parts=[]
    for tag,d in VAL_BY_TAG.items():
        mt=read_csv(os.path.join(base,d,'validation_metrics_monthly.csv'))
        if mt is None or met not in mt: continue
        mt=mt[~mt['model'].astype(str).str.startswith('baseline_')]
        parts.append(mt.assign(edition=edition_label(tag,runs)))
    if parts:
        mm=pd.concat(parts,ignore_index=True)
        mm['mon']=mm['month'].astype(int).map(MONTH_NAMES)
        f=px.line(mm,x='mon',y=met,color='edition',facet_col='domain',markers=True,height=400,
                  category_orders={'mon':[MONTH_NAMES[m] for m in SEASON_MONTHS]},title=f'monthly {met}, all rows',
                  labels={'mon':'month','edition':''})
        f.for_each_annotation(lambda a:a.update(text=a.text.split('=')[-1]))
        f.update_layout(legend=dict(orientation='h',y=-0.2))
        chart(f,key='valx_cmp_monthly')
    runs_todo=[t for t in dict.fromkeys(r['optuna_tag'] for r in runs) if t not in VAL_BY_TAG]
    if runs_todo: st.caption('not validated yet: '+', '.join(edition_label(t,runs) for t in runs_todo))

# ---------------------------------------------------------------
# Location time series and maps (every validation row: observed, XGBoost and baselines)
# ---------------------------------------------------------------
def set_loc(lt,lat,lon): st.session_state['loc']=nearest(lt,lat,lon)

def val_locations(tag):
    files=sorted(glob.glob(os.path.join(VAL_DIR,'SWE_xgboost_validation_*model.csv.gz')))
    if not files:
        st.warning(f'no SWE_xgboost_validation_*model.csv.gz in {VAL_DIR}. Set save_predictions=True in xgb-validate-cryo-swe.py to write it.'); return
    f=files[0] if len(files)==1 else st.selectbox('prediction file',files,format_func=os.path.basename)
    mt=mtime(f)
    df=load_pred(f,mt); lt=loc_table(f,mt)
    preds=[c for c in PRED_STYLE if c in df]
    st.caption(f"{os.path.basename(f)} | {len(df):,} rows | {len(lt):,} locations | {df['utctime'].min():%Y-%m-%d} to {df['utctime'].max():%Y-%m-%d} | written {stamp(f)}")

    tabs=st.tabs(['Location time series','Daily map','Domain mean'])

    with tabs[0]:
        c1,c2=st.columns([1,3])
        tags=c1.multiselect('domains',sorted(lt['tag'].unique()),default=sorted(lt['tag'].unique()))
        colby=c1.selectbox('color map by',[c for c in ['RMSE','ME','obs_mean','obs_max','pred_mean','height','dist_to_train_m'] if c in lt])
        worst=lt.nlargest(50,'RMSE')
        c1.selectbox('jump to a worst RMSE location',[None]+list(zip(worst['latitude'],worst['longitude'])),key='worst_pick',
                     format_func=lambda x:'' if x is None else f'{x[0]:.3f}, {x[1]:.3f}',
                     on_change=lambda: st.session_state['worst_pick'] and set_loc(lt,*st.session_state['worst_pick']))
        la=c1.number_input('latitude',value=float(lt['latitude'].median()),format='%.4f')
        lo=c1.number_input('longitude',value=float(lt['longitude'].median()),format='%.4f')
        c1.button('go to nearest location',on_click=set_loc,args=(lt,la,lo))
        sub=lt[lt['tag'].isin(tags)]
        if 'loc' not in st.session_state:
            r=lt.loc[lt['obs_mean'].idxmax()]; st.session_state['loc']=(float(r['latitude']),float(r['longitude']))
        with c2:
            if sub.empty: st.info('no locations for the chosen domains')
            else:
                fig=map_fig(sub,colby,hover={c:':.2f' for c in ['RMSE','ME','obs_mean','height'] if c in sub},height=520)
                lat0,lon0=st.session_state['loc']
                fig.add_trace(go.Scattermap(lat=[lat0],lon=[lon0],mode='markers',marker=dict(size=16,color='red'),name='selected',
                                            customdata=[[lat0,lon0]],showlegend=False)
                              if hasattr(go,'Scattermap') else go.Scattermapbox(lat=[lat0],lon=[lon0],mode='markers',marker=dict(size=16,color='red'),
                                                                                 customdata=[[lat0,lon0]],showlegend=False))
                try:
                    ev=chart(fig,on_select='rerun',selection_mode='points',key='locmap')
                    pts=ev.selection.points if ev is not None else []
                except Exception:
                    st.caption('click selection needs streamlit ≥ 1.35, use the inputs on the left'); pts=[]
                ll=next((x for x in (click_latlon(p) for p in pts) if x is not None),None) if pts else None
                if ll is not None and ll!=st.session_state.get('last_click'):
                    st.session_state['last_click']=ll; set_loc(lt,*ll); st.rerun()
                st.caption('click a point on the map to show its time series')

        lat,lon=st.session_state['loc']
        idx=loc_index(f,mt).get((lat,lon))
        if idx is None: st.info('selected location not in this file, pick another'); return
        d=df.iloc[idx].sort_values('utctime')
        info=lt[(lt['latitude']==lat)&(lt['longitude']==lon)].iloc[0]
        seasons=sorted(d['season'].unique())
        sel=st.select_slider('snow year',options=['all']+[f'{s}/{(s+1)%100:02d}' for s in seasons],value='all')
        if sel!='all': d=d[d['season']==int(sel[:4])]
        st.subheader(f"{info['tag']} {lat:.4f}N {lon:.4f}E, {info['height']:.0f} m"
                     + (f", {info['dist_to_train_m']/1000:.1f} km from nearest training point" if 'dist_to_train_m' in info and pd.notna(info['dist_to_train_m']) else ''))
        fig=go.Figure()
        for c in preds:
            name,color,dash=PRED_STYLE[c]
            fig.add_scatter(x=d['utctime'],y=d[c],mode='lines',name=name,line=dict(color=color,dash=dash,width=2.2 if c in ('SWE_observed','SWE_xgboost') else 1.4))
        fig.update_layout(height=420,yaxis_title='SWE (mm)',hovermode='x unified',legend=dict(orientation='h',y=1.1),margin=dict(t=30))
        chart(fig)
        sc=pd.DataFrame({PRED_STYLE[c][0]:quick_scores(d['SWE_observed'],d[c]) for c in preds if c!='SWE_observed'}).T
        table(sc.round(2))

    with tabs[1]:
        dmin,dmax=df['utctime'].min().date(),df['utctime'].max().date()
        last=pd.Timestamp(f"{int(df['season'].max())+1}-03-01").date()
        c1,c2=st.columns([1,3])
        day=c1.date_input('day',value=min(max(last,dmin),dmax),min_value=dmin,max_value=dmax)
        comp=c1.selectbox('compare with',[c for c in preds if c!='SWE_observed'],format_func=lambda c:PRED_STYLE[c][0])
        t0=pd.Timestamp(day); m=(df['utctime']>=t0)&(df['utctime']<t0+pd.Timedelta(days=1))
        dd=df.loc[m,['latitude','longitude','tag','height']+preds].copy()
        if dd.empty: st.info('no data for this day'); return
        dd[f'{comp} minus observed']=dd[comp]-dd['SWE_observed']
        s=quick_scores(dd['SWE_observed'],dd[comp])
        for k,v in [('locations',f"{len(dd):,}"),('RMSE (mm)',fmt(s.get('RMSE'),'{:.1f}')),('ME (mm)',fmt(s.get('ME'),'{:+.1f}')),('KGE',fmt(s.get('KGE'),'{:.3f}'))]:
            c1.metric(k,v)
        hi=float(np.nanpercentile(np.r_[dd['SWE_observed'],dd[comp]],99)) or 1.0
        with c2:
            a,b=st.columns(2)
            with a: chart(map_fig(dd,'SWE_observed',hover={'height':':.0f'},height=430,title='observed SWE (mm)',rng=(0,hi),cscale='Blues'))
            with b: chart(map_fig(dd,comp,hover={'height':':.0f'},height=430,title=f'{PRED_STYLE[comp][0]} SWE (mm)',rng=(0,hi),cscale='Blues'))
            chart(map_fig(dd,f'{comp} minus observed',hover={'SWE_observed':':.1f',comp:':.1f','height':':.0f'},height=480,
                          title=f'{PRED_STYLE[comp][0]} minus observed (mm)'))

    with tabs[2]:
        dm=domain_mean(f,mt)
        long=dm.melt(id_vars=['date','tag'],var_name='predictor',value_name='SWE')
        long['predictor']=long['predictor'].map(lambda c:PRED_STYLE[c][0])
        fig=px.line(long,x='date',y='SWE',color='predictor',facet_row='tag',height=330*dm['tag'].nunique(),
                    color_discrete_map={v[0]:v[1] for v in PRED_STYLE.values()},labels={'SWE':'domain mean SWE (mm)'})
        fig.for_each_annotation(lambda a:a.update(text=a.text.split('=')[-1]))
        fig.update_yaxes(matches=None); fig.update_layout(hovermode='x unified')
        chart(fig)

# ---------------------------------------------------------------
# Pages: SWE predictions (past) 10 and 15 days (final model, divided by start date) and compare model predictions 10 and 15 days
# ECXENS forecasts validated against HKL, figures and scores from the forecast validation script
# ---------------------------------------------------------------
def fc_no_runs():
    st.markdown('Copy the output of the forecast validation script here as `<model>/<start date>/<prediction>/`, '
                'for example `FIN5/20180119/ensmean/` (Alps: `FIN5_Alps/20180119/ensmean/`) with `swe_metrics.csv` '
                'and the png figures in each prediction folder:')
    st.code(f'rsync -av smartmet@smart.nsdc.fmi.fi:/home/users/smartmet/data/ec-ens/validation/figs_swe_validation/ {PRED_DIR}/',language='bash')

def fc_case_scores(r):
    # mean scores over all lead days of the ens mean (or the first prediction) of one run, for the start date table
    p=next((p for p in r['preds'] if os.path.basename(p)=='ensmean'),r['preds'][0])
    d=read_csv(os.path.join(p,'swe_metrics.csv'))
    if d is None: return {}
    out={}
    for k,l in [('bias','bias (mm)'),('rmse','RMSE (mm)'),('kge','KGE'),('csi_1','CSI 1 mm'),('skill_mse_vs_ecens','MSE skill vs ECENS')]:
        if k in d: out[l]=d[k].mean()
    return out

def page_forecasts():
    st.title('SWE predictions (past): 10 and 15 days (500 m, ECXENS)')
    allruns=fc_runs(PRED_DIR)
    model=used_model(prod_first({r['model'] for r in allruns})) or USED_MODEL
    runs=[r for r in allruns if r['model']==model]
    region,runs=region_choice(runs,FC_CASES,'fc_region_sel')
    by=case_overview(FC_CASES.get(region,{}),runs,fc_date_key,fc_case_scores,key=f'fc_cases_{region}',chart_box=not FC_SIMPLE,
                     sources=case_sources(FC_CASE_SOURCES,region))
    if not by:
        st.info(f'no {edition_name(model)} results for {region} in {PRED_DIR} yet'); fc_no_runs(); return
    keys=sorted(by)
    skey=f'fc_case_sel_{region}'
    if st.session_state.get(skey) not in keys: st.session_state[skey]=keys[0]
    k=st.selectbox('start date',keys,key=skey,format_func=lambda x:case_label(FC_CASES,region,x))
    st.divider()
    st.header(case_label(FC_CASES,region,k))
    fc_run_view(by[k],region,model,simple=FC_SIMPLE)

def fc_detail_any(runs,region):
    # one model in detail, any model edition and start date (the compare page)
    models=prod_first({r['model'] for r in runs})
    nstart={m:sum(r['model']==m for r in runs) for m in models}
    # widgets with fixed keys: their value lives in session_state, so a pick is never reset by a changing default
    mkey=f'fc_model_sel_{region}'   # one model choice per region
    if st.session_state.get(mkey) not in models:
        st.session_state[mkey]=default_model(models)
    c1,c2=st.columns([2,3])
    with c1:
        model=st.radio('model',models,horizontal=True,key=mkey,
                       format_func=lambda m:f'{edition_name(m)} ({nstart[m]} start date{"s" if nstart[m]!=1 else ""})')
    runs=[r for r in runs if r['model']==model and not case_hidden(fc_date_key(r))]
    paths=[r['run'] for r in runs]
    if not paths:
        st.info('no start dates with results for this model'); return
    skey=f'fc_run_sel_{region}_{model}'   # one start date choice per region and model
    if st.session_state.get(skey) not in paths: st.session_state[skey]=paths[0]
    def lab(p):
        r=next(r for r in runs if r['run']==p)
        return case_label(FC_CASES,region,fc_date_key(r))+f"  ({fc_rel(r['run'])}, {len(r['preds'])} predictions)"
    with c2:
        run_path=st.selectbox('start date',paths,key=skey,format_func=lab)
    fc_run_view(next(r for r in runs if r['run']==run_path),region,model)

def fc_run_view(R,region,model,simple=False):
    # simple: fewer score tabs below the daily maps (FC_SIMPLE on the SWE predictions (past) 10 and 15 days page)
    # daily maps, scores, summaries, members and figures of one forecast run (one model, region and start date)
    run_path,preds,base_time=R['run'],R['preds'],R['base_time']
    df=fc_metrics(preds)
    if df is None:
        st.warning('no readable swe_metrics.csv in this run'); return
    names=[fc_name(p) for p in preds]
    # the folder's start date and the dates inside swe_metrics.csv must agree, otherwise the folder holds another run's results
    bt_scores=fc_base_from_scores(preds[0])
    if base_time is not None and bt_scores is not None and abs((bt_scores-base_time).days)>0:
        st.warning(f"`{fc_rel(run_path)}` says start date {base_time:%Y-%m-%d}, but its swe_metrics.csv has dates "
                   f"{df['date'].min():%Y-%m-%d} to {df['date'].max():%Y-%m-%d} (start {bt_scores:%Y-%m-%d}): "
                   "these files come from another forecast run, check what was copied into this folder")

    # ---- daily maps first: per prediction raw ECENS, ECXENS, HKL (one figure) and ECXENS minus HKL
    st.subheader('Daily maps')
    dates=sorted({os.path.basename(f)[len('swe_compare_'):-4] for p in preds for f in glob.glob(os.path.join(p,'swe_compare_*.png'))})
    if not dates:
        st.info('no swe_compare_YYYYMMDD.png figures in this run')
    else:
        lab=lambda t: f"{pd.Timestamp(t):%Y-%m-%d}"+(f" (lead {(pd.Timestamp(t)-base_time).days} d)" if base_time is not None else '')
        c1,c2=st.columns([3,2])
        if len(dates)>1:
            day=c1.select_slider('day',options=dates,value=dates[0],format_func=lab,key=f'fc_day_{run_path}')
        else:
            day=dates[0]; c1.markdown(f'**day** {lab(day)}')
        show=c2.multiselect('predictions',preds,default=preds,format_func=fc_name,key=f'fc_show_{run_path}')
        st.caption(f'Each row: raw ECENS (as in the file, 0.1° grid), ECXENS and HKL SWE with the same legend; ECXENS minus HKL (500 m); '
                   f'ECENS minus HKL (0.1°, HKL averaged into the ECENS cells); |ECXENS bias| minus |ECENS bias| (0.1°, greens and blues = ECXENS closer '
                   f'to HKL, yellows and reds = further). All bias maps: soft greens and yellows within ±{MILD_BIAS:g} mm, strong blues and reds beyond.')
        ts=pd.Timestamp(day)
        for p in show:
            # wettest and driest members collapsed: their maps open with a click
            collapsed=os.path.basename(p) in FC_COLLAPSED
            box=st.expander(fc_name(p),expanded=False) if collapsed else st.container()
            with box:
                if not collapsed: st.markdown(f'#### {fc_name(p)}')
                r=df[(df['pred']==os.path.basename(p))&(df['date']==ts)]
                if not r.empty:
                    r=r.iloc[0]
                    parts=[('bias',r.get('bias'),'{:+.1f} mm'),('RMSE',r.get('rmse'),'{:.1f} mm'),('KGE',r.get('kge'),'{:.3f}'),
                           ('CSI 1 mm',r.get('csi_1'),'{:.3f}'),('POD 1 mm',r.get('pod_1'),'{:.3f}'),('FAR 1 mm',r.get('far_1'),'{:.3f}')]
                    if pd.notna(r.get('vol_ecx_km3')) and pd.notna(r.get('vol_hkl_km3')) and r.get('vol_hkl_km3'):
                        parts.append(('SWE volume ECXENS / HKL',r['vol_ecx_km3']/r['vol_hkl_km3'],'{:.2f}'))
                    if pd.notna(r.get('skill_mse_vs_ecens')):
                        parts+=[('0.1° RMSE ECENS',r.get('ecens_rmse'),'{:.1f} mm'),('ECXENS',r.get('ecx01_rmse'),'{:.1f} mm'),
                                ('MSE skill vs ECENS',r.get('skill_mse_vs_ecens'),'{:+.2f}'),('cells improved',r.get('frac_cells_improved'),'{:.0%}')]
                    st.markdown(' | '.join(f'{k} **{fmt(v,f)}**' for k,v,f in parts))
                # widths follow the png aspect ratios, so all images in the row get the same height
                row=[os.path.join(p,f'swe_compare_{day}.png'),os.path.join(p,f'swe_bias_{day}.png')]
                # ECENS bias and bias size change; runs from before that have the ECXENS minus ECENS map or nothing
                extra=[os.path.join(p,f'{n}_{day}.png') for n in ('swe_ecens_bias','swe_bias_gain')]
                extra=[x for x in extra if os.path.exists(x)] or [x for x in [os.path.join(p,f'swe_ecx_minus_ecens_{day}.png')] if os.path.exists(x)]
                row+=extra
                ratios=[img_ratio(fp,d) for fp,d in zip(row,[17.7/8]+[7/8]*(len(row)-1))]
                if len(row)>1 and sum(ratios)>6:
                    # wide domains (Alps, landscape figures): comparison on its own row, the bias maps side by side below it
                    img(row[0])
                    for col,fp in zip(st.columns(ratios[1:]),row[1:]):
                        with col: img(fp)
                else:
                    for col,fp in zip(st.columns(ratios),row):
                        with col: img(fp)

    # ---- everything else below the maps
    st.divider()
    st.subheader('Scores and summaries')
    # simple (the final model page): ensemble members first with only the domain mean plots, then the period summary
    names=(['Ensemble members','Period summary'] if simple else
           ['Scores vs lead day','Period summary','Ensemble members']+(['All figures'] if SHOW_ALL_FIGURES else [])+['Scores table'])
    T=dict(zip(names,st.tabs(names)))
    order={'name':names}

    if 'Scores vs lead day' in T:
        with T['Scores vs lead day']:
            c1,c2=st.columns([3,1])
            mets=c1.multiselect('scores',list(FC_METRICS),default=['bias','rmse','kge'],format_func=lambda m:FC_METRICS[m][0])
            snow=c2.radio('pixels',['all','snow'],horizontal=True,format_func=lambda s:'all' if s=='all' else 'snow (≥ 1 mm)')
            cols=st.columns(2)
            for i,m in enumerate(mets):
                k=m if snow=='all' else f'snow_{m}'
                if k not in df: continue
                f=px.line(df,x='lead_day',y=k,color='name',markers=True,height=340,title=FC_METRICS[m][0],category_orders=order,
                          labels={'lead_day':'lead day','name':'prediction',k:FC_METRICS[m][1] or m})
                if m=='bias': f.add_hline(y=0,line_color='black',line_width=0.6)
                with cols[i%2]: chart(f,key=f'fc_met_{m}')
            thrs=sorted({int(c.split('_')[1]) for c in df.columns if c.startswith('csi_')})
            if thrs:
                st.markdown('**Snow detection** (forecast and HKL above the threshold)')
                thr=st.radio('threshold (mm)',thrs,horizontal=True)
                cat=df.melt(id_vars=['lead_day','name'],value_vars=[f'{s}_{thr}' for s in ['pod','far','csi','hss'] if f'{s}_{thr}' in df],
                            var_name='score',value_name='value')
                cat['score']=cat['score'].str.split('_').str[0].str.upper()
                f=px.line(cat,x='lead_day',y='value',color='name',facet_col='score',markers=True,height=330,category_orders=order,
                          labels={'lead_day':'lead day','name':'prediction','value':''})
                f.for_each_annotation(lambda a:a.update(text=a.text.split('=')[-1]))
                f.update_yaxes(range=[min(0,float(cat['value'].min()) if cat['value'].notna().any() else 0),1])   # HSS can be negative
                chart(f,key='fc_cat')
            if {'vol_ecx_km3','vol_hkl_km3'}<=set(df.columns):
                vol=pd.concat([df[['lead_day','name','vol_ecx_km3']].rename(columns={'vol_ecx_km3':'km3'}),
                               df.drop_duplicates('lead_day')[['lead_day','vol_hkl_km3']].rename(columns={'vol_hkl_km3':'km3'}).assign(name='HKL')])
                f=px.line(vol,x='lead_day',y='km3',color='name',markers=True,height=340,title='total SWE volume (common valid pixels)',
                          category_orders={'name':names+['HKL']},labels={'lead_day':'lead day','name':'','km3':'km³ water'})
                chart(f,key='fc_vol')
            if 'skill_mse_vs_ecens' in df and df['skill_mse_vs_ecens'].notna().any():
                st.markdown('**ECXENS vs raw ECENS, both against HKL on the 0.1° ECENS grid** (same cells for both)')
                sc=st.radio('score',['rmse','bias','mae','kge'],horizontal=True,format_func=lambda m:FC_METRICS[m][0],key='fc_ecens_score')
                if f'ecens_{sc}' in df and f'ecx01_{sc}' in df:
                    both=pd.concat([df[['lead_day','name',f'ecens_{sc}']].rename(columns={f'ecens_{sc}':'v'}).assign(source='ECENS raw'),
                                    df[['lead_day','name',f'ecx01_{sc}']].rename(columns={f'ecx01_{sc}':'v'}).assign(source='ECXENS')])
                    c1,c2=st.columns(2)
                    f=px.line(both,x='lead_day',y='v',color='name',line_dash='source',markers=True,height=360,category_orders=order,
                              title=f'{FC_METRICS[sc][0]}, 0.1° (solid ECENS raw, dashed ECXENS)',
                              labels={'lead_day':'lead day','name':'prediction','v':FC_METRICS[sc][1] or sc,'source':''})
                    if sc=='bias': f.add_hline(y=0,line_color='black',line_width=0.6)
                    with c1: chart(f,key='fc_ecens_both')
                    sk=df.melt(id_vars=['lead_day','name'],value_vars=[c for c in ['skill_mse_vs_ecens','frac_cells_improved'] if c in df],
                               var_name='score',value_name='value')
                    sk['score']=sk['score'].map({'skill_mse_vs_ecens':'MSE skill vs ECENS','frac_cells_improved':'fraction of cells improved'})
                    f=px.line(sk,x='lead_day',y='value',color='name',facet_col='score',markers=True,height=360,category_orders=order,
                              labels={'lead_day':'lead day','name':'prediction','value':''})
                    f.for_each_annotation(lambda a:a.update(text=a.text.split('=')[-1]))
                    f.add_hline(y=0,line_color='black',line_width=0.6,col=1); f.add_hline(y=0.5,line_dash='dot',line_color='gray',col=2)
                    with c2: chart(f,key='fc_ecens_skill')
                    st.caption('MSE skill above 0 and more than half of the cells improved mean the downscaling brings the forecast closer to HKL.')

    if 'Period summary' in T:
        with T['Period summary']:
            summ=[]
            for p in preds:
                d=df[df['pred']==os.path.basename(p)]
                row={'prediction':fc_name(p),'days':len(d)}
                for k,l in [('bias','bias (mm)'),('mae','MAE (mm)'),('rmse','RMSE (mm)'),('r','r'),('kge','KGE'),
                            ('snow_bias','bias snow px (mm)'),('snow_rmse','RMSE snow px (mm)'),('snow_kge','KGE snow px'),
                            ('csi_1','CSI 1 mm'),('pod_1','POD 1 mm'),('far_1','FAR 1 mm')]:
                    if k in d: row[l]=d[k].mean()
                if {'vol_ecx_km3','vol_hkl_km3'}<=set(d.columns): row['SWE volume ECXENS / HKL']=(d['vol_ecx_km3']/d['vol_hkl_km3']).mean()
                for k,l in [('ecens_bias','0.1° bias ECENS (mm)'),('ecx01_bias','0.1° bias ECXENS (mm)'),('ecens_rmse','0.1° RMSE ECENS (mm)'),
                            ('ecx01_rmse','0.1° RMSE ECXENS (mm)'),('skill_mse_vs_ecens','MSE skill vs ECENS'),('frac_cells_improved','cells improved')]:
                    if k in d: row[l]=d[k].mean()
                summ.append(row)
            st.markdown('**Means over all lead days**')
            table(pd.DataFrame(summ).round(3),hide_index=True)
            figs={'swe_bias_mean_rmse.png':'mean bias and RMSE maps','swe_ecens_bias_gain_mean.png':'ECENS bias and MAE change vs ECENS',
                  'swe_density_scatter.png':'pixel density, all days pooled','swe_scores_vs_lead.png':'scores per lead day'}
            kind=st.radio('figure',list(figs),horizontal=True,format_func=figs.get,key='fc_period_fig')
            cols=st.columns(2)
            for i,p in enumerate(preds):
                with cols[i%2]:
                    st.markdown(f'**{fc_name(p)}**')
                    img(os.path.join(p,kind))

    if 'Ensemble members' in T:
        with T['Ensemble members']:
            if simple:
                fc_member_ts(run_path,base_time)   # only the domain mean plots of every member
            else:
                rk=read_csv(os.path.join(run_path,'member_ranking.csv'))
                if rk is None:
                    st.info('no member_ranking.csv in this run (written by the validation script when ADD_EXTREME_MEMBERS = True)')
                else:
                    stats=[c for c in MEMBER_STATS if c in rk]
                    rank_col=next((c for c in rk.columns if c.startswith('rank_')),None)
                    rank_by=rank_col[len('rank_'):] if rank_col and rank_col[len('rank_'):] in rk else (stats[0] if stats else None)
                    if rank_by is None:
                        st.info('member_ranking.csv has no snow amount columns')
                    else:
                        srt=rk.sort_values(rank_by)
                        dri,wet=int(srt['member'].iloc[0]),int(srt['member'].iloc[-1])
                        c1,c2=st.columns([1,4])
                        mcol=c1.radio('show',stats,index=stats.index(rank_by),format_func=MEMBER_STATS.get,key='fc_member_stat')
                        c1.metric('driest member',dri); c1.metric('wettest member',wet)
                        c1.caption(f'ranked by {MEMBER_STATS.get(rank_by,rank_by)}, {len(rk)} members')
                        missing=sorted(set(range(1,int(rk['member'].max())+1))-set(rk['member'].astype(int)))
                        if missing: c1.caption('missing members: '+', '.join(map(str,missing)))
                        d=rk.sort_values(mcol).copy()
                        d['member_s']=d['member'].astype(int).astype(str)
                        d['role']=np.select([d['member']==dri,d['member']==wet,d['member']==1],['driest','wettest','member 1'],'other')
                        f=px.bar(d,x='member_s',y=mcol,color='role',height=420,category_orders={'member_s':d['member_s'].tolist()},
                                 color_discrete_map={'driest':'#f5b800','wettest':'#1747c9','member 1':'#3fae49','other':'#b8b8b8'},
                                 labels={'member_s':'member (sorted)',mcol:f'{MEMBER_STATS.get(mcol,mcol)} (mm)','role':''},
                                 title=f'{MEMBER_STATS.get(mcol,mcol)} per ECXENS member')
                        f.add_hline(y=float(rk[mcol].mean()),line_dash='dot',line_color='black',annotation_text='mean of members')
                        with c2: chart(f,key='fc_members')
                        table(rk.round(3),hide_index=True)
                st.divider()
                fc_member_ts(run_path,base_time)

    if 'All figures' in T:
        with T['All figures']:
            p=st.selectbox('prediction',preds,format_func=fc_name,key='fc_allfig_pred')
            figs=sorted(glob.glob(os.path.join(p,'*.png')))
            if not figs:
                st.info(f'no figures in {p}')
            else:
                fnames=[os.path.basename(x) for x in figs]
                pick=st.multiselect(f'{len(figs)} figures in {p}',fnames,default=[n for n in fnames if not n.startswith(('swe_compare_','swe_bias_2'))],key='fc_allfig_pick')
                ncol=st.radio('columns',[1,2,3],index=1,horizontal=True,key='fc_allfig_cols')
                gc=st.columns(ncol)
                for i,x in enumerate([x for x in figs if os.path.basename(x) in pick]):
                    with gc[i%ncol]: img(x,os.path.basename(x))

    if 'Scores table' in T:
        with T['Scores table']:
            show=df.drop(columns=['pred']).rename(columns={'name':'prediction'})
            show['date']=show['date'].dt.date
            table(show.round(4),hide_index=True)
            st.download_button('download scores (csv)',show.to_csv(index=False),file_name=f'swe_metrics_{fc_rel(run_path).replace(os.sep,"_")}.csv',mime='text/csv')

def fc_member_ts(run_path,base_time):
    # figures of the validation script (MEMBER_TIMESERIES = True): domain mean SWE per day of every ECXENS / ECENS member and HKL
    st.markdown('**Domain mean SWE of every member against HKL**')
    pngs=[(os.path.join(run_path,'swe_members_ecxens.png'),'ECXENS members'),(os.path.join(run_path,'swe_members_ecens.png'),'raw ECENS members')]
    if any(os.path.exists(p) for p,_ in pngs): img_row(pngs)
    else: st.info('no swe_members_ecxens.png / swe_members_ecens.png in this run (written by the validation script when MEMBER_TIMESERIES = True)')

def fc_pred_label(lab):
    if lab in FC_NAMES: return FC_NAMES[lab].replace('ECXENS ','')
    if lab.startswith('member') and lab[6:].isdigit(): return f'member {int(lab[6:])}'
    return lab

def fc_date_key(r):
    return f"{r['base_time']:%Y-%m-%d}" if r['base_time'] is not None else os.path.basename(r['run'].rstrip('/'))

CMP_VIEWS=['Models side by side','One model in detail']

def page_compare():
    st.title('Compare model predictions: 10 and 15 days (500 m, ECXENS)')
    runs=fc_runs(PRED_DIR)
    if not runs:
        st.warning(f'no forecast validation results in {PRED_DIR}'); fc_no_runs(); return
    view=st.radio('view',CMP_VIEWS,horizontal=True,key='cmpm_view')
    region,runs=pick_region(runs,'cmpm_region_sel')
    if view==CMP_VIEWS[1]:
        fc_detail_any(runs,region); return
    st.caption('Raw ECENS, the ECXENS forecast of each model edition and HKL (Snowcap) for the same start date, prediction and day. '
               'Every map uses the same SWE legend and the same area.')

    # ---- start date, models with that start date, prediction shared by all picked models
    keys=sorted({fc_date_key(r) for r in runs if not case_hidden(fc_date_key(r))},reverse=True)
    c1,c2,c3=st.columns([2,3,1])
    key0=c1.selectbox('start date',keys,key=f'cmpm_date_{region}',format_func=lambda x:case_label(FC_CASES,region,x))
    key=f'{region}_{key0}'   # widget keys below are per region and start date
    by_model={}
    for r in runs:
        if fc_date_key(r)==key0: by_model.setdefault(r['model'],r)
    models=prod_first(by_model)
    pick=c2.multiselect('models',models,default=models,format_func=edition_name,key=f'cmpm_models_{key}')
    if not pick:
        st.info('pick at least one model'); return
    common=None
    for m in pick:
        labs={os.path.basename(p) for p in by_model[m]['preds']}
        common=labs if common is None else common&labs
    if not common:
        st.warning('the picked models have no prediction in common for this start date'); return
    labs=sorted(common,key=fc_sort_key)
    pred=c3.selectbox('prediction',labs,format_func=fc_pred_label,key=f'cmpm_pred_{key}')
    pdir={m:os.path.join(by_model[m]['run'],pred) for m in pick}
    if pred in ('driest','wettest'):
        st.caption(f'{pred} is ranked separately for each model, so the member can differ: '+
                   ', '.join(f'{edition_name(m)}: {fc_name(pdir[m])}' for m in pick))

    dates=None
    for m in pick:
        d={os.path.basename(f)[len('swe_compare_'):-4] for f in glob.glob(os.path.join(pdir[m],'swe_compare_*.png'))}
        dates=d if dates is None else dates&d
    dates=sorted(dates or [])
    if not dates:
        st.info('no daily maps shared by the picked models'); return
    bt=by_model[pick[0]]['base_time']
    lab=lambda t: f"{pd.Timestamp(t):%Y-%m-%d}"+(f" (lead {(pd.Timestamp(t)-bt).days} d)" if bt is not None else '')
    c1,c2=st.columns([4,1])
    day=c1.select_slider('day',options=dates,value=dates[0],format_func=lab,key=f'cmpm_day_{key}') if len(dates)>1 else dates[0]
    ref=pdir[pick[0]]   # raw ECENS and HKL are the same in every model folder, taken from the first model

    # ---- maps: raw ECENS, ECXENS per model, HKL
    panels=[(os.path.join(ref,f'swe_map_ecens_{day}.png'),'raw ECENS (0.1°)')]
    panels+=[(os.path.join(pdir[m],f'swe_map_ecxens_{day}.png'),f'ECXENS {edition_name(m)}') for m in pick]
    panels+=[(os.path.join(ref,f'swe_map_hkl_{day}.png'),'HKL (Snowcap)')]
    if all(os.path.exists(p) for p,_ in panels):
        ncol=c2.number_input('maps per row',min_value=2,max_value=8,value=min(len(panels),5),step=1,key='cmpm_ncol')
        for i in range(0,len(panels),int(ncol)):
            row=panels[i:i+int(ncol)]
            for col,(p,cap) in zip(st.columns(int(ncol)),row):
                with col:
                    st.markdown(f'**{cap}**'); img(p)
        if os.path.exists(os.path.join(ref,'swe_legend.png')):
            st.image(os.path.join(ref,'swe_legend.png'),width=900)
    else:
        # runs without single panels: each model's own figures side by side, one column per model
        st.caption('Each model\'s own figures side by side (raw ECENS | ECXENS | HKL, then ECXENS minus HKL). '
                   'For one row with a single ECENS and HKL panel, rerun the forecast validation with SAVE_PANELS = True.')
        for kind in ('swe_compare','swe_bias'):
            for col,m in zip(st.columns(len(pick)),pick):
                with col:
                    st.markdown(f'**{edition_name(m)}**'); img(os.path.join(pdir[m],f'{kind}_{day}.png'))

    # ---- bias maps of the same day
    with st.expander('bias maps for this day (forecast minus HKL)',expanded=False):
        bpan=[(os.path.join(ref,f'swe_ecens_bias_{day}.png'),'raw ECENS minus HKL (0.1°)')]
        bpan+=[(os.path.join(pdir[m],f'swe_bias_{day}.png'),f'ECXENS {edition_name(m)} minus HKL') for m in pick]
        bpan=[(p,c) for p,c in bpan if os.path.exists(p)]
        for i in range(0,len(bpan),4):
            for col,(p,cap) in zip(st.columns(4),bpan[i:i+4]):
                with col:
                    st.markdown(f'**{cap}**'); img(p)

    # ---- scores of this day and over all lead days, model against model
    parts=[]
    for m in pick:
        d=read_csv(os.path.join(pdir[m],'swe_metrics.csv'))
        if d is not None: parts.append(d.assign(model=edition_name(m)))
    if not parts: return
    df=pd.concat(parts,ignore_index=True); df['date']=pd.to_datetime(df['date'])
    ts=pd.Timestamp(day)
    st.subheader('Scores')
    cols={'bias':'bias (mm)','mae':'MAE (mm)','rmse':'RMSE (mm)','r':'r','kge':'KGE','csi_1':'CSI 1 mm',
          'ecx01_rmse':'0.1° RMSE ECXENS (mm)','ecens_rmse':'0.1° RMSE raw ECENS (mm)','skill_mse_vs_ecens':'MSE skill vs ECENS'}
    dd=df[df['date']==ts]
    if not dd.empty:
        st.markdown(f'**{lab(day)}**')
        table(dd.set_index('model')[[c for c in cols if c in dd]].rename(columns=cols).round(3))
    c1,c2=st.columns([2,2])
    met=c1.radio('score',['rmse','bias','mae','kge'],horizontal=True,format_func=lambda m:FC_METRICS[m][0],key='cmpm_met')
    res=c2.radio('grid',['500 m','0.1°'],horizontal=True,key='cmpm_res',
                 help='0.1°: ECXENS and raw ECENS against HKL on the same ECENS cells, so raw ECENS can be drawn too')
    if res=='500 m':
        long=df[['lead_day','model',met]].rename(columns={met:'v'})
    else:
        k=f'ecx01_{met}'
        if k not in df:
            st.info('no 0.1° scores in these runs'); return
        long=df[['lead_day','model',k]].rename(columns={k:'v'})
        if f'ecens_{met}' in df:
            e=df[df['model']==df['model'].iloc[0]][['lead_day',f'ecens_{met}']].rename(columns={f'ecens_{met}':'v'}).assign(model='raw ECENS')
            long=pd.concat([e,long])
    f=px.line(long,x='lead_day',y='v',color='model',markers=True,height=380,
              title=f'{FC_METRICS[met][0]} per lead day, {fc_pred_label(pred)} ({res})',
              labels={'lead_day':'lead day','v':FC_METRICS[met][1] or met,'model':''})
    if met=='bias': f.add_hline(y=0,line_color='black',line_width=0.6)
    f.update_traces(selector=dict(name='raw ECENS'),line=dict(dash='dash',color='gray'))
    chart(f,key='cmpm_lead')

    # ---- every start date both models have: mean scores model against model
    with st.expander('all start dates, model against model (means over all lead days of this prediction)',expanded=False):
        rows=[]
        for r in runs:
            if r['model'] not in pick: continue
            p=os.path.join(r['run'],pred)
            d=read_csv(os.path.join(p,'swe_metrics.csv'))
            if d is None or met not in d: continue
            rows.append({'start date':fc_date_key(r),'model':edition_name(r['model']),FC_METRICS[met][0]:d[met].mean()})
        if rows:
            t=pd.DataFrame(rows)
            table(t.pivot_table(index='start date',columns='model',values=FC_METRICS[met][0]).round(3))
            f=px.bar(t,x='start date',y=FC_METRICS[met][0],color='model',barmode='group',height=360)
            f.update_layout(xaxis_type='category',legend=dict(orientation='h',y=1.1),margin=dict(t=30))
            chart(f,key='cmpm_alldates')

# ---------------------------------------------------------------
# Seasonal pages: ECXSF monthly means validated against HKL, figures and scores from xgb-validate-swe-sf.py
# layout: validation_sf/<model>/<start date>/ as the script writes OUTDIR, e.g. validation_sf/FIN5/201803/ with
#           run_info.json                    base_time, region, grid, forecast_window, timeseries_domain, hkl_monthly
#           swe_domain_mean_timeseries.csv   domain mean SWE per time step: HKL, ECXSF ens mean and every member (and the png)
#           swe_metrics_monthly_all.csv      the predictions' monthly scores together (not needed, read per prediction)
#           <prediction>/                    ensmean, member00, ...: swe_metrics_monthly.csv, pred_info.json and the png maps
# one start date per model still works: validation_sf/<model>/<prediction>/ with run_info.json in <model>/
# the model folder name gives the model edition (MODEL_ALIASES: FIN5 -> FIN5noclim); 'model' in run_info.json is used only when
# the folder name is generic (ECXSF); a region suffix on the model folder (FIN5_Alps) gives the region
# ---------------------------------------------------------------
SF_ORDER=['ensmean','member00']
SF_NAMES={'ensmean':'ECXSF ens mean'}
# reference of a seasonal run: HKL (2.8 km script) or ERA5-Land (9 km script, 'reference' in run_info.json);
# key = name part of its map files (hkl_YYYYMM.png, bias_ecxsf_hkl_YYYYMM.png / era5l_..., bias_ecxsf_era5l_...) and time series column
SF_REFS={'hkl':'HKL','era5l':'ERA5-Land'}
def sf_ref(info):
    return 'era5l' if str(info.get('reference','')).lower().replace('-','').startswith('era5') else 'hkl'
def sf_fields(rk): return [('ecsf','raw ECSF sd'),('ecxsf','ECXSF SWE'),(rk,f'{SF_REFS[rk]} SWE')]
def sf_bias(rk): return [(f'bias_ecxsf_{rk}',f'ECXSF minus {SF_REFS[rk]}'),(f'bias_ecsf_{rk}',f'ECSF minus {SF_REFS[rk]} (1°)')]
SF_LEAD_FILE='swe_scores_lead_month_ensmean.csv'   # 9 km run folder: ens mean RMSE and ME per lead month and region
SF_REGION_FILE='swe_metrics_monthly_regions.csv'   # 9 km prediction folder: monthly scores per region
SF_SOURCES={'ecx':'ECXSF (own grid)','ecx1':'ECXSF (1° cells)','ecsf':'ECSF raw (1° cells)'}
# scores in swe_metrics_monthly.csv (with a source prefix ecx_, ecx1_, ecsf_); 'mean' is the mean SWE, written also without HKL
SF_METRICS=dict(FC_METRICS,mean=('mean SWE','mm'))
# seasonal grids: one page each; runs are sorted to a resolution by run_info.json, a folder suffix or the folder they are in
SF_RES=['2.8 km','9 km']
SF_RES_SPLIT_DEG=0.05   # grid step (degrees) at or below this is 2.8 km (ECXSF 0.025°), above it 9 km
SF_RES_SPLIT_KM=5.0     # same split for names given in km (2.8km, 3km, 9km)
SF_RES_SEP=' @ '        # compare page, both resolutions: model key 'FIN5noclim @ 9 km'
SF_GENERIC_MODELS={'ECXSF'}   # 'model' values in run_info.json that name the product, not a model edition (the script's default)
SF_TS_FILE='swe_domain_mean_timeseries.csv'   # run folder: domain mean SWE per time step, columns valid_time, hkl, ecxsf_ensmean, ecxsf_m00..
SF_TS_ONLY={'2.8 km'}   # simpler seasonal SWE predictions (past) pages: only the domain mean time series below the maps (no other score tabs) and no 'scores per start date, chart' expander

def sf_res_name(v):
    # '0.025°', 0.1, '9km', '2.8 km' -> '2.8 km' or '9 km'; None when it can not be read
    if v is None or v=='': return None
    s=str(v).strip().lower().replace(' ','')
    try:
        if s.endswith('km'): return SF_RES[0] if float(s[:-2])<=SF_RES_SPLIT_KM else SF_RES[1]
        return SF_RES[0] if float(s.rstrip('°').replace('deg',''))<=SF_RES_SPLIT_DEG else SF_RES[1]
    except ValueError:
        return None

def sf_split_res(folder):
    # 'FIN5_9km' -> ('FIN5','9 km'), 'FIN5' -> ('FIN5',None)
    for sep in ('_','-'):
        if sep in folder:
            head,tail=folder.rsplit(sep,1)
            if tail.lower().endswith('km') and sf_res_name(tail): return head,sf_res_name(tail)
    return folder,None

def sfm_name(m):
    # shown model name; 'FIN5noclim @ 9 km' (both resolutions on the compare page) -> 'FIN5noclim (final model) (9 km)'
    ed,_,rs=str(m).partition(SF_RES_SEP)
    return edition_name(ed)+(f' ({rs})' if rs else '')

def sf_model_order(models):
    # final model first (at every resolution), then by name
    return sorted(models,key=lambda m:(str(m).partition(SF_RES_SEP)[0]!=PROD_MODEL,str(m)))

def sf_res_runs(runs,res):
    # runs of one resolution; 'both' keeps every run and makes model + resolution the model key
    if res=='both': return [dict(r,model=f"{r['model']}{SF_RES_SEP}{r['res']}") for r in runs]
    return [r for r in runs if r['res']==res]

def sf_has_hkl(df): return df is not None and 'ecx_bias' in df and df['ecx_bias'].notna().any()
def sf_metric_opts(df): return [m for m in SF_METRICS if f'ecx_{m}' in df and df[f'ecx_{m}'].notna().any()]

def sf_sort_key(p):
    lab=os.path.basename(p.rstrip('/'))
    return (SF_ORDER.index(lab) if lab in SF_ORDER else len(SF_ORDER),lab)

def sf_name(p):
    # pred_info.json written by the seasonal validation script, then SF_NAMES, then member<N> -> 'ECXSF member N'
    name=read_json(os.path.join(p,'pred_info.json')).get('name')
    if name: return name
    lab=os.path.basename(p.rstrip('/'))
    if lab in SF_NAMES: return SF_NAMES[lab]
    if lab.startswith('member') and lab[6:].isdigit(): return f'ECXSF member {int(lab[6:])}'
    return lab

def sf_short(lab):
    if lab=='ensmean': return 'ens mean'
    if lab.startswith('member') and lab[6:].isdigit(): return f'member {int(lab[6:])}'
    return lab

@st.cache_data(ttl=60,show_spinner=False)
def sf_runs(sdir,default_res=SF_RES[0],skip=()):
    # every folder with swe_metrics_monthly.csv is a prediction; predictions with the same parent folder form one run
    # default_res: resolution of runs that do not say it themselves; skip: folders left out (the 9 km folder inside SF_DIR)
    sdir=os.path.normpath(sdir)
    if not os.path.isdir(sdir): return []
    skip={os.path.normpath(s) for s in skip}
    preds=[]
    for root,dirs,files in os.walk(sdir):
        dirs[:]=sorted(d for d in dirs if d not in FC_SKIP_DIRS and not d.startswith('.') and os.path.normpath(os.path.join(root,d)) not in skip)
        if os.path.relpath(root,sdir).count(os.sep)>=4: dirs[:]=[]
        if 'swe_metrics_monthly.csv' in files: preds.append(os.path.normpath(root))
    runs={}
    for p in preds:
        par=os.path.dirname(p)
        run=par if (p!=sdir and par!=sdir) else p   # <model>/<prediction>/ and <model>/<start date>/<prediction>/ both group by parent
        runs.setdefault(run,[]).append(p)
    out=[]
    for run,ps in runs.items():
        ps=sorted(ps,key=sf_sort_key)
        info=read_json(os.path.join(run,'run_info.json'))
        bt=None
        if info.get('base_time'):
            try: bt=pd.Timestamp(info['base_time'])
            except (ValueError,TypeError): bt=None
        if bt is None: bt=fc_date_from_name(os.path.basename(run))
        if bt is None:   # first month minus (lead month - 1)
            try:
                d=pd.read_csv(os.path.join(ps[0],'swe_metrics_monthly.csv'),usecols=['month','lead_month']).iloc[0]
                bt=pd.Timestamp(f"{d['month']}-01")-pd.DateOffset(months=int(d['lead_month'])-1)
            except Exception: bt=None
        rp=os.path.dirname(run)
        # the model folder name wins over 'model' in run_info.json: the validation script writes its default 'ECXSF'
        # there when it is run without a model name, and the folder is what the results were copied into
        if run==sdir: mfolder=info.get('model') or 'ECXSF'
        elif rp==sdir: mfolder=os.path.basename(run)   # validation_sf/FIN5/ensmean/
        else: mfolder=os.path.basename(rp)              # validation_sf/FIN5/20231101/ensmean/
        if mfolder in SF_GENERIC_MODELS and info.get('model') and info['model'] not in SF_GENERIC_MODELS: mfolder=info['model']
        model,res_f=sf_split_res(mfolder)              # FIN5_Alps_9km
        model,region=fc_split_model(model)
        if res_f is None: model,res_f=sf_split_res(model)   # FIN5_9km_Alps
        model=MODEL_ALIASES.get(model,model)
        region=fc_region_name(info.get('region')) or region or FC_DEFAULT_REGION
        res=sf_res_name(info.get('resolution')) or sf_res_name(info.get('grid')) or res_f or default_res
        out.append({'run':run,'preds':ps,'base_time':bt,'model':model,'region':region,'res':res})
    # start dates in CASE_HIDE are left out here, so they are in no table, chart or menu of any page
    out=[r for r in out if r['base_time'] is None or not case_hidden(f"{r['base_time']:%Y-%m-%d}")]
    return sorted(out,key=lambda r:r['base_time'] if r['base_time'] is not None else pd.Timestamp('1900-01-01'),reverse=True)

def sf_all_runs():
    # 2.8 km and 9 km runs: SF_DIR (default 2.8 km) and SF9_DIR (default 9 km), newest start date first
    same=os.path.normpath(SF9_DIR)==os.path.normpath(SF_DIR)
    out=sf_runs(SF_DIR,SF_RES[0],() if same else (SF9_DIR,))
    if not same: out=out+sf_runs(SF9_DIR,SF_RES[1])
    # start dates in CASE_HIDE are left out here, so they are in no table, chart or menu of any page
    out=[r for r in out if r['base_time'] is None or not case_hidden(f"{r['base_time']:%Y-%m-%d}")]
    return sorted(out,key=lambda r:r['base_time'] if r['base_time'] is not None else pd.Timestamp('1900-01-01'),reverse=True)

def sf_rel(run):
    # run folder relative to the seasonal folder it is in
    rn=os.path.normpath(run)
    for root in (SF9_DIR,SF_DIR):
        root=os.path.normpath(root)
        if rn==root: return os.path.basename(root)
        if rn.startswith(root+os.sep): return os.path.relpath(rn,root)
    return run

def sf_date_key(r):
    return f"{r['base_time']:%Y-%m-%d}" if r['base_time'] is not None else sf_rel(r['run'])

def sf_metrics(preds):
    parts=[]
    for p in preds:
        d=read_csv(os.path.join(p,'swe_metrics_monthly.csv'))
        if d is not None: parts.append(d.assign(pred=os.path.basename(p),name=sf_name(p)))
    return pd.concat(parts,ignore_index=True) if parts else None

def sf_months(folders):
    # forecast months (YYYYMM) that have the ECXSF map in every folder
    out=None
    for p in folders:
        m={os.path.basename(f)[len('ecxsf_'):-4] for f in glob.glob(os.path.join(p,'ecxsf_*.png'))}
        out=m if out is None else out&m
    return sorted(out or [])

def sf_month_label(m,bt):
    t=pd.Timestamp(f'{m[:4]}-{m[4:]}-01')
    if bt is None: return f'{t:%Y-%m}'
    return f'{t:%Y-%m} (lead month {(t.year-bt.year)*12+t.month-bt.month+1})'

def sf_no_runs(res=SF_RES[0]):
    if res==SF_RES[1]:
        st.markdown(f'Copy the output of `xgb-validate-swe-sf-9km.py` here as `<model>/<start date>/`, for example `FIN5/20180301/` '
                    'with `ensmean/`, `member00/`, `run_info.json` and the time series files:')
        st.code(f'rsync -av ubuntu@smartmet-destine:/home/ubuntu/data/snowmodel_validation/figs_swe_seasonal_9km/ {SF9_DIR}/',language='bash')
        return
    st.markdown(f'Copy the {res} output of `xgb-validate-swe-sf.py` here as `<model>/<start date>/<prediction>/`, for example '
                '`FIN5/20231101/ensmean/` (Alps: `FIN5_Alps/20231101/ensmean/`). '
                "A run's own `grid` in run_info.json, or a `_9km` / `_2.8km` suffix on the model folder, overrides the folder it is in.")
    st.code(f'rsync -av smartmet@firedanger.nsdc.fmi.fi:/home/users/smartmet/data/swe-validation/figs_swe_seasonal/ {SF_DIR}/',language='bash')

def sf_ts_regions(run,info=None):
    # regions with a time series in this run, in the order of run_info.json: [(key, name, points)]; [] for one domain runs (2.8 km)
    info=info if info is not None else read_json(os.path.join(run,'run_info.json'))
    regs=info.get('regions') or {}
    keys=list(regs)+sorted(os.path.basename(p)[len('swe_domain_mean_timeseries_'):-4]
                           for p in glob.glob(os.path.join(run,'swe_domain_mean_timeseries_*.csv')))
    out=[]
    for k in dict.fromkeys(keys):
        if os.path.exists(os.path.join(run,f'swe_domain_mean_timeseries_{k}.csv')):
            r=regs.get(k,{}); out.append((k,r.get('name',k),r.get('points')))
    return out

def sf_timeseries(run,region=None):
    # domain mean time series csv of one run (region: key of a regional file, 9 km); None when there is none.
    # Without a region: swe_domain_mean_timeseries.csv, else the whole domain regional file (europe)
    p=os.path.join(run,f'swe_domain_mean_timeseries_{region}.csv' if region else SF_TS_FILE)
    if region is None and not os.path.exists(p):
        p=os.path.join(run,'swe_domain_mean_timeseries_europe.csv')
    d=read_csv(p)
    if d is None or 'valid_time' not in d: return None
    d=d.copy(); d['valid_time']=pd.to_datetime(d['valid_time'])
    return d.sort_values('valid_time')

def sf_ts_members(d): return [c for c in d.columns if c.startswith('ecxsf_m') and c[7:].isdigit()]

def sf_ts_ref(d):
    # reference column of a time series and its name: ('hkl','HKL') or ('era5l','ERA5-Land'); (None, None) without one
    return next(((c,n) for c,n in SF_REFS.items() if c in d and d[c].notna().any()),(None,None))

def sf_ts_scores(d):
    # domain mean series: ECXSF ens mean against the reference, peaks and how often the reference is inside the member range
    out={}
    if 'ecxsf_ensmean' in d:
        i=d['ecxsf_ensmean'].idxmax()
        if pd.notna(i): out['peak ens mean']=(d.at[i,'ecxsf_ensmean'],d.at[i,'valid_time'])
    rc,rn=sf_ts_ref(d)
    if rc is None: return out
    out['ref']=rn
    i=d[rc].idxmax(); out['peak ref']=(d.at[i,rc],d.at[i,'valid_time'])
    if 'ecxsf_ensmean' in d:
        b=d.dropna(subset=[rc,'ecxsf_ensmean'])
        if len(b)>1:
            e=b['ecxsf_ensmean']-b[rc]
            out['bias']=e.mean(); out['RMSE']=float(np.sqrt((e**2).mean()))
            out['r']=b['ecxsf_ensmean'].corr(b[rc])
    mem=sf_ts_members(d)
    if mem:
        b=d.dropna(subset=[rc])
        lo,hi=b[mem].min(axis=1),b[mem].max(axis=1)
        ok=lo.notna()
        if ok.any(): out['ref inside members']=float(((b[rc]>=lo)&(b[rc]<=hi))[ok].mean())
    return out

def sf_ts_traces(fig,d,show='lines',row=None,col=None,legend=True):
    # members (thin blue lines or a 10 to 90 % band), ensemble mean black dotted, reference red, as in the validation script figure
    mem=sf_ts_members(d)
    t=d['valid_time']
    kw=dict(row=row,col=col) if row else {}
    if mem and show=='lines':
        for i,c in enumerate(mem):
            fig.add_scatter(x=t,y=d[c],mode='lines',line=dict(color='#1f77b4',width=0.7),opacity=0.45,hoverinfo='skip',
                            name=f'ECXSF members ({len(mem)})',legendgroup='members',showlegend=legend and i==0,**kw)
    elif mem and show=='band':
        q=d[mem].quantile([0.1,0.9],axis=1).T
        fig.add_scatter(x=t,y=q[0.9],mode='lines',line=dict(width=0),hoverinfo='skip',showlegend=False,legendgroup='band',**kw)
        fig.add_scatter(x=t,y=q[0.1],mode='lines',line=dict(width=0),fill='tonexty',fillcolor='rgba(31,119,180,0.25)',
                        name=f'ECXSF members 10 to 90 % ({len(mem)})',hoverinfo='skip',legendgroup='band',showlegend=legend,**kw)
    rc,rn=sf_ts_ref(d)
    if rc is not None:   # reference below the ens mean, so the dotted line stays visible
        h=d.dropna(subset=[rc])
        fig.add_scatter(x=h['valid_time'],y=h[rc],mode='lines',line=dict(color='#d7191c',width=2),name=rn,hovertemplate='%{y:.1f} mm',
                        legendgroup='ref',showlegend=legend,**kw)
    if 'ecxsf_ensmean' in d:
        fig.add_scatter(x=t,y=d['ecxsf_ensmean'],mode='lines',line=dict(color='black',dash='dot',width=2.2),name='ECXSF ensemble mean',
                        hovertemplate='%{y:.1f} mm',legendgroup='ens',showlegend=legend,**kw)

def sf_ts_fig(d,show='lines',height=460,title=None):
    fig=go.Figure()
    sf_ts_traces(fig,d,show)
    fig.update_layout(height=height,title=title,yaxis_title='domain mean SWE (mm)',hovermode='x unified',
                      legend=dict(orientation='h',y=1.08),margin=dict(t=60 if title else 30))
    return fig

def sf_ts_view(run,info,key):
    # domain mean time series of one run, interactive from the csv (the validation script png when there is no csv);
    # 9 km runs: one series per region, picked here, or all regions side by side
    regs=sf_ts_regions(run,info)
    region=None
    if regs:
        names={k:n for k,n,_ in regs}
        opts=[k for k,_,_ in regs]+(['_all'] if len(regs)>1 else [])
        rkey=f'sf_ts_region_{key}'
        if st.session_state.get(rkey) not in opts: st.session_state[rkey]=opts[0]
        c0,c00=st.columns([5,1])
        region=c0.radio('region',opts,horizontal=True,key=rkey,format_func=lambda k:'all regions' if k=='_all' else names[k])
        mp=os.path.join(run,'regions_map.png')
        if os.path.exists(mp):
            with c00.popover('regions on the map') if hasattr(st,'popover') else c00.expander('regions on the map'):
                img(mp)
        if region=='_all':
            sf_ts_all_regions(run,regs,key); return
    d=sf_timeseries(run,region)
    if d is None:
        p=os.path.join(run,f'swe_domain_mean_timeseries_{region}.png' if region else 'swe_domain_mean_timeseries.png')
        if os.path.exists(p): img(p)
        else: st.info(f'no {SF_TS_FILE} in this run (written by the validation script when the daily ECXSF member files are there)')
        return
    mem=sf_ts_members(d)
    c1,c2=st.columns([1,4])
    show=c1.radio('members',['lines','band','off'],horizontal=False,key=f'sf_ts_show_{key}',disabled=not mem,
                  format_func={'lines':'every member','band':'10 to 90 % band','off':'ens mean only'}.get)
    s=sf_ts_scores(d); rn=s.get('ref','reference')
    for lab,k,f in [('peak ECXSF ens mean','peak ens mean','{:.0f} mm'),(f'peak {rn}','peak ref','{:.0f} mm')]:
        if k in s: c1.metric(lab,fmt(s[k][0],f),f"{s[k][1]:%Y-%m-%d}",delta_color='off')
    for lab,k,f in [(f'bias, ens mean minus {rn}','bias','{:+.1f} mm'),('RMSE','RMSE','{:.1f} mm'),('r','r','{:.3f}'),
                    (f'{rn} inside the member range','ref inside members','{:.0%}')]:
        if k in s: c1.metric(lab,fmt(s[k],f))
    with c2: chart(sf_ts_fig(d,show),key=f'sf_ts_{key}_{region}')
    if region:
        r=(info.get('regions') or {}).get(region,{})
        dom=f"{r.get('name',region)}, {r.get('points','?')} points valid in {SF_REFS[sf_ref(info)]} and ECXSF (glaciers left out)"
    else:
        dom=info.get('timeseries_domain','forecast domain')
    st.caption(f"domain mean SWE over {dom}, every time step of the forecast window"
               + (f" ({info['forecast_window'][0]} to {info['forecast_window'][1]})" if info.get('forecast_window') else '')
               + '. Scores and peaks use the domain mean series, so they say how well the total snow amount evolves, not where it is.')
    with st.expander('time series table'):
        table(d.set_index('valid_time').round(2).reset_index(),hide_index=True)
        st.download_button('download time series (csv)',d.to_csv(index=False),
                           file_name=f'swe_domain_mean_{sf_rel(run).replace(os.sep,"_")}{"_"+region if region else ""}.csv',
                           mime='text/csv',key=f'sf_ts_dl_{key}_{region}')

def sf_ts_all_regions(run,regs,key):
    # every region side by side, members as a 10 to 90 % band, and their scores in one table
    from plotly.subplots import make_subplots
    ncol=3; nrow=-(-len(regs)//ncol)
    fig=make_subplots(rows=nrow,cols=ncol,subplot_titles=[n for _,n,_ in regs],shared_xaxes=True,vertical_spacing=0.08,horizontal_spacing=0.05)
    rows=[]
    for i,(k,n,pts) in enumerate(regs):
        d=sf_timeseries(run,k)
        if d is None: continue
        sf_ts_traces(fig,d,'band',row=i//ncol+1,col=i%ncol+1,legend=i==0)
        s=sf_ts_scores(d)
        rows.append({'region':n,'points':pts,'peak ens mean (mm)':s.get('peak ens mean',(np.nan,))[0],
                     f"peak {s.get('ref','reference')} (mm)":s.get('peak ref',(np.nan,))[0],
                     'bias (mm)':s.get('bias'),'RMSE (mm)':s.get('RMSE'),'r':s.get('r'),'reference inside members':s.get('ref inside members')})
    fig.update_layout(height=260*nrow+60,hovermode='x unified',legend=dict(orientation='h',y=1.04),margin=dict(t=60))
    fig.update_yaxes(matches=None)
    chart(fig,key=f'sf_ts_all_{key}')
    st.caption('Each panel has its own SWE scale (mm). Members as the 10 to 90 % band, ens mean black dotted, reference red.')
    if rows: table(pd.DataFrame(rows).round(3),hide_index=True)

def sf_lead_table(run,height=None):
    # lead month table of a 9 km run (ens mean RMSE and ME per lead month and region against ERA5-Land); None when there is none
    t=read_csv(os.path.join(run,SF_LEAD_FILE))
    if t is None: return None
    return t

def sf_lead_view(run,key):
    t=sf_lead_table(run)
    if t is None:
        st.info(f'no {SF_LEAD_FILE} in this run (written by xgb-validate-swe-sf-9km.py)'); return
    num=[c for c in t.columns if c not in ('lead month','month','ERA5-Land days')]
    cfg={c:st.column_config.NumberColumn(c,format='%.1f') for c in num}
    cfg['lead month']=st.column_config.NumberColumn('lead month',help='0 = forecast start month')
    table(t,hide_index=True,column_config=cfg)
    st.caption('ECXSF ensemble mean monthly mean against ERA5-Land, over the points of each region (glaciers left out). '
               'ME = forecast minus ERA5-Land, positive = too much snow. Lead month 0 is the forecast start month; '
               'ERA5-Land days shows how many days the month averages, fewer on the partial last month.')
    long=t.melt(id_vars=['lead month'],value_vars=num,var_name='column',value_name='mm')
    long['score']=long['column'].str.split(' ').str[0]
    long['region']=long['column'].str.split(' ',n=1).str[1].str.replace(' (mm)','',regex=False)
    f=px.line(long,x='lead month',y='mm',color='region',facet_col='score',markers=True,height=340,labels={'region':''})
    f.for_each_annotation(lambda a:a.update(text=a.text.split('=')[-1]))
    f.update_yaxes(matches=None); f.add_hline(y=0,line_color='black',line_width=0.6,col=2)
    chart(f,key=f'sf_lead_{key}')
    st.download_button('download table (csv)',t.to_csv(index=False),file_name=f'lead_month_{sf_rel(run).replace(os.sep,"_")}.csv',
                       mime='text/csv',key=f'sf_lead_dl_{key}')

def sf_regions_view(preds,run,key):
    # monthly scores per region of one prediction (swe_metrics_monthly_regions.csv, 9 km)
    have=[p for p in preds if os.path.exists(os.path.join(p,SF_REGION_FILE))]
    if not have:
        st.info(f'no {SF_REGION_FILE} in this run'); return
    c1,c2=st.columns([2,3])
    p=c1.selectbox('prediction',have,format_func=sf_name,key=f'sf_reg_pred_{key}') if len(have)>1 else have[0]
    rd=read_csv(os.path.join(p,SF_REGION_FILE))
    mets={'rmse':'RMSE (mm)','bias':'mean error (mm)','mae':'MAE (mm)','r':'correlation r','kge':'KGE','mean':'mean SWE (mm)'}
    mets={k:v for k,v in mets.items() if (k=='mean' and 'ecx_mean' in rd) or f'ecx_{k}' in rd}
    met=c2.radio('score',list(mets),horizontal=True,format_func=mets.get,key=f'sf_reg_met_{key}')
    if met=='mean':
        long=pd.concat([rd[['month','region_name','ecx_mean']].rename(columns={'ecx_mean':'v'}).assign(source='ECXSF'),
                        rd[['month','region_name','ref_mean']].rename(columns={'ref_mean':'v'}).assign(source='ERA5-Land')])
        f=px.line(long,x='month',y='v',color='region_name',line_dash='source',markers=True,height=400,
                  labels={'month':'forecast month','v':mets[met],'region_name':'','source':''})
    else:
        f=px.line(rd,x='month',y=f'ecx_{met}',color='region_name',markers=True,height=400,
                  labels={'month':'forecast month',f'ecx_{met}':mets[met],'region_name':''})
        if met=='bias': f.add_hline(y=0,line_color='black',line_width=0.6)
    f.update_traces(selector=dict(name='Europe (whole domain)'),line=dict(width=3.5,color='black'))
    chart(f,key=f'sf_reg_{key}')
    col='ecx_mean' if met=='mean' else f'ecx_{met}'
    table(rd.pivot_table(index='month',columns='region_name',values=col,sort=False).round(2))
    st.caption(f'ECXSF {sf_name(p)} against ERA5-Land, monthly means over each region\'s points (glaciers left out).')
    mp=os.path.join(run,'regions_map.png')
    if os.path.exists(mp):
        with st.expander('regions on the map'): img(mp)

def img_row(paths,default_ratio=0.9):
    # images side by side with the same height (column widths follow the png aspect ratios); missing pngs are left out
    paths=[(p,c) for p,c in paths if os.path.exists(p)]
    if not paths: return
    for col,(p,cap) in zip(st.columns([img_ratio(p,default_ratio) for p,_ in paths]),paths):
        with col:
            if cap: st.markdown(f'**{cap}**')
            img(p)

def sf_case_scores(r):
    # mean scores over all forecast months of the ens mean (or the first prediction) of one run, for the start date table
    p=next((p for p in r['preds'] if os.path.basename(p)=='ensmean'),r['preds'][0])
    d=read_csv(os.path.join(p,'swe_metrics_monthly.csv'))
    if d is None: return {}
    out={}
    for k,l in [('ecx_mean','mean SWE (mm)'),('ecx_bias','bias (mm)'),('ecx_rmse','RMSE (mm)'),('ecx_kge','KGE'),('skill_mse_vs_ecsf','MSE skill vs ECSF')]:
        if k in d: out[l]=d[k].mean()
    out['compared with']=SF_REFS[sf_ref(read_json(os.path.join(r['run'],'run_info.json')))] if sf_has_hkl(d) else 'none'
    return out

def page_seasonal_28(): page_seasonal(SF_RES[0])
def page_seasonal_9(): page_seasonal(SF_RES[1])

def page_seasonal(res):
    st.title(f'SWE predictions (past): seasonal 215 days ({res}, ECXSF)')
    allruns=[r for r in sf_all_runs() if r['res']==res]
    model=used_model(prod_first({r['model'] for r in allruns})) or USED_MODEL
    runs=[r for r in allruns if r['model']==model]
    cases={k:v for k,v in SF_CASES.items() if k in SF_CASE_REGIONS.get(res,SF_CASES)}
    region,runs=region_choice(runs,cases,f'sf_region_sel_{res}')
    other=[x for x in SF_RES if x!=res]
    by=case_overview(cases.get(region,{}),runs,sf_date_key,sf_case_scores,key=f'sf_cases_{res}_{region}',chart_box=res not in SF_TS_ONLY,
                     sources=case_sources(SF_CASE_SOURCES,region))
    if not by:
        st.info(f'no {edition_name(model)} {res} seasonal results for {region} yet'); sf_no_runs(res); return
    keys=sorted(by)
    skey=f'sf_case_sel_{res}_{region}'
    if st.session_state.get(skey) not in keys: st.session_state[skey]=keys[0]
    k=st.selectbox('start date',keys,key=skey,format_func=lambda x:case_label(SF_CASES,region,x))
    st.divider()
    st.header(case_label(SF_CASES,region,k))
    sf_run_view(by[k],region,model,ts_only=res in SF_TS_ONLY)

def sf_detail_any(runs,region,res):
    # one model in detail, any model edition and start date (the compare page); res: resolution picked there (or 'both')
    models=sf_model_order({r['model'] for r in runs})
    nstart={m:sum(r['model']==m for r in runs) for m in models}
    mkey=f'sf_model_pick_{res}_{region}'
    if st.session_state.get(mkey) not in models:
        st.session_state[mkey]=default_model(models)
    c1,c2=st.columns([2,3])
    with c1:
        model=st.radio('model',models,horizontal=True,key=mkey,
                       format_func=lambda m:f'{sfm_name(m)} ({nstart[m]} start date{"s" if nstart[m]!=1 else ""})')
    runs=[r for r in runs if r['model']==model and not case_hidden(sf_date_key(r))]
    paths=[r['run'] for r in runs]
    if not paths:
        st.info('no start dates with results for this model'); return
    skey=f'sf_run_sel_{res}_{region}_{model}'
    if st.session_state.get(skey) not in paths: st.session_state[skey]=paths[0]
    def lab(p):
        r=next(r for r in runs if r['run']==p)
        return case_label(SF_CASES,region,sf_date_key(r))+f"  ({sf_rel(r['run'])}, {len(r['preds'])} predictions)"
    with c2:
        run_path=st.selectbox('start date',paths,key=skey,format_func=lab)
    sf_run_view(next(r for r in runs if r['run']==run_path),region,model)

def sf_run_view(R,region,model,ts_only=False):
    # ts_only: below the monthly maps only the domain mean time series, the other score tabs are left out
    # monthly maps, scores and figures of one seasonal run (one model, region and start date)
    run_path,preds,bt=R['run'],R['preds'],R['base_time']
    df=sf_metrics(preds)
    if df is None:
        st.warning('no readable swe_metrics_monthly.csv in this run'); return
    hkl=sf_has_hkl(df)   # scores against the reference (HKL or ERA5-Land) are there
    info=read_json(os.path.join(run_path,'run_info.json'))
    rk=sf_ref(info); REF=SF_REFS[rk]
    fw=info.get('forecast_window')
    if not hkl:
        st.info(f'No {REF} for this start date: the validation script wrote the forecast maps and mean SWE only, no {REF} comparison.')

    # ---- monthly maps: raw ECSF, ECXSF and HKL on their own grids, then the two bias maps
    st.subheader('Monthly maps')
    months=sorted({m for p in preds for m in sf_months([p])})
    if not months:
        st.info('no ecxsf_YYYYMM.png figures in this run')
    else:
        c1,c2=st.columns([3,2])
        if len(months)>1:
            mon=c1.select_slider('month',options=months,value=months[0],format_func=lambda m:sf_month_label(m,bt),key=f'sf_mon_{run_path}')
        else:
            mon=months[0]; c1.markdown(f'**month** {sf_month_label(mon,bt)}')
        show=c2.multiselect('predictions',preds,default=preds,format_func=sf_name,key=f'sf_show_{run_path}')
        if hkl:
            st.caption(f'Top row: raw ECSF, ECXSF and {REF}, each on its own grid, same SWE legend. Bottom row: ECXSF minus {REF} '
                       f'and raw ECSF minus {REF} (1°, {REF} averaged into the cells). Bias colours: soft within ±{MILD_BIAS:g} mm, strong beyond.')
        else:
            st.caption(f'Raw ECSF and ECXSF, each on its own grid and area, same SWE legend (no {REF} for this run).')
        ym=f'{mon[:4]}-{mon[4:]}'
        for p in show:
            st.markdown(f'#### {sf_name(p)}')
            r=df[(df['pred']==os.path.basename(p))&(df['month']==ym)]
            if not r.empty:
                r=r.iloc[0]
                parts=[('mean SWE',r.get('ecx_mean'),'{:.1f} mm'),('max',r.get('ecx_max'),'{:.0f} mm'),
                       ('1° mean ECXSF',r.get('ecx1_mean'),'{:.1f} mm'),('ECSF',r.get('ecsf_mean'),'{:.1f} mm'),
                       ('bias',r.get('ecx_bias'),'{:+.1f} mm'),('RMSE',r.get('ecx_rmse'),'{:.1f} mm'),('KGE',r.get('ecx_kge'),'{:.3f}'),
                       ('1° RMSE ECSF',r.get('ecsf_rmse'),'{:.1f} mm'),('ECXSF',r.get('ecx1_rmse'),'{:.1f} mm'),
                       ('MSE skill vs ECSF',r.get('skill_mse_vs_ecsf'),'{:+.2f}'),('cells improved',r.get('frac_cells_improved'),'{:.0%}')]
                hd=r.get('hkl_days',r.get('ref_days'))   # days the reference was averaged over (forecast window only): fewer on a partial last month
                if pd.notna(hd) and hd>0:
                    dim=pd.Timestamp(ym+'-01').days_in_month
                    parts.append((f'{REF} days',f"{int(hd)} of {dim}"+(' (partial month)' if hd<dim else ''),'{}'))
                if pd.notna(r.get('ref_mean')): parts.insert(2,(f'{REF} mean',r.get('ref_mean'),'{:.1f} mm'))
                st.markdown(' | '.join(f'{k} **{fmt(v,f)}**' for k,v,f in parts if pd.notna(v)))
            img_row([(os.path.join(p,f'{k}_{mon}.png'),None) for k,_ in sf_fields(rk)])
            img_row([(os.path.join(p,f'{k}_{mon}.png'),None) for k,_ in sf_bias(rk)])

    st.divider()
    if ts_only:
        # only the domain mean time series, no tabs (SF_TS_ONLY pages)
        st.subheader('Domain mean time series')
        sf_ts_view(run_path,info,key=run_path)
        return
    st.subheader('Scores and summaries')
    lead=os.path.exists(os.path.join(run_path,SF_LEAD_FILE))                               # 9 km runs
    regional=any(os.path.exists(os.path.join(p,SF_REGION_FILE)) for p in preds)
    names=(['Lead month table'] if lead else [])+['Domain mean time series']+(['Scores per region'] if regional else [])
    names+=['Scores per month','Period summary']+(['All figures'] if SHOW_ALL_FIGURES else [])+['Scores table']
    T=dict(zip(names,st.tabs(names)))
    order={'name':[sf_name(p) for p in preds]}

    if 'Lead month table' in T:
        with T['Lead month table']:
            sf_lead_view(run_path,key=run_path)

    with T['Domain mean time series']:
        sf_ts_view(run_path,info,key=run_path)

    if 'Scores per region' in T:
        with T['Scores per region']:
            sf_regions_view(preds,run_path,key=run_path)

    with T['Scores per month']:
        c1,c2=st.columns([3,2])
        opts=sf_metric_opts(df)
        defs=[m for m in (['bias','rmse','kge'] if hkl else ['mean']) if m in opts]
        # options depend on the run (with or without HKL), so the widget key does too
        mets=c1.multiselect('scores',opts,default=defs,format_func=lambda m:SF_METRICS[m][0],key=f'sf_mets_{int(hkl)}')
        srcs=c2.multiselect('compared on',list(SF_SOURCES),default=list(SF_SOURCES),format_func=SF_SOURCES.get,key='sf_srcs')
        cols=st.columns(2)
        for i,m in enumerate(mets):
            parts=[df[['month','name',f'{s}_{m}']].rename(columns={f'{s}_{m}':'v'}).assign(source=SF_SOURCES[s])
                   for s in srcs if f'{s}_{m}' in df]
            if not parts: continue
            long=pd.concat(parts,ignore_index=True)
            f=px.line(long,x='month',y='v',color='name',line_dash='source',markers=True,height=360,title=SF_METRICS[m][0],
                      category_orders=order,labels={'month':'forecast month','name':'prediction','v':SF_METRICS[m][1] or m,'source':''})
            if m=='bias': f.add_hline(y=0,line_color='black',line_width=0.6)
            with cols[i%2]: chart(f,key=f'sf_met_{m}')
        if 'skill_mse_vs_ecsf' in df and df['skill_mse_vs_ecsf'].notna().any():
            sk=df.melt(id_vars=['month','name'],value_vars=[c for c in ['skill_mse_vs_ecsf','frac_cells_improved'] if c in df],var_name='score',value_name='value')
            sk['score']=sk['score'].map({'skill_mse_vs_ecsf':'MSE skill vs raw ECSF','frac_cells_improved':'fraction of 1° cells improved'})
            f=px.line(sk,x='month',y='value',color='name',facet_col='score',markers=True,height=340,category_orders=order,
                      labels={'month':'forecast month','name':'prediction','value':''})
            f.for_each_annotation(lambda a:a.update(text=a.text.split('=')[-1]))
            f.add_hline(y=0,line_color='black',line_width=0.6,col=1); f.add_hline(y=0.5,line_dash='dot',line_color='gray',col=2)
            chart(f,key='sf_skill')
            st.caption(f'Both on the same 1° ECSF cells. MSE skill above 0 and more than half of the cells improved mean the downscaling brings the forecast closer to {REF}.')

    with T['Period summary']:
        summ=[]
        for p in preds:
            d=df[df['pred']==os.path.basename(p)]
            row={'prediction':sf_name(p),'months':len(d)}
            for s,sl in [('ecx','ECXSF'),('ecx1','1° ECXSF'),('ecsf','1° ECSF')]:
                for k,l in [('mean','mean SWE (mm)'),('bias','bias (mm)'),('rmse','RMSE (mm)'),('kge','KGE')]:
                    if f'{s}_{k}' in d and d[f'{s}_{k}'].notna().any(): row[f'{sl} {l}']=d[f'{s}_{k}'].mean()
            for k,l in [('skill_mse_vs_ecsf','MSE skill vs ECSF'),('frac_cells_improved','cells improved')]:
                if k in d: row[l]=d[k].mean()
            summ.append(row)
        st.markdown('**Means over all forecast months**')
        table(pd.DataFrame(summ).round(3),hide_index=True)
        cols=st.columns(2)
        for i,p in enumerate(preds):
            with cols[i%2]:
                st.markdown(f'**{sf_name(p)}**'); img(os.path.join(p,'swe_scores_vs_month.png'))
                if os.path.exists(os.path.join(p,'swe_scores_vs_month_regions.png')): img(os.path.join(p,'swe_scores_vs_month_regions.png'))

    if 'All figures' in T:
        with T['All figures']:
            p=st.selectbox('prediction',preds,format_func=sf_name,key='sf_allfig_pred')
            figs=sorted(glob.glob(os.path.join(p,'*.png')))
            if not figs:
                st.info(f'no figures in {p}')
            else:
                fnames=[os.path.basename(x) for x in figs]
                pick=st.multiselect(f'{len(figs)} figures in {p}',fnames,default=fnames,key='sf_allfig_pick')
                ncol=st.radio('columns',[1,2,3],index=2,horizontal=True,key='sf_allfig_cols')
                gc=st.columns(ncol)
                for i,x in enumerate([x for x in figs if os.path.basename(x) in pick]):
                    with gc[i%ncol]: img(x,os.path.basename(x))

    with T['Scores table']:
        show=df.drop(columns=['pred']).rename(columns={'name':'prediction'})
        table(show.round(4),hide_index=True)
        st.download_button('download scores (csv)',show.to_csv(index=False),file_name=f'swe_metrics_monthly_{sf_rel(run_path).replace(os.sep,"_")}.csv',mime='text/csv')

def page_compare_seasonal():
    st.title('Compare model predictions: seasonal 215 days (ECXSF)')
    allruns=sf_all_runs()
    if not allruns:
        st.warning(f'no seasonal validation results (swe_metrics_monthly.csv) in {SF_DIR} or {SF9_DIR}'); sf_no_runs(); return
    c1,c2=st.columns([2,3])
    view=c1.radio('view',CMP_VIEWS,horizontal=True,key='sfc_view')
    have=[x for x in SF_RES if any(r['res']==x for r in allruns)]
    ropts=have+(['both'] if len(have)>1 else [])
    if st.session_state.get('sfc_res_sel') not in ropts: st.session_state['sfc_res_sel']=ropts[0]
    res=c2.radio('resolution',ropts,horizontal=True,key='sfc_res_sel',
                 format_func=lambda x:'2.8 km and 9 km' if x=='both' else x,
                 help='2.8 km and 9 km: each model edition at each resolution is its own model, '
                      'so the same edition can be compared at the two grids')
    runs=sf_res_runs(allruns,res)
    region,runs=pick_region(runs,'sfc_region_sel')
    if view==CMP_VIEWS[1]:
        sf_detail_any(runs,region,res); return
    st.caption('Raw ECSF and the reference (HKL at 2.8 km, ERA5-Land at 9 km) once, then the ECXSF monthly mean of each model edition, '
               'for the same start date, prediction and month.')
    keys=sorted({sf_date_key(r) for r in runs if not case_hidden(sf_date_key(r))},reverse=True)
    c1,c2,c3=st.columns([2,3,1])
    key0=c1.selectbox('start date',keys,key=f'sfc_date_{res}_{region}',format_func=lambda x:case_label(SF_CASES,region,x))
    key=f'{res}_{region}_{key0}'
    by_model={}
    for r in runs:
        if sf_date_key(r)==key0: by_model.setdefault(r['model'],r)
    models=sf_model_order(by_model)
    pick=c2.multiselect('models',models,default=models,format_func=sfm_name,key=f'sfc_models_{key}')
    if not pick:
        st.info('pick at least one model'); return
    common=None
    for m in pick:
        labs={os.path.basename(p) for p in by_model[m]['preds']}
        common=labs if common is None else common&labs
    if not common:
        st.warning('the picked models have no prediction in common for this start date'); return
    pred=c3.selectbox('prediction',sorted(common,key=sf_sort_key),format_func=sf_short,key=f'sfc_pred_{key}')
    pdir={m:next(p for p in by_model[m]['preds'] if os.path.basename(p)==pred) for m in pick}
    months=sf_months(pdir.values())
    if not months:
        st.info('no monthly maps shared by the picked models'); return
    bt=by_model[pick[0]]['base_time']
    c1,c2=st.columns([4,1])
    mon=c1.select_slider('month',options=months,value=months[0],format_func=lambda m:sf_month_label(m,bt),key=f'sfc_mon_{key}') if len(months)>1 else months[0]
    ref=pdir[pick[0]]   # raw ECSF and the reference are the same in every model folder, taken from the first model
    rks={m:sf_ref(read_json(os.path.join(by_model[m]['run'],'run_info.json'))) for m in pick}   # reference of each model's run
    rk=rks[pick[0]]
    if len(set(rks.values()))>1:
        st.warning('The picked runs have different references ('+', '.join(f'{sfm_name(m)}: {SF_REFS[r]}' for m,r in rks.items())+
                   '), so their scores are not against the same data. The reference map is from the first model.')

    panels=[(os.path.join(ref,f'ecsf_{mon}.png'),'raw ECSF (1°)'),(os.path.join(ref,f'{rk}_{mon}.png'),'HKL (Snowcap)' if rk=='hkl' else SF_REFS[rk])]
    panels+=[(os.path.join(pdir[m],f'ecxsf_{mon}.png'),f'ECXSF {sfm_name(m)}') for m in pick]
    ncol=int(c2.number_input('maps per row',min_value=2,max_value=8,value=min(len(panels),5),step=1,key='sfc_ncol'))
    for i in range(0,len(panels),ncol):
        img_row(panels[i:i+ncol])

    # ---- domain mean SWE over the whole forecast window: ens mean of each model and the reference
    with st.expander('domain mean SWE time series (ECXSF ensemble mean of each model, reference)',expanded=True):
        regsets=[{k:n for k,n,_ in sf_ts_regions(by_model[m]['run'])} for m in pick]
        common=[k for k in regsets[0] if all(k in r for r in regsets)] if regsets and all(regsets) else []
        region=None
        if common:   # 9 km runs: the same region in every run
            region=st.radio('region',common,horizontal=True,key=f'sfc_ts_region_{key}',format_func=regsets[0].get)
        fig=go.Figure(); doms=set(); ref_from=None; rows=[]
        for m in pick:
            run=by_model[m]['run']; d=sf_timeseries(run,region)
            if d is None: continue
            inf=read_json(os.path.join(run,'run_info.json'))
            doms.add(regsets[0][region] if region else inf.get('timeseries_domain','?'))
            if 'ecxsf_ensmean' in d:
                fig.add_scatter(x=d['valid_time'],y=d['ecxsf_ensmean'],mode='lines',name=f'ECXSF {sfm_name(m)} ens mean',hovertemplate='%{y:.1f} mm')
            rc,rn=sf_ts_ref(d)
            if ref_from is None and rc is not None:
                h=d.dropna(subset=[rc]); ref_from=(m,rn)
                fig.add_scatter(x=h['valid_time'],y=h[rc],mode='lines',line=dict(color='#d7191c',width=2.4),name=rn,hovertemplate='%{y:.1f} mm')
            s_=sf_ts_scores(d)
            rows.append({'model':sfm_name(m),'reference':s_.get('ref',''),'members':len(sf_ts_members(d)),
                         'peak ens mean (mm)':s_.get('peak ens mean',(np.nan,))[0],
                         'peak date':f"{s_['peak ens mean'][1]:%Y-%m-%d}" if 'peak ens mean' in s_ else '',
                         'peak reference (mm)':s_.get('peak ref',(np.nan,))[0],
                         'reference peak date':f"{s_['peak ref'][1]:%Y-%m-%d}" if 'peak ref' in s_ else '',
                         'bias (mm)':s_.get('bias'),'RMSE (mm)':s_.get('RMSE'),'r':s_.get('r'),'reference inside members':s_.get('ref inside members')})
        if not fig.data:
            st.info('no domain mean time series in the runs of the picked models')
        else:
            fig.update_layout(height=440,yaxis_title='domain mean SWE (mm)',hovermode='x unified',legend=dict(orientation='h',y=1.08),margin=dict(t=30))
            chart(fig,key=f'sfc_ts_{key}_{region}')
            table(pd.DataFrame(rows).round(3),hide_index=True)
            st.caption('Each series is the mean over that run\'s own domain (' + '; '.join(sorted(doms)) + '). '
                       + (f'{ref_from[1]} is drawn from the {sfm_name(ref_from[0])} run; on another grid its domain mean differs slightly, '
                          'the scores in the table use each run\'s own reference.' if ref_from is not None and len(pick)>1 else ''))

    # ---- lead month tables (9 km runs) of the picked models, side by side
    leads=[(m,sf_lead_table(by_model[m]['run'])) for m in pick]
    leads=[(m,t) for m,t in leads if t is not None]
    if leads:
        with st.expander('lead month table, ens mean against ERA5-Land (9 km runs)',expanded=False):
            t=pd.concat([t.assign(model=sfm_name(m)) for m,t in leads],ignore_index=True)
            t=t[['model']+[c for c in t.columns if c!='model']]
            table(t.round(1),hide_index=True)
            st.caption('Lead month 0 = forecast start month. ME = forecast minus ERA5-Land.')

    with st.expander('bias maps for this month (forecast minus reference)',expanded=False):
        bpan=[(os.path.join(ref,f'bias_ecsf_{rk}_{mon}.png'),f'raw ECSF minus {SF_REFS[rk]} (1°)')]
        bpan+=[(os.path.join(pdir[m],f'bias_ecxsf_{rks[m]}_{mon}.png'),f'ECXSF {sfm_name(m)} minus {SF_REFS[rks[m]]}') for m in pick]
        bpan=[(p,c) for p,c in bpan if os.path.exists(p)]
        for i in range(0,len(bpan),4):
            img_row(bpan[i:i+4])

    parts=[]
    for m in pick:
        d=read_csv(os.path.join(pdir[m],'swe_metrics_monthly.csv'))
        if d is not None: parts.append(d.assign(model=sfm_name(m)))
    if not parts: return
    df=pd.concat(parts,ignore_index=True)
    st.subheader('Scores')
    ym=f'{mon[:4]}-{mon[4:]}'
    cols={'ecx_bias':'bias (mm)','ecx_mae':'MAE (mm)','ecx_rmse':'RMSE (mm)','ecx_r':'r','ecx_kge':'KGE',
          'ecx1_rmse':'1° RMSE ECXSF (mm)','ecsf_rmse':'1° RMSE raw ECSF (mm)','skill_mse_vs_ecsf':'MSE skill vs ECSF'}
    dd=df[df['month']==ym]
    if not dd.empty:
        st.markdown(f'**{sf_month_label(mon,bt)}**')
        table(dd.set_index('model')[[c for c in cols if c in dd]].rename(columns=cols).round(3))
    c1,c2=st.columns(2)
    opts=[m for m in ['rmse','bias','mae','kge','mean'] if m in sf_metric_opts(df)] or ['mean']
    if st.session_state.get('sfc_met') not in opts: st.session_state['sfc_met']=opts[0]
    met=c1.radio('score',opts,horizontal=True,format_func=lambda m:SF_METRICS[m][0],key='sfc_met')
    if not sf_has_hkl(df): c1.caption('no reference for these runs, mean SWE only')
    res=c2.radio('grid',['own grid','1° cells'],horizontal=True,key='sfc_res',
                 help='1° cells: ECXSF and raw ECSF against the reference on the same ECSF cells, so raw ECSF can be drawn too')
    pre='ecx' if res=='own grid' else 'ecx1'
    if f'{pre}_{met}' not in df:
        st.info(f'no {res} scores in these runs'); return
    long=df[['month','model',f'{pre}_{met}']].rename(columns={f'{pre}_{met}':'v'})
    if res!='own grid' and f'ecsf_{met}' in df:
        e=df[df['model']==df['model'].iloc[0]][['month',f'ecsf_{met}']].rename(columns={f'ecsf_{met}':'v'}).assign(model='raw ECSF')
        long=pd.concat([e,long])
    f=px.line(long,x='month',y='v',color='model',markers=True,height=380,title=f'{SF_METRICS[met][0]} per forecast month, {sf_short(pred)} ({res})',
              labels={'month':'forecast month','v':SF_METRICS[met][1] or met,'model':''})
    if met=='bias': f.add_hline(y=0,line_color='black',line_width=0.6)
    f.update_traces(selector=dict(name='raw ECSF'),line=dict(dash='dash',color='gray'))
    chart(f,key='sfc_lead')

    # ---- every start date the picked models have: mean scores model against model
    with st.expander('all start dates, model against model (means over all forecast months of this prediction)',expanded=False):
        rows=[]
        for r in runs:
            if r['model'] not in pick: continue
            p=next((x for x in r['preds'] if os.path.basename(x)==pred),None)
            d=read_csv(os.path.join(p,'swe_metrics_monthly.csv')) if p else None
            if d is None or f'{pre}_{met}' not in d: continue
            rows.append({'start date':sf_date_key(r),'model':sfm_name(r['model']),SF_METRICS[met][0]:d[f'{pre}_{met}'].mean()})
        if rows:
            t=pd.DataFrame(rows)
            table(t.pivot_table(index='start date',columns='model',values=SF_METRICS[met][0]).round(3))
            f=px.bar(t,x='start date',y=SF_METRICS[met][0],color='model',barmode='group',height=360)
            f.update_layout(xaxis_type='category',legend=dict(orientation='h',y=1.1),margin=dict(t=30))
            chart(f,key='sfc_alldates')

# ---------------------------------------------------------------
# Page: latest SWE predictions
# web map with SmartMet WMS layers of the newest runs; click a spot (or pick one) and the ECXENS (10 / 15 days) and
# ECXSF (215 days) member time series of that spot open below the map, with observed SWE of this snow season where set
# point time series: SmartMet timeseries plugin, one request per product, every member as its own FMI key
#   FMI key = param:producer:geometry:leveltype:level:forecasttype:forecastnumber, the member number is the last field
# no origintime is sent, so the server answers from its newest run (add origintime=... to 'extra' to pin a run)
# ---------------------------------------------------------------
LATEST_TIMEOUT=120
LATEST_MAP_HEIGHT=580
LATEST_START=(58.0,15.0,4)   # map start: lat, lon, zoom (Scandinavia and the Alps both in view)
SNOW_MM=1.0                  # mm, a member has snow at or above this
# WMS layers on the map: shown name -> layer in the SmartMet WMS, the product whose valid times it has, and a probe spot inside
# the layer's domain (the run of the layer is read from a timeseries request there when GetCapabilities gives no reference_time).
# The map asks the tiles itself in web mercator, so the BBOX of a test GetMap url does not matter here.
LATEST_LAYERS={
    'ECXENS SWE, Scandinavia (10 / 15 days, 500 m)':dict(layer='gui:isobands:SWE-Scandinavia',product='ecxens',probe=(67.367,26.629)),
    # 'ECXENS SWE, Alps (10 / 15 days, 500 m)':dict(layer='gui:isobands:SWE-Alps',product='ecxens',probe=(46.830,9.809)),   # check the name
}
# forecast products for the point time series: tmpl is the FMI key with {m} for the member number; members are sent one key each.
# scale: factor to mm. ECXENS SD-M comes from the server already in mm SWE (despite the name), so 1
# check forecasttype (the field before {m}) and the member numbers in grid-gui; ECXSF tmpl is empty until its geometry is known
LATEST_PRODUCTS={
    'ecxens':dict(name='ECXENS',title='10 / 15 day ensemble (ECXENS, 500 m)',tmpl='SD-M:ECXENS:5085:1:0:0:{m}',members=list(range(1,51)),
                  scale=1.0,days_back=1,days_ahead=16,extra='',color='#1f77b4'),
    'ecxsf':dict(name='ECXSF',title='seasonal ensemble (ECXSF, 215 days)',tmpl='',members=list(range(0,51)),
                 scale=1.0,days_back=1,days_ahead=216,extra='',color='#9467bd'),
}
# observed SWE of this snow season (from 1 Sep) at the clicked spot; entries with an empty param are left out
LATEST_OBS={
    'HKL (Snowcap)':dict(param='',scale=1.0,extra='',color='#d7191c'),
    'ERA5-Land':dict(param='',scale=1000.0,extra='',color='#ff7f0e'),
}
# go to spots on the map (lat, lon); edit freely
LATEST_SPOTS={'Sodankylä':(67.367,26.629),'Kilpisjärvi':(69.046,20.791),'Abisko':(68.355,18.816),'Finse':(60.594,7.524),
              'Rovaniemi':(66.503,25.729),'Weissfluhjoch':(46.830,9.809),'Col de Porte':(45.295,5.766),'Sonnblick':(47.054,12.958)}
LATEST_SHOW={'lines':'every member','band':'percentile bands','off':'ens mean only'}

def latest_now(): return pd.Timestamp.now(tz='UTC').tz_localize(None)
def latest_season_start(now):
    y=now.year if now.month>=9 else now.year-1
    return pd.Timestamp(y,9,1)
def latest_rgba(hexc,a):
    h=hexc.lstrip('#'); return f'rgba({int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)},{a})'

def latest_query(keys,lat,lon,start,end,extra=''):
    q=dict(latlon=f'{lat:.4f},{lon:.4f}',param=','.join(['utctime']+list(keys)),starttime=f'{start:%Y%m%dT%H%M%S}',
           endtime=f'{end:%Y%m%dT%H%M%S}',timestep='data',format='json',tz='utc',timeformat='sql',precision='double')
    if extra and extra.strip(): q.update(dict(parse_qsl(extra.strip().lstrip('?&'),keep_blank_values=True)))
    return q
def latest_url(q): return f'{SMARTMET_URL}/timeseries?'+urlencode(q,safe=':,')

@st.cache_data(ttl=1800,show_spinner=False)
def latest_fetch(server,q,names):
    # one timeseries request -> utctime + names; columns renamed by position, so it does not matter how the server echoes the FMI keys
    r=requests.get(server+'/timeseries',params=q,timeout=LATEST_TIMEOUT)
    if r.status_code!=200: raise RuntimeError(f'HTTP {r.status_code}: {r.text[:300]}')
    try: js=r.json()
    except ValueError: raise RuntimeError(f'no json in the answer: {r.text[:300]}')
    if isinstance(js,dict): js=js.get('data',js)
    if not isinstance(js,list): raise RuntimeError(f'unexpected answer: {str(js)[:300]}')
    d=pd.DataFrame(js)
    if d.empty: return pd.DataFrame(columns=['utctime']+list(names))
    if len(d.columns)!=len(names)+1:
        raise RuntimeError(f'{len(d.columns)} columns in the answer, expected {len(names)+1}: {list(d.columns)[:4]} ...')
    d.columns=['utctime']+list(names)
    d['utctime']=pd.to_datetime(d['utctime'])
    for c in names:
        if c!='origintime': d[c]=pd.to_numeric(d[c],errors='coerce')
    return d

def latest_product_origin(key):
    # run (origintime, YYYYmmddTHHMMSS) of a product: the run of its first map layer (WMS default reference_time), else None
    L=next((L for L in LATEST_LAYERS.values() if L['product']==key),None)
    return latest_origin(L)[0] if L else None

def latest_product(key,lat,lon,origin=None):
    # member time series of one forecast product at one spot: dict(d, mem, origin, url, err); d None when not set or failed
    # origin: the run to ask (origintime in the request, also the start of the window); None = the server's newest run
    P=LATEST_PRODUCTS[key]
    out=dict(d=None,mem=[],origin=None,url=None,err=None)
    if not P.get('tmpl'): return out
    mem=[f'm{m:02d}' for m in P['members']]
    keys=[P['tmpl'].format(m=m) for m in P['members']]
    if origin:
        start=pd.Timestamp(origin); end=start+pd.Timedelta(days=P['days_ahead'])   # the whole run, from its first step
        extra='&'.join(x for x in (P.get('extra',''),f'origintime={origin}') if x)
    else:
        now=latest_now().normalize(); extra=P.get('extra','')
        start,end=now-pd.Timedelta(days=P['days_back']),now+pd.Timedelta(days=P['days_ahead'])
    q=latest_query(['origintime']+keys,lat,lon,start,end,extra)
    try:
        d=latest_fetch(SMARTMET_URL,q,tuple(['origintime']+mem))
    except Exception:
        q=latest_query(keys,lat,lon,start,end,extra)   # without the origintime column, for servers that do not give it
        try: d=latest_fetch(SMARTMET_URL,q,tuple(mem))
        except Exception as e:
            out.update(url=latest_url(q),err=str(e)); return out
    out['url']=latest_url(q)
    d=d.dropna(subset=mem,how='all').copy()
    if 'origintime' in d and d['origintime'].notna().any():
        try: out['origin']=pd.Timestamp(d['origintime'].dropna().iloc[-1])
        except (ValueError,TypeError): pass
    if not d.empty:
        d[mem]=d[mem]*P['scale']
        d['ensmean']=d[mem].mean(axis=1)
    out.update(d=d,mem=mem)
    return out

def latest_obs(lat,lon):
    # observed SWE since 1 Sep at the spot: {name: data}, request urls, errors
    obs,urls,errs={},{},{}
    now=latest_now().normalize()
    for name,O in LATEST_OBS.items():
        if not O.get('param'): continue
        q=latest_query([O['param']],lat,lon,latest_season_start(now),now+pd.Timedelta(days=1),O.get('extra',''))
        urls[name]=latest_url(q)
        try:
            d=latest_fetch(SMARTMET_URL,q,('v',)).dropna(subset=['v'])
            if not d.empty: obs[name]=d.assign(v=d['v']*O['scale'])
        except Exception as e:
            errs[name]=str(e)
    return obs,urls,errs

def latest_dim_values(text):
    # WMS Dimension text (comma separated times and start/end/period ranges) -> YYYYmmddTHHMMSS strings, oldest first
    out=[]
    for item in (text or '').strip().split(','):
        item=item.strip()
        if '/' in item:   # start/end/period
            a,b,per=(item.split('/')+['PT1H'])[:3]
            out+=list(pd.date_range(pd.Timestamp(a),pd.Timestamp(b),freq=pd.Timedelta(per))[:20000])
        elif item: out.append(pd.Timestamp(item))
    return sorted({f'{x.tz_convert(None) if x.tzinfo else x:%Y%m%dT%H%M%S}' for x in out})

@st.cache_data(show_spinner=False,ttl=1800)
def latest_wms_dims(url,layer):
    # time and reference_time (the runs) of a WMS layer from GetCapabilities: {'time':[...],'reference_time':[...]}, oldest first
    import xml.etree.ElementTree as ET
    r=requests.get(url,params=dict(service='WMS',request='GetCapabilities',version='1.3.0'),timeout=LATEST_TIMEOUT)
    r.raise_for_status()
    root=ET.fromstring(r.content)
    tag=lambda e: e.tag.split('}')[-1]
    for lay in root.iter():
        if tag(lay)!='Layer': continue
        if next((c.text for c in lay if tag(c)=='Name'),None)!=layer: continue
        out={}
        for c in lay:
            if tag(c)=='Dimension' and c.get('name','').lower() in ('time','reference_time','origintime'):
                k='time' if c.get('name').lower()=='time' else 'reference_time'
                out[k]=latest_dim_values(c.text)
                if c.get('default'): out[k+'_default']=latest_dim_values(c.get('default'))[-1:]
        return out
    return {}

@st.cache_data(show_spinner=False,ttl=1800)
def latest_wms_times(url,layer):
    # valid times of a WMS layer from GetCapabilities (Dimension time), as YYYYmmddTHHMMSS, oldest first; [] if not found
    import xml.etree.ElementTree as ET
    r=requests.get(url,params=dict(service='WMS',request='GetCapabilities',version='1.3.0'),timeout=LATEST_TIMEOUT)
    r.raise_for_status()
    root=ET.fromstring(r.content)
    tag=lambda e: e.tag.split('}')[-1]
    for lay in root.iter():
        if tag(lay)!='Layer': continue
        if next((c.text for c in lay if tag(c)=='Name'),None)!=layer: continue
        dim=next((c for c in lay if tag(c)=='Dimension' and c.get('name','').lower()=='time'),None)
        if dim is None or not (dim.text or '').strip(): return []
        out=[]
        for item in dim.text.strip().split(','):
            item=item.strip()
            if '/' in item:   # start/end/period
                a,b,per=(item.split('/')+['PT1H'])[:3]
                out+=list(pd.date_range(pd.Timestamp(a),pd.Timestamp(b),freq=pd.Timedelta(per))[:20000])
            elif item: out.append(pd.Timestamp(item))
        return sorted({f'{x.tz_convert(None) if x.tzinfo else x:%Y%m%dT%H%M%S}' for x in out})
    return []

def latest_times(L):
    # valid times of a map layer: from the WMS, else every day 00 UTC of the product's forecast length from today
    try: ts=latest_wms_dims(WMS_URL,L['layer']).get('time',[]) if WMS_URL else []
    except Exception: ts=[]
    if ts: return ts,'WMS GetCapabilities'
    now=latest_now().normalize()
    return [f'{now+pd.Timedelta(days=i):%Y%m%dT%H%M%S}' for i in range(LATEST_PRODUCTS[L['product']]['days_ahead']+1)],'generated, daily 00 UTC'

@st.cache_data(ttl=1800,show_spinner=False)
def latest_ts_origin(server,tmpl,member,lat,lon):
    # newest run (origintime) of a product from one timeseries request at a spot; None when the server does not give it
    now=latest_now().normalize()
    q=latest_query(['origintime',tmpl.format(m=member)],lat,lon,now-pd.Timedelta(days=1),now+pd.Timedelta(days=2))
    try:
        d=latest_fetch(server,q,('origintime','v')).dropna(subset=['v'])
        return f"{pd.Timestamp(d['origintime'].dropna().iloc[-1]):%Y%m%dT%H%M%S}" if len(d) and d['origintime'].notna().any() else None
    except Exception:
        return None

def latest_origin(L):
    # run of a map layer for the WMS origintime: newest reference_time in GetCapabilities, else the newest run in the timeseries
    # plugin at the layer's probe spot, else None (no origintime sent, the server picks its default)
    # the default reference_time is the run the time dimension belongs to; it is normally also the newest one
    try: dims=latest_wms_dims(WMS_URL,L['layer']) if WMS_URL else {}
    except Exception: dims={}
    if dims.get('reference_time_default'): return dims['reference_time_default'][0],'WMS GetCapabilities, default run'
    if dims.get('reference_time'): return dims['reference_time'][-1],'WMS GetCapabilities, newest run'
    P=LATEST_PRODUCTS[L['product']]
    if SMARTMET_URL and P.get('tmpl') and L.get('probe'):
        o=latest_ts_origin(SMARTMET_URL,P['tmpl'],P['members'][0],*L['probe'])
        if o: return o,'timeseries plugin'
    return None,'server default'

def latest_legend_url(layer):
    return WMS_URL+'?'+urlencode(dict(service='WMS',request='GetLegendGraphic',version='1.3.0',sld_version='1.1.0',style='default',
                                      format='image/png',layer=layer))

def latest_spot_name(lat,lon,km=3.0):
    best,dmin=None,1e9
    for n,(la,lo) in LATEST_SPOTS.items():
        d=111.2*np.hypot(lat-la,(lon-lo)*np.cos(np.deg2rad(lat)))
        if d<dmin: best,dmin=n,d
    return f'{best} ({lat:.3f}N {lon:.3f}E)' if dmin<=km else f'{lat:.4f}N {lon:.4f}E'

def latest_set_loc(lat,lon): st.session_state['latest_loc']=(round(float(lat),4),round(float(lon),4))
def latest_go():
    s=st.session_state.get('latest_spot')
    if s:
        la,lo=LATEST_SPOTS[s]; st.session_state['latest_view']={'center':[la,lo],'zoom':9}; latest_set_loc(la,lo)
def latest_go_coords():
    la,lo=st.session_state['latest_lat_in'],st.session_state['latest_lon_in']
    st.session_state['latest_view']={'center':[la,lo],'zoom':9}; latest_set_loc(la,lo)

def latest_map(layer_name,t,opacity,origin=None):
    ss=st.session_state; view=ss['latest_view']
    m=folium.Map(location=view['center'],zoom_start=view['zoom'],tiles=None,control_scale=True,min_zoom=3,max_zoom=16)
    folium.TileLayer('OpenStreetMap',name='OpenStreetMap').add_to(m)
    folium.TileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                     attr='Esri, Maxar, Earthstar Geographics',name='satellite image').add_to(m)
    if layer_name in LATEST_LAYERS and WMS_URL:
        kw=dict(time=t) if t else {}
        if origin: kw['origintime']=origin
        folium.raster_layers.WmsTileLayer(url=WMS_URL,layers=LATEST_LAYERS[layer_name]['layer'],fmt='image/png',transparent=True,
                                          version='1.3.0',name=layer_name,overlay=True,attr='FMI SmartMet Server',opacity=opacity,**kw).add_to(m)
    folium.LayerControl(collapsed=True).add_to(m)
    fg=folium.FeatureGroup(name='spot')
    lat,lon=ss['latest_loc']
    folium.CircleMarker([lat,lon],radius=9,color='white',weight=2,fill=True,fill_color='#d7191c',fill_opacity=1,
                        tooltip=f'selected: {latest_spot_name(lat,lon)}').add_to(fg)
    kw=dict(key='latest_map',height=LATEST_MAP_HEIGHT,returned_objects=['last_clicked'])
    # dynamic marker and view (streamlit-folium >= 0.15), so the map is not drawn again on every click; plain map on older versions
    for extra in (dict(feature_group_to_add=fg,center=view['center'],zoom=view['zoom'],use_container_width=True),
                  dict(feature_group_to_add=fg,center=view['center'],zoom=view['zoom'],width=1400)):
        try: return st_folium(m,**extra,**kw)
        except TypeError: continue
    fg.add_to(m)
    return st_folium(m,width=1400,**kw)

def latest_fig(r,P,obs,show,height=470,xstart=None,extra=None):
    # members (thin lines or percentile bands), ens mean black dotted, observations, a dotted line at now
    d,mem=r['d'],r['mem']; t=d['utctime']; col=P['color']
    fig=go.Figure()
    if show=='lines':
        for i,c in enumerate(mem):
            fig.add_scatter(x=t,y=d[c],mode='lines',line=dict(color=col,width=0.7),opacity=0.45,hoverinfo='skip',
                            name=f"{P['name']} members ({len(mem)})",legendgroup='members',showlegend=i==0)
    elif show=='band':
        q=d[mem].quantile([0.1,0.25,0.75,0.9],axis=1).T
        for lo,hi,a,lab in [(0.1,0.9,0.18,'10 to 90 %'),(0.25,0.75,0.38,'25 to 75 %')]:
            fig.add_scatter(x=t,y=q[hi],mode='lines',line=dict(width=0),hoverinfo='skip',showlegend=False)
            fig.add_scatter(x=t,y=q[lo],mode='lines',line=dict(width=0),fill='tonexty',fillcolor=latest_rgba(col,a),
                            name=f"{P['name']} members {lab}",hoverinfo='skip')
    if extra is not None:   # e.g. the ECXENS ens mean on top of the seasonal plot
        fig.add_scatter(x=extra[0]['utctime'],y=extra[0]['ensmean'],mode='lines',line=dict(color=extra[1],width=2),
                        name=extra[2],hovertemplate='%{y:.1f} mm')
    for name,o in obs.items():
        fig.add_scatter(x=o['utctime'],y=o['v'],mode='lines',line=dict(color=LATEST_OBS[name]['color'],width=2.2),
                        name=f'{name} (observed)',hovertemplate='%{y:.1f} mm')
    fig.add_scatter(x=t,y=d['ensmean'],mode='lines',line=dict(color='black',dash='dot',width=2.4),name=f"{P['name']} ensemble mean",
                    hovertemplate='%{y:.1f} mm')
    # line at today (00 UTC) as a plain shape: add_vline with a datetime x fails in some plotly versions
    tn=latest_now().normalize().to_pydatetime()
    fig.add_shape(type='line',xref='x',yref='paper',x0=tn,x1=tn,y0=0,y1=1,line=dict(color='gray',dash='dot',width=1))
    fig.update_layout(height=height,yaxis_title='SWE (mm)',hovermode='x unified',legend=dict(orientation='h',y=1.12),margin=dict(t=40),
                      xaxis_range=[xstart,t.max()] if xstart is not None else None)
    return fig

def latest_cards(r,obs,seasonal):
    # key numbers above the plot; dates go in the labels (a metric delta would draw an arrow)
    d,mem=r['d'],r['mem']
    now=d.loc[(d['utctime']-latest_now()).abs().idxmin()]   # time step closest to now
    last=d.iloc[-1]
    q10,q90=np.nanpercentile(last[mem].astype(float),[10,90])
    cards=[(f"ens mean, {now['utctime']:%Y-%m-%d}",fmt(now['ensmean'],'{:.1f} mm')),
           (f"ens mean, {last['utctime']:%Y-%m-%d} (last step)",fmt(last['ensmean'],'{:.1f} mm')),
           ('members 10 to 90 %, last step',f'{q10:.1f} to {q90:.1f} mm'),
           (f'members with snow (≥ {SNOW_MM:g} mm), last step',f"{int((last[mem]>=SNOW_MM).sum())} of {len(mem)}")]
    if seasonal:
        i=d['ensmean'].idxmax()
        cards.insert(2,(f"ens mean peak, {d.at[i,'utctime']:%Y-%m-%d}",fmt(d.at[i,'ensmean'],'{:.1f} mm')))
        after=d.loc[i:]; mo=after[after['ensmean']<SNOW_MM]
        cards.insert(3,('ens mean melt out',f"{mo['utctime'].iloc[0]:%Y-%m-%d}" if len(mo) else 'not in the forecast'))
    for name,o in obs.items():
        cards.append((f"{name}, {o['utctime'].iloc[-1]:%Y-%m-%d}",fmt(o['v'].iloc[-1],'{:.1f} mm')))
        break
    for col,(lab,v) in zip(st.columns(len(cards)),cards):
        col.metric(lab,v)

def latest_product_view(key,r,obs,show,other=None):
    P=LATEST_PRODUCTS[key]
    if not P.get('tmpl'):
        st.info(f"No FMI key for {P['name']} yet: set LATEST_PRODUCTS['{key}']['tmpl'] at the latest predictions section of the app, "
                f"e.g. 'SD-M:{P['name']}:<geometry>:1:0:0:{{m}}'"); return
    if r['err']:
        st.warning(f"{P['name']} request failed: {r['err'][:300]}"); st.code(r['url'],language=None); return
    if r['d'] is None or r['d'].empty:
        st.info(f"no {P['name']} forecast at this spot (outside the forecast domains, or no run on the server yet)")
        st.code(r['url'],language=None); return
    seasonal=key=='ecxsf'
    latest_cards(r,obs,seasonal)
    now=latest_now()
    xstart=latest_season_start(now) if seasonal else now.normalize()-pd.Timedelta(days=21)
    xstart=min(xstart,d0) if (d0:=r['d']['utctime'].min()) is not None else xstart
    chart(latest_fig(r,P,obs,show,xstart=xstart if obs else None,extra=other),key=f'latest_fig_{key}')
    d=r['d']
    st.caption(f"{P['title']} | run {r['origin']:%Y-%m-%d %H} UTC" if r['origin'] is not None else f"{P['title']} | newest run on the server")

def page_latest():
    st.title('Latest SWE predictions')
    if folium is None or st_folium is None:
        st.error('folium and streamlit-folium are missing: pip install folium streamlit-folium'); return
    if not SMARTMET_URL:
        st.warning('Set the SmartMet Server (and the public WMS address) in the sidebar, data paths, or with SWE_SMARTMET_URL and SWE_WMS_URL.')
    ss=st.session_state
    ss.setdefault('latest_view',{'center':list(LATEST_START[:2]),'zoom':int(LATEST_START[2])})
    ss.setdefault('latest_loc',next(iter(LATEST_SPOTS.values())))

    # ---- map controls: go to, layer, valid time, opacity
    c=st.columns([1.6,2.6,3.2,1.3])
    c[0].selectbox('go to',[None]+list(LATEST_SPOTS),key='latest_spot',on_change=latest_go,format_func=lambda x:'' if x is None else x)
    lname=c[1].selectbox('map layer',list(LATEST_LAYERS)+['none'],key='latest_layer')
    t=None; tsrc=''; origin=None; osrc=''
    if lname in LATEST_LAYERS:
        origin,osrc=latest_origin(LATEST_LAYERS[lname])
        times,tsrc=latest_times(LATEST_LAYERS[lname])
        tkey=f'latest_time_{lname}'
        if ss.get(tkey) not in times: ss.pop(tkey,None)   # a new run has other valid times
        now=f'{latest_now():%Y%m%dT%H%M%S}'
        default=next((x for x in times if x>=now),times[-1])
        if len(times)>1:
            t=c[2].select_slider('valid time (UTC)',options=times,value=ss.get(tkey,default),key=tkey,
                                 format_func=lambda x:f'{x[:4]}-{x[4:6]}-{x[6:8]} {x[9:11]}')
        else:
            t=times[0]; c[2].markdown(f'**valid time** {t[:4]}-{t[4:6]}-{t[6:8]} {t[9:11]} UTC')
    opacity=c[3].slider('layer opacity',0.0,1.0,0.7,0.05,key='latest_op')
    with st.expander('map run and coordinates',expanded=False):
        o=st.text_input('map run (origintime, YYYYmmddTHHMMSS), empty = '+(f'newest, {origin}' if origin else 'server default'),'',key='latest_origin_in')
        if o.strip(): origin,osrc=o.strip(),'typed'
        a,b,cc=st.columns([2,2,1])
        lat0,lon0=ss['latest_loc']
        a.number_input('latitude',value=float(lat0),format='%.4f',key='latest_lat_in')
        b.number_input('longitude',value=float(lon0),format='%.4f',key='latest_lon_in')
        cc.button('go',on_click=latest_go_coords,key='latest_go_btn')

    # ---- map with the WMS legend to the right of it
    if lname in LATEST_LAYERS and WMS_URL:
        mc,lc=st.columns([7,1])
        with mc: out=latest_map(lname,t,opacity,origin)
        lc.markdown(f'<div style="height: {LATEST_MAP_HEIGHT}px; display: flex; align-items: center; justify-content: center;">'
                    f'<img src="{latest_legend_url(LATEST_LAYERS[lname]["layer"])}" style="max-height: 100%; width: auto;"></div>',unsafe_allow_html=True)
    else:
        out=latest_map(lname,t,opacity,origin)
    clk=(out or {}).get('last_clicked')
    if clk:
        ll=(float(clk['lat']),float(clk['lng']))
        if ll!=ss.get('latest_last_click'):
            ss['latest_last_click']=ll; latest_set_loc(*ll); st.rerun()
    st.caption('click the map for the forecast time series of that spot | layers (top right): OpenStreetMap or satellite image'
               +(f' | valid times: {tsrc}' if tsrc else '')
               +(f' | run {origin[:4]}-{origin[4:6]}-{origin[6:8]} {origin[9:11]} UTC ({osrc})' if origin else (' | run: server default' if lname in LATEST_LAYERS else '')))
    if lname in LATEST_LAYERS and WMS_URL:
        with st.expander('test GetMap request of this layer (opens in a browser)'):
            q=dict(SERVICE='WMS',REQUEST='GetMap',VERSION='1.3.0',LAYERS=LATEST_LAYERS[lname]['layer'],STYLES='',FORMAT='image/png',
                   TRANSPARENT='true',HEIGHT=900,WIDTH=800,CRS='EPSG:4326',BBOX='54.0,4.0,71.5,32.0')
            if t: q['time']=t
            if origin: q['origintime']=origin
            st.code(WMS_URL+'?'+urlencode(q,safe=':,'),language=None)

    # ---- time series of the selected spot
    st.divider()
    lat,lon=ss['latest_loc']
    c1,c2=st.columns([5,2])
    c1.header(latest_spot_name(lat,lon))
    show=c2.radio('members',list(LATEST_SHOW),horizontal=True,format_func=LATEST_SHOW.get,key='latest_show')
    if not SMARTMET_URL: return
    with st.spinner('fetching the forecasts of this spot'):
        # the run of each product: the run picked for the map when the map layer is of that product, else the product's newest run
        runs={k:(origin if lname in LATEST_LAYERS and LATEST_LAYERS[lname]['product']==k and origin else latest_product_origin(k)) for k in LATEST_PRODUCTS}
        res={k:latest_product(k,lat,lon,runs[k]) for k in LATEST_PRODUCTS}
        obs,ourls,oerrs=latest_obs(lat,lon)
    for name,e in oerrs.items(): st.warning(f'{name} request failed: {e[:300]}')
    if not any(O.get('param') for O in LATEST_OBS.values()):
        st.caption('no observed SWE set yet (LATEST_OBS in the app)')
    T=st.tabs(['10 / 15 days (ECXENS)','Season, 215 days (ECXSF)','Requests and data'])
    with T[0]:
        latest_product_view('ecxens',res['ecxens'],obs,show)
    with T[1]:
        rx=res['ecxens']
        both=rx['d'] is not None and not rx['d'].empty and st.checkbox('ECXENS ensemble mean on top',value=True,key='latest_sf_ecx')
        latest_product_view('ecxsf',res['ecxsf'],obs,show,
                            other=(rx['d'],LATEST_PRODUCTS['ecxens']['color'],'ECXENS ensemble mean') if both else None)
    with T[2]:
        st.markdown('**Requests** (open in a browser to check the server answers)')
        for k,r in res.items():
            if r['url']: st.markdown(f"{LATEST_PRODUCTS[k]['name']}"); st.code(r['url'],language=None)
        for name,u in ourls.items(): st.markdown(name); st.code(u,language=None)
        P=LATEST_PRODUCTS['ecxens']
        if P.get('tmpl'):
            m0,m1=P['members'][0],P['members'][-1]
            q=latest_query([P['tmpl'].format(m=f'{m0}-{m1}')],lat,lon,latest_now().normalize(),latest_now().normalize()+pd.Timedelta(days=2))
            st.markdown('**Member range test**: if this answers with one column per member, the server takes a range in one key '
                        'and the explicit member list above could be shortened')
            st.code(latest_url(q),language=None)
        for k,r in res.items():
            if r['d'] is not None and not r['d'].empty:
                st.download_button(f"download {LATEST_PRODUCTS[k]['name']} members (csv)",r['d'].to_csv(index=False),
                                   file_name=f"{k}_swe_{lat:.4f}N_{lon:.4f}E.csv",mime='text/csv',key=f'latest_dl_{k}')

# ---------------------------------------------------------------
# Navigation: sidebar sections, in this order. (full page name, name in the menu under its section, page, url)
# Streamlit without st.navigation (< 1.36) gets one radio with the full names in the same order.
NAV={'':[('Overview','Overview',page_overview,'overview')],
     'Latest SWE predictions':[('Latest SWE predictions','Map and time series',page_latest,'latest')],
     'Final model':[('Training and model performance','Training and model performance',page_training,'training'),
                    ('Independent validation','Independent validation',page_validation,'validation')],
     'All model versions':[('Training explorer','Training explorer',page_training_explorer,'training-explorer'),
                           ('Validation explorer','Validation explorer',page_validation_explorer,'validation-explorer')],
     'SWE predictions (past)':[('SWE predictions (past): 10 and 15 days (500 m)','10 and 15 days (500 m)',page_forecasts,'prediction-10-15-days'),
                               ('SWE predictions (past): seasonal 215 days (2.8 km)','seasonal 215 days (2.8 km)',page_seasonal_28,'prediction-seasonal-2p8km'),
                               ('SWE predictions (past): seasonal 215 days (9 km)','seasonal 215 days (9 km)',page_seasonal_9,'prediction-seasonal-9km')],
     'Compare model predictions':[('Compare model predictions: 10 and 15 days','10 and 15 days',page_compare,'compare-10-15-days'),
                                  ('Compare model predictions: seasonal','seasonal',page_compare_seasonal,'compare-seasonal')]}
# the menu is drawn by the app itself (navigation hidden), so it sits under the sidebar title and not above it;
# streamlit versions without the hidden position get streamlit's own menu at the top of the sidebar
if hasattr(st,'navigation'):
    _PAGES={sec:[st.Page(fn,title=nav,url_path=url,default=(url=='overview')) for _,nav,fn,url in pages] for sec,pages in NAV.items()}
    try:
        _pg=st.navigation(_PAGES,position='hidden'); _own_menu=True
    except Exception:
        _pg=st.navigation(_PAGES); _own_menu=False
    if _own_menu:
        with _NAV_BOX:
            for _sec,_pages in _PAGES.items():
                if _sec: st.caption(_sec)
                for _p in _pages: st.page_link(_p)
    _pg.run()
else:
    _all=[p for pages in NAV.values() for p in pages]
    _page=_NAV_BOX.radio('page',[p[0] for p in _all])
    next(p[2] for p in _all if p[0]==_page)()

# bottom of the sidebar, on every page
st.sidebar.caption(f'🤖 {AI_NOTICE}')

# footer on every page but the overview: CryoSCOPE contact and funding statement (EU, SERI, MoES)
if SHOW_FOOTER and logo('cryoscope.png'):
    st.divider()
    _ftr=st.columns([3,2])
    _ftr[1].image(logo('cryoscope.png'),width=640)