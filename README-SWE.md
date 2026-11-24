# ML for Snow Water Equivalent (SWE) and Snow Depth from SNOWCAP data

This document describes the workflow and requirements for combining Snowcap SWE and snow depth as targets with meteorological and satellite data to forecast snow conditions with XGBoost​. 

## Target: SNOWCAP SWE and snow depth

|target in ML|units|spatial coverage|spatial resolution|temporal coverage|xgb name|
|:-|:-|:-|:-|:-|:-|
|Snow Water Equivalent|mm|Fenno-Scandinavia / Alps|500m|2015-2023 / 2016-2024|swe|
|Snow depth|m|Fenno-Scandinavia / Alps|500m|2015-2023 / 2016-2024|sde|

https://snowcap.info/ 