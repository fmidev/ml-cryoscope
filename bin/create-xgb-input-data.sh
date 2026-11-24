#!/usr/bin/env bash
# Snow Water Equivalent (SWE) and Snow Depth (SD)
# training / validation dataset preparation script
# script assumes you have selected training/validation locations and have already fetched the SNOWCAP SWE and SD data

cd /home/ubuntu/data/cryoscope/snowmodel/

training_lonlats_file='snowcap_scandi_alps_training_lonlats.txt'
validation_lonlats_file='snowcap_scandi_alps_validation_lonlats.txt'

snowcap_input_training='training_inputs/HKL_swe_sde_scandi_alps_training.csv'
snowcap_input_validation='validation_inputs/HKL_swe_sde_scandi_alps_validation.csv'

# fetch static data 

# fetch ERA5 Land data (era5l_features.yml)

# fetch ERA5 data

# merge datasets






