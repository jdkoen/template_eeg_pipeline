"""
Script: config.py
Creator: Joshua D. Koen
Description: This is the main variables and parameters to define
for preprocessing EEG data. 
"""

# Import libraries
from pathlib import Path
import platform
from mne.channels import read_custom_montage

import os
os.chdir(os.path.split(__file__)[0])
my_os = platform.system()

# STEP 1: TASK AND SUBJECT SPECIFIC VARIABLES
# Project ID and Experiment number
project_id = 'nd012'
experiment_id = 'exp1'

# Define task name
task = '1back'

# Bad Subjects not included in group analysis
bad_subs = []

# List of Known Bad Channels
# This is a dictionary of dictionaries. The top dictionary has keys
# for the subject ID, and the value is another dictionary. This other
# dictionary should have two fields: channels and reason. Both should be
# a list with the channel name that is bad, and the reson why.
bad_chans = {
    '999': {
        'channels': ['TP10'],
        'reason': ['excessive line noise']
        }
}

# STEP 2: Define Preprocessing Options
# Dictionary of preprocessing options
preprocess_options = {
    'reference_chan': 'FCz',
    'blink_thresh': 150e-6,
    'ext_val_thresh': 100e-6,
    'perc_good_chans': .10,
    'resample': 250,
    'highpass': .1,
    'tmin': -1.0,
    'tmax': 1.0,
    'baseline': (-.2, 0),
    'evoked_tmin': -.2,
    'evoked_tmax': .6,
    'evoked_lowpass': 20.0,
    'ica_highpass': 1,
    'ica_baseline': (None, None)
}

# BVEF File and Montage
bv_montage = read_custom_montage('old_64ch.bvef', head_size=.08)

# Rename mapper for BV Stimulus
rename_markers = {
    'New Segment/': 'boundary',
    'Stimulus/S 11': 'scene/novel',
    'Stimulus/S 12': 'scene/1back',
    'Stimulus/S 21': 'object/novel',
    'Stimulus/S 22': 'object/1back',
    'Stimulus/S 31': 'face/novel',
    'Stimulus/S 32': 'face/1back'
}

# Define event dictionary
event_id = {
    'scene/novel': 11,
    'scene/1back': 12,
    'object/novel': 21,
    'object/1back': 22,
    'face/novel': 31,
    'face/1back': 32,
}

# STEP 3: DEFINE THE SERVER AND DATA DIRECTORIES
# This is platform dependent and returns a Path class object
# Get the server directory
# UNCOMMENT THIS FOR A REAL PROJECT
# if my_os == 'Darwin':
#     server_dir = Path('/Volumes/koendata/EXPT')
# elif my_os == 'Linux':
#     server_dir = Path('/koenlab/koendata/EXPT')
# else:
#     server_dir = Path('X:\EXPT')
# data_dir = server_dir / project_id / experiment_id / 'data'

# DELETE THESE TWO LINES FOR A REAL PROJECT
data_dir = Path('data')

# STEP 4: DEFINE PATHS (DO NOT CHANGE AFTER THIS)
# This is the source_data directory
source_dir = data_dir / 'sourcedata'

# This is the bids formatted output directory
bids_dir = data_dir / 'bids'
bids_dir.mkdir(parents=True, exist_ok=True)

# Derivatives directory
deriv_dir = data_dir / 'derivatives' / f'task-{task}'
deriv_dir.mkdir(parents=True, exist_ok=True)

# Report directory
report_dir = deriv_dir / 'reports'
report_dir.mkdir(parents=True, exist_ok=True)

# Analysis Directory
analysis_dir = data_dir / 'analyses' / f'task-{task}'
analysis_dir.mkdir(parents=True, exist_ok=True)
