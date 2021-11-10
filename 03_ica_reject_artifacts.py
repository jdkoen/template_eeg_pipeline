"""
Script: 02_ICA_estimation.py
Author: Joshua D. Koen
Description: This script handles initial data preprocessing through
ICA estimation. The following is done:

1) Subject information and directories defined
2) Raw data loaded, events extracted and adjusted via photosensor
3) Raw data and events resampled
4) Raw data high pass filtered
5) Epochs created from Raw data
6) Bad channels and epochs identified identified with:
    * MNE-FASTER (using threshold = 4)
    * EOG Blinks at stim onset (peak-to-peak)
    * Extreme absolute voltages
    * Visual inspection
7) Run ICA on data using 'Piccard' algorithm
"""

# Import Libraries
import os
os.chdir(os.path.split(__file__)[0])

import json

from mne import read_epochs
from mne.preprocessing import read_ica

from mne_faster import (find_bad_components)

from config import (deriv_dir, task, preprocess_opts)
from functions import get_sub_list

# Ask for subject IDs to analyze
sub_list = get_sub_list(deriv_dir, allow_all=False)

# Loop through subjects
for sub in sub_list:

    # STEP 1: SUBJECT INFORMATION DEFINITION ###
    # Define the Subject ID and paths
    sub_id = sub.replace('sub-', '')
    deriv_sub_dir = deriv_dir / sub
    fig_sub_dir = deriv_sub_dir / 'figures'
    print(f'Loading ICA task-{task} data for {sub}')

    # STEP 2: LOAD EPOCHS AND ICA OBJECTS
    # Load ICA Epochs
    epochs = read_epochs(
        deriv_sub_dir / f'{sub}_task-{task}_ref-avg_desc-forica_epo.fif.gz')
    epochs.apply_baseline(preprocess_opts['baseline'])

    # Load ICA data and json_info
    ica_file = deriv_sub_dir / f'{sub}_task-{task}_ref-avg_ica.fif.gz'
    ica = read_ica(ica_file)
    ica.exclude = []
    json_file = deriv_sub_dir / f'{sub}_task-{task}_ref-avg_ica.json'
    with open(json_file, 'r') as f:
        json_info = json.load(f)

    # STEP 3: AUTO DETECT BAD ICS
    # Run find_bad_components
    bad_ics = find_bad_components(
        ica, epochs, thres=preprocess_opts['faster_thresh'],
        return_by_metric=True, max_iter=2)

    # Gather all unique bad epochs
    for key, value in bad_ics.items():
        for v in value:
            if v not in ica.exclude:
                ica.exclude.append(v)
    ica.exclude.sort()

    # Plot ICA
    ica.plot_components(inst=epochs, reject=None)
    ica.exclude.sort()
    ica.save(ica_file)
    print(f'ICs Flagged for Removal: {ica.exclude}')

    # Update ICA json sidecar file and resave file
    json_info['FASTER_flagged_ics'] = bad_ics
    json_info['flagged_components'] = [int(x) for x in ica.exclude]
    json_info['proportion_components_flagged'] = \
        len(ica.exclude) / len(ica.info['ch_names'])
    with open(json_file, 'w') as f:
        json.dump(json_info, f, indent=4)
