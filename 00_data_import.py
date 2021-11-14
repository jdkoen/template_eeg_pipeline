"""
Script: 00_data_import.py
Creator: Joshua D. Koen
Description: This script imports data from sourcedata to bids format.
"""

# Import Libraries
import os
os.chdir(os.path.split(__file__)[0])

import numpy as np
import pandas as pd
import shutil
import json
from random import (randrange, seed)

from mne.io import read_raw_brainvision
from mne import events_from_annotations
import mne

from mne_bids import (BIDSPath, write_raw_bids, mark_bad_channels)

from config import (bids_dir, source_dir, deriv_dir, task,
                    cols_to_keep, cols_to_add, cols_to_rename,
                    rename_events, unchanged_markers, bad_chans,
                    preprocess_opts)
from functions import get_sub_list

# Get subject list to process
sub_list = get_sub_list(source_dir, allow_all=True)
study_seed = int(input('Enter digits for study/project id: '))

# Get Subject List
for sub in sub_list:

    # STEP 1: SUBJECT INFORMATION DEFINITION
    # Define the Subject and Source Path
    sub_id = sub.replace('sub-', '')
    source_sub_dir = source_dir / sub

    # Handle Bids Path and ID for EEG data
    bids_id = sub[-3:]
    bids_sub_dir = BIDSPath(subject=bids_id, task=task,
                            datatype='eeg', root=bids_dir)

    # Derivative Paths
    deriv_sub_dir = deriv_dir / f'sub-{bids_id}'
    deriv_sub_dir.mkdir(parents=True, exist_ok=True)

    # Print Info to Screen
    print(f'Making BIDS data for sub-{bids_id} ({sub_id}) for task-{task}')
    print(f'  Source Path: {source_sub_dir}')
    print(f'  BIDS Path: {bids_sub_dir.directory}')
    print(f'  Derivative Path: {deriv_sub_dir}')

    # Ask for info to specify subject_info
    age = int(input('Enter age: '))
    sex = int(input('Enter sex/gender (0=unknown, 1=male, 2=female): '))
    hand = int(input('Enter handedness (1=right, 2=left, 3=ambidextrous): '))

    # STEP 2: BIDS-FY EEG DATA
    # Define the source data file
    source_vhdr = source_sub_dir / f'{sub_id}_1back.vhdr'

    # Read in raw bv from source and anonymize
    raw = read_raw_brainvision(
        source_vhdr, misc=[preprocess_opts['photosensor_chan']],
        eog=['VEOG', 'HEOG'])

    # Update line frequency to 60 Hz and indicate it is properly referenced
    raw.info['line_freq'] = 60.0

    # Anonymize
    seed(a=study_seed)
    raw.anonymize(
        daysback=(365 * randrange(100, 110)) + (randrange(-120, 120)))

    # Update subject_info
    bdate = raw.info['meas_date'].date()
    bdate = bdate.replace(year=bdate.year-age)
    subject_info = {
        'id': int(bids_id),
        'his_id': f'sub-{bids_id}',
        'birthday': (bdate.year, bdate.month, bdate.day),
        'sex': sex,
        'hand': hand,
        'last_name': 'mne_anonymize',
        'first_name': 'mne_anonymize',
        'middle_name': 'mne_anonymize',
    }
    raw.info['subject_info'] = subject_info

    # Extract Events and remove annotations
    events, event_id = events_from_annotations(raw)

    # Write BIDS Output
    if bids_sub_dir.directory.is_dir():
        shutil.rmtree(bids_sub_dir.directory)
    bids_sub_dir = write_raw_bids(
        raw, bids_path=bids_sub_dir,
        overwrite=True, verbose=False)

    # UPDATE CHANNELS.TSV
    # Get bad channels and update
    sub_bad_chans = bad_chans.get(bids_id)
    if sub_bad_chans is not None:
        print(f'{bids_id} has bad channels.')
        mark_bad_channels(
            sub_bad_chans['channels'], sub_bad_chans['reason'],
            bids_path=bids_sub_dir)

    # Load *channels.tsv file
    bids_sub_dir.update(suffix='channels', extension='.tsv')
    chans_data = pd.read_csv(bids_sub_dir.fpath, sep='\t')

    # Add EEG Reference
    chans_data['reference'] = 'FCz'

    # Remove online reference from auxillary channels
    for chan in ['VEOG', 'HEOG', 'Photosensor']:
        chans_data.loc[chans_data['name'] == chan, ['reference']] = 'n/a'

    # Overwrite file
    chans_data.to_csv(bids_sub_dir.fpath, sep='\t', index=False)

    # STEP 3: PROCESS BEHAVIORAL DATA FILE
    # Read in the *beh.tsv behavioral file
    beh_source_file = source_sub_dir / f'{sub_id}_1back_beh.tsv'
    beh_data = pd.read_csv(beh_source_file, sep='\t')[cols_to_keep]
    beh_data.rename(columns=cols_to_rename, inplace=True)

    # Replace NaN and -99 with 'n/a' for resp and rt, respectively
    beh_data['resp'].fillna('n/a', inplace=True)
    beh_data['rt'].replace(-99.0, 'n/a', inplace=True)

    # Replace subject id and select needed data columns
    beh_data['id'] = bids_id

    # Fil in some more values
    beh_data.replace(['None', '', '--'], 'n/a', inplace=True)

    # Save behavioral data
    bids_sub_dir.update(datatype='beh')
    bids_sub_dir.directory.mkdir(parents=True, exist_ok=True)
    beh_save_file = bids_sub_dir.directory / \
        f'sub-{bids_id}_task-{task}_beh.tsv'
    beh_data.to_csv(beh_save_file, sep='\t', index=False)

    # STEP 4: UPDATE *_EVENTS.TSV WITH BEHAVIORAL DATA
    # Load *events.tsv
    bids_sub_dir.update(datatype='eeg', suffix='events')
    events_data = pd.read_csv(bids_sub_dir.fpath, sep='\t')

    # Add new columnas as "n/a" values
    events_data[cols_to_add] = 'n/a'

    # Update with values
    counter = 0  # Keep track of current row in beh_data
    for index, row in events_data.iterrows():
        trial_type = rename_events[row['trial_type']]
        events_data.at[index, 'trial_type'] = trial_type
        if trial_type not in unchanged_markers:
            this_trial = beh_data.iloc[counter]
            for col in cols_to_add:
                events_data.at[index, col] = this_trial[col]
            counter += 1

    # Overwrite *events.tsv
    events_data.to_csv(bids_sub_dir.fpath, sep='\t', index=False)

    # STEP 5: UPDATE *eeg_json
    # Load JSON
    bids_sub_dir.update(suffix='eeg', extension='json')
    with open(bids_sub_dir.fpath, 'r') as file:
        eeg_json = json.load(file)

    # Update keys
    eeg_json['EEGReference'] = 'FCz'
    eeg_json['EEGGround'] = 'Fpz'
    eeg_json['EEGPlacementScheme'] = \
        [x for x in raw.ch_names if x not in
            ['VEOG', 'HEOG', 'Photosensor']]

    # Save EEG JSON
    with open(bids_sub_dir.fpath, 'w') as file:
        json.dump(eeg_json, file)
