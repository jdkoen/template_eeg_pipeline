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
import json
from random import (random, randrange)

from mne.io import read_raw_brainvision
from mne import events_from_annotations
import mne

from mne_bids import BIDSPath, write_raw_bids

from config import (bids_dir, source_dir, deriv_dir, task,
                    rename_markers, event_id, bad_chans, preprocess_options)

from functions import get_sub_list

# Get subject list to process
sub_list = get_sub_list(source_dir, allow_all=True)

# Add boundary to event_dict
event_id['boundary'] = -99

# BELOW CAN BE MODIFIED

# DATA COLUMNS TO KEEP AND ADD
# List of data columns to drop from behavioral data file(s)
cols_to_keep = ['id', 'stim_list', 'experimenter', 'frameRate',
                'psychopyVersion', 'phase', 'task', 'TrialNumber',
                'image', 'category', 'subcategory', 'repeat', 'presentation',
                'jitter', 'resp', 'rt', 'correct']

# Rename mapper for behavioral data file
cols_to_rename = {
    'frameRate': 'frame_rate',
    'psychopyVersion': 'psychopy_version',
    'TrialNumber': 'trial_number'
}

# List of columns to add to *events.tsv from behavioral data
cols_to_add = ['trial_number', 'category', 'subcategory', 'repeat', 'resp',
               'rt', 'correct']

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

    # STEP 2: PROCESS EEG DATA
    # Define the source data file
    source_vhdr = source_sub_dir / f'{sub_id}_1back.vhdr'

    # Anonymize Dictionary
    anonymize = {
        'daysback': (365 * randrange(100, 110)) +
                    (randrange(-120, 120) + random())
    }

    # Read in raw bv from source and anonymize
    raw = read_raw_brainvision(source_vhdr, misc=['Photosensor'],
                               eog=['VEOG', 'HEOG'], preload=True)
    raw.anonymize(daysback=anonymize['daysback'])

    # Update line frequency to 60 Hz and indicate it is properly referenced
    raw.info['line_freq'] = 60.0
    raw.set_eeg_reference(ref_channels=[])

    # Update event descriptions
    # (does it inplace on raw.annotations.description)
    print('Renaming and extracting needed markers:')
    onset = raw.annotations.onset.copy()
    duration = raw.annotations.duration.copy()
    description = raw.annotations.description.copy()
    for old_name, new_name in rename_markers.items():
        print('  ', old_name, 'to', new_name)
        description[description == old_name] = new_name
    annots_2_keep = np.isin(description, list(event_id.keys()))
    raw._annotations = mne.Annotations(
        onset[annots_2_keep], duration[annots_2_keep],
        description[annots_2_keep])

    # Extract Events and remove annotations
    events, event_id = events_from_annotations(raw, event_id=event_id)
    raw._annotations = mne.Annotations([], [], [])

    # Get bad channels and update
    sub_bad_chans = bad_chans.get(bids_id)
    if sub_bad_chans is not None:
        raw.info['bads'] = sub_bad_chans['channels']

    # Write BIDS Output
    write_raw_bids(raw, bids_path=bids_sub_dir, events_data=events,
                   event_id=event_id, overwrite=True, verbose=False)

    # UPDATE CHANNELS.TSV
    # Load *channels.tsv file
    bids_sub_dir.update(suffix='channels', extension='.tsv')
    chans_data = pd.read_csv(bids_sub_dir.fpath, sep='\t')

    # Add status_description
    chans_data['status_description'] = 'n/a'
    if sub_bad_chans is not None:
        for chan, reason in sub_bad_chans.items():
            chans_data.loc[chans_data['name'] == chan,
                           ['status_description']] = reason

    # Add EEG Reference
    chans_data['reference'] = preprocess_options['reference_chan']

    # Remove online reference from auxillary channels
    for chan in ['VEOG', 'HEOG', 'Photosensor']:
        chans_data.loc[chans_data['name'] == chan, ['reference']] = 'n/a'

    # Overwrite file
    chans_data.to_csv(bids_sub_dir.fpath, sep='\t', index=False)

    # STEP 3: PROCESS BEHAVIORAL DATA FILE
    # Read in the *beh.tsv behavioral file
    beh_source_file = source_sub_dir / f'{sub_id}_beh.tsv'
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
        if row['trial_type'] != 'boundary':
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

    # Save EEG JSON
    with open(bids_sub_dir.fpath, 'w') as file:
        json.dump(eeg_json, file)

    # STEP 6: MAKE COPY IN DERIVATIVES
    # Write Raw instance
    raw_out_file = deriv_sub_dir / \
        f'sub-{bids_id}_task-{task}_ref-FCz_desc-import_raw.fif.gz'
    raw.save(raw_out_file, overwrite=True)

    # Make a JSON
    json_info = {
        'Description': 'Import from BrainVision Recorder',
        'sfreq': raw.info['sfreq'],
        'reference': 'FCz'
    }
    json_file = deriv_sub_dir / \
        f'sub-{bids_id}_task-{task}_ref-FCz_desc-import_raw.json'
    with open(json_file, 'w') as outfile:
        json.dump(json_info, outfile, indent=4)

    # Write events
    events_out_file = deriv_sub_dir / \
        f'sub-{bids_id}_task-{task}_desc-import_eve.txt'
    mne.write_events(events_out_file, events)

    # Make a JSON
    json_info = {
        'Description': 'Events from Brain Vision Import',
        'columns': ['onset', 'duration', 'code'],
        'onset_units': 'samples',
        'sfreq': raw.info['sfreq'],
        'codes': event_id
    }
    json_file = deriv_sub_dir / \
        f'sub-{bids_id}_task-{task}_desc-import_eve.json'
    with open(json_file, 'w') as outfile:
        json.dump(json_info, outfile, indent=4)

    # Write a copy of the behavioral data file to derivatives
    beh_save_file = deriv_sub_dir / \
        f'sub-{bids_id}_task-{task}_beh.tsv'
    beh_data.to_csv(beh_save_file, sep='\t', index=False)
