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

import numpy as np
import pandas as pd

import json

from mne.io import read_raw_brainvision
from mne import events_from_annotations
from mne.preprocessing import ICA
import mne

from mne_faster import (find_bad_channels, find_bad_epochs)

from config import (bids_dir, deriv_dir, task, bv_montage, bv_event_ids,
                    rename_markers, preprocess_opts, bad_chans)
from functions import (get_sub_list, adjust_events_photosensor, inspect_epochs)

# Ask for subject IDs to analyze
sub_list = get_sub_list(deriv_dir, allow_all=False)

# Loop through subjects
for sub in sub_list:

    # STEP 1: SUBJECT INFORMATION DEFINITION ###
    # Define the Subject ID and paths
    sub_id = sub.replace('sub-', '')
    bids_sub_dir = bids_dir / sub / 'eeg'
    deriv_sub_dir = deriv_dir / sub
    fig_sub_dir = deriv_sub_dir / 'figures'
    print(f'Preprocessing task-{task} data for {sub}')

    # STEP 2: LOAD DATA AND ADJUST EVENT MARKERS
    # Load Raw EEG data from derivatives folder
    bids_vhdr = bids_sub_dir / f'sub-{sub_id}_task-{task}_eeg.vhdr'
    raw = read_raw_brainvision(bids_vhdr, preload=True, misc=['Photosensor'],
                               eog=['VEOG', 'HEOG'])

    # Update montage
    raw.set_montage(bv_montage)

    # Get bad channels and update
    sub_bad_chans = bad_chans.get(sub_id)
    if sub_bad_chans is not None:
        raw.info['bads'] = sub_bad_chans['channels']

    # Read events from annotations, and make appropriate event_id
    events_orig, _ = events_from_annotations(raw, event_id=bv_event_ids)
    event_id = {}
    for key, value in bv_event_ids.items():
        if key in rename_markers.keys():
            print(f'Updating BV marker {key} to {rename_markers[key]}')
            event_id[rename_markers[key]] = value

    # Adjust events with photosensor
    print('Adjusting marker onsets with Photosensor signal')
    events, delays, n_adjusted = adjust_events_photosensor(
        raw, events_orig.copy(), tmin=-.02, tmax=.05, threshold=.80,
        min_tdiff=.0085, return_diagnostics=True)
    print(f'  {n_adjusted} events were shifted')

    # Remove Photosensor from channels
    raw.drop_channels('Photosensor')

    # Save EEG data with json
    eeg_file = deriv_sub_dir / \
        f'{sub}_task-{task}_ref-FCz_desc-markadj_raw.fif.gz'
    raw.save(eeg_file, overwrite=True)
    json_info = {
        'Description': 'EEG data with adjusted markers',
        'adjust_photosensor_settings': {
            'tmin': -.02,
            'tmax': .05,
            'min_tdiff': .0085,
            'threshold': .80,
            'return_diagnostics': True
        },
        'eeg_file': eeg_file.name
    }
    with open(str(eeg_file).replace('.fif.gz', '.json'), 'w') as outfile:
        json.dump(json_info, outfile, indent=4)

    # Save events
    events_file = deriv_sub_dir / f'{sub}_task-{task}_desc-markadj_eve.txt'
    mne.write_events(events_file, events)
    json_info = {
        'Description': 'Photosensor adjusted EEG markers',
        'columns': ['onset', 'duration', 'code'],
        'onset_units': 'samples',
        'codes': event_id,
        'num_events_adjusted': n_adjusted,
        'delays': list(delays)
    }
    with open(str(eeg_file).replace('.txt', '.json'), 'w') as outfile:
        json.dump(json_info, outfile, indent=4)

    # STEP 3: RESAMPLE DATA AND EVENTS
    # Resample events and raw EEG data
    # This causes jitter, but it will be minimal
    raw, events = raw.resample(preprocess_opts['resample'], events=events)

    # Save EEG data with json
    eeg_file = deriv_sub_dir / \
        f'{sub}_task-{task}_ref-FCz_desc-resamp_raw.fif.gz'
    raw.save(eeg_file, overwrite=True)
    json_info = {
        'Description': 'EEG data with adjusted markers',
        'resample_rate': raw.info['sfreq'],
        'eeg_file': eeg_file.name
    }
    with open(str(eeg_file).replace('.fif.gz', '.json'), 'w') as outfile:
        json.dump(json_info, outfile, indent=4)

    # Save events
    events_file = deriv_sub_dir / f'{sub}_task-{task}_desc-resamp_eve.txt'
    mne.write_events(events_file, events)
    json_info = {
        'Description': 'Resampled marker events',
        'columns': ['onset', 'duration', 'code'],
        'onset_units': 'samples',
        'codes': event_id,
        'resample_rate': raw.info['sfreq']
    }
    with open(str(eeg_file).replace('.txt', '.json'), 'w') as outfile:
        json.dump(json_info, outfile, indent=4)

    # STEP 4: HIGH-PASS FILTER DATA
    # High-pass filter data from all channels
    raw.filter(
        l_freq=preprocess_opts['l_freq'], h_freq=preprocess_opts['h_freq'],
        skip_by_annotation=['boundary'])

    # Low-pass and notch filter eog channels only
    raw.filter(
        l_freq=None, h_freq=40, picks=['eog'], skip_by_annotation=['boundary'])
    raw.notch_filter(60, picks=['eog'])

    # Save EEG data with json
    eeg_file = deriv_sub_dir / \
        f'{sub}_task-{task}_ref-FCz_desc-hpfilt_raw.fif.gz'
    raw.save(eeg_file, overwrite=True)
    json_info = {
        'Description': 'filtered continuous EEG data',
        'filter_options': {
            'l_freq': preprocess_opts['l_freq'],
            'h_freq': preprocess_opts['h_freq'],
            'skip_by_annotation': ['boundary']
        },
        'notch_filter_options': {
            'freqs': 60,
            'picks': ['eog']
        },
        'eeg_file': eeg_file.name
    }
    with open(str(eeg_file).replace('.fif.gz', '.json'), 'w') as outfile:
        json.dump(json_info, outfile, indent=4)

    # STEP 5: EPOCH DATA
    # Load metadata
    metadata_file = bids_sub_dir / f'{sub}_task-{task}_events.tsv'
    metadata = pd.read_csv(metadata_file, sep='\t')
    metadata = metadata[metadata['trial_type'].isin(event_id.keys())]

    # Make the epochs from extract event data
    epochs = mne.Epochs(raw, events, event_id=event_id,
                        tmin=preprocess_opts['tmin'],
                        tmax=preprocess_opts['tmax'],
                        baseline=None,
                        metadata=metadata,
                        reject=None,
                        preload=True)

    # Save metadata
    metadata_file = deriv_sub_dir / f'{sub}_task-{task}_desc-orig_metadata.tsv'
    epochs.metadata.to_csv(metadata_file, sep='\t', index=False)

    # Save epochs
    eeg_file = deriv_sub_dir / \
        f'{sub}_task-{task}_ref-FCz_desc-orig_epo.fif.gz'
    epochs.save(eeg_file, overwrite=True)

    # Make a JSON
    json_info = {
        'Description': 'Epochs extracted from raw data',
        'reference': 'FCz',
        'tmin': epochs.tmin,
        'tmax': epochs.tmax,
        'baseline': epochs.baseline,
        'bad_channels': epochs.info['bads'],
        'metadata_file': metadata_file.name,
        'eeg_file': eeg_file.name
    }
    with open(str(eeg_file).replace('.fif.gz', '.json'), 'w') as outfile:
        json.dump(json_info, outfile, indent=4)

    # STEP 6: DETECT BAD CHANNELS AND DATA FOR ICA
    # Make epochs with baseline
    epochs.apply_baseline((-.2, 0))

    # Detect bad channels using FASTER (no kurtosis)
    bads = find_bad_channels(epochs, thres=4, return_by_metric=True)
    for m in bads.values():
        for x in m:
            if x not in epochs.info['bads']:
                epochs.info['bads'].append(x)

    # Plot PSD
    epochs.plot_psd(picks=['eeg'], xscale='linear',
                    show=True, n_jobs=4)

    # Find bad epochs using FASTER
    bad_epochs = find_bad_epochs(epochs, thres=4, return_by_metric=True)

    # Find extreme voltages
    epoch_data = epochs.get_data(picks=['eeg'])
    ext_voltage = np.sum(
        np.abs(epoch_data).max(axis=2) > preprocess_opts['ext_voltage'],
        axis=1)
    bad_epochs['ext_voltage'] = list(np.where(ext_voltage)[0])

    # Find VEOG at stim onset
    veog_data = epochs.copy().crop(tmin=-.1, tmax=.1).get_data(picks=['VEOG'])
    veog_p2p = np.ptp(
        np.squeeze(veog_data), axis=1) > preprocess_opts['blink_thresh']
    bad_epochs['blink_onset'] = list(np.where(veog_p2p)[0])
    for key, value in bad_epochs.items():
        bad_epochs[key] = [int(x) for x in value]

    # Gather all unique bad epochs
    bad_epoch_ids = []
    for key, value in bad_epochs.items():
        for v in value:
            if v not in bad_epoch_ids:
                bad_epoch_ids.append(v)
    bad_epoch_ids = sorted(bad_epoch_ids)

    # Drop the epochs marked as bad
    epochs.drop(bad_epoch_ids)

    # Inspect the epochs
    epochs = inspect_epochs(
        epochs, bad_epochs=None, events=events, event_id=event_id,
        scalings=dict(eeg=30e-6, eog=200e-6), block=True)

    # Find epochs and channels that were confirmed dropped
    dropped_epochs = []
    for i, epo in enumerate(epochs.drop_log):
        if len(epo) > 0:
            dropped_epochs.append(i)
    bad_channels = epochs.info['bads']

    # Load old epochs
    epochs = mne.read_epochs(
        deriv_sub_dir / f'{sub}_task-{task}_ref-FCz_desc-orig_epo.fif.gz')
    epochs.drop(dropped_epochs)
    epochs.info['bads'] = bad_channels

    # Save epochs
    eeg_file = deriv_sub_dir / \
        f'{sub}_task-{task}_ref-FCz_desc-ica_epo.fif.gz'
    epochs.save(eeg_file, overwrite=True)

    # Make a JSON
    json_info = {
        'Description': 'Epochs to submit to ICA',
        'reference': 'FCz',
        'tmin': epochs.tmin,
        'tmax': epochs.tmax,
        'baseline': epochs.baseline,
        'bad_channels': epochs.info['bads'],
        'bad_epochs': bad_epochs,
        'dropped_epochs': dropped_epochs,
        'eeg_file': eeg_file.name
    }
    with open(str(eeg_file).replace('.fif.gz', '.json'), 'w') as outfile:
        json.dump(json_info, outfile, indent=4)

    # STEP 7: ESTIMATE ICA
    # Estimate ICA
    ica = ICA(method='picard', max_iter=1000,
              random_state=int(sub_id),
              fit_params=dict(ortho=True, extended=True),
              verbose=True)
    ica.fit(epochs)

    # Save ICA
    ica_file = deriv_sub_dir / f'{sub}_task-{task}_ref-FCz_desc-ica_ica.fif.gz'
    ica.save(ica_file)

    # Make a JSON
    json_info = {
        'Description': 'ICA components',
        'reference': 'FCz',
        'ica_method': {
            'max_iter': 1000,
            'method': 'picard',
            'fit_params': dict(ortho=True, extended=True),
            'random_state': int(sub_id)
        },
        'n_components': len(ica.info['chs']),
        'input_epochs_file': eeg_file.name
    }
    with open(str(ica_file).replace('.fif.gz', '.json'), 'w') as outfile:
        json.dump(json_info, outfile, indent=4)
