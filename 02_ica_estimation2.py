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

from mne_bids import (BIDSPath, read_raw_bids)

from mne import events_from_annotations
from mne.preprocessing import ICA
import mne

from mne_faster import (find_bad_epochs)
from mne_bids import (BIDSPath, read_raw_bids)

from autoreject import Ransac

from config import (bids_dir, deriv_dir, task, bv_montage,
                    event_id, preprocess_opts)
from functions import (get_sub_list, adjust_events_photosensor, inspect_epochs)

# Ask for subject IDs to analyze
sub_list = get_sub_list(deriv_dir, allow_all=False)

# Loop through subjects
for sub in sub_list:

    # STEP 1: SUBJECT INFORMATION DEFINITION ###
    # Define the Subject ID and paths
    sub_id = sub.replace('sub-', '')
    deriv_sub_dir = deriv_dir / sub
    fig_sub_dir = deriv_sub_dir / 'figures'
    print(f'Preprocessing task-{task} data for {sub}')

    # STEP 2: LOAD DATA AND ADJUST EVENT MARKERS
    # Load Raw EEG data from derivatives folder
    bids_sub_dir = BIDSPath(subject=sub_id, task=task,
                            datatype='eeg', root=bids_dir)
    read_options = dict(preload=True, misc=['Photosensor'],
                        eog=['VEOG', 'HEOG'])
    raw = read_raw_bids(bids_sub_dir, extra_params=read_options)

    # Update montage and state there is a 'normal' reference
    raw.set_montage(bv_montage)
    raw.set_eeg_reference([])

    # Read events from annotations (this is the events file)
    events, event_id = events_from_annotations(raw, event_id=event_id)

    # Adjust events with photosensor
    print('Adjusting marker onsets with Photosensor signal')
    events, delays, n_adjusted = adjust_events_photosensor(
        raw, events, tmin=-.02, tmax=.05, threshold=.80,
        min_tdiff=.0085, return_diagnostics=True)
    print(f'  {n_adjusted} events were shifted')

    # Remove Photosensor from channels
    raw.drop_channels('Photosensor')

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
        'bads': raw.info['bads'],
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
    raw.filter(l_freq=preprocess_opts['l_freq'],
               h_freq=preprocess_opts['h_freq'],
               skip_by_annotation=['boundary'])
    raw.notch_filter(60)

    # STEP 5: EPOCH DATA
    # Load metadata
    metadata_file = bids_sub_dir.directory / f'{sub}_task-{task}_events.tsv'
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

    # STEP 6: DETECT BAD CHANNELS
    # Detect bad channels using RANSAC
    ransac = Ransac(min_corr=.7, n_jobs=4)
    ransac.fit(epochs.copy().set_eeg_reference('average'))

    # Update epochs and raw with bad channels
    for x in ransac.bad_chs_:
        epochs.info['bads'].append(x)

    # Plot PSD TODO ADD BLOCKING FUNCTION
    epochs.plot_psd(xscale='linear', show=True, n_jobs=4)

    # Any other channels to exclude?
    other_bads = input(
        'Other channels to exclude? '
        'Separate multiple channels with commas. '
        'Leave blank if none. '
        'ENTER CHANNEL NAMES: ')
    for x in other_bads.split(','):
        if x in epochs.info['ch_names']:
            epochs.info['bads'].append(x)
        else:
            print(f'WARNING: Channel {x} is not present. '
                  'Maybe you made a typo?!?!?')

    # Save epochs
    eeg_file = deriv_sub_dir / \
        f'{sub}_task-{task}_ref-FCz_desc-orig_epo.fif.gz'
    epochs.save(eeg_file, overwrite=True)

    # Make a JSON
    json_info = {
        'Description': 'Epochs extracted from raw data for ICA',
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

    # STEP 7: INTERPOLATE BAD CHANNELS AND AVERAGE REFERENCE
    # Interpolate channels
    epochs.interpolate_bads(verbose=True)

    # Re-reference to average and recover FCz
    epochs.add_reference_channels(preprocess_opts['reference_chan'])
    epochs.set_eeg_reference(ref_channels='average', ch_type='eeg')

    # Save epochs
    eeg_file = deriv_sub_dir / \
        f'{sub}_task-{task}_ref-avg_desc-interpbads_epo.fif.gz'
    epochs.save(eeg_file, overwrite=True)

    # Make a JSON
    json_info = {
        'Description': 'Epochs with bads interpolated and re-referenced',
        'reference': 'average',
        'tmin': epochs.tmin,
        'tmax': epochs.tmax,
        'baseline': epochs.baseline,
        'bad_channels': epochs.info['bads'],
        'eeg_file': eeg_file.name
    }
    with open(str(eeg_file).replace('.fif.gz', '.json'), 'w') as outfile:
        json.dump(json_info, outfile, indent=4)

    # STEP 8: REJECT BAD EPOCHS FOR ICA
    # Find bad epochs using FASTER
    bad_epochs = find_bad_epochs(epochs, return_by_metric=True,
                                 thres=preprocess_opts['faster_thresh'])

    # Find VEOG at stim onset
    veog_data = epochs.copy().crop(
        tmin=-.075, tmax=.075).get_data(picks=['VEOG'])
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

    # Inspect the remaining epochs
    epochs.plot(
        n_channels=len(epochs.ch_names), n_epochs=10, events=events,
        event_id=event_id, scalings=dict(eeg=30e-6, eog=200e-6),
        block=False)

    # Find epochs and channels that were confirmed dropped
    dropped_epochs = []
    for i, epo in enumerate(epochs.drop_log):
        if len(epo) > 0:
            dropped_epochs.append(i)

    # Save epochs
    eeg_file = deriv_sub_dir / \
        f'{sub}_task-{task}_ref-avg_desc-forica_epo.fif.gz'
    epochs.save(eeg_file, overwrite=True)

    # Make a JSON
    json_info = {
        'Description': 'Epochs for use with ICA',
        'reference': 'average',
        'tmin': epochs.tmin,
        'tmax': epochs.tmax,
        'baseline': epochs.baseline,
        'bad_epochs': bad_epochs,
        'dropped_epochs': dropped_epochs,
        'eeg_file': eeg_file.name
    }
    with open(str(eeg_file).replace('.fif.gz', '.json'), 'w') as outfile:
        json.dump(json_info, outfile, indent=4)

    # STEP 9: ESTIMATE ICA
    # Estimate ICA
    ica = ICA(method='picard', max_iter=1000,
              random_state=int(sub_id),
              fit_params=dict(ortho=True, extended=True),
              verbose=True)
    ica.fit(epochs)

    # Save ICA
    ica_file = deriv_sub_dir / f'{sub}_task-{task}_ref-avg_ica.fif.gz'
    ica.save(ica_file)

    # Make a JSON
    json_info = {
        'Description': 'ICA components',
        'reference': 'average',
        'ica_method': {
            'max_iter': 1000,
            'method': 'picard',
            'fit_params': dict(ortho=True, extended=True),
            'random_state': int(sub_id)
        },
        'n_components': ica.get_components().shape[1],
        'input_epochs_file': eeg_file.name
    }
    with open(str(ica_file).replace('.fif.gz', '.json'), 'w') as outfile:
        json.dump(json_info, outfile, indent=4)



        