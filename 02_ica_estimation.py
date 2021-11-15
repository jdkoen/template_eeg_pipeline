"""
Script: 02_ICA_estimation.py
Author: Joshua D. Koen
Description: This script handles initial data preprocessing through
ICA estimation. The following is done:

1) Subject information and directories defined
2) Raw data loaded
3) Events adjusted with photosensor (optional)
4) Raw data and events resampled and notch filter (notch is optional)
5) Raw data high pass filtered at 1.0Hz (1.0 Hz transition bandwidth)
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

from mne_bids import (BIDSPath, read_raw_bids)

from mne import events_from_annotations
from mne.preprocessing import ICA
import mne

from mne_faster import (find_bad_epochs, find_bad_channels)

from config import (bids_dir, deriv_dir, task, bv_montage,
                    event_id, preprocess_opts, ransac)
from functions import (get_sub_list, adjust_events_photosensor)

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
    if preprocess_opts['adjust_events']:
        print('Adjusting marker onsets with Photosensor signal')
        events, delays, n_adjusted = adjust_events_photosensor(
            raw, events, tmin=-.02, tmax=.05, threshold=.80,
            min_tdiff=.0085, return_diagnostics=True)
        print(f'  {n_adjusted} events were shifted')

    # Remove Photosensor from channels
    if preprocess_opts['drop_photosensor']:
        raw.drop_channels(preprocess_opts['photosensor_chan'])

    # STEP 3: RESAMPLE DATA AND EVENTS
    # Resample events and raw EEG data
    # This causes jitter, but it will be minimal
    raw, events = raw.resample(preprocess_opts['resample'], events=events)

    # Notch filter
    if preprocess_opts['notch_filter'] is not None:
        raw.notch_filter(60)

    # Save EEG data with json
    eeg_file = deriv_sub_dir / \
        f'{sub}_task-{task}_ref-FCz_desc-resamp_raw.fif.gz'
    raw.save(eeg_file, overwrite=True)

    # Save events
    events_file = deriv_sub_dir / f'{sub}_task-{task}_desc-resamp_eve.txt'
    mne.write_events(events_file, events)

    # STEP 4: HIGH-PASS FILTER DATA
    # High-pass filter data from all channels
    raw.filter(l_freq=preprocess_opts['ica_l_freq'],
               h_freq=preprocess_opts['ica_h_freq'],
               skip_by_annotation=['boundary'])

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
    if preprocess_opts['bad_chan_method'] == 'faster':
        bad_chans = find_bad_channels(epochs, max_iter=1, thres=4.0)
    elif preprocess_opts['bad_chan_method'] == 'ransac':
        ransac.fit(epochs.copy().drop_channels(epochs.info['bads']))
        bad_chans = ransac.bad_chs_
    for bc in bad_chans:
        epochs.info['bads'].append(bc)

    # Plot PSD TODO ADD BLOCKING FUNCTION (MATPLOTLIB)
    epochs.plot_psd(xscale='linear', show=True, n_jobs=4)

    # STEP 7: REJECT BAD EPOCHS FOR ICA
    # Find bad epochs using FASTER and drop them
    bad_epochs = find_bad_epochs(
        epochs.copy().drop_channels(epochs.info['bads']),
        return_by_metric=False, thres=preprocess_opts['faster_thresh'])
    epochs.drop(bad_epochs, reason='FASTER')

    # Find VEOG at stim onset and drop them
    veog_data = epochs.copy().crop(
        tmin=-.075, tmax=.075).get_data(picks=['VEOG'])
    veog_p2p = np.ptp(
        np.squeeze(veog_data), axis=1) > preprocess_opts['blink_thresh']
    epochs.drop(list(np.where(veog_p2p)[0]), reason='Onset Blink')

    # Inspect the remaining epochs
    epochs.plot(
        n_channels=len(epochs.ch_names), n_epochs=10,
        events=epochs.events, event_id=epochs.event_id,
        picks=epochs.ch_names, scalings=dict(eeg=40e-6, eog=150e-6),
        block=True)

    # Save epochs
    eeg_file = deriv_sub_dir / \
        f'{sub}_task-{task}_ref-FCz_desc-forica_epo.fif.gz'
    epochs.save(eeg_file, overwrite=True)

    # save bad channels
    bad_ch_file = deriv_sub_dir / f"{sub}_task-{task}_badchans.tsv"
    with open(bad_ch_file, 'w') as f:
        for bi, bc in enumerate(epochs.info['bads']):
            if bi == len(epochs.info['bads']):
                f.write(bc)
            else:
                f.write(bc + '\t')

    # Find epochs and channels that were confirmed dropped
    dropped_epochs = []
    for i, epo in enumerate(epochs.drop_log):
        if len(epo) > 0:
            dropped_epochs.append(i)

    # Save Dropped Epochs
    bad_epo_file = deriv_sub_dir / \
        f"{sub}_task-{task}_desc-forica_droppedepochs.tsv"
    with open(bad_epo_file, 'w') as f:
        for bi, be in enumerate(dropped_epochs):
            f.write(str(be))
            if bi != len(dropped_epochs):
                f.write('\t')

    # STEP 8: ESTIMATE ICA
    # Estimate ICA
    ica = ICA(method='picard', max_iter=1000,
              random_state=int(sub_id),
              fit_params=dict(ortho=True, extended=True),
              verbose=True)
    ica.fit(epochs)

    # Save ICA
    ica_file = deriv_sub_dir / f'{sub}_task-{task}_ref-FCz_ica.fif.gz'
    ica.save(ica_file)
