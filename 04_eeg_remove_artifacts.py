"""
Script: 03_sof_preprocess_eeg.py
Creator: Joshua D. Koen
Description: This script imports data from sourcedata to bids format for
the SOF (scene, object, face) task and runs some preprocessing on the data.
"""

# Import Libraries
import os
os.chdir(os.path.split(__file__)[0])

import numpy as np
import json

from mne import read_epochs
from mne.preprocessing import read_ica
import mne

from mne_faster import (find_bad_epochs, find_bad_channels_in_epochs)

from config import (deriv_dir, task, preprocess_opts)

from functions import get_sub_list

# Ask for subject IDs to analyze
sub_list = get_sub_list(deriv_dir, allow_all=True)

# Loop through subjecgts
for sub in sub_list:

    # STEP 1: SUBJECT INFORMATION DEFINITION ###
    # Define the Subject ID and paths
    sub_id = sub.replace('sub-', '')
    deriv_sub_dir = deriv_dir / sub
    fig_sub_dir = deriv_sub_dir / 'figures'
    print(f'Loading ICA task-{task} data for {sub}')

    # STEP 2: LOAD EPOCHS AND ICA OBJECTS, CLEAN BAD ICS, AND BASELINE
    # Load ICA
    ica_file = deriv_sub_dir / f'{sub}_task-{task}_ref-avg_ica.fif.gz'
    ica = read_ica(ica_file)

    # Load Epochs with channels interpolated, and apply ICA in place
    epoch_fif_file = deriv_sub_dir / \
        f'{sub}_task-{task}_ref-avg_desc-interpbads_epo.fif.gz'
    epochs = read_epochs(epoch_fif_file)
    ica.apply(epochs)

    # Re-reference to average
    epochs.set_eeg_reference(ref_channels='average')

    # Baseline correct
    epochs.apply_baseline(preprocess_opts['baseline'])

    # STEP 3: ARTIFACT REJECTION
    # Find bad epochs using FASTER
    bad_epochs = find_bad_epochs(epochs, return_by_metric=True,
                                 thres=preprocess_opts['faster_thresh'])
    for reason, drop_epochs in bad_epochs.items():
        epochs.drop(drop_epochs, reason=reason)

    # Find VEOG at stim onset
    veog_data = epochs.copy().crop(
        tmin=-.075, tmax=.075).get_data(picks=['VEOG'])
    veog_p2p = np.ptp(
        np.squeeze(veog_data), axis=1) > preprocess_opts['blink_thresh']
    bad_epochs['blink_onset'] = list(np.where(veog_p2p)[0])
    print('\tEpochs with blinks at onset: ', bad_epochs['blink_onset'])
    epochs.drop(bad_epochs['blink_onset'], reason='Onset Blink')

    # Find bad channels in each epoch
    # If # bad channels fewer than preprocess_opts['faster_bad_n_chans'],
    # interpolate the bads. Otherwise, reject the epoch
    metrics = ['amplitude', 'deviation', 'variance', 'median_gradient']
    bad_chans_in_epo = find_bad_channels_in_epochs(
        epochs, thres=preprocess_opts['faster_thresh'], use_metrics=metrics)
    bad_epochs['bad_chans_in_epo'] = []
    for i, b in enumerate(bad_chans_in_epo):
        n_bad_epoch = len(b)
        if n_bad_epoch > preprocess_opts['faster_bad_n_chans']:
            bad_epochs['bad_chans_in_epo'].append(i)
        elif n_bad_epoch > 0:
            ep = epochs[i]
            ep.info['bads'] = b
            ep.interpolate_bads(verbose=False)
            epochs._data[i, :, :] = ep._data[0, :, :]
    print('\tEpochs with bad n_chans > threshold: ',
          bad_epochs['bad_chans_in_epo'])
    epochs.drop(bad_epochs['bad_chans_in_epo'],
                reason='High Bad Chans in Epoch')

    # Inspect the remaining epochs
    epochs.plot(
        n_channels=len(epochs.ch_names), n_epochs=10,
        events=epochs.events, event_id=epochs.event_id,
        picks=epochs.ch_names, scalings=dict(eeg=25e-6, eog=150e-6),
        block=True)

    # Save epochs
    eeg_file = deriv_sub_dir / \
        f'{sub}_task-{task}_ref-avg_desc-cleaned_epo.fif.gz'
    epochs.save(eeg_file, overwrite=True)

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
        'Description': 'Epochs after artifact correction and rejection',
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
