"""
Script: 04_eeg_remove_artifacts.py
Creator: Joshua D. Koen
Description: This script imports data from sourcedata to bids format for
the SOF (scene, object, face) task and runs some preprocessing on the data.
"""

# Import Libraries
import os
os.chdir(os.path.split(__file__)[0])

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv

from mne import read_events
from mne.io import read_raw_fif
from mne.preprocessing import read_ica
from mne.viz import plot_compare_evokeds
import mne

from mne_faster import (find_bad_epochs, find_bad_channels_in_epochs)

from config import (deriv_dir, task, preprocess_opts, event_id,
                    bv_montage)

from functions import get_sub_list

# Ask for subject IDs to analyze
sub_list = get_sub_list(deriv_dir, allow_all=True)

# Loop through subjecgts
for sub in sub_list:

    # STEP 1: SUBJECT INFORMATION DEFINITION ###
    # Define the Subject ID and paths
    sub_id = sub.replace('sub-', '')
    deriv_sub_dir = deriv_dir / sub
    print(f'Loading task-{task} data for {sub}')

    # STEP 2: LOAD RESAMPLE RAW, BAD CHANNELS, EVENTS, METADATA, & ICA
    # Load raw fif file
    eeg_file = deriv_sub_dir / \
        f'{sub}_task-{task}_ref-FCz_desc-resamp_raw.fif.gz'
    raw = read_raw_fif(eeg_file, preload=True)

    # Read in bad channels and add to raw
    bad_chans = []
    bad_ch_file = deriv_sub_dir / f"{sub}_task-{task}_badchans.tsv"
    with open(bad_ch_file, 'r') as f:
        for bcs in csv.reader(f, dialect="excel-tab"):
            [bad_chans.append(x) for x in bcs if x]
    raw.info['bads'] = bad_chans

    # Read in events
    events_file = deriv_sub_dir / f'{sub}_task-{task}_desc-resamp_eve.txt'
    events = read_events(events_file)

    # Load metadata file
    metadata_file = deriv_sub_dir / f'{sub}_task-{task}_desc-orig_metadata.tsv'
    metadata = pd.read_csv(metadata_file, sep='\t')

    # Load ICA
    ica_file = deriv_sub_dir / f'{sub}_task-{task}_ref-FCz_ica.fif.gz'
    ica = read_ica(ica_file)

    # STEP 3: FILTER RAW DATA
    # High-pass filter data from all channels
    raw.filter(l_freq=preprocess_opts['l_freq'],
               h_freq=preprocess_opts['h_freq'],
               skip_by_annotation=['boundary'])

    # STEP 4: EPOCH DATA, REMOVE BAD ICs, INTERPOLATE BAD
    # CHANNELSS, REREFERENCE, & BASELINE CORRECTION
    # Make the epochs from extract event data
    epochs = mne.Epochs(raw, events, event_id=event_id,
                        tmin=preprocess_opts['tmin'],
                        tmax=preprocess_opts['tmax'],
                        baseline=None,
                        metadata=metadata,
                        reject=None,
                        preload=True)

    # Save original eeg epochs
    eeg_file = deriv_sub_dir / \
        f'{sub}_task-{task}_ref-FCz_desc-orig_epo.fif.gz'
    epochs.save(eeg_file, overwrite=True)

    # Make ERP over all trials with IC artifacts still there
    erps_to_check = [epochs.average()]
    erps_to_check[0].comment = 'Original'

    # Subtract bad ICs (operates on epochs in-place) and add to erps_to_check
    ica.apply(epochs)

    # Check IC artifact cleaning interactively
    erps_to_check.append(epochs.average())
    erps_to_check[-1].comment = 'Cleaned'
    plot_compare_evokeds(
        erps_to_check, picks='eeg', show=True, axes='topo', colors=['k', 'r'],
        title='Signals before (black) and after (red) cleaning)')
    plt.show()

    # Interpolate bad channels
    epochs.interpolate_bads(reset_bads=True, mode='accurate')

    # Re-reference to average (recovering FCz)
    epochs.add_reference_channels('FCz')
    epochs.set_montage(bv_montage)
    epochs.set_eeg_reference(ref_channels='average')

    # Baseline correct
    epochs.apply_baseline(preprocess_opts['baseline'])

    # STEP 3: ARTIFACT REJECTION
    # Find bad epochs using FASTER
    bad_epochs = find_bad_epochs(epochs, return_by_metric=False,
                                 thres=preprocess_opts['faster_thresh'])
    epochs.drop(bad_epochs, reason='FASTER (Epochs)')

    # Find VEOG at stim onset and drop them
    veog_data = epochs.copy().crop(
        tmin=-.075, tmax=.075).get_data(picks=['VEOG'])
    veog_p2p = np.ptp(
        np.squeeze(veog_data), axis=1) > preprocess_opts['blink_thresh']
    epochs.drop(list(np.where(veog_p2p)[0]), reason='Onset Blink')

    # Find bad channels in each epoch
    # If # bad channels fewer than preprocess_opts['faster_bad_n_chans'],
    # interpolate the bads. Otherwise, reject the epoch
    metrics = ['amplitude', 'deviation', 'variance', 'median_gradient']
    bad_chans_in_epo = find_bad_channels_in_epochs(
        epochs, thres=preprocess_opts['faster_thresh'], use_metrics=metrics)
    bad_epochs_by_chan = []
    for i, b in enumerate(bad_chans_in_epo):
        n_bad_epoch = len(b)
        if n_bad_epoch > preprocess_opts['faster_bad_n_chans']:
            bad_epochs_by_chan.append(i)
        elif n_bad_epoch > 0:
            ep = epochs[i]
            ep.info['bads'] = b
            ep.interpolate_bads(verbose=False)
            epochs._data[i, :, :] = ep._data[0, :, :]
    print('Epochs with bad n_chans > threshold: ', bad_epochs_by_chan)
    epochs.drop(bad_epochs_by_chan, reason='FASTER (Chans)')

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

    # Save Dropped Epochs
    bad_epo_file = deriv_sub_dir / \
        f"{sub}_task-{task}_desc-cleaned_droppedepochs.tsv"
    with open(bad_epo_file, 'w') as f:
        for bi, be in enumerate(dropped_epochs):
            f.write(str(be))
            if bi != len(dropped_epochs):
                f.write('\t')

    # Save metadata
    metadata_file = deriv_sub_dir / \
        f'{sub}_task-{task}_desc-cleaned_metadata.tsv'
    epochs.metadata.to_csv(metadata_file, sep='\t', index=False)
