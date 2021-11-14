"""
Script: 99_make_report.py
Creator: Joshua D. Koen
Description: Makes an html report of the data
"""
# Import Libraries
import os

os.chdir(os.path.split(__file__)[0])

import csv

from mne.io import read_raw_fif
from mne import (Report, read_epochs)

from config import (deriv_dir, report_dir, task, event_id, preprocess_opts)
from functions import get_sub_list

# Ask for subject IDs to analyze
sub_list = get_sub_list(deriv_dir, allow_all=True)

# Loop through subjects
for sub in sub_list:

    # STEP 1: SUBJECT INFORMATION DEFINITION
    # Define id and directory information
    sub_id = sub.replace('sub-', '')
    deriv_sub_dir = deriv_dir / sub
    fig_sub_dir = deriv_sub_dir / 'figures'
    print(f'Creating Report for task-{task} data for {sub}')

    # Gather file names of interest
    behav_fig_file = fig_sub_dir / f'{sub}_task-{task}_beh_performance.png'
    raw_fif_file = deriv_sub_dir / \
        f'{sub}_task-{task}_ref-FCz_desc-resamp_raw.fif.gz'
    events_file = deriv_sub_dir / f'{sub}_task-{task}_desc-resamp_eve.txt'
    ica_file = deriv_sub_dir / f'{sub}_task-{task}_ref-FCz_ica.fif.gz'
    orig_epo_file = deriv_sub_dir / \
        f'{sub}_task-{task}_ref-FCz_desc-orig_epo.fif.gz'
    clean_epo_file = deriv_sub_dir / \
        f'{sub}_task-{task}_ref-avg_desc-cleaned_epo.fif.gz'
    erp_file = deriv_sub_dir / f'{sub}_task-{task}_ref-avg_ave.fif.gz'

    # STEP 2: INITIALIZE REPORT
    report = Report(subject=sub, title=f'{sub}: task-{task} report',
                    image_format='png', verbose=True, projs=False,
                    subjects_dir=None)

    # STEP 3: ADD BEHAVIOR DATA SECTION
    report.add_image(behav_fig_file, title='Behavior Summary')

    # STEP 3: ADD RAW DATA with bad channels
    raw = read_raw_fif(raw_fif_file, preload=True)
    bad_chans = []
    bad_ch_file = deriv_sub_dir / f"{sub}_task-{task}_badchans.tsv"
    with open(bad_ch_file, 'r') as f:
        for bcs in csv.reader(f, dialect="excel-tab"):
            [bad_chans.append(x) for x in bcs if x]
    raw.info['bads'] = bad_chans
    report.add_figure(
        fig=raw.plot_sensors(show_names=True, show=False),
        title='EEG Channels',
        caption='Good (black) and bad (red) channels in dataset',
        image_format='PNG')
    report.add_raw(raw=raw, title='Continuous EEG', psd=True,
                   butterfly=False)

    # STEP 4: ADD IN EVENTS
    report.add_events(
        events_file, title='Events', event_id=event_id,
        sfreq=raw.info['sfreq'])

    # STEP 6: ADD IN ICA
    epochs_orig = (read_epochs(orig_epo_file)
                   .apply_baseline(preprocess_opts['baseline']))
    report.add_ica(ica_file, title='ICA', inst=epochs_orig, n_jobs=4)

    # STEP 7: ADD IN CLEANED EPOCHS
    report.add_epochs(clean_epo_file, title='Cleaned Epochs', psd=True)

    # STEP 8: ADD IN EVOKEDS
    report.add_evokeds(erp_file, n_time_points=50,
                       topomap_kwargs=dict(average=.4))

    # Save report
    report_file = report_dir / f'{sub}_task-{task}_report.html'
    report.save(report_file, overwrite=True, open_browser=True)
    report.save(report_file.with_suffix('.hdf5'), overwrite=True)
