"""
Script: 05_make_erps.py
Creator: Joshua D. Koen
Description: This script loads cleaned epoch data and makes
evoked objects for conditions of interest.
"""

# Import Libraries
import os
os.chdir(os.path.split(__file__)[0])

import matplotlib.pyplot as plt

from mne import (read_epochs, combine_evoked, write_evokeds)
from mne.viz import plot_compare_evokeds

from config import (deriv_dir, task, preprocess_opts,
                    erp_queries, erp_contrasts, erps_to_plot)
from functions import get_sub_list

# Ask for subject IDs to analyze
sub_list = get_sub_list(deriv_dir, allow_all=True)

# Loop through subjects
for sub in sub_list:

    # STEP 1: SUBJECT INFORMATION DEFINITION
    # Define the Subject ID and paths
    sub_id = sub.replace('sub-', '')
    deriv_sub_dir = deriv_dir / sub
    print(f'Creating ERPs for task-{task} data for {sub}')

    # STEP 2: LOAD CLEANED EPOCHS
    # Read in Cleaned Epochs
    eeg_file = deriv_sub_dir / \
        f'{sub}_task-{task}_ref-avg_desc-cleaned_epo.fif.gz'
    epochs = read_epochs(eeg_file)

    # STEP 3: MAKE EVOKEDS
    # Make empty list to store data
    evokeds = []

    # Loop through erp_queries
    for comment, query in erp_queries.items():
        evoked = epochs[query].average()
        evoked.comment = comment
        evokeds.append(evoked)

    # STEP 4: MAKE DIFFERENCE WAVES
    # Add in difference waves
    for comment, contrast in erp_contrasts.items():
        contrast_evokeds = [
            e for e in evokeds if e.comment in contrast['conds']]
        evoked = combine_evoked(contrast_evokeds, weights=contrast['weights'])
        evoked.comment = comment
        evokeds.append(evoked)

    # STEP 5: CROP EVOKEDS IN NEEDED
    # Crop evokeds
    tmin = preprocess_opts['erp_tmin']
    tmax = preprocess_opts['erp_tmax']
    evokeds = [x.crop(tmin=tmin, tmax=tmax) for x in evokeds]

    # STEP 5: WRITE EVOKEDS
    # Write evoked file
    erp_file = deriv_sub_dir / f'{sub}_task-{task}_ref-avg_ave.fif.gz'
    write_evokeds(erp_file, evokeds)

    # STEP 6: PLOT EVOKEDS ON TOPO
    for title, conds in erps_to_plot.items():
        evokeds_to_plot = [e for e in evokeds if e.comment in conds]
        plot_compare_evokeds(
            evokeds_to_plot, picks='eeg', title=title, axes='topo',
            show=True)
        plt.show()
