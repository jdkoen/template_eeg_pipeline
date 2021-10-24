"""
Script: 01_compute_behavioral_data.py
Creator: Joshua D. Koen
Description: This script analyzes the behavioral data for the
1-back task. The script returns a trial level file and a summary
level file.

The trial level file name is formatted as:

sub-<id>_task-{task}_desc-triallevel_beh.tsv

This file contains RT and accuracy data (and other information)
for each trial. It is basically a copy of the original BIDS
behavioral data file.

The summary data file name is formatted as:

sub-<id>_task-{task}_desc-summary_beh.tsv



"""

# Import Libraries
import os
os.chdir(os.path.split(__file__)[0])

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict

from config import (bids_dir, deriv_dir, task)

from functions import get_sub_list

# Get Subject List
sub_list = get_sub_list(bids_dir, allow_all=True)

# Loop through subjects
for sub in sub_list:

    # Get subject information
    sub_id = sub.replace('sub-', '')
    bids_sub_dir = bids_dir / sub
    deriv_sub_dir = deriv_dir / sub
    deriv_sub_dir.mkdir(parents=True, exist_ok=True)
    fig_sub_dir = deriv_sub_dir / 'figures'
    fig_sub_dir.mkdir(parents=True, exist_ok=True)

    # Make figure
    plt.close('all')
    fig = plt.figure()
    fig.set_size_inches([11, 7])
    fig.suptitle(sub, fontsize=18)

    # Load the behavioral data file
    beh_file = bids_sub_dir / 'beh' / f'{sub}_task-{task}_beh.tsv'
    beh_data = pd.read_csv(beh_file, sep='\t')

    # Variables use for grouping summary measures.
    groups = ['category', 'presentation']

    # STEP 1: COMPUTE TRIAL COUNTS
    # Calculate trial counts in a new data frame
    counts = beh_data.groupby(groups)['correct'].value_counts().unstack(groups)
    counts.fillna('0', inplace=True)

    # Make A table for trial counts
    colLabels = [f'{x[0]} Rep{x[1]}' for x in counts.columns.to_flat_index()]
    colLabels = [x.replace('Rep1', 'first') for x in colLabels]
    colLabels = [x.replace('Rep2', '1back') for x in colLabels]
    ax1 = plt.subplot(2, 1, 1)
    ax1.set_title('Trial Counts', fontsize=18, fontweight='bold')
    rowLabels = ['Incorrect', 'Correct']
    if counts.shape[0] == 1 and counts.index[0] == 1:
        rowLabels = ['Correct']
    ax1.table(cellText=counts.values, colLabels=colLabels, rowLabels=rowLabels,
              loc='upper center')
    ax1.axis('off')

    # Compute proportions (accuracy) and plot it
    proportions = beh_data.groupby(
        groups)['correct'].mean().unstack(['category'])

    # Plot accuracy
    ax2 = plt.subplot(2, 2, 3)
    proportions.plot(kind='bar', ax=ax2)
    ax2.legend(loc='upper center', mode='expand', ncol=3,
               bbox_to_anchor=(.7, 1, .7, .12),
               frameon=False)
    ax2.tick_params(axis='x', rotation=0)
    ax2.set_title('Accuracy', y=1.15, fontsize=18, fontweight='bold')
    ax2.set_ylabel('p(Correct)')
    ax2.set_xticklabels(['first', '1back'])

    # Get Colors
    colors = []
    colors.append(ax2.get_children()[0].get_facecolor())
    colors.append(ax2.get_children()[2].get_facecolor())
    colors.append(ax2.get_children()[4].get_facecolor())

    # STEP 2: MEDIAN RT MEASURES
    # Compute RT measures
    ax3 = plt.subplot(2, 2, 4)
    median_rts = beh_data.query('correct==1 and presentation==2')
    median_rts = median_rts.groupby('category')['rt'].median().to_frame()
    mean_rts = beh_data.query('correct==1 and presentation==2')
    mean_rts = mean_rts.groupby('category')['rt'].mean().to_frame()
    sd_rts = beh_data.query('correct==1 and presentation==2')
    sd_rts = sd_rts.groupby('category')['rt'].std().to_frame()
    ax3.bar(np.array([1, 1.5, 2]),
            median_rts.pivot_table(columns='category',
                                   values='rt').values.squeeze(),
            width=.3, color=colors)
    ax3.set_xlim(.25, 2.75)
    ax3.set_ylim(.25, ax3.get_ylim()[-1]+.1)
    ax3.tick_params(axis='x', which='both', bottom=False, top=False,
                    labelbottom=False)
    ax3.set_title('Median RT', y=1.15, fontsize=18, fontweight='bold')
    ax3.set_ylabel('RT (sec)')
    ax3.set_xlabel('*for correct 1back trials only')

    # Plot labels
    x_pos = .85
    for v in median_rts.values:
        ax3.text(x_pos, v[0]+.015, f'{v[0]: 1.2f}', color='black')
        x_pos += .5

    # Save plot to figures
    fig_file = fig_sub_dir / f'{sub}_task-{task}_beh_performance.png'
    fig.savefig(fig_file, dpi=600)

    # STEP 3: MAKE DATA FRAME TO CREATE OUTPUTS
    # Make dictionary and basic values
    out_dict = OrderedDict()
    out_dict = {
        'id': sub,
        'age': ['young' if int(sub_id) < 200 else 'older'],
        'set': [beh_data.iloc[0]['stim_list']]
    }

    # Add in accuracy values
    for cat in ['scenes', 'objects', 'faces']:
        for i, rep in enumerate(['first', '1back']):
            out_dict[f'{cat}_{rep}_acc'] = [proportions.loc[i+1][cat]]

    # Add in RT values
    for val in ['median', 'mean', 'sd']:
        for cat in ['scenes', 'objects', 'faces']:
            out_dict[f'{cat}_{val}_rt'] = \
                eval(f'[{val}_rts.loc["{cat}"]["rt"]]')

    # Write summary data file
    summary_file = deriv_sub_dir / f'{sub}_task-{task}_desc-summary_beh.tsv'
    summary_df = pd.DataFrame().from_dict(out_dict).to_csv(
        summary_file, index=False, sep='\t')

    # Write copy of behavior data to derivatives
    trial_file = deriv_sub_dir / f'{sub}_task-{task}_desc-triallevel_beh.tsv'
    beh_data.to_csv(trial_file, sep='\t', index=False)

    # Show plot
    plt.show(block=True)
