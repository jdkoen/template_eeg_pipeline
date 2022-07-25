# Template EEG Processing Pipeline

## Purpose

This is a series of scripts developed by myself for the [Koen Lab](https://github.com/koenlab) to process EEG data. These are Python scripts that use the [MNE-Python](https://mne.tools/stable/index.html) as well as custom function.

At present, these are primarily for pre-processing of single subject data. No group analysis scripts are included in this template. 

The code is accompanied by example data store with [Git LFS](https://git-lfs.github.com/). 

## Brief Overview of Use

To use this template, fork this repository and create a new repository using this as a template. You will then go through the scripts and update information where needed. 

The environment used for these scripts can be installed using `anaconda` by importing the `mne_environment.yml` file. 

The data are imported from how they are stored and into the [Brain Imaging Data Structure (BIDS)](https://bids-specification.readthedocs.io/en/stable/05-derivatives/01-introduction.html) using [`mne-bids`](https://mne.tools/mne-bids/stable/index.html).

### Config file

The main preprocessing options are locating in the `config.py` file. These options can be modified there and control all other aspects of processing. These control aspects of processing such as:

* Epoch length
* Filtering methods
* Automated bad channel and epoch methods
* ERP plotting options for reports
* Directories

### Data Import (`00_data_import.py`)

This script handles importing the data (both EEG and behavioral data) from a source data folder into the BIDS format. This script is not standardized and must be updated with proper file names specific to your experiment. Relevant variables for this script located in the `config.py` are:

* `cols_to_keep`
* `cols_to_rename`
* `cols_to_add`

### Step 1. Process Behavioral Data (`01_compute_behavioral_data.py`)

This script will need to be customized for each experiment. The purpose is to derive the behavioral dependent variables of interest from the behavioral tasks.

### Step 2. ICA Preprocessing (`02_ica_estimation.py`)

This script filters the continuous data, segments the data into epochs, and performs semi-automated artifact rejection prior to running ICA estimation. 

### Step 3. ICA Artifact Rejection (`03_ica_reject_artifats.py`)

This script loads the estimated ICA components output from Step 1 and plots the component properties (using `mne.preprocessing.ICA.plot_components`) to be inspected for rejection. Some components are automatically flagged based on algorithms implemented in [me-faster](https://github.com/wmvanvliet/mne-faster. The components automatically flagged are not necessarily artifcatual, and require confirmation via visual inspection to mark for rejection. 

### Step 4. Remove Artifacts (`04_eeg_remove_artifacts.py`)

This script first loads the EEG data, filters the continuous data, and segments the data into epochs. The ICA components flagged as artifactual are subtracted from the newly created epoched data and then channels marked as bad are interpolated. Then, a semi-automated artifct detection routine is implmented that does the following (in order):

1. Reject epochs with a blink at stimulus onset
2. Reject epochs flagged as artifacts based on mne-faster algorithms
3. Visual inspection of the remaining artifacts. 

During the final visual inspection, additional EEG channels can be marked as artifactual, which are subsequently interpolated. The cleaned epochs are written to file. 

### Step 5. Make ERPs (`05_make_erps.py`)

This script makes the ERPs and plots some for inspection of data quality. The ERPs that are written and shown can be controlled by changing the `erp_queries`, `erp_contrasts`, and `erps_to_plot` variables in `config.py`.

### Make Report (`99_make_report.py`)

This script makes an html report using the `Report` class from mne-python. This follows a standardized format that will largely read from the above scripts output. It will need some modification for naming of the behavioral data files that are written (including figures). 

### Other

The `functions` script contains utility functions used throughout the scripts, such as `get_sub_list` and `adjust_events_photosensor`. 

## Notes

This codebase is under development and may experience constant change. For this reason, it is import to use this as a template only when starting a new experiment. Do not update the 'version' of this code base for an ungoing experiment unless otherwise necessary

At present, the scripts do not contain a standard method for time-frequency analysis of EEG data. This will be developed in a future iteration of the template. 
