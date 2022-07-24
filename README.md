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

### Step 1. ICA Preprocessing (`01_ica_estimation.py`)

This script filters the continuous data, segments the data into epochs, and performs semi-automated artifact rejection prior to running ICA estimation. 

### Step 2. ICA Artifact Rejection (`02_ica_reject_artifats.py`)



*TODO* At present, the scripts do not contain a standard method for time-frequency analysis of EEG data. This will be developed in a future iteration of the template. 
