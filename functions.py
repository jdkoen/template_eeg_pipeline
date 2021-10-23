

# Define subject list function
def get_sub_list(data_dir, allow_all=False):

    # Ask for subject IDs to analyze
    print('What IDs are being preprocessed?')
    print('(Enter multiple values separated by a comma; e.g., 101,102)')
    if allow_all:
        print('To process all subjects, type all')
    sub_list = input('Enter IDs: ')

    if sub_list == 'all' and allow_all:
        sub_list = [x.name for x in data_dir.glob('sub-*')]
    else:
        sub_list = [f'sub-{x}' for x in sub_list.split(',')]

    return sorted(sub_list)



    # STEP 6: MAKE COPY IN DERIVATIVES
    # Write Raw instance
    raw_out_file = deriv_path / \
        f'sub-{bids_id}_task-{task}_ref-FCz_desc-import_raw.fif.gz'
    raw.save(raw_out_file, overwrite=overwrite)

    # Make a JSON
    json_info = {
        'Description': 'Import from BrainVision Recorder',
        'sfreq': raw.info['sfreq'],
        'reference': 'FCz'
    }
    json_file = deriv_path / \
        f'sub-{bids_id}_task-{task}_ref-FCz_desc-import_raw.json'
    with open(json_file, 'w') as outfile:
        json.dump(json_info, outfile, indent=4)
    del json_info, json_file

    # Write events
    events_out_file = deriv_path / \
        f'sub-{bids_id}_task-{task}_desc-import_eve.txt'
    mne.write_events(events_out_file, events)

    # Make a JSON
    json_info = {
        'Description': 'Events from Brain Vision Import',
        'columns': ['onset', 'duration', 'code'],
        'onset_units': 'samples',
        'sfreq': raw.info['sfreq'],
        'codes': event_id
    }
    json_file = deriv_path / f'{sub}_task-{task}_desc-import_eve.json'
    try:
        json_file.unlink()
    except OSError:
        pass
    json_file = deriv_path / f'sub-{bids_id}_task-{task}_desc-import_eve.json'
    with open(json_file, 'w') as outfile:
        json.dump(json_info, outfile, indent=4)
    del json_info, json_file
