import numpy as np
from mne import pick_channels


def get_sub_list(data_dir, allow_all=False):

    # TODO Add docstring
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


def adjust_events_photosensor(raw, events, photosensor='Photosensor',
                              tmin=-.02, tmax=.05, threshold=.80,
                              min_tdiff=.0085, return_diagnostics=False):

    # TODO Add docstring
    # TODO Add input checks

    # Extract the needed data
    data, times = raw.copy().pick_channels(
        [photosensor]).get_data(return_times=True)
    data = np.squeeze(data)
    sfreq = raw.info['sfreq']
    latencies = (times * sfreq).astype(int)

    # Convert tmin, tmax and min_tdiff to samples
    lmin = int(tmin * sfreq)
    lmax = int(tmax * sfreq)
    min_ldiff = np.ceil(min_tdiff * sfreq).astype(int)

    # Loop through events
    delays = np.array([])
    n_events_adjusted = 0
    for event in events:

        # Get segment latency window and baseline window
        seg_lonset = event[0]
        seg_lstart = seg_lonset + lmin
        seg_lend = seg_lonset + lmax
        seg_window = np.logical_and(
            latencies >= seg_lstart, latencies < seg_lend)
        seg_baseline = np.logical_and(
            latencies >= seg_lstart, latencies < seg_lonset)

        # Extract data segment and substract baseline, and latencies
        # for this event
        data_segment = data[seg_window] - data[seg_baseline].mean()
        latency_segment = latencies[seg_window]
        psensor_lonset = latency_segment[np.where(
            data_segment > (data_segment.max() * threshold))[0][0]]

        # Compute the delay in samples
        this_delay = psensor_lonset - seg_lonset
        delays = np.append(delays, this_delay)

        # Correct onset if delayed by too much (either direction)
        if this_delay > min_ldiff:
            event[0] = psensor_lonset
            n_events_adjusted += 1

    if return_diagnostics:
        return (events, delays * sfreq, n_events_adjusted)
    else:
        return events


def inspect_epochs(inst, bad_epochs=None, events=None, event_id=None,
                   n_epochs=10, block=True, scalings=None, return_copy=True):

    # If return copy is true, make a copy
    if return_copy:
        epochs = inst.copy()
    else:
        epochs = inst

    # Update bad epochs
    if bad_epochs is None:
        bad_epochs = []

    # Make color index for artifacts
    epoch_colors = list()
    n_channels = len(epochs.ch_names)
    for epoch_idx in np.arange(len(epochs)):
        if epoch_idx in bad_epochs:
            epoch_color = ['m'] * n_channels
        else:
            epoch_color = ['k'] * n_channels
        epoch_colors.append(epoch_color)

    # Mark bad channels as grey
    bad_chans = pick_channels(epochs.ch_names, epochs.info['bads']).tolist()
    for i, _ in enumerate(epoch_colors):
        for c in bad_chans:
            epoch_colors[i][c] = (.8, .8, .8, 1)

    # Visually inspect epochs
    epochs.plot(n_channels=n_channels, n_epochs=n_epochs, block=block,
                scalings=scalings, epoch_colors=epoch_colors,
                events=events, event_id=event_id, picks=epochs.ch_names)

    return epochs
