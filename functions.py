import numpy as np

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


# def adjust_events_photosensor(raw, events, photosensor='Photosensor',
#                               tmin=-.02, tmax=.05, threshold=.90,
#                               min_dif=.005):
    
#     # TODO Add input checks

#     # Extract the needed data
#     data, times = raw.copy().pick_channels(
#         [photosensor]).get_data(return_times=True)
#     sfreq = raw.info['sfreq']
#     latencies = (times * sfreq).astype(int)
    
#     # Loop through events
#     for event in events:
#         print(event)

        




#     return raw, adjustments