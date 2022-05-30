import os
import numpy as np
import pandas as pd
import xarray as xr

import neural_analysis.matIO as io
import neural_analysis.spikes as spk

from stabda.utils import get_data_path


def bin_trials_mean(trial_data, bin_size):
    """
    Takes trial data and bins by taking mean within each bin. Even n_timepts /
    bin_size is not 0, data is removed from end of time series.

    Args:
        trial_data: (ndarray) Neural data with dims (trials, units, time).
        bin_size: (int) Length (in indicies) of bin size. 

    Returns:
        trials_binned: (ndarray) Result of binning and taking mean, with dims
            (trials, units, bins).
    """
    trial_length = trial_data.shape[2]
    n_bins = np.floor(trial_length / bin_size)

    cut_ind = int(n_bins * bin_size)
    clipped = trial_data[:, :, :cut_ind]

    split = np.split(clipped, n_bins, axis=2)
    means = [x.mean(2) for x in split]
    trial_data_binned = np.stack(means, axis=2)

    return trial_data_binned


def load_session_data(
    session_id, trial_info_keys, drop_dead_units=True, pool_units_on_electrode=True
):
    """
    Load data from experimental session. 
    
    Args:
        session_id: (str) ID for experimental session
            options include ["1203", "1204", "1205", "1206", "1207", "1210"]
        trial_info_keys: (list) - Trial information to return 
            - typically use ["sample", "isDistractorShown", "delayLength"]
        drop_dead_units: (bool) Option to drop dead neurons (neurons with no
            spikes). Defaults to True. 
        pool_units_on_electrode: (bool) Option to pool spiking units if they
            come from the same electrode source. Default is True. 

    Returns:
        spike_times: (ndarray) Numpy object array with dims (trials, units).
            Each entry contains a 1d-arrya of spike times. 
        trial_info_df: (DataFrame) Pandas dataframe containing selected trial
            information. Each column has length n_trials.
    """
    # load tuple
    # select relevant parts
    data_dir = get_data_path()
    session_path = os.path.join(data_dir, f"Tiergan-DMTS-2018{session_id}.mat")

    load_vars = ["unitInfo", "trialInfo", "spikeTimes"]

    # load data
    unit_info, trial_info, spike_times = io.loadmat(
        session_path, variables=load_vars, verbose=False
    )

    # pool units if source is same electrode
    if pool_units_on_electrode:
        spike_times = spk.pool_electrode_units(spike_times, unit_info["electrode"])

    # drop units with no activity
    if drop_dead_units:
        n_units = spike_times.shape[1]
        spike_counts = []
        for i in range(n_units):
            max_spikes = np.array([x.size for x in spike_times[:, i]]).max()
            spike_counts.append(max_spikes)

        dead_filt = np.array(spike_counts) == 0
        spike_times = spike_times[:, ~dead_filt]

    new_dict = {k: trial_info[k] for k in trial_info_keys}
    trial_info_df = pd.DataFrame.from_dict(new_dict)

    return spike_times, trial_info_df
