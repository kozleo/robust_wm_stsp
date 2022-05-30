import os
import argparse
from functools import partial
import multiprocessing as mp

import numpy as np
import pandas as pd
import xarray as xr

import neural_analysis.spikes as spk
from stabda.decode import run_svc, create_decode_storage
from stabda.data import load_session_data, bin_trials_mean
from stabda.utils import xr_save, get_session_ids, get_results_path


def run():
    """Run experiment for decoding sample identity activity.
    Parallelized across sessions."""
    parser = argparse.ArgumentParser()
    parser.add_argument("kernel_width", type=int)
    parser.add_argument("bin_size", type=int)
    args = parser.parse_args()
    print(f"Running sample decode with args: {args.kernel_width}, {args.bin_size}")

    save_name = f"decode_sample_{args.kernel_width}_{args.bin_size}.nc"
    results_path = get_results_path()
    save_path = os.path.join(results_path, save_name)

    n_cores = 8
    pool = mp.Pool(n_cores)

    func = partial(
        run_session_decode, kernel_width=args.kernel_width, bin_size=args.bin_size
    )

    sessions = get_session_ids()
    results_list = pool.map(func, sessions)

    session_index = pd.Index(sessions, name="session")
    result = xr.concat(results_list, dim=session_index)

    xr_save(result, save_path)


def run_session_decode(session, kernel_width, bin_size):
    """
    Run decoding for sample identity for individual session.

    Args:
        session: (str) Session id.
        kernel_width: (float) Width of Gaussian kernel for smoothing
            spike trains. Units are in milliseconds.
        bin_size: (int) Length (in number of timepoints) of bin used.
            Mean is taken across time within each bin.

    Returns:
        ssn_result: (xarray DataArray) Decoding results.
    """
    kw = kernel_width * 1e-3

    # set time range for experiment
    Fs = 1000
    sample_on = 2000
    pre = 200
    post = 5000

    st = sample_on - pre
    en = sample_on + post

    timepts = (np.arange(-pre, post, bin_size) / Fs) + bin_size / Fs / 2

    # create storage for results
    ssn_result = create_decode_storage(timepts)

    # load spike_times array and trial_info dataframe
    info_keys = ["sample", "isDistractorShown", "delayLength"]
    spike_times, trial_info = load_session_data(session, info_keys)

    delays = np.unique(trial_info["delayLength"])
    distr_options = [0, 1]

    # TODO improve printing
    print(f"{session}: {spike_times.shape}")
    for i, delay in enumerate(delays):
        print(delay)
        for has_distr in distr_options:

            # select data
            distr_filt = trial_info["isDistractorShown"] == has_distr
            delay_filt = trial_info["delayLength"] == delay

            subset_df = trial_info[distr_filt & delay_filt]
            subset_spike_times = spike_times[subset_df.index, :]

            # convert spike_times to smoothed rates
            smoothed, _ = spk.density(subset_spike_times, width=kw, lims=[-2, 5])

            # clip time to experiment range
            smoothed_use = smoothed[:, :, st:en]

            # set up training data / labels
            y = subset_df["sample"].values
            X_all = bin_trials_mean(smoothed_use, bin_size)

            # run decoder over binned time series and get test accuracy
            T = X_all.shape[2]
            test_acc = np.zeros(T)

            for i in range(T):
                X = X_all[:, :, i]
                test_acc[i] = run_svc(y, X)

            ssn_result.loc[delay, has_distr, :] = test_acc

    return ssn_result


if __name__ == "__main__":
    run()
