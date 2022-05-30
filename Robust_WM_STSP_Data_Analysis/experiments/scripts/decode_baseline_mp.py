import os
import argparse
from functools import partial
import multiprocessing as mp

import numpy as np
import pandas as pd
import xarray as xr


import neural_analysis.spikes as spk
from stabda.decode import run_svc, create_decode_storage
from stabda.data import bin_trials_mean, load_session_data
from stabda.utils import get_session_ids, xr_save, get_results_path


def run():
    """Run experiment for decoding pre-sample activity vs post-sample
    activity. Parallelized across sessions."""
    parser = argparse.ArgumentParser()
    parser.add_argument("kernel_width", type=int)
    parser.add_argument("bin_size", type=int)
    args = parser.parse_args()

    save_name = f"decode_baseline_{args.kernel_width}_{args.bin_size}.nc"
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
    Run decoding pre-sample activity vs post-sample activity for
    individual session.

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

    # set time range info for experiment
    Fs = 1000
    timepts = (np.arange(-2000, 5000, bin_size) / Fs) + bin_size / Fs / 2
    n_timepts = timepts.size

    sample_on = 2000
    pre_sample_bin_start = sample_on - bin_size
    baseline_pre_stim = 350

    baseline_start = pre_sample_bin_start - baseline_pre_stim
    baseline_end = sample_on - baseline_pre_stim

    # create storage for results
    ssn_result = create_decode_storage(timepts)

    # load spike_times array and trial_info dataframe
    info_keys = ["sample", "isDistractorShown", "delayLength"]
    spike_times, trial_info = load_session_data(session, info_keys)
    delays = np.unique(trial_info["delayLength"])

    samples = np.array([x for x in range(1, 9)])
    n_samples = samples.size

    print(f"{session}: {spike_times.shape}")
    for delay in delays:
        print(delay)
        for has_distr in [0, 1]:
            sample_results = np.zeros((n_samples, n_timepts))
            for s, samp in enumerate(samples):

                # select data
                subset_df = trial_info[
                    (trial_info["isDistractorShown"] == has_distr)
                    & (trial_info["sample"] == samp)
                    & (trial_info["delayLength"] == delay)
                ]
                subset_spike_times = spike_times[subset_df.index, :]

                # convert spike_times to smoothed rates
                temp, time = spk.times_to_bool(subset_spike_times, lims=[-2, 5])
                smoothed, _ = spk.density(temp, width=kw, timepts=time)

                n_trials = smoothed.shape[0]

                # select baseline period
                baseline = smoothed[:, :, baseline_start:baseline_end].mean(2)

                # construct binned time series
                X_test_all = bin_trials_mean(smoothed, bin_size)
                T = X_test_all.shape[2]
                test_acc = np.zeros(T)

                # run decoder over binned time series and get test accuracy
                for t in range(T):

                    test = X_test_all[:, :, t]

                    y = np.concatenate([np.repeat(0, n_trials), np.repeat(1, n_trials)])
                    X = np.concatenate([baseline, test], axis=0)

                    test_acc[t] = run_svc(y, X)

                sample_results[s, :] = test_acc
            ssn_result.loc[delay, has_distr, :] = sample_results.mean(0)

    return ssn_result


if __name__ == "__main__":
    run()
