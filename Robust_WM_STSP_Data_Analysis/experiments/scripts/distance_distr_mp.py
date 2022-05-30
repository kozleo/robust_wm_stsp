import os
import numpy as np
import argparse
import itertools as it
import multiprocessing as mp
from functools import partial


import neural_analysis.matIO as io
import neural_analysis.spikes as spk

from stabda.utils import get_results_path, get_session_ids, xr_save, get_session_id_map
from stabda.data import load_session_data
from stabda.bootstrap import create_bs_distance_storage, draw_bs_dist_distr


def run():
    """Run experiment calculating distance between neural trajectories of
    trials with distractor and trials without distractor. Produces
    samples for multi-level bootstrap. Parallelized across sessions."""
    parser = argparse.ArgumentParser()
    parser.add_argument("kernel_width", type=int)
    args = parser.parse_args()
    print(f"Running bootstrap distr distance for args: {args.kernel_width}")

    # set up saving and storage
    sessions = get_session_ids()
    results_path = get_results_path()

    savename = f"distance_distr_{args.kernel_width}.nc"
    savepath = os.path.join(results_path, savename)

    result = create_bs_distance_storage()
    session_id_map = get_session_id_map()

    # bootstrap ids (bsids) track which session to draw trials from in
    # first level of multilevel bootstrap
    bsids = result.bsid.data
    n_timepts = result.timepts.data.size

    n_cores = 8
    pool = mp.Pool(n_cores)

    func = partial(
        bootstrap_session,
        session_id_map=session_id_map,
        bsids=bsids,
        n_timepts=n_timepts,
        kernel_width=args.kernel_width,
    )

    bs_results = pool.map(func, sessions)

    for i, session in enumerate(sessions):
        bsid = session_id_map[session]
        bsid_filt = result.bsid == bsid

        result.loc[bsid_filt, :, :] = bs_results[i]

    xr_save(result, savepath)


def bootstrap_session(session, session_id_map, bsids, n_timepts, kernel_width):
    """
    Produce bootstrap estimates of distractor vs no distractor
    distances for a session.

    Args:
        session: (str) Session id.
        session_id_map: (dict) Mapping from session id (str) to
            (int) used for tracking multi-level bootstrap samples.
        bsids: (ndarray) Bootstrap session ids used for tracking
            multi-level bootstrap samples.
        n_timepts: (int) Length of distance array that will be calculated.
        kernel_width: (float) Width of Gaussian kernel for smoothing
            spike trains. Units are in milliseconds.

    Returns:
        ssn_result: (xarray DataArray) Bootstrap distance results.
    """
    n_bs_draws = (bsids == session_id_map[session]).sum()
    kw = kernel_width * 1e-3

    # load session data
    info_keys = ["sample", "isDistractorShown", "delayLength"]
    spike_times, trial_info = load_session_data(session, info_keys)

    delays = np.unique(trial_info["delayLength"])

    samples = np.array([x for x in range(1, 9)])

    results = np.zeros((n_bs_draws, delays.size, n_timepts))

    for d, delay in enumerate(delays):
        bs_distances = np.zeros((n_bs_draws, samples.size, n_timepts))

        for s, samp in enumerate(samples):

            # select
            subset_df_distr = trial_info[
                (trial_info["isDistractorShown"] == False)
                & (trial_info["sample"] == samp)
                & (trial_info["delayLength"] == delay)
            ]
            subset_df_nodistr = trial_info[
                (trial_info["isDistractorShown"] == True)
                & (trial_info["sample"] == samp)
                & (trial_info["delayLength"] == delay)
            ]

            distr_spike_times = spike_times[subset_df_distr.index, :]
            nodistr_spike_times = spike_times[subset_df_nodistr.index, :]

            smoothed_distr, _ = spk.density(distr_spike_times, width=kw, lims=[-2, 5])
            smoothed_nodistr, _ = spk.density(
                nodistr_spike_times, width=kw, lims=[-2, 5]
            )

            for b in range(n_bs_draws):
                if b % 200 == 0:
                    print(
                        f"Completed {kernel_width} / {session} / {delay} / {samp} / {b}"
                    )

                # resample with replacement (over trials) and compute distances
                bs_draw = draw_bs_dist_distr(smoothed_distr, smoothed_nodistr)
                bs_distances[b, s, :] = bs_draw

        results[:, d, :] = bs_distances.mean(1)

    return results


if __name__ == "__main__":
    run()
