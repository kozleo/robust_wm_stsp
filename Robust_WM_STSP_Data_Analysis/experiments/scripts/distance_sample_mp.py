import os
import numpy as np
import argparse
import itertools as it
import multiprocessing as mp

from functools import partial

import neural_analysis.spikes as spk

from stabda.bootstrap import draw_bs_dist_sample_id, create_bs_distance_storage
from stabda.utils import get_results_path, get_session_ids, xr_save, get_session_id_map
from stabda.data import load_session_data


def run():
    """Run experiment calculating mean distance between neural trajectories with
    different sample identities. Produces samples for multi-level bootstrap.
    Parallelized across sessions."""
    parser = argparse.ArgumentParser()
    parser.add_argument("kernel_width", type=int)
    args = parser.parse_args()
    print(f"Running bootstrap sample distance for args: {args.kernel_width}")

    # set up saving and storage
    sessions = get_session_ids()
    results_path = get_results_path()

    savename = f"distance_sample_{args.kernel_width}.nc"
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
    Produce bootstrap estimates of mean distance between neural trajectories
    with different sample identities for a given session.

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

    sample_ids = [x for x in range(1, 9)]
    pairs = it.combinations(sample_ids, 2)
    pairs_list = list(pairs)

    results = np.zeros((n_bs_draws, delays.size, n_timepts))

    for d, delay in enumerate(delays):
        print(f"Running {session}, delay: {delay}")
        subset_df = trial_info[
            (trial_info["isDistractorShown"] == False)
            & (trial_info["delayLength"] == delay)
        ]
        subset_spike_times = spike_times[subset_df.index, :]
        smoothed, _ = spk.density(subset_spike_times, width=kw, lims=[-2, 5])

        # create sample_id -> sample_inds dict for given delay
        sample_id_inds = {}
        subset_df = subset_df.reset_index()
        for i in sample_ids:
            sample_id_inds[i] = subset_df[subset_df["sample"] == i].index

        # resample with replacement (over trials) and compute distances
        for b in range(n_bs_draws):
            if b % 200 == 0:
                print(f"Completed {kernel_width} / {session} / {delay} / {b}")
            bs_draw = draw_bs_dist_sample_id(smoothed, sample_id_inds, pairs_list)
            results[b, d, :] = bs_draw

    return results


if __name__ == "__main__":
    run()
