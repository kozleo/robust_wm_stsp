import numpy as np
import xarray as xr

from stabda.utils import get_session_ids, get_delay_times


def gen_bootstrap_info(n_bs=1000):
    """Generate bootstrap information for multi-level bootstrap
    construction of confidence intervals (distance experiments).

    Args:
        n_bs: (int) number of bootstrap samples to use

    Returns:
        bs_mat: (ndarray) Matrix containing session id ints (0-5)
            sampled with replacement. Size is (n_bs, n_sessions).
        session_id_key: (dict) Mapping from session_id (str) to int (0-5)
    """

    sessions = get_session_ids()
    session_id_key = {k: v for v, k in enumerate(sessions)}

    bsids = np.array(list(session_id_key.values()))

    bs_mat = np.random.choice(bsids, (n_bs, bsids.size))

    return bs_mat, session_id_key


def draw_bs_dist_sample_id(smoothed_data, sample_id_inds, pairs_list):
    """
    Creates a bootstrap sample of the mean distance between trajectories with
    different sample ids.

    Args:
        smoothed_data: (ndarray) Numpy array containing smoothed firing rates.
            Dims are (trials, neurons, time).
        sample_id_inds: (dict) Mapping from sample ids (1-8) to corresponding
            trial indicies in smoothed_data.
        pairs_list: (list) List of lists containing unique sample pairs.
            E.g. an entry may be [1,8]

    Returns:
        mean_distance: (ndarray) Bootstrap sample of ean distance between
            neural trajectories across sample pairs.
    """
    n_pairs = len(pairs_list)
    bs_draw_inds = {}
    sample_ids = list(sample_id_inds.keys())
    n_timepts = smoothed_data.shape[2]

    # resample trial inds with replacement (bootstrap sample)
    for k, v in sample_id_inds.items():
        bs_draw_inds[k] = np.random.choice(v, v.size)

    pairwise_distances = np.zeros((n_pairs, n_timepts))
    trial_means = {}

    # get trial means per sample_id
    for sample_id in sample_ids:
        sample_id_inds = bs_draw_inds[sample_id]
        sample_id_trials = smoothed_data[sample_id_inds, :]
        trial_means[sample_id] = sample_id_trials.mean(0)

    # compute distances between pairs of sample_id means
    for i, pair in enumerate(pairs_list):
        sample1_id = pair[0]
        sample2_id = pair[1]

        sample1_mean = trial_means[sample1_id]
        sample2_mean = trial_means[sample2_id]

        distances = np.sqrt(np.linalg.norm(sample1_mean - sample2_mean, axis=0))
        pairwise_distances[i, :] = distances

    mean_distance = pairwise_distances.mean(0)

    return mean_distance


def draw_bs_dist_distr(d_smooth, nd_smooth):
    """
    Creates a bootstrap sample of the distance between trajectories of
        trials with distractor and trials without distractor.

    Args:
        d_smooth: (ndarray) Numpy array of firing rates for trials with distractor.
            Dims are (trials, neurons, time).
        nd_smooth: (ndarray) Numpy array of firing rates for trials without distractor.
            Dims are (trials, neurons, time).

    Returns:
        distance: (ndarray) Bootstrap sample of distance between neural trajectories
            with distractor and without distractor.
    """
    bs_d_filt = np.random.choice(d_smooth.shape[0], d_smooth.shape[0])
    bs_nd_filt = np.random.choice(nd_smooth.shape[0], nd_smooth.shape[0])

    distr = d_smooth[bs_d_filt, :]
    no_distr = nd_smooth[bs_nd_filt, :]

    distance = np.sqrt(np.linalg.norm(distr.mean(0) - no_distr.mean(0), axis=0))

    return distance


def create_bs_distance_storage():
    """Create xarray for storing bootstrap distance results."""
    bs_mat, session_id_key = gen_bootstrap_info()

    bsids = bs_mat.flatten()
    n_bs = bsids.size

    delays = get_delay_times()
    n_delays = delays.size

    timepts = np.arange(-2, 5 + 1e-3, 1e-3)
    n_timepts = timepts.size

    res = xr.DataArray(
        np.zeros((n_bs, n_delays, n_timepts)),
        dims=["bsid", "delay", "timepts"],
        coords={
            "bsid": bsids,
            "delay": delays,
            "timepts": timepts,
        },
    )

    res.attrs["bs_mat"] = bs_mat.flatten()

    return res
