import os

import pickle
import numpy as np
import xarray as xr


def get_proj_path():
    current_path = os.path.dirname(os.path.abspath(__file__))
    project_path = os.path.split(current_path)[0]

    return project_path


def get_data_path():
    project_path = get_proj_path()
    data_dir = "datasets"

    return os.path.join(project_path, data_dir)


def get_results_path():
    project_path = get_proj_path()
    results_dir = os.path.join("experiments", "results")

    return os.path.join(project_path, results_dir)


def xr_load(nc_path):
    """Load xarray DataArray object from netCDF save file."""
    xarr = xr.open_dataarray(nc_path)
    xarr.load()
    return xarr


def xr_save(xarr, nc_path):
    """Save xarray DataArray object to netCDF save file."""
    xarr.to_netcdf(nc_path)


def pickle_load(file):
    """Open pickle file."""
    with open(file, "rb") as handle:
        data = pickle.load(handle)
    return data


def pickle_save(data, save_name):
    """Save object as pickle file."""
    with open(save_name, "wb") as handle:
        pickle.dump(data, handle)


def get_session_ids():
    """Load NHP experiment session IDs."""
    return np.array(["1203", "1204", "1205", "1206", "1207", "1210"])


def get_session_id_map():
    """Load dictionary containing session ids (keys) to integer values (values)
    used for tracking multilevel bootstrap."""
    session_ids = np.array(["1203", "1204", "1205", "1206", "1207", "1210"])
    session_id_ints = np.array([0, 1, 2, 3, 4, 5])

    session_id_map = {session_ids[i]: session_id_ints[i] for i in range(6)}

    return session_id_map


def get_distr_on_times():
    """Load dictionary containing delay times (keys)
    and distractor on times (values) in seconds."""
    return {1.0: 1.0, 1.41: 1.21, 2.0: 1.501, 2.83: 1.921, 4.0: 2.501}


def get_delay_times():
    return np.array([1.0, 1.41, 2.0, 2.83, 4.0])
