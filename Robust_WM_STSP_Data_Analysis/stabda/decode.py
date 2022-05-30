import numpy as np
import xarray as xr

from sklearn.model_selection import KFold
from sklearn.svm import SVC

from stabda.utils import get_delay_times


def run_svc(y, X, svc_type="linear", n_split=10, C=1):
    """
        Train SVM classifier on data and return test accuracy.

    Args:
        y: (ndarray) Labels array with dims (samples,)
        X: (ndarray) Covariates with dims (samples, features);
            features are typically some function of neural data
        svc_type: (str) Option for SVC type. Options are "linear" or "rbf".
            Default is "linear".
        n_split: (int) Number of splits for cross-validation. Default is 10.
        C: (float) Regularization parameter for SVC. Default is 1.

    Returns:
        res_test: (float) Resulting test accuracy of trained classifier on test
            data.
    """
    kf = KFold(n_splits=n_split, shuffle=True, random_state=0)
    kf.get_n_splits(X, y)
    y_true = []
    y_pred = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        if svc_type == "linear":
            clf = SVC(kernel="linear", C=C)
        if svc_type == "rbf":
            clf = SVC(C=C)
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)

        y_true.append(y_test)
        y_pred.append(pred)

    true = np.concatenate(y_true)
    pred = np.concatenate(y_pred)

    test_acc = (pred == true).sum() / true.size
    return test_acc


def create_decode_storage(timepts):
    """Create xarray for storing decoding results."""
    has_distr = np.array([0, 1])
    n_distr_options = has_distr.size

    delays = get_delay_times()
    n_delays = delays.size

    n_timepts = timepts.size

    res = xr.DataArray(
        np.zeros((n_delays, n_distr_options, n_timepts)),
        dims=["delay", "has_distr", "timepts"],
        coords={"delay": delays, "has_distr": has_distr, "timepts": timepts},
    )

    return res
