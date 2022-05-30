import pickle

def pickle_load(file):
    """Open pickle file."""
    with open(file, "rb") as handle:
        data = pickle.load(handle)
    return data

def pickle_save(data, save_name):
    """Save object as pickle file."""
    with open(save_name, "wb") as handle:
        pickle.dump(data, handle)


def remove_all_ticks(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
