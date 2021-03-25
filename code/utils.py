import numpy as np
import json
import yaml
import os

def read_yaml(x: str) -> dict:
    """
    Read yaml files and return content as dictionary
        x: name of the file
    """
    with open(x, "r") as con:
        config = yaml.safe_load(con)
    return config


def check_dir(path: str) -> None:
    # Create target Directory if don't exist
    if not os.path.exists(path):
        os.mkdir(path)


def get_model_save_path(network_type, interval, device, window_length, use_weather, use_occupancy) -> str:
    check_dir(f"F://training_outputs/saved_models/{device}")
    save_model_dir = f"F://training_outputs//saved_models/{device}/{network_type}_{interval}_{window_length}"
    if use_weather:
        save_model_dir += "_weather"
    if use_occupancy:
        save_model_dir += "_occupancy"
    return save_model_dir + ".h5"

def save_dict(data, path):
    with open(f"{path}.json", 'w') as f:
        json.dump(data, f, indent=4)


def clip(arr):
    arr[arr < 0] = 0
    return arr


def mean_squared_error(y_true, y_pred, offset=None, clip_values=False):
    if clip_values:
        y_true = clip(y_true)
        y_pred = clip(y_pred)

    if offset:
        y_true = y_true[offset:-offset]

    assert y_true.shape == y_pred.shape
    return np.mean(np.power((y_true - y_pred), 2))


def mean_absolute_error(y_true, y_pred, offset=None, clip_values=False):
    if clip_values:
        y_true = clip(y_true)
        y_pred = clip(y_pred)

    if offset:
        y_true = y_true[offset:-offset]
    assert y_true.shape == y_pred.shape

    return np.mean(np.abs(y_true - y_pred))


def normalised_signal_aggregate_error(y_true, y_pred, offset=None, clip_values=False):
    if clip_values:
        y_true = clip(y_true)
        y_pred = clip(y_pred)

    if offset:
        y_true = y_true[offset:-offset]

    assert y_true.shape == y_pred.shape

    r = np.sum(y_true)
    r_hat = np.sum(y_pred)
    signal_aggregate_error = np.abs(r_hat - r) / r
    return signal_aggregate_error


def match_rate(y_true, y_pred, offset=None, clip_values=False):
    if clip_values:
        y_true = clip(y_true)
        y_pred = clip(y_pred)


    if offset:
        y_true = y_true[offset:-offset]

    assert y_true.shape == y_pred.shape


    minimum = np.sum(np.minimum(y_true, y_pred))
    maximum = np.sum(np.maximum(y_true, y_pred))
    return minimum / maximum


def denormalize(readings, name, path="../stats_DRED_1min.json"):
    with open(path) as file:
        stats = json.load(file)
    read_norm = readings * stats["std"][name] + stats["mean"][name]

    # Can't have negative energy readings - set any results below 0 to 0.
    read_norm[read_norm < 0] = 0

    # To make sure arrays have the same shape
    read_norm = np.array(read_norm).reshape((-1, 1))
    return read_norm


def merge_overlapping_predictions(arr):
    # If we use overlapping test batches to predict we receive overlapping predictions. This function returns a merged
    # array (where overlapping regions are averaged)
    arr = np.squeeze(arr)
    length, window_size = arr.shape
    out = np.zeros(shape=(length+window_size, 1))

    norm_arr = np.concatenate(
        [np.linspace(1, window_size, num=window_size),
         np.repeat(window_size, length-window_size),
         np.linspace(window_size, 1, num=window_size)]
    ).reshape((-1, 1))

    for idx in range(length):
        out[idx:idx+window_size] += arr[idx].reshape((-1, 1))
    out /= norm_arr
    return out


