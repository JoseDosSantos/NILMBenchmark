import numpy as np
import json


def clip(arr):
    arr[arr < 0] = 0
    return arr


def mean_squared_error(y_true, y_pred, offset=None, clip_values=False):
    if clip_values:
        y_true = clip(y_true)
        y_pred = clip(y_pred)

    if offset:
        y_true = y_true[offset:-offset]

    return np.mean(np.power((y_true - y_pred), 2))


def mean_absolute_error(y_true, y_pred, offset=None, clip_values=False):
    if clip_values:
        y_true = clip(y_true)
        y_pred = clip(y_pred)

    if offset:
        y_true = y_true[offset:-offset]

    return np.mean(np.abs(y_true - y_pred))


def normalised_signal_aggregate_error(y_true, y_pred, offset=None, clip_values=False):
    if clip_values:
        y_true = clip(y_true)
        y_pred = clip(y_pred)

    if offset:
        y_true = y_true[offset:-offset]

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

    minimum = np.sum(np.minimum(y_true, y_pred))
    maximum = np.sum(np.maximum(y_true, y_pred))
    return minimum / maximum


def denormalize(readings, name):
    with open("../stats_DRED_1min.json") as file:
        stats = json.load(file)
    read_norm = readings * stats["std"][name] + stats["mean"][name]

    # Can't have negative energy readings - set any results below 0 to 0.
    read_norm[read_norm < 0] = 0

    # To make sure arrays have the same shape
    read_norm = np.array(read_norm).reshape((-1, 1))
    return read_norm
