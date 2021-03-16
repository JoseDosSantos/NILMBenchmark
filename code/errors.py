import numpy as np


def clip(arr):
    arr[arr < 0] = 0
    return arr


def mean_squared_error(y_true, y_pred, offset=None, clip_values=True):
    if clip_values:
        y_true = clip(y_true)
        y_pred = clip(y_pred)
    return np.mean(np.power((y_true[offset:offset] - y_pred), 2))


def mean_absolute_error(y_true, y_pred, offset=None, clip_values=True):
    if clip_values:
        y_true = clip(y_true)
        y_pred = clip(y_pred)
    return np.mean(np.abs(y_true[offset:-offset] - y_pred))


def normalised_signal_aggregate_error(y_true, y_pred, offset=None, clip_values=True):
    if clip_values:
        y_true = clip(y_true)
        y_pred = clip(y_pred)

    r = np.sum(y_true[offset:-offset])
    r_hat = np.sum(y_pred)
    signal_aggregate_error = np.abs(r_hat - r) / r
    return signal_aggregate_error


def match_rate(y_true, y_pred, offset=None, clip_values=True):
    if clip_values:
        y_true = clip(y_true)
        y_pred = clip(y_pred)

    minimum = np.sum(np.minimum(y_true[offset:-offset], y_pred))
    maximum = np.sum(np.maximum(y_true[offset:-offset], y_pred))
    return minimum / maximum
