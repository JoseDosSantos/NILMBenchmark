#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from typing import Union


class DataStreamerNilm:
    """Returns batches of a given dataset.

    Takes a given dataset, optionally enriches it with additional data and 
    returns an iterator over that dataset with the given batch size. Note that
    this function applies no preprocessing, so the input data needs to be 
    processed beforehand.
    """

    def __init__(
        self,
        dataset,
        mains_col: str,
        appliance_cols: Union[str, list],
        batch_size: int = 8192,
        window_size: int = 1,
        shuffle: bool = False,
        chunksize: int = -1,
        random_state: int = None
    ):
        """Initialize NILM data streamer.

            Args:
            dataset: pd.DataFrame of mains and appliance data.
              TODO: Load file from disk.
            mains_col: Name of the columns containing the mains readings.
            appliance_col: Either single name or list of appliance names to 
              return.
            batch_size: Number of datapoints returned.
            window_size: In case sequential training data is needed, each 
              batch item consists of a time window with given length. Leave at 
              1 to return independent singular observations.
            shuffle: Shuffle data before yielding. If window length is given,
              the data is first split into window-sized continuous chunks and
              then shuffled to preserve order.
            chunksize: Currently not implemented. Number of observations to
              load from disk.
              TODO: If file is loaded from memory, enable chunkwise loading.
            random_state: Use to get reproducable shuffling results.

        Yields:
            An iterable over the input dataset.
        """

        self.mains_col = mains_col
        self.appliance_cols = appliance_cols
        self.batch_size = batch_size
        self.window_size = window_size
        self.shuffle = shuffle
        self.chunksize = chunksize
        self.random_state = random_state
        if self.random_state:
            np.random.seed(self.random_state)
        
        
        # We only need to keep mains + selected appliances in memory
        if type(appliance_cols) is str:
            self.dataset = dataset.filter([mains_col, appliance_cols])
        else:
            self.dataset = dataset.filter([mains_col] + appliance_cols)
        
        self.reset_iterator(self.dataset)
        
    def generate_batch(self):
        target, features = next(self.dataset_iterator)
        return target, features


    def _dataset_iterator(self, data):
        """
        Expects 
        """
        for batch in data:
            yield batch


    def reset_iterator(self, data: pd.DataFrame) -> None:
        """Reset data streamer and empty sample cache"""
        df_length_original, n_cols = data.shape

        if self.window_size > 1:
            # A bit hacky, but to make the reshape work we cut off a small part
            # at the end so the dataset nicely divides into window_sized parts
            cutoff = df_length_original % self.window_size
            if cutoff > 0:
                data = data[:-cutoff]
        df_length = data.shape[0]
        n_splits = df_length // self.window_size

        # Reshape the data into window_sized parts
        data = data.to_numpy().reshape((n_splits, self.window_size, n_cols))

        if self.shuffle:
            np.random.shuffle(data)
        
        # There might be a better way to make sure the data exactly divides into
        # the given amount of batches, but probably not an issue with sufficient
        # training samples.
        batch_cutoff = n_splits % self.batch_size
        if batch_cutoff > 0:
            data = data[:-batch_cutoff]
        
        # Now separate the shuffled and windowed observations into target and
        # feature lists.
        # TODO: Maybe this step can be done before and both lists can instead
        # be shuffled separately with same seeds.
        target_list = []
        feature_list = []
        for window in data:
            target, features = np.hsplit(window,[1])
            target_list.append(target)
            feature_list.append(features)
        
        # Finally split the data into batches, consisting of a list of target
        # windows and a list of corresponding feature windows.
        n_batches = len(target_list) // self.batch_size
        batches = []
        
        # TODO: Create batch-indexes in a nicer way
        for i in range(n_batches):
            batches.append([target_list[i*self.batch_size:i*self.batch_size+self.batch_size],
                            feature_list[i*self.batch_size:i*self.batch_size+self.batch_size]])
        
        self.dataset_iterator = self._dataset_iterator(batches)