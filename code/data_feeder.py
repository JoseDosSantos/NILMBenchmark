import numpy as np 
import pandas as pd 

# batch_size: the number of rows fed into the network at once.
# crop: the number of rows in the data set to be used in total.
# chunk_size: the number of lines to read from the file at once.


class TrainSlidingWindowGenerator:

    """Yields features and targets for training a ConvNet.

    Parameters:
    __file_name (string): The path where the training dataset is located.
    __batch_size (int): The size of each batch from the dataset to be processed.
    __chunk_size (int): The size of each chunk of data to be processed.
    __shuffle (bool): Whether the dataset should be shuffled before being returned.
    __offset (int):
    __crop (int): The number of rows of the dataset to return.
    __skip_rows (int): The number of rows of a dataset to skip before reading data.
    __ram_threshold (int): The maximum amount of RAM to utilise at a time.
    total_size (int): The number of rows read from the dataset.

    """

    def __init__(
            self,
            file_name,
            appliance,
            chunk_size,
            shuffle,
            offset,
            odd_input,
            output_length=1,
            use_weather=False,
            use_occupancy=False,
            batch_size=1000,
            crop=None,
            skip_rows=0,
            ram_threshold=5 * 10 ** 5
    ):
        self.__file_name = file_name
        self.__appliance = appliance
        self.__batch_size = batch_size
        self.__chunk_size = chunk_size
        self.__shuffle = shuffle
        self.__offset = offset
        self.__output_length = output_length
        self.__odd_input = odd_input
        self.__use_weather = use_weather
        self.__use_occupancy = use_occupancy
        self.__crop = crop
        self.__skip_rows = skip_rows
        self.__ram_threshold = ram_threshold
        self.total_size = 0
        self.__total_num_samples = crop

        if self.total_size == 0:
            self.check_if_chunking()

        self.n_cols = 1 + self.__use_weather * 2 + self.__use_occupancy * 1

    @property
    def total_num_samples(self):
        if self.__crop:
            return self.__total_num_samples
        else:
            return self.total_size - 2 * self.__offset
    
    @total_num_samples.setter
    def total_num_samples(self, value):
        self.__total_num_samples = value

    def check_if_chunking(self):

        """Count the number of rows in the dataset and determine whether this is larger than the chunking 
        threshold or not. """

        # Loads the file and counts the number of rows it contains.
        print("Importing training file...")
        chunks = pd.read_csv(self.__file_name, header=0, nrows=self.__crop)
        print("Counting number of rows...")
        self.total_size = len(chunks)
        del chunks
        print("Done.")

        print("The dataset contains ", self.total_size, " rows")

        # Display a warning if there are too many rows to fit in the designated amount RAM.
        if self.total_size > self.__ram_threshold:
            print("There is too much data to load into memory, so it will be loaded in chunks. Please note that this "
                  "may result in decreased training times.")
    
    def load_dataset(self):

        """Yields pairs of features and targets that will be used directly by a neural network for training.

        Yields:
        input_data (numpy.array): A 1D array of size batch_size containing features of a single input. 
        output_data (numpy.array): A 1D array of size batch_size containing the target values corresponding to 
        each feature set.

        """


        # If we use evenly sized windows with odd input lengths, we run into problems when trying to predict the center
        # point of odd observations and vice versa. Therefore we only allow both odd (indicated by self.__odd_input) or
        # or both even windows. This is not the most concise code, but it does the job.

        if self.__odd_input:
            if self.__output_length % 2 == 0:
                raise ValueError("Odd input length with even output window detected. Currently, only odd length "
                                 "output windows are supported with odd input windows.")
        else:
            if self.__output_length % 2 == 1:
                raise ValueError("Even input length with odd output window detected. Currently, only eben length "
                                 "output windows are supported with even input windows.")

        out_offset = self.__offset - self.__output_length // 2
        odd_offset = int(self.__odd_input)


        # Returns an array of the content from the CSV file.
        data_frame = pd.read_csv(self.__file_name, header=0, nrows=self.__crop)

        inputs = np.array(data_frame["mains"])
        outputs = np.array(data_frame[self.__appliance])

        if self.__use_occupancy:
            inputs = np.stack((inputs, np.array(data_frame["occupied"])))

        # Transpose and vstack because the weather array is 2D
        if self.__use_weather:
            inputs = np.vstack((inputs, np.array(data_frame[["temperature", "humidity"]]).T))

        del data_frame

        inputs = inputs.T

        maximum_batch_size = len(inputs) - 2 * self.__offset

        if self.__batch_size < 0:
            self.__batch_size = maximum_batch_size

        indices = np.arange(maximum_batch_size)
        if self.__shuffle:
            np.random.shuffle(indices)

        while True:
            for start_index in range(0, maximum_batch_size, self.__batch_size):
                splice = indices[start_index:start_index + self.__batch_size]
                input_data = np.array([inputs[index:index + 2 * self.__offset + odd_offset] for index in splice])
                output_data = np.array(
                    [outputs[index + out_offset:index + out_offset + self.__output_length + odd_offset]
                     for index in splice])
                yield input_data, output_data




class TestSlidingWindowGenerator(object):

    """Yields features and targets for testing and validating a ConvNet.

    Parameters:
    __number_of_windows (int): The number of sliding windows to produce.
    __offset (int): The offset of the infered value from the sliding window.
    __inputs (numpy.ndarray): The available testing / validation features.
    __targets (numpy.ndarray): The target values corresponding to __inputs.
    __total_size (int): The total number of inputs.

    """

    def __init__(self, number_of_windows, inputs, targets, offset, odd_input):
        self.__number_of_windows = number_of_windows
        self.__offset = offset
        self.__inputs = inputs
        self.__targets = targets
        self.__odd_input = odd_input
        self.total_size = max(inputs.shape)
        self.max_number_of_windows = self.total_size - 2 * self.__offset

    def load_dataset(self):

        """Yields features and targets for testing and validating a ConvNet.

        Yields:
        input_data (numpy.array): An array of features to test / validate the network with.

        """

        self.__inputs = self.__inputs.T
        #max_number_of_windows = self.__inputs.size - 2 * self.__offset

        if self.__number_of_windows < 0:
            self.__number_of_windows = self.max_number_of_windows

        odd_offset = int(self.__odd_input)

        indices = np.arange(self.max_number_of_windows, dtype=int)
        for start_index in range(0, self.max_number_of_windows, self.__number_of_windows):
            splice = indices[start_index:start_index + self.__number_of_windows]
            input_data = np.array([self.__inputs[index:index + 2 * self.__offset + odd_offset] for index in splice])
            target_data = self.__targets[splice + self.__offset].reshape(-1, 1)
            yield input_data, target_data

