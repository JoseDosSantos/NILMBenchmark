import logging
import pandas as pd
import time
import json
import tensorflow as tf
from models.neural_network_architectures import create_model, load_model
from data_feeder import TestSlidingWindowGenerator
import matplotlib.pyplot as plt
from utils import *


class Tester:

    """ Used to test and evaluate a pre-trained seq2point model with or without pruning applied. 
    
    Parameters:
    __appliance (string): The target appliance.
    __algorithm (string): The (pruning) algorithm the model was trained with.
    __network_type (string): The architecture of the model.
    __crop (int): The maximum number of rows of data to evaluate the model with.
    __batch_size (int): The number of rows per testing batch.
    __window_size (int): The size of eaech sliding window
    __window_offset (int): The offset of the inferred value from the sliding window.
    __test_directory (string): The directory of the test file for the model.
    
    """

    def __init__(
            self,
            appliance,
            algorithm,
            crop,
            batch_size,
            network_type,
            test_directory,
            saved_model_dir,
            log_file_dir,
            input_window_length,
            output_length,
            use_weather,
            use_occupancy,
            plot_first,
            dataset,
            return_time=False,
            return_predictions=False
    ):
        self.__appliance = appliance
        self.__algorithm = algorithm
        self.__network_type = network_type
        self.__use_weather = use_weather
        self.__use_occupancy = use_occupancy
        self.__n_cols = 1 + self.__use_weather * 2 + self.__use_occupancy * 1

        self.__crop = crop
        self.__batch_size = batch_size
        self.__input_window_length = input_window_length
        self.__window_offset = self.__input_window_length // 2
        self.__odd_input = self.__input_window_length % 2 == 1
        self.__output_length = output_length

        self.__test_directory = test_directory
        self.__saved_model_dir = saved_model_dir
        self.__dataset = dataset

        self.__log_file = log_file_dir

        # In Python 3.8 we can just add force=True to the basic config, but project is written in 3.7
        # so clear and reset path manually (there's probably a better way)
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(filename=log_file_dir, format='%(message)s', level=logging.INFO)

        self.__plot_first = plot_first
        self.__return_time = return_time
        self.__return_predictions = return_predictions

    def test_model(self):

        """ Tests a fully-trained model using a sliding window generator as an input. Measures inference time, gathers,
        and plots evaluation metrics. """

        start_time = time.time()

        # This is necessary to make the code work with Tensorflow 2.
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)

        test_input, test_target = self.load_dataset(self.__test_directory)
        model = create_model(self.__input_window_length, self.__n_cols, self.__network_type)
        model = load_model(
            model,
            self.__network_type,
            self.__algorithm,
            self.__appliance,
            self.__saved_model_dir
        )

        test_generator = TestSlidingWindowGenerator(
            number_of_windows=self.__batch_size,
            inputs=test_input,
            targets=test_target,
            offset=self.__window_offset,
            odd_input=self.__odd_input
        )

        # Calculate the optimum steps per epoch.
        steps_per_test_epoch = np.round(
            int(test_generator.max_number_of_windows / self.__batch_size),
            decimals=0
        ) + 1

        # Test the model.
        start_time = time.time()
        testing_history = model.predict(x=test_generator.load_dataset(), steps=steps_per_test_epoch, verbose=2)
        end_time = time.time()

        if self.__output_length > 1:
            testing_history = merge_overlapping_predictions(testing_history)

        self.log_results(test_target, testing_history)
        if self.__plot_first != -1:
            self.plot_results(testing_history, test_input, test_target)

        if self.__return_predictions:
            if self.__return_time:
                return denormalize(readings=testing_history, name=self.__appliance), end_time - start_time
            else:
                return denormalize(readings=testing_history, name=self.__appliance)
        if self.__return_time:
            return end_time - start_time

    def load_dataset(self, directory):
        """Loads the testing dataset from the location specified by file_name.

        Parameters:
        directory (string): The location at which the dataset is stored, concatenated with the file name.

        Returns:
        test_input (numpy.array): The first n (crop) features of the test dataset.
        test_target (numpy.array): The first n (crop) targets of the test dataset.

        """
        # Returns an array of the content from the CSV file.
        data_frame = pd.read_csv(directory, header=0, nrows=self.__crop)
        test_target = np.array(data_frame[self.__appliance])

        test_input = np.array(data_frame["mains"])

        if self.__use_occupancy:
            test_input = np.stack((test_input, np.array(data_frame["occupied"])))

        # Transpose and vstack because the weather array is 2D
        if self.__use_weather:
            test_input = np.vstack((test_input, np.array(data_frame[["temperature", "humidity"]]).T))

        del data_frame
        return test_input, test_target

    def log_results(self, y_true, y_pred):

        """Logs the inference time, MAE and MSE of an evaluated model.

        Parameters:
        model (tf.keras.Model): The evaluated model.
        test_time (float): The time taken by the model to infer all required values.
        evaluation metrics (list): The MSE, MAE, and various compression ratios of the model.

        """

        test_log = f"Test dataset:{self.__dataset}"\
                   f"Window size: {self.__input_window_length} "\
                   f"Weather: {self.__use_weather} "\
                   f"Occupancy: {self.__use_occupancy} "
        logging.info(test_log)

        # If we predict an output sequence, the difference in length will not be equal to the window offset, so
        # we recalculate here

        trim_offset = abs(len(y_true) - len(y_pred)) // 2

        y_true = denormalize(readings=y_true, name=self.__appliance)
        y_pred = denormalize(readings=y_pred, name=self.__appliance)

        mse = mean_squared_error(y_true=y_true, y_pred=y_pred, offset=trim_offset)
        mae = mean_absolute_error(y_true=y_true, y_pred=y_pred, offset=trim_offset)
        sae = normalised_signal_aggregate_error(y_true=y_true, y_pred=y_pred, offset=trim_offset)
        mr = match_rate(y_true=y_true, y_pred=y_pred, offset=trim_offset)

        metric_string = f"MSE: {mse}" \
                        f" MAE: {mae}" \
                        f" SAE: {sae}" \
                        f" Match Rate: {mr}\n"
        logging.info(metric_string)



    def plot_results(self, testing_history, test_input, test_target):

        """ Generates and saves a plot of the testing history of the model against the (actual) 
        aggregate energy values and the true appliance values.

        Parameters:
        testing_history (numpy.ndarray): The series of values inferred by the model.
        test_input (numpy.ndarray): The aggregate energy data.
        test_target (numpy.ndarray): The true energy values of the appliance.

        """

        if len(test_input.shape) > 1:
            test_agg = denormalize(test_input[0], "mains")
        else:
            test_agg = denormalize(test_input.flatten(), "mains")

        test_target = denormalize(test_target, self.__appliance)
        testing_history = denormalize(testing_history, self.__appliance)

        trim_offset = abs(len(test_target) - len(testing_history)) // 2
        if trim_offset > 0:
            test_agg = test_agg[trim_offset: -trim_offset]
            test_target = test_target[trim_offset:-trim_offset]


        # Plot testing outcomes against ground truth.
        plt.figure(1)
        plt.plot(test_agg[:self.__plot_first], label="Aggregate")
        plt.plot(test_target[:self.__plot_first], label="Ground Truth")
        plt.plot(testing_history[:self.__plot_first], label="Predicted")
        plt.title(f"{self.__appliance}  {self.__network_type} (Window: {self.__input_window_length}")
        plt.ylabel("Power Value (Watts)")
        plt.xlabel("Testing Window")
        plt.legend()

        # file_path = "./" + self.__appliance + "/saved_models/" + self.__appliance + "_" + self.__algorithm + "_" + self.__network_type + "_test_figure.png"
        # plt.savefig(fname=file_path)

        plt.show()
