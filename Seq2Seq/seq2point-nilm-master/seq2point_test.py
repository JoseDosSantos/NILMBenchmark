import logging
import numpy as np
import pandas as pd
import time
import json
from model_structure import create_model, load_model
from data_feeder import TestSlidingWindowGenerator
import matplotlib.pyplot as plt


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
            use_weather,
            use_occupancy
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
        self.__window_size = self.__input_window_length + 2
        self.__window_offset = int(0.5 * self.__window_size - 1)
        #self.__number_of_windows = 100

        self.__test_directory = test_directory
        self.__saved_model_dir = saved_model_dir

        self.__log_file = log_file_dir
        logging.basicConfig(filename=self.__log_file, level=logging.INFO)

    def test_model(self):

        """ Tests a fully-trained model using a sliding window generator as an input. Measures inference time, gathers,
        and plots evaluation metrics. """

        test_input, test_target = self.load_dataset(self.__test_directory)
        model = create_model(self.__input_window_length, self.__n_cols)
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
            offset=self.__window_offset
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
        test_time = end_time - start_time

        #evaluation_metrics = model.evaluate(x=test_generator.load_dataset(), steps=steps_per_test_epoch)

        self.log_results(model, test_time, test_time, test_target, testing_history)
        self.plot_results(testing_history, test_input, test_target)

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


    def mean_squared_error(self, y_true, y_pred):
        return np.mean(np.power((y_true[self.__window_offset:-self.__window_offset] - y_pred), 2))

    def mean_absolute_error(self, y_true, y_pred):
        return np.mean(np.abs(y_true[self.__window_offset:-self.__window_offset]-y_pred))

    def normalised_signal_aggregate_error(self, y_true, y_pred):
        r = np.sum(y_true[self.__window_offset:-self.__window_offset])
        r_hat = np.sum(y_pred)
        signal_aggregate_error = np.abs(r_hat - r) / r
        return signal_aggregate_error

    def log_results(self, model, test_time, evaluation_metrics, y_true, y_pred):

        """Logs the inference time, MAE and MSE of an evaluated model.

        Parameters:
        model (tf.keras.Model): The evaluated model.
        test_time (float): The time taken by the model to infer all required values.
        evaluation metrics (list): The MSE, MAE, and various compression ratios of the model.

        """

        inference_log = "Inference Time: " + str(test_time)
        logging.info(inference_log)


        mean_squared_error = self.mean_squared_error(
            y_true=self.denormalize(readings=y_true, name=self.__appliance),
            y_pred=self.denormalize(readings=y_pred, name=self.__appliance)
        )

        mean_absolute_error = self.mean_absolute_error(
            y_true=self.denormalize(readings=y_true, name=self.__appliance),
            y_pred=self.denormalize(readings=y_pred, name=self.__appliance)
        )

        signal_aggregate_error = self.normalised_signal_aggregate_error(
            y_true=self.denormalize(readings=y_true, name=self.__appliance),
            y_pred=self.denormalize(readings=y_pred, name=self.__appliance)
        )

        metric_string = f"MSE: {mean_squared_error}" \
                        f" MAE: {mean_absolute_error}" \
                        f" SAE: {signal_aggregate_error}"
        logging.info(metric_string)

        # self.count_pruned_weights(model)

    def count_pruned_weights(self, model):

        """ Counts the total number of weights, pruned weights, and weights in convolutional 
        layers. Calculates the sparsity ratio of different layer types and logs these values.

        Parameters:
        model (tf.keras.Model): The evaluated model.

        """
        num_total_zeros = 0
        num_dense_zeros = 0
        num_dense_weights = 0
        num_conv_zeros = 0
        num_conv_weights = 0
        for layer in model.layers:
            if np.shape(layer.get_weights())[0] != 0:
                layer_weights = layer.get_weights()[0].flatten()

                if "conv" in layer.name:
                    num_conv_weights += np.size(layer_weights)
                    num_conv_zeros += np.count_nonzero(layer_weights == 0)

                    num_total_zeros += np.size(layer_weights)
                else:
                    num_dense_weights += np.size(layer_weights)
                    num_dense_zeros += np.count_nonzero(layer_weights == 0)

        conv_zeros_string = "CONV. ZEROS: " + str(num_conv_zeros)
        conv_weights_string = "CONV. WEIGHTS: " + str(num_conv_weights)
        conv_sparsity_ratio = "CONV. RATIO: " + str(num_conv_zeros / num_conv_weights)

        dense_weights_string = "DENSE WEIGHTS: " + str(num_dense_weights)
        dense_zeros_string = "DENSE ZEROS: " + str(num_dense_zeros)
        dense_sparsity_ratio = "DENSE RATIO: " + str(num_dense_zeros / num_dense_weights)

        total_zeros_string = "TOTAL ZEROS: " + str(num_total_zeros)
        total_weights_string = "TOTAL WEIGHTS: " + str(model.count_params())
        total_sparsity_ratio = "TOTAL RATIO: " + str(num_total_zeros / model.count_params())

        print("LOGGING PATH: ", self.__log_file)

        logging.info(conv_zeros_string)
        logging.info(conv_weights_string)
        logging.info(conv_sparsity_ratio)
        logging.info("")
        logging.info(dense_zeros_string)
        logging.info(dense_weights_string)
        logging.info(dense_sparsity_ratio)
        logging.info("")
        logging.info(total_zeros_string)
        logging.info(total_weights_string)
        logging.info(total_sparsity_ratio)

    def denormalize(self, readings, name):
        with open("stats_DRED_1min.json") as file:
            stats = json.load(file)
        read_norm = readings * stats["std"][name] + stats["mean"][name]

        # Can't have negative energy readings - set any results below 0 to 0.
        read_norm[read_norm < 0] = 0

        # To make sure arrays have the same shape
        read_norm = read_norm.reshape((-1, 1))
        return read_norm

    def plot_results(self, testing_history, test_input, test_target):

        """ Generates and saves a plot of the testing history of the model against the (actual) 
        aggregate energy values and the true appliance values.

        Parameters:
        testing_history (numpy.ndarray): The series of values inferred by the model.
        test_input (numpy.ndarray): The aggregate energy data.
        test_target (numpy.ndarray): The true energy values of the appliance.

        """
        testing_history = self.denormalize(testing_history, self.__appliance)
        test_target = self.denormalize(test_target, self.__appliance)
        test_agg = self.denormalize(test_input.flatten(), "mains")
        test_agg = test_agg[:testing_history.size]

        # testing_history = ((testing_history * stats["std"][self.__appliance])
        #                    + stats["mean"][self.__appliance])
        # test_target = ((test_target * stats["std"][self.__appliance])
        #                    + stats["mean"][self.__appliance])
        # test_agg = (test_input.flatten() * stats["std"]["mains"]) + stats["mean"]["mains"]
        # test_agg = test_agg[:testing_history.size]

        # test_target[test_target < 0] = 0
        # testing_history[testing_history < 0] = 0
        # test_input[test_input < 0] = 0

        # Plot testing outcomes against ground truth.
        plt.figure(1)
        plt.plot(test_agg[self.__window_offset: -self.__window_offset], label="Aggregate")
        plt.plot(test_target[:test_agg.size - (2 * self.__window_offset)], label="Ground Truth")
        plt.plot(testing_history[:test_agg.size - (2 * self.__window_offset)], label="Predicted")
        plt.title(self.__appliance + " " + self.__network_type + "(" + self.__algorithm + ")")
        plt.ylabel("Power Value (Watts)")
        plt.xlabel("Testing Window")
        plt.legend()

        # file_path = "./" + self.__appliance + "/saved_models/" + self.__appliance + "_" + self.__algorithm + "_" + self.__network_type + "_test_figure.png"
        # plt.savefig(fname=file_path)

        plt.show()
