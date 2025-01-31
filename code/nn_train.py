import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
import time

from data_feeder import TrainSlidingWindowGenerator
from models.neural_network_architectures import create_model, save_model
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class Trainer:

    """ Used to train a seq2point model with or without pruning applied Supports 
    various alternative architectures. 
    
    Parameters:
    __appliance (string): The target appliance.
    __network_type (string): The architecture of the model.
    __batch_size (int): The number of rows per testing batch.
    __window_size (int): The size of eaech sliding window
    __window_offset (int): The offset of the inferred value from the sliding window.
    __max_chunk_size (int): The largest possible number of row per chunk.
    __validation_frequency (int): The number of epochs between model validation.
    __training_directory (string): The directory of the model's training file.
    __validation_directory (string): The directory of the model's validation file.
    __training_chunker (TrainSlidingWindowGenerator): A sliding window provider 
    that returns feature / target pairs. For training use only.
    __validation_chunker (TrainSlidingWindowGenerator): A sliding window provider 
    that returns feature / target pairs. For validation use only.
    
    """

    def __init__(
            self,
            appliance,
            batch_size,
            crop,
            network_type,
            training_directory,
            validation_directory,
            save_model_dir,
            use_occupancy=False,
            use_weather=False,
            learning_rate=0.001,
            epochs=10,
            input_window_length=599,
            output_length=1,
            validation_frequency=1,
            early_stopping=True,
            patience=5,
            restore_weights=True,
            min_delta=1e-6,
            verbose=1,
            return_time=False,
            plot_training=False
    ):
        self.__appliance = appliance
        self.__algorithm = network_type
        self.__network_type = network_type
        self.__crop = crop
        self.__batch_size = batch_size
        self.__epochs = epochs
        self.__early_stopping = early_stopping
        self.__patience = patience
        self.__restore_weights = restore_weights
        self.__min_delta = min_delta
        self.__verbose = verbose
        self.__return_time = return_time
        self.__loss = "mse"
        self.__metrics = ["mse", "msle", "mae"]
        self.__learning_rate = learning_rate
        self.__beta_1 = 0.9
        self.__beta_2 = 0.999
        self.__save_model_dir = save_model_dir
        self.__plot_training = plot_training
        self.__use_occupancy = use_occupancy
        self.__use_weather = use_weather
        self.__n_cols = 1 + self.__use_weather * 2 + self.__use_occupancy * 1

        self.__input_window_length = input_window_length
        # self.__window_size = 2 + self.__input_window_length
        self.__output_length = output_length
        self.__odd_input = self.__input_window_length % 2 == 1
        self.__window_offset = self.__input_window_length // 2
        self.__max_chunk_size = 5 * 10 ** 2
        self.__validation_frequency = validation_frequency
        self.__ram_threshold = 5*10**6
        self.__skip_rows_train = 0
        self.__validation_steps = self.__epochs
        self.__skip_rows_val = 0

        # Directories of the training and validation files. Always has the structure 
        # ./dataset_management/refit/{appliance_name}/{appliance_name}_training_.csv for training or 
        # ./dataset_management/refit/{appliance_name}/{appliance_name}_validation_.csv
        self.__training_directory = training_directory
        self.__validation_directory = validation_directory

        self.__training_chunker = TrainSlidingWindowGenerator(
            file_name=self.__training_directory,
            appliance=self.__appliance,
            chunk_size=self.__max_chunk_size,
            batch_size=self.__batch_size,
            crop=self.__crop, shuffle=True,
            skip_rows=self.__skip_rows_train,
            offset=self.__window_offset,
            output_length=self.__output_length,
            odd_input=self.__odd_input,
            ram_threshold=self.__ram_threshold,
            use_occupancy=self.__use_occupancy,
            use_weather=self.__use_weather
        )

        self.__validation_chunker = TrainSlidingWindowGenerator(
            file_name=self.__validation_directory,
            appliance=self.__appliance,
            chunk_size=self.__max_chunk_size,
            batch_size=self.__batch_size,
            crop=self.__crop,
            shuffle=True,
            skip_rows=self.__skip_rows_val,
            offset=self.__window_offset,
            output_length=self.__output_length,
            odd_input=self.__odd_input,
            ram_threshold=self.__ram_threshold,
            use_occupancy=self.__use_occupancy,
            use_weather=self.__use_weather
        )

    def train_model(self):

        """ Trains an energy disaggregation model using a user-selected pruning algorithm (default is no pruning). 
        Plots and saves the resulting model. """

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

        # Calculate the optimum steps per epoch.
        # self.__training_chunker.check_if_chunking()
        # steps_per_training_epoch = np.round(int(self.__training_chunker.total_size / self.__batch_size), decimals=0)
        steps_per_training_epoch = np.round(
            int(self.__training_chunker.total_num_samples / self.__batch_size),
            decimals=0
        ) + 1
        
        model = create_model(self.__input_window_length, self.__n_cols, self.__network_type)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.__learning_rate,
                beta_1=self.__beta_1,
                beta_2=self.__beta_2
            ),
            loss=self.__loss,
            metrics=self.__metrics
        )

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            min_delta=self.__min_delta,
            patience=self.__patience,
            verbose=self.__verbose,
            mode="auto",
            restore_best_weights=self.__restore_weights
        )

        # can use checkpoint ###############################################
        # checkpoint_filepath = "checkpoint/housedata/refit/"+ self.__appliance + "/"
        # model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        #     filepath = checkpoint_filepath,
        #     monitor='val_loss',
        #     verbose=0,
        #     save_best_only=True,
        #     save_weights_only=False,
        #     mode='auto',
        #     save_freq='epoch')        
        # callbacks=[early_stopping, model_checkpoint_callback]
        ###################################################################

        callbacks = []
        if self.__early_stopping:
            callbacks = [early_stopping]

        start_time = time.time()
        training_history = self.default_train(model, callbacks, steps_per_training_epoch)
        end_time = time.time()

        training_history.history["val_loss"] = np.repeat(
            training_history.history["val_loss"],
            self.__validation_frequency
        )

        model.summary()
        save_model(model, self.__network_type, self.__algorithm, 
                   self.__appliance, self.__save_model_dir)

        if self.__plot_training:
            self.plot_training_results(training_history)
        if self.__return_time:
            return  end_time - start_time


    def default_train(self, model, callbacks, steps_per_training_epoch):

        """ The default training method the neural network will use. No pruning occurs.

        Parameters:
        model (tensorflow.keras.Model): The seq2point model being trained.
        early_stopping (tensorflow.keras.callbacks.EarlyStopping): An early stopping callback to 
        prevent overfitting.
        steps_per_training_epoch (int): The number of training steps to occur per epoch.

        Returns:
        training_history (numpy.ndarray): The error metrics and loss values that were calculated 
        at the end of each training epoch.

        """
        # ########### this is retired ##############################
        # training_history = model.fit_generator(self.__training_chunker.load_dataset(),
        #     steps_per_epoch=steps_per_training_epoch,
        #     epochs=1,
        #     verbose=1,
        #     validation_data = self.__validation_chunker.load_dataset(),
        #     validation_steps=100,
        #     validation_freq=self.__validation_frequency,
        #     callbacks=[early_stopping])
        ############################################################

        training_history = model.fit(
            self.__training_chunker.load_dataset(),
            steps_per_epoch=steps_per_training_epoch,
            epochs=self.__epochs,
            verbose=self.__verbose,
            callbacks=callbacks,
            validation_data=self.__validation_chunker.load_dataset(),
            validation_freq=self.__validation_frequency,
            validation_steps=self.__validation_steps
        )

        return training_history

    def plot_training_results(self, training_history):

        """ Plots and saves a graph of training loss against epoch.

        Parameters:
        training_history (numpy.ndarray): A timeseries of loss against epoch count.

        """

        plt.plot(training_history.history["loss"], label="MSE (Training Loss)")
        plt.plot(training_history.history["val_loss"], label="MSE (Validation Loss)")
        plt.title(f'Training History for {self.__appliance}. Model: {self.__network_type}')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        plt.show()

        file_name = "outputs/plots/" + self.__appliance + "_" + self.__network_type + "_training_results.png"
        plt.savefig(fname=file_name)
