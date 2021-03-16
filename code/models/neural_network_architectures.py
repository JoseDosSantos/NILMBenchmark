import tensorflow as tf
import os
from keras.layers import *


def create_seq2seq(input_window_length, ncols):

    """Specifies the structure of a seq2point model using Keras' functional API.
    Taken from Zhang, Zhong, Wang, Goddard, Sutton et al. (2018) Sequence-to-Point Learning With Neural Networks for
    Non-Intrusive Load Monitoring

    Returns:
    model (tensorflow.keras.Model): The uncompiled seq2point model.

    """

    input_layer = Input(shape=(input_window_length, ncols))
    reshape_layer = Reshape((1, input_window_length, ncols))(input_layer)
    conv_layer_1 = Convolution2D(filters=30, kernel_size=(10, ncols), strides=(1, 1), padding="same",
                                 activation="relu")(reshape_layer)
    conv_layer_2 = Convolution2D(filters=30, kernel_size=(8, 1), strides=(1, 1), padding="same",
                                 activation="relu")(conv_layer_1)
    conv_layer_3 = Convolution2D(filters=40, kernel_size=(6, 1), strides=(1, 1), padding="same",
                                 activation="relu")(conv_layer_2)
    conv_layer_4 = Convolution2D(filters=50, kernel_size=(5, 1), strides=(1, 1), padding="same",
                                 activation="relu")(conv_layer_3)
    conv_layer_5 = Convolution2D(filters=50, kernel_size=(5, 1), strides=(1, 1), padding="same",
                                 activation="relu")(conv_layer_4)
    flatten_layer = Flatten()(conv_layer_5)
    label_layer = Dense(1024, activation="relu")(flatten_layer)
    output_layer = Dense(1, activation="linear")(label_layer)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    return model


def create_bi_gru(input_window_length, ncols):

    """Specifies the structure of a Bidirectional GRU model using Keras' functional API.
    Taken from Krystalakos, Nalmpantis, Vrakas et al. (2018) Sliding Window Approach for Online Energy
    Disaggregation Using Artificial Neural Networks

    Returns:
    model (tensorflow.keras.Model): The uncompiled GRU model.

    """

    input_layer = Input(shape=(input_window_length, ncols))
    reshape_layer_1 = Reshape((1, input_window_length, ncols))(input_layer)
    conv_layer_1 = Convolution2D(filters=16, kernel_size=(4, ncols), strides=(1, 1), padding="same",
                                 activation="relu")(reshape_layer_1)
    reshape_layer_2 = Reshape((input_window_length, 16))(conv_layer_1)
    gru_layer_1 = Bidirectional(GRU(units=64, activation="tanh", return_sequences=True),
                                merge_mode="concat")(reshape_layer_2)
    dropout_1 = Dropout(rate=0.5)(gru_layer_1)
    gru_layer_2 = Bidirectional(GRU(units=128, activation="tanh", return_sequences=False),
                                merge_mode="concat")(dropout_1)
    dropout_2 = Dropout(rate=0.5)(gru_layer_2)
    dense_1 = Dense(units=128, activation="relu")(dropout_2)
    dropout_3 = Dropout(rate=0.5)(dense_1)
    output_layer = Dense(units=1, activation="linear")(dropout_3)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    return model


def create_autoencoder(input_window_length, ncols):
    """Specifies the structure of a Denoising Autoencoder model using Keras' functional API.
    See Kelly, Knottenbelt et al. (2015) Neural NILM Deep Neural Networks Applied to Energy Disaggregation

    Returns:
    model (tensorflow.keras.Model): The uncompiled DAE model.

    """

    input_layer = Input(shape=(input_window_length, ncols))
    reshape_layer_1 = Reshape((1, input_window_length, ncols))(input_layer)
    conv_layer_1 = Conv2D(filters=8, kernel_size=(4, ncols), strides=(1, 1), padding="same",
                                 activation="linear")(reshape_layer_1)
    flatten_layer_1 = Flatten()(conv_layer_1)
    dense_1 = Dense(units=(input_window_length)*8, activation="relu")(flatten_layer_1)
    dense_2 = Dense(units=128, activation="relu")(dense_1)
    dense_3 = Dense(units=(input_window_length)*8, activation="relu")(dense_2)
    reshape_layer_2 = Reshape((input_window_length, 8))(dense_3)
    conv_layer_2 = Conv1D(filters=1, kernel_size=4, strides=1, activation="linear", padding="same")(reshape_layer_2)
    flatten_layer_2 = Flatten()(conv_layer_2)
    output_layer = Dense(units=1, activation="linear")(flatten_layer_2)
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    return model


def create_model(input_window_length, ncols, model_type="Seq2Seq"):
    if model_type == "Seq2Seq":
        return create_seq2seq(input_window_length=input_window_length, ncols=ncols)

    if model_type == "GRU":
        return create_bi_gru(input_window_length=input_window_length, ncols=ncols)

    if model_type == "DAE":
        return create_autoencoder(input_window_length=input_window_length, ncols=ncols)

    raise AttributeError(f"Model type {model_type} not recognized. Please provide a valid model type.")


def save_model(model, network_type, algorithm, appliance, save_model_dir):

    """ Saves a model to a specified location. Models are named using a combination of their
    target appliance, architecture, and pruning algorithm.

    Parameters:
    model (tensorflow.keras.Model): The Keras model to save.
    network_type (string): The architecture of the model ('', 'reduced', 'dropout', or 'reduced_dropout').
    algorithm (string): The pruning algorithm applied to the model.
    appliance (string): The appliance the model was trained with.

    """

    #model_path = "saved_models/" + appliance + "_" + algorithm + "_" + network_type + "_model.h5"
    model_path = save_model_dir

    if not os.path.exists(model_path):
        open(model_path, 'a').close()

    model.save(model_path)


def load_model(model, network_type, algorithm, appliance, saved_model_dir):

    """ Loads a model from a specified location.

    Parameters:
    model (tensorflow.keras.Model): The Keas model to which the loaded weights will be applied to.
    network_type (string): The architecture of the model ('', 'reduced', 'dropout', or 'reduced_dropout').
    algorithm (string): The pruning algorithm applied to the model.
    appliance (string): The appliance the model was trained with.

    """

    #model_name = "saved_models/" + appliance + "_" + algorithm + "_" + network_type + "_model.h5"
    model_name = saved_model_dir
    print("PATH NAME: ", model_name)

    model = tf.keras.models.load_model(model_name)
    num_of_weights = model.count_params()
    print("Loaded model with ", str(num_of_weights), " weights")
    return model
