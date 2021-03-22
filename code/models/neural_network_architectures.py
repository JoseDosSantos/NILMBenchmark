import tensorflow as tf
import os
from keras.layers import *

def create_glu_conv(input_window_length, ncols):
    input_layer = Input(shape=(input_window_length, ncols))
    reshape_layer = Reshape((1, input_window_length, ncols))(input_layer)
    conv_1_bl1 = Conv2D(filters=100, kernel_size=(4, ncols), strides=(1, 1), padding="same",
                                 activation="sigmoid")(reshape_layer)
    conv_2_bl1 = Conv2D(filters=100, kernel_size=(4, ncols), strides=(1, 1), padding="same",
                                 activation="linear")(reshape_layer)

    out_bl1 = Multiply()([conv_1_bl1, conv_2_bl1])

    reshape_layer_2 = Reshape((input_window_length, 100))(out_bl1)
    max_pool_1 = MaxPooling1D(pool_size=2)(reshape_layer_2)

    conv_1_bl2 = Conv1D(filters=100, kernel_size=4, strides=1, padding="same",
                        activation="sigmoid")(max_pool_1)
    conv_2_bl2 = Conv1D(filters=100, kernel_size=4, strides=1, padding="same",
                        activation="linear")(max_pool_1)
    out_bl2 = Multiply()([conv_1_bl2, conv_2_bl2])
    max_pool_2 = MaxPooling1D(pool_size=2)(out_bl2)

    conv_1_bl3 = Conv1D(filters=100, kernel_size=4, strides=1, padding="same",
                        activation="sigmoid")(max_pool_2)
    conv_2_bl3 = Conv1D(filters=100, kernel_size=4, strides=1, padding="same",
                        activation="linear")(max_pool_2)
    out_bl3 = Multiply()([conv_1_bl3, conv_2_bl3])
    max_pool_3 = MaxPooling1D(pool_size=2)(out_bl3)

    flatten_layer = Flatten()(max_pool_3)

    dense_1_res1 = Dense(units=50, activation="relu")(flatten_layer)
    dense_2_res1 = Dense(units=50, activation=None)(dense_1_res1)
    out_res1 = Add()([dense_1_res1, dense_2_res1])

    dense_1_res2 = Dense(units=50, activation="relu")(out_res1)
    dense_2_res2 = Dense(units=50, activation=None)(dense_1_res2)
    out_res2 = Add()([dense_1_res2, dense_2_res2])

    dense_1_res3 = Dense(units=50, activation="relu")(out_res2)
    dense_2_res3 = Dense(units=50, activation=None)(dense_1_res3)
    out_res3 = Add()([dense_1_res3, dense_2_res3])

    dense_1 = Dense(units=input_window_length//8, activation="relu")(out_res3)
    dense_2 = Dense(units=input_window_length//8, activation=None)(dense_1)

    model = tf.keras.Model(inputs=input_layer, outputs=dense_2)

    return model



def create_fully_convolutional(input_window_length, ncols):
    input_layer = Input(shape=(input_window_length, ncols))
    reshape_layer = Reshape((1, input_window_length, ncols))(input_layer)
    conv_layer_1 = Conv2D(filters=128, kernel_size=(9, ncols), strides=(1, 1), padding="same",
                                 activation="relu")(reshape_layer)
    dilated_conv_1 = Conv1D(filters=128, kernel_size=3, strides=1, padding="same", activation="relu",
                            dilation_rate=2)(conv_layer_1)
    dilated_conv_2 = Conv1D(filters=128, kernel_size=3, strides=1, padding="same", activation="relu",
                            dilation_rate=4)(dilated_conv_1)
    dilated_conv_3 = Conv1D(filters=128, kernel_size=3, strides=1, padding="same", activation="relu",
                            dilation_rate=8)(dilated_conv_2)
    dilated_conv_4 = Conv1D(filters=128, kernel_size=3, strides=1, padding="same", activation="relu",
                            dilation_rate=16)(dilated_conv_3)
    dilated_conv_5 = Conv1D(filters=128, kernel_size=3, strides=1, padding="same", activation="relu",
                            dilation_rate=32)(dilated_conv_4)
    dilated_conv_6 = Conv1D(filters=128, kernel_size=3, strides=1, padding="same", activation="relu",
                            dilation_rate=64)(dilated_conv_5)
    dilated_conv_7 = Conv1D(filters=128, kernel_size=3, strides=1, padding="same", activation="relu",
                            dilation_rate=128)(dilated_conv_6)
    dilated_conv_8 = Conv1D(filters=128, kernel_size=3, strides=1, padding="same", activation="relu",
                            dilation_rate=256)(dilated_conv_7)
    dilated_conv_9 = Conv1D(filters=128, kernel_size=3, strides=1, padding="same", activation="relu",
                            dilation_rate=512)(dilated_conv_8)
    conv_layer_2 = Conv1D(filters=128, kernel_size=1, strides=1, padding="same", activation="relu")(dilated_conv_9)
    conv_layer_3 = Conv1D(filters=1, kernel_size=1, strides=1, padding="same", activation="relu")(conv_layer_2)
    flatten_layer_1 = Flatten()(conv_layer_3)
    output_layer = Dense(1, activation="linear")(flatten_layer_1)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    return model



def create_seq2point(input_window_length, ncols):

    """Specifies the structure of a seq2point model using Keras' functional API.
    Taken from Zhang, Zhong, Wang, Goddard, Sutton et al. (2018) Sequence-to-Point Learning With Neural Networks for
    Non-Intrusive Load Monitoring

    Returns:
    model (tensorflow.keras.Model): The uncompiled seq2point model.

    """

    input_layer = Input(shape=(input_window_length, ncols))
    reshape_layer = Reshape((1, input_window_length, ncols))(input_layer)
    conv_layer_1 = Conv2D(filters=30, kernel_size=(10, ncols), strides=(1, 1), padding="same",
                                 activation="relu")(reshape_layer)
    conv_layer_2 = Conv2D(filters=30, kernel_size=(8, 1), strides=(1, 1), padding="same",
                                 activation="relu")(conv_layer_1)
    conv_layer_3 = Conv2D(filters=40, kernel_size=(6, 1), strides=(1, 1), padding="same",
                                 activation="relu")(conv_layer_2)
    conv_layer_4 = Conv2D(filters=50, kernel_size=(5, 1), strides=(1, 1), padding="same",
                                 activation="relu")(conv_layer_3)
    conv_layer_5 = Conv2D(filters=50, kernel_size=(5, 1), strides=(1, 1), padding="same",
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
    conv_layer_1 = Conv2D(filters=8 * ncols, kernel_size=(4, ncols), strides=(1, 1), padding="same",
                                 activation="linear")(reshape_layer_1)
    flatten_layer_1 = Flatten()(conv_layer_1)
    dense_1 = Dense(units=(input_window_length)*8, activation="relu")(flatten_layer_1)
    dense_2 = Dense(units=128, activation="relu")(dense_1)
    dense_3 = Dense(units=(input_window_length)*8, activation="relu")(dense_2)
    reshape_layer_2 = Reshape((input_window_length, 8))(dense_3)
    conv_layer_2 = Conv1D(filters=1, kernel_size=4, strides=1, activation="linear", padding="same")(reshape_layer_2)
    output = Reshape((-1, input_window_length))(conv_layer_2)
    model = tf.keras.Model(inputs=input_layer, outputs=output)

    return model


def create_model(input_window_length, ncols, model_type="Seq2Seq"):
    if model_type == "Seq2Point":
        return create_seq2point(input_window_length=input_window_length, ncols=ncols)

    if model_type == "BiGRU":
        return create_bi_gru(input_window_length=input_window_length, ncols=ncols)

    if model_type == "DAE":
        return create_autoencoder(input_window_length=input_window_length, ncols=ncols)

    if model_type == "FullConv":
        return create_fully_convolutional(input_window_length=input_window_length, ncols=ncols)

    if model_type == "GLUConv":
        return create_glu_conv(input_window_length=input_window_length, ncols=ncols)

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
