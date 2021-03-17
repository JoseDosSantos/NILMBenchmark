import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

from models.co import CO
from models.afhmm import AFHMM
from errors import *
from typing import Union

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# If not specified otherwise, use all appliances available
APPLIANCES = {"fridge", "microwave", "washing machine", "television", "laptop computer"}

# We only implement Combinatorial Optimisation and AFHMM models here. Both of these need no validation data and
# need most of the time for testing (especially for the AFHMM). Therefore we currently do not export the trained model
# and to not use a separate test-script .Instead, we always retrain and evaluate performance on test immediately.

class Trainer:

    def __init__(self,
                 interval,
                 network_type,
                 data_directory,
                 use_occupancy=False,
                 use_weather=False,
                 ):
        self.__interval = interval
        self.__network_type = network_type
        self.__data_directory = data_directory
        self.__use_occupancy = use_occupancy
        self.__use_weather = use_weather

if __name__ == '__main__':

    interval = "6s"
    params = {}
    model = AFHMM(params)

    def load_df(app, freq, cols, dataset="training", denorm=False):
        df = pd.read_csv(f"../data/appliances/{app}/{app}_{dataset}_{freq}_.csv")[cols]
        if denorm:
            return pd.Series(denormalize(df, app).reshape(-1,))
        return df

    apps_training = {}
    for app in APPLIANCES:
        apps_training[app] = load_df(app, interval, cols=app, dataset="training", denorm=True)

    mains_training = load_df("fridge", interval, cols=["mains"], dataset="training", denorm=True)

    model.partial_fit(train_main=[mains_training], train_appliances=apps_training)

    apps_test = {}
    for app in APPLIANCES:
        apps_test[app] = load_df(app, interval, cols=app, dataset="test", denorm=True)

    mains_test = load_df("fridge", interval, cols=["mains"], dataset="test", denorm=True)

    n = 720*12

    start_time = time.time()
    results = model.disaggregate_chunk(test_mains_list=[mains_test[:n]])[0]

    end_time = time.time()
    test_time = end_time - start_time
    print(test_time)

    for app in APPLIANCES:
        true_apps = np.array(apps_test[app][:n])#enormalize(apps_test[app][:n], app)
        pred_apps = np.array(results[app])# denormalize(results[app], app)

        mse = mean_squared_error(true_apps, pred_apps)
        mae = mean_absolute_error(true_apps, pred_apps)
        sae = normalised_signal_aggregate_error(true_apps, pred_apps)
        mr = match_rate(true_apps, pred_apps)

        print(f"{app} : MSE: {mse}, MAE: {mae}, SAE: {sae} MR: {mr}")

    res_df = pd.DataFrame(results)

    for app in APPLIANCES:
        true_apps = np.array(apps_test[app][:n])#denormalize(apps_test[app][:n], app)
        pred_apps = np.array(results[app]) #denormalize(results[app], app)

        plt.figure(1)
        plt.plot(true_apps, label="Ground Truth")
        plt.plot(pred_apps, label="Predicted")
        # plt.title(self.__appliance + " " + self.__network_type + "(" + self.__algorithm + ")")
        plt.ylabel("Power Value (Watts)")
        plt.xlabel("Testing Window")
        plt.title(app)
        plt.legend()
        plt.show()

        # file_path = "./" + self.__appliance + "/saved_models/" + self.__appliance + "_" + self.__algorithm + "_" + self.__network_type + "_test_figure.png"
        # plt.savefig(fname=file_path)




    res_df.plot()
    print(results)
