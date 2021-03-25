import pandas as pd
import numpy as np
import time
import logging

import matplotlib.pyplot as plt
import math
from models.co import CO
from models.afhmm import AFHMM
import utils
from multiprocessing import cpu_count
from typing import Union

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# If not specified otherwise, use all appliances available
APPLIANCES = ["fridge", "microwave", "washing machine", "television", "electric heating element", "cooker"]

# We only implement Combinatorial Optimisation and AFHMM models here. Both of these need no validation data and
# need most of the time for testing (especially for the AFHMM). Therefore we currently do not export the trained model
# and to not use a separate test-script. Instead, we always retrain and evaluate performance on test immediately.


def load_df(app, freq, col, dataset="train", denorm=False):
    df = pd.read_csv(f"../data/appliances/{app}/{app}_{dataset}_{freq}_.csv")[col]
    if denorm:
        return pd.Series(utils.denormalize(df, app).reshape(-1, ))
    return df


def run_model(
        model_type,
        appliances,
        interval,
        test_dataset,
        experiment_name,
        train_denorm=True,
        plot_results=False,
        return_time=False,
        export_predictions=False,
        verbose=False
):
    if appliances:
        appliance_list = appliances
    else:
        appliance_list = APPLIANCES

    train_appliances = {}
    for app in appliance_list:
        train_appliances[app] = load_df(app, interval, col=app, dataset="train", denorm=train_denorm)
    train_mains = load_df("fridge", interval, col="mains", dataset="train", denorm=train_denorm)

    if model_type == "CO":
        model = CO({})
    elif model_type == "AFHMM":
        model = AFHMM({})
    else:
        raise ValueError(f"Model type {model_type} not understood. Available currently are only 'CO' and 'AFHMM'.")

    train_start_time = time.time()
    model.partial_fit(train_main=[train_mains], train_appliances=train_appliances)
    train_time = time.time() - train_start_time

    test_appliances = {}
    # Average test time across datasets
    test_time = 0
    test_appliances = {}
    for app in appliance_list:
        try:
            test_appliances[app] = load_df(app, interval, col=app, dataset=test_dataset, denorm=train_denorm)
        except:
            pass

    if model_type == "AFHMM" and test_dataset == "ECO":
        raise ValueError("Do not use AFHMM with ECO. It is not currently implemented due to long testing times.")

    test_time_agg = 0
    eval_counter = 0

    if model_type == "AFHMM":
        num_workers = cpu_count()

        # hardcoded fix for now
        chunk_length = 720
        test_mains = load_df(appliance_list[0], interval, col="mains", dataset=test_dataset, denorm=train_denorm)

        test_mains = test_mains.values.flatten().reshape((-1, 1))
        n = len(test_mains)
        n_chunks = int(math.ceil(len(test_mains) / chunk_length))

        # test_mains_chunks = [test_mains_big[i:i+self.time_period] for i in range(0, test_mains_big.shape[0], self.time_period)]
        n_iter = math.ceil(n_chunks / num_workers)
        results = []
        test_start_time = time.time()

        print(f"Starting disaggregation for {n_iter} chunks.")
        for i in tqdm(range(n_iter)):
            # print(i * num_workers * chunk_length, i * num_workers * chunk_length + chunk_length * num_workers)
            mains = test_mains[
                    i * num_workers * chunk_length:i * num_workers * chunk_length + chunk_length * num_workers]
            # print(len(mains))
            results.append(model.disaggregate_chunk(mains)[0])
            pd.concat(results, axis=0).to_csv(f"quicksaves/checkpoint{i}_{interval}.csv", sep=";")
        test_time = time.time() - test_start_time
        results = pd.concat(results, axis=0)[:n]

    for app in appliance_list:
        try:
            if model_type == "CO":
                test_mains = load_df(app, interval, col="mains", dataset=test_dataset, denorm=train_denorm)
                n = len(test_mains)
                test_start_time = time.time()
                results = model.disaggregate_chunk(mains=pd.Series([test_mains[:n]]))[0]
                test_time = time.time() - test_start_time

            if train_denorm:
                true_apps = np.array(test_appliances[app][:n])
                pred_apps = np.array(results[app])
            else:
                true_apps = utils.denormalize(test_appliances[app][:n], app)
                pred_apps = utils.denormalize(results[app], app)

            mse = utils.mean_squared_error(true_apps, pred_apps)
            mae = utils.mean_absolute_error(true_apps, pred_apps)
            sae = utils.normalised_signal_aggregate_error(true_apps, pred_apps)
            mr = utils.match_rate(true_apps, pred_apps)

            log_file_dir = f"Nilmtk/logs/{experiment_name}/{model_type}_{app}.log"

            # In Python 3.8 we can just add force=True to the basic config, but project is written in 3.7
            # so clear and reset path manually (there's probably a better way)
            for handler in logging.root.handlers[:]:
                logging.root.removeHandler(handler)
            logging.basicConfig(filename=log_file_dir, format='%(message)s', level=logging.INFO)

            test_log = f"Test dataset: {test_dataset}"
            logging.info(test_log)

            metric_string = f"MSE: {mse}" \
                            f" MAE: {mae}" \
                            f" SAE: {sae}" \
                            f" Match Rate: {mr}\n"
            logging.info(metric_string)

            if export_predictions:
                utils.check_dir(f"Nilmtk/model_predictions/{experiment_name}/")
                results_path = f"Nilmtk/model_predictions/{experiment_name}/{model_type}_{app}_{test_dataset}.csv"
                pd.DataFrame(pred_apps).to_csv(results_path, sep=";")

            test_time_agg += test_time
            eval_counter += 1
        except Exception as e:
            if verbose:
                print(app, e)

    test_time_agg /= eval_counter

    if return_time:
        return train_time, test_time_agg


    # if plot_results:
    #     for app in APPLIANCES:
    #         if train_denorm:
    #             true_apps = np.array(test_appliances[app][:n])
    #             pred_apps = np.array(results[app])
    #         else:
    #             true_apps = utils.denormalize(test_appliances[app][:n], app)
    #             pred_apps = utils.denormalize(results[app], app)
    #
    #         plt.figure(1)
    #         plt.plot(true_apps, label="Ground Truth")
    #         plt.plot(pred_apps, label="Predicted")
    #         # plt.title(self.__appliance + " " + self.__network_type + "(" + self.__algorithm + ")")
    #         plt.ylabel("Power Value (Watts)")
    #         plt.xlabel("Testing Window")
    #         plt.title(app)
    #         plt.legend()
    #         plt.show()
    #
    #         # file_path = "./" + self.__appliance + "/saved_models/" + self.__appliance + "_" + self.__algorithm + "_" + self.__network_type + "_test_figure.png"
    #         # plt.savefig(fname=file_path)




    #res_df.plot()

if __name__ == '__main__':
    t1, t2 = run_model("AFHMM", APPLIANCES, "6s", "test", "experiment_4", return_time=True, export_predictions=True, verbose=True)
    print(t1, t2)
