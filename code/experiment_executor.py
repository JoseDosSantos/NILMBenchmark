import pandas as pd
import utils
from nn_train import Trainer
from nn_test import Tester
import nilmtk_run
import os
import argparse


def run(exp_name):
    nilmtk_models = ["CO", "AFHMM"]
    nilmtk_appliances = ["fridge", "microwave", "washing machine", "television", "electric heating element", "cooker"]

    experiment = utils.read_yaml(f"experiment_configs/{exp_name}.yaml")
    pass

    devices = experiment["devices"]
    interval = experiment["interval"]
    use_weather = experiment["use_weather"]
    use_occupancy = experiment["use_occupancy"]

    times = {
        "train_times": {},
        "test_times": {}
    }

    for model in experiment["models"].values():
        network_type = model["network_type"]
        window_length = model["window_length"]
        output_length = model["output_length"]
        epochs = model["epochs"]
        learning_rate = model["learning_rate"]
        batch_size = model["batch_size"]
        early_stopping = model["early_stopping"]
        patience = model["patience"]
        restore_weights = model["restore_weights"]

        times["train_times"][network_type] = {}
        times["test_times"][network_type] = {}

        if network_type in nilmtk_models:
            if use_weather or use_occupancy:
                continue
            else:
                train_time_agg = 0
                test_time_agg = 0
                eval_counter = 0

                for test_set in experiment["evaluate"]:
                    train_time, test_time = nilmtk_run.run_model(
                        model_type=network_type,
                        appliances=nilmtk_appliances,
                        interval=interval,
                        test_dataset=test_set,
                        experiment_name=exp_name,
                        return_time=True,
                        export_predictions=True,
                        verbose=False)

                    train_time_agg += train_time
                    test_time_agg += test_time
                    eval_counter += 1

                times["train_times"][network_type] = train_time_agg / eval_counter
                times["test_times"][network_type] = test_time_agg / eval_counter

        else:
            for device in devices:
                training_directory = f"../data/appliances/{device}/{device}_train_{interval}_.csv"
                validation_directory = f"../data/appliances/{device}/{device}_val_{interval}_.csv"

                save_model_dir = utils.get_model_save_path(
                    network_type=network_type,
                    interval=interval,
                    device=device,
                    window_length=window_length,
                    use_weather=use_weather,
                    use_occupancy=use_occupancy
                )

                trainer = Trainer(
                    appliance=device,
                    batch_size=batch_size,
                    crop=None,
                    network_type=network_type,
                    use_weather=use_weather,
                    use_occupancy=use_occupancy,
                    learning_rate=learning_rate,
                    epochs=epochs,
                    early_stopping=early_stopping,
                    patience=patience,
                    restore_weights=restore_weights,
                    training_directory=training_directory,
                    validation_directory=validation_directory,
                    save_model_dir=save_model_dir,
                    input_window_length=window_length,
                    output_length=output_length,
                    validation_frequency=1,
                    return_time=True,
                    plot_training=False
                )
                train_time = trainer.train_model()
                times["train_times"][network_type][device] = train_time

                # Average test_times across all datasets

                test_time_agg = 0
                eval_counter = 0

                for test_set in experiment["evaluate"]:
                    test_directory = f"../data/appliances/{device}/{device}_{test_set}_{interval}_.csv"

                    if os.path.isfile(test_directory):
                        # The logs including results will be recorded to this log file
                        utils.check_dir(f"outputs/logs/{exp_name}/")
                        log_file_dir = f"outputs/logs/{exp_name}/{network_type}_{device}.log"

                        tester = Tester(
                            appliance=device,
                            algorithm=network_type,
                            crop=None,
                            batch_size=batch_size,
                            network_type=network_type,
                            test_directory=test_directory,
                            saved_model_dir=save_model_dir,
                            log_file_dir=log_file_dir,
                            input_window_length=window_length,
                            output_length=output_length,
                            use_weather=use_weather,
                            use_occupancy=use_occupancy,
                            plot_first=-1,
                            dataset=test_set,
                            return_time=True,
                            return_predictions=True
                        )
                        pred, test_time = tester.test_model()
                        test_time_agg += test_time
                        eval_counter += 1

                        utils.check_dir(f"outputs/model_predictions/{exp_name}/")
                        results_path = f"outputs/model_predictions/{exp_name}/{network_type}_{device}_{test_set}.csv"
                        pd.DataFrame(pred).to_csv(results_path, sep=";")

                times["test_times"][network_type][device] = test_time_agg / eval_counter

    utils.save_dict(times, f"outputs/experiment_logs/{exp_name}_training_times")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run experiment from experiment config. ")
    parser.add_argument("--appliance_name", type=str, default="experiment4",
                        help="Experiment to execute. Provide name of yaml without '.yaml'")
    run(exp_name="experiment_special")
