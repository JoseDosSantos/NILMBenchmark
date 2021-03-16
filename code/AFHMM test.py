from code.models.co import CO
import pandas as pd
import numpy as np
import time

if __name__ == '__main__':
    import warnings

    warnings.filterwarnings("ignore", category=UserWarning)
    device = "fridge"
    interval = "1min"
    params = {}
    model = CO(params)

    data_fridge = pd.read_csv(f"../../data/appliances/{device}/{device}_training_{interval}_.csv")

    device = "microwave"
    data_microwave = pd.read_csv(f"../../data/appliances/{device}/{device}_training_{interval}_.csv")

    device = "washing machine"
    data_washing_machine = pd.read_csv(f"../../data/appliances/{device}/{device}_training_{interval}_.csv")

    device = "television"
    data_television = pd.read_csv(f"../../data/appliances/{device}/{device}_training_{interval}_.csv")

    device = "laptop computer"
    data_laptop_computer = pd.read_csv(f"../../data/appliances/{device}/{device}_training_{interval}_.csv")

    mains = data_fridge["mains"]
    appliances = {"fridge": data_fridge["fridge"],
                  "microwave": data_microwave["microwave"],
                  "washing machine": data_washing_machine["washing machine"],
                  "television": data_television["television"],
                  "laptop computer": data_laptop_computer["laptop computer"]}
    model.partial_fit(train_main=[mains], train_appliances=appliances)

    device = "fridge"
    data_fridge_test = pd.read_csv(f"../../data/appliances/{device}/{device}_test_{interval}_.csv")

    device = "microwave"
    data_microwave_test = pd.read_csv(f"../../data/appliances/{device}/{device}_test_{interval}_.csv")

    device = "washing machine"
    data_washing_machine_test = pd.read_csv(f"../../data/appliances/{device}/{device}_test_{interval}_.csv")

    device = "television"
    data_television_test = pd.read_csv(f"../../data/appliances/{device}/{device}_test_{interval}_.csv")

    device = "laptop computer"
    data_laptop_computer_test = pd.read_csv(f"../../data/appliances/{device}/{device}_test_{interval}_.csv")


    n = 100000



    start_time = time.time()
    results = model.disaggregate_chunk(mains=[data_fridge_test["mains"][:n]])

    end_time = time.time()
    test_time = end_time - start_time
    print(test_time)

    def mse(y_true, y_pred):
        y_pred[y_pred < 0] = 0
        return np.mean(np.power(y_true-y_pred, 2))

    def sae(y_true, y_pred):
        y = y_pred
        y[y < 0] = 0
        return np.abs(np.sum(y_true)-np.sum(y_pred)) / np.sum(y_true)

    def mae(y_true, y_pred):
        y_pred[y_pred < 0] = 0

        return np.mean(np.abs(y_true-y_pred))

    device = "fridge"
    print(device, ": MSE: ", mse(data_fridge_test[device][:n], results[0][device]),
          " MAE: ", mae(data_fridge_test[device][:n], results[0][device]),
          " SAE: ", sae(data_fridge_test[device][:n], results[0][device]))

    device = "microwave"
    print(device, ": MSE: ", mse(data_microwave_test[device][:n], results[0][device]),
          " MAE: ", mae(data_microwave_test[device][:n], results[0][device]),
          " SAE: ", sae(data_microwave_test[device][:n], results[0][device]))

    device = "washing machine"
    print(device, ": MSE: ", mse(data_washing_machine_test[device][:n], results[0][device]),
          " MAE: ", mae(data_washing_machine_test[device][:n], results[0][device]),
          " SAE: ", sae(data_washing_machine_test[device][:n], results[0][device]))

    device = "television"
    print(device, ": MSE: ", mse(data_television_test[device][:n], results[0][device]),
          " MAE: ", mae(data_television_test[device][:n], results[0][device]),
          " SAE: ", sae(data_television_test[device][:n], results[0][device]))

    device = "laptop computer"
    print(device, ": MSE: ", mse(data_laptop_computer_test[device][:n], results[0][device]),
          " MAE: ", mae(data_laptop_computer_test[device][:n], results[0][device]),
          " SAE: ", sae(data_laptop_computer_test[device][:n], results[0][device]))