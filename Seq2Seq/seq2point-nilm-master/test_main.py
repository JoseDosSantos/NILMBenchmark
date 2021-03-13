import argparse
from remove_space import remove_space
from seq2point_test import Tester

# Allows a model to be tested from the terminal.

# You need to input your test data directory
test_directory = "../../data/appliances/microwave/microwave_test_1min_.csv"

parser = argparse.ArgumentParser(description="Train a pruned neural network for energy disaggregation. ")

parser.add_argument("--appliance_name", type=remove_space, default="microwave", help="The name of the appliance to perform disaggregation with. Default is kettle. Available are: kettle, fridge, dishwasher, microwave. ")
parser.add_argument("--batch_size", type=int, default="1024", help="The batch size to use when training the network. Default is 1000. ")
parser.add_argument("--crop", type=int, default=None, help="The number of rows of the dataset to take training data from. Default is None. ")
parser.add_argument("--algorithm", type=remove_space, default="seq2point", help="The pruning algorithm of the model to test. Default is none. ")
parser.add_argument("--network_type", type=remove_space, default="", help="The seq2point architecture to use. Only use if you do not want to use the standard architecture. Available are: default, dropout, reduced, and reduced_dropout. ")
parser.add_argument("--input_window_length", type=int, default="599", help="Number of input data points to network. Default is 599. ")
parser.add_argument("--test_directory", type=str, default=test_directory, help="The dir for training data. ")
parser.add_argument("--use_occupancy", type=bool, default=True, help="Include occupancy data for training. Default is False. ")
parser.add_argument("--use_weather", type=bool, default=True, help="Include weather data for training. Default is False. ")

arguments = parser.parse_args()

# You need to provide the trained model
saved_model_dir = "saved_models/" + arguments.appliance_name + "_" + arguments.algorithm + "_model.h5"

# The logs including results will be recorded to this log file
log_file_dir = "saved_models/" + arguments.appliance_name + "_" + arguments.algorithm + "_" + arguments.network_type + ".log"

tester = Tester(
    appliance=arguments.appliance_name,
    algorithm=arguments.algorithm,
    crop=arguments.crop,
    batch_size=arguments.batch_size,
    network_type=arguments.network_type,
    test_directory=arguments.test_directory,
    saved_model_dir=saved_model_dir,
    log_file_dir=log_file_dir,
    input_window_length=arguments.input_window_length,
    use_weather=arguments.use_weather,
    use_occupancy=arguments.use_occupancy,
)
tester.test_model()

