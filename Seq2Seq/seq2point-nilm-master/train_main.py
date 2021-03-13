import argparse
from remove_space import remove_space
from seq2point_train import Trainer

# Allows a model to be trained from the terminal.

training_directory = "../../data/appliances/microwave/microwave_training_1min_.csv"
validation_directory = "../../data/appliances/microwave/microwave_validation_1min_.csv"

parser = argparse.ArgumentParser(description="Train sequence-to-point learning for energy disaggregation. ")

parser.add_argument("--appliance_name", type=remove_space, default="microwave", help="The name of the appliance to train the network with. Default is kettle. Available are: kettle, fridge, washing machine, dishwasher, and microwave. ")
parser.add_argument("--batch_size", type=int, default="256", help="The batch size to use when training the network. Default is 1000. ")
parser.add_argument("--crop", type=int, default=None, help="The number of rows of the dataset to take training data from. If not specified all are taken. ")
# parser.add_argument("--pruning_algorithm", type=remove_space, default="default", help="The pruning algorithm that the network will train with. Default is none. Available are: spp, entropic, threshold. ")
parser.add_argument("--network_type", type=remove_space, default="seq2point", help="The seq2point architecture to use. ")
parser.add_argument("--epochs", type=int, default="10", help="Number of epochs. Default is 10. ")
parser.add_argument("--learning_rate", type=float, default="0.00001", help="Learning rate. Default is 0.001. ")
parser.add_argument("--input_window_length", type=int, default="599", help="Number of input data points to network. Default is 599.")
parser.add_argument("--validation_frequency", type=int, default="1", help="How often to validate model. Default is 1. ")
parser.add_argument("--training_directory", type=str, default=training_directory, help="The dir for training data. ")
parser.add_argument("--validation_directory", type=str, default=validation_directory, help="The dir for validation data. ")
parser.add_argument("--use_occupancy", type=bool, default=True, help="Include occupancy data for training. Default is False. ")
parser.add_argument("--use_weather", type=bool, default=True, help="Include weather data for training. Default is False. ")


arguments = parser.parse_args()

# Need to provide the trained model
save_model_dir = "saved_models/" + arguments.appliance_name + "_" + arguments.network_type + "_model.h5"

trainer = Trainer(
    appliance=arguments.appliance_name,
    batch_size=arguments.batch_size,
    crop=arguments.crop,
    network_type=arguments.network_type,
    use_weather=arguments.use_weather,
    use_occupancy=arguments.use_occupancy,
    learning_rate=arguments.learning_rate,
    training_directory=arguments.training_directory,
    validation_directory=arguments.validation_directory,
    save_model_dir=save_model_dir,
    epochs=arguments.epochs,
    input_window_length=arguments.input_window_length,
    validation_frequency=arguments.validation_frequency
)
trainer.train_model()

