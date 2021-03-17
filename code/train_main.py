import argparse
from remove_space import remove_space
from nn_train import Trainer


# Defaults / settings for IDE execution
network_type = "GLUConv"
device = "cooker"
interval = "1min"       # 1min or 6s
use_weather = False
use_occupancy = False
window_length = 64
output_length = 8
epochs = 100
learning_rate = 0.00003
batch_size = 1024
early_stopping = True
patience = 10
restore_weights = True

# Allows a model to be trained from the terminal.

training_directory = f"../data/appliances/{device}/{device}_train_{interval}_.csv"
validation_directory = f"../data/appliances/{device}/{device}_val_{interval}_.csv"

parser = argparse.ArgumentParser(description="Train sequence-to-point learning for energy disaggregation. ")

parser.add_argument("--appliance_name", type=str, default=device, help="The name of the appliance to train the network with. Default is kettle. Available are: kettle, fridge, washing machine, dishwasher, and microwave. ")
parser.add_argument("--batch_size", type=int, default=batch_size, help="The batch size to use when training the network. Default is 1000. ")
parser.add_argument("--crop", type=int, default=None, help="The number of rows of the dataset to take training data from. If not specified all are taken. ")
parser.add_argument("--network_type", type=remove_space, default=network_type, help="The network architecture to use. Options are 'Seq2Seq', 'GRU'. ")
parser.add_argument("--epochs", type=int, default=epochs, help="Number of epochs. Default is 10. ")
parser.add_argument("--learning_rate", type=float, default=learning_rate, help="Learning rate. Default is 0.001. ")
parser.add_argument("--early_stopping", type=bool, default=early_stopping, help="Use early stopping when training. Default is True. ")
parser.add_argument("--patience", type=int, default=patience, help="Patience for early stopping. Default is 5." )
parser.add_argument("--restore_weights", type=bool, default=restore_weights, help="Restore best weights when early stopping. Default is True. ")
parser.add_argument("--input_window_length", type=int, default=window_length, help="Number of input data points to network. Default is 599. ")
parser.add_argument("--output_length", type=int, default=output_length, help="Number of output data points of network. Default is 1 ")
parser.add_argument("--validation_frequency", type=int, default="1", help="How often to validate model. Default is 1. ")
parser.add_argument("--training_directory", type=str, default=training_directory, help="The dir for training data. ")
parser.add_argument("--validation_directory", type=str, default=validation_directory, help="The dir for validation data. ")
parser.add_argument("--use_occupancy", type=bool, default=use_occupancy, help="Include occupancy data for training. Default is False. ")
parser.add_argument("--use_weather", type=bool, default=use_weather, help="Include weather data for training. Default is False. ")
parser.add_argument("--interval", type=str, default=interval, help="Sampling interval to be used. Either '1min' or '6s'. ")


arguments = parser.parse_args()

# Need to provide the trained model
save_model_dir = f"outputs/saved_models/{arguments.appliance_name}_{arguments.interval}_{arguments.network_type}_model.h5"

trainer = Trainer(
    appliance=arguments.appliance_name,
    batch_size=arguments.batch_size,
    crop=arguments.crop,
    network_type=arguments.network_type,
    use_weather=arguments.use_weather,
    use_occupancy=arguments.use_occupancy,
    learning_rate=arguments.learning_rate,
    epochs=arguments.epochs,
    early_stopping=arguments.early_stopping,
    patience=arguments.patience,
    restore_weights=arguments.restore_weights,
    training_directory=arguments.training_directory,
    validation_directory=arguments.validation_directory,
    save_model_dir=save_model_dir,
    input_window_length=arguments.input_window_length,
    output_length=arguments.output_length,
    validation_frequency=arguments.validation_frequency
)
trainer.train_model()

