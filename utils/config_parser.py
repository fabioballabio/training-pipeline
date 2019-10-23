import json


class ConfigParser:
    """
    Parses the configuration file provided.
        Attributes:
                config_file_path : path to the json file containing the configuration
    """

    def __init__(self, config_file_path):
        self.config_file_path = config_file_path

    def get_config(self):
        """
        Parses json config file.

            Returns:
                data_path : path to common root directory of data.
                dataset : str identifying name of the dataset to be loaded from data_path.
                model : class name of the model to be trained.
                epochs : integer number of training epochs.
                batch_size : size of batches.
                loss : str identifying the desired loss function to be optimized while training.
                optimizer : str identifying the desired optimizer to be employed while training.
                lr : learning rate.
                log_dir : str identifying the path where training results are written.
                save_dir : str identifying the path where model is saved once training is finished.
        """
        try:
            config_file = open(self.config_file_path).read()
        except FileNotFoundError:
            print("Oops! No such file at %s" % self.config_file_path)
        try:
            parsed_json = json.loads(config_file)
        except ValueError:
            print("Config file not in valid JSON format")

        data_path = parsed_json["input_data"].get("data_path", "./data")
        try:
            dataset_name = parsed_json["input_data"]["dataset"]
        except KeyError:
            print(
                "You have to specify the name of a valid dataset to train your model!"
            )
        try:
            model = parsed_json["training"]["model"]
        except KeyError:
            print("You have to specify the name of a valid model to train it!")
        epochs = parsed_json["training"].get("epochs", None)
        batch_size = parsed_json["training"].get("batch_size", None)
        loss = parsed_json["training"].get("loss", "mse")
        optimizer = parsed_json["training"].get("optimizer", "adam")
        lr = parsed_json["training"].get("learning_rate", 1e-3)
        log_dir = parsed_json["output_data"].get("logs_path", None)
        save_dir = parsed_json["output_data"].get("model_save_path", None)
        return (
            data_path,
            dataset_name,
            model,
            epochs,
            batch_size,
            loss,
            optimizer,
            lr,
            log_dir,
            save_dir,
        )
