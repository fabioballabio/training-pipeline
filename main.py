from datasets.datasets import COCODataset, KittiDataset
from utils.arg_parser import ArgParser
from utils.config_parser import ConfigParser
from trainer.torch_trainer import TorchTrainer
from utils.training_utils import (
    get_labels_counts,
    rebalance_dataset,
    build_classification_dataset,
    get_model,
)
from general_utils import TorchDataset


if __name__ == "__main__":
    AVAILABLE_DATASETS = {"kitti": KittiDataset, "coco": COCODataset}
    # Parse command line interface to get config file path
    CONFIG_FILE_PATH = ArgParser().parse_cli()

    # Parse json config file to get params for training
    DATA_PATH, DATASET_NAME, MODEL_CLASS, EPOCHS, BATCH_SIZE, LOSS, OPTIMIZER, LR, LOG_DIR, SAVE_DIR = ConfigParser(
        CONFIG_FILE_PATH
    ).get_config()

    # Get dataset
    dataset = AVAILABLE_DATASETS[DATASET_NAME.lower()](DATA_PATH)
    torch_dataset = TorchDataset(dataset, resize_to=(256, 512))
    torch_dataset = build_classification_dataset(torch_dataset, ["Car"])
    # Bring detection dataset to classification dataset shape
    get_labels_counts(torch_dataset)
    torch_dataset = rebalance_dataset(torch_dataset)
    get_labels_counts(torch_dataset)
    # Get the model and fix last layers to accomplish our binary
    # classification task
    model = get_model(MODEL_CLASS)

    trainer = TorchTrainer(
        model=model,
        dataset=torch_dataset,
        epochs=EPOCHS,
        loss_func=LOSS,
        optimizer=OPTIMIZER,
        lr=LR,
        batch_size=BATCH_SIZE,
        log_dir=LOG_DIR,
    ).start_training()
