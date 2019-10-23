# Configurable training pipeline based on PyTorch

## Features:
- [x] Configurable: .json file to specify training details
- [x] Expandable: Add new custom datasets and/or models as simple as writing a new python class
- [x] Visualization: Training evolution and results visualized through tensorboard
- [x] GPU training supported
## TODO:
- [ ] Improve config.json file structure
- [ ] Support different tasks, not only classification 
- [ ] Allow sklearn-like models training 
## PACKAGE STRUCTURE:
```
.
├── configs                 # Config files in .json format
├── datasets                # Datasets and methods to handle them as python classes
├── datasets                # Raw data folder
├── models                  # Models defined as python classes
├── trainer                 # Responsible to get data and model, and handle actual training
├── utils                   # Parse CLI and config file plus training utilities
├── general_utils.py        # General utils beyond training
├── main.py                 # Launch your training instance
└── README.md
```
## RUN:
### Run through:
```
python main.py - p path / to / config / file
```
### Observe training running tensorboard:
```
tensorboard - -logdir = . / absolute / path / to / run / file
```
## CONFIGURATION FILE:
Configuration file is a .json file aimed to build a customized training instance.
It is structured in three different sections: input data, training and output data
### Input data properties:
- input_data (mandatory): path to all data parent directory - str
- dataset (mandatory): name of the dataset to be employed in training (Kitti/COCO) - str
### Training properties:
- model (mandatory): name of the python class identifying the model to be trained - str
- epochs (optional): number of epochs training should last - int
- batch_size (optional): number of samples processed per training iteration - int
- loss (mandatory): performance measure to be optimized during training - str
- optimizer (optional): algorithm responsible for gradients calculation and parameters update - str
- learning_rate (optional): hyperparam controlling how fast the model learns - float
### Output data properties:
- model_save_path (optional): path where model checkpoints will be saved during training - str
- logs_path (optional): path where tensorboard runs will be saved for visualization - str
```
{
    "input_data": {
        "data_path": "./path/to/common/data/parent/dir",
        "dataset": "Kitti" or "COCO"
    },
    "training": {
        "model": "Model",
        "epochs": 100,
        "batch_size": 64,
        "loss": "cross entropy",
        "optimizer": "adam",
        "learning_rate": 0.001
    },
    "output_data": {
        "model_save_path": "./path/where/model/should/be/saved",
        "logs_path": "./path/where/tensorboard/runs/should/be/saved"
    }
}
```
