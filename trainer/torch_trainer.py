import numpy as np
import cv2
import os
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split
from general_utils import TorchDataset
from utils.training_utils import (
    get_metrics,
    format_time,
    count_parameters,
    get_device,
)
from general_utils import precision, recall, accuracy
import time


class TorchTrainer:
    """
    Class representing the entity who handles training.

    # Parameters:
        model : PyTorch model inheriting from nn.Module
        data : TorchDataset wrapping a BasicDataset
        epochs : integer number of training epochs.
        loss_func : str identifying loss function to be optimized.
        optimizer : str identifying optimizer to be employed.
        lr : learning rate.
        model : python class of model to be trained, or actual model.
        log_dir : str identifying path where training results are written.
        save_dir : str identifying path where model is saved.
    """

    def __init__(
        self,
        model: nn.Module,
        dataset: TorchDataset,
        epochs: int,
        loss_func: str,
        optimizer: str,
        batch_size: int,
        lr: float = 1e-3,
        log_dir: str = "./runs",
        save_dir: str = "./checkpoints",
    ):
        self.device = get_device()

        self.model = model.float().to(self.device)

        self.dataset = dataset

        self.epochs = epochs

        if loss_func.replace(" ", "").lower() == "classification":
            self.loss_func = nn.BCELoss()
        else:
            self.loss_func = nn.MSELoss()

        self.batch_size = batch_size
        self.lr = lr

        if optimizer.replace(" ", "").lower() == "adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        else:
            self.optimizer = optim.rmsprop(self.model.parameters(), lr=self.lr)

        self.log_dir = os.path.join(log_dir, self.model.__class__.__name__)
        self.save_dir = (
            os.path.join(save_dir, self.model.__class__.__name__) + ".pth"
        )
        self.writer = SummaryWriter(log_dir=self.log_dir)

    def _training_step(self, X, y):
        # moving from NHWC convention to NCHW
        X = X.permute(0, 3, 1, 2)
        # reshape to (N_BATCHES, N_CLASSES)
        y = y.view(-1, 1).float()
        # zeros out previous gradients
        self.model.zero_grad()
        # run model and get predictions
        y_hat = self.model(X.float())
        # calculate loss
        loss = self.loss_func(y_hat, y)
        # calculate gradients updating
        loss.backward()
        # update weights
        self.optimizer.step()
        return y_hat, loss

    def _validation_step(self, X, y):
        # no optimization involved while validating, so turn of backprop
        with torch.no_grad():
            X = X.permute(0, 3, 1, 2)
            # reshape to (N_BATCHES, N_CLASSES)
            y = y.view(-1, 1).float()
            y_hat = self.model(X.float())
            # calculate loss
            loss = self.loss_func(y_hat, y)
            return y_hat, loss

    def _log_metrics(self, metrics: dict, epoch: int, stage: str = "train"):
        if not metrics:
            print("Empty metrics dict, nothing logged in tensorboard")

        # make everything lowercase explicitly to avoid errors do to different
        # spells of the same metric between function and caller
        supported_metrics = ["accuracy", "loss", "precision", "recall"]
        metrics = {k.lower(): v for k, v in metrics.items()}

        for metric in supported_metrics:
            if metric in metrics.keys():
                scalar = metrics[metric.lower()]
                log_label = os.path.join(
                    metric.capitalize(), stage.capitalize()
                )
                self.writer.add_scalar(log_label, scalar, epoch)

        self.writer.flush()

    def _log_images(self, X, y_hat, epoch: int, n_imgs: int = 4):
        # tensor in NCHW convention
        batch_size, _, h_img, w_img = X.size()
        # Extract n random indeces to pick up images from last batch
        # processed
        log_idxs = []
        while len(log_idxs) != n_imgs:
            log_idxs = np.unique(
                np.random.randint(0, y_hat.size()[0], size=n_imgs)
            )
        # Display each of the n images in tensorboard inserting text
        # showing how much the model is confident about predicting
        # a car in the image
        for idx in log_idxs:
            # from CHW to HWC to work with opencv
            raw_img = np.moveaxis(X[idx].numpy(), 0, -1)
            # annotate image with the prediction (car probability)
            annotated_img = cv2.putText(
                raw_img,
                "{:.4f}".format(y_hat[idx].item()),
                (round(h_img * 0.1), round(w_img * 0.1)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (1, 0, 0),
                2,
                cv2.LINE_AA,
            )

            # Log annotated image on tensorboard
            self.writer.add_image(
                "Epoch{}/IMG{}".format(epoch, idx),
                annotated_img,
                global_step=epoch,
                dataformats="HWC",
            )
            self.writer.flush()

    def start_training(self):
        """ Starts training the model. """
        # Make training reproducible
        if str(self.device) == "cuda":
            torch.backend.cudnn.deterministic = True
            torch.backend.cudnn.benchmark = True

        # Get some infos about image we will deal with, if code get split up,
        # it would be no more required here
        h_img, w_img, ch_img = self.dataset[0][0].shape
        # fraction of data to be used to validate training results
        val_frac = 0.2
        # frequency of validations
        val_step = 2
        # Randomly split dataset in training and validation
        lenghts = [
            round((1 - val_frac) * len(self.dataset)),
            round(val_frac * len(self.dataset)),
        ]
        torch.manual_seed(0)
        train_dataset, val_dataset = random_split(self.dataset, lenghts)
        # Get loaders to shuffle, batch and lazily load data
        train_data = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            worker_init_fn=np.random.seed(0),
        )
        val_data = DataLoader(
            val_dataset,
            batch_size=self.batch_size // 2,
            shuffle=True,
            pin_memory=True,
            worker_init_fn=np.random.seed(0),
        )

        # Advise user about where and how monitor training
        print(
            "Monitor training running tensorboard through the following cmd:"
            + "\n"
            + "tensorboard --logdir=./absolute/path/to/runs/directory"
        )
        # tensorboard --logdir=./Teraki/rd_video/benchmarks/training/runs

        # PyTorch training loop
        start = time.time()
        print("\nTraining started on {}".format(self.device))
        for epoch in range(self.epochs):
            epoch_start = time.time()
            # Activate layers like dropout and batch norm telling our model
            # we are in training stage
            self.model.train()
            losses = []
            preds = []
            ground_truth = []
            for batch in train_data:
                X, y = batch
                # moving from NHWC convention to NCHW
                X = X.permute(0, 3, 1, 2).float().to(self.device)
                # reshape to (N_BATCHES, N_CLASSES)
                y = y.view(-1, 1).float().to(self.device)
                ground_truth.extend(y.tolist())
                # zeros out previous gradients
                self.model.zero_grad()
                # run model and get predictions
                y_hat = self.model(X)
                preds.extend(y_hat.tolist())
                # calculate loss
                loss = self.loss_func(y_hat, y)
                losses.append(loss.item())
                # calculate gradients updating
                loss.backward()
                # update weights
                self.optimizer.step()

            # Validation loop
            if (
                epoch % val_step == 0 and epoch != 0
            ) or epoch == self.epochs - 1:
                self.model.eval()
                val_losses = []
                val_preds = []
                val_ground_truth = []
                # no optimization involved while validating
                for val_batch in val_data:
                    with torch.no_grad():
                        X_val, y_val = val_batch
                        # moving from NHWC convention to NCHW
                        X_val = (
                            X_val.permute(0, 3, 1, 2).float().to(self.device)
                        )
                        # reshape to (N_BATCHES, N_CLASSES)
                        y_val = y_val.view(-1, 1).float().to(self.device)
                        val_ground_truth.extend(y_val.tolist())
                        y_val_hat = self.model(X_val)
                        val_preds.extend(y_val_hat.tolist())
                        # calculate loss
                        val_loss = self.loss_func(y_val_hat, y_val)
                        val_losses.append(val_loss.item())

                # Log validation scalar results
                val_tp, val_fp, val_fn, val_tn = get_metrics(
                    val_preds, val_ground_truth
                )
                val_accuracy = accuracy(val_tp, val_fp, val_fn, val_tn)
                val_precision = precision(val_tp, val_fp)
                val_recall = recall(val_tp, val_fn)
                self.writer.add_scalar(
                    "Loss/val", np.array(val_losses).mean(), epoch
                )
                self.writer.add_scalar("Accuracy/val", val_accuracy, epoch)
                self.writer.add_scalar("Precision/val", val_precision, epoch)
                self.writer.add_scalar("Recall/val", val_recall, epoch)

            # Each 4 epochs log examples of images with predictions
            if epoch % 4 == 0 or epoch == self.epochs - 1:
                # Extract 5 random indeces to pick up images from last batch
                # processed
                log_idxs = []
                while len(log_idxs) != 5:
                    log_idxs = np.unique(
                        np.random.randint(0, y.size()[0], size=5)
                    )
                # Display each of the 5 images in tensorboard inserting text
                # showing how much the model is confident about predicting
                # a car in the image
                for idx in log_idxs:
                    # from CHW to HWC to work with opencv
                    raw_img = np.moveaxis(X.cpu()[idx].numpy(), 0, -1)
                    # annotate image with the prediction (car probability)
                    annotated_img = cv2.putText(
                        raw_img,
                        "{:.4f}".format(y_hat[idx].item()),
                        (round(h_img * 0.1), round(w_img * 0.1)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (1, 0, 0),
                        2,
                        cv2.LINE_AA,
                    )

                    # Log annotated image on tensorboard
                    self.writer.add_image(
                        "Epoch{}/IMG{}".format(epoch, idx),
                        annotated_img.get().astype("float"),
                        global_step=epoch,
                        dataformats="HWC",
                    )

            # Log training scalar results
            tp, fp, fn, tn = get_metrics(preds, ground_truth)
            train_accuracy = accuracy(tp, fp, fn, tn)
            train_precision = precision(tp, fp)
            train_recall = recall(tp, fn)
            self.writer.add_scalar(
                "Loss/train", np.array(losses).mean(), epoch
            )
            self.writer.add_scalar("Accuracy/train", train_accuracy, epoch)
            self.writer.add_scalar("Precision/train", train_precision, epoch)
            self.writer.add_scalar("Recall/train", train_recall, epoch)
            self.writer.flush()
            epoch_end = time.time()
            # Printing on console training results
            if (
                epoch % val_step == 0 and epoch != 0
            ) or epoch == self.epochs - 1:
                print(
                    "EPOCH {} - "
                    "Train Loss: {:.4f} - "
                    "Train Accuracy: {:.4f} - "
                    "Val Loss: {:.4f} - "
                    "Val Accuracy: {:.4f} - "
                    "Time: {}".format(
                        epoch,
                        np.array(losses).mean(),
                        train_accuracy,
                        np.array(val_losses).mean(),
                        val_accuracy,
                        format_time(epoch_end - epoch_start),
                    )
                )
            else:
                print(
                    "EPOCH {} - "
                    "Train Loss: {:.4f} - "
                    "Train Accuracy: {:.4f} - "
                    "Time: {}".format(
                        epoch,
                        np.array(losses).mean(),
                        train_accuracy,
                        format_time(epoch_end - epoch_start),
                    )
                )

        self.writer.flush()
        self.writer.close()

        # saving just model params (recommended pytorch way)
        torch.save(self.model.state_dict(), self.save_dir)

        end = time.time()
        elapsed = end - start

        # Print basic training statistics
        print(
            "\nWell done! Model trained for {} epochs in {}, "
            "taking on average {} per epoch "
            "\nModel params saved in: {} "
            "\nTraining results log at: {}".format(
                self.epochs,
                format_time(elapsed),
                format_time(elapsed / self.epochs),
                self.save_dir,
                self.log_dir,
            )
        )
