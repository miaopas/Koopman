from contextlib import contextmanager
from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torch import nn
from torch.optim import lr_scheduler as lr_scheduler
from torch.optim.optimizer import Optimizer as Optimizer
from torchmetrics import MaxMetric, MeanMetric


class LitModule(LightningModule):
    """Example of LightningModule for LF.

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        psi,
        model_K,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        criterion: str = "mse",
    ) -> None:
        """Initialize a `LFLitModule`.

        :param model: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        :param compile: Whether compile model for fast training, use with cautious.
        """
        super().__init__()

        if criterion.lower() == "mse":
            self.criterion = torch.nn.MSELoss()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        # self.save_hyperparameters(ignore=["psi", "model_K"],logger=False)
        self.save_hyperparameters(logger=False)

        self.psi = psi
        self.model_K = model_K

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model."""
        psi_x = self.psi(x)
        K_psi_x = self.model_K(psi_x)
        
        return K_psi_x



    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of target labels.
        """
        x = batch
        yhat = self(x[:,:-1,:]) - self.psi(x[:,1:,:])
        y = torch.zeros_like(yhat)
        loss = self.criterion(yhat, y)
        return loss

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of inputs and target
            outputs.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)

        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", lr, on_step=True, on_epoch=False, prog_bar=True, logger=True)

        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """_summary_

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): _description_
            batch_idx (int): _description_
        """
        loss = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        """_summary_"""
        pass

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.model = torch.compile(self.model)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
    


    def predict_multistep(self, x0, Nt):
        self.eval()
        x_pred_list = [x0]

        psi_x0 = self.psi(x0)
        psi_x_pred_list = [psi_x0]

        B = self.psi.generate_B(x0)

        for _ in range(Nt):
            psi_pred = self.model_K(psi_x_pred_list[-1])
            x_pred = torch.matmul(psi_pred, B)
            x_pred_list.append(x_pred.detach())
            psi_x_pred_list.append(psi_pred.detach())

        return torch.stack(x_pred_list, dim=1)

    def predict_onestep(self, x0, Nt):
        self.eval()
        x_pred_list = [x0]

        psi_x = self.psi(x0)
       

        B = self.psi.generate_B(x0)

        for _ in range(Nt):
            psi_pred = self.model_K(psi_x)
            x_pred = torch.matmul(psi_pred, B)
            x_pred_list.append(x_pred.detach())
            psi_x = self.psi(x_pred)

        return torch.stack(x_pred_list, dim=1)
