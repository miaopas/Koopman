from omegaconf import OmegaConf
import hydra
from functools import partial
import lightning as L
import rootutils
from torch import nn
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
import logging

# configure logging at the root level of Lightning
# logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
import torch
from lightning.pytorch.loggers import WandbLogger
from src.data.datamodule import ODEDataModule
from src.data.ode_targets import DuffingOscillator
from src.modules import ConstantMatrixMultiplier, PsiNN, LitModule


from src.utils import (
    RankedLogger,
    log_hyperparameters,
    instantiate_callbacks,
    instantiate_loggers,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)


def train(cfg):
    
    L.seed_everything(123, workers=True)

    datamodule =  ODEDataModule(
        train_val_test_split=(512*50, 1024, 1024),
        batch_size=2048, num_workers=31, pin_memory=False,
        target=DuffingOscillator, length=50, dt=1e-3,t_step=0.25,dim=2,
    )

    state_dim = 2
    layer_sizes = [256, 256, 256]
    n_psi_train = 22
    activation_func = "tanh"
    n_psi = 1 + state_dim + n_psi_train

    dict_nn = PsiNN(
        inputs_dim=state_dim,
        layer_sizes=layer_sizes,
        n_psi_train=n_psi_train,
        activation_func=activation_func,
    )
    # model_K = ConstantMatrixMultiplier(n_psi=n_psi)

    model_K = nn.Linear(n_psi, n_psi)

    optimizer = partial(torch.optim.Adam, lr=5e-3)
    scheduler = partial(torch.optim.lr_scheduler.ReduceLROnPlateau,mode="min", factor=0.5, patience=3)

    model = LitModule(dict_nn, model_K, optimizer, scheduler, compile=False)

    callbacks = instantiate_callbacks(cfg.get("callbacks"))

    logger = instantiate_loggers(cfg.get("logger"))

    for lg in logger:
        if isinstance(lg, WandbLogger):
            lg.watch(model, log_freq=100)

    trainer = L.Trainer(
        default_root_dir=cfg.paths.run_dir,
        max_epochs=1000,
        accelerator='gpu', devices=1, callbacks=callbacks, logger=logger, precision="64-true")

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    
    trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":

    cfg = OmegaConf.load('configs/config.yaml')
    train(cfg)
