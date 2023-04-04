from typing import Any, List, Optional

import lightning as L
import torch
from ema_pytorch import EMA
from lightning import Trainer, Callback
from lightning.pytorch.loggers import WandbLogger
from torch import nn
from torch.utils.data import DataLoader
import os

import sys
sys.path.append("..") 
from data.utils import fractional_random_split


""" Model """


class Model(L.LightningModule):
    def __init__(
        self,
        lr: float,
        lr_beta1: float,
        lr_beta2: float,
        lr_eps: float,
        lr_weight_decay: float,
        ema_beta: float,
        ema_power: float,
        model: nn.Module,        
    ):
        super().__init__()
        self.lr = lr
        self.lr_beta1 = lr_beta1
        self.lr_beta2 = lr_beta2
        self.lr_eps = lr_eps
        self.lr_weight_decay = lr_weight_decay

        # CLIP Model
        self.model = model
        self.model_ema = EMA(self.model, beta=ema_beta, power=ema_power)

    @property
    def device(self):
        return next(self.model.parameters()).device
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            list(self.parameters()),
            lr=self.lr,
            betas=(self.lr_beta1, self.lr_beta2),
            eps=self.lr_eps,
            weight_decay=self.lr_weight_decay,
        )
        return optimizer
    
    def common_step(self, batch):
        batch = batch | {"return_loss": True}
        loss = self.model(**batch)['loss']
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.common_step(batch)
        self.log("train_loss", loss)
        # Update EMA model and log decay
        self.model_ema.update()
        self.log("ema_decay", self.model_ema.get_current_decay())
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self.common_step(batch)
        self.log("valid_loss", loss)
        return loss


""" Datamodule """


class Datamodule(L.LightningDataModule):
    def __init__(
        self,
        dataset,
        *,
        val_split: float,
        batch_size: int,        
        num_workers: int,
        pin_memory: bool = False,
        **kwargs: int,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.val_split = val_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.data_train: Any = None
        self.data_val: Any = None

    def setup(self, stage: Any = None) -> None:
        split = [1.0 - self.val_split, self.val_split]
        self.data_train, self.data_val = fractional_random_split(self.dataset, split)
        assert len(self.data_val) > 0, "Validation set is empty, increase split ratio."

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )


""" Callbacks """


def get_wandb_logger(trainer: Trainer) -> Optional[WandbLogger]:
    """Safely get Weights&Biases logger from Trainer."""

    if isinstance(trainer.logger, WandbLogger):
        return trainer.logger

    print("WandbLogger not found.")
    return None


class SampleLogger(Callback):
    def __init__(
        self,
        num_items: int,
        processor,
    ) -> None:
        self.num_items = num_items
        self.processor = processor

        self.log_next = False

    def on_validation_epoch_start(self, trainer, pl_module):
        self.log_next = True

    def on_validation_batch_start(
        self, trainer, pl_module, batch, batch_idx
    ):
        if self.log_next:
            self.log_sample(trainer, pl_module, batch)
            self.log_next = False

    @torch.no_grad()
    def log_sample(self, trainer, pl_module, batch):
        is_train = pl_module.training
        if is_train:
            pl_module.eval()

        if not os.environ['SKIP_LOGGER']:
            wandb_logger = get_wandb_logger(trainer).experiment


        model = pl_module.model_ema.ema_model


        pixel_values = batch['pixel_values']
                
        with torch.no_grad():
            image_embeddings = [model.get_image_features(pixel_values)[i] for i in range(self.num_items)]

        texts = ['sks', 'ksk']
        text_inputs = self.processor(text=texts, return_tensors='pt', padding=True)
        text_embeddings = [model.get_text_features(**text_inputs)[i] for i in range(2)]

        # find cosine similarity for every text and image combination
        ## ## ##


        # log images on wandb and the probabilities


            
            
        if is_train:
            pl_module.train()
