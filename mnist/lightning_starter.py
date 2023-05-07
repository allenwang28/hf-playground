"""An MNIST autoencoder using PyTorch, Huggingface and PyTorch Lightning."""
import os
import torch
import logging
from torch import optim, nn, utils, Tensor
import torch.nn.functional as F
from torchvision.datasets import MNIST
import torchvision
from torchvision.transforms import ToTensor
import lightning.pytorch as pl
from pytorch_lightning.loggers import WandbLogger
from lightning.pytorch.profilers import AdvancedProfiler
from lightning.pytorch.callbacks import DeviceStatsMonitor
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.utilities import grad_norm

import wandb


class LitAutoEncoder(pl.LightningModule):
  def __init__(self, encoder, decoder, lr=1e-3):
    super().__init__()
    self.encoder = encoder
    self.decoder = decoder
    self.lr = lr

  def training_step(self, batch, batch_idx):
    x, y = batch
    x = x.view(x.size(0), -1)
    z = self.encoder(x)
    x_hat = self.decoder(z)
    loss = F.mse_loss(x_hat, x)
    self.log("train_loss", loss)
    return loss

  def configure_optimizers(self):
    optimizer = optim.Adam(self.parameters(), lr=self.lr)
    return optimizer

  def test_step(self, batch, batch_idx):
    x, y = batch
    x = x.view(x.size(0), -1)
    z = self.encoder(x)
    x_hat = self.decoder(z)
    loss = F.mse_loss(x_hat, x)
    self.log("test_loss", loss)
    wandb_logger = self.logger.experiment
    wandb_logger.log({"generated_images": [wandb.Image(x_hat, caption="Generated images")]})
    return loss

  def validation_step(self, batch, batch_idx):
    x, y = batch
    x = x.view(x.size(0), -1)
    z = self.encoder(x)
    x_hat = self.decoder(z)
    loss = F.mse_loss(x_hat, x)
    self.log("val_loss", loss)
    wandb_logger = self.logger.experiment
    wandb_logger.log({"generated_images": [wandb.Image(x_hat, caption="Generated images")]})
    return loss

  def on_before_optimizer_step(self, optimizer):
    # Compute the 2-norm for each layer
    # If using mixed precision, the gradients are already unscaled here
    encoder_norms = grad_norm(self.encoder, norm_type=2)
    decoder_norms = grad_norm(self.decoder, norm_type=2)
    self.log_dict(encoder_norms)
    self.log_dict(decoder_norms)


class MNISTDataModule(pl.LightningDataModule):
  def __init__(self, data_dir: str = os.getcwd(), batch_size: int = 64, train_eval_split: float = 0.8):
    super().__init__()
    self.data_dir = data_dir
    self.batch_size = batch_size
    self.train_eval_split = train_eval_split
    self.transform = torchvision.transforms.Compose([
      torchvision.transforms.ToTensor(),
      torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])

  def prepare_data(self):
    # download
    MNIST(self.data_dir, train=True, download=True)
    MNIST(self.data_dir, train=False, download=True)

  def setup(self, stage: str):
    logging.info("Setting up data for stage: %s", stage)
    if stage == "fit":
      dataset = MNIST(self.data_dir, download=True, train=True, transform=self.transform)
      train_set_size = int(len(dataset) * self.train_eval_split)
      val_set_size = int(len(dataset)) - train_set_size
      seed = torch.Generator().manual_seed(42)
      self.train_set, self.valid_set = utils.data.random_split(dataset, [train_set_size, val_set_size], generator=seed)
    if stage == "test":
      self.mnist_test = MNIST(self.data_dir, download=True, train=False, transform=self.transform)
    if stage == "predict":
      self.mnist_predict = MNIST(self.data_dir, download=True, train=False, transform=self.transform)

  def train_dataloader(self):
    logging.info("Getting train dataloader")
    return utils.data.DataLoader(self.train_set, batch_size=self.batch_size)

  def val_dataloader(self):
    logging.info("Getting eval dataloader")
    return utils.data.DataLoader(self.valid_set, batch_size=self.batch_size)

  def test_dataloader(self):
    return utils.data.DataLoader(self.mnist_test, batch_size=self.batch_size)

  def predict_dataloader(self):
    return utils.data.DataLoader(self.mnist_predict, batch_size=self.batch_size)


def main():
  encoder = nn.Sequential(nn.Linear(28*28, 64), nn.ReLU(), nn.Linear(64, 3))
  decoder = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28*28))
  autoencoder = LitAutoEncoder(encoder=encoder, decoder=decoder)
  mnist = MNISTDataModule(batch_size=128)

  logger = WandbLogger()
  profiler = AdvancedProfiler(dirpath=".", filename="perf_logs")
  checkpoint_callback = ModelCheckpoint(
    save_top_k=10,
    monitor="val_loss",
    mode="min",
    dirpath=os.path.join(os.getcwd(), "checkpoints"),
    filename="sample-mnist-{epoch:02d}-{val_loss:.2f}",
)

  callbacks = [DeviceStatsMonitor(), checkpoint_callback]
  trainer = pl.Trainer(
    max_epochs=100,
    logger=logger,
    profiler=profiler,
    accelerator="gpu",
    accumulate_grad_batches=8,
    devices=1,
    precision="16-mixed",
    callbacks=callbacks)
  trainer.fit(model=autoencoder, datamodule=mnist)
  trainer.test(model=autoencoder, dataloaders=mnist)


if __name__ == '__main__':
  logging.basicConfig(level=logging.INFO)
  main()
