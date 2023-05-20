"""Datamodule for the WMT transformer model."""
from datasets import load_dataset
from transformers import AutoTokenizer
import logging
import lightning.pytorch as pl
from torch import utils
import os


class WmtModule(pl.LightningDataModule):
  def __init__(
    self, data_dir: str = os.path.join(os.getcwd(), 'data'),
    cache_dir: str = os.path.join(os.getcwd(), 'data-cache'),
    batch_size: int = 64, train_eval_split: float = 0.8,
    max_length: int = 128,
    num_workers: int = 1,
    debug: bool = False):
    super().__init__()
    self.data_dir = data_dir
    self.cache_dir = cache_dir
    self.batch_size = batch_size
    self.train_eval_split = train_eval_split
    self.max_length = max_length
    self._debug = debug
    self.num_workers = num_workers

  def prepare_data(self):
    # download
    load_dataset("wmt16", "ro-en", cache_dir=self.cache_dir)
    AutoTokenizer.from_pretrained(
      "Helsinki-NLP/opus-mt-en-ro",
      cache_dir=self.cache_dir,
      use_fast_tokenizer=True)

  def preprocess_examples(self, examples):
    inputs = [ex['en'] for ex in examples['translation']]
    targets = [ex['ro'] for ex in examples['translation']]
    model_inputs = self.tokenizer(
      inputs, max_length=self.max_length,
      padding="max_length", truncation=True)
    labels = self.tokenizer(targets, max_length=self.max_length,
      padding="max_length", truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

  def setup(self, stage: str):
    logging.info("Setting up data for stage: %s", stage)
    books = load_dataset("wmt16", "ro-en", cache_dir=self.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(
      "Helsinki-NLP/opus-mt-en-ro",
      cache_dir=self.cache_dir,
      use_fast_tokenizer=True)
    tokenizer.src_lang = 'en'
    tokenizer.tgt_lang = 'ro'
    def get_dataset(raw_dataset, split, debug):
      if debug:
        num_samples = 10
      else:
        num_samples = len(raw_dataset)
      raw_dataset = raw_dataset[split].select(range(num_samples))
      return raw_dataset.map(
        self.preprocess_examples,
        num_proc=self.num_workers,
        batched=True,
        remove_columns=raw_dataset.columns,
        desc="Running tokenizer on {} set.".format(split),
      )
      
    if stage == "fit":
      self.train = get_dataset(books, "train", self._debug)
      self.val = get_dataset(books, "validation", self._debug)
    if stage == "test":
      self.test = get_dataset(books, "test", self._debug)
    if stage == "predict":
      self.predict = get_dataset(books, "test", self._debug)

  def train_dataloader(self):
    logging.info("Getting train dataloader")
    #return utils.data.DataLoader(self.train_set, batch_size=self.batch_size)
    return self.train

  def val_dataloader(self):
    logging.info("Getting eval dataloader")
    #return utils.data.DataLoader(self.val_set, batch_size=self.batch_size)
    return self.val

  def test_dataloader(self):
    logging.info("Getting test dataloader")
    return self.test

  def predict_dataloader(self):
    logging.info("Getting predict dataloader")
    return self.predict
