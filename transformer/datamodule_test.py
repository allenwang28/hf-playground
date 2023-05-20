"""Tests for datamodule.py"""
import os
import pytest
from unittest.mock import Mock, patch
from datamodule import WmtModule

# This sets up the WmtModule with default parameters for testing
@pytest.fixture
def wmt_module():
  return WmtModule()

# Test that default parameters are set correctly
def test_default_parameters(wmt_module):
  assert wmt_module.data_dir == os.path.join(os.getcwd(), 'data')
  assert wmt_module.cache_dir == os.path.join(os.getcwd(), 'data-cache')
  assert wmt_module.batch_size == 64
  assert wmt_module.train_eval_split == 0.8
  assert wmt_module.max_length == 128
  assert wmt_module._debug == False
  assert wmt_module.num_workers == 1

# Test that prepare_data method correctly calls load_dataset and AutoTokenizer.from_pretrained
@patch('your_module.load_dataset')
@patch('your_module.AutoTokenizer.from_pretrained')
def test_prepare_data(mock_from_pretrained, mock_load_dataset, wmt_module):
  wmt_module.prepare_data()
  mock_load_dataset.assert_called_once_with("wmt16", "ro-en", cache_dir=wmt_module.cache_dir)
  mock_from_pretrained.assert_called_once_with("Helsinki-NLP/opus-mt-en-ro", cache_dir=wmt_module.cache_dir, use_fast_tokenizer=True)

# Test that the setup method correctly sets the train, val, and test properties
@patch('your_module.load_dataset')
@patch('your_module.AutoTokenizer.from_pretrained')
def test_setup(mock_from_pretrained, mock_load_dataset, wmt_module):
  # Mock the return values from load_dataset and from_pretrained methods
  mock_dataset = Mock()
  mock_load_dataset.return_value = mock_dataset
  mock_tokenizer = Mock()
  mock_from_pretrained.return_value = mock_tokenizer

  # Call the setup method for each stage and check that the properties are set correctly
  for stage in ['fit', 'test', 'predict']:
    wmt_module.setup(stage)
    if stage == 'fit':
      assert wmt_module.train is not None
      assert wmt_module.val is not None
    if stage == 'test':
      assert wmt_module.test is not None
    if stage == 'predict':
      assert wmt_module.predict is not None

# Test that the train_dataloader, val_dataloader, test_dataloader, and predict_dataloader methods return the correct properties
def test_dataloaders(wmt_module):
  assert wmt_module.train_dataloader() == wmt_module.train
  assert wmt_module.val_dataloader() == wmt_module.val
  assert wmt_module.test_dataloader() == wmt_module.test
  assert wmt_module.predict_dataloader() == wmt_module.predict