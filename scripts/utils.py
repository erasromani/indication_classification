import torch
import datetime
import os, pickle5 as pickle, json, requests
import pandas as pd

from pathlib import Path


def epoch_iter(num_epochs, dataloader):
    steps_per_epoch = len(dataloader)
    for epoch in range(num_epochs):
      for step in range(steps_per_epoch):
        yield epoch

def load_data(path):
  path = Path(path)
  if path.suffix == ".xlsx":
    data = pd.read_excel(path, engine='openpyxl', index_col=0)
    data = data.drop_duplicates(subset='Acc', keep='first')
    return data
  elif path.suffix == ".pkl":
    with open(path, 'rb') as f:
      data = pickle.load(f)
    return data

def save_weights(model, optimizer, filename, **kwargs):
    """
    Save all weights necessary to resume training
    """
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    for key, val in kwargs.items():
      state[key] = val
    torch.save(state, filename)

def get_save_path(output_dir):
  timestamp = datetime.datetime.now()
  date = timestamp.strftime("%Y%m%d")
  time = timestamp.strftime("%H%M%S")
  save_path = f'{output_dir}/{date}/{time}'
  return save_path

