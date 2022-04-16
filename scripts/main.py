import argparse
from data import get_dataloaders
from train import train_loop
import numpy as np
import torch
import random
from model import ClinicalBERT
from torch.optim import Adam
import os
import json
import datetime

def get_save_path(output_dir):
  timestamp = datetime.datetime.now()
  date = timestamp.strftime("%Y%m%d")
  time = timestamp.strftime("%H%M%S")
  save_path = f'{output_dir}/{date}/{time}'
  return save_path

def main(args):

  # set seed
  if args.seed is None:
      seed = np.random.randint(low=0, high=99999)
  else:
    seed = args.seed
  print("seed = {}".format(seed))
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = True
  random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  np.random.seed(seed)  

  save_path = get_save_path(args.output_dir)  
  try:
      os.makedirs(save_path, exist_ok = True)
      print(f"Directory {save_path} created successfully")
  except OSError as error:
      print(f"Directory {save_path} can not be created")

  dataloaders = get_dataloaders(args.data_path, args.batch_size)
  model = ClinicalBERT(args.num_classes)
  optimizer = Adam(model.parameters(), lr=args.max_lr, weight_decay=args.weight_decay)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f'running on device {device}')
  
  config = vars(args)
  config['save_path'] = save_path
  config['seed'] = seed
  config['device'] = str(device)
  with open(os.path.join(save_path, "config.json"),"w") as f:
    config = json.dumps(config, indent=4, sort_keys=True)
    f.write(config)
  
  train_loop(dataloaders["train"], 
             dataloaders["val"],
             model, 
             optimizer, 
             args.num_epochs, 
             args.warmup_steps,
             device, 
             save_path, 
             logging_steps=args.logging_steps)


if __name__ == "__main__": 
  parser = argparse.ArgumentParser(description='train indication classifer')
  parser.add_argument('--max_lr', required=True, type=float, help='max learning rate')
  parser.add_argument('--weight_decay', required=True, type=float, help='weight decay')
  parser.add_argument('--warmup_steps', default=None, type=int, help='warmup steps for learning rate scheduler')
  parser.add_argument('--num_epochs', required=True, type=int, help='total number of epochs')
  parser.add_argument('--output_dir', required=True, type=str, help='output directory for model weights and logs')
  parser.add_argument('--logging_steps', default=5, type=int, help='logging interval for validation set metrics')
  parser.add_argument('--batch_size', type=int, help='batch_size')
  parser.add_argument('--seed', default=None, type=int, help='random seed')
  parser.add_argument('--num_classes', default=5, type=int, help='number of classes in dataset')
  parser.add_argument('--data_path', default='../data/dataset.pkl', type=str, help='path to dataset pickle file')

  args = parser.parse_args()

  main(args)
  