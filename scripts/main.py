import argparse
from data import get_dataloaders
from train import train_loop
import numpy as np
import torch
import random
from models import resolve_model
from torch.optim import Adam
import os
import json
from utils import get_save_path


def main(args):

  # set seed
  if args.seed is None:
      seed = np.random.randint(low=0, high=99999)
  else:
    seed = args.seed
  print("seed = {}".format(seed))
  torch.backends.cudnn.deterministic = True
  if args.model == 'clinicalslidingwindow':
    torch.backends.cudnn.benchmark = False
  else:
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

  dataloaders = get_dataloaders(args.data_path, args.batch_size, args.model, exclude_classes=args.exclude_classes)
  model = resolve_model(args.model, args.num_classes, reduction=args.reduction, c=args.c)
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
             logging_steps=args.logging_steps,
             clip_grad_norm=args.clip_grad_norm,
             grad_accum_steps=args.grad_accum_steps)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='train indication classifer')
  parser.add_argument('--model', default='clinicalslidingwindow', type=str, help='model name')
  parser.add_argument('--max_lr', required=True, type=float, help='max learning rate')
  parser.add_argument('--weight_decay', required=True, type=float, help='weight decay')
  parser.add_argument('--warmup_steps', default=None, type=int, help='warmup steps for learning rate scheduler')
  parser.add_argument('--num_epochs', required=True, type=int, help='total number of epochs')
  parser.add_argument('--output_dir', required=True, type=str, help='output directory for model weights and logs')
  parser.add_argument('--logging_steps', default=10, type=int, help='logging interval for validation set metrics')
  parser.add_argument('--batch_size', type=int, help='batch_size')
  parser.add_argument('--seed', default=None, type=int, help='random seed')
  parser.add_argument('--c', default=2.0, type=float, help='weighting parameter for mean/max clinicalslidingwindow reduction')
  parser.add_argument('--clip_grad_norm', default=1.0, type=float, help='gradient clipping value for gradient norm')
  parser.add_argument('--num_classes', default=5, type=int, help='number of classes in dataset')
  parser.add_argument('--grad_accum_steps', default=4, type=int, help='number of steps to perform gradient accumation for')
  parser.add_argument('--reduction', default="mean/max", type=str, help='aggregation method for sliding window')
  parser.add_argument('--data_path', default='../data/dataset.pkl', type=str, help='path to dataset pickle file')
  parser.add_argument('--exclude_classes', default=['unknown'], nargs='+', type=str, help='class names to exclude')

  args = parser.parse_args()

  main(args)
  