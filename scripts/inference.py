from __future__ import print_function
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
from train import eval_loop
from data import id2category
from transformers import AutoTokenizer


def process_data(data, tokenizer, truncation=True, padding=True, max_length=512):
  texts = [data] if isinstance(data, str) else data
  encodings = tokenizer(texts, truncation=truncation, padding=padding, max_length=max_length)
  return encodings


def get_data(data_path, model):

  with open(data_path, 'r') as f:
    data = f.read()

  if model == 'clinicalbiobert':
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    data = process_data(data, tokenizer, truncation=True, padding=True, max_length=512)
    return data

  elif model == 'clinicallongformer':

    tokenizer = AutoTokenizer.from_pretrained("yikuan8/Clinical-Longformer")    
    data = process_data(data, tokenizer, truncation=True, padding=True, max_length=1024)
    return data

  elif model == 'clinicalslidingwindow':

    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    data = process_data(data, tokenizer, truncation=False, padding=False, max_length=None)
    return data

  else:
    ValueError(f'invalid model name {model} entered')


def main(args):

    processed_data = get_data(args.data_path, args.model)
    model = resolve_model('clinicallongformer', 5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    input_ids = torch.tensor(processed_data['input_ids']).to(device)
    attention_mask = torch.tensor(processed_data['attention_mask']).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask, inference=True)
        preds = logits.argmax(-1)
    preds = [id2category[id] for id in preds.cpu()]

    with open(args.output_path, 'w') as f:
        f.writelines(preds)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='train indication classifer')
    parser.add_argument('--model', default='clinicallongformer', type=str, help='model name')
    parser.add_argument('--c', default=None, type=float, help='weighting parameter for mean/max clinicalslidingwindow reduction')
    parser.add_argument('--num_classes', default=5, type=int, help='number of classes in dataset')
    parser.add_argument('--reduction', default="attention", type=str, help='aggregation method for sliding window')
    parser.add_argument('--pretrain_path', default=None, type=str, help='path to model pretrained weights')
    parser.add_argument('--data_path', required=True, type=str, help='path to text file containing report')    
    parser.add_argument('--output_path', required=True, type=str, help='path to output prediction')
    args = parser.parse_args()

    main(args)
    