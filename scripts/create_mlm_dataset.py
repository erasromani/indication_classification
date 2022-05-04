import torch
import pandas as pd
import numpy as np
import random
import os, pickle5 as pickle, json, requests
import argparse

from functools import partial
from datasets import load_dataset, load_from_disk
from pathlib import Path
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
from utils import load_data
from datasets import Dataset
from utils import get_save_path

def tokenize_and_chunk(texts, tokenizer):
  return tokenizer(
                  texts["text"],
                  truncation=True,
                  max_length=1024,
                  return_overflowing_tokens=True
                  )

def main(args):

    # set seed
    if args.seed is None:
        seed = np.random.randint(low=0, high=99999)
    else:
        seed = args.seed
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
    config = vars(args)
    config['save_path'] = save_path
    config['seed'] = seed
    with open(os.path.join(save_path, "config.json"),"w") as f:
        config = json.dumps(config, indent=4, sort_keys=True)
        f.write(config)

    if args.exclude_acn_path is None:
        exclude_acn = []
    else:
        exclude_acn = np.loadtxt(args.exclude_acn_path, dtype=str, skiprows=0)

    # TODO: log Acc in train and test set
    all_texts = []
    all_acns = []
    for data_file in args.data_files:
        data = load_data(data_file)
        data = data.drop_duplicates(subset='Acc', keep='first')
        column = list({'RawReport', 'Report'}.intersection(data.columns))
        assert len(column) == 1, 'invalid number of matching columns'
        column = column[0]
        acns = list(data['Acc'].values)
        texts = list(data[column].values)
        all_texts.extend(texts)
        all_acns.extend(acns)
    assert len(all_acns) == len(all_texts), "acn list and report list have mismatch lengths"

    text_data = []
    for acn, data in zip(all_acns, all_texts):
        if isinstance(data, str) and str(acn) not in exclude_acn:
            text_data.append(data)
    del all_texts
    del all_acns

    df = pd.DataFrame({'text': text_data})
    dataset = Dataset.from_pandas(df)

    tokenizer = AutoTokenizer.from_pretrained("yikuan8/Clinical-Longformer")
    dataset = dataset.map(partial(tokenize_and_chunk, tokenizer=tokenizer), batched=True, remove_columns=["text"])
    dataset = dataset.train_test_split(test_size=args.eval_size)
    dataset.save_to_disk(args.output_dir)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='construct tokenized dataset for masked language modeling with clinical longformer')
    parser.add_argument('--output_dir',
                        required=True,
                        type=str,
                        help='output directory for mlm dataset')
    parser.add_argument('--seed', default=42, type=int, help='random seed')
    parser.add_argument('--data_files', 
                        nargs='+',
                        default=[
                            "/gpfs/data/geraslab/Jan/data/rad_reports/2022.01.combined_rad_reports/ultrasound.xlsx",
                            "/gpfs/data/geraslab/Jan/data/rad_reports/2022.01.combined_rad_reports/MRI.xlsx",
                            "/gpfs/data/geraslab/Jan/data/rad_reports/2022.01.combined_rad_reports/MG.pkl",
                        ],
                        type=str,
                        help='paths to radiology all radiology reports')
    parser.add_argument('--exclude_acn_path',
                        default=None,
                        type=str,
                        help='path to text file containing list of accession numbers to exclude; first row will be skipped upon loading')
    parser.add_argument('--eval_size',
                        default=5000,
                        type=int,
                        help='size of evaluation dataset')

    args = parser.parse_args()

    main(args)
