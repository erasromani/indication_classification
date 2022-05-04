import torch
import pandas as pd
import numpy as np
import random
import os, pickle5 as pickle, json, requests
import argparse

from transformers import DataCollatorForLanguageModeling
from datasets import load_dataset, load_from_disk
from pathlib import Path
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
from utils import load_data
from datasets import Dataset
from transformers import Trainer, TrainingArguments
from dotenv import load_dotenv
from utils import get_save_path

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
    config['num_gpus'] = torch.cuda.device_count()
    with open(os.path.join(save_path, "config.json"),"w") as f:
        config = json.dumps(config, indent=4, sort_keys=True)
        f.write(config)

    dataset = load_from_disk(args.data_dir)

    tokenizer = AutoTokenizer.from_pretrained("yikuan8/Clinical-Longformer")
    model = AutoModelForMaskedLM.from_pretrained("yikuan8/Clinical-Longformer")
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    training_args = TrainingArguments(
        output_dir=save_path,
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        warmup_steps=args.warmup_steps,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        logging_first_step=True,
        save_strategy="steps",
        save_steps=args.save_steps,
        evaluation_strategy = "steps",
        eval_steps=args.eval_steps,
        learning_rate=args.max_lr,
        weight_decay=args.weight_decay,
        fp16=True,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test']
    )

    trainer.train()
    trainer.save_model(os.path.join(save_path, "model_last_epoch.pt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='finetune clinical longformer model with masked language modeling')
    parser.add_argument('--seed', default=None, type=int, help='random seed')
    parser.add_argument('--max_lr', default=2e-5, type=float, help='max learning rate')
    parser.add_argument('--weight_decay', default=0.01, type=float, help='weight decay')
    parser.add_argument('--per_device_batch_size', default=7, type=int, help='per device batch size')
    parser.add_argument('--warmup_steps', default=1000, type=int, help='warmup steps for learning rate scheduler')
    parser.add_argument('--logging_steps', default=100, type=int, help='logging steps')
    parser.add_argument('--eval_steps', default=2000, type=int, help='evaluation steps')
    parser.add_argument('--save_steps', default=2000, type=int, help='save steps')
    parser.add_argument('--num_epochs', default=2, type=int, help='total number of epochs')
    parser.add_argument('--data_dir',
                        default="/gpfs/data/geraslab/ekr6072/projects/study_indication/data/mlm_dataset",
                        type=str,
                      help='directory to for mlm dataset')
    parser.add_argument('--output_dir',
                        required=True,
                        type=str,
                        help='output directory for language model')

    args = parser.parse_args()
    load_dotenv()

    main(args)
