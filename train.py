import os
import time
import json

import torch
from torch.utils.data import DataLoader
from transformers import BertForTokenClassification

from utils import get_args, setup_logger, set_seed
from src import preprocess, NerToknizer, NerTokenizerForTest, NerTrainer, NerDataset

def main(args, logger):

    # build tokenizer, model
    tokenizer = NerToknizer.from_pretrained(args.model_name)
    model = BertForTokenClassification.from_pretrained(args.model_name, num_labels=9).to(args.device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)

    # build dataloader
    dataset = json.load(open(args.data_path, "r"))
    label2id = json.load(open(args.ner_config_path, "r"))
    dataset = preprocess(dataset, label2id)

    num_train = int(len(dataset)*args.train_split)
    num_val = int(len(dataset)*args.val_split)
    dataset_train = dataset[:num_train]
    dataset_val = dataset[num_train:num_train+num_val]
    dataset_test = dataset[num_train+num_val:]

    dataloader_train = DataLoader(NerDataset(dataset_train, tokenizer, args.max_length), batch_size=args.batch_size, shuffle=True)
    dataloader_val = DataLoader(NerDataset(dataset_val, tokenizer, args.max_length), batch_size=args.batch_size*2, shuffle=True)
    dataloader = {"train": dataloader_train, "val": dataloader_val}

    logger.info(f'\n{"-"*20} Dataset Stat {"-"*20}\n [TRAIN NUM]: {len(dataset_train)} \n [VAL NUM]: {len(dataset_val)} \n [TEST NUM]: {len(dataset_test)} \n{"-"*20} Dataset Stat {"-"*20}')
    
    # train
    trainer = NerTrainer(
        args=args,
        model=model,
        loader=dataloader,
        optimizer=optimizer,
    )
    trainer.run()


if __name__ == "__main__":

    args = get_args()
    set_seed(args.seed)

    cur_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    os.makedirs(args.output_dir, exist_ok=True)
    args.output_dir = f'{args.output_dir}/train/{args.model_name}'
    os.makedirs(args.output_dir, exist_ok=True)

    logger = setup_logger('Japanese NER Training', args.output_dir)
    logger.info(str(args).replace(',','\n'))

    # train
    main(args, logger)