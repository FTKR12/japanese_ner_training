import os
import time
import json

import torch
from torch.utils.data import DataLoader
from transformers import BertForTokenClassification

from utils import get_args, setup_logger, set_seed
from src import preprocess, NerToknizer, NerTokenizerForTest, NerTester, NerDataset

def main(args):
    # build tokenizer and model
    tokenizer = NerTokenizerForTest.from_pretrained(args.model_name)
    model = BertForTokenClassification.from_pretrained(args.model_name, num_labels=9).to(args.device)
    model.load_state_dict(torch.load(args.model_save_path, weights_only=True))
    
    # build dataloader
    dataset = json.load(open(args.data_path, "r"))
    label2id = json.load(open(args.ner_config_path, "r"))
    dataset = preprocess(dataset, label2id)
    num_train = int(len(dataset)*args.train_split)
    num_val = int(len(dataset)*args.val_split)
    dataset_test = dataset[num_train+num_val:]
    
    tester = NerTester(
        args=args,
        label2id=label2id,
        dataset=dataset_test,
        tokenizer=tokenizer,
        model=model
    )
    tester.run()

if __name__ == "__main__":

    args = get_args()
    set_seed(args.seed)

    cur_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    os.makedirs(args.output_dir, exist_ok=True)
    args.output_dir = f'{args.output_dir}/test/{args.model_name}'
    os.makedirs(args.output_dir, exist_ok=True)

    logger = setup_logger('Japanese NER Training', args.output_dir)
    logger.info(str(args).replace(',','\n'))

    # train
    main(args)