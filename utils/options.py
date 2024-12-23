import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Japanese NER Traning")
    
    # general params
    parser.add_argument('--seed', default=123, type=int)
    parser.add_argument('--output_dir', default='./output')
    parser.add_argument('--data_path', default='./dataset/ner.json')
    parser.add_argument('--device', default='cuda:7')
    parser.add_argument('--ner_config_path', default='./dataset/ner_config.json')
    parser.add_argument('--train_split', default=0.75, type=float)
    parser.add_argument('--val_split', default=0.15, type=float)
    parser.add_argument('--test_split', default=0.10, type=float)

    # model params
    parser.add_argument('--model_name', default='cl-tohoku/bert-base-japanese-whole-word-masking')
    parser.add_argument('--max_length', default=128, type=int)
    
    # experimental params
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--lr', default=2e-5, type=float)
    parser.add_argument('--model_save_path', default='./model/model.bin')
    
    args = parser.parse_args()
    return args