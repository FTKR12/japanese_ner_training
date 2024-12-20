from typing import List, Dict
import unicodedata

def preprocess(dataset: List, label2id: Dict) -> List:
    for sample in dataset:
        sample['text'] = unicodedata.normalize('NFKC', sample['text'])
        for e in sample["entities"]:
            e['type_id'] = label2id[e['type']]
            del e['type']
    return dataset