import torch
from torch.utils.data import Dataset

class NerDataset(Dataset):
  
  def __init__(self, dataset, tokenizer, max_length):
    self.dataset = dataset
    self.tokenizer = tokenizer
    self.max_length = max_length

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, index):
    text = self.dataset[index]["text"]
    entities = self.dataset[index]["entities"]
    encoding = self.tokenizer.encode_plus_tagged(text, entities, max_length=self.max_length)

    input_ids = torch.tensor(encoding["input_ids"])
    token_type_ids = torch.tensor(encoding["token_type_ids"])
    attention_mask = torch.tensor(encoding["attention_mask"])
    labels = torch.tensor(encoding["labels"])

    return {
      "input_ids": input_ids,
      "token_type_ids": token_type_ids,
      "attention_mask": attention_mask,
      "labels": labels
    }