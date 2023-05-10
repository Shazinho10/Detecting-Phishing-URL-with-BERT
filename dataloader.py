from torch.utils.data import Dataset, DataLoader
import torch

class extracted_data(Dataset):
  def __init__(self, text, url, labels, tokenizer):

    self.tokenizer = tokenizer
    self.text = text
    self.url = url
    self.source_max_len = 512
    self.labels = labels

  def __len__(self):
    return len(self.text)

  def __getitem__(self, idx):
    self.text_encoding = self.tokenizer(self.text[idx],
                                  max_length=self.source_max_len,
                                  padding='max_length',
                                  truncation=True,
                                  return_attention_mask=True,
                                  add_special_tokens=True,
                                  return_tensors='pt')
    

    self.url_encoding = self.tokenizer(self.url[idx],
                                  max_length=self.source_max_len,
                                  padding='max_length',
                                  truncation = True,
                                  return_attention_mask=True,
                                  add_special_tokens=True,
                                  return_tensors='pt')
    
    return {"text_input_ids": self.text_encoding["input_ids"],
            "text_token_type_ids": self.text_encoding["token_type_ids"],
            "text_attention_mask": self.text_encoding["attention_mask"],
            "labels": self.labels}
            
            # "url_input_ids": torch.tensor(self.url_encoding(["input_ids"])),
            # "url_token_type_ids": torch.tensor(self.url_encoding["token_type_ids"]),
            # "url_attention_mask": torch.tensor(self.url_encoding["attention_mask"])}