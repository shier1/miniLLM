import os
import torch
import json
from transformers import AutoTokenizer
from torch.utils.data import Dataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class PretrainedDataset(Dataset):
    def __init__(self, config):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(config.dataset.tokenizer.path)
        self.max_length = config.dataset.max_length
        self.samplers = self.load_data(config.dataset.data_path)
    
    def load_data(self, path):
        samplers = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                samplers.append(data)
        return samplers
    
    def __len__(self):
        return len(self.samplers)
    
    def __getitem__(self, index):
        sample = self.samplers[index]
        text = f"{self.tokenizer.bos_token}{str(sample['text'])}{self.tokenizer.eos_token}"
        encoding = self.tokenizer(
            text,
            max_length = self.max_length,
            padding = "max_length",
            truncation = True,
            return_tensors="pt"
        )
        input_ids = encoding.input_ids.squeeze()
        loss_mask = (input_ids != self.tokenizer.pad_token_id)

        print(input_ids.shape)
        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        return X, Y, loss_mask

# if __name__ == "__main__":
#     from config import get_config
#     config = get_config("./configs/miniLLM.yaml")
#     dataset = PretrainedDataset(config=config)
#     print(len(dataset))
#     for X, Y, loss_mask in dataset:
#         print(X.shape, Y.shape, loss_mask.shape)
#         break