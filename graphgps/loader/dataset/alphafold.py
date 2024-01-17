from typing import List
import os
import os.path as osp
import torch
from torch_geometric.data import Dataset
import re

class Alphafold(Dataset):
    def __init__(self):
        super().__init__()
        self.root = '/content/41k_prot_foldseek'

    @property
    def processed_file_names(self) -> List[str]:
        return [f for f in os.listdir(self.root) if f.endswith('.pt')]

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data_path = osp.join(self.root, self.processed_file_names[idx])
        data = torch.load(data_path)

        # Convert the data to bfloat16
        for key in data.keys():
            if torch.is_tensor(data[key]):
                data[key] = data[key].to(dtype=torch.bfloat16)

        return data
