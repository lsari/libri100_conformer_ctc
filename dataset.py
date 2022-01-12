# Author: Leda Sari
import os
import json
import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

seed = 2020
# torch.manual_seed(seed)
np.random.seed(seed)

class LibriSpeechDataset(Dataset):
    def __init__(self, feat_csv, args_dict):
        self.args = args_dict
        self.df = pd.read_csv(feat_csv)
        self.sample_size = len(self.df)
        self.df['id'] = list(range(self.sample_size))
        
    def __getitem__(self, idx):
        sample = self.df.iloc[idx]
        trn_ids = [] 
        if self.args["blank_symbol"] is not None:
            # augment the transcription with blank label id
            for i in json.loads(sample["trans_ids"]):
                trn_ids += [self.args["blank_symbol"], i]
            trn_ids.append(self.args["blank_symbol"])
        else:
            trn_ids = json.loads(sample["trans_ids"])
            
        feature = np.load(sample["features"])
        index = sample["id"]
        # assert len(trn_ids)>0
        return index, feature, trn_ids

    def __len__(self):
        return self.sample_size

    def pad_collate(self, batch):
        max_input_len = float('-inf')
        max_target_len = float('-inf')

        for elem in batch:
            index, feature, trn = elem
            max_input_len = max_input_len if max_input_len > feature.shape[0] else feature.shape[0]
            max_target_len = max_target_len if max_target_len > len(trn) else len(trn)

        for i, elem in enumerate(batch):
            index, f, trn = elem
            # trn = np.array(trn)
            input_length = np.array(f.shape[0])
            input_dim = f.shape[1]
            feature = np.zeros((max_input_len, input_dim), dtype=np.float)
            feature[:f.shape[0], :f.shape[1]] = f
            trn = np.pad(trn, (0, max_target_len - len(trn)), 'constant', constant_values=self.args["pad_token"])

            batch[i] = (int(index), feature, trn, input_length)

        batch.sort(key=lambda x: x[3], reverse=True)
        
        return default_collate(batch)
