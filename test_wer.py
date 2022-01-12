# Author: Leda Sari

import sys
import json
import numpy as np
import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
import torch.optim as optim

from dataset import LibriSpeechDataset
import models
import conformer_layer

# For decoding
# from ctcdecode import CTCBeamDecoder
from ctc_decoder import best_path, beam_search
from Levenshtein import distance as levenshtein_distance


# TODO: Add some seed 
args_json = sys.argv[1]
ckpt  = sys.argv[2]
with open(args_json, 'r') as f:
    args = json.load(f)

# ctc_loss = nn.CTCLoss(blank=args["pad_token"], reduction="none")

    
train_set = LibriSpeechDataset(args["train_set"], args)
# valid_set = LibriSpeechDataset(args["valid_set"], args)
valid_set = LibriSpeechDataset(args["test_set"], args)

train_collate_fn = train_set.pad_collate

valid_loader = DataLoader(
    valid_set,
    batch_size=args["batch_size"],
    collate_fn=train_collate_fn,
    num_workers=args["num_workers"]
)

device = torch.device('cuda' if args["use_gpu"] else "cpu")
model = conformer_layer.ConformerModel(args)
model.load_state_dict(torch.load(ckpt, map_location='cpu'))
model = model.to(device)



# Test phase:
print("==========================")
print("===     TEST PHASE     ===")
print("==========================")


with open(args["labels_file"], 'r') as f:
    label_dict=  json.load(f)

rev_label_dict = {v: k for k, v in label_dict.items()}
print(label_dict)
char_str = ("".join([k for k, v in label_dict.items()]))[1:]
print(char_str)

# decoder = CTCBeamDecoder(
#     labels=[str(c) for c in rev_label_dict.keys()], beam_width=1
# )

model.eval()
total_error = 0.0
total_length = 0.0
total_word_err = 0.0
total_num_words = 0.0
with torch.no_grad():
    for (index, features, trns, input_lengths) in (valid_loader):
        features = features.float().to(device)
        # features = features.transpose(1,2).unsqueeze(1)
        trns = trns.long().to(device)
        input_lengths = input_lengths.long().to(device)
        
        log_y, output_lengths = model(features, input_lengths)
        
        target_lengths = torch.IntTensor([
            len(y[y != args["pad_token"]]) for y in trns
        ])

        # print(log_y.size(), input_lengths.size())
        mat = torch.exp(log_y).transpose(0,1).detach().cpu().numpy()
        out_mat = np.concatenate([mat[:, :, 1:], mat[:,:, 0, np.newaxis]], 2)
        # print(np.shape(out_mat))
        # print(np.shape(out_mat))
        for k, l in enumerate(output_lengths):
            hyp = beam_search(out_mat[k,:l], char_str) # .rstrip('QZ')
            t = trns[k].detach().cpu().tolist()
            t = [ll for ll in t if ll != 0]
            tlength = len(t)
            tt = ''.join([rev_label_dict[i] for i in t])
            print(tt, hyp)
            error = levenshtein_distance(tt, hyp)
            word_err = levenshtein_distance(tt.split(" "), hyp.split(" "))
            num_words = len(tt.split(" "))
            total_error += error
            total_length += tlength
            total_word_err += word_err
            total_num_words += num_words
            print(error, tlength, word_err, num_words)
            # break
        # break
    print(total_error/total_length, total_word_err/total_num_words)
