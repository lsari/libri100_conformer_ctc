# Author: Leda Sari

import sys
import json
import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim

from dataset import LibriSpeechDataset
import models
import conformer_layer

# For decoding
# from ctcdecode import CTCBeamDecoder
# from Levenshtein import distance as levenshtein_distance

seed=2020
torch.manual_seed(seed)
np.random.seed(seed)

def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.
    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


# TODO: Add some seed 

with open(sys.argv[1], 'r') as f:
    args = json.load(f)

ctc_loss = nn.CTCLoss(blank=args["pad_token"], reduction="mean")

    
train_set = LibriSpeechDataset(args["train_set"], args)
valid_set = LibriSpeechDataset(args["valid_set"], args)

train_collate_fn = train_set.pad_collate

train_loader = DataLoader(
    train_set,
    batch_size=args["batch_size"],
    collate_fn=train_collate_fn,
    shuffle=False,
    num_workers=args["num_workers"]
)

valid_loader = DataLoader(
    valid_set,
    batch_size=args["batch_size"],
    collate_fn=train_collate_fn,
    num_workers=args["num_workers"]
)

device = torch.device('cuda' if args["use_gpu"] else "cpu")
model = conformer_layer.ConformerModel(args)
model = model.to(device)

optimizer = optim.Adam(model.parameters(), args["lr"])
scaler = GradScaler()

for e in range(args["num_epochs"]):
    model.train()
    for (index, features, trns, input_lengths) in (train_loader):
        features = features.float().to(device)
        # print(features.size())
        # features = features.transpose(0, 1) # .unsqueeze(1)
        # print(features.size())
        trns = trns.long().to(device)
        input_lengths = input_lengths.long().to(device)
        with autocast():
            optimizer.zero_grad()
            log_y, output_lengths = model(features, input_lengths)
        
            target_lengths = torch.IntTensor([
                len(y[y != args["pad_token"]]) for y in trns
        ])
            # train_ctc_loss = torch.mean(ctc_loss(log_y, trns, output_lengths, target_lengths)/(target_lengths.float().to(device))) 
            train_ctc_loss = ctc_loss(log_y, trns, output_lengths, target_lengths)

        scaler.scale(train_ctc_loss).backward()
        scaler.unscale_(optimizer)
        clip_gradient(optimizer, args["clip"])
        scaler.step(optimizer) #.step()
        scaler.update()
        
        # train_ctc_loss.backward()
        # total_norm = nn.utils.clip_grad_norm_(model.parameters(), args["clip"])
        # optimizer.step()
        print(train_ctc_loss.data)
        # break
    
    valid_ctc_loss = model.eval_loss(valid_loader, ctc_loss, device)
    print("Valid loss at epoch {}: {}".format(e, valid_ctc_loss.data))

    # TODO: Save model, update lr?
    # if e % == 0 
    torch.save(model.state_dict(), "{}/checkpoint_{}.pt".format(sys.argv[2], e))


