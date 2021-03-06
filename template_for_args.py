# Author: Leda Sari
import json

args_dict = {
    "train_set": "data/features_train-clean-100.csv",
    "valid_set": "data/features_dev-clean.csv",
    "test_set": "data/features_test-clean.csv",
    "labels_file": "data/labels.json",
    "batch_size": 8,
    "num_workers": 0,
    "use_gpu": True,
    "lr": 0.0001,
    "clip": 10,
    "pad_token": 0,
    "blank_symbol": None,
    "num_epochs": 100,
    "bidirectional": True,
    "rnn_input_dim": 80,
    "nb_layers": 5,
    "rnn_hidden_size": 768,
    "num_classes": 29,
    "rnn_type": "lstm",
    "context": 20,
    'encoder_embed_dim': 144,
    'encoder_attention_heads': 4,
    'depthwise_conv_kernel_size': 5,
    'encoder_ffn_embed_dim': 144,
    'dropout': 0.1,
    'input_feat_per_channel': 80,
    'input_channels': 1,
    'conv_channels': 32,
    'conv_kernel_sizes': [3],
    'max_source_positions': 11,
    'encoder_layers': 16,
    "decoder_hidden_size": 256,
    "decoder_num_layers": 1,
    "prev_checkpoint": None,
    "num_embeddings": 1024
}

with open("train_1.json", 'w') as f:
    json.dump(args_dict, f, indent=4)
