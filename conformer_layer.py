import math
from collections import OrderedDict
import torch
import torch.onnx.operators
from torch import Tensor, nn
from typing import Optional, Any, List
# from fairseq.modules import LayerNorm, MultiheadAttention
from models import BatchRNN

supported_rnns = {
    'lstm': nn.LSTM,
    'rnn': nn.RNN,
    'gru': nn.GRU
}


def lengths_to_padding_mask(lens):
    bsz, max_lens = lens.size(0), torch.max(lens).item()
    mask = torch.arange(max_lens).to(lens.device).view(1, max_lens)
    mask = mask.expand(bsz, -1) >= lens.view(bsz, 1).expand(-1, max_lens)
    return mask


def padding_mask_to_lengths(mask):
    # print(torch.sum(0 + ~mask, 1))
    return torch.sum(0 + ~mask, 1).cpu() # .to(mask.device)


def make_positions(tensor, padding_idx: int, onnx_trace: bool = False):
    """Replace non-padding symbols with their position numbers.
    Position numbers begin at padding_idx+1. Padding symbols are ignored.
    """
    # The series of casts and type-conversions here are carefully
    # balanced to both work with ONNX export and XLA. In particular XLA
    # prefers ints, cumsum defaults to output longs, and ONNX doesn't know
    # how to handle the dtype kwarg in cumsum.
    mask = tensor.ne(padding_idx).int()
    return (torch.cumsum(mask, dim=1).type_as(mask) * mask).long() + padding_idx


class SinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length.
    Padding symbols are ignored.
    """

    def __init__(self, embedding_dim, padding_idx, init_size=1024):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx if padding_idx is not None else 0
        self.weights = SinusoidalPositionalEmbedding.get_embedding(
            init_size, embedding_dim, padding_idx
        )
        self.onnx_trace = False
        self.register_buffer("_float_tensor", torch.FloatTensor(1))
        self.max_positions = int(1e5)

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    @staticmethod
    def get_embedding(
        num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None
    ):
        """Build sinusoidal embeddings.
        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(
            1
        ) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(
            num_embeddings, -1
        )
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    def forward(
        self,
        input,
        incremental_state: Optional[Any] = None,
        timestep: Optional[Tensor] = None,
        positions: Optional[Any] = None,
    ):
        """Input is expected to be of size [bsz x seqlen]."""
        bspair = torch.onnx.operators.shape_as_tensor(input)
        bsz, seq_len = bspair[0], bspair[1]
        max_pos = self.padding_idx + 1 + seq_len
        if self.weights is None or max_pos > self.weights.size(0):
            # recompute/expand embeddings if needed
            self.weights = SinusoidalPositionalEmbedding.get_embedding(
                max_pos, self.embedding_dim, self.padding_idx
            )
        self.weights = self.weights.to(self._float_tensor)

        if incremental_state is not None:
            # positions is the same for every token when decoding a single step
            pos = timestep.view(-1)[0] + 1 if timestep is not None else seq_len
            if self.onnx_trace:
                return (
                    self.weights.index_select(index=self.padding_idx + pos, dim=0)
                    .unsqueeze(1)
                    .repeat(bsz, 1, 1)
                )
            return self.weights[self.padding_idx + pos, :].expand(bsz, 1, -1)

        positions = make_positions(
            input, self.padding_idx, onnx_trace=self.onnx_trace
        )
        if self.onnx_trace:
            flat_embeddings = self.weights.detach().index_select(0, positions.view(-1))
            embedding_shape = torch.cat(
                (bsz.view(1), seq_len.view(1), torch.tensor([-1], dtype=torch.long))
            )
            embeddings = torch.onnx.operators.reshape_from_tensor_shape(
                flat_embeddings, embedding_shape
            )
            return embeddings
        return (
            self.weights.index_select(0, positions.view(-1))
            .view(bsz, seq_len, -1)
            .detach()
        )

    
class Conv1dSubsampler(nn.Module):
    """Convolutional subsampler: a stack of 1D convolution (along temporal
    dimension) followed by non-linear activation via gated linear units
    (https://arxiv.org/abs/1911.08460)
    Args:
        in_channels (int): the number of input channels
        mid_channels (int): the number of intermediate channels
        out_channels (int): the number of output channels
        kernel_sizes (List[int]): the kernel size for each convolutional layer
    """

    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        kernel_sizes: List[int] = (3, 3),
    ):
        super(Conv1dSubsampler, self).__init__()
        self.n_layers = len(kernel_sizes)
        self.conv_layers = nn.ModuleList(
            nn.Conv1d(
                in_channels if i == 0 else mid_channels // 2,
                mid_channels if i < self.n_layers - 1 else out_channels * 2,
                k,
                stride=2,
                padding=k // 2,
            )
            for i, k in enumerate(kernel_sizes)
        )

    def get_out_seq_lens_tensor(self, in_seq_lens_tensor):
        out = in_seq_lens_tensor.clone()
        for _ in range(self.n_layers):
            out = ((out.float() - 1) / 2 + 1).floor().long()
        return out

    def forward(self, src_tokens, src_lengths):
        bsz, in_seq_len, _ = src_tokens.size()  # B x T x (C x D)
        x = src_tokens.transpose(1, 2).contiguous()  # -> B x (C x D) x T
        for conv in self.conv_layers:
            x = conv(x)
            x = nn.functional.glu(x, dim=1)
        _, _, out_seq_len = x.size()
        x = x.transpose(1, 2).transpose(0, 1).contiguous()  # -> T x B x (C x D)
        return x, self.get_out_seq_lens_tensor(src_lengths)

    
class ConvolutionModule(torch.nn.Module):
    def __init__(
        self,
        embed_dim,
        channels,
        depthwise_kernel_size,
        dropout,
        bias=False,
        export=False,
    ):
        super(ConvolutionModule, self).__init__()
        # kernerl_size should be a odd number for 'SAME' padding
        assert (depthwise_kernel_size - 1) % 2 == 0
        self.layer_norm = torch.nn.LayerNorm(embed_dim) # , export=export)
        self.pointwise_conv1 = torch.nn.Conv1d(
            embed_dim,
            2 * channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        self.glu = torch.nn.GLU(dim=1)
        self.depthwise_conv = torch.nn.Conv1d(
            channels,
            channels,
            depthwise_kernel_size,
            stride=1,
            padding=(depthwise_kernel_size - 1) // 2,
            groups=channels,
            bias=bias,
        )
        self.batch_norm = torch.nn.BatchNorm1d(channels)
        self.swish = torch.nn.SiLU(channels)
        self.pointwise_conv2 = torch.nn.Conv1d(
            channels,
            embed_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        """
        x: B X T X C
        Output:  B X T X C
        """
        x = self.layer_norm(x)
        # exchange the temporal dimension and the feature dimension
        x = x.transpose(1, 2)

        # GLU mechanism
        x = self.pointwise_conv1(x)  # (batch, 2*channel, dim)
        x = self.glu(x)  # (batch, channel, dim)

        # 1D Depthwise Conv
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.swish(x)

        x = self.pointwise_conv2(x)

        return x.transpose(1, 2)


class FeedForwardModule(torch.nn.Module):
    """Positionwise feed forward layer."""

    def __init__(self, input_feat, hidden_units, dropout1, dropout2, bias=True):
        """Construct an PositionwiseFeedForward object."""
        super(FeedForwardModule, self).__init__()
        self.layer_norm = nn.LayerNorm(input_feat)
        self.w_1 = torch.nn.Linear(input_feat, hidden_units, bias=True)
        self.w_2 = torch.nn.Linear(hidden_units, input_feat, bias=True)
        self.dropout1 = torch.nn.Dropout(dropout1)
        self.dropout2 = torch.nn.Dropout(dropout2)
        self.activation = torch.nn.SiLU(hidden_units)

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.w_1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.w_2(x)
        return self.dropout2(x)


class ConformerEncoderLayer(torch.nn.Module):
    def __init__(self, args):
        super(ConformerEncoderLayer, self).__init__()

        self.ffn1 = FeedForwardModule(
            args['encoder_embed_dim'],
            args['encoder_ffn_embed_dim'],
            args['dropout'],
            args['dropout'],
        )

        self.self_attn_layer_norm = nn.LayerNorm(args['encoder_embed_dim']) # , export=False)
        self.self_attn_dropout = torch.nn.Dropout(args['dropout'])
        self.self_attn = torch.nn.MultiheadAttention(
            args['encoder_embed_dim'], args['encoder_attention_heads'], dropout=args['dropout']
        )
        # nn.MultiheadAttention(embed_dim, num_heads, dropout=0.0, bias=True,
        # add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None,
        # batch_first=False, device=None, dtype=None)

        self.conv_module = ConvolutionModule(
            embed_dim=args['encoder_embed_dim'],
            channels=args['encoder_embed_dim'],
            depthwise_kernel_size=args['depthwise_conv_kernel_size'],
            dropout=args['dropout'],
        )

        self.ffn2 = FeedForwardModule(
            args['encoder_embed_dim'],
            args['encoder_ffn_embed_dim'],
            args['dropout'],
            args['dropout'],
        )
        self.final_layer_norm = torch.nn.LayerNorm(args['encoder_embed_dim']) #, export=False)

    def forward(self, x, encoder_padding_mask: Optional[torch.Tensor]):
        """x: T X B X C"""
        residual = x
        x = self.ffn1(x)
        x = x * 0.5 + residual

        residual = x
        x = self.self_attn_layer_norm(x)
        x, attn = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=encoder_padding_mask,
            need_weights=False,
        )
        x = self.self_attn_dropout(x)
        x = x + residual

        residual = x
        # TBC to BTC
        x = x.transpose(0, 1)
        x = self.conv_module(x)
        # BTC to TBC
        x = x.transpose(0, 1)
        x = residual + x

        residual = x
        x = self.ffn2(x)
        x = x * 0.5 + residual

        x = self.final_layer_norm(x)
        return x, attn

    
class S2TConformerEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.padding_idx = 1 # args["PADDING_TOKEN"]
        self.subsample = Conv1dSubsampler(
            args['input_feat_per_channel'] * args['input_channels'],
            args['conv_channels'],
            args['encoder_embed_dim'],
            [int(k) for k in args['conv_kernel_sizes']],
        )
        # max_pos = self.padding_idx + 1 + seq_len 
        self.embed_positions = SinusoidalPositionalEmbedding(
            args['encoder_embed_dim'], self.padding_idx # , init_size=args["num_embeddings"] + self.padding_idx + 1
        )
        self.linear = torch.nn.Linear(args['encoder_embed_dim'], args['encoder_embed_dim'])
        self.dropout = torch.nn.Dropout(args['dropout'])
        self.conformer_layers = torch.nn.ModuleList(
            [ConformerEncoderLayer(args) for _ in range(args['encoder_layers'])]
        )

    def forward(self, src_tokens, src_lengths, return_all_hiddens=False):
        """
        src_tokens: T X B X C -> should be B x T x C, check subsample! 
        """
        x, input_lengths = self.subsample(src_tokens, src_lengths)
        # print("in encoderModel", input_lengths)
        encoder_padding_mask = lengths_to_padding_mask(input_lengths)
        positions = self.embed_positions(encoder_padding_mask).transpose(0, 1)
        x += positions
        x = self.linear(x)
        x = self.dropout(x)
        encoder_states = []

        for layer in self.conformer_layers:
            x, _ = layer(x, encoder_padding_mask)
            if return_all_hiddens:
                encoder_states.append(x)

        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask]
            if encoder_padding_mask.any()
            else [],  # B x T
            "encoder_embedding": [],  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [],
        }

    # def reorder_encoder_out(self, encoder_out, new_order):
    #     return S2TTransformerEncoder.reorder_encoder_out(self, encoder_out, new_order)

    
class ConformerModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.num_classes = args["num_classes"] 
        self.encoder = S2TConformerEncoder(args)
        self.rnns = nn.Sequential(
            BatchRNN(
                input_size=args["encoder_embed_dim"],
                hidden_size=args["decoder_hidden_size"],
                rnn_type=supported_rnns[args["rnn_type"]],
                bidirectional=args["bidirectional"]
            ),
            *[BatchRNN(
                input_size=args["decoder_hidden_size"],
                hidden_size=args["decoder_hidden_size"],
                rnn_type=supported_rnns[args["rnn_type"]],
                bidirectional=args["bidirectional"]
            ) for n in range(1, args["decoder_num_layers"])]
        )
        
        self.fc = nn.Sequential(
            nn.Linear(args["decoder_hidden_size"], self.num_classes),
            nn.LogSoftmax(2)
        )
        self.config = args
        self.pad_token = args["pad_token"]
        
    def forward(self, x, lengths):
        encoder_result = self.encoder(x, lengths)
        encoder_out_lengths = padding_mask_to_lengths(encoder_result["encoder_padding_mask"][0])
        for k, rnn in enumerate(self.rnns):
            if k==0:
                penultimate_y  = rnn(encoder_result["encoder_out"][0], encoder_out_lengths)
            else:
                penultimate_y  = rnn(penultimate_y, encoder_out_lengths)
                
        log_y = self.fc(penultimate_y)
        
        return log_y, encoder_out_lengths

    def eval_loss(self, dataset, loss_fn, device):
        self.eval()

        total_loss = 0.0
        total_count = 0.0
        for (index, features, trns, input_lengths) in dataset:
            features = features.float().to(device)
            # features = features.transpose(0, 1) # .unsqueeze(1)
            trns = trns.long().to(device)
            input_lengths = input_lengths.long().to(device)

            log_y, output_lengths = self(features, input_lengths)
            target_lengths = torch.IntTensor([
                len(y[y != self.config["pad_token"]]) for y in trns
            ])
            batch_loss = loss_fn(log_y, trns, output_lengths, target_lengths) 

            # total_loss += torch.sum(batch_loss).data
            # total_count += features.size(0)
            total_loss += batch_loss.data * features.size(0)
            total_count += features.size(0)
            
        return total_loss/total_count
        
