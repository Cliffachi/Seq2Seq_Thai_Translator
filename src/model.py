import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import (
    pack_padded_sequence,
    pad_packed_sequence,
)

class Seq2Seq(nn.Module):
    def __init__(self, 
                 src_vocab_size,
                 tgt_vocab_size,
                 embed_dim=128,
                 hidden_dim=256,
                 padding_idx=0,
                 attention_dim=None):

        super().__init__()
        self.padding_idx = padding_idx
        attention_dim = attention_dim or hidden_dim

        self.src_embedding = nn.Embedding(src_vocab_size, embed_dim, padding_idx=padding_idx)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, embed_dim, padding_idx=padding_idx)

        self.encoder = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.decoder = nn.GRU(embed_dim, hidden_dim, batch_first=True)

        # Attention mechanism
        self.Wk = nn.Linear(hidden_dim, attention_dim, bias=False)
        self.Wq = nn.Linear(hidden_dim, attention_dim, bias=False)

        self.decoder_fc = nn.Linear(hidden_dim * 2, tgt_vocab_size)

    def encode(self, src, src_lengths):
        embedded_src = self.src_embedding(src)

        packed_src = pack_padded_sequence(
            embedded_src, src_lengths.cpu(), 
            batch_first=True, 
            enforce_sorted=False
            )

        pack_out, hidden = self.encoder(packed_src)

        encoder_outputs, _ = pad_packed_sequence(pack_out,
                                                 batch_first=True,
                                                 total_length=src.size(1))

        return encoder_outputs, hidden

    def attention(self, encoder_outputs, decoder_outputs, src):
        # encoder_outputs: [B, S, H]
        # decoder_outputs: [B, T, H]

        # Project encoder outputs into attention keys
        keys = self.Wk(encoder_outputs) # [B, S, A]

        # Project every decoder timestep into a query
        queries = self.Wq(decoder_outputs) # [B, T, A]

        # Scaled dot-product attention
        scores = torch.bmm(
            queries, keys.transpose(1, 2)
        ) / math.sqrt(keys.size(-1)) # [B, T, S]

        # Mask source padding
        pad_mask = src.eq(self.padding_idx)
        scores = scores.masked_fill(
            pad_mask.unsqueeze(1),
            float("-inf")
        )

        # Normalize across source positions
        weights = torch.softmax(scores, dim=-1) # [B, T, S]
        # Weighted sum of encoder outputs
        context = torch.bmm(weights, encoder_outputs) # [B, T, H]

        return context, weights


    def forward(self, src, src_lengths, tgt):
        # Encode source
        encoder_outputs, hidden = self.encode(src, src_lengths)

        # Decode target
        embedded_tgt = self.tgt_embedding(tgt)
        decoder_outputs, _ = self.decoder(embedded_tgt, hidden)

        # Attend the encoder outputs
        context, weights = self.attention(
            encoder_outputs,
            decoder_outputs,
            src
        )

        # Concatenate context and decoder outputs
        combined = torch.cat([decoder_outputs, context], dim=-1)

        logits = self.decoder_fc(combined)
        return logits, weights