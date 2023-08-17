import torch
import torch.nn as nn

class MultimodalAttention(nn.Module):
    def __init__(self, heads, embed_size):
        super(MultimodalAttention, self).__init__()
        self.heads = heads
        self.embed_size = embed_size
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), f"Embedding size needs to be divisible by heads. self.head_dim = {self.head_dim}, embed_size = {self.embed_size}, num_heads = {self.heads}"

        self.values = nn.Linear(embed_size, embed_size)
        self.keys = nn.Linear(embed_size, embed_size)
        self.queries = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)


    def forward(self, queries, keys, values, mask=None):
        batch_size = queries.shape[0]

        value_len, query_len, key_len = values.shape[1], queries.shape[1], keys.shape[1]

        values = self.values(values).reshape(batch_size, value_len, self.heads, self.head_dim)
        keys = self.keys(keys).reshape(batch_size, key_len, self.heads, self.head_dim)
        queries = self.queries(queries).reshape(batch_size, query_len, self.heads, self.head_dim)

        print(f'queries.shape: {queries.shape}, keys.shape: {keys.shape}, values.shape{values.shape}')

        head_tensors = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        if mask is not None:
            head_tensors = head_tensors.masked_fill_(mask == 0, float("-1e20"))

        attention = torch.softmax(head_tensors / (self.embed_size ** (1/2)), dim=-1)
        return self.fc_out(
            torch.einsum("nhql,nlhd->nqhd", [attention, values])
                .reshape(batch_size, query_len, self.heads * self.head_dim)
        )


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super().__init__()
        self.attention = MultimodalAttention(heads, embed_size)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, mask=None):
        attention = self.attention(queries, keys, values, mask)
        x = self.dropout(self.norm1(attention + queries))
        forward = self.feed_forward(x)
        return self.dropout(self.norm2(forward + x))


class Encoder(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        embed_size,
        num_layers,
        heads,
        forward_expansion,
        dropout,
        max_length,
        device, 
    ):
        super().__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion
                )
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        batch_size, sequence_len = x.shape
        positions = torch.arange(0, sequence_len).expand(batch_size, sequence_len).to(self.device)
        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))

        for layer in self.layers:
            out = layer(out, out, out, mask)
        return out

class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super().__init__()
        self.attention = MultimodalAttention(heads, embed_size)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = TransformerBlock(embed_size, heads, dropout, forward_expansion)
        self.dropout = nn.Dropout(dropout)
        self.device = device

    def forward(self, x, value, key, src_mask, trg_mask):
        attention = self.attention(x, x, x, trg_mask)
        query = self.dropout(self.norm(attention + x))
        return self.transformer_block(value, key, query, src_mask)
    
class Decoder(nn.Module):
    def __init__(self, trg_vocab_size, embed_size, num_layers, heads, forward_expansion, dropout, device, max_length):
        super().__init__()
        self.device = device
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [DecoderBlock(embed_size, heads, forward_expansion, dropout, device) for _ in range(num_layers)]
        )
        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, trg_mask):
        N, seq_length = x.shape
        positions = torch.arange(0,seq_length).expand(N, seq_length).to(self.device)
        x = self.dropout(self.word_embedding(x) + self.position_embedding(positions))

        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)

        return self.fc_out(x)
    
class Transformer(nn.Module):
    def __init__(
            self,
            src_vocab_size,
            trg_vocab_size,
            src_pad_idx,
            # trg_pad_idx,
            embed_size=512,
            num_layers=6,
            heads=8,
            forward_expansion=4,
            dropout=0,
            max_length=100,
            device='cpu',
    ):
        super().__init__()
        self.encoder = Encoder(
            src_vocab_size, 
            embed_size,
            num_layers, 
            heads, 
            forward_expansion, 
            dropout, 
            max_length, 
            device,
        )
        self.decoder = Decoder(
            trg_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            device,
            max_length,
        )
        # self.decoder.word_embedding.weight = self.encoder.word_embedding.weight
        # self.decoder.position_embedding.weight = self.encoder.position_embedding.weight
        self.src_pad_idx = src_pad_idx
        # self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):
        if self.src_pad_idx is not None:
            src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2) # will depend on the tensor shape being passed
            return src_mask.to(self.device)
        return None
    
    def make_trg_mask(self, trg):
        print(f'trg shape = {trg.shape}')
        N, _, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones(trg_len, trg_len)).expand(
            N, 1, trg_len, trg_len
        )
        return trg_mask.to(self.device)
    
    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        # print(src.shape, enc_src.shape)
        return self.decoder(trg, enc_src, src_mask, trg_mask)
