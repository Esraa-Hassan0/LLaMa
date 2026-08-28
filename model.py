from dataclasses import dataclass
from typing import Optional
import torch
import math
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1  # will be set when load the tokenizer
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 2048

    device: str = None


def precompute_theta_pos_freqs(
    head_dim: int, seq_len: int, device, theta: float = 10000.0
):
    assert head_dim % 2 == 0, "Head dimension must be divided by 2"

    # (Head_dim / 2)
    theta_numerator = torch.arange(0, head_dim, 2).float()

    # (Head_dim / 2)
    theta = 1.0 / (theta ** (theta_numerator / head_dim)).to(device)

    # (Seq_len)
    m = torch.arange(seq_len, device=device)

    #  (seq_len) outer_prod (Head_dim / 2) => (seq_len, Head_dim / 2)
    freqs = torch.outer(m, theta).float()

    # (seq_len, Head_dim / 2) => (seq_len, Head_dim / 2)
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex


def applyRotaryEmbeddings(x: torch.Tensor, freqs_complex: torch.Tensor, device: str):
    # (B, Seq_len, H, Head_dim) => (B, Seq_len, H, Head_dim/2)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))

    #  (Seq_len, Head_dim/2) => (1, Seq_len, 1, Head_Dim/2)
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)

    # (B, Seq_len, H, Head_dim/2) * (1, Seq_len, 1, Head_Dim/2) => (B, Seq_len, H, Head_dim/2)
    x_rotated = x_complex * freqs_complex

    # (B, Seq_len, H, Head_dim/2) => (B, Seq_len, H, Head_dim/2 , 2)
    x_out = torch.view_as_real(x_rotated)

    # (B, Seq_len, H, Head_dim/2 , 2) => (B, Seq_len, H, Head_dim)
    x_out = x_out.reshape(*x.shape)
    return x_out.type_as(x).to(device)


def repeat_kv(x: torch.Tensor, n_rep: int):
    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    return (
        # (B, Seq_len, N_KV_Heads, 1, Head_Dim)
        x[:, :, :, None, :]
        # (B, Seq_len, N_KV_Heads, N_rep, Head_Dim)
        .expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)
        # (B, Seq_len, N_KV_Heads * N_rep, Head_Dim)
        .reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)
    )


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        # (B, seq_len, Dim) * (B, seq_len, 1)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        # (Dim) * (B, Seq_len, Dim) =  (B, Seq_len, Dim)
        return self.weight * self._norm(x.float()).type_as(x)


class SelfAttention(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()

        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads

        self.n_q_heads = args.n_heads

        self.n_rep = self.n_q_heads // self.n_kv_heads

        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(
            args.n_heads * self.head_dim, args.dim , bias=False
        )

        self.cache_k = torch.zeros(
            (args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim),
            device=args.device,
        )
        self.cache_v = torch.zeros(
            (args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim),
            device=args.device,
        )

    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        batch_size, seq_len, _ = x.shape  # (B,1,Dim)

        # (B, 1, Dim) -> (B, 1, H_Q * Head_Dim)
        xq = self.wq(x)
        # (B, 1, Dim) -> (B, 1, H_KV * Head_Dim)
        xk = self.wk(x)
        # (B, 1, Dim) -> (B, 1, H_KV * Head_Dim)
        xv = self.wv(x)

        # (B, 1, H_Q * Head_Dim) => (B, 1, H_Q, Head_Dim)
        xq = xq.view(batch_size, seq_len, self.n_q_heads, self.head_dim)
        # (B, 1, H_KV * Head_Dim) => (B, 1, H_KV, Head_Dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        # (B, 1, H_KV * Head_Dim) => (B, 1, H_KV, Head_Dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        # (B, 1 , H_Q, Head_Dim) => (B, 1, H_Q, Head_Dim)
        xq = applyRotaryEmbeddings(xq, freqs_complex, device=x.device)

        # (B, 1 , H_KV, Head_Dim) => (B, 1, H_KV, Head_Dim)
        xk = applyRotaryEmbeddings(xk, freqs_complex, device=x.device)

        # Place the entry in the cache
        self.cache_k[:batch_size, start_pos : start_pos + seq_len] = xk
        self.cache_v[:batch_size, start_pos : start_pos + seq_len] = xv

        # (B, Seq_len_KV, H_KV, Head_Dim)
        keys = self.cache_k[:batch_size, : start_pos + seq_len]
        # (B, Seq_len_KV, H_KV, Head_Dim)
        values = self.cache_v[:batch_size, : start_pos + seq_len]

        # (B, Seq_Len_KV, H_KV, Head_Dim) => (B, Seq_Len_KV, H_Q, Head_Dim)
        keys = repeat_kv(keys, self.n_rep)
        # (B, Seq_Len_KV, H_KV, Head_Dim) => (B, Seq_Len_KV, H_Q, Head_Dim)
        values = repeat_kv(values, self.n_rep)

        # (B,1, H_Q, Head_Dim) => (B, H_Q, 1, Head_Dim)
        xq = xq.transpose(1, 2)
        # (B,Seq_Len_KV, H_Q, Head_Dim) => (B, H_Q, Seq_Len_KV, Head_Dim)
        keys = keys.transpose(1, 2)

        # (B,Seq_Len_KV, H_Q, Head_Dim) => (B, H_Q, Seq_Len_KV, Head_Dim)
        values = values.transpose(1, 2)

        # (B, H_Q, 1, Head_Dim) @  (B, H_Q, Head_Dim, Seq_Len_KV) => (B, H_Q, 1, Seq_Len_KV)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)

        # (B, H_Q, 1, Seq_Len_KV) => (B, H_Q, 1, Seq_Len_KV)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)

        # (B, H_Q, 1, Seq_Len_KV) @ (B, H_Q, Seq_Len_KV, Head_Dim) => (B, H_Q, 1, Head_Dim)
        output = torch.matmul(scores, values)

        # (B, H_Q, 1, Head_Dim) => (B, 1, H_Q, Head_Dim) => (B, 1, Dim)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        # (B, 1, Dim) => (B, 1, Dim)
        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        hidden_dim = 4 * args.dim
        hidden_dim = int(2 * hidden_dim / 3)
        if args.ffn_dim_multiplier is not None:
            hidden_dim *= args.ffn_dim_multiplier

        hidden_dim = args.multiple_of * (
            (args.multiple_of + hidden_dim - 1) // args.multiple_of
        )

        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor):
        # (B, seq_len ,Dim) => (B, seq_len ,Hidden_dim)
        swish = F.silu(self.w1(x))
        # (B, seq_len ,Dim) => (B, seq_len ,Hidden_dim)
        x_V = self.w3(x)
        # (B, seq_len ,Hidden_dim) * (B, seq_len ,Hidden_dim) => (B, seq_len ,Hidden_dim)
        x = swish * x_V
        #  (B, seq_len ,Hidden_dim) => (B, seq_len ,Dim)
        x = self.w2(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()

        self.feed_forward = FeedForward(args)

        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)

        self.attention = SelfAttention(args)
        

    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):

        h = x + self.attention(self.attention_norm(x), start_pos, freqs_complex)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        assert args.vocab_size != -1, "Vocab size must be set"

        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.tok_embedd = nn.Embedding(self.vocab_size, args.dim)

        self.layers = nn.ModuleList()

        for _ in range(args.n_layers):
            self.layers.append(EncoderBlock(args))

        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, self.vocab_size, bias=False)

        self.freq_complex = precompute_theta_pos_freqs(
            self.args.dim // self.args.n_heads,
            self.args.max_seq_len * 2,
            self.args.device,
        )

    def forward(self, tokens: torch.Tensor, start_pos: int):
        batch_size, seq_len = tokens.shape
        assert seq_len == 1, "Only one token at a time can be proceed"

        # (B, seq_len) => (B, seq_len, Dim)
        h = self.tok_embedd(tokens)

        freqs_complex = self.freq_complex[start_pos : start_pos + seq_len]

        for layer in self.layers:
            h = layer(h, start_pos, freqs_complex)

        h = self.norm(h)

        output = self.output(h).float()

        return output
