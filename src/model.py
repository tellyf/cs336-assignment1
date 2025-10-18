import torch
import math
from einops import rearrange, einsum
from jaxtyping import Float, Int
from torch import Tensor
from torch import nn


def silu(x:torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    x = x - torch.max(x, dim=dim, keepdim=True)[0]
    x = torch.exp(x)
    x_sum = torch.sum(x, dim=dim, keepdim=True)
    # Epsilon prevents division by zero
    x_sum = torch.clamp(x_sum, min=1e-12)
    return x / x_sum


def scaled_dot_product_attention(
        query:Tensor,
        key:Tensor,
        value:Tensor,
        mask:Tensor|None,
        ) -> Tensor:
    head_dim = query.shape[-1]
    attn_scores = query @ key.transpose(-2, -1)
    # attn_scores = einsum(query, key, "... q k, ... p k -> ... q p")
    if mask is not None:
        attn_scores.masked_fill_(mask == 0, float("-inf"))
    # attn_weights = torch.softmax(attn_scores / (head_dim ** 0.5), dim=-1)
    attn_weights = softmax(attn_scores / (head_dim ** 0.5), dim=-1)
    context_vec = attn_weights @ value
    return context_vec


class Linear (nn.Module):
    def __init__(self,
                in_features: int,
                out_features: int,
                device: str = "cpu",
                dtype: torch.dtype = torch.float32,
                ):
        super().__init__()
        w = torch.empty(out_features, in_features, device=device, dtype=dtype)
        # Per assignment#1 3.4.1 parameter initialization.
        std = math.sqrt(2 / (in_features + out_features))        
        nn.init.trunc_normal_(w, mean=0.0, std=std, a=-3*std, b=3*std)
        self.weights = nn.Parameter(w)

    def set_weights(self, weights: Float[Tensor, "d_out d_in"]):
        """Load weights into the linear module."""
        self.weights = nn.Parameter(weights)

    def forward(self, x: Float[Tensor, "... d_in"]) -> Float[Tensor, "... d_out"]:
        # return x @ self.weights.T
        return einsum(x, self.weights, "batch x d_in, d_out d_in -> batch x d_out")


class Embedding(nn.Module):
    # vocab_size, d_model, weights, token_ids
    def __init__(self, 
                num_embeddings: int, 
                d_model: int, 
                device: str = "cpu",
                dtype: torch.dtype = torch.float32,
            ):
        super().__init__()
        w = torch.empty(
            num_embeddings, 
            d_model, 
            device=device, 
            dtype=dtype,
        )
        # Per assignment#1 3.4.1 parameter initialization.
        nn.init.trunc_normal_(w, mean = 0.0, std=1.0, a = -3, b = 3)
        self.weights = nn.Parameter(w)
    
    def set_weights(self, weights: Float[Tensor, "vocab_size d_model"]):
        """Load weights into the embedding module."""
        self.weights = nn.Parameter(weights)

    def forward(self, token_ids: Int[Tensor, "..."]):
        return self.weights[token_ids]


class RMSnorm (nn.Module):
    def __init__(self, 
                 d_model: int, 
                 eps: float = 1e-5, 
                 device: str = "cpu", 
                 dtype: torch.dtype = torch.float32,
                 ):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.eps = eps

    def set_weights(self, weights: Float[Tensor, "d_model"]):
        """Load weights into the RMSNorm module."""
        self.weight = nn.Parameter(weights)

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        """ process an input tensor of shape (batch_size, squence_length, d_model) """
        in_type = x.dtype
        x = x.to(torch.float32) # upcast to prevent overflow.
        # calculate the mean over the last dimension
        norm = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        x = (x / norm) * self.weight
        return x.to(in_type)


class RoPE(nn.Module):
    def __init__(self,                  
                 theta: float,
                 d_k: int,
                 max_seq_len: int, 
                 device: str = "cpu", 
                 ):
        super().__init__()
        assert d_k % 2 == 0, "d_k must be even for RoPE"
        self.d_k = d_k
        half_dim = d_k // 2
        feq_seq = torch.arange(0, half_dim, device=device)
        inv_freq = 1.0 / (theta ** (feq_seq * 2.0 / d_k))
        t = torch.arange(max_seq_len, device=device)

        freqs = einsum(t, inv_freq, "seq_len, half_dim -> seq_len half_dim") 
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)
        

    def forward(self, x: Tensor, token_positions: Tensor) -> Tensor:
        *_, d_k = x.shape
        assert d_k == self.d_k

        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]

        x1 = x[..., ::2] 
        x2 = x[..., 1::2]

        result = torch.zeros_like(x)
        result[..., ::2] = x1 * cos - x2 * sin
        result[..., 1::2] = x1 * sin + x2 * cos
        return result


class FFN(nn.Module):
    def __init__(self, 
                 d_model: int,
                 d_ff: int,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 ):
        super().__init__()
        def _init_weights(in_features, out_features, device, dtype):
            w = torch.empty(out_features, in_features, device=device, dtype=dtype)
            std = (2 / (in_features + out_features)) ** 0.5
            nn.init.trunc_normal_(w, mean=0.0, std=std, a=-3 * std, b=3 * std)
            w = nn.Parameter(w)
            return w
        self.w1 = _init_weights(d_ff, d_model, device, dtype)
        self.w2 = _init_weights(d_model, d_ff, device, dtype)
        self.w3 = _init_weights(d_ff, d_model, device, dtype)
        self.silu = silu

    def set_weights(self, w1, w2, w3):
        self.w1 = nn.Parameter(w1)
        self.w2 = nn.Parameter(w2)
        self.w3 = nn.Parameter(w3)

    def forward(self, x: Tensor) -> Tensor:
        # (x) ((batch, sequence), d_model)  [ (d_model, d_ff), (d_ff, d_model) ] -> (d_ff, d_model) T-> (d_model, d_ff)
        return (self.silu(x @ self.w1.T) * (x @ self.w3.T)) @ self.w2.T               


class MultiHeadAttention(nn.Module):
    def __init__(self, 
                 d_model: int, 
                 num_heads: int,
                 theta: float|None = None,
                 max_seq_len: int|None = None, 
                 token_positions: Tensor|None = None,
                 device: str = "cpu", 
                 dtype: torch.dtype = torch.float32,
                 ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        def _init_weights(in_features, out_features, device, dtype):
            w = torch.empty(out_features, in_features, device=device, dtype=dtype)
            std = (2 / (in_features + out_features)) ** 0.5
            nn.init.trunc_normal_(w, mean=0.0, std=std, a=-3 * std, b=3 * std)
            w = nn.Parameter(w)
            return w

        self.query = _init_weights(d_model, d_model, device=device, dtype=dtype)
        self.key = _init_weights(d_model, d_model, device=device, dtype=dtype)
        self.value = _init_weights(d_model, d_model, device=device, dtype=dtype)
        self.out_proj = _init_weights(d_model, d_model, device=device, dtype=dtype)

        self.rope = None
        if theta is not None and max_seq_len is not None:
            self.rope = RoPE(theta, self.head_dim, max_seq_len, device=device) if theta is not None and max_seq_len is not None else None

        self.token_positions = token_positions

    def set_weights(self, q: Tensor, k:Tensor, v:Tensor, o:Tensor):
        """Load weights into the MultiHeadAttention module."""
        self.query = nn.Parameter(q)
        self.key = nn.Parameter(k)
        self.value = nn.Parameter(v)
        self.out_proj = nn.Parameter(o)

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        batch_size, token_len, _ = x.shape
        if self.token_positions is None:
            token_positions = torch.arange(token_len, device=x.device, dtype=torch.long)
            token_positions = token_positions.unsqueeze(0)
            token_positions = token_positions.expand(batch_size, token_len)
        else:
            token_positions = self.token_positions

        query = x @ self.query.T
        key = x @ self.key.T
        value = x @ self.value.T

        query = query.view(batch_size, token_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, token_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, token_len, self.num_heads, self.head_dim).transpose(1, 2)

        if self.rope is not None:
            query = self.rope.forward(query, token_positions)
            key = self.rope.forward(key, token_positions) 

        mask = torch.tril(torch.ones(token_len, token_len), diagonal = 0)
        context_vec = scaled_dot_product_attention(query, key, value, mask)
        context_vec = rearrange(context_vec, "... head_num token_num head_dim -> ... token_num (head_num head_dim)")
        context_vec = context_vec @ self.out_proj.T
        return context_vec


class TransformerBlock(nn.Module):
    def __init__(self, 
                d_model: int, 
                num_heads: int, 
                d_ff: int, 
                theta: float,
                max_seq_len: int,
                device: str = "cpu", 
                dtype: torch.dtype = torch.float32,
                ):
        super().__init__()
        self.att_rmsnorm = RMSnorm(d_model, device=device, dtype=dtype)
        self.mha = MultiHeadAttention(d_model, num_heads, theta, max_seq_len, device=device, dtype=dtype)
        self.att_unit = nn.Sequential(
            self.att_rmsnorm,
            self.mha,
        )
        self.ffn_rmsnorm = RMSnorm(d_model, device=device, dtype=dtype)
        self.ffn = FFN(d_model, d_ff, device, dtype)
        self.ffn_unit = nn.Sequential(
            self.ffn_rmsnorm,
            self.ffn,
        )

    def set_weights(self, weights):
        q = weights['attn.q_proj.weight']
        k = weights['attn.k_proj.weight']
        v = weights['attn.v_proj.weight']
        o = weights['attn.output_proj.weight']
        r1 = weights['ln1.weight']
        r2 = weights['ln2.weight']
        w1 = weights['ffn.w1.weight']
        w2 = weights['ffn.w2.weight']
        w3 = weights['ffn.w3.weight']
        self.att_rmsnorm.set_weights(r1)
        self.mha.set_weights(q, k, v, o)
        self.ffn_rmsnorm.set_weights(r2)
        self.ffn.set_weights(w1, w2, w3)

    def forward(self, x:Tensor) -> Tensor:
        x += self.att_unit(x)
        x += self.ffn_unit(x)
        return x


class TransformerLM(nn.Module):
    def __init__(self,
                vocab_size: int,
                context_length: int,
                d_model: int,
                num_layers: int,
                num_heads: int,
                d_ff: int,
                rope_theta: float,
                device: str = "cpu", 
                dtype: torch.dtype = torch.float32,
                ):
        super().__init__()
        self.embedding = Embedding(vocab_size, d_model, device, dtype)
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    num_heads,
                    d_ff,
                    rope_theta,
                    context_length,
                    device,
                    dtype,
                )
                for _ in range(num_layers)
            ]
        )
        self.rms_norm = RMSnorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device, dtype)

    def forward(self, in_indices: Tensor) -> Tensor:
        x = self.embedding(in_indices)
        for block in self.transformer_blocks:
            x = block(x)
        x = self.rms_norm(x)
        x = self.lm_head(x)
        return x        
