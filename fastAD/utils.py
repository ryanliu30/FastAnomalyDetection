import math
from functools import partial
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

def get_positional_encoding(d_model, max_len):
    """
    Generate sinusoidal positional encoding. Implementation is taken from LabML
    https://nn.labml.ai/transformers/positional_encoding.html
    """
    encodings = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
    two_i = torch.arange(0, d_model, 2, dtype=torch.float32)
    div_term = torch.exp(two_i * -(math.log(10000.0) / d_model))
    encodings[:, 0::2] = torch.sin(position * div_term)
    encodings[:, 1::2] = torch.cos(position * div_term)
    encodings = encodings.requires_grad_(False)

    return encodings

def stack_to_seq(stack, length):
    seq = []
    for i, (event, l) in enumerate(zip(stack, length)):
        seq.append(event[:l])
    return torch.cat(seq, dim = 0)

class SetNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        self.weight = Parameter(torch.ones((1, 1, normalized_shape)))
        self.bias = Parameter(torch.zeros((1, 1, normalized_shape)))
    def forward(self, x, mask = None):
        """
        x: [P, N, C]
        mask: [N, P]
        """
        if mask is None:
            mask = torch.zeros(
                (x.shape[1], x.shape[0]),
                device = x.device,
                dtype = bool
            )
        weights = (
            (~mask).float() / (~mask).sum(1, keepdim = True)
        ).permute(1, 0).unsqueeze(2)
        means = (x * weights).sum(0, keepdim = True).mean(2, keepdim = True)
        variances = (
            (x - means).square() * weights
        ).sum(0, keepdim = True).mean(2, keepdim = True) # [1, N, 1]
        std_div = torch.sqrt(variances + 1e-5) # [1, N, 1]
        return (
            (x - means) / std_div * self.weight + self.bias
        ).masked_fill_(mask.permute(1, 0).unsqueeze(2), 0)

class MaskedBatchNorm1D(nn.Module):
    def __init__(self, normalized_shape, momentum = 0.01):
        super().__init__()
        self.register_buffer(
            "mean", 
            torch.zeros((1, 1, normalized_shape)), 
            True
        )
        self.register_buffer(
            "var",
            torch.ones((1, 1, normalized_shape)),
            True
        )
        self.weight = Parameter(torch.ones((1, 1, normalized_shape)))
        self.bias = Parameter(torch.zeros((1, 1, normalized_shape)))
        self.momentum = momentum
        
    def forward(self, x, mask = None):
        """
        x: [N, P, C]
        mask: [N, P]
        """
        if self.training:
            with torch.no_grad():
                if mask is not None:
                    weights = (
                        mask.float() / mask.float().sum(1, keepdim = True)
                    ).unsqueeze(2) / x.shape[0]
                else:
                    weights = torch.ones(
                        x.shape[:2]
                    ).to(x).unsqueeze(2) / x.shape[0] / x.shape[1]
                mean = (weights * x).sum((0, 1), keepdim = True)
                var = (
                    weights * (x - mean).square()
                ).sum((0, 1), keepdim = True)
                self.mean = (1 - self.momentum)*self.mean + self.momentum*mean
                self.var = (1 - self.momentum)*self.var + self.momentum*var                
            
        x = (x - self.mean)/torch.sqrt(self.var + 1e-5)*self.weight + self.bias
        return x
    
class MaskedBatchNorm2D(nn.Module):
    def __init__(self, normalized_shape, momentum = 0.01):
        super().__init__()
        self.register_buffer(
            "mean", 
            torch.zeros((1, 1, 1, normalized_shape)),
            True
        )
        self.register_buffer(
            "var",
            torch.ones((1, 1, 1, normalized_shape)),
            True
        )
        self.weight = Parameter(torch.ones((1, 1, 1, normalized_shape)))
        self.bias = Parameter(torch.zeros((1, 1, 1, normalized_shape)))
        self.momentum = momentum
        
    def forward(self, x, mask = None):
        """
        x: [N, P, P, C]
        mask: [N, P]
        """
        if self.training:
            with torch.no_grad():
                if mask is not None:
                    mask = mask[:, :, None] & mask[:, None, :]
                    weights = (
                        mask.float() / mask.float().sum((1, 2), keepdim = True)
                    ).unsqueeze(3) / x.shape[0]
                else:
                    weights = torch.ones(
                        x.shape[:3]
                    ).to(x).unsqueeze(3) / x.shape[0] / x.shape[1] / x.shape[2]
                mean = (weights * x).sum((0, 1, 2), keepdim = True)
                var = (
                    weights * (x - mean).square()
                ).sum((0, 1, 2), keepdim = True)
                self.mean = (1 - self.momentum)*self.mean + self.momentum*mean
                self.var = (1 - self.momentum)*self.var + self.momentum*var                
            
        x = (x - self.mean)/torch.sqrt(self.var + 1e-5)*self.weight + self.bias
        return x
    
class TransformerLayer(nn.Module):
    
    def __init__(self, 
                 d_model = 512, 
                 heads = 8, 
                 dropout = 0,
                 d_source = 512,
                 d_ff = 2048,
                 self_attn = True,
                 src_attn = False,
                ):

        super().__init__()
        if self_attn:
            self.self_attn = nn.MultiheadAttention(
                d_model,
                heads,
                dropout=dropout
            )
        if src_attn:
            self.src_attn = nn.MultiheadAttention(
                d_model,
                heads,
                dropout=dropout,
                kdim=d_source,
                vdim=d_source
            ) 
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=True),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model, bias=True)
        )
        self.dropout = nn.Dropout(dropout)
        if self_attn:
            self.norm_self_attn = SetNorm(d_model)
        if src_attn:
            self.norm_src_attn = SetNorm(d_model)
        self.norm_ff = SetNorm(d_model)

    def forward(self,
                x: torch.Tensor,
                src: torch.Tensor = None,
                padding_mask: torch.Tensor = None, 
                src_padding_mask: torch.Tensor = None
               ):
        if hasattr(self, "self_attn"):
            z = self.norm_self_attn(x, padding_mask)
            self_attn, *_ = self.self_attn(z, z, z, padding_mask)
            x = x + self.dropout(self_attn)
        if hasattr(self, "src_attn"):
            z = self.norm_src_attn(x, padding_mask)
            src_attn, *_ = self.src_attn(z, src, src, src_padding_mask)
            x = x + self.dropout(src_attn)
        
        z = self.norm_ff(x, padding_mask)
        ff = self.feed_forward(z)
        x = x + self.dropout(ff)

        return x
    
class DeepSetLayer(nn.Module):
    
    def __init__(self, 
                 d_model = 512, 
                 d_ff = 1024,
                 dropout = 0,
                ):

        super().__init__()
        
        self.phi = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=True),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
        )
            
        self.rho = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=True),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
        )
        self.ff = nn.Sequential(
            nn.Linear(d_ff, d_model, bias=True),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
        )
        
        self.act = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)
        self.norm = MaskedBatchNorm1D(d_model)

    def forward(self,
                x: torch.Tensor,
                mask: torch.Tensor
               ):
        
        weights = (mask.float() / mask.sum(1, keepdim = True))[:, :, None]
        
        x = self.norm(x, mask)
        
        phi = self.phi(x)
        rho = self.rho((x * weights).sum(1, keepdim = True))
        z = self.ff(rho + phi)

        return z