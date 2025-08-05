"""Simple, minimal implementation of Mamba in one file of PyTorch.

Suggest reading the following before/while reading the code:
    [1] Mamba: Linear-Time Sequence Modeling with Selective State Spaces (Albert Gu and Tri Dao)
        https://arxiv.org/abs/2312.00752
    [2] The Annotated S4 (Sasha Rush and Sidd Karamcheti)
        https://srush.github.io/annotated-s4

Glossary:
    b: batch size                       (`B` in Mamba paper [1] Algorithm 2)
    l: sequence length                  (`L` in [1] Algorithm 2)
    d or d_model: hidden dim
    n or d_state: latent state dim      (`N` in [1] Algorithm 2)
    expand: expansion factor            (`E` in [1] Section 3.4)
    d_in or d_inner: d * expand         (`D` in [1] Algorithm 2)
    A, B, C, D: state space parameters  (See any state space representation formula)
                                        (B, C are input-dependent (aka selective, a key innovation in Mamba); A, D are not)
    Δ or delta: input-dependent step size
    dt_rank: rank of Δ                  (See [1] Section 3.6 "Parameterization of ∆")

"""
from __future__ import annotations
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass



def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

@dataclass
class ModelArgs:
    d_model: int
    n_layer: int
    num_F_in_J_layers: int
    num_F_in_J_head: int
    temp_emb_dim: int
    expand: int
    dropout: int = 0.0

    ## extra param for inner dct
    cfg: object = None
    
    

    
    ## extra param for mamba robust loss
    return_loss: bool = True

    
    def __post_init__(self):
        self.d_inner = int(self.expand * self.d_model)
            
            
        ## extra param for GCN
        self.adj = [[0, 1], [1, 2],
                            [3, 4], [4, 5],
                            [6, 7], [7, 8], [8,9],
                            [7, 10], [10, 11], [11, 12],
                            [7, 13], [13, 14], [14, 15]]
        ## extra param for inner dct
        self.inner_dct_enable = self.cfg.inner_dct_enable
        self.dct_m = self.cfg.dct_m_all
        self.idct_m = self.cfg.idct_m_all
        self.n_pre = self.cfg.n_pre
        self.n_pre_con = self.cfg.n_pre_cond




class RMSNorm(nn.Module):
    def __init__(self,
                 d_model: int,
                 eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))


    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

        return output
        