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
from einops import rearrange, repeat, einsum

from .mamba.mamba_ssm import Mamba as MambaBlock_org
from .mamba_paral.mamba_kalman import MambaBlock as MambaBlock_paral
from .mamba_paral.vim import VMambaBlock as VMambaBlock


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
    num_T_in_M_layers: int
    num_T_in_M_head: int

    ## extra param for model design
    vocab_length: int
    vocab_size: int
    cond_length: int
    temp_emb_dim: int
    
    dropout: int = 0.0
    d_state: int = 16
    expand: int = 2
    dt_rank: Union[int, str] = 'auto'
    d_conv: int = 4 
    pad_vocab_size_multiple: int = 8
    conv_bias: bool = True
    bias: bool = False

    ## extra param for inner dct
    cfg: object = None
    
    

    
    ## extra param for mamba robust loss
    return_loss: bool = True

    ## extra param for mamba parallel
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init: str = "random" # "random" or "constant"
    dt_scale: float = 1.0
    dt_init_floor = 1e-4

    rms_norm_eps: float = 1e-5
    base_std: float = 0.02

    inner_layernorms: bool = False # apply layernorms to internal activations
    mup: bool = False
    mup_base_width: float = 128 # width=d_model
    pscan: bool = True # use parallel scan mode or sequential mode when training
    use_cuda: bool = False # use official CUDA implementation when training (not compatible with (b)float16)

    ## extra param for vmamba
    bidirectional: bool = True # use bidirectional MambaBlock

    divide_output: bool = True
    
    def __post_init__(self):
        self.d_inner = int(self.expand * self.d_model)
        
        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)
            
        if self.vocab_size % self.pad_vocab_size_multiple != 0:
            self.vocab_size += (self.pad_vocab_size_multiple
                                - self.vocab_size % self.pad_vocab_size_multiple)
            
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






class StylizationBlock(nn.Module):

    def __init__(self, latent_dim, time_embed_dim, dropout):
        super().__init__()
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, 2 * latent_dim),
        )
        self.norm = nn.LayerNorm(latent_dim)
        self.out_layers = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(nn.Linear(latent_dim, latent_dim)),
        )

    def forward(self, h, emb):
        """
        h: B, T, D
        emb: B, D
        """
        # B, 1, 2D

        if len(emb.shape) == 2:
            emb_out = self.emb_layers(emb).unsqueeze(1)
        else:
            emb_out = self.emb_layers(emb)

        # scale: B, 1, D / shift: B, 1, D
        scale, shift = torch.chunk(emb_out, 2, dim=2)

        h = self.norm(h) * (1 + scale) + shift
        h = self.out_layers(h)
        return h
    
class ResidualBlock_simple(nn.Module):
    def __init__(self, args: ModelArgs):
        """Simple block wrapping Mamba block with normalization and residual connection."""
        super().__init__()
        self.args = args
        self.proj_out = StylizationBlock(self.args.d_model, self.args.temp_emb_dim, self.args.dropout)
        self.mixer = MambaBlock(args)
        # self.mixer = MambaBlock_org(
        #     # This module uses roughly 3 * expand * d_model^2 parameters
        #     d_model=args.d_model, # Model dimension d_model
        #     d_state=args.d_state,  # SSM state expansion factor
        #     d_conv=args.d_conv,    # Local convolution width
        #     expand=args.expand,    # Block expansion factor
        # )
        self.norm = RMSNorm(args.d_model)
        

    def forward(self, x, emb):
        """
        Args:
            x: shape (b, l, d)    (See Glossary at top for definitions of b, l, d_in, n...)
    
        Returns:
            output: shape (b, l, d)

        Official Implementation:
            Block.forward(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py#L297
            
            Note: the official repo chains residual blocks that look like
                [Add -> Norm -> Mamba] -> [Add -> Norm -> Mamba] -> [Add -> Norm -> Mamba] -> ...
            where the first Add is a no-op. This is purely for performance reasons as this
            allows them to fuse the Add->Norm.

            We instead implement our blocks as the more familiar, simpler, and numerically equivalent
                [Norm -> Mamba -> Add] -> [Norm -> Mamba -> Add] -> [Norm -> Mamba -> Add] -> ....
            
        """
        y = self.mixer(self.norm(x)) 
        output = self.proj_out(y, emb) + x

        return output

class ResidualBlock_nondiff(nn.Module):
    def __init__(self, args: ModelArgs):
        """Simple block wrapping Mamba block with normalization and residual connection."""
        super().__init__()
        self.args = args
        self.proj_out = nn.Sequential(
            RMSNorm(args.d_model),
            nn.SiLU(),
            nn.Dropout(p=self.args.dropout),
            zero_module(nn.Linear(self.args.d_model, self.args.d_model)),
        )
        self.mixer = MambaBlock(args)
        # self.mixer = MambaBlock_org(
        #     # This module uses roughly 3 * expand * d_model^2 parameters
        #     d_model=args.d_model, # Model dimension d_model
        #     d_state=args.d_state,  # SSM state expansion factor
        #     d_conv=args.d_conv,    # Local convolution width
        #     expand=args.expand,    # Block expansion factor
        # )
        self.norm = RMSNorm(args.d_model)
        

    def forward(self, x):
        """
        Args:
            x: shape (b, l, d)    (See Glossary at top for definitions of b, l, d_in, n...)
    
        Returns:
            output: shape (b, l, d)

        Official Implementation:
            Block.forward(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py#L297
            
            Note: the official repo chains residual blocks that look like
                [Add -> Norm -> Mamba] -> [Add -> Norm -> Mamba] -> [Add -> Norm -> Mamba] -> ...
            where the first Add is a no-op. This is purely for performance reasons as this
            allows them to fuse the Add->Norm.

            We instead implement our blocks as the more familiar, simpler, and numerically equivalent
                [Norm -> Mamba -> Add] -> [Norm -> Mamba -> Add] -> [Norm -> Mamba -> Add] -> ....
            
        """
        y = self.mixer(self.norm(x)) 
        output = self.proj_out(y) + x

        return output

class ResidualBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        """Simple block wrapping Mamba block with normalization and residual connection."""
        super().__init__()
        self.args = args
        self.proj_out = StylizationBlock(self.args.d_model, self.args.temp_emb_dim, self.args.dropout)
        # self.mixer = MambaBlock(args)
        # self.mixer = MambaBlock_paral(args)
        self.mixer = VMambaBlock(args)
        # self.mixer = MambaBlock_org(
        #     # This module uses roughly 3 * expand * d_model^2 parameters
        #     d_model=args.d_model, # Model dimension d_model
        #     d_state=args.d_state,  # SSM state expansion factor
        #     d_conv=args.d_conv,    # Local convolution width
        #     expand=args.expand,    # Block expansion factor
        # )
        self.norm = RMSNorm(args.d_model)
        

    def forward(self, x, emb):
        """
        Args:
            x: shape (b, l, d)    (See Glossary at top for definitions of b, l, d_in, n...)
    
        Returns:
            output: shape (b, l, d)

        Official Implementation:
            Block.forward(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py#L297
            
            Note: the official repo chains residual blocks that look like
                [Add -> Norm -> Mamba] -> [Add -> Norm -> Mamba] -> [Add -> Norm -> Mamba] -> ...
            where the first Add is a no-op. This is purely for performance reasons as this
            allows them to fuse the Add->Norm.

            We instead implement our blocks as the more familiar, simpler, and numerically equivalent
                [Norm -> Mamba -> Add] -> [Norm -> Mamba -> Add] -> [Norm -> Mamba -> Add] -> ....
            
        """
        y = self.mixer(self.norm(x)) 
        output = self.proj_out(y, emb) + x

        return output
            

class MambaBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        """A single Mamba block, as described in Figure 3 in Section 3.4 in the Mamba paper [1]."""
        super().__init__()
        self.args = args

        self.in_proj = nn.Linear(args.d_model, args.d_inner * 2, bias=args.bias)

        self.conv1d = nn.Conv1d(
            in_channels=args.d_inner,
            out_channels=args.d_inner,
            bias=args.conv_bias,
            kernel_size=args.d_conv,
            groups=args.d_inner,
            padding=args.d_conv - 1,
        )

        # x_proj takes in `x` and outputs the input-specific Δ, B, C
        self.x_proj = nn.Linear(args.d_inner, args.dt_rank + args.d_state * 2, bias=False)
        
        # dt_proj projects Δ from dt_rank to d_in
        self.dt_proj = nn.Linear(args.dt_rank, args.d_inner, bias=True)

        A = repeat(torch.arange(1, args.d_state + 1), 'n -> d n', d=args.d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(args.d_inner))
        self.out_proj = nn.Linear(args.d_inner, args.d_model, bias=args.bias)


        # for attention
        self.kalman_attn = CrossAttention(args.d_state, args.d_state, hid_dim=16, num_head=1, dropout=args.dropout)
        self.attn_dim = 128
        self.state_proj = nn.Linear(args.d_state, self.attn_dim, bias=False)
        self.control_proj = nn.Linear(args.d_state, self.attn_dim, bias=False)
        self.norm_cos = nn.LayerNorm(args.d_state)
        self.cos = nn.CosineSimilarity(dim=1)
        

    def forward(self, x):
        """Mamba block forward. This looks the same as Figure 3 in Section 3.4 in the Mamba paper [1].
    
        Args:
            x: shape (b, l, d)    (See Glossary at top for definitions of b, l, d_in, n...)
    
        Returns:
            output: shape (b, l, d)
        
        Official Implementation:
            class Mamba, https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py#L119
            mamba_inner_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L311
            
        """
        (b, l, d) = x.shape
        
        x_and_res = self.in_proj(x)  # shape (b, l, 2 * d_in)
        (x, res) = x_and_res.split(split_size=[self.args.d_inner, self.args.d_inner], dim=-1)

        x = rearrange(x, 'b l d_in -> b d_in l')
        x = self.conv1d(x)[:, :, :l]
        x = rearrange(x, 'b d_in l -> b l d_in')
        
        x = F.silu(x)
        y = self.ssm(x)
        
        y = y * F.silu(res)
        
        output = self.out_proj(y)

        return output

    
    def ssm(self, x):
        """Runs the SSM. See:
            - Algorithm 2 in Section 3.2 in the Mamba paper [1]
            - run_SSM(A, B, C, u) in The Annotated S4 [2]

        Args:
            x: shape (b, l, d_in)    (See Glossary at top for definitions of b, l, d_in, n...)
    
        Returns:
            output: shape (b, l, d_in)

        Official Implementation:
            mamba_inner_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L311
            
        """
        (d_in, n) = self.A_log.shape

        # Compute ∆ A B C D, the state space parameters.
        #     A, D are input independent (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
        #     ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
        #                                  and is why Mamba is called **selective** state spaces)
        
        A = -torch.exp(self.A_log.float())  # shape (d_in, n)
        D = self.D.float()

        x_dbl = self.x_proj(x)  # (b, l, dt_rank + 2*n)
        
        (delta, B, C) = x_dbl.split(split_size=[self.args.dt_rank, n, n], dim=-1)  # delta: (b, l, dt_rank). B, C: (b, l, n)
        delta = F.softplus(self.dt_proj(delta))  # (b, l, d_in)
        
        y = self.selective_scan(x, delta, A, B, C, D)  # This is similar to run_SSM(A, B, C, u) in The Annotated S4 [2]
        
        return y

    
    def selective_scan(self, u, delta, A, B, C, D):
        """Does selective scan algorithm. See:
            - Section 2 State Space Models in the Mamba paper [1]
            - Algorithm 2 in Section 3.2 in the Mamba paper [1]
            - run_SSM(A, B, C, u) in The Annotated S4 [2]

        This is the classic discrete state space formula:
            x(t + 1) = Ax(t) + Bu(t)
            y(t)     = Cx(t) + Du(t)
        except B and C (and the step size delta, which is used for discretization) are dependent on the input x(t).
    
        Args:
            u: shape (b, l, d_in)    (See Glossary at top for definitions of b, l, d_in, n...)
            delta: shape (b, l, d_in)
            A: shape (d_in, n)
            B: shape (b, l, n)
            C: shape (b, l, n)
            D: shape (d_in,)
    
        Returns:
            output: shape (b, l, d_in)
    
        Official Implementation:
            selective_scan_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L86
            Note: I refactored some parts out of `selective_scan_ref` out, so the functionality doesn't match exactly.
            
        """
        (b, l, d_in) = u.shape
        n = A.shape[1]
        
        # Discretize continuous parameters (A, B)
        # - A is discretized using zero-order hold (ZOH) discretization (see Section 2 Equation 4 in the Mamba paper [1])
        # - B is discretized using a simplified Euler discretization instead of ZOH. From a discussion with authors:
        #   "A is the more important term and the performance doesn't change much with the simplification on B"
        deltaA = torch.exp(einsum(delta, A, 'b l d_in, d_in n -> b l d_in n'))
        deltaB_u = einsum(delta, B, u, 'b l d_in, b l n, b l d_in -> b l d_in n')
        
        # Perform selective scan (see scan_SSM() in The Annotated S4 [2])
        # Note that the below is sequential, while the official implementation does a much faster parallel scan that
        # is additionally hardware-aware (like FlashAttention).
        x = torch.zeros((b, d_in, n), device=deltaA.device)
        ys = []    
        for i in range(l):
            state = deltaA[:, i] * x
            control = deltaB_u[:, i]
            state, control = self.kalman_attn(state, control)

            # weight = self.cos(self.state_proj(self.norm_cos(state.mean(dim=1))), self.control_proj(self.norm_cos(control.mean(dim=1))))
            # weight = (weight + 1) / 2
            # weight = weight.view(b, 1, 1)
            # print(i, weight.shape, weight.min(), weight.max())



            # x = state * (1-weight) + control * weight

            x = state + control
            y = einsum(x, C[:, i, :], 'b d_in n, b n -> b d_in')
            ys.append(y)
        y = torch.stack(ys, dim=1)  # shape (b, l, d_in)
        
        y = y + u * D
    
        return y


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
        
class CrossAttention(nn.Module):

    def __init__(self, latent_dim, mod_dim, hid_dim, num_head, dropout):
        super().__init__()
        self.hid_dim = hid_dim
        self.num_head = num_head
        self.norm = nn.LayerNorm(latent_dim)
        self.text_norm = nn.LayerNorm(mod_dim)
        self.query = nn.Linear(latent_dim, self.hid_dim, bias=False)
        self.key = nn.Linear(mod_dim, self.hid_dim, bias=False)
        self.value = nn.Linear(mod_dim, self.hid_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, xf):
        """
        x: B, T, D
        xf: B, N, L
        """
        """
        x: B, T, D
        xf: B, N, L
        """
        B, T, _ = x.shape
        D = self.hid_dim
        N = xf.shape[1]
        H = self.num_head
        # B, T, 1, D
        query = self.query(self.norm(x)).unsqueeze(2)
        # B, 1, N, D
        key = self.key(self.text_norm(xf)).unsqueeze(1)
        query = query.view(B, T, H, -1)
        key = key.view(B, N, H, -1)
        # B, T, N, H
        attention = torch.einsum('bnhd,bnhd->bnh', query, key) / math.sqrt(D // H)
        # weight = self.dropout(F.softmax(attention, dim=2))
        weight = self.dropout(attention)
        # print(weight[0])
        value = self.value(self.text_norm(xf)).view(B, N, H, -1)
        y2 = torch.einsum('bnh,bnhd->bnhd', weight, value).reshape(B, T, D)
        y1 = torch.einsum('bnh,bnhd->bnhd', 1 - weight, query).reshape(B, T, D)
        return y1, y2