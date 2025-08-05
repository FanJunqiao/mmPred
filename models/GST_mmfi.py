import torch
from models.GST_utils import *
from models.MotionTransformer import *
import copy
from einops import rearrange, repeat, einsum

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    if len(timesteps.shape) == 1:
        
        args = timesteps[:, None].float() * freqs[None, :]
    elif len(timesteps.shape) == 2:
        args = timesteps[:, :, None].float() * freqs[None, None, :]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[..., :1])], dim=-1)
    
    return embedding


class GST(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        print(self.args.d_model, self.args.temp_emb_dim)

        if self.args.d_model != self.args.temp_emb_dim: #"d_model and temp_emb_dim must be equal for model."
            self.args.temp_emb_dim = self.args.d_model
            print(f"temp_emb_dim is set to {self.args.temp_emb_dim} to match d_model.")
        
        self.joint_num = 16
        self.T_num = args.cfg.n_pre
        self.joint_latent_dim = self.args.d_model // self.joint_num

        # time domain conversion
        # Register DCT and IDCT bases
        self.cfg = args.cfg
        self.dct_m = self.cfg.dct_m
        self.idct_m = self.cfg.idct_m
        self.t_his = self.cfg.t_his    # Number of history frames
        self.t_pred = self.cfg.t_pred


        self.trans_latent = self.joint_latent_dim * self.joint_num
        self.mamb_latent = self.joint_latent_dim * self.T_num


        self.sequence_embedding = nn.Parameter(torch.zeros(1, self.T_num, self.joint_num, self.joint_latent_dim))
        
        self.embedding = nn.Linear(3, self.joint_latent_dim)        


        self.time_embed_base = nn.Sequential(
            nn.Linear(self.args.d_model, self.args.temp_emb_dim),
            nn.SiLU(),
            nn.Linear(self.args.temp_emb_dim, self.args.temp_emb_dim),
            
        )
    
        


        self.layers_M = nn.ModuleList([TemporalDiffusionTransformerDecoderLayer(
                    latent_dim=self.mamb_latent,
                    time_embed_dim=self.args.temp_emb_dim,
                    ffn_dim=2*self.mamb_latent,
                    num_head=8,
                    dropout=self.args.dropout,
                ) for _ in range(args.n_layer)])
        self.layer_T = nn.ModuleList()
        for i in range(self.args.n_layer):
            self.layer_T.append(
                TemporalDiffusionTransformerDecoderLayer(
                    latent_dim=self.trans_latent,
                    time_embed_dim=self.args.temp_emb_dim,
                    ffn_dim=2*self.trans_latent,
                    num_head=4,
                    dropout=self.args.dropout,
                )
            )
        print(self.joint_latent_dim, self.mamb_latent, self.trans_latent, self.args.temp_emb_dim)

        self.norm_f = RMSNorm(self.trans_latent)

        self.lm_head = nn.Linear(self.joint_latent_dim, 3)

        cfg = self.args.cfg
        self.cfg = cfg

        self.past_process_block = DCTDenoiseTransformer(
            N=cfg.n_pre,
            T=cfg.t_his + cfg.t_pred,
            t_his=cfg.t_his,
            J=cfg.joint_num,
            h_dim=128,
            out_dim=self.args.temp_emb_dim,
            depth=2,
            num_heads=2,
            mask_prob=0.3,
            noise_std=0.1,
            cfg=cfg
        )

        self.point_encode = RadarPointEncoder(mlp_hidden=128, point_feat_dim=64, joint_feat_dim=self.args.temp_emb_dim)
        self.motion_pred_encode = nn.Sequential(
            nn.Linear(3*self.T_num, self.args.temp_emb_dim),
            nn.SiLU(),
            nn.Linear(self.args.temp_emb_dim, self.args.temp_emb_dim),
        )
        print(cfg.dataset)
        in_feat_size = 32 if cfg.dataset == "mmfi" else 64
        self.motion_feat_encode = nn.Sequential(
            nn.Linear(in_feat_size, self.args.temp_emb_dim),
            nn.SiLU(),
            nn.Linear(self.args.temp_emb_dim, self.args.temp_emb_dim),
        )
        # Register DCT and IDCT bases
        self.cfg = cfg
        self.dct_m = cfg.dct_m
        self.idct_m = cfg.idct_m
        


    def forward(self, x, timesteps, 
                mod=None, 
                feat_time = None, 
                radar_time=None, 
                limb_len = None, 
                mod_time = None, gt_time = None,
                motion_pred = None, motion_feat = None,
                pose_var=None,
                return_attn = False):

    
    
        '''
        Input Projection and Positional Embedding
        '''
        
        B, T, C = x.shape
        J = C//3
        x = x.reshape(B,T,J,3)
        h = self.embedding(x)
        h = h + self.sequence_embedding
        b,t,v,c = h.shape


        '''
        Diffusion Time-step Embedding
        '''
        emb = self.time_embed_base(timestep_embedding(timesteps, self.args.d_model)).unsqueeze(dim=1)
        emb_T = emb
        emb_M = emb
        
        '''
        Dual-domain feature fusion
        '''
        
        loss_aux = None
        if mod is not None:
            
            motion_pred = torch.matmul(self.cfg.dct_m_all[:self.cfg.n_pre], motion_pred)
            motion_pred = motion_pred.view(B,T,J,3).permute(0,2,1,3).reshape(B,J,-1)
            motion_feat_proj = self.motion_feat_encode(motion_feat)
            motion_pred_proj = self.motion_pred_encode(motion_pred)
            predicted_gt, mod_proj = self.past_process_block(mod, mode='noise')
                        
            mod_proj =  mod_proj  +  motion_feat_proj + motion_pred_proj 
            
            emb_M = emb_M + mod_proj #+ feat_proj
            emb_T = emb_T + mod_proj.mean(dim=1, keepdim=True) #+ feat_proj.mean(dim=1, keepdim=True)
            


            

        '''
        U-Net architecture
        '''
        attn_list = []
        
        i = 0
        prelist = []
        for layer_M in self.layers_M:
            if i < (self.args.n_layer // 2):
                prelist.append(h)

                '''
                Reshape for S-Transformer
                '''
                h = h.permute(0, 2, 1, 3).reshape(b,v,t*c)
                if return_attn:
                    h, attn = layer_M(h, emb_M, return_attn=return_attn)
                    attn_list.append(attn)
                else:
                    h = layer_M(h, emb_M, return_attn=return_attn)
                h = h.view(b,v,t,c).permute(0,2,1,3)


                '''
                Reshape for F-Transformer
                '''
                h = h.reshape(b,t,v*c)
                h = self.layer_T[i](h, emb_T)
                h = h.view(b,t,v,c)
                
            elif i >= (self.args.n_layer // 2):


                '''
                Reshape for S-Transformer
                '''
                h = h.permute(0, 2, 1, 3).reshape(b,v,t*c)
                if return_attn:
                    h, attn = layer_M(h, emb_M, return_attn=return_attn)
                    attn_list.append(attn)
                else:
                    h = layer_M(h, emb_M, return_attn=return_attn)
                h = h.view(b,v,t,c).permute(0,2,1,3)

                '''
                Reshape for F-Transformer
                '''
                h = h.reshape(b,t,v*c)
                h = self.layer_T[i](h, emb_T)
                h = h.view(b,t,v,c)

                h += prelist[-1]
                prelist.pop()
            i += 1


        logits = self.lm_head(h)
        logits = logits.reshape(B,T,C).contiguous()

        if return_attn:
            print(len(attn_list), attn_list[0].shape, attn_list[-1].shape)
            return logits, loss_aux, attn_list
        else:
            return logits, loss_aux
    



class MotionConfidenceModulator(nn.Module):
    """
    Predict per-joint log-variance from features, output normalized attention scores via softmax,
    and compute heteroscedastic uncertainty loss.

    Args:
        feat_dim: dimension of joint features
        hidden_dim: hidden units in variance MLP
        lambda_var: weight for variance regularization in loss
    """
    def __init__(self, feat_dim: int = 64, hidden_dim: int = 32, lambda_var: float = 1.0):
        super().__init__()
        self.lambda_var = lambda_var
        # MLP: feat_dim -> hidden_dim -> 1 (log σ² per joint)
        self.variance_net = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, feat: torch.Tensor, gt_motion: torch.Tensor, pred_motion: torch.Tensor):
        """
        Args:
            feat: [B, T, J, C] joint features
            gt_motion: [B, T, J, 3] ground-truth joint coords
            pred_motion: [B, T, J, 3] predicted joint coords
        Returns:
            log_var:    [B, T, J, 1] predicted log variance per joint
            attn_score: [B, T, J, 1] normalized attention via softmax over J
            loss_unc:   scalar tensor, heteroscedastic uncertainty loss
        """
        B, T, J, C = feat.shape
        log_var = self.variance_net(feat.reshape(-1, C)).reshape(B, T, J, 1)

        # raw confidence per joint
        attn_score = torch.sigmoid(-0.5 * log_var * 2)
        loss_unc = None
        if gt_motion != None:

            loss_unc = self._uncertainty_loss(gt_motion, pred_motion, log_var)

        return log_var, attn_score, loss_unc

    def _uncertainty_loss(self, gt: torch.Tensor, pred: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Heteroscedastic uncertainty loss:
          L = mean( se/(2*var) + 0.5*log_var*lambda_var )
        where se = sum over coords of (gt-pred)^2 and var=exp(log_var).
        """
        var = torch.exp(log_var)
        se = (gt - pred).pow(2).sum(dim=-1, keepdim=True)
        mse_term = (se / (2 * var)).mean()
        var_term = (0.5 * log_var).mean()
        return mse_term + self.lambda_var * var_term


class DCTDenoiseTransformer(nn.Module):
    def __init__(self, N=20, T=25, t_his=10, J=17, h_dim=128,out_dim=512, depth=4, num_heads=8, mask_prob=0.3, noise_std=0.1, cfg=None):
        super().__init__()
        self.N = N            # Number of frequency components
        self.T = T            # Total number of time frames
        self.t_his = t_his    # Number of history frames
        self.J = J            # Number of joints
        self.mask_prob = mask_prob
        self.noise_std = noise_std

        self.input_proj = nn.Linear(J * 3, h_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=h_dim, nhead=num_heads, dim_feedforward=h_dim * 2, batch_first=True),
            num_layers=depth
        )
        self.output_proj = nn.Linear(h_dim, J * 3)

        # 1D convolution + mean pooling + MLP for projecting features
        self.feat_conv1d = nn.Sequential(
            nn.Conv1d(h_dim, h_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(h_dim, h_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # Global average pooling
            nn.Flatten()
        )
        self.feat_proj = nn.Linear(h_dim, out_dim)
        self.feat_proj1 = nn.Linear(3 * N, out_dim)

        # Register DCT and IDCT bases
        self.cfg = cfg
        self.dct_m = cfg.dct_m
        self.idct_m = cfg.idct_m

        self.mask_token = nn.Parameter(torch.randn(1, 1, self.J, 3))

    def forward(self, x_dct, mode='mask'):
        """
        x_dct: (B, N, J*3)
        Returns: denoised x_dct (B, N, J*3)
        """
        B, device = x_dct.size(0), x_dct.device
        # 1. Inverse DCT to time domain
        x_time = torch.matmul(self.cfg.idct_m_all[:, :self.cfg.n_pre], x_dct)
        
        # 2. Corrupt history (first t_his frames)
        x_time = x_time.view(B, self.T, self.J, 3)
        if self.training == True: # Only apply corruption during training
            
            if mode == 'mask':
                mask = (torch.rand(B, self.T, 1, 1, device=device) < self.mask_prob)
                mask[:, self.t_his:] = False
                x_time = x_time.masked_fill(mask, 0.)
            elif mode == 'noise':
                x_time[:, :self.t_his] += torch.randn_like(x_time[:, :self.t_his]) * self.noise_std
            elif mode == 'maskLeg':
                x_time[:, :self.t_his] += torch.randn_like(x_time[:, :self.t_his]) * self.noise_std
                # Mask out 80% of the first 6 joints (along J dimension) for [:,:,0:6,:]
                leg_parts = [0, 1, 2, 3, 4, 5]  # example leg joint indices
                mask = (torch.rand(B, self.T, len(leg_parts), 1, device=device) < 0.8)

                x_time[:, :, leg_parts, :] = torch.where(
                    mask,
                    self.mask_token[:, :, leg_parts, :],  # [1, 1, 4, 3] → broadcastable
                    x_time[:, :, leg_parts, :]
                )


        x_time = x_time.view(B, self.T, self.J * 3)

        x_out = x_time[:,:self.t_his,:]
        # pad the output use the last frame of the x_out
        x_out = torch.cat([x_out, x_out[:, [-1], :].repeat(1,self.T-self.t_his,1)], dim=1)
        x_out_dct = torch.matmul(self.cfg.dct_m_all[:self.cfg.n_pre], x_out)
        x_out_dct = x_out_dct.view(B, self.N, self.J, 3).permute(0,2,1,3).reshape(B, self.J, -1)

        x_feat_out = self.feat_proj1(x_out_dct)


        # 4. DCT back to frequency domain
        return x_out_dct, x_feat_out


import torch
import torch.nn as nn
import torch.nn.functional as F

class RadarPointEncoder(nn.Module):
    def __init__(self, mlp_hidden=64, point_feat_dim=64, joint_feat_dim=312, num_joints=16):
        super().__init__()

        self.joint_feat_dim = joint_feat_dim
        self.point_feat_dim = point_feat_dim
        self.num_joints = num_joints

        # Project 7D (xyz + 3feat + t) → 64D
        self.point_mlp = nn.Sequential(
            nn.Linear(7, mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden, point_feat_dim)
        )

        # Temporal projection: 64 → 312 via Conv1D
        self.temporal_proj = nn.Sequential(
            nn.Conv1d(point_feat_dim, joint_feat_dim, kernel_size=1),
            nn.ReLU(inplace=True)
        )

        # Temporal max pooling
        self.temporal_pool = nn.AdaptiveMaxPool1d(1)

        # MLP to reduce [B, J*312] → [B, 312]
        self.joint_mlp = nn.Sequential(
            nn.Linear(joint_feat_dim * num_joints, joint_feat_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        x: [B, T, J, N, 6] – radar point cloud input
        Output: [B, 312] – final encoded feature
        """
        B, T, J, N, C = x.shape
        assert C == 6, "Expected input with 6 features (xyz + 3 feat)"
        assert J == self.num_joints, f"Expected {self.num_joints} joints, got {J}"

        # Add normalized time as an additional feature channel
        t_vals = torch.linspace(0, 1, T, device=x.device).view(1, T, 1, 1, 1).expand(B, T, J, N, 1)
        x = torch.cat([x, t_vals], dim=-1)  # [B, T, J, N, 7]

        # Point MLP
        x = x.view(B * T * J * N, 7)
        x = self.point_mlp(x)  # [B*T*J*N, 64]
        x = x.view(B, T, J, N, self.point_feat_dim)  # [B, T, J, N, 64]

        # Max-pool over N
        x = x.max(dim=3)[0]  # [B, T, J, 64]

        # Temporal Conv1D projection
        x = x.permute(0, 2, 3, 1)            # [B, J, 64, T]
        x = x.reshape(B * J, self.point_feat_dim, T)  # [B*J, 64, T]
        x = self.temporal_proj(x)           # [B*J, 312, T]

        # Temporal pooling
        x = self.temporal_pool(x).squeeze(-1)  # [B*J, 312]
        x = x.view(B, J, self.joint_feat_dim)  # [B, J, 312]

        # Flatten joint dim and apply MLP
        x = x.view(B, J * self.joint_feat_dim)  # [B, J*312]
        x = self.joint_mlp(x)                   # [B, 312]


        return x