import torch
from models.mamba_simple_jointscan import *
from models.MotionTransformer import *
import copy

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


class Mamba(nn.Module):
    def __init__(self, args: ModelArgs):
        """My Mamba model."""
        super().__init__()
        self.args = args
        print(self.args.d_model, self.args.temp_emb_dim)

        if self.args.d_model != self.args.temp_emb_dim: #"d_model and temp_emb_dim must be equal for model."
            self.args.temp_emb_dim = self.args.d_model
            print(f"temp_emb_dim is set to {self.args.temp_emb_dim} to match d_model.")
        self.args.temp_emb_dim = 192

        # time domain conversion
        # Register DCT and IDCT bases
        self.cfg = args.cfg
        self.dct_m = self.cfg.dct_m
        self.idct_m = self.cfg.idct_m
        self.t_his = self.cfg.t_his    # Number of history frames
        self.t_pred = self.cfg.t_pred
        

        self.joint_num = 16
        self.N_pre_num = 10
        self.joint_latent_dim = self.args.d_model // self.joint_num
        self.trans_latent = self.joint_latent_dim * self.joint_num
        self.mamb_latent = self.joint_latent_dim * self.N_pre_num


        self.sequence_embedding = nn.Parameter(torch.zeros(1, self.N_pre_num, self.joint_num, self.joint_latent_dim))
        
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
                    num_head=4,
                    dropout=self.args.dropout,
                ) for _ in range(args.n_layer)])
        self.layer_T = nn.ModuleList()
        for i in range(self.args.n_layer):
            self.layer_T.append(
                # ResidualBlock(args, self.trans_latent)
                TemporalDiffusionTransformerDecoderLayer(
                    latent_dim=self.trans_latent,
                    time_embed_dim=self.args.temp_emb_dim,
                    ffn_dim=2*self.trans_latent,
                    num_head=4,
                    dropout=self.args.dropout,
                )
            )
        print(self.joint_latent_dim, self.mamb_latent, self.trans_latent, self.args.temp_emb_dim)

        self.lm_head = nn.Linear(self.joint_latent_dim, 3)

        self.past_process_block = DCTDenoiseTransformer(
            N=self.cfg.n_pre,
            T=self.cfg.t_his + self.cfg.t_pred,
            t_his=self.cfg.t_his,
            J=self.cfg.joint_num,
            mask_prob=0.3,
            noise_std=0.0,
            cfg=self.cfg
        )

        
        
        in_feat_size = 32 if self.cfg.dataset == "mmfi" else 64
        self.motion_feat_encode = nn.Sequential(
            nn.Linear(in_feat_size, self.args.temp_emb_dim),
            nn.SiLU(),
            nn.Dropout(0.5),
            nn.Linear(self.args.temp_emb_dim, self.args.temp_emb_dim),
        )
        
        self.motion_pred_encode = nn.Sequential(
            nn.Linear(3*self.N_pre_num, self.args.temp_emb_dim),
            nn.SiLU(),
            nn.Dropout(0.5),
            nn.Linear(self.args.temp_emb_dim, self.args.temp_emb_dim),
        )
        
        self.pose_pred_encode = nn.Sequential(
            nn.Linear(3*self.N_pre_num, self.args.temp_emb_dim),
            nn.SiLU(),
            nn.Dropout(0.5),
            nn.Linear(self.args.temp_emb_dim, self.args.temp_emb_dim),
        )
        
        


    def forward(self, x, timesteps, 
                mod=None, 
                feat_time = None, 
                radar_time=None, 
                limb_len = None, 
                mod_time = None, gt_time = None,
                motion_pred = None, motion_feat = None,
                pose_var=None,
                return_attn=False):

        B, N_pre, C = x.shape
        J = C//3
        x = x.reshape(B,N_pre,J,3)
        h = self.embedding(x)
        h = h + self.sequence_embedding
        b,t,v,c = h.shape



        emb = self.time_embed_base(timestep_embedding(timesteps, self.args.d_model)).unsqueeze(dim=1)
        emb_T,emb_M = emb, emb
        
        attn_list = []

        if mod is not None:
            
            motion_pred = torch.matmul(self.cfg.dct_m_all[:self.cfg.n_pre], motion_pred)
            motion_pred = motion_pred.view(B,N_pre,J,3).permute(0,2,1,3).reshape(B,J,-1)
            mod_noise = self.past_process_block(mod, mode='noise')
            
            # motion_feat_proj = self.motion_feat_encode(motion_feat)
            motion_pred_proj = self.motion_pred_encode(motion_pred)
            pose_pred_proj = self.pose_pred_encode(mod_noise)
            
                        
            mod_proj = pose_pred_proj + motion_pred_proj
            
            emb_M = emb_M + mod_proj
            emb_T = emb_T + mod_proj.mean(dim=1, keepdim=True)
            
      
        i = 0
        prelist = []
        for layer_M in self.layers_M:
            if i < (self.args.n_layer // 2):
                prelist.append(h)

                h = h.permute(0, 2, 1, 3).reshape(b,v,t*c)
                if return_attn:
                    h, attn = layer_M(h, emb_M, return_attn=return_attn)
                    attn_list.append(attn)
                else:
                    h = layer_M(h, emb_M, return_attn=return_attn)
                h = h.view(b,v,t,c).permute(0,2,1,3)

                h = h.reshape(b,t,v*c)
                h = self.layer_T[i](h, emb_T)
                h = h.view(b,t,v,c)
                
            elif i >= (self.args.n_layer // 2):

                h = h.permute(0, 2, 1, 3).reshape(b,v,t*c)
                if return_attn:
                    h, attn = layer_M(h, emb_M, return_attn=return_attn)
                    attn_list.append(attn)
                else:
                    h = layer_M(h, emb_M, return_attn=return_attn)
                h = h.view(b,v,t,c).permute(0,2,1,3)

                h = h.reshape(b,t,v*c)
                h = self.layer_T[i](h, emb_T)
                h = h.view(b,t,v,c)

                h += prelist[-1]
                prelist.pop()
            i += 1

        logits = self.lm_head(h)
        logits = logits.reshape(B,N_pre,C).contiguous()

        if return_attn:
            print(len(attn_list), attn_list[0].shape, attn_list[-1].shape)
            return logits, None, attn_list
        else:
            return logits, None
    

class DCTDenoiseTransformer(nn.Module):
    def __init__(self, N=20, T=25, t_his=10, J=17, mask_prob=0.3, noise_std=0.1, cfg=None):
        super().__init__()
        self.N = N            # Number of frequency components
        self.T = T            # Total number of time frames
        self.t_his = t_his    # Number of history frames
        self.J = J            # Number of joints
        self.mask_prob = mask_prob
        self.noise_std = noise_std

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
        x_time = torch.matmul(self.cfg.idct_m_all[:, :self.cfg.n_pre], x_dct)[:, :self.t_his] 
        x_time = x_time.view(B, self.t_his, self.J, 3)
        if self.training == True: # Only apply corruption during training
            if mode == 'mask':
                mask = (torch.rand(B, self.t_his, 1, 1, device=device) < self.mask_prob)
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
        x_time = x_time.view(B, self.t_his, self.J * 3)
        
        
        # pad the output use the last frame of the x_out
        x_out = torch.cat([x_time, x_time[:, [-1], :].repeat(1,self.T-self.t_his,1)], dim=1)
        x_out_dct = torch.matmul(self.cfg.dct_m_all[:self.cfg.n_pre], x_out)
        x_out_dct = x_out_dct.view(B, self.N, self.J, 3).permute(0,2,1,3).reshape(B, self.J, -1)

        return x_out_dct

