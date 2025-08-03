import copy

import torch

from utils import *
import numpy as np
import math
from copy import deepcopy


def sqrt_beta_schedule(timesteps, s=0.0001):
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps) / timesteps
    alphas_cumprod = 1 - torch.sqrt(t + s)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def sigmoid_beta_schedule(timesteps, start=-3., end=3., tau=0.7, clamp_min=1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class Diffusion:
    def __init__(self, noise_steps=1000,
                 beta_start=1e-4,
                 beta_end=0.02,
                 motion_size=(35, 66),
                 device="cuda",
                 padding=None,
                 EnableComplete=True,
                 ddim_timesteps=100,
                 refine_timesteps = 10,
                 scheduler='Linear',
                 model_type='data',
                 mod_enable=True,
                 mod_test=0.5,
                 dct=None,
                 idct=None,
                 cfg = None,
                 n_pre=10,
                 n_pre_cond=10,
                 dct_enable = True,
                 vel_enable = False,
                 diff_err_loss = True,
                 diff_err_mask = False,
                 diff_err_mask_rate = 0.8):
        self.noise_steps = noise_steps
        self.beta_start = (1000 / noise_steps) * beta_start
        self.beta_end = (1000 / noise_steps) * beta_end
        self.motion_size = motion_size
        self.device = device

        self.scheduler = scheduler  # 'Cosine', 'Sqrt', 'Linear', 'Sigmoid'
        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        self.ddim_timesteps = ddim_timesteps
        self.refine_timesteps = refine_timesteps

        self.model_type = model_type
        self.padding = padding  # 'Zero', 'Repeat', 'LastFrame'
        self.EnableComplete = EnableComplete
        self.mod_enable = mod_enable
        self.mod_test = mod_test
        
        self.ddim_timestep_seq = np.asarray(
            list(range(0, self.noise_steps, self.noise_steps // self.ddim_timesteps))) + 1
        self.ddim_timestep_prev_seq = np.append(np.array([0]), self.ddim_timestep_seq[:-1])

        self.dct = dct
        self.idct = idct
        self.cfg = cfg
        self.n_pre = n_pre
        self.n_pre_cond = n_pre_cond
        self.dct_enable = dct_enable
        self.vel_enable = vel_enable
        self.diff_err_loss = diff_err_loss
        self.diff_err_mask = diff_err_mask
        self.diff_err_mask_rate = diff_err_mask_rate


    def prepare_noise_schedule(self):
        if self.scheduler == 'Linear':
            return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
        elif self.scheduler == 'Cosine':
            return cosine_beta_schedule(self.noise_steps)
        elif self.scheduler == 'Sqrt':
            return sqrt_beta_schedule(self.noise_steps)
        elif self.scheduler == 'Sigmoid':
            return sigmoid_beta_schedule(self.noise_steps)
        else:
            raise NotImplementedError(f"unknown scheduler: {self.scheduler}")

    def noise_motion(self, x, x_real, t, is_train=False):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None]
        Ɛ = torch.randn_like(x)
        x_t = sqrt_alpha_hat * x_real + sqrt_one_minus_alpha_hat * Ɛ
        # if is_train and torch.rand(1).item() < 0.2:  # 20% probability
        #     x_t = sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ
        #     Ɛ = (x_t - sqrt_alpha_hat * x_real) / sqrt_one_minus_alpha_hat

        
        if self.diff_err_loss:
            return x_t, Ɛ
        else:
            return x_t, x
        
    def calculate_grad(self, x_dct, x_time):
        with torch.enable_grad():
            x_dct = torch.autograd.Variable(x_dct, requires_grad = True)
            x_time = torch.autograd.Variable(x_time, requires_grad = True)

            x_time_pr = torch.matmul(self.cfg.idct_m_all[:,:self.cfg.n_pre], x_dct[:,:self.cfg.n_pre])
            x_time = torch.matmul(self.cfg.idct_m_all[:, :self.cfg.n_pre], x_time[:,:self.cfg.n_pre])

            # print(x_time[0, :25, :])

            F_x_time = torch.nn.functional.mse_loss(x_time_pr[:, :self.cfg.t_his], x_time[:, :self.cfg.t_his])
            # print(torch.pow((x_time_pr - x_time)[:, :self.cfg.t_his],2))
            # F_x_time = torch.pow((x_time_pr - x_time)[:, :self.cfg.t_his],2).sum(dim=-1).mean(dim=0).mean()

            out_delta = torch.autograd.grad(F_x_time, x_dct)[0]
            print("loss", F_x_time)

            return out_delta
    

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def inpaint_complete(self, x, prev_t, mode_dict, traj_dct_norm, root):
        """
        perform mask completion

        Args:
            step: current diffusion timestep
            x: x in prev_t step
            prev_t:  timestep in prev_t
            traj_dct: DCT coefficient of the traj,
                    shape as [sample_num, n_pre, 3 * joints_num]
            mode_dict: mode helper dict as sample_ddim()

        Returns:
            completed sample
        """
        # add noise in DCT domain
        alpha_hat = self.alpha_hat[prev_t][:, :, None, None]
        traj_dct_norm_noise = torch.sqrt(alpha_hat) * traj_dct_norm + torch.sqrt(1 - alpha_hat) * torch.randn_like(traj_dct_norm)


        # back to time domain
        x_prev_t_known_noise = self.cfg.input_detransform(self.cfg, traj_dct_norm_noise, root)
        x_prev_t_unknown = self.cfg.input_detransform(self.cfg, x, root)


        # concatenation
        x_prev_t_known = x_prev_t_known_noise
        K,n,t,C = x_prev_t_known.shape
        x = (torch.mul(mode_dict['mask'], x_prev_t_known.reshape(K,n,t,C//3,3)) + torch.mul((1 - mode_dict['mask']), x_prev_t_unknown.reshape(K,n,t,C//3,3))).reshape(K,n,t,C)  # mask

        # back to dct domain
        x, _, _ = self.cfg.input_transform(self.cfg, x, x, x_H = root)


        return x
    


    def sample_ddim_progressive(self, model, mode_dict, noise=None):
        """
        Generate samples from the model and yield samples from each timestep.

        Args are the same as sample_ddim()
        Returns a generator contains x_{prev_t}, shape as [sample_num, n_pre, 3 * joints_num]
        """
        
        n_hypothesis, sample_num = mode_dict['sample_num']
        EnableComplete = self.EnableComplete
        EnableComplete_step = int(self.ddim_timesteps * 1.0)
        timesteps = self.ddim_timesteps

        traj_dct_mod = mode_dict['noise']["pad_dct"]
        traj_dct = mode_dict['gt']["all_dct"]
        traj_gt_time = mode_dict['gt']["all_time"]
        root = mode_dict['noise']["root"]
        
        return_attn = self.cfg.return_attn
        attn_list_list = []
        



        


            
        t = (torch.ones(n_hypothesis, sample_num) * self.ddim_timestep_seq[timesteps-1]).long().to(self.device)

        alpha_hat = self.alpha_hat[t][:, :, None, None]
        


        traj_dct_norm = self.cfg.input_detransform(self.cfg,traj_dct_mod,root=None)
        traj_dct_norm,_,_ = self.cfg.input_transform(self.cfg, traj_dct_norm, traj_dct_norm, x_H=root)

        Ɛ = torch.randn_like(traj_dct_norm)
        x = torch.sqrt(alpha_hat) * traj_dct_norm + torch.sqrt(1 - alpha_hat) * Ɛ 

        x = x.to(self.device)


        
    
   
        with torch.no_grad():
            for i in reversed(range(0, timesteps)):
                t = (torch.ones(n_hypothesis, sample_num) * self.ddim_timestep_seq[i]).long().to(self.device)
                prev_t = (torch.ones(n_hypothesis, sample_num) * self.ddim_timestep_prev_seq[i]).long().to(self.device)
                
                alpha_hat = self.alpha_hat[t][:, :, None, None]
                alpha_hat_prev = self.alpha_hat[prev_t][:, :, None, None]

                if EnableComplete is True: # and i != timesteps-1:
                    x = self.inpaint_complete(x,
                                              t,
                                              mode_dict,
                                              traj_dct_norm,
                                              root)                    
                    EnableComplete_step -= 1
                    if EnableComplete_step == 0:
                        EnableComplete = False

                # incorporate K multi-hypothesis, input traj_dct and traj_dct_mod K, b, T, C
                K, b, T, C = x.shape
                x = x.reshape(K*b, T, C).contiguous()
                t = t.reshape(K*b).contiguous()
                



                # Reshape mode_dict values from (K, b, ...) to (K*b, ...)
                pad_dct = mode_dict["noise"]["pad_dct"]
                mod = pad_dct.reshape(K * b, *pad_dct.shape[2:]) if pad_dct is not None else None

                feat_time = mode_dict["others"]["feat"].reshape(K * b, *mode_dict["others"]["feat"].shape[2:])
                radar_time = mode_dict["others"]["radar"].reshape(K * b, *mode_dict["others"]["radar"].shape[2:])
                limb_len = mode_dict["others"]["limb_len"].reshape(K * b, *mode_dict["others"]["limb_len"].shape[2:])
                motion_pred = mode_dict["others"]["motion_pred"].reshape(K * b, *mode_dict["others"]["motion_pred"].shape[2:])
                motion_feat = mode_dict["others"]["motion_feat"].reshape(K * b, *mode_dict["others"]["motion_feat"].shape[2:])
                pose_var = mode_dict["others"]["observed_var"].reshape(K * b, *mode_dict["others"]["observed_var"].shape[2:])
                mod_time = mode_dict["noise"]["pad_time"].reshape(K * b, *mode_dict["noise"]["pad_time"].shape[2:])
                gt_time = None

                if return_attn:
                    
                    predicted_noise, loss_aux, attn_list = model(
                        x, t,
                        mod=mod,
                        feat_time=feat_time,
                        radar_time=radar_time,
                        limb_len=limb_len,
                        mod_time=mod_time[:, :self.cfg.t_his],
                        gt_time=gt_time,
                        motion_pred = motion_pred,
                        motion_feat = motion_feat,
                        pose_var = pose_var[:, :self.cfg.t_his],
                        return_attn=return_attn
                    )
                    attn_list_list.append(attn_list)
                    print(f"attn_list{i}", len(attn_list_list), len(attn_list_list[0]))
                    if len(attn_list_list) == 100:
                        print(motion_pred.shape, mod_time.shape)
                        from plot_paper import visualize_hypothesis_attn,plot_dct_energy_heatmap,plot_selected_poses,plot_dct_N_variants_multi_joints_flat
                        visualize_hypothesis_attn(attn_list_list, save_dir="./attn_matrix", step_range=(95, -1), layer_range=(3, -1),attn_value_range=(0,0.4))
                        
                        plot_dct_energy_heatmap(motion_pred[0,:8]-motion_pred[0,[0]], N=8, save_path="./attn_matrix/pred_dct_energy_heatmap.png", legend_range=[0,0.08])
                        plot_dct_energy_heatmap(mod_time[0,:8]-mod_time[0,[0]], N=8, save_path="./attn_matrix/pred_time_energy_heatmap.png", legend_range=[0,0.1])
                        plot_dct_energy_heatmap(traj_gt_time[0,0], N=10, save_path="./attn_matrix/gt_dct_energy_heatmap.png")
                        plot_selected_poses(traj_gt_time[0,0], save_path="./attn_matrix/gt_time_domain_poses.png", num_frames=6, N=0)
                        plot_dct_N_variants_multi_joints_flat(traj_gt_time[0,0],coord=2,joint_ids=[1,2], save_path="./attn_matrix/gt_time_domain_poses_line.png",N_list=[1,2,3,4])

                else:
                    predicted_noise, loss_aux = model(
                        x, t,
                        mod=mod,
                        feat_time=feat_time,
                        radar_time=radar_time,
                        limb_len=limb_len,
                        mod_time=mod_time[:, :self.cfg.t_his],
                        gt_time=gt_time,
                        motion_pred = motion_pred,
                        motion_feat = motion_feat,
                        pose_var = pose_var[:, :self.cfg.t_his],
                        return_attn=return_attn
                    )

                # incorporate K multi-hypothesis, input traj_dct and traj_dct_mod K, b, T, C
                predicted_noise = predicted_noise.reshape(K, b, T, C).contiguous()
                x = x.reshape(K, b, T, C).contiguous()


                if self.diff_err_loss:    
                    predicted_x0 = (x - torch.sqrt((1. - alpha_hat)) * predicted_noise) / torch.sqrt(alpha_hat)
                    x_prev = torch.sqrt(alpha_hat_prev) * predicted_x0 + torch.sqrt(1 - alpha_hat_prev) * (predicted_noise)
                else:
                    
                    predicted_x0 = predicted_noise
                    predicted_noise_step = (x - torch.sqrt(alpha_hat) * predicted_x0) / torch.sqrt(1 - alpha_hat)
                    x_prev = torch.sqrt(alpha_hat_prev) * predicted_x0 + torch.sqrt(1 - alpha_hat_prev) * predicted_noise_step
                

                x = x_prev

                yield x 

    def sample_ddim(self,
                    model,
                    mode_dict):
        """
        Generate samples from the model.

        Args:
            model: the model to predict noise
            traj_dct: DCT coefficient of the traj,
                shape as [sample_num, n_pre, 3 * joints_num]
            traj_dct_mod: equal to traj_dct or None when no modulation
            mode_dict: a dict containing the following keys:
                 - 'mask': [[1, 1, ..., 0, 0, 0]
                            [1, 1, ..., 0, 0, 0]
                            ...
                            [1, 1, ..., 0, 0, 0]], mask for observation
                 - 'sample_num': sample_num for different modes
                 - 'mode': mode name, e.g. 'pred', 'control'....
                 when mode is 'switch', there are two additional keys:
                 - 'traj_switch': traj to switch to
                 - 'mask_end': [[0, 0, ...., 1, 1, 1]
                                [0, 0, ...., 1, 1, 1]
                                ...
                                [0, 0, ...., 1, 1, 1]], mask for switch
                 when mode is 'control', there are one additional key:
                 - 'traj_fix': retained the fixed part of the traj
                    and current mask will be:
                                [[0, 0, ...., 1, 1, 1]
                                [1, 1, ...., 1, 1, 1]
                                ...
                                [0, 0, ...., 1, 1, 1]]
        Returns: sample
        """
        # refinement
        # if self.refine_timesteps > 0:
        #     final_cond = traj_dct
        #     for sample in self.sample_ddim_progressive(model,
        #                                             traj_dct,
        #                                             traj_dct_mod,
        #                                             mode_dict,
        #                                             noise=traj_dct):
        #         final_cond = sample
            

        #     # update traj_dct_mod
        #     if self.dct_enable:
        #         traj_dct = final_cond
        #         x_cond = torch.matmul(self.idct[:, :self.n_pre],
        #                                 final_cond[:, :self.n_pre])  # return time domain
                
        #         traj_dct_mod = torch.matmul(self.dct[:self.n_pre_cond], x_cond)

        # generation


        final = None
        for sample in self.sample_ddim_progressive(model,
                                                   mode_dict,
                                                   noise=None):
            final = sample
        
        # # incorporate K multi-hypothesis
        # final.reshape(K, b, T, C)

        return final
    

    
