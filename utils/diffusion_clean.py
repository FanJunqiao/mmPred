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

    def noise_motion(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None]
        Ɛ = torch.randn_like(x)
        
        if self.diff_err_loss:
            return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ
        else:
            return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, x


    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def inpaint_complete(self, step, x, prev_t, traj_dct, mode_dict, traj_dct_cond, root, dct_transform, dct_detransform):
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
        traj_dct_cond = torch.sqrt(alpha_hat) * traj_dct_cond + torch.sqrt(1 - alpha_hat) * torch.randn_like(traj_dct_cond)


        # back to time domain
        traj_dct_cond = dct_detransform(self.cfg, traj_dct_cond, root)
        x_prev_t_unknown = dct_detransform(self.cfg, x, root)

        # print(traj_dct_cond.shape, traj_dct_cond)


        # concatenation
        x_prev_t_known = traj_dct_cond

        x = torch.mul(mode_dict['mask'], x_prev_t_known) + torch.mul((1 - mode_dict['mask']), x_prev_t_unknown)  # mask

        # back to dct domain
        x, _, _ = dct_transform(self.cfg, x, traj_dct_cond)


        return x
    


    def sample_ddim_progressive(self, model, traj_dct, traj_dct_mod, mode_dict, root, dct_transform, dct_detransform, noise=None, gpu_split = True):
        """
        Generate samples from the model and yield samples from each timestep.

        Args are the same as sample_ddim()
        Returns a generator contains x_{prev_t}, shape as [sample_num, n_pre, 3 * joints_num]
        """
        
        n_hypothesis, sample_num = mode_dict['sample_num']
        EnableComplete = self.EnableComplete
        EnableComplete_step = int(self.ddim_timesteps * 1.0)
        timesteps = self.ddim_timesteps

        


        t = (torch.ones(n_hypothesis, sample_num) * self.ddim_timestep_seq[timesteps-1]).long().to(self.device)
        prev_t = (torch.ones(n_hypothesis, sample_num) * self.ddim_timestep_prev_seq[timesteps-2]).long().to(self.device)
        # x, _ = self.noise_motion(traj_dct, prev_t)

        alpha_hat = self.alpha_hat[t][:, :, None, None]
        alpha_hat_prev = self.alpha_hat[prev_t][:, :, None, None]

        Ɛ = torch.randn_like(traj_dct)
        x = torch.sqrt(alpha_hat) * traj_dct + torch.sqrt(1 - alpha_hat) * Ɛ

        x = x.to(self.device)

   
        with torch.no_grad():
            for i in reversed(range(0, timesteps)):
                t = (torch.ones(n_hypothesis, sample_num) * self.ddim_timestep_seq[i]).long().to(self.device)
                prev_t = (torch.ones(n_hypothesis, sample_num) * self.ddim_timestep_prev_seq[i]).long().to(self.device)
                
                alpha_hat = self.alpha_hat[t][:, :, None, None]
                alpha_hat_prev = self.alpha_hat[prev_t][:, :, None, None]

                if EnableComplete is True: 
                    x = self.inpaint_complete(i,
                                              x,
                                              t,
                                              traj_dct,
                                              mode_dict,
                                              traj_dct_mod,
                                              root, dct_transform, dct_detransform)                    

                # incorporate K multi-hypothesis, input traj_dct and traj_dct_mod K, b, T, C
                K, b, T, C = x.shape
                x = x.reshape(K*b, T, C).contiguous()
                traj_dct_mod = traj_dct_mod.reshape(K*b, T, C).contiguous()
                t = t.reshape(K*b).contiguous()



                predicted_noise = model(x, t, mod=traj_dct_mod)
                
                
                predicted_noise = predicted_noise.reshape(K, b, T, C).contiguous()
                x = x.reshape(K, b, T, C).contiguous()
                traj_dct_mod = traj_dct_mod.reshape(K, b, T, C).contiguous()


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
                    traj_dct,
                    traj_dct_mod,
                    mode_dict,
                    root, dct_transform, dct_detransform):
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
                                                   traj_dct,
                                                   traj_dct_mod,
                                                   mode_dict,
                                                   root, dct_transform, dct_detransform,
                                                   noise=None):
            final = sample
        
        # # incorporate K multi-hypothesis
        # final.reshape(K, b, T, C)

        return final
    

    
