import torch
import numpy as np
from utils.script import *

def dct_transform(cfg, traj, traj_pad, x_H=None):
    
    traj_dct = torch.matmul(cfg.dct_m_all[:cfg.n_pre], traj)
    traj_dct_mod = torch.matmul(cfg.dct_m_all[:cfg.n_pre_cond], traj_pad)
    root = traj_pad[...,[0],:]
    return traj_dct, traj_dct_mod, root

def dct_detransform(cfg, sampled_motion, root):
    return torch.matmul(cfg.idct_m_all[:, :cfg.n_pre], sampled_motion)

# def dct_transform(cfg, traj, traj_pad, x_H=None):

#     # step 1: find X_H and normalized with X_H
#     if x_H == None:
#         x_H = traj_pad[...,[cfg.t_his-1],:]
#     traj = traj - x_H

#     traj_dct = torch.matmul(cfg.dct_m_all[:cfg.n_pre], traj)
#     traj_dct_mod = torch.matmul(cfg.dct_m_all[:cfg.n_pre_cond], traj_pad)
    

#     root = x_H
#     return traj_dct, traj_dct_mod, root

# def dct_detransform(cfg, sampled_motion, root=None):

#     sampled_motion = torch.matmul(cfg.idct_m_all[:, :cfg.n_pre], sampled_motion)

#     # step 4
#     if root != None:
#         sampled_motion = sampled_motion + root
#     return sampled_motion


def dct_dev1_transform(cfg, traj, traj_pad):
    traj_dct = (torch.matmul(cfg.dct_m_all1[:cfg.n_pre], traj)*100)[...,1:,:]
    traj_dct_mod = (torch.matmul(cfg.dct_m_all1[:cfg.n_pre_cond], traj_pad)*100)[...,1:,:]
    root = traj_pad[...,[0],:]
    return traj_dct, traj_dct_mod, root

def dct_dev1_detransform(cfg, sampled_motion, root):
    zeros_pad = torch.zeros_like(sampled_motion[...,[1],:])
    sampled_motion = torch.cat([zeros_pad, sampled_motion], dim=-2)
    vel_pr = torch.matmul(cfg.idct_m_all1[:, :cfg.n_pre], sampled_motion/100)
    return vel_to_cart(vel_pr, root)

def null_transform(cfg, traj, traj_pad):
    root = traj_pad[...,[0],:]
    return traj, traj_pad, root

def null_detransform(cfg, sampled_motion, root):
    return sampled_motion




def vel_transform(cfg, traj, traj_pad):
    traj_dct = cart_to_vel(traj)
    traj_dct_mod = cart_to_vel(traj_pad)
    root = traj_pad[...,[0],:]
    return traj_dct, traj_dct_mod, root

def vel_detransform(cfg, sampled_motion, root):
    return vel_to_cart(sampled_motion, root)

