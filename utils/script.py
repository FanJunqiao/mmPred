import copy
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from data_loader.mmBody_pc import mmBody
from data_loader.mmfi import mmfi
from data_loader.MMVR import MMVR

from utils import padding_traj
from utils import *
from utils.visualization import render_animation
from utils.diffusion_shortterm import Diffusion

from tqdm import tqdm

from scipy.spatial.distance import pdist, squareform

def create_diffusion(cfg):
    """
    create TransLinear model and Diffusion
    """
    diffusion = Diffusion(
        noise_steps=cfg.noise_steps,
        motion_size=(cfg.n_pre, 3 * cfg.joint_num),  # 3 means x, y, z
        device=cfg.device, padding=cfg.padding,
        EnableComplete=cfg.Complete,
        ddim_timesteps=cfg.ddim_timesteps,
        refine_timesteps=cfg.refine_timesteps,
        scheduler=cfg.scheduler,
        mod_test=cfg.mod_test,
        dct=cfg.dct_m_all,
        idct=cfg.idct_m_all,
        n_pre=cfg.n_pre,
        n_pre_cond=cfg.n_pre_cond,
        dct_enable=cfg.dct_enable,
        vel_enable = cfg.vel_enable,
        diff_err_loss = cfg.diff_err_loss,
        diff_err_mask = cfg.diff_err_mask,
        cfg = cfg,
    )
    return diffusion

def set_dataloader_mmBody(cfg):
    data_dir = cfg.data_dir
    output_dir = f'{cfg.output_dir}'
    input_n = cfg.t_his
    output_n = cfg.t_pred_duplicate if cfg.train_stage == 1 else cfg.t_pred
    joint_n = cfg.joint_n
    skip_rate = cfg.skip_rate_train

    all_data = True if cfg.data_all == 'all' else False

    test_dataset_list = []
    
    load_train = (cfg.mode.split("_")[0] == "train")
    train_loader, val_loader = None, None
    if load_train or cfg.train_stage == 1:
        train_dataset = mmBody(data_dir, input_n, output_n, skip_rate, split=0, miss_rate=(cfg.miss_rate / 100), miss_scale=cfg.miss_scale,
                            miss_type=cfg.miss_type_train, all_data=all_data, joint_n=joint_n, dct_i=cfg.dct_i)
        print('>>> train_dataset length: {:d}'.format(train_dataset.__len__()))
        val_dataset = mmBody(data_dir, input_n, output_n, skip_rate, split=2, miss_rate=(cfg.miss_rate / 100), miss_scale=cfg.miss_scale,
                                miss_type=cfg.miss_type_test[0], all_data=all_data, joint_n=joint_n, dct_i=cfg.dct_i)
        print('>>> val_dataset length: {:d}'.format(val_dataset.__len__()))
        train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=16,
                                pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=16,
                                    pin_memory=True)
    
    for miss_type_test in cfg.miss_type_test:
        test_dataset = mmBody(data_dir, input_n, output_n, skip_rate, split=2, miss_rate=(cfg.miss_rate / 100), miss_scale=cfg.miss_scale,
                            miss_type=miss_type_test, all_data=all_data, joint_n=joint_n, dct_i=cfg.dct_i)
        test_dataset_list.append(test_dataset)
        print('>>> test_dataset length: {:d}'.format(test_dataset.__len__()))
    
    return train_loader, val_loader, test_dataset_list

def set_dataloader_mmfi(cfg):
    data_dir = cfg.data_dir
    output_dir = f'{cfg.output_dir}'
    input_n = cfg.t_his
    output_n = cfg.t_pred_duplicate if cfg.train_stage == 1 else cfg.t_pred
    joint_n = cfg.joint_n
    skip_rate = cfg.skip_rate_train

    all_data = True if cfg.data_all == 'all' else False
    test_dataset_list = []
    
    load_train = (cfg.mode.split("_")[0] == "train")
    train_loader, val_loader = None, None
    if load_train or cfg.train_stage == 1:
        train_dataset = mmfi(data_dir, input_n, output_n, skip_rate, split=0, miss_rate=(cfg.miss_rate / 100), miss_scale=cfg.miss_scale,
                        miss_type=cfg.miss_type_train, all_data=all_data, joint_n=joint_n, dct_i=cfg.dct_i)
        print('>>> train_dataset length: {:d}'.format(train_dataset.__len__()))
        val_dataset = mmfi(data_dir, input_n, output_n, skip_rate, split=2, miss_rate=(cfg.miss_rate / 100), miss_scale=cfg.miss_scale,
                            miss_type=cfg.miss_type_test[0], all_data=all_data, joint_n=joint_n, dct_i=cfg.dct_i)
        print('>>> val_dataset length: {:d}'.format(val_dataset.__len__()))
        train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=16,
                                pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=16,
                                    pin_memory=True)
    
    for miss_type_test in cfg.miss_type_test:
        test_dataset = mmfi(data_dir, input_n, output_n, skip_rate, split=2, miss_rate=(cfg.miss_rate / 100), miss_scale=cfg.miss_scale,
                            miss_type=miss_type_test, all_data=all_data, joint_n=joint_n, dct_i=cfg.dct_i)
        test_dataset_list.append(test_dataset)
        print('>>> test_dataset length: {:d}'.format(test_dataset.__len__()))
    

    
    return train_loader, val_loader, test_dataset_list

def set_dataloader_MMVR(cfg):
    data_dir = cfg.data_dir
    output_dir = f'{cfg.output_dir}'
    input_n = cfg.t_his
    output_n = cfg.t_pred
    joint_n = cfg.joint_n
    skip_rate = cfg.skip_rate_train

    all_data = True if cfg.data_all == 'all' else False

    val_dataset_list = []
    test_dataset_list = []
    
    for miss_type_test in cfg.miss_type_test:
        test_dataset = MMVR(input_n, output_n, split=2,
                        miss_type=cfg.miss_type_train)
        test_dataset_list.append(test_dataset)
    
    print('>>> Training dataset length: {:d}'.format(test_dataset.__len__()))
    # test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=16,
    #                             pin_memory=True)
    
    return None, None, test_dataset_list

def get_multimodal_gt_full(logger, dataset_multi_test_list, args, cfg, multiFlag=True):
    """
    calculate the multi-modal data, currently only single modal, TO BE WRITE
    """
    multi_dataset_list = []
    idx = 0
    for dataset_multi_test in dataset_multi_test_list:
        # convert to dataloader
        dataset_multi_test = DataLoader(dataset_multi_test, batch_size=cfg.batch_size, shuffle=False, num_workers=16,
                                pin_memory=True)
        logger.info(f'preparing full evaluation dataset for the {idx} scene....')
        data_group = []
        gt_group = []
        feat_group = []
        sub_group = []
        num_samples = 0
        # data_gen_multi_test = dataset_multi_test.iter_generator(step=cfg.t_his)
        for batch in tqdm(dataset_multi_test, total=len(dataset_multi_test), desc='Test prepare round', unit='batch', leave=False):
            num_samples += 1
            data, gt, mask, timepoints, feat, sub = batch["observed"], batch["pose"], batch["mask"], batch["timepoints"], batch["pose_feat"], batch["sub"]
            data = data.view(-1, (cfg.t_his + cfg.t_pred), cfg.joint_n, 3)
            gt = gt.view(-1, (cfg.t_his + cfg.t_pred), cfg.joint_n, 3)
            
            if cfg.dataset == "mmBody": sub = sub//10

            data_group.append(data.detach().cpu().numpy())
            gt_group.append(gt.detach().cpu().numpy())
            feat_group.append(feat.detach().cpu().numpy())
            sub_group.append(sub.detach().cpu().numpy())
        data_group = np.concatenate(data_group, axis=0)
        gt_group = np.concatenate(gt_group, axis=0)
        feat_group = np.concatenate(feat_group, axis=0)
        sub_group = np.concatenate(sub_group, axis=0)
        print(data_group.shape, gt_group.shape, feat_group.shape, sub_group.shape)
        # all_data = data_group[..., 1:, :].reshape(data_group.shape[0], data_group.shape[1], -1)
        # all_gt = gt_group[..., 1:, :].reshape(data_group.shape[0], data_group.shape[1], -1)
        # gt_group = all_gt[:, cfg.t_his:, :]

        if multiFlag:
            all_start_pose = all_gt[:, cfg.t_his - 1, :]
            pd = squareform(pdist(all_start_pose))
            traj_gt_arr = []
            num_mult = []
            for i in range(pd.shape[0]):
                ind = np.nonzero(pd[i] < args.multimodal_threshold)
                traj_gt_arr.append(all_data[ind][:, cfg.t_his:, :])
                num_mult.append(len(ind[0]))
            # np.savez_compressed('./data/data_3d_h36m_test.npz',data=all_data)
            # np.savez_compressed('./data/data_3d_humaneva15_test.npz',data=all_data)
            num_mult = np.array(num_mult)
            logger.info('=' * 80)
            logger.info(f'#1 future: {len(np.where(num_mult == 1)[0])}/{pd.shape[0]}')
            logger.info(f'#<10 future: {len(np.where(num_mult < 10)[0])}/{pd.shape[0]}')
        logger.info('done...')
        logger.info('=' * 80)
        if multiFlag:
            multi_dataset_list.append({'traj_gt_arr': traj_gt_arr,
                    'data_group': data_group,
                    # 'gt_group': gt_group,
                    'num_samples': num_samples})
            del traj_gt_arr, data_group, gt_group, num_samples, num_mult
        else:
            multi_dataset_list.append({
                    'data_group': data_group,
                    'gt_group': gt_group,
                    'feat_group': feat_group,
                    'num_samples': num_samples,
                    'sub_group': sub_group})
        idx += 1
    
    return multi_dataset_list


def display_exp_setting(logger, cfg):
    """
    log the current experiment settings.
    """
    logger.info('=' * 80)
    log_dict = cfg.__dict__.copy()
    for key in list(log_dict):
        if 'dir' in key or 'path' in key or 'dct' in key:
            del log_dict[key]
    del log_dict['zero_index']
    del log_dict['idx_pad']
    logger.info(log_dict)
    logger.info('=' * 80)


def process_batch(batch, cfg):
    observed, pose, feat, radar, limb_len, motion_pred, motion_feat, observed_multi, observed_var, mask = batch["observed"], batch["pose"], batch["pose_feat"], batch["radar"], batch["limb_len"], batch["motion_pred"], batch["motion_feat"], batch["observed_multi"], batch["observed_variance"], batch["mask"]
    mode_dict = {}

    pose = pose[:,:(cfg.t_his + cfg.t_pred),:]
    observed = observed[:,:(cfg.t_his + cfg.t_pred),:]

    traj_gt = pose.view(-1, (cfg.t_his + cfg.t_pred), cfg.joint_n, 3)
    traj_gt = traj_gt[..., 1:, :].reshape([traj_gt.shape[0], cfg.t_his + cfg.t_pred, -1])
    traj_gt = traj_gt.to(torch.float).to(cfg.device) #tensor(traj_np, device=cfg.device, dtype=cfg.dtype)


    traj_pad_gt = padding_traj(traj_gt, cfg.padding, cfg.idx_pad, cfg.zero_index)
    traj_dct_gt, traj_dct_mod_gt, root_gt = cfg.input_transform(cfg, traj_gt, traj_pad_gt)
    mode_dict["gt"] = {"pad_time": traj_pad_gt, "all_time": traj_gt, "all_dct": traj_dct_gt, "pad_dct": traj_dct_mod_gt, "root": root_gt}
    

    traj_np_noise = observed.view(-1, (cfg.t_his + cfg.t_pred), cfg.joint_n, 3)
    traj_np_noise = traj_np_noise[..., 1:, :].reshape([traj_np_noise.shape[0], cfg.t_his + cfg.t_pred, -1])
    traj_np_noise = traj_np_noise.to(torch.float).to(cfg.device) #tensor(traj_np, device=cfg.device, dtype=cfg.dtype)
    

    traj_pad_noise = padding_traj(traj_np_noise, cfg.padding, cfg.idx_pad, cfg.zero_index)
    traj_dct_noise, traj_dct_mod_noise, root_noise = cfg.input_transform(cfg, traj_np_noise, traj_pad_noise)

    mode_dict["noise"] = {"pad_time": traj_pad_noise, "all_time": traj_np_noise, "all_dct": traj_dct_noise, "pad_dct": traj_dct_mod_noise, "root": root_noise}

    feat = feat.to(torch.float).to(cfg.device) 
    radar = radar.to(torch.float).to(cfg.device)
    limb_len = limb_len.to(torch.float).to(cfg.device)
    motion_pred = motion_pred.to(torch.float).to(cfg.device)
    motion_feat = motion_feat.to(torch.float).to(cfg.device)
    observed_multi = observed_multi.to(torch.float).to(cfg.device)
    observed_var = observed_var[:,:,1:].to(torch.float).to(cfg.device)
    mask = mask[:,:,1:].to(torch.float).to(cfg.device)

    mode_dict["others"] = {"feat": feat, "radar": radar, "limb_len": limb_len, "motion_pred": motion_pred, "motion_feat": motion_feat, "observed_multi": observed_multi, "observed_var":observed_var, "mask": mask}
    
    return mode_dict

def sample_preprocessing(batch, K, cfg, n = -1):
    """
    This function is used to preprocess traj for sample_ddim().
    input : traj_seq, cfg, mode
    output: a dict for specific mode,
            traj_dct,
            traj_dct_mod
    """
    mode_dict = process_batch(batch, cfg)

    if n == -1:
        n = mode_dict["gt"]["all_time"].shape[0]
    else:
        for key1 in mode_dict.keys():
            for key2 in mode_dict[key1].keys():
                value = mode_dict[key1][key2]
                mode_dict[key1][key2] = value[n:n+1]
        n = mode_dict["gt"]["all_time"].shape[0]


    mask = torch.zeros([1, n, cfg.t_his + cfg.t_pred, mode_dict["gt"]["all_time"].shape[-1]//3, 1]).to(cfg.device)
    for i in range(0, cfg.t_his):
        mask[:, :, i, :, :] = 1
    if mode_dict["others"]["mask"] is not None:
        mask = mode_dict["others"]["mask"][None,...]

    for key1 in mode_dict.keys():
        for key2 in mode_dict[key1].keys():
            value = mode_dict[key1][key2]
            mode_dict[key1][key2] = value.unsqueeze(0).repeat(K, *([1] * value.dim()))

    if np.random.random() > cfg.mod_test:
        mode_dict["noise"]["pad_dct"] = None

    mode_dict.update({
        'mask': mask,
        'sample_num': (K, n),
    })
    return mode_dict


def sample_preprocessing_nondiff(traj, cfg, mode, mask, dct_transform=None):
    """
    This function is used to preprocess traj for sample_ddim().
    input : traj_seq, cfg, mode
    output: a dict for specific mode,
            traj_dct,
            traj_dct_mod
    """

    K, n = traj.shape[0], traj.shape[1]

    mask = torch.zeros([K, n, cfg.t_his + cfg.t_pred, traj.shape[-1]]).to(cfg.device)
    for i in range(0, cfg.t_his):
        mask[:, :, i, :] = 1
    
    traj_pad = padding_traj(traj, cfg.padding, cfg.idx_pad, cfg.zero_index)

    traj_dct, traj_dct_mod, root = dct_transform(cfg, traj, traj_pad)

    if np.random.random() > cfg.mod_test:
        traj_dct_mod = None

    return {'mask': mask,
            'sample_num': (K,n),
            'mode': mode}, traj_dct, traj_dct_mod, root
