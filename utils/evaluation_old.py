import csv

import pandas as pd
from utils.metrics import *
from tqdm import tqdm
from utils import *
from utils.script import sample_preprocessing

tensor = torch.tensor
DoubleTensor = torch.DoubleTensor
FloatTensor = torch.FloatTensor
LongTensor = torch.LongTensor
ByteTensor = torch.ByteTensor
ones = torch.ones
zeros = torch.zeros


def compute_stats(diffusion, multimodal_dict_list, model, logger, cfg, mask, multiFlag=True, dct_transform = None, dct_detransform=None):
    """
    The GPU is strictly needed because we need to give predictions for multiple samples in parallel and repeat for
    several (K=50) times.
    """

    # TODO reduce computation complexity
    def get_prediction(data, model_select, mask, feat=None):
        mode_dict = sample_preprocessing(data, feat, cfg, mode='metrics', mask=mask, dct_transform=dct_transform)

        sampled_motion = diffusion.sample_ddim(model_select,
                                               mode_dict,
                                               dct_transform, dct_detransform,)

        traj_est = dct_detransform(cfg, sampled_motion, mode_dict["root"])
        traj_est = traj_est

        # traj_est.shape (K, 125, 48)
        traj_est = traj_est.cpu().numpy()
        return traj_est

    if multiFlag:
        stats_names = ['APD', 'ADE', 'FDE', 'MMADE', 'MMFDE']
    else:
        stats_names = ['APD', 'ADE', 'FDE']
    stats_meter = {x: {f'mmPred_scene{y}': AverageMeter() for y in range(len(multimodal_dict_list))} for x in stats_names}

    for idx in range(len(multimodal_dict_list)):
        multimodal_dict = multimodal_dict_list[idx]

        gt_group = multimodal_dict['gt_group']
        data_group = multimodal_dict['data_group']
        feat_group = multimodal_dict['feat_group']
        gt_group = gt_group[..., 1:, :].reshape(data_group.shape[0], data_group.shape[1], -1)[:, cfg.t_his:, :]
        if multiFlag:
            traj_gt_arr = multimodal_dict['traj_gt_arr']
        num_samples = data_group.shape[0]



        K = 10
        pred = []
        batch_size = 16
        test_size = data_group.shape[0]

        # for k in tqdm(range(K), position=0):
        #     data_k = tensor(data_group, device=cfg.device, dtype=torch.float32)[:,:,1:,:]
        #     data_k = data_k.reshape(data_k.shape[0], data_k.shape[1], -1)
        #     data_k = data_k[None, :, :, :]
        #     feat_k = tensor(feat_group, device=cfg.device, dtype=torch.float32)
        #     feat_k = feat_k[None, :, :, :]
        #     pred_i_nd = get_prediction(data_k, model, mask, feat_k)
        #     pred.append(pred_i_nd)

    
 
        # pred = np.concatenate(pred, axis=0)
        for b in tqdm(range(0, data_group.shape[0]//batch_size+1), position=0):
            idx_start = b * batch_size
            if idx_start < test_size:
                idx_end = min(test_size, (b+1)*batch_size)
                data_b = tensor(data_group[idx_start:idx_end, ...], device=cfg.device, dtype=torch.float32)[:,:,1:,:]
                data_b = data_b.reshape(data_b.shape[0], data_b.shape[1], -1)
                data_b = data_b[None,...].repeat(K, 1, 1, 1)
                print(data_b.shape, data_group.shape)

                feat_b = tensor(feat_group[idx_start:idx_end, ...], device=cfg.device, dtype=torch.float32)
                feat_b = feat_b[None, :, :, :].repeat(K, 1, 1, 1)

                pred_i_nd = get_prediction(data_b, model, mask, feat_b)
                pred.append(pred_i_nd)

                break
        pred = np.concatenate(pred, axis=1)
        # pred [50, 5187, 125, 48] in h36m
        pred = pred[:, :, cfg.t_his:, :]
        # Use GPU to accelerate
        try:
            gt_group = torch.from_numpy(gt_group).to('cuda')
        except:
            pass
        try:
            pred = torch.from_numpy(pred).to('cuda')
        except:
            pass
        # pred [50, 5187, 100, 48]

        for j in range(0, 16):
            if multiFlag:
                apd, ade, fde, mmade, mmfde = compute_all_metrics(pred[:, j, :, :],
                                                                        gt_group[j][np.newaxis, ...],
                                                                        traj_gt_arr[j])
                stats_meter['APD'][f'mmPred_scene{idx}'].update(apd)
                stats_meter['ADE'][f'mmPred_scene{idx}'].update(ade)
                stats_meter['FDE'][f'mmPred_scene{idx}'].update(fde)
                stats_meter['MMADE'][f'mmPred_scene{idx}'].update(mmade)
                stats_meter['MMFDE'][f'mmPred_scene{idx}'].update(mmfde)
            else:

                apd, ade, fde = compute_metrics(pred[:, j, :, :],
                                                gt_group[j][np.newaxis, ...])
                stats_meter['APD'][f'mmPred_scene{idx}'].update(apd)
                stats_meter['ADE'][f'mmPred_scene{idx}'].update(ade)
                stats_meter['FDE'][f'mmPred_scene{idx}'].update(fde)
                
            
        for stats in stats_names:
            str_stats = f'{stats}: ' + ' '.join(
                [f'{x}: {y.avg:.4f}' for x, y in stats_meter[stats].items()]
            )
            logger.info(str_stats)


    # save stats in csv
    file_latest = '%s/stats_latest.csv'
    file_stat = '%s/stats.csv'
    with open(file_latest % cfg.result_dir, 'w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=['Metric'] + [f'mmPred_scene{y}'for y in range(len(multimodal_dict_list))])
        writer.writeheader()
        for stats, meter in stats_meter.items():
            new_meter = {}
            for x, y in meter.items():
                print(x, y.avg.cpu().numpy())
                new_meter[x] = y.avg.cpu().numpy()
            new_meter['Metric'] = stats
            writer.writerow(new_meter)
    df1 = pd.read_csv(file_latest % cfg.result_dir)

    if os.path.exists(file_stat % cfg.result_dir) is False:
        df1.to_csv(file_stat % cfg.result_dir, index=False)
    else:
        df2 = pd.read_csv(file_stat % cfg.result_dir)
        df = pd.concat([df2]+ [df1[f'mmPred_scene{y}'] for y in range(len(multimodal_dict_list))], axis=1, ignore_index=True)
        df.to_csv(file_stat % cfg.result_dir, index=False)


def compute_stats_nondiff(multimodal_dict_list, model, logger, cfg, mask, multiFlag=True, dct_transform = None, dct_detransform=None):
    """
    The GPU is strictly needed because we need to give predictions for multiple samples in parallel and repeat for
    several (K=50) times.
    """

    # TODO reduce computation complexity
    def get_prediction(data, model_select, mask):
        model_select.eval()
        traj_np = data[..., 1:, :].transpose([0, 2, 3, 1])
        traj = tensor(traj_np, device=cfg.device, dtype=torch.float32)
        traj = traj.reshape([traj.shape[0], -1, traj.shape[-1]]).transpose(1, 2)
        # traj.shape: [*, t_his + t_pre, 3 * joints_num]

        mode_dict, traj_dct, traj_dct_cond, root = sample_preprocessing(traj, cfg, mode='metrics', mask=mask, dct_transform = dct_transform)
        traj_est_list = []
        for i in range(0, traj_dct_cond.shape[0]//128+1):
            sampled_motion = model_select(traj_dct_cond[i*128:(i+1)*128])

            # if cfg.dct_enable:
            #     traj_est = torch.matmul(cfg.idct_m_all[:, :cfg.n_pre], sampled_motion)
            # else:
            #     if cfg.vel_enable:
            #         # traj_est = vel_to_cart(sampled_motion, traj_dct_cond[:,[0],:])
            #         traj_est = sampled_motion / 10
            #     else:
            #         traj_est = sampled_motion

            traj_est = dct_detransform(cfg, sampled_motion, root)

            # traj_est.shape (K, 125, 48)
            traj_est = traj_est.cpu().detach().numpy()
            traj_est_list.append(traj_est)
        traj_est = np.concatenate(traj_est_list, axis=0)
        print(traj_est.shape)
        return traj_est


    if multiFlag:
        stats_names = ['APD', 'ADE', 'FDE', 'MMADE', 'MMFDE']
    else:
        stats_names = ['APD', 'ADE', 'FDE']
    stats_meter = {x: {f'mmPred_scene{y}': AverageMeter() for y in range(len(multimodal_dict_list))} for x in stats_names}

    for idx in range(len(multimodal_dict_list)):
        multimodal_dict = multimodal_dict_list[idx]

        # gt_group = multimodal_dict['gt_group']
        data_group = multimodal_dict['data_group']
        gt_group = data_group[..., 1:, :].reshape(data_group.shape[0], data_group.shape[1], -1)[:, cfg.t_his:, :]
        if multiFlag:
            traj_gt_arr = multimodal_dict['traj_gt_arr']
        num_samples = data_group.shape[0]

    


        # It generates a prediction for all samples in the test set
        # So we need loop for K times
        pred = get_prediction(data_group, model, mask)
        pred = np.expand_dims(pred, axis=0)
        # Use GPU to accelerate
        try:
            gt_group = torch.from_numpy(gt_group).to('cuda')
        except:
            pass
        try:
            pred = torch.from_numpy(pred).to('cuda')
        except:
            pass
        # pred [50, 5187, 100, 48]
        for j in range(0, num_samples):
            if multiFlag:
                apd, ade, fde, mmade, mmfde = compute_all_metrics(pred[:, j, cfg.t_his:, :],
                                                                        gt_group[j][np.newaxis, ...],
                                                                        traj_gt_arr[j])
                stats_meter['APD'][f'mmPred_scene{idx}'].update(apd)
                stats_meter['ADE'][f'mmPred_scene{idx}'].update(ade)
                stats_meter['FDE'][f'mmPred_scene{idx}'].update(fde)
                stats_meter['MMADE'][f'mmPred_scene{idx}'].update(mmade)
                stats_meter['MMFDE'][f'mmPred_scene{idx}'].update(mmfde)
            else:
                apd, ade, fde = compute_metrics(pred[:, j, cfg.t_his:, :],
                                                    gt_group[j][np.newaxis, ...])
                stats_meter['APD'][f'mmPred_scene{idx}'].update(apd)
                stats_meter['ADE'][f'mmPred_scene{idx}'].update(ade)
                stats_meter['FDE'][f'mmPred_scene{idx}'].update(fde)
        for stats in stats_names:
            str_stats = f'{stats}: ' + ' '.join(
                [f'{x}: {y.avg:.4f}' for x, y in stats_meter[stats].items()]
            )
            logger.info(str_stats)

    # save stats in csv
    file_latest = '%s/stats_latest.csv'
    file_stat = '%s/stats.csv'
    with open(file_latest % cfg.result_dir, 'w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=['Metric'] + [f'mmPred_scene{y}'for y in range(len(multimodal_dict_list))])
        writer.writeheader()
        for stats, meter in stats_meter.items():
            new_meter = {}
            for x, y in meter.items():
                print(x, y.avg.cpu().numpy())
                new_meter[x] = y.avg.cpu().numpy()
            new_meter['Metric'] = stats
            writer.writerow(new_meter)
    df1 = pd.read_csv(file_latest % cfg.result_dir)

    if os.path.exists(file_stat % cfg.result_dir) is False:
        df1.to_csv(file_stat % cfg.result_dir, index=False)
    else:
        df2 = pd.read_csv(file_stat % cfg.result_dir)
        df = pd.concat([df2]+ [df1[f'mmPred_scene{y}'] for y in range(len(multimodal_dict_list))], axis=1, ignore_index=True)
        df.to_csv(file_stat % cfg.result_dir, index=False)
