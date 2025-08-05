import csv

import pandas as pd
from utils.metrics import *
from tqdm import tqdm
from utils import *
from utils.script import sample_preprocessing, sample_preprocessing_nondiff
from torch.utils.data import DataLoader


tensor = torch.tensor
DoubleTensor = torch.DoubleTensor
FloatTensor = torch.FloatTensor
LongTensor = torch.LongTensor
ByteTensor = torch.ByteTensor
ones = torch.ones
zeros = torch.zeros


def compute_stats(diffusion, test_dataset_list, model, logger, cfg, mask=None, multiFlag=True):
    """
    The GPU is strictly needed because we need to give predictions for multiple samples in parallel and repeat for
    several (K=50) times.
    """

    # TODO reduce computation complexity
    def get_prediction(batch, model_select, K):
        mode_dict = sample_preprocessing(batch, K=K, cfg=cfg, n = -1)

        sampled_motion = diffusion.sample_ddim(model_select,
                                               mode_dict)

        traj_est = cfg.input_detransform(cfg, sampled_motion, mode_dict["noise"]["root"])

        # traj_est.shape (K, 125, 48)
        traj_est = traj_est.cpu()
        return traj_est

    partition = [10]
    scenes = list(range(1,8)) if cfg.dataset == "mmBody" else [2,3,4,5,13,14,17,18,19,20,21,22,23,27]
    metrics = ['ADE', 'FDE', "R-ADE", "R-FDE"]
    stats_names = ['APD', "limb_err", "limb_var","jit_err", "jit_var"]
    for par in partition:
        for me in metrics:
            stats_names.append(me + str(par))
    for sce in scenes:
        for me in metrics:
            stats_names.append(me + "s" + str(sce))
    print(stats_names)
    stats_meter = {x: {f'mmPred_scene{y}': AverageMeter() for y in range(len(test_dataset_list))} for x in stats_names}

    for idx in range(len(test_dataset_list)):
        test_dataset = test_dataset_list[idx]
        # convert to dataloader
        test_dataset = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4,
                                pin_memory=True)
        logger.info(f'preparing evaluation dataset for the {idx} scene....')

        num_samples = 0
        K = 10
        pred = []
        gt_group = []
        scene_group = []
        for batch in tqdm(test_dataset, total=len(test_dataset), desc='Test prepare round', unit='batch', leave=False):    

            data, gt, sub = batch["observed"], batch["pose"], batch["sub"]
            num_samples += data.shape[0]

            gt = gt.view(-1, (cfg.t_his + cfg.t_pred), cfg.joint_n, 3)

            gt = gt.float()[:,:,1:,:]
            gt = gt.reshape(gt.shape[0], gt.shape[1], -1)
            gt = gt[None,...]

            pred_i_nd = get_prediction(batch, model, K)
            pred.append(pred_i_nd)
            gt_group.append(gt)
            if cfg.dataset == "mmBody": sub = sub//10
            scene_group.append(sub)

    
 
        pred = torch.cat(pred, dim=1)
        gt_group = torch.cat(gt_group, dim=1)
        scene_group = torch.cat(scene_group, dim=0)
        # pred [50, 5187, 125, 48] in h36m
        r_pred = pred[:, :, cfg.t_his:, :] - pred[:, :, [cfg.t_his], :]
        r_gt_group = gt_group[:, :, cfg.t_his:, :] - gt_group[:, :, [cfg.t_his], :]
        pred = pred[:, :, cfg.t_his:, :]
        gt_group = gt_group[:, :, cfg.t_his:, :]


        # pred [50, 5187, 100, 48]
        for j in range(0, num_samples):

            apd, ade, fde = compute_metrics(pred[:, j, :, :],
                                            gt_group[:, j, :, :],
                                            partitions=partition)
            
            _, rade, rfde = compute_metrics(r_pred[:, j, :, :],
                                            r_gt_group[:, j, :, :],
                                            partitions=partition)
            limb_err, limb_var, jit_err, jit_var = compute_limb_metrics(pred[:, j, :, :],
                                            gt_group[:, j, :, :])
            
            stats_meter['APD'][f'mmPred_scene{idx}'].update(apd)
            stats_meter['limb_err'][f'mmPred_scene{idx}'].update(limb_err)
            stats_meter['limb_var'][f'mmPred_scene{idx}'].update(limb_var)
            stats_meter['jit_err'][f'mmPred_scene{idx}'].update(jit_err)
            stats_meter['jit_var'][f'mmPred_scene{idx}'].update(jit_var)
            for i in range(len(partition)):   
                stats_meter["ADE"+str(partition[i])][f'mmPred_scene{idx}'].update(ade[i])
                stats_meter['FDE'+str(partition[i])][f'mmPred_scene{idx}'].update(fde[i])
                stats_meter["R-ADE"+str(partition[i])][f'mmPred_scene{idx}'].update(rade[i])
                stats_meter['R-FDE'+str(partition[i])][f'mmPred_scene{idx}'].update(rfde[i])
                
            sce_i = scene_group[j]
            stats_meter["ADE"+f"s{sce_i}"][f'mmPred_scene{idx}'].update(ade[-1])
            stats_meter['FDE'+f"s{sce_i}"][f'mmPred_scene{idx}'].update(fde[-1])
            stats_meter["R-ADE"+f"s{sce_i}"][f'mmPred_scene{idx}'].update(rade[-1])
            stats_meter['R-FDE'+f"s{sce_i}"][f'mmPred_scene{idx}'].update(rfde[-1])
                
            
        for stats in stats_names:
            str_stats = f'{stats}: ' + ' '.join(
                [f'{x}: {y.avg:.4f}' for x, y in stats_meter[stats].items()]
            )
            logger.info(str_stats)


    # save stats in csv
    file_latest = '%s/stats_latest.csv'
    file_stat = '%s/stats.csv'
    with open(file_latest % cfg.result_dir, 'w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=['Metric'] + [f'mmPred_scene{y}'for y in range(len(test_dataset_list))])
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
        df = pd.concat([df2]+ [df1[f'mmPred_scene{y}'] for y in range(len(test_dataset_list))], axis=1, ignore_index=True)
        df.to_csv(file_stat % cfg.result_dir, index=False)


def compute_stats_nondiff(test_dataset_list, model, logger, cfg, mask, multiFlag=True, dct_transform = None, dct_detransform=None):
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

        mode_dict, traj_dct, traj_dct_cond, root = sample_preprocessing_nondiff(traj, cfg, mode='metrics', mask=mask, dct_transform = dct_transform)
        traj_est_list = []
        for i in range(0, traj_dct_cond.shape[0]//128+1):
            sampled_motion = model_select(traj_dct_cond[i*128:(i+1)*128])
            traj_est = dct_detransform(cfg, sampled_motion, root)

            # traj_est.shape (K, 125, 48)
            traj_est = traj_est.cpu().detach().numpy()
            traj_est_list.append(traj_est)
        traj_est = np.concatenate(traj_est_list, axis=0)
        print(traj_est.shape)
        return traj_est


    partition = [3,6,9,12,16]
    scenes = list(range(1,8)) if cfg.dataset == "mmBody" else [2,3,4,5,13,14,17,18,19,20,21,22,23,27]
    metrics = ['ADE', 'FDE', "R-ADE", "R-FDE"]
    stats_names = ['APD']
    for par in partition:
        for me in metrics:
            stats_names.append(me + str(par))
    for sce in scenes:
        for me in metrics:
            stats_names.append(me + "s" + str(sce))
    print(stats_names)
    stats_meter = {x: {f'mmPred_scene{y}': AverageMeter() for y in range(len(test_dataset_list))} for x in stats_names}

    for idx in range(len(test_dataset_list)):
        multimodal_dict = test_dataset_list[idx]

        gt_group = multimodal_dict['gt_group']
        data_group = multimodal_dict['data_group']
        scene_group = multimodal_dict['sub_group']
        gt_group = gt_group[..., 1:, :].reshape(gt_group.shape[0], gt_group.shape[1], -1)[:, cfg.t_his:, :]
        num_samples = data_group.shape[0]

    

        # It generates a prediction for all samples in the test set
        # So we need loop for K times
        pred = get_prediction(data_group, model, mask)
        pred = np.expand_dims(pred, axis=0)
        gt_group = np.expand_dims(gt_group, axis=0)
        # Use GPU to accelerate
        gt_group = torch.from_numpy(gt_group).to('cuda')


        pred = torch.from_numpy(pred).to('cuda')
        
        r_pred = pred[:, :, cfg.t_his:, :] - pred[:, :, [cfg.t_his], :]
        r_gt_group = gt_group[:, :, :, :] - gt_group[:, :, [0], :]
        pred = pred[:, :, cfg.t_his:, :]
        gt_group = gt_group[:, :, :, :]
        
        print(pred.shape, gt_group.shape, r_pred.shape, r_gt_group.shape)

        # pred [50, 5187, 100, 48]
        for j in range(0, num_samples):

            apd, ade, fde = compute_metrics(pred[:, j, :, :],
                                            gt_group[:, j, :, :],
                                            partitions=partition)
            
            _, rade, rfde = compute_metrics(r_pred[:, j, :, :],
                                            r_gt_group[:, j, :, :],
                                            partitions=partition)
            
            stats_meter['APD'][f'mmPred_scene{idx}'].update(apd)
            for i in range(len(partition)):   
                stats_meter["ADE"+str(partition[i])][f'mmPred_scene{idx}'].update(ade[i])
                stats_meter['FDE'+str(partition[i])][f'mmPred_scene{idx}'].update(fde[i])
                stats_meter["R-ADE"+str(partition[i])][f'mmPred_scene{idx}'].update(rade[i])
                stats_meter['R-FDE'+str(partition[i])][f'mmPred_scene{idx}'].update(rfde[i])
                
            sce_i = scene_group[j]
            stats_meter["ADE"+f"s{sce_i}"][f'mmPred_scene{idx}'].update(ade[-1])
            stats_meter['FDE'+f"s{sce_i}"][f'mmPred_scene{idx}'].update(fde[-1])
            stats_meter["R-ADE"+f"s{sce_i}"][f'mmPred_scene{idx}'].update(rade[-1])
            stats_meter['R-FDE'+f"s{sce_i}"][f'mmPred_scene{idx}'].update(rfde[-1])
                
            
        for stats in stats_names:
            str_stats = f'{stats}: ' + ' '.join(
                [f'{x}: {y.avg:.4f}' for x, y in stats_meter[stats].items()]
            )
            logger.info(str_stats)

    # save stats in csv
    file_latest = '%s/stats_latest.csv'
    file_stat = '%s/stats.csv'
    with open(file_latest % cfg.result_dir, 'w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=['Metric'] + [f'mmPred_scene{y}'for y in range(len(test_dataset_list))])
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
        df = pd.concat([df2]+ [df1[f'mmPred_scene{y}'] for y in range(len(test_dataset_list))], axis=1, ignore_index=True)
        df.to_csv(file_stat % cfg.result_dir, index=False)
