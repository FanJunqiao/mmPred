from torch import tensor
from utils import *
from utils.script import sample_preprocessing, sample_preprocessing_nondiff
import copy


def pose_generator(data_loader, model_select, diffusion, cfg, mode=None,
                   action=None, nrow=1, n_pre = -1):
    """
    stack k rows examples in one gif

    The logic of 'draw_order_indicator' is to cheat the render_animation(),
    because this render function only identify the first two as context and gt, which is a bit tricky to modify.
    """
    traj_np = None
    j = None
    while True:
        poses = {}
        draw_order_indicator = -1
        for k in range(0, nrow):

            if mode == 'test':
                batch = data_loader # this dataloader is a data
                batch = {k: torch.tensor(v) if not isinstance(v, torch.Tensor) and v is not None else v for k, v in batch.items()}
                batch = {k: v if v is None else v.unsqueeze(0) for k, v in batch.items()}
                data, gt = batch["observed"], batch["pose"]
                mode = "gif"

            elif mode == 'gif':
                batch = next(iter(data_loader))
                data, gt= batch["observed"], batch["pose"]

  

            # gt
            gt = gt[0].view((cfg.t_his + cfg.t_pred), cfg.joint_n, 3).numpy()
            data = data[0].view((cfg.t_his + cfg.t_pred), cfg.joint_n, 3).numpy()


            if draw_order_indicator == -1:
                poses['context'] = data
                poses['gt'] = gt
            else:
                poses[f'mmPred_{draw_order_indicator + 1}'] = gt
                poses[f'mmPred_{draw_order_indicator + 2}'] = gt
                


            n = cfg.vis_col if n_pre == -1 else n_pre
            mode_dict= sample_preprocessing(batch, K=n, cfg=cfg, n = 0)
            sampled_motion = diffusion.sample_ddim(model_select,
                                                   mode_dict)
                                                   




            traj_est = cfg.input_detransform(cfg, sampled_motion, mode_dict["noise"]["root"])[:,0,:,:]

            traj_est = traj_est.cpu().numpy()
            traj_est = post_process(traj_est, cfg)

            if k == 0:
                for j in range(traj_est.shape[0]):
                    poses[f'mmPred_{j}'] = traj_est[j]
            else:
                for j in range(traj_est.shape[0]):
                    poses[f'mmPred_{j + draw_order_indicator + 2 + 1}'] = traj_est[j]

            if draw_order_indicator == -1:
                draw_order_indicator = j
            else:
                draw_order_indicator = j + draw_order_indicator + 2 + 1
        yield poses

def pose_generator_nondiff(data_loader, model_select, cfg, mode=None,
                   action=None, nrow=1, dct_transform=None, dct_detransform = None):
    """
    stack k rows examples in one gif

    The logic of 'draw_order_indicator' is to cheat the render_animation(),
    because this render function only identify the first two as context and gt, which is a bit tricky to modify.
    """
    traj_np = None
    j = None
    model_select.eval()
    while True:
        poses = {}
        draw_order_indicator = -1
        for k in range(0, nrow):
            if mode == 'switch':
                data = data_set.sample_all_action()
            elif mode == 'test':
                batch = data_loader # this dataloader is a data
                data, gt, mask, timepoints = batch["observed"], batch["pose"], batch["mask"], batch["timepoints"]
                data = torch.from_numpy(data)
                gt = torch.from_numpy(gt)
                data = data.view(-1, (cfg.t_his + cfg.t_pred), cfg.joint_n, 3)
                gt = gt.view(-1, (cfg.t_his + cfg.t_pred), cfg.joint_n, 3)
                mode = "gif"

            elif mode == 'gif' or 'fix' in mode:
                batch = next(iter(data_loader))
                data, gt, mask, timepoints = batch["observed"], batch["pose"], batch["mask"], batch["timepoints"]

                data = data.view(-1, (cfg.t_his + cfg.t_pred), cfg.joint_n, 3)
                gt = gt.view(-1, (cfg.t_his + cfg.t_pred), cfg.joint_n, 3)

            elif mode == 'zero_shot':
                data = data_set[np.random.randint(0, data_set.shape[0])].copy()
                data = np.expand_dims(data, axis=0)
            else:
                raise NotImplementedError(f"unknown pose generator mode: {mode}")
            
            # # modification of x, y, z -> y, x, z for occ dataloader
            # data_temp = copy.deepcopy(data)
            # data_temp[:, :, :, 0] = data[:, :, :, 2]
            # data_temp[:, :, :, 2] = data[:, :, :, 0]
            # data = data_temp
            # del data_temp

            # gt
            gt = copy.deepcopy(gt[0])
            data = copy.deepcopy(data[0])
            # gt[:, :1, :] = 0
            # data[:, :, :1, :] = 0

            if mode == 'switch':
                poses = {}
                traj_np = data[..., 1:, :].reshape([data.shape[0], cfg.t_his + cfg.t_pred, -1])
            elif mode == 'pred' or mode == 'gif' or 'fix' in mode or mode == 'zero_shot':
                if draw_order_indicator == -1:
                    poses['context'] = data
                    poses['gt'] = gt
                else:
                    poses[f'mmPred_{draw_order_indicator + 1}'] = gt
                    poses[f'mmPred_{draw_order_indicator + 2}'] = gt
                gt = np.expand_dims(gt, axis=0)
                data = np.expand_dims(data, axis=0)
                traj_np = data[..., 1:, :].reshape([data.shape[0], cfg.t_his + cfg.t_pred, -1])

            traj = tensor(traj_np, device=cfg.device, dtype=cfg.dtype)

            mode_dict, traj_dct, traj_dct_mod, root = sample_preprocessing_nondiff(traj, cfg, mode=mode, mask=None, dct_transform=dct_transform)
            sampled_motion = model_select(traj_dct_mod)

            # if cfg.dct_enable:
            #     traj_est = torch.matmul(cfg.idct_m_all[:, :cfg.n_pre], sampled_motion)
            # else: 
            #     if cfg.vel_enable:
            #         # traj_est = vel_to_cart(sampled_motion, traj_dct_mod[:,[0],:])
            #         traj_est = sampled_motion / 10
            #     else:
            #         traj_est = sampled_motion
            traj_est = dct_detransform(cfg, sampled_motion, root)

            traj_est = traj_est.detach().cpu().numpy()
            traj_est = post_process(traj_est, cfg)

            if k == 0:
                for j in range(traj_est.shape[0]):
                    poses[f'mmPred_{j}'] = traj_est[j]
            else:
                for j in range(traj_est.shape[0]):
                    poses[f'mmPred_{j + draw_order_indicator + 2 + 1}'] = traj_est[j]

            if draw_order_indicator == -1:
                draw_order_indicator = j
            else:
                draw_order_indicator = j + draw_order_indicator + 2 + 1

        yield poses
