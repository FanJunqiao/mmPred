import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt

import torch
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
torch.cuda.set_device(0)

from utils import create_logger, seed_set
from utils.script import *
from utils.training import Trainer
from utils.training_regress import Trainer as Trainer_regress
from utils.training_non_diff import Trainer_nondiff
from utils.input_transform import *

from utils.evaluation import compute_stats, compute_stats_nondiff
from config import Config, update_config
from utils.demo_visualize import demo_visualize
from tensorboardX import SummaryWriter

# stage 1
from Point_models.mmwave_point_transformer_origin import PointTransformerReg
from Point_models.p4Transformer_encode import P4Transformer

# stage 2


from models.humanMAC_transformer import MotionTransformer

from models.PSGCN.model import GCN

from models.mamba_simple_kalman import ModelArgs ### no robust config
from models.MambaTransendBase_trans_time_mmfi import Mamba ###  test implementation





def setup():
    parser = argparse.ArgumentParser(description='Arguments for running the scripts')
    
    # Common args
    parser.add_argument('--cfg',
                        default='mmfi', help='h36m or humaneva')
    parser.add_argument('--mode', default='test_mmBody', help='train / test / pred / switch/ control/ zero_shot')
    parser.add_argument('--model_type', default='transformer', help='mamba / transformer')
    parser.add_argument('--finetune', type=bool, default=False)
    parser.add_argument('--exp_name', default='exp')
    parser.add_argument('--diffusion', type=bool, default=True)
    parser.add_argument('--iter', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str,
                        default=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    parser.add_argument('--multimodal_threshold', type=float, default=0.5)
    parser.add_argument('--multimodal_th_high', type=float, default=0.1)
    parser.add_argument('--milestone', type=list, default=[75, 150, 225, 275, 350, 450])
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--save_model_interval', type=int, default=50)
    parser.add_argument('--save_gif_interval', type=int, default=10)
    parser.add_argument('--save_metrics_interval', type=int, default=50)
    parser.add_argument('--ckpt', type=str, default='../ckpt/ckpt_ema_mmfi.pt')
    parser.add_argument('--ema', type=bool, default=True)
    parser.add_argument('--vis_switch_num', type=int, default=10)
    parser.add_argument('--vis_col', type=int, default=5)
    parser.add_argument('--vis_row', type=int, default=3)
    parser.add_argument('--train_stage', type=int, default=2)
    

    # DePos dataset args
    parser.add_argument("--joint_n", type=int, default=17)
    parser.add_argument("--miss_rate", type=int, default=20)
    parser.add_argument("--miss_scale", type=int, default=50)
    parser.add_argument('--miss_type_train', type=str, default='pred',
                        help='Choose the missing type of input sequence')
    parser.add_argument('--miss_type_test', type=list, default=["diff_pred"], # 'random_joints', 'random_left_arm_right_leg', 'structured_frame', 'random_frame' 'no_miss',"diff_pred", "pred"
                    help='Choose the missing type of input sequence')
    parser.add_argument("--skip_rate_train", type=int, default=1)
    parser.add_argument("--skip_rate_val", type=int, default=25)
    parser.add_argument('--data_all', type=str, default='all', choices=['one', 'all'],
                    help='Choose to train on one subject or all')
    parser.add_argument('--data_dir', type=str, default='./data/')
    parser.add_argument('--output_dir', type=str, default='default')

    parser.add_argument("--alpha_kalman", type=int, default=5)
    parser.add_argument("--alpha_kalman_enable", type=bool, default=False)
    parser.add_argument("--multiFlag", type=bool, default=False)

    args = parser.parse_args()

    """seed setup"""
    seed_set(args.seed)

    """config setup"""
    if args.train_stage == 1:
        args.cfg = args.cfg + "_stage_1"
    cfg = Config(f'{args.cfg}', test=(args.mode[:5] != 'train'))
    cfg = update_config(cfg, vars(args))


    return args, cfg

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device: %s' % device)

    args, cfg = setup()


    if cfg.train_stage == 2:
        """logger"""
        tb_logger = SummaryWriter(cfg.tb_dir)
        logger = create_logger(os.path.join(cfg.log_dir, 'log.txt'))
        print(f"Create logger: {os.path.join(cfg.log_dir, 'log.txt')}")
        display_exp_setting(logger, cfg)



        """create model"""
        if args.diffusion == True:
            diffusion = create_diffusion(cfg)

            expand = 1
            mambaArgs = ModelArgs(d_model= 512// expand,
                                num_T_in_M_layers=cfg.num_T_in_M_layers,
                                num_T_in_M_head=cfg.num_T_in_M_head,
                                n_layer = 4, 
                                temp_emb_dim= cfg.latent_dims // expand, 
                                vocab_length=cfg.n_pre, 
                                cond_length=cfg.n_pre_cond,
                                vocab_size= 3 * cfg.joint_num,
                                d_state=16,
                                cfg=cfg,
                                dropout=cfg.dropout, 
                                expand=expand)
            model = Mamba(mambaArgs).to(cfg.device)
        else:
            upJ = np.array([8,9,10,11,12,13,14,15])
            downJ = np.array([0,1,2,3,4,5,6])
            dim_up = np.concatenate((upJ * 3, upJ * 3 + 1, upJ * 3 + 2))
            dim_down = np.concatenate((downJ * 3, downJ * 3 + 1, downJ * 3 + 2))
            n_up = dim_up.shape[0]
            n_down = dim_down.shape[0]
            part_sep = (dim_up, dim_down, n_up, n_down)
            model = GCN(in_d=cfg.n_pre, hid_d=256, p_dropout=0.5,
                            num_stage=4, node_n=cfg.joint_num * 3, J=1, part_sep=part_sep,
                            W_pg=0.6, W_p=0.6).to(cfg.device)

        

        """create dataloader"""
        # prepare full evaluation dataset
        if args.mode.split("_")[1] == 'mmfi':
            train_loader, val_loader, test_dataset_list = set_dataloader_mmfi(cfg)
        if args.mode.split("_")[1] == 'mmBody':
            train_loader, val_loader, test_dataset_list = set_dataloader_mmBody(cfg)

        if args.mode.split("_")[0] == "test":
            print("Running ", args.mode)
            print(f"csv saving to {cfg.result_dir}")
            ckpt = torch.load(args.ckpt)
            model.load_state_dict(ckpt)
            model.eval()
            print("Model complete.")

            if args.diffusion == True:
                np.random.seed(5)
                plot_indexes = np.random.choice(range(0,3024), size=100, replace=False)
                demo_visualize(args.mode, cfg, model, diffusion, test_dataset_list, plot_indexes=plot_indexes) #818(wave), 368(raise), 2087(throw), 1145(dun), 1648(bend) 

                compute_stats(diffusion, test_dataset_list, model, logger, cfg, multiFlag=args.multiFlag)
            else:
                # test_multimodal_loader_list = get_multimodal_gt_full(logger, test_dataset_list, args, cfg, multiFlag=args.multiFlag)
                compute_stats_nondiff(test_dataset_list, model, logger, cfg,)

        elif args.mode.split("_")[0] == "train":
            if args.diffusion == True:
                trainer = Trainer(
                    model=model,
                    diffusion=diffusion,
                    dataloaders=[train_loader, val_loader],
                    test_dataset_list = test_dataset_list,
                    cfg=cfg,
                    args = args,
                    logger=logger,
                    tb_logger=tb_logger)
                trainer.loop()
            else:
                trainer = Trainer_nondiff(
                    model=model,
                    dataloaders=[train_loader, val_loader],
                    test_dataset_list = test_dataset_list,
                    cfg=cfg,
                    args = args,
                    logger=logger,
                    tb_logger=tb_logger)
                trainer.loop()

                    
    elif cfg.train_stage == 1:
        """logger"""
        tb_logger = SummaryWriter(cfg.tb_dir)
        logger = create_logger(os.path.join(cfg.log_dir, 'log.txt'))
        print(f"Create logger: {os.path.join(cfg.log_dir, 'log.txt')}")
        display_exp_setting(logger, cfg)

        

        """create dataloader"""
        if args.mode.split("_")[1] == 'mmfi':
            train_loader, val_loader, test_dataset_list = set_dataloader_mmfi(cfg)
            model = PointTransformerReg(
                input_dim = 5,
                nblocks = 5,
                n_p = 16,
                n_pre = cfg.n_pre
            ).to(cfg.device)
        if args.mode.split("_")[1] == 'mmBody':
            train_loader, val_loader, test_dataset_list = set_dataloader_mmBody(cfg)
            model = P4Transformer(radius=0.1, nsamples=32, spatial_stride=32,
                    temporal_kernel_size=3, temporal_stride=2,
                    emb_relu=False,
                    dim=1024, depth=10, heads=8, dim_head=256,
                    mlp_dim=2048, num_classes=17*3, dropout1=0.2, dropout2=0.2, dct_co=cfg.n_pre).to(cfg.device)

        if args.mode.split("_")[0] == "train":

            trainer = Trainer_regress(
                model=model,
                diffusion=None,
                dataloaders=[train_loader, val_loader],
                test_dataset_list = test_dataset_list,
                cfg=cfg,
                args = args,
                logger=logger,
                tb_logger=tb_logger)
            trainer.loop()
        else:
            trainer = Trainer_regress(
                model=model,
                diffusion=None,
                dataloaders=[train_loader, val_loader],
                test_dataset_list = test_dataset_list,
                cfg=cfg,
                args = args,
                logger=logger,
                tb_logger=tb_logger,
                is_test=True,
                test_plot=False)
            trainer.loop()




        