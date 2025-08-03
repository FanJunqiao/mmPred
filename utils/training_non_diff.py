import copy
import time

from torch import optim, nn

from utils.visualization import render_animation
from utils.ema import EMA
from utils import *
from utils.evaluation import compute_stats_nondiff
from utils.pose_gen import *
from tqdm import tqdm
from utils.script import *
from utils.input_transform import *


class Trainer_nondiff:
    def __init__(self,
                 model,
                 dataloaders,
                 test_dataset_list,
                 cfg,
                 args,
                 logger,
                 tb_logger):
        super().__init__()

        self.val_losses = None
        self.t_s = None
        self.train_losses = None

        self.criterion = None
        self.lr_scheduler = None
        self.optimizer = None

        self.model = model
        self.train_loader, self.val_loader = dataloaders[0], dataloaders[1]
        self.test_dataset_list = test_dataset_list
        self.cfg = cfg
        self.args = args
        self.multiFlag = args.multiFlag
        self.logger = logger
        self.tb_logger = tb_logger
        self.test_multimodal_loader_list = get_multimodal_gt_full(self.logger, self.test_dataset_list, self.args, self.cfg, multiFlag=self.multiFlag)

        self.iter = 0

        self.lrs = []

        if self.cfg.ema is True:
            self.ema = EMA(0.995)
            self.ema_model = copy.deepcopy(model).eval().requires_grad_(False)
            self.ema_setup = (self.cfg.ema, self.ema, self.ema_model)
        else:
            self.ema_model = None
            self.ema_setup = None
        
        if self.cfg.dct_enable == True:
            self.input_transform = dct_transform
            self.input_detransform = dct_detransform
        elif self.cfg.dct_dev1_enable == True:
            self.input_transform = dct_dev1_transform
            self.input_detransform = dct_dev1_detransform
        elif self.cfg.vel_enable == True:
            self.input_transform = vel_transform
            self.input_detransform = vel_detransform
        else:
            self.input_transform = null_transform
            self.input_detransform = null_detransform


    def loop(self):
        self.before_train()
        for self.iter in range(0, self.cfg.num_epoch):
            self.before_train_step()
            self.run_train_step()
            self.after_train_step()
            self.before_val_step()
            self.run_val_step()
            self.after_val_step()

    def before_train(self):
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.cfg.lr)
        self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.cfg.milestone,
                                                           gamma=self.cfg.gamma)
        self.criterion = nn.MSELoss()

    def before_train_step(self):
        self.model.train()
        # self.generator_train = self.dataset['train'].sampling_generator(num_samples=self.cfg.num_data_sample,
        #                                                                 batch_size=self.cfg.batch_size)
        self.t_s = time.time()
        self.train_losses = AverageMeter()
        self.logger.info(f"Starting training epoch {self.iter}:")

    def run_train_step(self):

        for batch in tqdm(self.train_loader, total=len(self.train_loader), desc='Train round', unit='batch', leave=False):
            with torch.no_grad():
                # (N, t_his + t_pre, joints, 3) -> (N, t_his + t_pre, 3 * (joints - 1))
                # discard the root joint and combine xyz coordinate
                observed, pose, mask, timepoints = batch["observed"], batch["pose"], batch["mask"], batch["timepoints"]
                traj_np = pose.view(-1, (self.cfg.t_his + self.cfg.t_pred), self.cfg.joint_n, 3)
                traj_np = traj_np[..., 1:, :].reshape([traj_np.shape[0], self.cfg.t_his + self.cfg.t_pred, -1])
                traj = traj_np.clone().to(torch.float).to(self.cfg.device) #tensor(traj_np, device=self.cfg.device, dtype=self.cfg.dtype)

                observed_np = observed.view(-1, (self.cfg.t_his + self.cfg.t_pred), self.cfg.joint_n, 3)
                observed_np = observed_np[..., 1:, :].reshape([observed_np.shape[0], self.cfg.t_his + self.cfg.t_pred, -1])
                observed_traj = observed_np.clone().to(torch.float).to(self.cfg.device) #tensor(traj_np, device=self.cfg.device, dtype=self.cfg.dtype)
                traj_pad = padding_traj(observed_traj, self.cfg.padding, self.cfg.idx_pad, self.cfg.zero_index)

                traj_dct, traj_dct_mod, root = self.input_transform(self.cfg, traj, traj_pad)
                

            # train
            predicted_traj_dct = self.model(traj_dct_mod)
            loss = self.criterion(traj_dct, predicted_traj_dct)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            args_ema, ema, ema_model = self.ema_setup[0], self.ema_setup[1], self.ema_setup[2]

            if args_ema is True:
                ema.step_ema(ema_model, self.model)

            self.train_losses.update(loss.item())
            self.tb_logger.add_scalar('Loss/train', loss.item(), self.iter)

            del loss, traj, traj_dct, traj_dct_mod, traj_pad, traj_np, observed_np, observed_traj

    def after_train_step(self):
        self.lr_scheduler.step()
        self.lrs.append(self.optimizer.param_groups[0]['lr'])
        self.logger.info(
            '====> Epoch: {} Time: {:.2f} Train Loss: {} lr: {:.5f}'.format(self.iter,
                                                                            time.time() - self.t_s,
                                                                            self.train_losses.avg,
                                                                            self.lrs[-1]))
        if self.iter % self.cfg.save_gif_interval == 0:
            pose_gen = pose_generator_nondiff(self.train_loader, self.model, self.cfg, mode='gif', dct_transform=self.input_transform, dct_detransform=self.input_detransform)
            render_animation(self.cfg, pose_gen, ['mmPred'], self.cfg.t_his, ncol=4,
                             output=os.path.join(self.cfg.gif_dir, f'training_{self.iter}.gif'))

    def before_val_step(self):
        self.model.eval()
        self.t_s = time.time()
        self.val_losses = AverageMeter()
        # self.generator_val = self.dataset['test'].sampling_generator(num_samples=self.cfg.num_val_data_sample,
        #                                                              batch_size=self.cfg.batch_size)
        self.logger.info(f"Starting val epoch {self.iter}:")

    def run_val_step(self):
        for batch in tqdm(self.val_loader, total=len(self.val_loader), desc='Validation round', unit='batch', leave=False):
            with torch.no_grad():
                # (N, t_his + t_pre, joints, 3) -> (N, t_his + t_pre, 3 * (joints - 1))
                # discard the root joint and combine xyz coordinate
                observed, pose, mask, timepoints = batch["observed"], batch["pose"], batch["mask"], batch["timepoints"]
                traj_np = pose.view(-1, (self.cfg.t_his + self.cfg.t_pred), self.cfg.joint_n, 3)
                traj_np = traj_np[..., 1:, :].reshape([traj_np.shape[0], self.cfg.t_his + self.cfg.t_pred, -1])
                traj = traj_np.clone().to(torch.float).to(self.cfg.device) #tensor(traj_np, device=self.cfg.device, dtype=self.cfg.dtype)

                observed_np = observed.view(-1, (self.cfg.t_his + self.cfg.t_pred), self.cfg.joint_n, 3)
                observed_np = observed_np[..., 1:, :].reshape([observed_np.shape[0], self.cfg.t_his + self.cfg.t_pred, -1])
                observed_traj = observed_np.clone().to(torch.float).to(self.cfg.device) #tensor(traj_np, device=self.cfg.device, dtype=self.cfg.dtype)
                traj_pad = padding_traj(observed_traj, self.cfg.padding, self.cfg.idx_pad, self.cfg.zero_index)

                traj_dct, traj_dct_mod, root = self.input_transform(self.cfg, traj, traj_pad)

                predicted_traj_dct = self.model(traj_dct_mod)
                loss = self.criterion(traj_dct, predicted_traj_dct)

                self.val_losses.update(loss.item())
                self.tb_logger.add_scalar('Loss/val', loss.item(), self.iter)

            del loss, traj, traj_dct, traj_dct_mod, traj_pad, traj_np, observed_np, observed_traj

    def after_val_step(self):
        self.logger.info('====> Epoch: {} Time: {:.2f} Val Loss: {}'.format(self.iter,
                                                                            time.time() - self.t_s,
                                                                            self.val_losses.avg))

        if self.iter % self.cfg.save_gif_interval == 0:
            if self.cfg.ema is True:
                pose_gen = pose_generator_nondiff(self.val_loader, self.ema_model, self.cfg, mode='gif', dct_transform=self.input_transform, dct_detransform=self.input_detransform)
            else:
                pose_gen = pose_generator_nondiff(self.val_loader, self.model, self.cfg, mode='gif', dct_transform=self.input_transform, dct_detransform=self.input_detransform)
            render_animation(self.cfg, pose_gen, ['mmPred'], self.cfg.t_his, ncol=4,
                             output=os.path.join(self.cfg.gif_dir, f'val_{self.iter}.gif'))

        if self.iter % self.cfg.save_metrics_interval == 0 and self.iter != 0:
            
            if self.cfg.ema is True:
                
                compute_stats_nondiff(self.test_multimodal_loader_list, self.ema_model, self.logger, self.cfg, mask=None, multiFlag=self.multiFlag, dct_transform=self.input_transform, dct_detransform=self.input_detransform)
            else:
                compute_stats_nondiff(self.test_multimodal_loader_list, self.model, self.logger, self.cfg, mask=None, multiFlag=self.multiFlag, dct_transform=self.input_transform, dct_detransform=self.input_detransform)
        if self.cfg.save_model_interval > 0 and (self.iter + 1) % self.cfg.save_model_interval == 0:
            if self.cfg.ema is True:
                torch.save(self.ema_model.state_dict(),
                           os.path.join(self.cfg.model_path, f"ckpt_ema_{self.iter + 1}.pt"))
            else:
                torch.save(self.model.state_dict(), os.path.join(self.cfg.model_path, f"ckpt_{self.iter + 1}.pt"))
