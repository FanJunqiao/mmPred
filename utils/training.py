import copy
import time

from torch import optim, nn

from utils.visualization import render_animation
from utils.ema import EMA
from utils import *
from utils.evaluation import compute_stats
from utils.pose_gen import pose_generator
from tqdm import tqdm
from utils.script import *
from utils.input_transform import *


class Trainer:
    def __init__(self,
                 model,
                 diffusion,
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
        self.diffusion = diffusion
        self.train_loader, self.val_loader = dataloaders[0], dataloaders[1]
        self.test_dataset_list = test_dataset_list
        self.cfg = cfg
        self.args = args
        self.multiFlag = args.multiFlag
        self.logger = logger
        self.tb_logger = tb_logger

        self.iter = 0

        self.lrs = []

        if self.cfg.ema is True:
            self.ema = EMA(0.995)
            self.ema_model = copy.deepcopy(model).eval().requires_grad_(False)
            self.ema_setup = (self.cfg.ema, self.ema, self.ema_model)
        else:
            self.ema_model = None
            self.ema_setup = None


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
        self.train_losses_aux = AverageMeter()
        self.logger.info(f"Starting training epoch {self.iter}:")

    def run_train_step(self):

        for batch in tqdm(self.train_loader, total=len(self.train_loader), desc='Train round', unit='batch', leave=False):

            mode_dict_train = process_batch(batch, self.cfg)
            t = self.diffusion.sample_timesteps(mode_dict_train["gt"]["all_dct"].shape[0]).to(self.cfg.device)
            x_t, noise = self.diffusion.noise_motion(mode_dict_train["noise"]["all_dct"],mode_dict_train["gt"]["all_dct"], t, is_train = True)

            if np.random.random() > self.cfg.mod_train:
                mode_dict_train["noise"]["pad_dct"] = None
            
            

            predicted_noise, loss_aux = self.model(x_t, t, 
                                                mod=mode_dict_train["noise"]["pad_dct"], 
                                                feat_time = mode_dict_train["others"]["feat"], 
                                                radar_time = mode_dict_train["others"]["radar"],
                                                limb_len = mode_dict_train["others"]["limb_len"],
                                                mod_time = mode_dict_train["noise"]["pad_time"][:, :self.cfg.t_his], 
                                                gt_time = mode_dict_train["gt"]["pad_time"][:, :self.cfg.t_his],
                                                motion_pred = mode_dict_train["others"]["motion_pred"],
                                                motion_feat = mode_dict_train["others"]["motion_feat"],
                                                pose_var = mode_dict_train["others"]["observed_var"][:, :self.cfg.t_his])

            
            loss = self.criterion(predicted_noise, noise) 
            if loss_aux!=None:
                loss_aux = loss_aux
                loss_total = loss + loss_aux
            else:
                loss_total = loss
                loss_aux = torch.zeros_like(loss)


            self.optimizer.zero_grad()
            loss_total.backward()
            self.optimizer.step()

            args_ema, ema, ema_model = self.ema_setup[0], self.ema_setup[1], self.ema_setup[2]

            if args_ema is True:
                ema.step_ema(ema_model, self.model)

            self.train_losses.update(loss.item())
            self.train_losses_aux.update(loss_aux.item())
            self.tb_logger.add_scalar('Loss/train', loss.item(), self.iter)
            self.tb_logger.add_scalar('Loss/train_aux', loss_aux.item(), self.iter)


    def after_train_step(self):
        self.lr_scheduler.step()
        self.lrs.append(self.optimizer.param_groups[0]['lr'])
        self.logger.info(
            '====> Epoch: {} Time: {:.2f} Train Loss: {} Train Loss Aux: {} lr: {:.5f}'.format(self.iter,
                                                                            time.time() - self.t_s,
                                                                            self.train_losses.avg, self.train_losses_aux.avg,
                                                                            self.lrs[-1]))
        if self.iter % self.cfg.save_gif_interval == 0:
            pose_gen = pose_generator(self.train_loader, self.model, self.diffusion, self.cfg, mode='gif')
            render_animation(self.cfg, pose_gen, ['mmPred'], self.cfg.t_his, ncol=4,
                             output=os.path.join(self.cfg.gif_dir, f'training_{self.iter}.gif'))

    def before_val_step(self):
        self.model.eval()
        self.t_s = time.time()
        self.val_losses = AverageMeter()
        self.val_losses_aux = AverageMeter()
        # self.generator_val = self.dataset['test'].sampling_generator(num_samples=self.cfg.num_val_data_sample,
        #                                                              batch_size=self.cfg.batch_size)
        self.logger.info(f"Starting val epoch {self.iter}:")

    def run_val_step(self):
        for batch in tqdm(self.val_loader, total=len(self.val_loader), desc='Validation round', unit='batch', leave=False):
            with torch.no_grad():
                # (N, t_his + t_pre, joints, 3) -> (N, t_his + t_pre, 3 * (joints - 1))
                # discard the root joint and combine xyz coordinate
                mode_dict_val = process_batch(batch, self.cfg)
                t = self.diffusion.sample_timesteps(mode_dict_val["gt"]["all_dct"].shape[0]).to(self.cfg.device)
                x_t, noise = self.diffusion.noise_motion(mode_dict_val["noise"]["all_dct"], mode_dict_val["gt"]["all_dct"], t, is_train = True)
                
                if np.random.random() > self.cfg.mod_train:
                    mode_dict_val["noise"]["pad_dct"] = None

                # from plot import plot_pose_sequence_as_gif
                # plot_pose_sequence_as_gif(mode_dict_val["gt"]["all_time"], mode_dict_val["others"]["motion_pred"])

                predicted_noise, loss_aux = self.model(x_t, t, 
                                                   mod=mode_dict_val["noise"]["pad_dct"], 
                                                   feat_time = mode_dict_val["others"]["feat"], 
                                                   radar_time = mode_dict_val["others"]["radar"],
                                                   limb_len = mode_dict_val["others"]["limb_len"],
                                                   mod_time = mode_dict_val["noise"]["pad_time"][:, :self.cfg.t_his], 
                                                   gt_time = mode_dict_val["gt"]["pad_time"][:, :self.cfg.t_his],
                                                   motion_pred = mode_dict_val["others"]["motion_pred"],
                                                   motion_feat = mode_dict_val["others"]["motion_feat"],
                                                   pose_var = mode_dict_val["others"]["observed_var"][:, :self.cfg.t_his])


                loss = self.criterion(predicted_noise, noise) 
                if loss_aux!=None:
                    loss_aux = loss_aux
                    loss = loss + loss_aux 
                else:
                    loss_total = loss
                    loss_aux = torch.zeros_like(loss)

                self.val_losses.update(loss.item())
                self.tb_logger.add_scalar('Loss/val', loss.item(), self.iter)
                self.val_losses_aux.update(loss_aux.item())
                self.tb_logger.add_scalar('Loss/val_aux', loss_aux.item(), self.iter)


    def after_val_step(self):
        self.logger.info('====> Epoch: {} Time: {:.2f} Val Loss: {} Val Loss Aux:{}'.format(self.iter,
                                                                            time.time() - self.t_s,
                                                                            self.val_losses.avg, self.val_losses_aux.avg))

        if self.iter % self.cfg.save_gif_interval == 0:
            if self.cfg.ema is True:
                pose_gen = pose_generator(self.val_loader, self.ema_model, self.diffusion, self.cfg, mode='gif')
            else:
                pose_gen = pose_generator(self.val_loader, self.model, self.diffusion, self.cfg, mode='gif')
            render_animation(self.cfg, pose_gen, ['mmPred'], self.cfg.t_his, ncol=4,
                             output=os.path.join(self.cfg.gif_dir, f'val_{self.iter}.gif'))

        if self.iter % self.cfg.save_metrics_interval == 0 and self.iter != 0:
            if self.cfg.ema is True:
                compute_stats(self.diffusion, self.test_dataset_list, self.ema_model, self.logger, self.cfg, mask=None, multiFlag=self.multiFlag)
            else:
                compute_stats(self.diffusion, self.test_dataset_list, self.model, self.logger, self.cfg, mask=None, multiFlag=self.multiFlag)
        if self.cfg.save_model_interval > 0 and (self.iter + 1) % self.cfg.save_model_interval == 0:
            if self.cfg.ema is True:
                torch.save(self.ema_model.state_dict(),
                           os.path.join(self.cfg.model_path, f"ckpt_ema_{self.iter + 1}.pt"))
            else:
                torch.save(self.model.state_dict(), os.path.join(self.cfg.model_path, f"ckpt_{self.iter + 1}.pt"))
