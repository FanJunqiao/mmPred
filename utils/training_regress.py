import copy
import time

from torch import optim, nn

from utils.visualization import render_animation
from utils.ema import EMA
from utils import *
from utils.logger import *
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
                 tb_logger, 
                 is_test = False,
                 test_plot = False):
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
        self.is_test = is_test
        self.dataset = cfg.dataset
        self.test_plot = test_plot
        
        self.ckpt_path = "./ckpt/"
        self.data_path = "./data/"

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
        if self.is_test:
            self.before_test_step()
            self.run_test_step()
        else:
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
        self.t_s = time.time()
        self.train_losses = AverageMeter()
        self.train_losses_aux = AverageMeter()
        self.logger.info(f"Starting training epoch {self.iter}:")

    def run_train_step(self):

        for batch in tqdm(self.train_loader, total=len(self.train_loader), desc='Train round', unit='batch', leave=False):

            mode_dict_train = process_batch(batch, self.cfg)
            
            predicted = self.model(radar = mode_dict_train["others"]["radar"])
            B, J, C = predicted.shape
            predicted = predicted.view(B, J, self.cfg.n_pre, C//self.cfg.n_pre).permute(0,2,1,3).reshape(B, self.cfg.n_pre, J*(C//self.cfg.n_pre))


            loss = self.criterion(predicted, mode_dict_train["gt"]["all_dct"]) * 100 
            
            loss_aux = None
            if loss_aux!=None:
                loss_aux = loss_aux * 10.0
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
            batch = next(iter(self.train_loader))
            mode_dict = process_batch(batch, self.cfg)
            radar = mode_dict["others"]["radar"]
            gt_all_time = mode_dict["gt"]["all_time"]
            print(f"Radar shape: {radar.shape}, GT all_time shape: {gt_all_time.shape}")

    def before_val_step(self):
        self.model.eval()
        self.t_s = time.time()
        self.val_losses = AverageMeter()
        self.val_losses_aux = AverageMeter()

        self.logger.info(f"Starting val epoch {self.iter}:")

    def run_val_step(self):
        i = 0
        for batch in tqdm(self.val_loader, total=len(self.val_loader), desc='Validation round', unit='batch', leave=False):
            i += 1
            with torch.no_grad():

                mode_dict_val = process_batch(batch, self.cfg)
                
                if np.random.random() > self.cfg.mod_train:
                    mode_dict_val["noise"]["pad_dct"] = None

                print(mode_dict_val["others"]["radar"].shape,  mode_dict_val["gt"]["all_dct"].shape)
                predicted = self.model(radar = mode_dict_val["others"]["radar"])

                B, J, C = predicted.shape
                predicted = predicted.view(B, J, self.cfg.n_pre, C//self.cfg.n_pre).permute(0,2,1,3).reshape(B, self.cfg.n_pre, J*(C//self.cfg.n_pre))
                loss = self.criterion(predicted, mode_dict_val["gt"]["all_dct"]) * 100

                if i % 100 ==0:
                    from plot_regress import visualize_radar_gt_pred_pose
                    visualize_radar_gt_pred_pose(radar=mode_dict_val["others"]["radar"], gt_dct= mode_dict_val["gt"]["all_dct"], pose_pr=predicted)
                
                loss_aux = None
                if loss_aux!=None:
                    loss_aux = loss_aux * 10.0
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

            # Print radar and all_time shapes similar to train
            batch = next(iter(self.val_loader))
            mode_dict = process_batch(batch, self.cfg)
            radar = mode_dict["others"]["radar"]
            gt_all_time = mode_dict["gt"]["all_dct"]
            print(f"Radar shape: {radar.shape}, GT all_time shape: {gt_all_time.shape}")
            

        # Save model only if current val loss is the smallest so far
        if not hasattr(self, 'best_val_loss'):
            self.best_val_loss = float('inf')
        if self.val_losses.avg < self.best_val_loss:
            self.best_val_loss = self.val_losses.avg
            if self.cfg.ema is True:
                torch.save(self.ema_model.state_dict(),
                    os.path.join(self.ckpt_path, f"ckpt_ema_best_{self.dataset}.pt"))
            else:
                torch.save(self.model.state_dict(), os.path.join(self.ckpt_path, f"ckpt_best.pt"))



    def before_test_step(self):

        ckpt_path = os.path.join(self.ckpt_path, f"ckpt_ema_best_{self.dataset}.pt")
            
        
        if self.cfg.ema is True and os.path.exists(ckpt_path):
            self.ema_model.load_state_dict(torch.load(ckpt_path, map_location=self.cfg.device))
            self.model.load_state_dict(self.ema_model.state_dict())
        elif os.path.exists(ckpt_path):
            self.model.load_state_dict(torch.load(ckpt_path, map_location=self.cfg.device))
        else:
            self.logger.error(f"Checkpoint {ckpt_path} does not exist. Please check the path.")
            raise FileNotFoundError(f"Checkpoint {ckpt_path} does not exist.")
        print("model loaded")
        self.model.eval()
        
        self.t_s = time.time()
        self.val_losses = AverageMeter()
        self.val_losses_aux = AverageMeter()

        self.logger.info(f"Starting teset epoch {self.iter}:")
        self.criterion = nn.MSELoss()

    def run_test_step(self):
        i = 0
        def reset_dataloader_without_shuffle(dataloader):
            return torch.utils.data.DataLoader(
                dataloader.dataset,
                batch_size=dataloader.batch_size,
                num_workers=dataloader.num_workers,
                pin_memory=getattr(dataloader, 'pin_memory', False),
                drop_last=getattr(dataloader, 'drop_last', False),
                collate_fn=getattr(dataloader, 'collate_fn', None),
                shuffle=False
            )
        self.train_loader_print = reset_dataloader_without_shuffle(self.train_loader)
        self.val_loader_print = reset_dataloader_without_shuffle(self.val_loader)
        
        for loader_name in ["test", "train"]:
            all_pred = []
            all_feature = []

            if loader_name == "train":
                loader_select = self.train_loader_print
            else:
                loader_select = self.val_loader_print

            for batch in tqdm(loader_select, total=len(loader_select), desc='Validation round', unit='batch', leave=False):
                i += 1
                with torch.no_grad():
                    mode_dict_val = process_batch(batch, self.cfg)
                    predicted = self.model(radar = mode_dict_val["others"]["radar"])

                    B, J, C = predicted.shape
                    predicted = predicted.view(B, J, self.cfg.n_pre, C//self.cfg.n_pre).permute(0,2,1,3).reshape(B, self.cfg.n_pre, J*(C//self.cfg.n_pre))

                    # Register a forward hook to capture the intermediate output of dim_reduce_head
                    hook_name = "dim_reduce_head" if self.dataset == "mmBody" else "fc2"
                    features = {}
                    if self.dataset == "mmBody":
                        def hook_fn(module, input, output):
                            features["fc2"] = output
                        handle = self.model.dim_reduce_head.register_forward_hook(hook_fn)
                    elif self.dataset == "mmfi":
                        def hook_fn(module, input, output):
                            features["fc2"] = output
                        handle = self.model.fc2.register_forward_hook(hook_fn)
                    # handle = self.model.fc2.register_forward_hook(hook_fn)
                    _ = self.model(radar=mode_dict_val["others"]["radar"])
                    feature = features[hook_name]
                    handle.remove()

                    all_pred.append(predicted.cpu().numpy())
                    all_feature.append(feature.cpu().numpy())

                    if self.test_plot:
                        from plot_regress import visualize_radar_gt_pred_pose
                        visualize_radar_gt_pred_pose(radar=mode_dict_val["others"]["radar"][:,-self.cfg.t_his:], gt_dct= mode_dict_val["gt"]["all_dct"], pose_pr=predicted)

            # Concatenate all predictions and features
            all_pred = np.concatenate(all_pred, axis=0)
            all_feature = np.concatenate(all_feature, axis=0)
            
            # Ensure output directory exists
            save_data_dir = os.path.join(self.data_path, f"data_{self.dataset}", "stage_1_process/")
            os.makedirs(save_data_dir, exist_ok=True)

            # Save to npy files
            np.save(os.path.join(save_data_dir, f"all_pred_{loader_name}_{self.cfg.n_pre}.npy"), all_pred)
            np.save(os.path.join(save_data_dir, f"all_feat_{loader_name}_{self.cfg.n_pre}.npy"), all_feature)

            print("Saved all_pred.npy with shape:", all_pred.shape)
            print("Saved all_feature.npy with shape:", all_feature.shape)
