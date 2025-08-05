import yaml
import os
from utils.input_transform import *


from utils import util, torch, generate_pad


def get_log_dir_index(out_dir):
    dirs = [x[0] for x in os.listdir(out_dir)]
    if '.' in dirs:  # minor change for .ipynb
        dirs.remove('.')
    log_dir_index = '_' + str(len(dirs))

    return log_dir_index


def update_config(cfg, args_dict):
    """
    update some configuration related to args
        - merge args to cfg
        - dct, idct matrix
        - save path dir
    """
    for k, v in args_dict.items():
        setattr(cfg, k, v)

    cfg.t_pred_duplicate = cfg.t_pred 
    if cfg.train_stage == 1:
        cfg.t_pred = 0
    cfg.idx_pad, cfg.zero_index = generate_pad(cfg.padding, cfg.t_his, cfg.t_pred)
    print(cfg.t_pred , cfg.t_pred_duplicate)

    dtype = torch.float32
    torch.set_default_dtype(dtype)
    cfg.dtype = dtype

    cfg.dct_m, cfg.idct_m = util.get_dct_matrix(cfg.t_pred + cfg.t_his)
    cfg.dct_m_all = cfg.dct_m.float().to(cfg.device)
    cfg.idct_m_all = cfg.idct_m.float().to(cfg.device)

    # dev 1
    cfg.dct_m1, cfg.idct_m1, cfg.aug_m1 = util.get_dct_dev1_matrix(cfg.t_pred + cfg.t_his)
    cfg.dct_m_all1 = cfg.dct_m1.float().to(cfg.device)
    cfg.idct_m_all1 = cfg.idct_m1.float().to(cfg.device)
    cfg.aug_m1 = cfg.aug_m1.float().to(cfg.device)

    # dev 2
    cfg.dct_m2, cfg.idct_m2 = util.get_dct_dev2_matrix(cfg.t_pred + cfg.t_his)
    cfg.dct_m_all2 = cfg.dct_m2.float().to(cfg.device)
    cfg.idct_m_all2 = cfg.idct_m2.float().to(cfg.device)

    index = get_log_dir_index(cfg.base_dir)
    if args_dict['mode'] == ('train' or 'pred' or 'eval'):
        cfg.cfg_dir = '%s/%s' % (cfg.base_dir, args_dict['cfg']+ "_" + cfg.exp_name + index)
    else:
        cfg.cfg_dir = '%s/%s' % (cfg.base_dir, args_dict['mode']+ "_" + cfg.exp_name + index)
    os.makedirs(cfg.cfg_dir, exist_ok=True)
    cfg.model_dir = '%s/models' % cfg.cfg_dir
    cfg.result_dir = '%s/results' % cfg.cfg_dir
    cfg.log_dir = '%s/log' % cfg.cfg_dir
    cfg.tb_dir = '%s/tb' % cfg.cfg_dir
    cfg.gif_dir = '%s/out' % cfg.cfg_dir
    os.makedirs(cfg.model_dir, exist_ok=True)
    os.makedirs(cfg.result_dir, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)
    os.makedirs(cfg.tb_dir, exist_ok=True)
    os.makedirs(cfg.gif_dir, exist_ok=True)
    cfg.model_path = os.path.join(cfg.model_dir)
    cfg.joint_num = cfg.joint_n - 1
    cfg.data_dir = cfg.data_dir + f"data_{cfg.dataset}"
    
        

    return cfg


class Config:

    def __init__(self, cfg_id, test=False):
        self.id = cfg_id
        cfg_name = './cfg/%s.yml' % cfg_id
        if not os.path.exists(cfg_name):
            print("Config file doesn't exist: %s" % cfg_name)
            exit(0)
        cfg = yaml.safe_load(open(cfg_name, 'r'))

        # create dirs
        self.base_dir = 'inference' if test else 'experiments'
        os.makedirs(self.base_dir, exist_ok=True)

        # common
        self.dataset = cfg.get('dataset', 'h36m')
        self.batch_size = cfg['batch_size']
        self.normalize_data = cfg.get('normalize_data', False)
        self.t_his = cfg['t_his']
        self.t_pred = cfg['t_pred']

        self.num_epoch = cfg['num_epoch']
        self.num_data_sample = cfg['num_data_sample']
        self.num_val_data_sample = cfg['num_val_data_sample']
        self.lr = cfg['lr']

        self.n_pre = cfg['n_pre']
        self.n_pre_cond = cfg['n_pre_cond']
        self.multimodal_path = cfg['multimodal_path']
        self.data_candi_path = cfg['data_candi_path']

        self.padding = cfg['padding']
        self.Complete = cfg['Complete']
        self.noise_steps = cfg['noise_steps']
        self.ddim_timesteps = cfg['ddim_timesteps']
        self.refine_timesteps = cfg['refine_timesteps']
        self.scheduler = cfg['scheduler']

        self.num_layers = cfg['num_layers']
        self.num_F_in_J_layers = cfg['num_F_in_J_layers']
        self.num_F_in_J_head = cfg['num_F_in_J_head']
        self.latent_dims = cfg['latent_dims']
        self.dropout = cfg['dropout']
        self.num_heads = cfg['num_heads']

        self.mod_train = cfg['mod_train']
        self.mod_test = cfg['mod_test']

        self.dct_norm_enable = cfg['dct_norm_enable']
        self.dct_enable = cfg['dct_enable']
        self.dct_dev1_enable = cfg['dct_dev1_enable']
        self.inner_dct_enable = cfg['inner_dct_enable']
        self.diff_err_loss = cfg["diff_err_loss"]
        self.diff_err_mask = cfg["diff_err_mask"]
        self.vel_enable = cfg["vel_enable"]
        
        self.dct_i = cfg["dct_i"] if self.dataset == 'mmBody' else 3

        # indirect variable
        self.joint_num = 16 if self.dataset == 'h36m' else 14
        self.idx_pad, self.zero_index = generate_pad(self.padding, self.t_his, self.t_pred)
        self.frame_rate = 10 if self.dataset == "mmfi" else 16
        
        self.return_attn = False

        if self.dct_enable == True:
            self.input_transform = dct_transform
            self.input_detransform = dct_detransform
        elif self.dct_dev1_enable == True:
            self.input_transform = dct_dev1_transform
            self.input_detransform = dct_dev1_detransform
        elif self.vel_enable == True:
            self.input_transform = vel_transform
            self.input_detransform = vel_detransform
        else:
            self.input_transform = null_transform
            self.input_detransform = null_detransform
