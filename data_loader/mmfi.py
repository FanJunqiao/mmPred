import os
import numpy as np
import torch
from torch.utils.data import Dataset


class mmfi(Dataset):

    def __init__(self, data_dir, input_n, output_n, skip_rate, actions=None, split=0, miss_rate=0.2, miss_scale=25,
                 miss_type='no_miss', all_data=False, joint_n=32, train_stage = 2, dct_i=3):
        """
        :param actions:
        :param input_n:
        :param output_n:
        :param dct_used:
        :param split: 0 train, 1 validation, 2 testing
        :param sample_rate:
        """
        self.data_dir = data_dir
        self.train_stage = train_stage
        splits = ["train", "test", "test"]
        self.split_num = split
        self.split = splits[split]
        self.in_n = input_n
        self.out_n = output_n
        self.skip_rate = skip_rate
        self.miss_rate = miss_rate
        self.miss_scale = miss_scale
        self.miss_type = miss_type
        self.mask_apply = True
        self.joint_n = joint_n
        self.sample_rate = 1
        self.all_data = all_data
        self.actions = actions
        self.p3d = {}
        self.params = {}
        self.masks = {}
        self.data_idx = []
        self.seq_len = self.in_n + self.out_n
        self.seq_len_before = int(self.seq_len*self.sample_rate)
        
        self.time_lag_before = 3
        self.dct_i = dct_i

        # for mmBody noise type
        if self.miss_type == "no_miss":
            self.joint_noise_type = "gt"
        else:
            self.joint_noise_type = self.miss_type

        self.data = None
        
        self.prepare_data()
        
        print(f"Generating {self.__len__()} samples...") 

    def prepare_data(self):

        self.data_file = os.path.join(self.data_dir, "gt" + '_poses_' + self.split +'.npz')
        self.data_noise_file = os.path.join(self.data_dir, self.joint_noise_type + '_poses_' + self.split +'.npz')
        self.radar_file = os.path.join(self.data_dir, 'radar_array_input_' + self.split +'_random.npz')
        
        if self.train_stage == 2:
            self.motion_feat_file = os.path.join(self.data_dir, "stage_1_process", f"all_feat_{self.split}_{self.dct_i}.npy")
            self.motion_pred_file = os.path.join(self.data_dir, "stage_1_process", f"all_pred_{self.split}_{self.dct_i}.npy")

        self.kept_joints = np.array([x for x in range(17)])
        self.process_data()  
        self.process_data_seq()

    def process_data(self):
        print("Process data")
        data = np.load(self.data_file, allow_pickle=True)
        data = {key: data[key] for key in data.keys()}

        self.data = {}
        for seq_id in data.keys():
            seq_id_keys = seq_id.split(",")
            scene = seq_id_keys[0]
            action = seq_id_keys[1]
            subject = seq_id_keys[2]
            if scene not in self.data.keys():
                self.data[scene] = {}
            if action not in self.data[scene].keys():
                self.data[scene][action] = {}
            self.data[scene][action][subject] = data[seq_id]

        data_noise = np.load(self.data_noise_file, allow_pickle=True)
        data_noise = {key: data_noise[key] for key in data_noise.keys()}

        self.data_noise = {}
        for seq_id in data_noise.keys():
            seq_id_keys = seq_id.split(",")
            scene = seq_id_keys[0]
            action = seq_id_keys[1]
            subject = seq_id_keys[2]
            
            if scene not in self.data_noise.keys():
                self.data_noise[scene] = {}
            if action not in self.data_noise[scene].keys():
                self.data_noise[scene][action] = {}
            self.data_noise[scene][action][subject] = data_noise[seq_id]
        

        data_radar = np.load(self.radar_file, allow_pickle=True)
        data_radar = {key: data_radar[key] for key in data_radar.keys()}

        self.data_radar = {}
        for seq_id in data_radar.keys():
            seq_id_keys = seq_id.split(",")
            scene = seq_id_keys[0]
            action = seq_id_keys[1]
            subject = seq_id_keys[2]
            
            
            if scene not in self.data_radar.keys():
                self.data_radar[scene] = {}
            if action not in self.data_radar[scene].keys():
                self.data_radar[scene][action] = {}
            self.data_radar[scene][action][subject] = data_radar[seq_id]
        
        if self.train_stage == 2:    
            self.motion_feat = np.load(self.motion_feat_file, allow_pickle=True)

            self.motion_pred = np.load(self.motion_pred_file, allow_pickle=True)

                   
        
    
    def process_data_seq(self):
        print("Process data seq...")
        # print(self.data.keys())
        for scene in self.data.keys():
            for action in self.data[scene].keys():
                for sub in self.data[scene][action].keys():
                    # print(sub, action, self.data[sub][action].shape)
                    the_sequence = self.data[scene][action][sub]
                    n, j, _ = the_sequence.shape
                    if self.sample_rate < 1:
                            even_list = range(0, n, 1)
                    else:
                        even_list = range(0, n, self.sample_rate)
                    num_frames = len(even_list) - self.seq_len - self.time_lag_before
                    the_sequence = np.array(the_sequence[even_list, :, :])
                    
                    step = 10
                    valid_frames_num = num_frames//step
                    valid_frames = np.arange(0, valid_frames_num) * step + self.time_lag_before
                    tmp_data_idx_1 = [scene] * len(valid_frames)
                    tmp_data_idx_2 = [action] * len(valid_frames)
                    tmp_data_idx_3 = [sub] * len(valid_frames)
                    tmp_data_idx_4 = list(valid_frames)
                    self.data_idx.extend(zip(tmp_data_idx_1, tmp_data_idx_2, tmp_data_idx_3, tmp_data_idx_4))
                

    def __len__(self):
        return np.shape(self.data_idx)[0]

    def __getitem__(self, item):
        # import pdb
        # pdb.set_trace()
        scene, action, sub, start_frame = self.data_idx[item]
        fs = np.arange(start_frame, start_frame + self.seq_len_before)
        fs_radar = np.arange(start_frame-self.time_lag_before, start_frame + self.seq_len_before)
        

        pose = self.data[scene][action][sub][fs].copy()
        # pose = low_pass_filter_motion(pose, n_pre=4)
        pose_noise = self.data_noise[scene][action][sub][fs].copy()
        radar = self.data_radar[scene][action][sub][fs_radar].copy()

        radar = pad_and_filter_radar(radar, max_points=100)
        
        if self.train_stage == 2:
            motion_pred = self.motion_pred[item]
            motion_feat = self.motion_feat[item]
            
            motion_pred = inverse_dct(motion_pred, self.in_n, T_out=self.in_n+self.out_n)
        
        else:
            motion_pred = None
            motion_feat = None
        
        if self.sample_rate < 1:
            pose = frame_rate_interpolate(pose, self.sample_rate, 1)
            pose_noise = frame_rate_interpolate(pose_noise, self.sample_rate, 1)
        


        pose = transform_pose(pose)
        pose_noise = transform_pose(pose_noise)
        
        radar[:,:,:3] = transform_pose(radar[:,:,:3])
        # radar_raw[:,:,2] = -radar_raw[:,:,2]

        pose = pose.reshape(self.in_n + self.out_n, -1)
        pose_noise = pose_noise.reshape(self.in_n + self.out_n, -1)

        
        mask = np.zeros((pose.shape[0], 17,3))
        mask[0:self.in_n, :,:] = 1
        mask[self.in_n:self.in_n + self.out_n, :,:] = 0

        observed = pose_noise.copy()
            
            

        data = {
           "observed": observed,
            "pose": pose,
            "pose_feat": np.zeros((1,17,32)),
            "observed_multi":np.zeros((1,1,17,3)),
            "observed_variance": np.zeros((1,1,17,3)),
            "mask": mask,
            "radar": radar[:self.in_n+self.time_lag_before],
            "radar_select": radar[:self.in_n+self.time_lag_before],
            "limb_len": np.zeros(1), 
            "timepoints": np.arange(self.in_n + self.out_n),
            "motion_pred": motion_pred,
            "motion_feat": motion_feat,
            "sub": int(action),
        }

        return data


def augment(observed, pose):
    _, F = observed.shape
    observed = observed.reshape(-1, F//3, 3)
    pose = pose.reshape(-1, F//3, 3)
    if np.random.uniform() > 0.5:  # x-y rotating
        theta = np.random.uniform(0, 2 * np.pi)
        rotate_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        rotate_xy = np.matmul(observed.transpose([1, 0, 2])[..., 0:2], rotate_matrix)
        observed[..., 0:2] = rotate_xy.transpose([1, 0, 2])
        rotate_xy = np.matmul(pose.transpose([1, 0, 2])[..., 0:2], rotate_matrix)
        pose[..., 0:2] = rotate_xy.transpose([1, 0, 2])
        del theta, rotate_matrix, rotate_xy
    if np.random.uniform() > 0.5:  # x-z mirroring
        observed[..., 0] = - observed[..., 0]
        pose[..., 0] = - pose[..., 0]
    if np.random.uniform() > 0.5:  # y-z mirroring
        observed[..., 1] = - observed[..., 1]
        pose[..., 1] = - pose[..., 1]
    observed = observed.reshape(-1, F)
    pose = pose.reshape(-1, F)
    return observed, pose


def transform_pose(pose):
    pose = pose.copy()
    pose[:, :, 1] = -pose[:, :, 1]
    pose_copy = pose[:, :, 2].copy()
    pose[:, :, 2] = pose[:, :, 1].copy()
    pose[:, :, 1] = pose_copy.copy()
    pose_copy = pose[:, :, 0].copy()
    pose[:, :, 0] = pose[:, :, 1].copy()
    pose[:, :, 1] = pose_copy.copy()
    return pose


def pad_and_filter_radar(radar, max_points=100):
    T, N, C = radar.shape
    filtered_radar = np.zeros((T, max_points, C), dtype=radar.dtype)
    for t in range(T):
        # Find non-null points (not all zeros in the 5 features)
        mask = ~(np.all(radar[t] == 0, axis=1))
        valid_points = radar[t][mask]
        num_valid = valid_points.shape[0]
        if num_valid > max_points:
            idx = np.random.choice(num_valid, max_points, replace=False)
            selected_points = valid_points[idx]
            filtered_radar[t, :max_points] = selected_points
        else:
            filtered_radar[t, :num_valid] = valid_points
        # The rest remain zeros (null points)
    return filtered_radar

def frame_rate_interpolate(before_array, before_rate, after_rate):
    before_seq_len, j, n = before_array.shape
    expand = after_rate/before_rate
    after_seq_len = int(before_seq_len * expand)
    after_array = np.zeros((after_seq_len, j, n), dtype=before_array.dtype)
    for i in range(before_seq_len):
        start = int(np.floor(i * expand))
        end = int(np.floor((i+1) * expand))
        for j in range(start, end):
            after_array[j] = before_array[i]
    return after_array

def low_pass_filter_motion(motion: np.ndarray, n_pre: int) -> np.ndarray:

    def dct_type_2(x: np.ndarray) -> np.ndarray:
        T = x.shape[0]
        n = np.arange(T)
        k = n.reshape(-1, 1)
        dct_basis = np.sqrt(2 / T) * np.cos(np.pi * (2 * n + 1) * k / (2 * T))
        dct_basis[0] /= np.sqrt(2)
        return dct_basis @ x

    def idct_type_3(x_dct: np.ndarray) -> np.ndarray:
        T = x_dct.shape[0]
        k = np.arange(T)
        n = k.reshape(-1, 1)
        idct_basis = np.sqrt(2 / T) * np.cos(np.pi * (2 * n + 1) * k / (2 * T))
        idct_basis[:, 0] /= np.sqrt(2)
        return idct_basis @ x_dct

    T, J, C = motion.shape
    motion_flat = motion.reshape(T, J * C)
    motion_normalized = motion_flat# - motion_flat[0]

    # Apply DCT
    motion_dct = dct_type_2(motion_normalized)

    # Apply low-pass filter (zero out high-frequency components)
    motion_dct[n_pre:] = 0

    # Reconstruct with inverse DCT
    motion_filtered = idct_type_3(motion_dct)

    # Restore original offset
    motion_filtered += 0# motion_flat[0]

    # Reshape back to (T, J, 3)
    return motion_filtered.reshape(T, J, C)


import math
def inverse_dct(data, T, T_out=24):
    """
    Convert low-pass DCT motion [N, C] or [B, N, C] to time domain [T, C] or [B, T, C]
    with optional padding to T_out by repeating the last frame.

    Args:
        data: torch.Tensor or np.ndarray, shape [N, C] or [B, N, C]
        T: number of time frames to reconstruct (IDCT output)
        T_out: optional, pad to this length by repeating last frame (T_out ≥ T)

    Returns:
        motion_time: [T, C] or [B, T, C] if T_out not specified,
                     or [T_out, C] / [B, T_out, C] if padding requested
    """
    is_numpy = isinstance(data, np.ndarray)
    xp = np if is_numpy else torch

    # Convert to torch tensor if needed
    if is_numpy:
        data = torch.from_numpy(data)

    # Expand to 3D if needed
    if data.ndim == 2:
        data = data.unsqueeze(0)  # [1, N, C]
        squeeze_output = True
    else:
        squeeze_output = False

    B, N, C = data.shape
    device = data.device

    # Construct full IDCT matrix [T, T], then truncate to [T, N]
    t = torch.arange(T, device=device).float().unsqueeze(1)
    k = torch.arange(T, device=device).float().unsqueeze(0)
    idct_mat = torch.cos(math.pi * (2 * t + 1) * k / (2 * T))  # [T, T]
    idct_mat[:, 0] *= 1 / math.sqrt(2)
    idct_mat *= math.sqrt(2 / T)
    idct = idct_mat[:, :N]  # [T, N]

    # IDCT: [B, T, C]
    motion_time = torch.matmul(idct.unsqueeze(0), data)  # [B, T, C]

    # Pad if needed
    if T_out is not None and T_out > T:
        pad_len = T_out - T
        last_frame = motion_time[:, -1:, :]  # [B, 1, C]
        pad = last_frame.expand(B, pad_len, C)  # [B, pad_len, C]
        motion_time = torch.cat([motion_time, pad], dim=1)  # [B, T_out, C]

    # Squeeze if input was unbatched
    if squeeze_output:
        motion_time = motion_time[0]  # [T/T_out, C]

    return motion_time.numpy() if is_numpy else motion_time

if __name__ == "__main__":
    # data_gt = np.load("./data_mmfi/gt_poses_train.npz",allow_pickle=True)
    # data_pred = np.load("./data_mmfi/pred_poses_train.npz",allow_pickle=True)
    # data_diff = np.load("./data_mmfi/diff_pred_poses_train.npz",allow_pickle=True)

    # for scene in data_gt.keys():
    #         # print(sub, action, self.data[sub][action].shape)
    #         plot_gt = data_gt[scene].reshape(-1,51)
    #         plot_diff = data_diff[scene].reshape(-1,51)

    #         break
    # from plot import plot_pose_sequence_as_gif
    # start = 50
    # plot_pose_sequence_as_gif(torch.tensor(plot_gt[start:start+25,3:])[None,:,:], torch.tensor(plot_diff[start:start+25,3:])[None,:,:], t_his=5)
    dataset = mmfi(None, input_n=10, output_n=20, skip_rate= None, split=0, miss_rate=(10 / 100), miss_scale=10,
                            miss_type="pred", all_data=True, joint_n=17)
    from torch.utils.data import DataLoader
    dataset_multi_test = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=16,
                                pin_memory=True)
    # data_gen_multi_test = dataset_multi_test.iter_generator(step=cfg.t_his)
    from plot import plot_pose_sequence_as_gif
    from mmBody_utils import plot_pose_sequence_as_gif_selected
    for batch in dataset_multi_test:
        pose_observe, pose_gt, radar = batch["observed"], batch["pose"], batch["radar"]
        print(pose_observe.shape, pose_gt.shape, radar.shape)
        # plot_pose_sequence_as_gif(pose_gt[...,3:], pose_observe[...,3:], t_his=10)
        plot_pose_sequence_as_gif_selected(pose_observe[...,3:], pose_gt[...,3:], selected_raw_gt=radar, selected_raw_pred=radar, t_his=10, elev=0, azim=0)

