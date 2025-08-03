import os
import numpy as np
import torch
from torch.utils.data import Dataset
import math
import copy
import glob

class MMVR(Dataset):

    def __init__(self, input_n=25, output_n=100, miss_type = "noisy_25", miss_rate=0.2, miss_scale=25, root='/home/junqiao/projects/data/MMVR/', coarse_3d_path = "/home/junqiao/projects/data/MMVR/results.npy", split=0):

        self.root = root
        self.coarse_3d_path = coarse_3d_path
        splits = ["train", "validate", "test"]
        self.split_num = split
        self.split = splits[split]
        self.data_idx = []
        self.sample_rate = 15/25

        self.in_n = input_n
        self.out_n = output_n
        self.mask_apply = True
        self.miss_rate = miss_rate
        self.miss_scale = miss_scale
        self.miss_type = miss_type
        self.data_idx = []
        self.seq_len = self.in_n + self.out_n
        self.seq_len_before = int(self.seq_len*self.sample_rate)

        self.load_split()
        self.load_data()
        self.load_coarse_3d_pose()
        self.process_data_seq()
    
    def load_split(self):
        self.train_list = []
        split_list = np.load(self.root+"data_split.npz", allow_pickle=True)["data_split_dict"][0]
        self.split_list = []
        for subject in split_list.keys():
            for seq in split_list[subject][self.split]:
                self.train_list.append(seq)


    def load_data(self):
        root = self.root

        self.pose_dict = {}
        self.radar_ver_dict = {}
        self.radar_hor_dict = {}
        self.meta_dict = {}

        for subject_path in glob.glob(os.path.join(root,"*")):
            subject = subject_path.split("/")[-1]
            if subject[:4] == "d1s1":
                print("find ", subject)
                self.pose_dict[subject] = {}
                self.radar_ver_dict[subject] = {}
                self.radar_hor_dict[subject] = {}
                self.meta_dict[subject] = {}
                for seq_path in glob.glob(os.path.join(subject_path,"*")):
                    seq = seq_path.split("/")[-1]
                    seq_idx = 0
                    pose_array = []
                    radar_ver_array = []
                    radar_hor_array = []
                    while True:
                        # Load pose file
                        pose_file = glob.glob(os.path.join(seq_path,f"{seq_idx:05}_pose.npz"))
                        radar_file = glob.glob(os.path.join(seq_path,f"{seq_idx:05}_radar.npz"))
                        # if len(radar_file) != 0:
                        #     radar = np.load(radar_file[0])
                        #     radar_hor_array.append(radar['hm_hori'])
                        #     radar_ver_array.append(radar['hm_vert'])
                        # else: 
                        #     radar_hor_array = np.concatenate(radar_hor_array)
                        #     radar_ver_array = np.concatenate(radar_ver_array)
                        if len(pose_file) != 0:
                            # Process pose file
                            pose = np.load(pose_file[0])['kp']
                            pose = coco2h36m(pose)
                            pose_array.append(pose)
                        else:
                            pose_array = np.concatenate(pose_array)
                            break
                        seq_idx += 1
                    self.pose_dict[subject][seq] = pose_array
                    self.radar_hor_dict[subject][seq] = radar_hor_array
                    self.radar_ver_dict[subject][seq] = radar_ver_array
                    self.meta_dict[subject][seq] = seq_idx

    def load_coarse_3d_pose(self):
        coarse_3d_pose = np.load(self.coarse_3d_path)    
        self.coarse_3d_dict = {}
        start = 0
        for subject in self.meta_dict.keys():
            self.coarse_3d_dict[subject] = {}
            for seq in self.meta_dict[subject].keys():
                store = coarse_3d_pose[start:start+self.meta_dict[subject][seq]][:,0,:,:]
                store_copy = store[:,:,0]
                store[:,:,0] = -store[:,:,2]
                store[:,:,2] = -store[:,:,1]
                store[:,:,1] = store_copy
                self.coarse_3d_dict[subject][seq] = store
                start = start+self.meta_dict[subject][seq]
        
    
    def process_data_seq(self):
        # print(self.data.keys())
        for sub in self.coarse_3d_dict.keys():
            for seq in self.coarse_3d_dict[sub].keys():
                # print(sub, action, self.data[sub][action].shape)
                the_sequence = self.coarse_3d_dict[sub][seq]
                n, j, _ = the_sequence.shape
                if self.sample_rate < 1:
                    even_list = range(0, n, 1)
                num_frames = len(even_list)
                the_sequence = np.array(the_sequence[even_list, :, :])
                step = self.seq_len_before//10
                valid_frames_num = num_frames//step
                valid_frames = np.arange(0, valid_frames_num) * step
                tmp_data_idx_1 = [sub] * len(valid_frames)
                tmp_data_idx_2 = [seq] * len(valid_frames)
                tmp_data_idx_3 = list(valid_frames)
                self.data_idx.extend(zip(tmp_data_idx_1, tmp_data_idx_2, tmp_data_idx_3))
        

    def __len__(self):
        return np.shape(self.data_idx)[0]

    def __getitem__(self, item):
        sub, seq, start_frame = self.data_idx[item]
        fs = np.arange(start_frame, start_frame + self.seq_len_before)
        


        pose = self.coarse_3d_dict[sub][seq][fs]
        pose = frame_rate_interpolate(pose, self.sample_rate, 1)
        pose = pose.reshape(self.in_n + self.out_n, -1)
        observed = pose.copy()

        if self.miss_type == 'no_miss':
            mask = np.zeros((pose.shape[0], pose.shape[1]))
            mask[0:self.in_n, :] = 1
            mask[self.in_n:self.in_n + self.out_n, :] = 0
        elif self.miss_type == 'noisy_25':
            # Noisy Leg with Sigma=25
            mask = np.ones((self.in_n, pose.shape[1]))
            leg = [1, 2, 3, 4, 5, 6] # [1, 2, 3, 6, 7, 8]
            sigma = 25
            noise = np.random.normal(0, sigma, size=observed.shape)
            noise[self.in_n:, :] = 0
            for i in range(0, self.in_n):
                missing_leg_joints = np.random.choice(leg, 3)
                for j in range(3):
                    mask[i, missing_leg_joints[j] * 3] = 0
                    mask[i, missing_leg_joints[j] * 3 + 1] = 0
                    mask[i, missing_leg_joints[j] * 3 + 2] = 0

                    noise[i, missing_leg_joints[j] * 3] = 0
                    noise[i, missing_leg_joints[j] * 3 + 1] = 0
                    noise[i, missing_leg_joints[j] * 3 + 2] = 0
            observed = (observed * 1000 + noise) / 1000
            predict = np.zeros((self.out_n, pose.shape[1]))
            mask = np.concatenate((mask, predict), axis=0)
        else:
            mask = np.zeros((pose.shape[0], pose.shape[1]))
            mask[0:self.in_n, :] = 1
            mask[self.in_n:self.in_n + self.out_n, :] = 0


        if self.split_num < 0:
            observed, pose = augment(observed, pose)
        
        if self.mask_apply:
            unknown_in =  1 - mask[:self.in_n, :]
            noise = np.random.normal(0, self.miss_scale, size=unknown_in.shape)
            observed[:self.in_n, :] = (observed[:self.in_n, :] * 1000 + noise * unknown_in) / 1000

        data = {
            "observed": observed,
            "pose": pose,
            "mask": mask.copy(),
            "timepoints": np.arange(self.in_n + self.out_n)
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

def coco2h36m(x):
    '''
        Input: x (M x T x V x C)
        
        COCO: {0-nose 1-Leye 2-Reye 3-Lear 4Rear 5-Lsho 6-Rsho 7-Lelb 8-Relb 9-Lwri 10-Rwri 11-Lhip 12-Rhip 13-Lkne 14-Rkne 15-Lank 16-Rank}
        
        H36M:
        0: 'root',
        1: 'rhip',
        2: 'rkne',
        3: 'rank',
        4: 'lhip',
        5: 'lkne',
        6: 'lank',
        7: 'belly',
        8: 'neck',
        9: 'nose',
        10: 'head',
        11: 'lsho',
        12: 'lelb',
        13: 'lwri',
        14: 'rsho',
        15: 'relb',
        16: 'rwri'
    '''
    y = np.zeros(x.shape)
    y[:,0,:] = (x[:,11,:] + x[:,12,:]) * 0.5
    y[:,1,:] = x[:,12,:]
    y[:,2,:] = x[:,14,:]
    y[:,3,:] = x[:,16,:]
    y[:,4,:] = x[:,11,:]
    y[:,5,:] = x[:,13,:]
    y[:,6,:] = x[:,15,:]
    y[:,8,:] = (x[:,5,:] + x[:,6,:]) * 0.5
    y[:,7,:] = (y[:,0,:] + y[:,8,:]) * 0.5
    y[:,9,:] = x[:,0,:]
    y[:,10,:] = (x[:,1,:] + x[:,2,:]) * 0.5
    y[:,11,:] = x[:,5,:]
    y[:,12,:] = x[:,7,:]
    y[:,13,:] = x[:,9,:]
    y[:,14,:] = x[:,6,:]
    y[:,15,:] = x[:,8,:]
    y[:,16,:] = x[:,10,:]
    return y

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
    