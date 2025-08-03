import os
import numpy as np
import torch
from torch.utils.data import Dataset
from depos_utils import data_utils
import pathlib
import math
import copy
from data_loader.skeleton import Skeleton

class H36M(Dataset):

    def __init__(self, data_dir, input_n, output_n, skip_rate, actions=None, split=0, miss_rate=0.2, miss_scale=25,
                 miss_type='no_miss', all_data=False, joint_n=32, use_vel = False):
        """
        :param path_to_data:
        :param actions:
        :param input_n:
        :param output_n:
        :param dct_used:
        :param split: 0 train, 1 validation, 2 testing
        :param sample_rate:
        """
        self.path_to_data = os.path.join(data_dir, 'h3.6m/dataset')
        splits = ["train", "validate", "test"]
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
        self.sample_rate = 2 #2
        self.all_data = all_data
        self.actions = actions
        self.use_vel = use_vel
        self.p3d = {}
        self.params = {}
        self.masks = {}
        self.data_idx = []
        self.seq_len = self.in_n + self.out_n

        self.data = None
        
        self.prepare_data()
        
        print(f"Generating {self.__len__()} samples...") 

    def prepare_data(self):
        self.data_file = os.path.join('data', 'data_3d_h36m.npz')
        self.subjects_split = {'train': [1, 5, 6, 7, 8],
                               'validate': [9],
                               'test': [11]}
        self.subjects = ['S%d' % x for x in self.subjects_split[self.split]]
        self.skeleton = Skeleton(parents=[-1, 0, 1, 2, 3, 4, 0, 6, 7, 8, 9, 0, 11, 12, 13, 14, 12,
                                          16, 17, 18, 19, 20, 19, 22, 12, 24, 25, 26, 27, 28, 27, 30],
                                 joints_left=[6, 7, 8, 9, 10, 16, 17, 18, 19, 20, 21, 22, 23],
                                 joints_right=[1, 2, 3, 4, 5, 24, 25, 26, 27, 28, 29, 30, 31])
        if self.joint_n == 17:
            self.removed_joints = {4, 5, 9, 10, 11, 16, 20, 21, 22, 23, 24, 28, 29, 30, 31}
            self.kept_joints = np.array([x for x in range(32) if x not in self.removed_joints])
        else:
            self.removed_joints = np.array([0, 1, 6, 11, 16, 20, 23, 24, 28, 31])
            self.kept_joints = np.array([x for x in range(32) if x not in self.removed_joints])

        # ignore constant joints and joints at same position with other joints
        

        self.skeleton.remove_joints(self.removed_joints)
        self.skeleton._parents[11] = 8
        self.skeleton._parents[14] = 8
        self.process_data()  
        self.process_data_seq()

    def process_data(self):
        data_o = np.load(self.data_file, allow_pickle=True)['positions_3d'].item()
        self.S1_skeleton = data_o['S1']['Directions'][:1, self.kept_joints].copy()
        data_f = dict(filter(lambda x: x[0] in self.subjects, data_o.items()))
        if self.all_data == False:
            print("Not using all data.")
            for key in list(data_f.keys()):
                data_f[key] = dict(filter(lambda x: all([a in x[0] for a in self.actions]), data_f[key].items()))
                if len(data_f[key]) == 0:
                    data_f.pop(key)
        for data_s in data_f.values():
            for action in data_s.keys():
                seq = data_s[action][:, self.kept_joints, :]
                if self.use_vel:
                    v = (np.diff(seq[:, :1], axis=0) * 50).clip(-5.0, 5.0)
                    v = np.append(v, v[[-1]], axis=0)
                # seq[:, :] -= seq[:, :1]
                if self.use_vel:
                    seq = np.concatenate((seq, v), axis=1)
                data_s[action] = seq
        self.data = data_f
    
    def process_data_seq(self):
        # print(self.data.keys())
        for sub in self.data.keys():
            for action in self.data[sub].keys():
                # print(sub, action, self.data[sub][action].shape)
                the_sequence = self.data[sub][action]
                n, j, _ = the_sequence.shape
                even_list = range(0, n, self.sample_rate)
                num_frames = len(even_list)
                the_sequence = np.array(the_sequence[even_list, :, :])
                step = self.seq_len//10
                valid_frames_num = num_frames//step
                valid_frames = np.arange(0, valid_frames_num) * step
                tmp_data_idx_1 = [sub] * len(valid_frames)
                tmp_data_idx_2 = [action] * len(valid_frames)
                tmp_data_idx_3 = list(valid_frames)
                self.data_idx.extend(zip(tmp_data_idx_1, tmp_data_idx_2, tmp_data_idx_3))
                

    def __len__(self):
        return np.shape(self.data_idx)[0]

    def __getitem__(self, item):
        sub, action, start_frame = self.data_idx[item]
        fs = np.arange(start_frame, start_frame + self.in_n + self.out_n)

        pose = self.data[sub][action][fs]
        pose = pose.reshape(self.in_n + self.out_n, -1)
        observed = pose.copy()

        if self.miss_type == 'no_miss':
            mask = np.zeros((pose.shape[0], pose.shape[1]))
            mask[0:self.in_n, :] = 1
            mask[self.in_n:self.in_n + self.out_n, :] = 0
        elif self.miss_type == 'random': 
            # Random Missing with Random Probability
            mask = np.zeros((self.in_n, pose.shape[1] // 3, 3))
            p_miss = np.random.uniform(0., 1., size=[self.in_n, pose.shape[1] // 3])
            p_miss_rand = np.random.uniform(0., 1.)
            mask[p_miss > p_miss_rand] = 1.0
            mask = mask.reshape((self.in_n, pose.shape[1]))
            predict = np.zeros((self.out_n, pose.shape[1]))
            mask = np.concatenate((mask, predict), axis=0)
        elif self.miss_type == 'random_joints': # Table 5 Setting C
            # Random Joint Missing
            mask = np.zeros((self.in_n, pose.shape[1]))
            p_miss = self.miss_rate * np.ones((pose.shape[1], 1))
            for i in range(0, pose.shape[1], 3):
                A = np.random.uniform(0., 1., size=[self.in_n, ])
                B = A > p_miss[i]
                mask[:, i] = 1. * B
                mask[:, i + 1] = 1. * B
                mask[:, i + 2] = 1. * B
            predict = np.zeros((self.out_n, pose.shape[1]))
            mask = np.concatenate((mask, predict), axis=0)
        elif self.miss_type == 'random_right_leg':
            # Right Leg Random Missing
            mask = np.ones((self.in_n, pose.shape[1]))
            rand = np.random.choice(self.in_n, size=math.floor(self.miss_rate * self.in_n), replace=False)
            right_leg = [1, 2, 3] # [1, 2, 3]
            for i in right_leg:
                mask[rand, 3 * i] = 0.
                mask[rand, 3 * i + 1] = 0.
                mask[rand, 3 * i + 2] = 0.
        elif self.miss_type == 'random_left_arm_right_leg': # Table 4 Setting B FDE
            # Left Arm and Right Leg Random Missing
            mask = np.ones((self.in_n, pose.shape[1]))
            rand = np.random.choice(self.in_n, size=math.floor(self.miss_rate * self.in_n), replace=False)
            left_arm_right_leg = [1, 2, 3, 11, 12, 13] # [1, 2, 3, 17, 18, 19]
            for i in left_arm_right_leg:
                mask[rand, 3 * i] = 0.
                mask[rand, 3 * i + 1] = 0.
                mask[rand, 3 * i + 2] = 0.
            predict = np.zeros((self.out_n, pose.shape[1]))
            mask = np.concatenate((mask, predict), axis=0)
        elif self.miss_type == 'structured_joint':
            # Structured Joint Missing (Continuous)
            mask = np.ones((self.in_n, pose.shape[1]))
            rand = np.random.choice(self.in_n - 10, size=1, replace=False)
            right_leg = [1, 2, 3] # [1, 2, 3]
            for i in right_leg:
                mask[rand[0]:rand[0] + 10, 3 * i] = 0.
                mask[rand[0]:rand[0] + 10, 3 * i + 1] = 0.
                mask[rand[0]:rand[0] + 10, 3 * i + 2] = 0.
            predict = np.zeros((self.out_n, pose.shape[1]))
            mask = np.concatenate((mask, predict), axis=0)
        elif self.miss_type == 'structured_frame':
            # Structured Frame Missing (Continuous)
            mask = np.ones((self.in_n, pose.shape[1]))
            rand = np.random.choice(self.in_n - 5, size=1, replace=False)
            mask[rand[0]:rand[0] + 5, ] = 0.
            predict = np.zeros((self.out_n, pose.shape[1]))
            mask = np.concatenate((mask, predict), axis=0)
        elif self.miss_type == 'random_frame':
            # Random Frame Missing
            mask = np.ones((self.in_n, pose.shape[1]))
            rand = np.random.choice(self.in_n, size=math.floor(self.miss_rate * self.in_n), replace=False)
            mask[rand, :] = 0.
            predict = np.zeros((self.out_n, pose.shape[1]))
            mask = np.concatenate((mask, predict), axis=0)
        elif self.miss_type == 'noisy_50':
            # Noisy Leg with Sigma=50
            mask = np.ones((self.in_n, pose.shape[1]))
            leg = [1, 2, 3, 4, 5, 6] # [1, 2, 3, 6, 7, 8]
            sigma = 50
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

