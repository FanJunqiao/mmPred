import os
import numpy as np
import torch
from torch.utils.data import Dataset

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN

from .mmBody_utils import *




class mmBody(Dataset):

    def __init__(self, data_dir, input_n, output_n, skip_rate, actions=None, split=0, miss_rate=0.2, miss_scale=25,
                 miss_type='no_miss', all_data=False, joint_n=32, limb_len_scale = True, aug = True, dct_i=4):
        """
        :param actions:
        :param input_n:
        :param output_n:
        :param dct_used:
        :param split: 0 train, 1 validation, 2 testing
        :param sample_rate:
        """
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
        self.limb_len_scale = limb_len_scale
        self.dct_i = dct_i

        # for mmBody noise type
        if self.miss_type == "no_miss":
            self.joint_noise_type = "gt"
        else:
            self.joint_noise_type = self.miss_type

        self.data = None
        self.aug = (aug and self.split_num == 0)
        
        self.prepare_data()

        print(f"Generating {self.__len__()} samples...") 

    def prepare_data(self):
        if self.split == "train":
            self.subjects = ['18', '17', '11', '0', '19', '5', '1', '14', '2', '10', '13', '12', '16', '3', '15', '6', '4', '8', '7', '9']
            self.data_file = os.path.join('./data_mmBody', "gt" + '_poses_' + self.split +'.npz')
            self.data_noise_file = os.path.join('./data_mmBody', self.joint_noise_type + '_poses_' + self.split +'.npz')
            self.feat_file = os.path.join('./data_mmBody', 'feat_' + self.split +'.npz')
            self.radar_file = os.path.join('./data_mmBody', 'radar_' + self.split +'.npz')
            self.data_noise_multi_file = os.path.join('./data_mmBody', 'diff_pred_poses_train_multi.npz')
        elif self.split == "validate":
            self.subjects = ['10', '11', '20', '21', '30', '31', '40', '41', '50', '51', '60', '61', '70', '71']
            self.data_file = os.path.join('./data_mmBody', "gt" + '_poses_' + self.split +'.npz')
            self.data_noise_file = os.path.join('./data_mmBody', self.joint_noise_type + '_poses_' + self.split +'.npz')
            self.feat_file = os.path.join('./data_mmBody', 'feat_' + self.split +'.npz')
            self.radar_file = os.path.join('./data_mmBody', 'radar_' + self.split +'.npz')
            self.data_noise_multi_file = os.path.join('./data_mmBody', 'diff_pred_poses_test_multi.npz')
        else:
            self.subjects = ['10', '11', '20', '21', '30', '31', '40', '41', '50', '51', '60', '61', '70', '71']
            self.data_file = os.path.join('./data_mmBody', "gt" + '_poses_' + self.split +'.npz')
            self.data_noise_file = os.path.join('./data_mmBody', self.joint_noise_type + '_poses_' + self.split +'.npz')
            self.feat_file = os.path.join('./data_mmBody', 'feat_' + self.split +'.npz')
            self.radar_file = os.path.join('./data_mmBody', 'radar_' + self.split +'.npz')
            self.data_noise_multi_file = os.path.join('./data_mmBody', 'diff_pred_poses_test_multi.npz')
        
        self.motion_feat_file = os.path.join(f"data_mmBody/dct_ablat_data_mmBody/all_feature_{self.split}_{self.dct_i}.npy")
        self.motion_pred_file = os.path.join(f"data_mmBody/dct_ablat_data_mmBody/all_pred_{self.split}_{self.dct_i}.npy")
        print(f"Motion feature file: {self.motion_feat_file}")
        print(f"Motion prediction file: {self.motion_pred_file}")

        self.kept_joints = np.array([x for x in range(17)])
        self.process_data()  
        self.process_data_seq()
        self.post_process_data_seq()
        self.limb_length_estimation()

    def process_data(self):
        self.data = np.load(self.data_file, allow_pickle=True)
        self.data = {key: self.data[key] for key in self.data.keys()}

        self.data_noise = np.load(self.data_noise_file, allow_pickle=True)
        self.data_noise = {key: self.data_noise[key] for key in self.data_noise.keys()}
        
        self.data_noise_multi = np.load(self.data_noise_multi_file, allow_pickle=True)
        self.data_noise_multi = {key: self.data_noise_multi[key] for key in self.data_noise_multi.keys()}

        self.data_feat = np.load(self.feat_file, allow_pickle=True)
        self.data_feat = {key: self.data_feat[key] for key in self.data_feat.keys()}

        self.data_radar = np.load(self.radar_file, allow_pickle=True)
        self.data_radar = {key: self.data_radar[key] for key in self.data_radar.keys()}



        self.motion_feat = np.load(self.motion_feat_file, allow_pickle=True)

        self.motion_pred = np.load(self.motion_pred_file, allow_pickle=True)

        print(self.motion_feat.shape, self.motion_pred.shape)
    
    def process_data_seq(self):
        # print(self.data.keys())
        for sub in self.data.keys():
            # print(sub, action, self.data[sub][action].shape)
            the_sequence = self.data[sub]
            n, j, _ = the_sequence.shape
            if self.sample_rate < 1:
                    even_list = range(0, n, 1)
            else:
                even_list = range(0, n, self.sample_rate)
            num_frames = len(even_list) - self.seq_len
            the_sequence = np.array(the_sequence[even_list, :, :])
            # step = self.seq_len_before//10
            step = 2
            valid_frames_num = num_frames//step
            valid_frames = np.arange(0, valid_frames_num) * step
            tmp_data_idx_1 = [sub] * len(valid_frames)
            tmp_data_idx_3 = list(valid_frames)
            self.data_idx.extend(zip(tmp_data_idx_1, tmp_data_idx_3))

    def post_process_data_seq(self):
        energy_dist = []
        select_idx = []
        
        for sub, start_frame in self.data_idx:
            fs = np.arange(start_frame, start_frame + self.seq_len_before)
            seq = self.data[sub][fs]      
            if compute_dct_energy(seq) > 3:
                select_idx.append((sub, start_frame))
                energy_dist.append(compute_dct_energy(seq))
            
        self.data_idx = select_idx

        from plot import plot_and_save_histogram
        plot_and_save_histogram(energy_dist, x_min=0, x_max=30)

    def limb_length_estimation(self):

        self.gt_data = np.load(f"limb_{self.split}.npz", allow_pickle=True)
        self.pred_data = np.load(f"limb_pred_{self.split}.npz", allow_pickle=True)

        self.gt_limb_data = {}
        self.pred_limb_data = {}

        
        for sub in self.gt_data.files:
            print("processing", sub)
            item_gt = self.gt_data[sub].item()["limb_length"]
            item_pred = self.pred_data[sub].item()["limb_length"]

            self.gt_limb_data[sub] = item_gt
            self.pred_limb_data[sub] = item_pred

            if self.limb_len_scale and self.joint_noise_type == "diff_pred": # debug
                # self.data[sub] = scale_pose_wild(self.data[sub], item_gt)
                self.data_noise[sub] = scale_pose_wild(self.data_noise[sub], item_pred)

            # self.data_radar[sub] = extract_framewise_jointwise_local(self.data_noise[sub], self.data_radar[sub], top_k=50, dist_thresh=0.3)
            # self.data_radar[sub] = self.data_radar[sub][:,:16*50,:].reshape(self.data_radar[sub].shape[0], 16, 50, 6)
            
        print(f"Loaded {len(self.gt_limb_data.keys())} subjects from split '{self.split}'.")

        # for sub in self.gt_limb_data.files:
        #     item_gt = self.gt_limb_data[sub].item()
        #     item_pred = self.pred_limb_data[sub].item()

        #     # limb_length 和 point_cloud
        #     pc = item_gt['point_cloud']             # [20000, 6]
        #     limblen_gt = item_gt['limb_length']       # [16]
        #     limblen_pred = item_pred['limb_length']  # [16]

        #     print(f"\n📍 Subject {sub}")
        #     print("GT limb len   :", np.round(limblen_gt, 3))
        #     print("Predicted len :", np.round(limblen_pred, 3))

        #     # Avoid division by zero by adding a small epsilon
        #     epsilon = 1e-6
        #     abs_percent_error = np.abs(limblen_pred - limblen_gt) / (np.abs(limblen_gt) + epsilon) * 100
        #     print("Abs % error   :", np.round(abs_percent_error, 3))

        #     # 可视化调用
        #     fs = np.arange(50)
        #     seq = self.data[sub][fs] 
        #     pcs = self.data_radar[sub][fs]    
        #     process_pc_and_plot_pose_xz_with_voxel_heatmap(
        #         pcs, 
        #         seq.mean(0),  # 可根据需要改为 limblen_gt
        #         gt_limb_len=limblen_gt,
        #         pred_limb_len=limblen_pred,
        #         k1=1, percentile1=100,
        #         k2=1, percentile2=100,
        #         ellip_percent=100,
        #         save_path=f"./radar_PC/limb{sub}_{self.split}.png"  # 不保存
        #     )


        # self.limb_length = []
        # self.limb_length_pc = []
        # limb_dict = {}  # 最终要保存的 dict
        # for sub in self.data.keys():
        #     fs = np.arange(50)
        #     pcs = self.data_radar[sub][fs]
        #     seq = self.data[sub][fs]      
        #     limb_len = compute_limb_lengths(seq).mean(0)

        #     filtered_pc = filter_points_in_ellipsoid_and_pad(pcs, max_points=20000)
        #     limb_dict[sub] = {
        #         'limb_length': limb_len.astype(np.float32),
        #         'point_cloud': filtered_pc.astype(np.float32)
        #     }
        #     print(f"{sub}: Radar PC shape {filtered_pc.shape}, limblen {limb_len}")


            

        #     # process_pc_and_plot_pose_xz_with_voxel_heatmap(pcs, seq[0], k1=1, percentile1=100, k2=1, percentile2=100, ellip_percent=100, save_path=f"./radar_PC/limb{sub}_{self.split}.png")

        # # 最后保存
        # np.savez(f"limb_{self.split}.npz", **limb_dict)
        # print(f"Saved to limb_{self.split}.npz")
    
    
    def analyze_mse_errors(self, prediction, ground_truth, idx, quantiles=3, selection = 0):
        """
        Analyze MSE errors between prediction and ground truth arrays.

        Parameters:
        - prediction (np.ndarray): Predicted values of shape (H, T, 17, 3).
        - ground_truth (np.ndarray): Ground truth values of shape (H, T, 17, 3).
        - bins (int): Number of bins for the histogram.
        - quantiles (int): Number of quantiles to divide the errors into.

        Returns:
        - dict: Dictionary with quantile names as keys and sample indices as values.
        - np.ndarray: Array of mean MSE errors for each sample.
        """

        # Calculate MSE for each sample
        mse_errors = np.mean((prediction - ground_truth) ** 2, axis=(1, 2, 3))

        

        # Split samples into quantiles based on error
        quantile_labels = [f"Quantile {i+1}" for i in range(quantiles)]
        quantiles_split = pd.qcut(mse_errors, q=quantiles, labels=quantile_labels)
        quantile_indices = {
            label: np.where(quantiles_split == label)[0]
            for label in quantile_labels
        }

        # Select the prediction subset corresponding to the lowest error quantile
        lowest_quantile_label = quantile_labels[selection]
        idx_new = [idx[i] for i in quantile_indices[lowest_quantile_label]]
        
        return idx_new

                

    def __len__(self):
        return np.shape(self.data_idx)[0]

    def __getitem__(self, item):
        sub, start_frame = self.data_idx[item]
        fs = np.arange(start_frame, start_frame + self.seq_len_before)

        pose = self.data[sub][fs].copy()
        pose_noise = self.data_noise[sub][fs].copy()
        pose_noise_multi = self.data_noise_multi[sub][fs].copy()
        # Variance over K, then mean over xyz
        pose_noise_variance = np.var(pose_noise_multi, axis=1)
        pose_feat = self.data_feat[sub][fs].reshape(self.seq_len_before, -1)[:self.in_n,:]
        radar = self.data_radar[sub][fs]
        limb_len = self.pred_limb_data[sub]
        radar_select = radar
        # radar_select = extract_framewise_jointwise_local(pose_noise, radar, top_k=50, dist_thresh=0.2)
        motion_pred = self.motion_pred[item]
        motion_feat = self.motion_feat[item]

        motion_pred = inverse_dct(motion_pred, self.in_n, T_out=self.in_n+self.out_n)
        

        



        # radar_select = dbscan_cluster_filter(radar, eps=0.01, min_samples=10, cluster_size_thresh=20, reduce_ratio=1, pad_ratio=1, dim_weight=[0.0,1,0.0])
        # radar_select = knn_outlier_aggressive_filter(radar, k=10, distance_percentile=90)
        # radar_select = extract_closest_points_global(pose_noise, radar, top_k=500)
        # radar_select = extract_framewise_jointwise_local(pose_noise, radar, top_k=50, dist_thresh=0.3)
        # radar_select = annotate_flow_from_pose(pose, radar, max_dist=0.5)


        if self.sample_rate < 1:
            pose = frame_rate_interpolate(pose, self.sample_rate, 1)
            pose_noise = frame_rate_interpolate(pose_noise, self.sample_rate, 1)
        
        # for mmBody
        pose = transform_axes_for_mmBody(pose)
        if self.miss_type != "rgb":
            pose_noise = transform_axes_for_mmBody(pose_noise)
        else:
            pose_noise[:,:,1] = -pose_noise[:,:,1]
            pose_noise[:,:,2] = -pose_noise[:,:,2]
            pose_noise = pose_noise[:,:,[2,0,1]]
        pose_noise_multi = transform_axes_for_mmBody(pose_noise_multi)
        radar = transform_axes_for_mmBody(radar)
        radar_select = transform_axes_for_mmBody(radar_select)
        radar_select[...,-3:] = transform_axes_for_mmBody(radar_select[...,-3:])
        

        # Example usage: you can add it to the data dict if needed
        # data["motion_pred_shift"] = padded_motion_pred_shift
        
        
    

        pose = pose.reshape(self.in_n + self.out_n, -1)
        pose_noise = pose_noise.reshape(self.in_n + self.out_n, -1)
        pose_noise_multi = pose_noise_multi.reshape(self.in_n + self.out_n,5,-1)

        

        

        # var_thre = 5e-4
        # var_mask = pose_noise_variance  > var_thre
        
        mask = np.zeros((pose.shape[0], 17,3))
        mask[0:self.in_n, :,:] = 1
        mask[self.in_n:self.in_n + self.out_n, :,:] = 0
        # mask[var_mask] = 0
        

        observed = pose_noise.copy()
        
            



        # if self.split_num < 0:
        #     observed, pose = augment(observed, pose)
        
        # if self.mask_apply:
        #     unknown_in =  1 - mask[:self.in_n, :]
        #     noise = np.random.normal(0, self.miss_scale, size=unknown_in.shape)
        #     observed[:self.in_n, :] = (observed[:self.in_n, :] * 1000 + noise * unknown_in) / 1000

        data = {
            "observed": observed,
            "pose": pose,
            "pose_feat": pose_feat[:self.in_n],
            "observed_multi": pose_noise_multi,
            "observed_variance": pose_noise_variance,
            "mask": mask,
            "radar": radar[:self.in_n],
            "radar_select": radar_select[:self.in_n],
            "limb_len": limb_len,
            "timepoints": np.arange(self.in_n + self.out_n),
            "motion_pred": motion_pred,
            "motion_feat": motion_feat,
            "sub": int(sub),
            "seq_id": fs[0],
        }

        return data






if __name__ == "__main__":
    dataset = mmBody(None, input_n=8, output_n=8, skip_rate= None, split=2, miss_rate=(10 / 100), miss_scale=10,
                            miss_type="diff_pred", all_data=True, joint_n=17)
    from torch.utils.data import DataLoader
    dataset_multi_test = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=16,
                                pin_memory=True)
    # data_gen_multi_test = dataset_multi_test.iter_generator(step=cfg.t_his)
    from mmBody_utils import *
    i = 0
    for batch in dataset_multi_test:
        if i>2000:
            pose_observe, pose_gt, pose_feat, radar, radar_select = batch["observed"], batch["pose"], batch["pose_feat"], batch["radar"], batch["radar_select"]
            print(pose_observe.shape, pose_gt.shape, pose_feat.shape, radar.shape, radar_select.shape)
            plot_pose_sequence_as_gif_selected(pose_observe[...,3:], pose_gt[...,3:], selected_raw_gt=radar_select, selected_raw_pred=radar, t_his=8, elev=90, azim=0)
        i += 16
