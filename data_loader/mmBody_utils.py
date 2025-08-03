import numpy as np
import torch

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN

def annotate_flow_from_pose(pose_seq, raw_pc, max_dist=0.5):
    """
    Annotate each radar point with flow vector from nearest joint.

    Args:
        pose_seq: [T, J, 3] — human pose sequence
        raw_pc: [T, N, 6] — radar point cloud
        max_dist: distance threshold to associate point with joint

    Returns:
        [T, N, 9] tensor with [x, y, z, f3, f4, f5, dx, dy, dz]
    """
    input_is_numpy = isinstance(pose_seq, np.ndarray)
    if input_is_numpy:
        pose_seq = torch.tensor(pose_seq, dtype=torch.float32)
        raw_pc = torch.tensor(raw_pc, dtype=torch.float32)

    T, N, _ = raw_pc.shape
    J = pose_seq.shape[1]

    output = []

    for t in range(T):
        points = raw_pc[t]  # [N, 6]
        xyz = points[:, :3]  # [N, 3]
        feats = points[:, 3:]  # [N, 3]

        # Default flow is 0
        flow = torch.zeros_like(xyz)

        if t < T - 1:
            joints_now = pose_seq[t]      # [J, 3]
            joints_next = pose_seq[t+1]   # [J, 3]

            # Compute distance [N, J]
            dist = torch.cdist(xyz, joints_now, p=2)  # [N, J]
            min_dist, nearest_idx = dist.min(dim=1)   # [N], [N]

            matched = min_dist <= max_dist            # [N] boolean mask
            joint_ids = nearest_idx[matched]          # indices of matched joints

            # Compute flow: kp[t+1] - kp[t]
            flow_vec = joints_next[joint_ids] - joints_now[joint_ids]  # [num_matched, 3]

            flow[matched] = flow_vec

        # Append: [xyz, feats, flow]
        annotated = torch.cat([xyz, feats, flow], dim=1)  # [N, 9]
        output.append(annotated)

    output = torch.stack(output, dim=0)  # [T, N, 9]
    return output.numpy() if input_is_numpy else output



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

def transform_axes_for_mmBody(data):
    data = data.copy()
    data[..., 1] *= -1
    data[..., [0, 1]] = data[..., [1, 0]]
    return data

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


def compute_dct_energy(motion: np.ndarray) -> np.ndarray:

    def dct_type_2(x: np.ndarray) -> np.ndarray:

        T = x.shape[0]
        n = np.arange(T)
        k = n.reshape(-1, 1)
        dct_basis = np.sqrt(2 / T) * np.cos(np.pi * (2 * n + 1) * k / (2 * T))
        dct_basis[0] /= np.sqrt(2)
        return dct_basis @ x

    T, J, C = motion.shape
    assert C == 3, "Last dimension should be 3 (XYZ coords)"

    # Step 1: Reshape (T, J, 3) → (T, J*3)
    motion_flat = motion.reshape(T, J * C)

    # Step 2: Subtract first frame (broadcasted subtraction)
    motion_normalized = motion_flat - motion_flat[0]

    # Step 3: Apply DCT (our custom DCT function along time axis)
    motion_dct = dct_type_2(motion_normalized)

    # Step 4: Energy = sum of squared DCT coefficients per dimension
    dct_energy = np.sum(motion_dct**2)

    return dct_energy

# from typing import Union

# ArrayType = Union[np.ndarray, torch.Tensor]

# def low_pass_filter_motion(motion: ArrayType, n_pre: int) -> ArrayType:
#     is_tensor = isinstance(motion, torch.Tensor)

#     def dct_type_2(x: ArrayType) -> ArrayType:
#         T = x.shape[0]
#         n = torch.arange(T) if is_tensor else np.arange(T)
#         k = n.reshape(-1, 1)
#         factor = np.sqrt(2 / T) if not is_tensor else torch.sqrt(torch.tensor(2.0 / T))
#         pi = np.pi if not is_tensor else torch.acos(torch.zeros(1)).item() * 2
#         dct_basis = factor * (torch.cos(pi * (2 * n + 1) * k / (2 * T)) if is_tensor else np.cos(pi * (2 * n + 1) * k / (2 * T)))
#         dct_basis[0] /= np.sqrt(2) if not is_tensor else torch.sqrt(torch.tensor(2.0))
#         return dct_basis @ x

#     def idct_type_3(x_dct: ArrayType) -> ArrayType:
#         T = x_dct.shape[0]
#         k = torch.arange(T) if is_tensor else np.arange(T)
#         n = k.reshape(-1, 1)
#         factor = np.sqrt(2 / T) if not is_tensor else torch.sqrt(torch.tensor(2.0 / T))
#         pi = np.pi if not is_tensor else torch.acos(torch.zeros(1)).item() * 2
#         idct_basis = factor * (torch.cos(pi * (2 * n + 1) * k / (2 * T)) if is_tensor else np.cos(pi * (2 * n + 1) * k / (2 * T)))
#         idct_basis[:, 0] /= np.sqrt(2) if not is_tensor else torch.sqrt(torch.tensor(2.0))
#         return idct_basis @ x_dct

#     T, J, C = motion.shape
#     motion_flat = motion.view(T, J * C) if is_tensor else motion.reshape(T, J * C)
#     motion_normalized = motion_flat - motion_flat[0]

#     motion_dct = dct_type_2(motion_normalized)
#     motion_dct[n_pre:] = 0
#     motion_filtered = idct_type_3(motion_dct)
#     motion_filtered += motion_flat[0]

#     return motion_filtered.view(T, J, C) if is_tensor else motion_filtered.reshape(T, J, C)


def extract_framewise_jointwise_local(pose_seq, raw_pc, top_k=50, dist_thresh=0.2):
    """
    For each frame and each joint (excluding root), extract up to top_k nearest radar points within a distance threshold.

    Args:
        pose_seq: [T, J, 3] — joint positions per frame
        raw_pc: [T, N, 6] — radar point cloud per frame
        top_k: number of closest points per joint
        dist_thresh: max distance to accept a point

    Returns:
        Tensor of shape [T, (J-1)*top_k, 6], with zero padding if necessary.
    """
    input_is_numpy = isinstance(pose_seq, np.ndarray)
    if input_is_numpy:
        pose_seq = torch.tensor(pose_seq, dtype=torch.float32)
        raw_pc = torch.tensor(raw_pc, dtype=torch.float32)

    T, J, _ = pose_seq.shape
    output = []

    for t in range(T):
        frame_pc = raw_pc[t]            # [N, 6]
        frame_pose = pose_seq[t]        # [J, 3]

        # Remove all-zero points
        valid_mask = ~(frame_pc == 0).all(dim=1)
        valid_pc = frame_pc[valid_mask]         # [N_valid, 6]
        xyz = valid_pc[:, :3]                   # [N_valid, 3]
        feats = valid_pc[:, 3:]                 # [N_valid, 3]

        frame_out = []

        for j in range(1, J):  # Exclude root joint
            joint_pos = frame_pose[j].unsqueeze(0)  # [1, 3]

            if xyz.shape[0] == 0:
                nearest = torch.zeros((top_k, 6), dtype=torch.float32)
            else:
                dist = torch.norm(xyz - joint_pos, dim=1)  # [N_valid]
                in_range_mask = dist <= dist_thresh
                in_range_idx = torch.nonzero(in_range_mask, as_tuple=False).squeeze(1)

                if in_range_idx.numel() == 0:
                    nearest = torch.zeros((top_k, 6), dtype=torch.float32)
                else:
                    # Sort the in-range points by distance
                    dists_in_range = dist[in_range_idx]
                    sorted_idx = torch.argsort(dists_in_range)[:top_k]
                    final_idx = in_range_idx[sorted_idx]
                    selected = torch.cat([xyz[final_idx], feats[final_idx]], dim=1)  # [<=top_k, 6]

                    # Pad if needed
                    pad_num = top_k - selected.shape[0]
                    if pad_num > 0:
                        pad = torch.zeros((pad_num, 6), dtype=selected.dtype)
                        selected = torch.cat([selected, pad], dim=0)
                    nearest = selected

            frame_out.append(nearest)

        frame_out = torch.cat(frame_out, dim=0)  # [ (J-1)*top_k, 6 ]
        output.append(frame_out)

    output = torch.stack(output, dim=0)  # [T, (J-1)*top_k, 6]
    return output.numpy() if input_is_numpy else output

def extract_closest_points_global(pose_seq, raw_pc, top_k=100, t_his=None):
    input_is_numpy = isinstance(pose_seq, np.ndarray)
    if input_is_numpy:
        pose_seq = torch.tensor(pose_seq, dtype=torch.float32)
        raw_pc = torch.tensor(raw_pc, dtype=torch.float32)

    T, J, _ = pose_seq.shape
    if t_his is None:
        t_his = T
    assert t_his <= T, f"t_his ({t_his}) must be <= total frames ({T})"

    # 1. Trim first t_his frames
    pose_seq = pose_seq[:t_his]                  # [t_his, J, 3]
    raw_pc = raw_pc[:t_his]                      # [t_his, 5000, 6]
    total_k = t_his * top_k

    # 2. Flatten joints (excluding root)
    pose_points = pose_seq[:, 1:, :].reshape(-1, 3)  # [t_his*(J-1), 3]

    # 3. Flatten raw point cloud and filter out all-zero 6D
    raw_pc_flat = raw_pc.reshape(-1, 6)              # [t_his*5000, 6]
    valid_mask = ~(raw_pc_flat == 0).all(dim=1)
    valid_pc = raw_pc_flat[valid_mask]               # [N_valid, 6]
    xyz = valid_pc[:, :3]                            # [N_valid, 3]

    if xyz.shape[0] == 0:
        result = torch.zeros((total_k, 7), dtype=torch.float32)
        return result.numpy() if input_is_numpy else result

    # 4. Compute distance from each point to joints
    dist = torch.cdist(xyz, pose_points, p=2)        # [N_valid, t_his*(J-1)]
    min_dist, _ = dist.min(dim=1)                    # [N_valid]

    # 5. Select top_k points (or all)
    selected_k = min(total_k, min_dist.shape[0])
    topk_val, topk_idx = torch.topk(-min_dist, selected_k)  # negative for ascending
    selected = valid_pc[topk_idx]                    # [<=total_k, 6]

    # 6. Recover time index
    original_indices = torch.nonzero(valid_mask, as_tuple=False).squeeze(1)
    frame_idx = original_indices[topk_idx] // 5000   # [selected_k]
    frame_idx = frame_idx.float().unsqueeze(1)

    # 7. Compose output [selected_k, 7]
    out = torch.cat([selected[:, :3], frame_idx, selected[:, 3:]], dim=1)

    # 8. Pad with zeros if needed
    if selected_k < total_k:
        pad = torch.zeros((total_k - selected_k, 7), dtype=out.dtype, device=out.device)
        out = torch.cat([out, pad], dim=0)

    return out.numpy() if input_is_numpy else out


def knn_outlier_zero_filter(data, k=10, distance_percentile=95):
    """
    Remove outliers from non-zero point cloud data using KNN per frame,
    while preserving original shape and restoring 0-points at original locations.

    Args:
        data: ndarray of shape (T, N, 6), where first 3 dims are (x, y, z)
        k: number of neighbors to use in KNN (default 10)
        distance_percentile: percentile threshold for filtering out outliers

    Returns:
        data_filtered: ndarray of same shape (T, N, 6) with KNN outliers + original zeros set to 0s
    """
    T, N, D = data.shape
    assert D == 6, "Expected input shape (T, N, 6)"

    data_filtered = data.copy()

    for t in range(T):
        frame = data[t]  # shape (N, 6)

        # Find non-zero points (assume zeros indicate already-removed or invalid points)
        nonzero_mask = ~(np.all(frame == 0, axis=-1))  # shape (N,)
        valid_points = frame[nonzero_mask]  # shape (M, 6) where M <= N

        if valid_points.shape[0] < k + 1:
            # Not enough points to apply KNN
            continue

        xyz = valid_points[:, :3]

        # KNN on non-zero valid points
        nbrs = NearestNeighbors(n_neighbors=k + 1).fit(xyz)
        distances, _ = nbrs.kneighbors(xyz)
        kth_dist = distances[:, -1]

        threshold = np.percentile(kth_dist, distance_percentile)
        inlier_mask = kth_dist <= threshold  # shape (M,)

        # Convert outliers to zero
        valid_points[~inlier_mask] = 0.0

        # Restore into full frame
        filtered_frame = np.zeros_like(frame)
        filtered_frame[nonzero_mask] = valid_points
        data_filtered[t] = filtered_frame

    return data_filtered


def knn_outlier_aggressive_filter(data, k=10, distance_percentile=95, worst_outlier_thresh=0.0):
    """
    Two-step aggressive KNN-based outlier filtering:
    1. Apply percentile-based mask to mark initial outliers.
    2. Further filter remaining points with kth-distance <= worst_outlier_thresh.

    Args:
        data: ndarray of shape (T, N, 6)
        k: number of neighbors
        distance_percentile: percentile threshold for initial outlier detection
        worst_outlier_thresh: further filter any remaining points with distance <= this

    Returns:
        data_filtered: ndarray of same shape (T, N, 6), with outliers zeroed
    """
    T, N, D = data.shape
    assert D == 6, "Expected input shape (T, N, 6)"

    data_filtered = data.copy()

    for t in range(T):
        frame = data[t]  # (N, 6)
        nonzero_mask = ~(np.all(frame == 0, axis=-1))  # keep only valid points
        valid_points = frame[nonzero_mask]  # shape (M, 6)

        if valid_points.shape[0] < k + 1:
            continue  # skip if too few points

        xyz = valid_points[:, :3]

        # Step 1: compute k-th nearest neighbor distances
        nbrs = NearestNeighbors(n_neighbors=k + 1).fit(xyz)
        distances, _ = nbrs.kneighbors(xyz)
        kth_dist = distances[:, -1]

        # Step 2: initial outlier mask by percentile
        initial_outlier_mask = kth_dist > np.percentile(kth_dist, distance_percentile)

        # Step 3: aggressive mask: among remaining inliers, if distance <= worst_outlier_thresh → outlier
        remaining_mask = ~initial_outlier_mask
        aggressive_mask = np.zeros_like(kth_dist, dtype=bool)
        aggressive_mask[remaining_mask] = kth_dist[remaining_mask] <= worst_outlier_thresh

        # Final outlier mask = initial + aggressive
        final_outlier_mask = initial_outlier_mask | aggressive_mask

        # Zero out final outliers
        valid_points[final_outlier_mask] = 0.0

        # Reinsert into full frame
        filtered_frame = np.zeros_like(frame)
        filtered_frame[nonzero_mask] = valid_points
        data_filtered[t] = filtered_frame

    return data_filtered




def dbscan_cluster_filter(
    data, min_samples = 5, eps=0.5, cluster_size_thresh=20,
    reduce_ratio=0.5, pad_ratio=1.0, dim_weight=[1.,1.,1.]
):
    """
    DBSCAN-based point filtering and reduction for shape (T, N, 6).
    Outputs (T, N', 7) where the last dimension includes cluster_id.

    Returns:
        output: ndarray of shape (T, N * pad_ratio, 7)
    """
    
    T, N, D = data.shape
    assert D == 6
    target_N = int(N * pad_ratio)
    output = np.zeros((T, target_N, 7), dtype=data.dtype)  # extra channel for cluster_id

    for t in range(T):
        frame = data[t]  # (N, 6)

        # Remove zero points
        valid_mask = ~(np.all(frame == 0, axis=-1))
        valid_points = frame[valid_mask]
        if len(valid_points) < min_samples:
            continue  # skip this frame

        xyz = valid_points[:, :3]
        xyz_weighted = xyz.copy()
        xyz_weighted[:, 0] *= dim_weight[0]  # Compress or stretch x
        xyz_weighted[:, 1] *= dim_weight[1]  # Compress or stretch x
        xyz_weighted[:, 2] *= dim_weight[2]  # Compress or stretch x
        # eps = eps / (3**0.5) *(dim_weight[0]**2 + dim_weight[0]**2 + dim_weight[0]**2)**0.5

        # DBSCAN
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(xyz_weighted)
        labels = clustering.labels_
        unique_labels = sorted(set(labels))

        # Step 1: Cluster sizes
        cluster_sizes = [
            np.sum(labels == label)
            for label in unique_labels if label != -1
        ]
        if len(cluster_sizes) == 0:
            continue

        dynamic_thresh = min(
            cluster_size_thresh,
            np.percentile(cluster_sizes, 8) - 1
        )

        # print(dynamic_thresh, len(valid_points), f"[T={t}] Cluster sizes:", cluster_sizes, sum(cluster_sizes))

        # Step 2: Filtering and reduction
        clustered_points = []
        for label in unique_labels:
            if label == -1:
                continue  # noise

            cluster_points = valid_points[labels == label]
            cluster_size = len(cluster_points)

            if cluster_size < dynamic_thresh:
                continue

            keep_n = min(cluster_size, max(int(cluster_size * reduce_ratio), cluster_size_thresh))
            selected_indices = np.random.choice(cluster_size, keep_n, replace=False)
            selected = cluster_points[selected_indices]
            selected_cluster_ids = np.full((keep_n, 1), label, dtype=data.dtype)
            selected_with_label = np.concatenate([selected, selected_cluster_ids], axis=1)  # (keep_n, 7)
            clustered_points.append(selected_with_label)

        # Step 3: Pad/truncate
        if clustered_points:
            filtered = np.concatenate(clustered_points, axis=0)
        else:
            filtered = np.zeros((0, 7), dtype=data.dtype)

        if len(filtered) >= target_N:
            output[t] = filtered[:target_N]
        else:
            padded = np.zeros((target_N, 7), dtype=data.dtype)
            padded[:len(filtered)] = filtered
            output[t] = padded

    return output


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, cm

def plot_pose_sequence_as_gif_selected(
    gt, pred,
    selected_raw_gt=None, selected_raw_pred=None,
    batch_idx=0, t_his=25,
    out_path='pose_comparison.gif',
    elev=30, azim=120
):
    pred = pred.detach().cpu().numpy() if hasattr(pred, "detach") else pred
    gt = gt.detach().cpu().numpy() if hasattr(gt, "detach") else gt
    if selected_raw_gt is not None:
        selected_raw_gt = selected_raw_gt.detach().cpu().numpy() if hasattr(selected_raw_gt, "detach") else selected_raw_gt
    if selected_raw_pred is not None:
        selected_raw_pred = selected_raw_pred.detach().cpu().numpy() if hasattr(selected_raw_pred, "detach") else selected_raw_pred

    B, T, D = pred.shape
    J = D // 3

    # Skeleton edges
    adj = np.zeros((J + 1, J + 1))
    edges = [
        (0, 1), (1, 2), (2, 3),           # Right leg
        (0, 4), (4, 5), (5, 6),           # Left leg
        (0, 7), (7, 8), (8, 9), (9, 10),  # Spine and head
        (8, 11), (11, 12), (12, 13),      # Left arm
        (8, 14), (14, 15), (15, 16)       # Right arm
    ]
    for i, j in edges:
        adj[i, j] = adj[j, i] = 1
    edges = np.array(np.nonzero(adj)).T

    ade = np.mean(np.linalg.norm(pred[batch_idx, t_his:] - gt[batch_idx, t_his:], axis=-1))

    fig = plt.figure(figsize=(12, 6))
    ax_gt = fig.add_subplot(121, projection='3d')
    ax_pred = fig.add_subplot(122, projection='3d')
    fig.suptitle(f"GT vs Prediction (ADE = {ade:.4f})")

    def update(frame_idx):
        ax_gt.cla()
        ax_pred.cla()

        gt_frame = gt[batch_idx, frame_idx].reshape(J, 3)
        pred_frame = pred[batch_idx, frame_idx].reshape(J, 3)

        gt_frame = np.concatenate([np.zeros_like(gt_frame[[0]]), gt_frame], axis=0)
        pred_frame = np.concatenate([np.zeros_like(pred_frame[[0]]), pred_frame], axis=0)

        color_gt = 'green' if frame_idx < t_his else 'blue'
        color_pred = 'gray' if frame_idx < t_his else 'red'

        # GT pose
        ax_gt.set_title('Ground Truth')
        ax_gt.scatter(*gt_frame.T, c=color_gt, s=20)
        for i, j in edges:
            ax_gt.plot([gt_frame[i, 0], gt_frame[j, 0]],
                       [gt_frame[i, 1], gt_frame[j, 1]],
                       [gt_frame[i, 2], gt_frame[j, 2]], c=color_gt)

        # Predicted pose
        ax_pred.set_title('Prediction')
        ax_pred.scatter(*pred_frame.T, c=color_pred, s=20)
        for i, j in edges:
            ax_pred.plot([pred_frame[i, 0], pred_frame[j, 0]],
                         [pred_frame[i, 1], pred_frame[j, 1]],
                         [pred_frame[i, 2], pred_frame[j, 2]], c=color_pred)

        def plot_pc_with_optional_flow(ax, pc_data):
            if pc_data.ndim == 4 and pc_data.shape[-1] == 6:
                points = pc_data[batch_idx, frame_idx, :, :3]
                ax.scatter(*points.T, c='black', s=2, alpha=0.8)
            elif pc_data.ndim == 4 and pc_data.shape[-1] == 7:
                pc = pc_data[batch_idx, frame_idx]
                xyz = pc[:, :3]
                cluster_ids = pc[:, 6].astype(int)
                colors = cm.tab20(cluster_ids % 20)
                ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=colors, s=2, alpha=0.8)
            elif pc_data.ndim == 4 and pc_data.shape[-1] == 9:
                pc = pc_data[batch_idx, frame_idx]
                xyz = pc[:, :3]
                flow = pc[:, 6:9]
                ax.quiver(xyz[:, 0], xyz[:, 1], xyz[:, 2],
                          flow[:, 0], flow[:, 1], flow[:, 2],
                          length=1.0, normalize=False, color='black', linewidth=0.5)
            elif pc_data.ndim == 3:
                pc = pc_data[batch_idx]
                pc = pc[pc[:, 3] == frame_idx]
                xyz = pc[:, :3]
                cluster_ids = pc[:, 6].astype(int)
                colors = cm.tab20(cluster_ids % 20)
                ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=colors, s=2, alpha=0.8)

        if selected_raw_gt is not None:
            plot_pc_with_optional_flow(ax_gt, selected_raw_gt)
        if selected_raw_pred is not None:
            plot_pc_with_optional_flow(ax_pred, selected_raw_pred)

        for ax in [ax_gt, ax_pred]:
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_zlim(-1, 1)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            # ax.view_init(elev=elev, azim=azim)

    ani = animation.FuncAnimation(fig, update, frames=T, interval=100)
    ani.save(out_path, writer='pillow')
    plt.close()
    print(f"Saved GIF to {out_path} with ADE: {ade:.4f}")

