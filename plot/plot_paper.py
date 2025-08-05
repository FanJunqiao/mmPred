import matplotlib.pyplot as plt
import numpy as np
import torch
from utils.script import sample_preprocessing
from utils import post_process
import os
import matplotlib.patches as patches


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches



def compute_ade(pred, gt, t_start):
    """
    pred, gt: (T, J, 3)
    Computes ADE = mean L2 over joints and time (from t_start to end)
    """
    assert pred.shape == gt.shape
    diff = pred[t_start:] - gt[t_start:]  # (T - t_start, J, 3)
    ade = np.linalg.norm(diff, axis=-1).mean()
    return ade

def render_3d_pose(ax, joints, parent=None, elev=0, azim=135, color=None, alpha=1.0, linewidth=2.0):
    J = joints.shape[0]
    joints = joints - joints[0:1]  # root-centered

    if parent is None:
        if J == 15:
            parent = [-1, 0, 1, 2, 3, 1, 5, 6, 0, 8, 9, 0, 11, 12, 1]
        elif J == 17:
            parent = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]
        elif J == 22:
            parent = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
        else:
            raise ValueError(f"Unsupported joint count: {J}")

    # Coloring rules for default skeleton
    left_joints = {4, 5, 6, 11, 12, 13}
    right_joints = {1, 2, 3, 14, 15, 16}

    for j in range(J):
        i = parent[j]
        if i == -1:
            continue

        if color is not None:
            line_color = color
        elif j in left_joints and i in left_joints:
            line_color = 'orange'
        elif j in right_joints and i in right_joints:
            line_color = 'green'
        else:
            line_color = 'blue'

        x1, y1, z1 = joints[i]
        x2, y2, z2 = joints[j]
        ax.plot([x1, x2], [y1, y2], [z1, z2], lw=linewidth, c=line_color, alpha=alpha)

    ax.set_xlim([-0.3, 0.3])
    ax.set_ylim([-0.3, 0.3])
    ax.set_zlim([-0.8, 0.8])
    ax.set_box_aspect([0.6, 1.0, 1.8])
    ax.view_init(elev=elev, azim=azim)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])

def plot_best_hypothesis_from_generator(
    generator,
    save_path="best_motion.png",
    frame_indices=[0, 3, 7, 11, 15, 19, 23],
    max_hypothesis=10,
    t_his=8,
    gt_flag = True,
    pred_flag = True,
    elev = 0,
    azim = -0
):
    """
    For one sample from generator, computes ADE for each hypothesis vs gt,
    selects best one, and overlays GT in red.
    """
    # === 1. Load one sample ===
    sample = next(generator)

    gt = sample['gt']         # (T, J, 3)
    t_total, J, _ = gt.shape

    best_id = -1
    best_ade = float("inf")
    best_hyp = None

    # === 2. Loop over hypotheses ===
    for i in range(max_hypothesis):
        key = f"mmPred_{i}"
        if key not in sample:
            continue
        pred = sample[key]
        ade = compute_ade(pred, gt, t_start=t_his)

        if ade < best_ade:
            best_ade = ade
            best_id = i
            best_hyp = pred

    print(f"[INFO] Best hypothesis: mmPred_{best_id}, ADE = {best_ade:.4f}")

    # === 3. Plot selected frames ===
    fig = plt.figure(figsize=(len(frame_indices) * 2, 3))
    axs = [fig.add_subplot(1, len(frame_indices), i + 1, projection='3d') for i in range(len(frame_indices))]

    for i, t in enumerate(frame_indices):
        ax = axs[i]
        ax.axis("off")

        # Plot GT (in red, faint)
        if gt_flag:
            alpha = 0.8 if pred_flag==False else 0.4
            color = "blue" if pred_flag==False else "red"
            render_3d_pose(ax, gt[t], color=color, alpha=alpha, linewidth=1.5, elev=elev, azim=azim)

        # Plot best prediction (default colors)
        if pred_flag:
            render_3d_pose(ax, best_hyp[t], elev=elev, azim=azim)

        # if i == 0:
        #     # Add legend to first subplot
        #     pred_patch = mpatches.Patch(color='blue', label='Prediction')
        #     gt_patch = mpatches.Patch(color='red', label='Ground Truth')
        #     ax.legend(handles=[pred_patch, gt_patch], loc='upper left')

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.0, hspace=0.0)
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()

    print(f"[INFO] Saved best hypothesis plot to: {save_path}")


def plot_best_hypothesis_overlay(
    generator,
    save_path="best_motion_overlay_alpha.png",
    frame_indices=[0, 3, 7, 11, 15, 19, 23],
    max_hypothesis=10,
    t_his=8,
    gt_flag=False,
    pred_flag=True,
    elev=20,
    azim=40,
    alpha_range=(0.2, 1.0),
    radar_pc=None,
):
    """
    Saves:
    (1) One full prediction overlay plot (PNG)
    (2) Three individual history frame plots (SVG), with optional radar PC overlay
    """
    sample = next(generator)
    gt = sample['gt']
    t_total, J, _ = gt.shape

    best_id = -1
    best_ade = float("inf")
    best_hyp = None

    for i in range(max_hypothesis):
        key = f"mmPred_{i}"
        if key not in sample:
            continue
        pred = sample[key]
        ade = compute_ade(pred, gt, t_start=t_his)
        if ade < best_ade:
            best_ade = ade
            best_id = i
            best_hyp = pred

    print(f"[INFO] Best hypothesis: mmPred_{best_id}, ADE = {best_ade:.4f}")

    # === (1) Full prediction overlay ===
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111, projection='3d')
    ax.axis("off")

    n_frames = len(frame_indices)
    min_alpha, max_alpha = alpha_range

    for i, t in enumerate(frame_indices):
        alpha = min_alpha if i < 3 else min_alpha + (max_alpha - min_alpha) * ((i - 3) / (n_frames - 4))

        if gt_flag:
            render_3d_pose(ax, gt[t], color="red", alpha=alpha,
                           linewidth=1.5, elev=elev, azim=azim)
        if pred_flag:
            render_3d_pose(ax, best_hyp[t], color=None, alpha=alpha,
                           linewidth=2.0, elev=elev, azim=azim)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"[INFO] Saved full prediction overlay to: {save_path}")

    # === (2) Three individual history plots ===
    hist_indices = frame_indices[:3]
    hist_alphas = [0.3, 0.5, 0.7]

    for i, t in enumerate(hist_indices):
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111, projection='3d')
        ax.axis("off")

        alpha = hist_alphas[i]

        if gt_flag:
            render_3d_pose(ax, gt[t], color="red", alpha=alpha,
                           linewidth=1.5, elev=elev, azim=azim)
        if pred_flag:
            render_3d_pose(ax, best_hyp[t], color=None, alpha=alpha,
                           linewidth=2.0, elev=elev, azim=azim)

        # === (Optional) Plot radar point cloud ===
        if radar_pc is not None:
            if isinstance(radar_pc, torch.Tensor):
                radar_pc_np = radar_pc.detach().cpu().numpy()
            else:
                radar_pc_np = radar_pc
            if t < radar_pc_np.shape[0]:
                pc_xyz = radar_pc_np[t, :, :3]  # (N, 3)
                ax.scatter(pc_xyz[:, 0], pc_xyz[:, 1], pc_xyz[:, 2],
                           c='grey', s=1.5, alpha=0.8, depthshade=False)

        save_hist_svg = save_path.replace(".png", f"_hist_frame{t}.svg")
        plt.tight_layout()
        plt.savefig(save_hist_svg, bbox_inches="tight", format='svg')
        plt.close()
        print(f"[INFO] Saved history frame {t} to: {save_hist_svg}")
        

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_best_hypothesis_overlay1(
    generator,
    save_path="best_motion_overlay_alpha.png",
    frame_indices=[0, 3, 7, 11, 15, 19, 23],
    max_hypothesis=10,
    t_his=8,
    gt_flag=False,
    pred_flag=True,
    elev=20,
    azim=40,
    radar_pc=None,
):
    """
    Saves:
    (1) One long prediction overlay plot (PNG), subplots per frame
    (2) Three individual history frame plots (SVG), with optional radar PC overlay
    """
    sample = next(generator)
    gt = sample['gt']
    t_total, J, _ = gt.shape

    best_id = -1
    best_ade = float("inf")
    best_hyp = None

    for i in range(max_hypothesis):
        key = f"mmPred_{i}"
        if key not in sample:
            continue
        pred = sample[key]
        ade = compute_ade(pred, gt, t_start=t_his)
        if ade < best_ade:
            best_ade = ade
            best_id = i
            best_hyp = pred

    print(f"[INFO] Best hypothesis: mmPred_{best_id}, ADE = {best_ade:.4f}")

    # === (1) Full prediction overlay with subplots ===
    n_frames = len(frame_indices)
    fig = plt.figure(figsize=(4 * n_frames, 4))
    for i, t in enumerate(frame_indices):
        ax = fig.add_subplot(1, n_frames, i + 1, projection='3d')
        ax.axis("off")
        ax.set_title(f"Frame {t}", fontsize=10)

        if gt_flag:
            render_3d_pose(ax, gt[t], color="red", alpha=1.0,
                           linewidth=1.5, elev=elev, azim=azim)
        if pred_flag:
            render_3d_pose(ax, best_hyp[t], color=None, alpha=1.0,
                           linewidth=2.0, elev=elev, azim=azim)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"[INFO] Saved full prediction overlay to: {save_path}")

    # === (2) Three individual history plots ===
    hist_indices = frame_indices[:3]
    hist_alphas = [0.3, 0.5, 0.7]

    for i, t in enumerate(hist_indices):
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111, projection='3d')
        ax.axis("off")

        alpha = hist_alphas[i]

        if gt_flag:
            render_3d_pose(ax, gt[t], color="red", alpha=alpha,
                           linewidth=1.5, elev=elev, azim=azim)
        if pred_flag:
            render_3d_pose(ax, best_hyp[t], color=None, alpha=alpha,
                           linewidth=2.0, elev=elev, azim=azim)

        # === (Optional) Plot radar point cloud ===
        if radar_pc is not None:
            if isinstance(radar_pc, torch.Tensor):
                radar_pc_np = radar_pc.detach().cpu().numpy()
            else:
                radar_pc_np = radar_pc
            if t < radar_pc_np.shape[0]:
                pc_xyz = radar_pc_np[t, :, :3]  # (N, 3)
                ax.scatter(pc_xyz[:, 0], pc_xyz[:, 1], pc_xyz[:, 2],
                           c='grey', s=1.5, alpha=0.8, depthshade=False)

        save_hist_svg = save_path.replace(".svg", f"_hist_frame{t}.svg")
        plt.tight_layout()
        plt.savefig(save_hist_svg, bbox_inches="tight", format='svg')
        plt.close()
        print(f"[INFO] Saved history frame {t} to: {save_hist_svg}")

        
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def visualize_hypothesis_attn(attn_list_list, save_dir='attn_vis',
                               step_range=(0, -1), layer_range=(0, -1),
                               attn_value_range: tuple[float, float] = (0.0, 0.4)):
    """
    Visualize and save averaged attention matrices for each hypothesis.

    Parameters:
    - attn_list_list: List[List[Tensor]] where
        attn_list_list[i][j] has shape (B, J, J, H)
    - save_dir: output folder
    - step_range: (start, end) indices for diffusion steps (inclusive)
    - layer_range: (start, end) indices for layers (inclusive)
    """

    os.makedirs(save_dir, exist_ok=True)

    num_steps = len(attn_list_list)
    num_layers = len(attn_list_list[0])
    B, J, _, H = attn_list_list[0][0].shape

    # Normalize ranges
    step_start, step_end = step_range
    layer_start, layer_end = layer_range
    if step_end == -1: step_end = num_steps
    if layer_end == -1: layer_end = num_layers

    step_indices = range(step_start, step_end)
    layer_indices = range(layer_start, layer_end)

    # === 1. Accumulate attention ===
    attn_sum = np.zeros((B, J, J))
    count = 0

    for i in step_indices:
        for j in layer_indices:
            attn = attn_list_list[i][j]  # (B, J, J, H)
            attn_np = attn.detach().cpu().numpy() if hasattr(attn, 'detach') else attn
            attn_np = attn_np.mean(-1)  # (B, J, J)
            attn_sum += attn_np
            count += 1

    if count == 0:
        raise ValueError("No attention matrices selected for averaging.")

    # === 2. Average ===
    attn_avg = attn_sum / count  # (B, J, J)

    # === 3. Save attention heatmaps ===
    for b in range(B):
        plt.figure(figsize=(6, 5))
        vmin, vmax = attn_value_range
        sns.heatmap(attn_avg[b], cmap='viridis', square=True, vmin=vmin, vmax=vmax)
        plt.title(f'Hypothesis {b} Attention\nSteps {step_start}-{step_end-1}, Layers {layer_start}-{layer_end-1}')
        plt.xlabel('Key Joint')
        plt.ylabel('Query Joint')
        plt.tight_layout()
        filename = f'attn_hypo_{b}_s{step_start}-{step_end-1}_l{layer_start}-{layer_end-1}.png'
        plt.savefig(os.path.join(save_dir, filename), dpi=300)
        plt.close()

    print(f"Saved {B} attention maps to {save_dir}, averaged over {count} matrices.")


import torch
import matplotlib.pyplot as plt
import numpy as np

def dct_torch(x, T=None, norm='ortho'):
    """
    Computes DCT-II of x along the first dimension (time).
    x: Tensor of shape (T, D)
    T: length of DCT (if None, inferred from x.shape[0])
    norm: 'ortho' or None
    Returns: Tensor of shape (T, D)
    """
    if T is None:
        T = x.shape[0]
    device = x.device
    x = x.float()

    n = torch.arange(T, device=device).unsqueeze(0)
    k = torch.arange(T, device=device).unsqueeze(1)
    dct_mat = torch.cos(np.pi * (n + 0.5) * k / T)

    if norm == 'ortho':
        dct_mat[0] *= 1 / np.sqrt(T)
        dct_mat[1:] *= np.sqrt(2 / T)

    return torch.matmul(dct_mat, x)

def idct_torch(X, T=None, norm='ortho'):
    """
    Computes inverse DCT-II (i.e., DCT-III) of X along the first dimension (time).
    X: Tensor of shape (N, D) -- N is number of DCT coefficients to use (low-pass)
    T: output length (should match original time length)
    norm: 'ortho' or None
    Returns: Tensor of shape (T, D)
    """
    N = X.shape[0]
    if T is None:
        T = N
    device = X.device
    X = X.float()

    k = torch.arange(N, device=device).unsqueeze(0)
    n = torch.arange(T, device=device).unsqueeze(1)
    idct_mat = torch.cos(np.pi * k * (n + 0.5) / T)

    if norm == 'ortho':
        idct_mat[:, 0] *= 1 / np.sqrt(T)
        idct_mat[:, 1:] *= np.sqrt(2 / T)

    return torch.matmul(idct_mat, X)

def low_pass_dct_filter(motion: torch.Tensor, N: int, norm='ortho'):
    """
    Apply low-pass DCT filter on motion sequence using first N DCT components.
    
    Args:
        motion: Tensor of shape (T, J*3), on CUDA or CPU.
        N: Number of DCT coefficients to keep.
        norm: 'ortho' or None (should match dct_torch / idct_torch).
    
    Returns:
        filtered: Tensor of shape (T, J*3), same device and dtype.
    """
    T, D = motion.shape
    device = motion.device

    # === DCT-II ===
    motion_dct = dct_torch(motion, T=T, norm=norm)  # (T, D)

    # === Zero-out high frequency components ===
    motion_dct_lp = motion_dct.clone()
    motion_dct_lp[N:, :] = 0  # low-pass: keep first N coefficients

    # === IDCT-III ===
    motion_filtered = idct_torch(motion_dct_lp[:N], T=T, norm=norm)  # (T, D)

    return motion_filtered

def plot_dct_energy_heatmap(
    motion: torch.Tensor,
    N: int = 10,
    save_path: str = "dct_energy_heatmap.png",
    legend_range: tuple = None
):
    """
    motion: Tensor of shape (T, J*3), assumed on CUDA
    N: number of DCT components to keep
    save_path: output PNG file path
    legend_range: (vmin, vmax) for colorbar, optional
    """
    T, D = motion.shape
    J = D // 3

    # === DCT on CPU ===
    motion_cpu = motion.detach().cpu()  # (T, J*3)
    dct_coeff = dct_torch(motion_cpu, T=motion_cpu.shape[0], norm='ortho')  # (T, J*3)

    # === Keep first N DCT coefficients ===
    dct_n = dct_coeff[:N].view(N, J, 3)

    # === Energy and dB ===
    energy_abs = dct_n.abs().mean(dim=2)

    # === Plot heatmap ===
    plt.figure(figsize=(10, 6))
    vmin, vmax = (legend_range if legend_range is not None else (None, None))
    plt.imshow(energy_abs.numpy(), cmap='viridis', aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
    plt.colorbar(label='Abs Value')
    plt.xlabel("Joint Index")
    plt.ylabel("DCT Coefficient Index")
    plt.title("Log-Scale DCT Energy Heatmap (dB)")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_selected_poses(motion: torch.Tensor, save_path: str = "time_domain_poses.png", num_frames: int = 6, N: int = 10, elev=0, azim=135):
    """
    motion: (T, J*3), pelvis removed
    N: number of low-pass DCT components
    """
    
    # Apply low-pass DCT filter
    T, D = motion.shape
    J_no_pelvis = D // 3
    motion_np = motion.detach().cpu().numpy()
    
    # === DCT on CPU ===
    motion_cpu = motion.detach().cpu()  # (T, J*3)
    # dct_coeff = dct_torch(motion_cpu,T=motion_cpu.shape[0], norm='ortho')  # (T, J*3)
    # dct_n = dct_coeff[:N].view(N, J_no_pelvis, 3)
    # motion_filtered = idct_torch(motion_cpu,T=motion_cpu.shape[0], norm='ortho').reshape(T, J_no_pelvis, 3)
    motion_filtered = low_pass_dct_filter(motion_cpu,N=1, norm='ortho').reshape(T, J_no_pelvis, 3)
    
    

    

    # Pad pelvis as (0, 0, 0)
    padded = np.concatenate([np.zeros((T, 1, 3)), motion_filtered], axis=1)  # [T, J+1, 3]

    selected_idx = np.linspace(0, T-1, num_frames, dtype=int)

    fig = plt.figure(figsize=(num_frames * 3.2, 4.2))
    for i, idx in enumerate(selected_idx):
        ax = fig.add_subplot(1, num_frames, i + 1, projection='3d')
        render_3d_pose(ax, padded[idx], elev=elev, azim=azim)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    
def plot_dct_N_variants_multi_joints_flat(
    motion: torch.Tensor,       # (T, J*3)
    joint_ids: list = [0, 1, 2, 3],
    coord: int = 1,             # 0:x, 1:y, 2:z
    N_list=[1, 2, 3, 4],
    save_path="dct_N_multi_joint_flat.png"
):
    """
    Plot selected joint trajectories (coord only) with DCT smoothing for N in N_list.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    T, D = motion.shape
    J = D // 3
    device = motion.device
    x = np.arange(T)

    # === Prepare index mapping ===
    joint_dim_indices = [j * 3 + coord for j in joint_ids]

    # === Extract the original signals ===
    signal_orig = motion[:, joint_dim_indices]  # (T, len(joint_ids))
    signal_orig_np = signal_orig.cpu().numpy()
    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5']

    # === Plot ===
    plt.figure(figsize=(6, len(N_list) * 2.2))

    for i, N in enumerate(reversed(N_list)):
        dct_coeff = dct_torch(signal_orig, T=T)  # (T, len(joint_ids))
        dct_coeff_lp = dct_coeff.clone()
        dct_coeff_lp[N:, :] = 0
        signal_filtered = idct_torch(dct_coeff_lp[:N], T=T)  # (T, len(joint_ids))
        signal_filtered_np = signal_filtered.cpu().numpy()

        ax = plt.subplot(len(N_list), 1, i + 1)
        for j, jid in enumerate(joint_ids):
            ax.plot(x, signal_orig_np[:, j], linestyle='--', linewidth=1, alpha=0.6,
                    color=colors[j], label=f"original joint {jid+1}")
            ax.plot(x, signal_filtered_np[:, j], linestyle='-', linewidth=1.8,
                    color=colors[j], label=f"DCT joint {jid+1}")

        ax.set_ylim(-1.1, -0.1)
        ax.set_ylabel("position")
        ax.set_title(f"N={N} DCT Components")
        if i == len(N_list) - 1:
            ax.set_xlabel("frame")
        if i ==0:
            ax.legend(loc="upper right", ncol=2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

