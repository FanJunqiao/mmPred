import torch
import numpy as np
import matplotlib.pyplot as plt
import imageio
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D



def visualize_radar_gt_pred_with_uncertainty(
    radar, gt, pred, pred_var,
    batch_idx=0,
    save_path='radar_gt_pred_uncertainty.png'
):
    device = radar.device
    radar_np = radar[batch_idx, :, :, :3].reshape(-1, 3).detach().cpu().numpy()

    J_minus1 = gt.shape[1] // 3
    J = J_minus1 + 1

    gt_full = torch.cat([
        torch.zeros((1, 3), device=device),
        gt[batch_idx].reshape(J_minus1, 3)
    ], dim=0).detach().cpu().numpy()

    pred_full = torch.cat([
        torch.zeros((1, 3), device=device),
        pred[batch_idx].reshape(J_minus1, 3)
    ], dim=0).detach().cpu().numpy()

    pred_var_full = torch.cat([
        torch.zeros((1, 3), device=device),
        pred_var[batch_idx].reshape(J_minus1, 3)
    ], dim=0).detach().cpu().numpy()

    edges = [
        (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6),
        (0, 7), (7, 8), (8, 9), (9, 10),
        (8, 11), (11, 12), (12, 13),
        (8, 14), (14, 15), (15, 16)
    ]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # --- Radar all T frames ---
    ax.scatter(radar_np[:, 0], radar_np[:, 1], radar_np[:, 2], c='blue', s=2, alpha=0.5)

    # --- GT ---
    ax.scatter(gt_full[:, 0], gt_full[:, 1], gt_full[:, 2], c='green', s=20, label='GT')
    for i, j in edges:
        ax.plot([gt_full[i, 0], gt_full[j, 0]],
                [gt_full[i, 1], gt_full[j, 1]],
                [gt_full[i, 2], gt_full[j, 2]],
                c='green')

    # --- Pred ---
    ax.scatter(pred_full[:, 0], pred_full[:, 1], pred_full[:, 2], c='red', s=20, label='Pred')
    for i, j in edges:
        ax.plot([pred_full[i, 0], pred_full[j, 0]],
                [pred_full[i, 1], pred_full[j, 1]],
                [pred_full[i, 2], pred_full[j, 2]],
                c='red')

    # --- Variance ellipsoids ---
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:20j]
    for j in range(J):
        center = pred_full[j]
        var_xyz = pred_var_full[j]
        scale_x = np.sqrt(var_xyz[0])# * 0.1
        scale_y = np.sqrt(var_xyz[1])# * 0.1
        scale_z = np.sqrt(var_xyz[2])# * 0.1

        if (scale_x > 0) and (scale_y > 0) and (scale_z > 0):
            x = scale_x * np.cos(u) * np.sin(v) + center[0]
            y = scale_y * np.sin(u) * np.sin(v) + center[1]
            z = scale_z * np.cos(v) + center[2]
            ax.plot_surface(
                x, y, z,
                color='red',
                alpha=0.2,
                linewidth=0,
                antialiased=True
            )

    ax.view_init(elev=0, azim=-0)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved 3D GT+Pred+Radar+Var-Ellipsoids plot to: {save_path}")


def visualize_radar_gt_pred_pose(
    radar: torch.Tensor,             # [B, T, N, C], radar point cloud (CUDA)
    gt_dct: torch.Tensor,            # [B, D, 16*3], GT pose in DCT domain
    pose_pr: torch.Tensor,           # [B, D, 16*3], predicted pose in DCT domain
    save_path: str = 'radar_pose_compare.gif',
    batch_idx: int = 0
):
    assert radar.device == gt_dct.device == pose_pr.device
    device = radar.device
    B, T, N, _ = radar.shape
    D = gt_dct.shape[1]
    J = 16
    
    # Calculate MSE difference between GT DCT and predicted DCT
    import torch.nn as nn
    mse_loss = nn.MSELoss()
    mse_diff = mse_loss(gt_dct, pose_pr).item() * 100
    print(f"MSE between GT DCT and predicted DCT: {mse_diff:.6f}")
    
    pose_pr = pose_pr

    # === Create IDCT matrix ===
    k = torch.arange(D, device=device).float()
    t = torch.arange(T, device=device).float().unsqueeze(1)
    idct_mat = torch.cos(torch.pi * (2 * t + 1) * k / (2 * T))  # [T, D]
    idct_mat[:, 0] *= 1 / torch.sqrt(torch.tensor(2.0, device=device))
    idct_mat *= torch.sqrt(torch.tensor(2.0 / T, device=device))

    # === Apply IDCT ===
    gt_pose = torch.matmul(idct_mat, gt_dct).view(B, T, J, 3)        # [B, T, 16, 3]
    pr_pose = torch.matmul(idct_mat, pose_pr).view(B, T, J, 3)       # [B, T, 16, 3]

    # === Add pelvis (joint 0)
    pelvis = torch.zeros((B, T, 1, 3), device=device)
    gt_pose = torch.cat([pelvis, gt_pose], dim=2)  # [B, T, 17, 3]
    pr_pose = torch.cat([pelvis, pr_pose], dim=2)  # [B, T, 17, 3]

    radar_xyz = radar[..., :3]  # [B, T, N, 3]

    # === Skeleton edges
    EDGES = [
        (0, 1), (1, 2), (2, 3),           # Right leg
        (0, 4), (4, 5), (5, 6),           # Left leg
        (0, 7), (7, 8), (8, 9), (9, 10),  # Spine and head
        (8, 11), (11, 12), (12, 13),      # Left arm
        (8, 14), (14, 15), (15, 16)       # Right arm
    ]

    frames = []
    for t in tqdm(range(T), desc='Rendering frames'):
        fig = plt.figure(figsize=(12, 6))

        for i, (title, pose) in enumerate([("GT Pose", gt_pose), ("Predicted Pose", pr_pose)]):
            ax = fig.add_subplot(1, 2, i + 1, projection='3d')

            pc = radar_xyz[batch_idx, t].detach().cpu().numpy()
            joints = pose[batch_idx, t].detach().cpu().numpy()

            ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], c='blue', s=3, alpha=0.5)
            ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], c='red', s=20)
            for j1, j2 in EDGES:
                ax.plot([joints[j1, 0], joints[j2, 0]],
                        [joints[j1, 1], joints[j2, 1]],
                        [joints[j1, 2], joints[j2, 2]], c='black')

            ax.set_title(f'{title} - Frame {t}')
            ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(0, 2)
            ax.view_init(elev=20, azim=45)
            ax.axis('off')

        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(image)
        plt.close(fig)

    imageio.mimsave(save_path, frames, duration=0.1)
    print(f'Saved GT vs Prediction GIF to: {save_path}')


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_points_and_anchors_with_circles(points, anchors, radius=0.0, batch_idx=0, save_path="points_and_anchors_3d.png"):
    """
    Args:
        points: [B, N, 3] - input points
        anchors: [B, N', 3] - anchor points
        radius: float - radius of circles on XY plane
        batch_idx: int - batch index to visualize
        save_path: str - path to save the image
    """
    pts = points[batch_idx]    # [N, 3]
    anc = anchors[batch_idx]   # [N', 3]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot input points (black) and anchors (red)
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c='black', s=1, alpha=0.5, label='Points')
    ax.scatter(anc[:, 0], anc[:, 1], anc[:, 2], c='red', s=20, label='Anchors')

    # Draw XY-plane circles around each anchor
    theta = np.linspace(0, 2 * np.pi, 100)
    for a in anc:
        x_circle = a[0] + radius * np.cos(theta)
        y_circle = a[1] + radius * np.sin(theta)
        z_circle = np.full_like(theta, a[2])  # flat circle at anchor's Z
        ax.plot(x_circle, y_circle, z_circle, c='red', alpha=0.6, linewidth=1)

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"3D Points and Anchors (Batch {batch_idx})")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ Saved to {save_path}")
