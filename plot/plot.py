import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

def plot_tensor_to_gif(tensor, save_path='/dct_vis/output.gif'):
    """
    Plots a T, 17, 3 torch tensor using a connectivity graph and saves it as a GIF.

    Parameters:
    - tensor (torch.Tensor): A tensor of shape (T, 17, 3) on CUDA or CPU.
    - connectivity (list of tuple): List of joint connections [(start, end), ...].
    - save_path (str): Path to save the resulting GIF.

    Returns:
    - None
    """
    # Example connectivity (17 joints)
    connectivity = [[0,1], [1,2], [2,3],
                    [0,4], [4,5], [5, 6],
                    [0,7], [7, 8], [8, 9], [9, 10],
                    [8,11], [11,12], [12, 13],
                    [8,14], [14,15], [15,16]]

    if not isinstance(tensor, torch.Tensor):
        raise ValueError("Input must be a PyTorch tensor.")

    if tensor.shape[-2:] != (17, 3):
        raise ValueError("Tensor must have shape (T, 17, 3).")

    # Move the tensor to CPU and convert to NumPy
    tensor_cpu = tensor.cpu().numpy() if tensor.is_cuda else tensor.numpy()

    # Create a 3D animation plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Initialize plot elements
    lines = []
    points, = ax.plot([], [], [], 'o', markersize=4, color='blue')
    for (start, end) in connectivity:
        line, = ax.plot([], [], [], lw=2, color='black')
        lines.append(line)

    # Update function for each frame
    def update(frame):
        coords = tensor_cpu[frame]
        points.set_data(coords[:, 0], coords[:, 1])
        points.set_3d_properties(coords[:, 2])
        for idx, (start, end) in enumerate(connectivity):
            line_coords = coords[[start, end]]
            lines[idx].set_data(line_coords[:, 0], line_coords[:, 1])
            lines[idx].set_3d_properties(line_coords[:, 2])
        return lines + [points]

    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=tensor_cpu.shape[0], interval=50, blit=True)

    # Save animation as a GIF
    ani.save(save_path, writer='pillow', fps=20)
    print(f"Animation saved to {save_path}")


def plot_frequency_diagram(tensor, save_path='/dct_vis/frequency_diagram.png'):
    """
    Plots a frequency diagram for a PyTorch tensor of shape (F, C) and saves it as a PNG.

    Parameters:
    - tensor (torch.Tensor): A PyTorch tensor of shape (F, C).
    - save_path (str): Path to save the resulting PNG file.

    Returns:
    - None
    """
    if not isinstance(tensor, torch.Tensor):
        raise ValueError("Input must be a PyTorch tensor.")

    if tensor.ndimension() != 2:
        raise ValueError("Tensor must have 2 dimensions (F, C).")

    # Move tensor to CPU and convert to NumPy for plotting
    tensor_cpu = tensor.cpu().numpy() if tensor.is_cuda else tensor.numpy()

    # Get the shape
    F, C = tensor_cpu.shape

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.imshow(tensor_cpu, aspect='auto', cmap='viridis', origin='lower')
    plt.colorbar(label='Magnitude')
    plt.title(f'Frequency Diagram (Frequencies: {F}, Channels: {C})')
    plt.xlabel('Channel Index')
    plt.ylabel('Frequency Index')
    plt.xticks(range(C))
    plt.yticks(range(F))
    
    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Frequency diagram saved to {save_path}")

def dct_test(traj_plot, traj_plot_dct, traj_plot_noise, traj_plot_dct_noise, traj_plot_dct_noise_after, index):
    plot_tensor_to_gif(traj_plot[index], save_path=f'traj_plot{index}.gif')
    plot_tensor_to_gif(traj_plot_noise[index], save_path=f'traj_plot_noise{index}.gif')
    plot_frequency_diagram(traj_plot_dct[index], save_path=f'traj_plot_dct{index}.png')
    plot_frequency_diagram(traj_plot_dct_noise[index], save_path=f'traj_plot_dct_noise{index}.png')
    plot_frequency_diagram(traj_plot_dct_noise_after[index], save_path=f'traj_plot_dct_noise_after{index}.png')


def dct_plot(traj, traj_pad, t, diffusion, cfg, index=8, t_def = 20, secondcol = 0):
    # dct_plot(traj, traj_pad, t, self.diffusion, self.cfg, index=8, t_def = 20, secondcol = 0)
    t_my = torch.ones_like(t) * t_def
    x_t_my , _  = diffusion.noise_motion(traj, traj, t_my)
    traj_plot = torch.cat([torch.zeros(traj.shape[0],traj.shape[1],3).cuda(), traj], dim=-1)
    traj_plot_noise = torch.cat([torch.zeros(traj.shape[0],traj.shape[1],3).cuda(), x_t_my], dim=-1)
    traj_plot_dct = torch.matmul(cfg.dct_m_all[:cfg.n_pre], traj)
    traj_plot_dct_noise = torch.matmul(cfg.dct_m_all[:cfg.n_pre], x_t_my)
    traj_plot_dct_noise_after , _  = diffusion.noise_motion(traj_plot_dct, t_my)
    traj_plot_noise = traj_plot_noise.view(-1,traj_plot.shape[1], 17, 3)
    traj_plot = traj_plot.view(-1,traj_plot.shape[1], 17, 3)
    dct_test(traj_plot, traj_plot_dct, traj_plot_noise[:,secondcol:,:], traj_plot_dct_noise[:,secondcol:,:], traj_plot_dct_noise_after[:,secondcol:,:], index=index)

def plot_frequency_diagram(tensor, save_path='frequency_diagram.jpg', vmin = None, vmax = None):
    """
    Plots a frequency diagram for a PyTorch tensor of shape (F, C) and saves it as a PNG.

    Parameters:
    - tensor (torch.Tensor): A PyTorch tensor of shape (F, C).
    - save_path (str): Path to save the resulting PNG file.

    Returns:
    - None
    """
    if not isinstance(tensor, torch.Tensor):
        raise ValueError("Input must be a PyTorch tensor.")

    if tensor.ndimension() != 2:
        raise ValueError("Tensor must have 2 dimensions (F, C).")

    # Move tensor to CPU and convert to NumPy for plotting
    tensor_cpu = tensor.cpu().numpy() if tensor.is_cuda else tensor.numpy()

    # Get the shape
    F, C = tensor_cpu.shape

    # Plotting
    plt.figure(figsize=(12, 6))
    if vmin != None and vmax != None:
        plt.imshow(tensor_cpu, aspect='auto', cmap='viridis', origin='lower', vmin=vmin, vmax=vmax)
    else:
        plt.imshow(tensor_cpu, aspect='auto', cmap='viridis', origin='lower')
    plt.colorbar(label='Magnitude')
    plt.title(f'Frequency Diagram (Frequencies: {F}, Channels: {C})')
    plt.xlabel('Channel Index')
    plt.ylabel('Frequency Index')
    plt.xticks(range(C))
    plt.yticks(range(F))
    
    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Frequency diagram saved to {save_path}")

def plot_dct_energy_comparison(gt_dct, cond_dct, pred_dcts, save_path='energy_comparison.png'):
    """
    Given the ground truth, conditional, and predicted DCT tensors, compute their energy in dB 
    and plot the energy for each GT, conditional, and predicted tensors together. 
    Saves the plot as a PNG file.

    Parameters:
    - gt_dct (torch.Tensor): Ground truth DCT tensor of shape (N, C)
    - cond_dct (torch.Tensor): Conditional DCT tensor of shape (N, C)
    - pred_dcts (list of torch.Tensor): List of predicted DCT tensors, each of shape (N, C)
    - save_path (str): Path where the plot will be saved as a PNG file (default is 'energy_comparison.png')
    """
    # Ensure the tensors are float tensors for numerical stability in log operations
    gt_dct = gt_dct.float()
    cond_dct = cond_dct.float()
    pred_dcts = pred_dcts.float()  # Make sure predictions are float tensors
    
    # Compute energy in dB for each tensor (GT, cond, and each pred)
    energies_db = []
    
    # Compute energy for ground truth
    gt_energy = torch.sum(torch.abs(gt_dct)**2, dim=1)
    gt_energy_db = 10 * torch.log10(gt_energy)
    energies_db.append(gt_energy_db.numpy())
    
    # Compute energy for conditional
    cond_energy = torch.sum(torch.abs(cond_dct)**2, dim=1)
    cond_energy_db = 10 * torch.log10(cond_energy)
    energies_db.append(cond_energy_db.numpy())
    
    # Compute energy for each predicted tensor
    # for pred_dct in pred_dcts:
    pred_energy = torch.sum(torch.abs(pred_dcts)**2, dim=1)
    pred_energy_db = 10 * torch.log10(pred_energy)
    energies_db.append(pred_energy_db.numpy())
    
    # Plot the energy comparison
    plt.figure(figsize=(10, 6))
    
    # Plot ground truth, conditional, and each predicted energy
    labels = ['Ground Truth', 'Conditional'] + [f'Predicted {i+1}' for i in range(len(pred_dcts))]
    colors = ['b', 'g'] + ['r', 'c', 'm', 'y', 'k'][:len(pred_dcts)]  # You can extend this for more colors
    
    for i, energy_db in enumerate(energies_db):
        plt.plot(range(gt_dct.size(0)), energy_db, label=labels[i], color=colors[i], linestyle='--' if i > 1 else '-')
    
    plt.xlabel('N (Index)')
    plt.ylabel('Energy (dB)')
    plt.title('Energy Comparison: Ground Truth, Conditional, and Predicted DCT')
    plt.grid(True)
    plt.legend()
    
    # Save the plot as a PNG file
    plt.savefig(save_path, format='png')
    print(f"Plot saved as {save_path}")
    
    # Optionally, you can also display the plot
    # plt.show()


# def plot_pose_sequence_as_gif(gt, pred, batch_idx=0, t_his=25, out_path='pose_comparison.gif'):
#     """
#     Visualize GT vs Predicted 3D poses as a side-by-side gif, with historical frames highlighted,
#     and show average displacement error (ADE) in the title.
 
#     Args:
#         pred: Tensor of shape [B, T, 3*J], predicted pose (on CUDA)
#         gt: Tensor of shape [B, T, 3*J], ground truth pose (on CUDA)
#         batch_idx: Index of the sample in batch to visualize
#         t_his: Number of historical frames (0 ~ t_his are GT, t_his~ are predicted)
#         out_path: Path to save the gif
#     """
#     pred = pred.detach().cpu().numpy()
#     gt = gt.detach().cpu().numpy()

#     B, T, D = pred.shape
#     J = D // 3

#     # Define skeleton connections (H36M 17 joints)
#     adj = np.zeros((J+1, J+1))
#     edges = [
#         (0, 1), (1, 2), (2, 3),           # Right leg
#         (0, 4), (4, 5), (5, 6),           # Left leg
#         (0, 7), (7, 8), (8, 9), (9, 10),  # Spine and head
#         (8,11), (11,12), (12,13),         # Left arm
#         (8,14), (14,15), (15,16)          # Right arm
#     ]
#     for i, j in edges:
#         adj[i, j] = adj[j, i] = 1
#     edges = np.array(np.nonzero(adj)).T

#     # Compute Average Displacement Error (ADE) over prediction frames
#     ade = np.mean(np.linalg.norm(
#         pred[batch_idx, t_his:] - gt[batch_idx, t_his:], axis=-1))

#     fig = plt.figure(figsize=(10, 5))
#     ax_gt = fig.add_subplot(121, projection='3d')
#     ax_pred = fig.add_subplot(122, projection='3d')
#     fig.suptitle(f"GT vs Prediction (ADE = {ade:.4f})")

#     def update(frame_idx):
#         ax_gt.cla()
#         ax_pred.cla()

#         gt_frame = gt[batch_idx, frame_idx].reshape(J, 3)
#         pred_frame = pred[batch_idx, frame_idx].reshape(J, 3)

#         gt_frame = np.concatenate([np.zeros_like(gt_frame[[0]]), gt_frame], axis=0)
#         pred_frame = np.concatenate([np.zeros_like(pred_frame[[0]]), pred_frame], axis=0)

#         # Plot GT
#         ax_gt.set_title('Ground Truth')
#         color_gt = 'green' if frame_idx < t_his else 'blue'
#         ax_gt.scatter(*gt_frame.T, c=color_gt, s=20)
#         for i, j in edges:
#             ax_gt.plot([gt_frame[i, 0], gt_frame[j, 0]],
#                        [gt_frame[i, 1], gt_frame[j, 1]],
#                        [gt_frame[i, 2], gt_frame[j, 2]], c=color_gt)

#         # Plot Pred
#         ax_pred.set_title('Prediction')
#         color_pred = 'gray' if frame_idx < t_his else 'red'
#         ax_pred.scatter(*pred_frame.T, c=color_pred, s=20)
#         for i, j in edges:
#             ax_pred.plot([pred_frame[i, 0], pred_frame[j, 0]],
#                          [pred_frame[i, 1], pred_frame[j, 1]],
#                          [pred_frame[i, 2], pred_frame[j, 2]], c=color_pred)

#         for ax in [ax_gt, ax_pred]:
#             ax.set_xlim(-1, 1)
#             ax.set_ylim(-1, 1)
#             ax.set_zlim(-1, 1)
#             ax.set_xlabel('X')
#             ax.set_ylabel('Y')
#             ax.set_zlabel('Z')

#     ani = animation.FuncAnimation(fig, update, frames=T, interval=100)
#     ani.save(out_path, writer='pillow')
#     plt.close()
#     print(f"Saved GIF to {out_path} with ADE: {ade:.4f}")

def plot_pose_sequence_as_gif(gt, pred, raw_gt=None, raw_pred=None, batch_idx=0, t_his=25, out_path='pose_comparison.gif', point_fraction=0.1):
    """
    Visualize GT vs Predicted 3D poses and point cloud as a side-by-side gif.
    
    Args:
        gt: Tensor of shape [B, T, 3*J], ground truth pose (on CUDA)
        pred: Tensor of shape [B, T, 3*J], predicted pose (on CUDA)
        raw_gt: Tensor of shape [B, T, 5000, 6], ground truth point cloud (optional, on CUDA)
        raw_pred: Tensor of shape [B, T, 5000, 6], predicted point cloud (optional, on CUDA)
        batch_idx: Index of the sample in batch to visualize
        t_his: Number of historical frames (0 ~ t_his are GT, t_his~ are predicted)
        out_path: Path to save the gif
        point_fraction: Fraction of 5000 points to randomly visualize
    """
    pred = pred.detach().cpu().numpy()
    gt = gt.detach().cpu().numpy()

    if raw_gt is not None and raw_pred is not None:
        raw_gt = raw_gt.detach().cpu().numpy()
        raw_pred = raw_pred.detach().cpu().numpy()
        N_points = raw_gt.shape[2]
        sample_size = int(N_points * point_fraction)
        point_indices = np.random.choice(N_points, size=sample_size, replace=False)
    else:
        raw_gt = raw_pred = None

    B, T, D = pred.shape
    J = D // 3

    # Define skeleton connections (H36M 17 joints)
    adj = np.zeros((J+1, J+1))
    edges = [
        (0, 1), (1, 2), (2, 3),           # Right leg
        (0, 4), (4, 5), (5, 6),           # Left leg
        (0, 7), (7, 8), (8, 9), (9, 10),  # Spine and head
        (8,11), (11,12), (12,13),         # Left arm
        (8,14), (14,15), (15,16)          # Right arm
    ]
    for i, j in edges:
        adj[i, j] = adj[j, i] = 1
    edges = np.array(np.nonzero(adj)).T

    # Compute ADE
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

        # Pose GT
        ax_gt.set_title('Ground Truth')
        ax_gt.scatter(*gt_frame.T, c=color_gt, s=20)
        for i, j in edges:
            ax_gt.plot([gt_frame[i, 0], gt_frame[j, 0]],
                       [gt_frame[i, 1], gt_frame[j, 1]],
                       [gt_frame[i, 2], gt_frame[j, 2]], c=color_gt)

        # Pose Prediction
        ax_pred.set_title('Prediction')
        ax_pred.scatter(*pred_frame.T, c=color_pred, s=20)
        for i, j in edges:
            ax_pred.plot([pred_frame[i, 0], pred_frame[j, 0]],
                         [pred_frame[i, 1], pred_frame[j, 1]],
                         [pred_frame[i, 2], pred_frame[j, 2]], c=color_pred)

        # Optional Point Cloud
        if raw_gt is not None and raw_pred is not None:
            raw_gt_xyz = raw_gt[batch_idx, frame_idx, point_indices, :3]
            raw_pred_xyz = raw_pred[batch_idx, frame_idx, point_indices, :3]
            ax_gt.scatter(*raw_gt_xyz.T, c='black', s=2, alpha=0.8)
            ax_pred.scatter(*raw_pred_xyz.T, c='black', s=2, alpha=0.8)

        for ax in [ax_gt, ax_pred]:
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_zlim(-1, 1)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')

    ani = animation.FuncAnimation(fig, update, frames=T, interval=100)
    ani.save(out_path, writer='pillow')
    plt.close()
    print(f"Saved GIF to {out_path} with ADE: {ade:.4f}")




# def plot_pose_sequence_as_gif_selected(
#     gt, pred,
#     selected_raw_gt=None, selected_raw_pred=None,
#     batch_idx=0, t_his=25,
#     out_path='pose_comparison.gif'
# ):
#     """
#     Visualize GT vs Predicted 3D poses and selected point clouds as a side-by-side GIF.

#     Args:
#         gt: [B, T, 3*J] tensor or ndarray, ground truth pose
#         pred: [B, T, 3*J] tensor or ndarray, predicted pose
#         selected_raw_gt: [B, T, N, 6] or [B, N, 7], raw or selected GT point cloud
#         selected_raw_pred: [B, T, N, 6] or [B, N, 7], raw or selected Pred point cloud
#         batch_idx: Index of sample to show
#         t_his: Number of historical frames
#         out_path: Path to save GIF
#     """

#     # Convert torch -> numpy if needed
#     pred = pred.detach().cpu().numpy() if hasattr(pred, "detach") else pred
#     gt = gt.detach().cpu().numpy() if hasattr(gt, "detach") else gt
#     if selected_raw_gt is not None:
#         selected_raw_gt = selected_raw_gt.detach().cpu().numpy() if hasattr(selected_raw_gt, "detach") else selected_raw_gt
#     if selected_raw_pred is not None:
#         selected_raw_pred = selected_raw_pred.detach().cpu().numpy() if hasattr(selected_raw_pred, "detach") else selected_raw_pred

#     B, T, D = pred.shape
#     J = D // 3

#     # Define skeleton edges
#     adj = np.zeros((J+1, J+1))
#     edges = [
#         (0, 1), (1, 2), (2, 3),           # Right leg
#         (0, 4), (4, 5), (5, 6),           # Left leg
#         (0, 7), (7, 8), (8, 9), (9, 10),  # Spine and head
#         (8,11), (11,12), (12,13),         # Left arm
#         (8,14), (14,15), (15,16)          # Right arm
#     ]
#     for i, j in edges:
#         adj[i, j] = adj[j, i] = 1
#     edges = np.array(np.nonzero(adj)).T

#     ade = np.mean(np.linalg.norm(pred[batch_idx, t_his:] - gt[batch_idx, t_his:], axis=-1))

#     fig = plt.figure(figsize=(12, 6))
#     ax_gt = fig.add_subplot(121, projection='3d')
#     ax_pred = fig.add_subplot(122, projection='3d')
#     fig.suptitle(f"GT vs Prediction (ADE = {ade:.4f})")

#     def update(frame_idx):
#         ax_gt.cla()
#         ax_pred.cla()

#         gt_frame = gt[batch_idx, frame_idx].reshape(J, 3)
#         pred_frame = pred[batch_idx, frame_idx].reshape(J, 3)

#         gt_frame = np.concatenate([np.zeros_like(gt_frame[[0]]), gt_frame], axis=0)
#         pred_frame = np.concatenate([np.zeros_like(pred_frame[[0]]), pred_frame], axis=0)

#         color_gt = 'green' if frame_idx < t_his else 'blue'
#         color_pred = 'gray' if frame_idx < t_his else 'red'

#         # Plot pose
#         ax_gt.set_title('Ground Truth')
#         ax_gt.scatter(*gt_frame.T, c=color_gt, s=20)
#         for i, j in edges:
#             ax_gt.plot([gt_frame[i, 0], gt_frame[j, 0]],
#                        [gt_frame[i, 1], gt_frame[j, 1]],
#                        [gt_frame[i, 2], gt_frame[j, 2]], c=color_gt)

#         ax_pred.set_title('Prediction')
#         ax_pred.scatter(*pred_frame.T, c=color_pred, s=20)
#         for i, j in edges:
#             ax_pred.plot([pred_frame[i, 0], pred_frame[j, 0]],
#                          [pred_frame[i, 1], pred_frame[j, 1]],
#                          [pred_frame[i, 2], pred_frame[j, 2]], c=color_pred)

#         # Plot point cloud (support B, T, N, 6 or B, N, 7)
#         if selected_raw_gt is not None:
#             if selected_raw_gt.ndim == 4:  # B, T, N, 6
#                 points_gt = selected_raw_gt[batch_idx, frame_idx, :, :3]
#             elif selected_raw_gt.ndim == 3:  # B, N, 7
#                 points_gt = selected_raw_gt[batch_idx]
#                 points_gt = points_gt[points_gt[:, 3] == frame_idx][:, :3]
#             else:
#                 raise ValueError("selected_raw_gt must be [B,T,N,6] or [B,N,7]")
#             ax_gt.scatter(*points_gt.T, c='black', s=2, alpha=0.8)

#         if selected_raw_pred is not None:
#             if selected_raw_pred.ndim == 4:  # B, T, N, 6
#                 points_pred = selected_raw_pred[batch_idx, frame_idx, :, :3]
#             elif selected_raw_pred.ndim == 3:  # B, N, 7
#                 points_pred = selected_raw_pred[batch_idx]
#                 points_pred = points_pred[points_pred[:, 3] == frame_idx][:, :3]
#             else:
#                 raise ValueError("selected_raw_pred must be [B,T,N,6] or [B,N,7]")
#             ax_pred.scatter(*points_pred.T, c='black', s=2, alpha=0.8)

#         for ax in [ax_gt, ax_pred]:
#             ax.set_xlim(-1, 1)
#             ax.set_ylim(-1, 1)
#             ax.set_zlim(-1, 1)
#             ax.set_xlabel('X')
#             ax.set_ylabel('Y')
#             ax.set_zlabel('Z')

#     ani = animation.FuncAnimation(fig, update, frames=T, interval=100)
#     ani.save(out_path, writer='pillow')
#     plt.close()
#     print(f"Saved GIF to {out_path} with ADE: {ade:.4f}")




def plot_pose_hypotheses_as_gif(pred, t_his=25, out_path='pose_hypotheses.gif'):
    """
    Visualize K hypotheses of 3D poses as a side-by-side GIF (no GT, no padding, 17 joints assumed).

    Args:
        pred: Tensor of shape [K, T, J, 3], predicted hypotheses (J=17)
        t_his: Number of historical frames (for color coding)
        out_path: Path to save the gif
    """
    # pred = pred.detach().cpu().numpy()
    K, T, J, _ = pred.shape

    # Define skeleton edges for 17-joint H36M
    edges = [
        (0, 1), (1, 2), (2, 3),           # Right leg
        (0, 4), (4, 5), (5, 6),           # Left leg
        (0, 7), (7, 8), (8, 9), (9, 10),  # Spine and head
        (8,11), (11,12), (12,13),         # Left arm
        (8,14), (14,15), (15,16)          # Right arm
    ]

    fig = plt.figure(figsize=(5 * K, 5))
    axes = [fig.add_subplot(1, K, k + 1, projection='3d') for k in range(K)]

    def update(frame_idx):
        for k in range(K):
            ax = axes[k]
            ax.cla()

            pose = pred[k, frame_idx]  # [17, 3]
            color = 'gray' if frame_idx < t_his else 'red'

            ax.set_title(f'Hypothesis {k+1}')
            ax.scatter(*pose.T, c=color, s=20)
            for i, j in edges:
                ax.plot([pose[i, 0], pose[j, 0]],
                        [pose[i, 1], pose[j, 1]],
                        [pose[i, 2], pose[j, 2]], c=color)

            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_zlim(-1, 1)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')

    frame_indices = list(range(0, T, 2))  # downsample for 2x motion
    ani = animation.FuncAnimation(fig, update, frames=frame_indices, interval=100, repeat=False)
    ani.save(out_path, writer='pillow')
    plt.close()
    print(f"Saved GIF to {out_path}")

import matplotlib.pyplot as plt

def plot_and_save_histogram(data, x_min, x_max, bins=50, save_path='histogram.png', title='Histogram'):
    """
    Plot and save a histogram of the data with a custom X range.

    Parameters:
    - data: list or 1D np.ndarray of numerical values
    - x_min: minimum x-axis value
    - x_max: maximum x-axis value
    - bins: number of bins in the histogram (default: 50)
    - save_path: file path to save the plot (default: 'histogram.png')
    - title: title of the plot
    """
    plt.figure(figsize=(8, 5))
    plt.hist(data, bins=bins, range=(x_min, x_max), edgecolor='black', alpha=0.75)
    plt.xlim(x_min, x_max)
    plt.title(title)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()  # Close the figure to avoid display in notebooks


# Example usage
if __name__ == "__main__":
    traj_plot = torch.tensor(np.load("traj_plot.npy"))
    traj_plot_dct = torch.tensor(np.load("traj_plot_dct.npy"))
    traj_plot_noise = torch.tensor(np.load("traj_plot_noise.npy"))
    traj_plot_dct_noise = torch.tensor(np.load("traj_plot_dct_noise.npy"))
    traj_plot_dct_noise_after = torch.tensor(np.load("traj_plot_dct_noise_after.npy"))

    index = 4

    plot_tensor_to_gif(traj_plot[index], save_path=f'traj_plot{index}.gif')
    plot_tensor_to_gif(traj_plot_noise[index], save_path=f'traj_plot_noise{index}.gif')
    plot_frequency_diagram(traj_plot_dct[index], save_path=f'traj_plot_dct{index}.png')
    plot_frequency_diagram(traj_plot_dct_noise[index], save_path=f'traj_plot_dct_noise{index}.png')
    plot_frequency_diagram(traj_plot_dct_noise_after[index], save_path=f'traj_plot_dct_noise_after{index}.png')
