from scipy.spatial.distance import pdist
import numpy as np
import torch

"""metrics"""


def compute_all_metrics(pred, gt, gt_multi):
    """
    calculate all metrics

    Args:
        pred: candidate prediction, shape as [50, t_pred, 3 * joints_num]
        gt: ground truth, shape as [1, t_pred, 3 * joints_num]
        gt_multi: multi-modal ground truth, shape as [multi_modal, t_pred, 3 * joints_num]

    Returns:
        diversity, ade, fde, mmade, mmfde
    """
    if pred.shape[0] == 1:
        diversity = 0.0
    dist_diverse = torch.pdist(pred.reshape(pred.shape[0], -1))
    diversity = dist_diverse.mean()

    gt_multi = torch.from_numpy(gt_multi).to('cuda')
    gt_multi_gt = torch.cat([gt_multi, gt], dim=0)

    gt_multi_gt = gt_multi_gt[None, ...]
    pred = pred[:, None, ...]

    diff_multi = pred - gt_multi_gt
    dist = torch.linalg.norm(diff_multi, dim=3)
    # we can reuse 'dist' to optimize metrics calculation

    mmfde, _ = dist[:, :-1, -1].min(dim=0)
    # mmfde = dist[:, :-1, -1].mean(dim=0)
    mmfde = mmfde.mean()
    mmade, _ = dist[:, :-1].mean(dim=2).min(dim=0)
    # mmade = dist[:, :-1].mean(dim=2).mean(dim=0)
    mmade = mmade.mean()

    ade, _ = dist[:, -1].mean(dim=1).min(dim=0)
    fde, _ = dist[:, -1, -1].min(dim=0)
    # ade = dist[:, -1].mean(dim=1).mean(dim=0)
    # fde = dist[:, -1, -1].mean(dim=0)
    ade = ade.mean()
    fde = fde.mean()

    return diversity, ade, fde, mmade, mmfde


def compute_metrics(pred, gt, partitions=[3,6,9,12,16]):
    if pred.shape[0] == 1:
        diversity = 0.0
    dist_diverse = torch.pdist(pred.reshape(pred.shape[0], -1))
    diversity = dist_diverse.mean()

    ade_list = []
    fde_list = []
    for pa in partitions:
        gt_par = gt[:,:pa,:][None, ...]
        pred_par = pred[:,:pa,:][:, None, ...]

        diff = gt_par - pred_par
        dist = torch.linalg.norm(diff, dim=3)

        ade_list.append(dist[:, 0].mean(dim=1).min(dim=0)[0].mean())
        fde_list.append(dist[:, -1, -1].min(dim=0)[0].mean())





    return diversity, ade_list, fde_list

# === Step 2: Compute metrics
def compute_limb_metrics(pred, gt, parent=[-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]):
    """
    Inputs:
        gt: (1, T, (J-1)*3) ground truth (shared across all predictions)
        pred: (N, T, (J-1)*3) predictions
        parent: list of length J (joint tree), with parent[j] = parent of joint j
    Returns:
        normalized_limb_length_error: scalar mean over all N, T, J-1
        limb_length_variance: scalar mean of var over T, averaged over N and J-1
    """
    N, T, D = pred.shape
    J = (D // 3) + 1

    # Step 1: Pad pelvis (j=0) as zero vector
    def pad_pelvis(x, N_batch):
        x = x.view(N_batch, T, J - 1, 3)
        pelvis = torch.zeros(N_batch, T, 1, 3, device=x.device, dtype=x.dtype)
        return torch.cat([pelvis, x], dim=2)  # (N, T, J, 3)

    # Broadcast gt from (1, T, D) to (N, T, D) before padding
    gt_expanded = gt.expand(N, -1, -1)  # (N, T, D)

    gt_joints = pad_pelvis(gt_expanded, N)  # (N, T, J, 3)
    pred_joints = pad_pelvis(pred, N)

    # Step 2: Compute limb lengths
    limb_error_list = []
    limb_ratio_list = []

    for j in range(1, J):
        p = parent[j]
        if p < 0:
            continue

        gt_len = torch.norm(gt_joints[:, :, j] - gt_joints[:, :, p], dim=-1)    # (N, T)
        pred_len = torch.norm(pred_joints[:, :, j] - pred_joints[:, :, p], dim=-1)
        # print(gt_len, pred_len)

        eps = 1e-6
        norm_error = (pred_len - gt_len).abs() / (gt_len + eps)      # (N, T)
        norm_ratio = (pred_len[:,1:] - pred_len[:,:-1]).abs() / (pred_len[:,:-1] + eps)#pred_len/pred_len.mean() #(pred_len[:,1:] - pred_len[:,:-1]).abs() / (pred_len[:,:-1] + eps) #/ (gt_len + eps)                 # (N, T)

        limb_error_list.append(norm_error)
        limb_ratio_list.append(norm_ratio)

    # Stack to shape (N, T, J-1)
    limb_error = torch.stack(limb_error_list, dim=-1)  # (N, T, J-1)
    limb_ratio = torch.stack(limb_ratio_list, dim=-1)  # (N, T, J-1)

    # Step 3a: Normalized limb length error (mean absolute error)
    norm_limb_error = limb_error.abs()


    return norm_limb_error.mean()*100, norm_limb_error.var(dim=1).sqrt().mean()*100,limb_ratio.mean()*100, limb_ratio.var(dim=1).sqrt().mean()*100

# def compute_metrics(pred, gt, partitions=[3,6,9,12,16]):
#     if pred.shape[0] == 1:
#         diversity = 0.0
#     dist_diverse = torch.pdist(pred.reshape(pred.shape[0], -1))
#     diversity = dist_diverse.mean()

#     gt = gt[None, ...]
#     pred = pred[:, None, ...]

#     diff = pred - gt
#     dist = torch.linalg.norm(diff, dim=3)

#     ade, _ = dist[:, 0].mean(dim=1).min(dim=0)
#     fde, _ = dist[:, -1, -1].min(dim=0)

#     ade = ade.mean()
#     fde = fde.mean()

#     return diversity, ade, fde