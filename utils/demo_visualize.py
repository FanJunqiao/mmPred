import os
import torch
import numpy as np
from utils.pose_gen import pose_generator
from utils.visualization import render_animation

def demo_visualize(mode, cfg, model, diffusion, test_dataset_list, plot_indexes=[2000], name = "my", plot_pc = False):
    """
    script for drawing gifs in different modes
    """


    if mode == 'test_mmBody':
        for i in range(len(test_dataset_list)):
            
            for idx, (sub, seq_id) in enumerate(test_dataset_list[i].data_idx):
                if int(sub) == 61 and int(seq_id) == 1848:
                    print(f"Index with sub=61 and seq_id=1848: {idx}")
                    plot_indexes = [idx]
                    break
            
            plot_indexes = [1723]#list(range(800,5535, 10))
            print(plot_indexes)
            for id_plot in plot_indexes:
                data_loader = test_dataset_list[i].__getitem__(id_plot)
                id = data_loader["sub"]
                seq_id = data_loader["seq_id"]
                radar_pc = data_loader["radar"] if plot_pc else None
                # pose_gen = pose_generator(data_loader, model, diffusion, cfg, mode='test',n_pre=10)
                pose_gen1 = pose_generator(data_loader, model, diffusion, cfg, mode='test',n_pre=10)
                # from plot.plot_paper import plot_best_hypothesis_from_generator, plot_best_hypothesis_overlay1
                # if name == "gt":
                #     plot_best_hypothesis_overlay1(pose_gen, radar_pc = radar_pc, save_path=f'./test_vis_mmBody4/{name}_best_test_case{id}_{seq_id}.svg',pred_flag=False, gt_flag=True)
                # else: plot_best_hypothesis_overlay1(pose_gen, radar_pc = radar_pc, save_path=f'./test_vis_mmBody4/{name}_best_test_case{id}_{seq_id}.svg',)
                # plot_best_hypothesis_from_generator(pose_gen,save_path=f'./test_vis_mmBody4/{name}_best_test_case{id}_{seq_id}.png')
                # render_animation(cfg, pose_gen1, ['mmPred'], cfg.t_his, ncol=cfg.vis_col + 2,
                #                 output=os.path.join("test_vis_mmBody4", f'{name}_best_test_case{id}_{seq_id}.gif'))
                render_animation(cfg, pose_gen1, ['mmPred'], cfg.t_his, ncol=cfg.vis_col + 2,
                                output=os.path.join("test_vis_mmBody4", f'select.gif'))

    elif mode == 'test_mmfi':
        for i in range(len(test_dataset_list)):
            for id_plot in plot_indexes:
                data_loader = test_dataset_list[i].__getitem__(id_plot)
                pose_gen = pose_generator(data_loader, model, diffusion, cfg, mode='test',n_pre=10)
                # from plot.plot_paper import plot_best_hypothesis_from_generator
                # plot_best_hypothesis_from_generator(pose_gen,save_path=f'./test_vis_mmfi/my_best_test_case{i}_{id_plot}.png',t_his=5,frame_indices=[0,2,4,6,8,10,12,14])
                render_animation(cfg, pose_gen, ['mmPred'], cfg.t_his, ncol=cfg.vis_col + 2,
                                output=os.path.join("test_vis_mmfi", f'test_case{id_plot}.gif'),radar_pc=data_loader["radar"])

