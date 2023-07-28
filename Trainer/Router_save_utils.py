import os
import numpy as np
import matplotlib.pyplot as plt;
from Trainer import Router_utils_plt as plt_utils

def save(result_dir,globali,episodes,gridParameters,
         reward_plot_combo, reward_plot_combo_pure,
         solution_combo_filled,sortedHalfWireLength, solutionTwoPin):
    print("saving reward plots and reward data ...")   
    result_img_dir = f"{result_dir}/imgs"
    result_reward_dir = f"{result_dir}/reward_ckpt"
    os.makedirs(result_reward_dir,exist_ok=True)

    save_path = f"{result_img_dir}/reward_plt/test_benchmark_{globali+1}.DRLRewardPlot.jpg";      print(save_path)        
    n = np.linspace(1,episodes,len(reward_plot_combo))
    plt_utils.plot_x_y(n,reward_plot_combo,'episodes','reward',save_path)
    
    save_path = f'{result_img_dir}/reward_plt_pure/test_benchmark_{globali+1}.DRLRewardPlotPure.jpg';   print(save_path)
    n = np.linspace(1,episodes,len(reward_plot_combo_pure))
    plt_utils.plot_x_y(n,reward_plot_combo_pure,'episodes','reward',save_path)

    #filename = benchmark_reduced/test_benchmark_i.gr.rewardData.npy
    np.save(f'{result_reward_dir}/test_benchmark_{globali+1}.gr.rewardData',reward_plot_combo)

    #! dump solution of DRL  -> solutionsDRL/test_benchmark_{globali+1}.gr.DRLsolution
    plt_utils.save_DRL_solutions(f'{result_dir}/test_benchmark_{globali+1}.gr.DRLsolution',
                        solution_combo_filled,gridParameters,sortedHalfWireLength)
    
    #* Visualize solution 1D
    plt_utils.saveDRL_Visual(f'{result_img_dir}/route1D/DRLRoutingVisualize_test_benchmark_{globali+1}.png',
                                solutionTwoPin,  gridParameters)
    #* Visualize results on 2D 
    # plt_utils.saveDRL_visual2D(f'{result_img_dir}/route2D/DRLRoutingVisualize_test_benchmark2d_{globali+1}.png',
    #                            routeListNotMerged, gridParameters)
    #todo Plot of routing for multilple net (RL solution) 3d
    # plt_utils.saveDRL_visual3D(f'{result_img_dir}/route3D/DRLRoutingVisualize_test_benchmark3d_{globali+1}.png', gridParameters)