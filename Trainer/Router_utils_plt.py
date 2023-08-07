import os
import numpy as np
import matplotlib.pyplot as plt;

def check_and_make_dir(filepath):
    dirname = os.path.dirname(filepath)
    os.makedirs(dirname, exist_ok=True)

def plot_x_y(x,y,x_label,y_label,save_path):
    check_and_make_dir(save_path)
    
    plt.figure()
    plt.plot(x,y)
    plt.xlabel(x_label);    plt.ylabel(y_label);
    plt.savefig(save_path);    # plt.show()
    plt.close()
    return

def save_DRL_solutions(save_path,solution_combo_filled,gridParameters,sortedHalfWireLength):  #f'solutionsDRL/test_benchmark_{globali+1}.gr.DRLsolution'
    check_and_make_dir(save_path)

    f = open(save_path, 'w+')
    twoPinSolutionPointer = 0
    routeListMerged = solution_combo_filled
    for i in range(gridParameters['numNet']):
        singleNetRouteCache = []
        indicator = i
        netNum = int(sortedHalfWireLength[i][0]) # i 
        
        i = netNum
        value = '{netName} {netID} {cost}\n'.format(netName=gridParameters['netInfo'][indicator]['netName'],
                                                netID = gridParameters['netInfo'][indicator]['netID'],
                                                cost = 0) #max(0,len(routeListMerged[indicator])-1))
        f.write(value)
        for j in range(len(routeListMerged[indicator])):
            for  k in range(len(routeListMerged[indicator][j])-1):
                a = routeListMerged[indicator][j][k]
                b = routeListMerged[indicator][j][k+1]
                if (a[3],a[4],a[2],b[3],b[4],b[2]) not in singleNetRouteCache:  
                    singleNetRouteCache.append((a[3],a[4],a[2],b[3],b[4],b[2]))
                    singleNetRouteCache.append((b[3],b[4],b[2],a[3],a[4],a[2]))

                    diff = [abs(a[2]-b[2]),abs(a[3]-b[3]),abs(a[4]-b[4])]
                    if diff[1] > 2 or diff[2] > 2:
                        continue
                    elif diff[1] == 2 or diff[2] == 2:
                        continue
                    elif diff[0] == 0 and diff[1] == 0 and diff[2] == 0:
                        continue
                    elif diff[0] + diff[1] + diff[2] >= 2:
                        continue
                    else:
                        value = '({},{},{})-({},{},{})\n'.format(int(a[0]),int(a[1]),a[2],int(b[0]),int(b[1]),b[2])
                        f.write(value)
            twoPinSolutionPointer = twoPinSolutionPointer + 1
        f.write('!\n')
    f.close()
    
    return


#**  can't see this ??
def saveDRL_Visual(save_path,solutionTwoPin,gridParameters, ):#(solutionTwoPin):
    check_and_make_dir(save_path)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for twoPinRoute in solutionTwoPin:
      x = []; y = []; z = [];
      for i in range(len(twoPinRoute)):
        x.append(twoPinRoute[i][3]);  y.append(twoPinRoute[i][4]); z.append(twoPinRoute[i][2])
        ax.plot(x,y,z,linewidth=2.5)
    
    plt.xlim([0, gridParameters['gridSize'][0]-1])
    plt.ylim([0, gridParameters['gridSize'][1]-1])
    plt.savefig(save_path);  # plt.show()
    plt.close()
    return

def saveDRL_visual2D(save_path,routeListNotMerged, gridParameters):
    check_and_make_dir(save_path)
    fig = plt.figure();     ax = fig.add_subplot(111)
    for routeList in routeListNotMerged:
        for route in routeList:
            num_points = len(route)
            for i in range(num_points-1):
                pair_x = [route[i][3], route[i+1][3]]
                pair_y = [route[i][4], route[i+1][4]]
                pair_z = [route[i][2], route[i+1][2]]
                if pair_z[0] ==pair_z[1] == 1:
                    ax.plot(pair_x, pair_y, color='blue', linewidth=2.5)
                if pair_z[0] ==pair_z[1] == 2:
                    ax.plot(pair_x, pair_y, color='red', linewidth=2.5)
    ax.axis('scaled')
    ax.invert_yaxis()
    plt.xlim([-0.1, gridParameters['gridSize'][0]-0.9])
    plt.ylim([-0.1, gridParameters['gridSize'][1]-0.9])
    plt.axis('off')
    plt.savefig(save_path)
    plt.close()
    return

def saveDRL_visual3D(save_path,gridParameters):
    check_and_make_dir(save_path)
    fig = plt.figure();      ax = fig.add_subplot(111, projection='3d')
    ax.set_zlim(0.75,2.25)
    ax.set_xticks([]); ax.set_yticks([]);   ax.set_zticks([]);
    x_meshP = np.linspace(0,gridParameters['gridSize'][0]-1,200)
    y_meshP = np.linspace(0,gridParameters['gridSize'][1]-1,200)
    x_mesh,y_mesh = np.meshgrid(x_meshP,y_meshP)
    z_mesh = np.ones_like(x_mesh)
    ax.plot_surface(x_mesh,y_mesh,z_mesh,alpha=0.3,color='r')
    ax.plot_surface(x_mesh,y_mesh,2*z_mesh,alpha=0.3,color='r')
    plt.savefig(save_path)
    plt.axis('off')
    plt.close()
    return



def save(result_dir,globali,episodes,gridParameters,
         reward_plot_combo, reward_plot_combo_pure,
         solution_combo_filled,sortedHalfWireLength, solutionTwoPin):
    print("saving reward plots and reward data ...")   
    result_img_dir = f"{result_dir}/imgs"
    result_reward_dir = f"{result_dir}/reward_ckpt"
    os.makedirs(result_reward_dir,exist_ok=True)

    save_path = f"{result_img_dir}/reward_plt/test_benchmark_{globali+1}.DRLRewardPlot.jpg";      print(save_path)        
    n = np.linspace(1,episodes,len(reward_plot_combo))
    plot_x_y(n,reward_plot_combo,'episodes','reward',save_path)
    
    save_path = f'{result_img_dir}/reward_plt_pure/test_benchmark_{globali+1}.DRLRewardPlotPure.jpg';   print(save_path)
    n = np.linspace(1,episodes,len(reward_plot_combo_pure))
    plot_x_y(n,reward_plot_combo_pure,'episodes','reward',save_path)

    #filename = benchmark_reduced/test_benchmark_i.gr.rewardData.npy
    np.save(f'{result_reward_dir}/test_benchmark_{globali+1}.gr.rewardData',reward_plot_combo)

    #! dump solution of DRL  -> solutionsDRL/test_benchmark_{globali+1}.gr.DRLsolution
    save_DRL_solutions(f'{result_dir}/test_benchmark_{globali+1}.gr.DRLsolution',
                        solution_combo_filled,gridParameters,sortedHalfWireLength)
    
    #* Visualize solution 1D
    saveDRL_Visual(f'{result_img_dir}/route1D/DRLRoutingVisualize_test_benchmark_{globali+1}.png',
                                solutionTwoPin,  gridParameters)
    #* Visualize results on 2D 
    # plt_utils.saveDRL_visual2D(f'{result_img_dir}/route2D/DRLRoutingVisualize_test_benchmark2d_{globali+1}.png',
    #                            routeListNotMerged, gridParameters)
    #todo Plot of routing for multilple net (RL solution) 3d
    # plt_utils.saveDRL_visual3D(f'{result_img_dir}/route3D/DRLRoutingVisualize_test_benchmark3d_{globali+1}.png', gridParameters)

