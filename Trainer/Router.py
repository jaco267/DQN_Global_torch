from __future__ import print_function
import datetime
import os
import matplotlib;   matplotlib.use('TkAgg');
import numpy as np
import random
import os
import random
import operator

import wandb

from Trainer import Initializer as init

from Trainer import Router_utils as utils
from Trainer import Router_utils_plt as save_utils
from Trainer import GridGraphV2
from datetime import datetime
import importlib
np.random.seed(10701);  random.seed(10701);
def saveResults():
    return
def timestamp():
    return datetime.now().strftime("%B %d, %H:%M:%S")

def train_one_epoch(filename,  # benchmark_reduced/test_benchmark_i.gr
                    env,
                    algos_fn,algos_name,
                    hid_layer,
                    globali,
                    self_play_episode_num,
                    result_dir,
                    logger,
                    save_ckpt:bool,
                    load_ckpt:bool,
                    early_stop,
                    ckpt_folder,
                    emb_dim,
                    context_len,
                    rainbow_mode):
    #* print("---data preprocessing---")
    # # Getting Net Info
    grid_info = init.read(filename)
    gridParameters:dict = init.gridParameters(grid_info)

    sortedHalfWireLength = sorted(init.VisualGraph(gridParameters).bounding_length().items(),  
                                key=operator.itemgetter(1),reverse=True) # Large2Small
    netSort = []
    for i in range(gridParameters['numNet']):   #20    #A1~A20  for each teset_benchmark.gr
        order = int(sortedHalfWireLength[i][0])
        netSort.append(order)           #arrange net by its wireLength 
    #* print(f"---netsort {netSort} len {len(netSort)}---")
    twoPinEachNetClear:list = utils.gen_2pinListClear(gridParameters)    

    twopinlist_nonet = utils.gen2pinlistNet(
                            utils.gen2pinListCombo(
                                    gridParameters,
                                    sortedHalfWireLength
                            ))
    assert np.sum(twoPinEachNetClear) == len(twopinlist_nonet)   #== 49,  20net has 49 pin connect, avg_connect/net ~=2.5
    #* print("---DRL Module from here---")
    os.makedirs(ckpt_folder,exist_ok=True);  
                                                
    Env_graph = env(gridParameters,
                                max_step=100,   #?20
                                twopin_combo=twopinlist_nonet,
                                net_pair=twoPinEachNetClear) 
    # Training DRL
    #!!!  core  DQN_implement.py
    success = 0
    #* print("----start training---")
    if algos_name in ['rainbow_dqn']:
        print(algos_name,">>>>>>>")
        agent = algos_fn.DQN_Agent( env=Env_graph, rainbow_mode=rainbow_mode, hid_layer=hid_layer,
        emb_dim=emb_dim, self_play_num =self_play_episode_num, context_len=context_len,)  
        results, solutionTwoPin,posTwoPinNum,success = agent.train(
                            twoPinEachNetClear,
                            netSort,
                            ckpt_path=f"{ckpt_folder}{algos_name}.ckpt",
                            logger=logger,
                            save_ckpt=save_ckpt,
                            load_ckpt=load_ckpt,
                            early_stop=early_stop,
            )
    else:
        print(algos_name,">>>>>>>")
        agent = algos_fn.DQN_Agent( Env_graph, hid_layer,emb_dim,
                                   self_play_episode_num =self_play_episode_num,
                                   context_len=context_len)  
        results, solutionTwoPin,posTwoPinNum,success = agent.train(
                            twoPinEachNetClear,
                            netSort,
                            ckpt_path=f"{ckpt_folder}{algos_name}.ckpt",
                            logger=logger,
                            save_ckpt=save_ckpt,
                            load_ckpt=load_ckpt,
                            early_stop=early_stop,
            )
    # print("======---saving results(Generate output file for DRL solver)---======") 
    assert len(Env_graph.twopin_combo)==len(twopinlist_nonet)
    print(f"=====posTwoPinNum  {posTwoPinNum}/{len(Env_graph.twopin_combo)}======")
    if posTwoPinNum >= len(twopinlist_nonet): 
        save_utils.save(result_dir,globali,agent.max_episodes,gridParameters,
            results['reward_plot_combo'], results['reward_plot_combo_pure'],
            results['solutionDRL'],sortedHalfWireLength, solutionTwoPin)
    else:
        print("DRL fails with existing max episodes! : (")
    return success

class Print_log:
    '''just a fake wandDB log (used when --enable_wandb=False)'''
    def __init__(self) -> None:
        pass
    def log(self,item):
        # print(item)
        pass
def main_fn(
        algos="dtqn",
        mode="train",  #train, eval
        hid_layer=1, emb_dim=64,
        context_len = 5,   #try other numbers 1~30,  I known 50 is bad and slow
        early_stop=False,
        result_dir = "solutionsDRL",
        save_ckpt=True,load_ckpt=True,
        self_play_episode_num = 150,    
        enable_wandb=True, wandbName="",    
        data_folder='train_data_/benchmark_reduced',
        run_benchmark_num = None,
        verbose = False,
        rainbow_mode = ['double','nstep']#['double','duel','noisy','per','cat','nstep']
    ):  
    '''
    [double] is nice (20/20),  
    In the current setting [duel, noisy] are bad, 
    todo : save replay ckpt when training (this will influence the beta update of PER)
    #[double] at train and [double,per] at eval is still bad 
    '''
    
    print(">>>>>>>>>>>>>>>\n",locals())
    algos_name = algos
    rain_dict = {}
    if 'rainbow' in algos:
        stardard_rainbow_list = ['double','duel','noisy','per','cat','nstep']
        for r_mode in stardard_rainbow_list:
            if r_mode in rainbow_mode:
                rain_dict[r_mode] = True
                algos_name+= f'_{r_mode}' 
            else:
                rain_dict[r_mode] = False
        print('rainbow_mode--->',rain_dict)
        print('algos_name--->',algos_name)
    if enable_wandb:
        wandb.login()
        project_name = "Global_route"
        config={
                "algos":algos_name,
                "mode":mode,
                "layer":hid_layer,
                "emb_dim":emb_dim,
                "episode":self_play_episode_num,
                "context_len":context_len,
        }
        if verbose:
            group = wandbName+"_"+"_".join(
                [f"{key}={val}" for key, val in config.items()]
            ),
        else:
            group = f"{algos_name}_{mode}"
        wandb.init(
            project=project_name,
            name = timestamp(),
            group = group,
            config=config,
        )    
        logger = wandb
    else:
        print("disable wandb")
        logger = Print_log()
    print(self_play_episode_num,result_dir)
    os.system(f'rm -r {result_dir}');    os.makedirs(result_dir);    
    benchmark_reduced_path = data_folder
    src_benchmark_file = [li for li in os.listdir(benchmark_reduced_path) if "rewardData" not in li]
    #*  ex. test_benchmark_1.gr,  test_benchmark_2.gr .....
    success_count = 0
    
    env = GridGraphV2.GridGraph 
    algos_fn = importlib.import_module(f'Trainer.algos.agent.{algos}')
    print(algos_fn.__name__,"...algos module name...")

    if run_benchmark_num == None:
        run_benchmark_num = len(src_benchmark_file)
    run_benchmark_num = min(run_benchmark_num,len(src_benchmark_file))
    for i in range(run_benchmark_num):
        #benchmark_reduced/test_benchmark_1.gr
        read_file_name = f"{benchmark_reduced_path}/test_benchmark_{i+1}.gr"  
        print (f'\n********{i+1}/{run_benchmark_num}******Working on {read_file_name}****************')
        success = train_one_epoch(read_file_name,
                         env,     
                         algos_fn, 
                         algos_name=algos,
                         hid_layer=hid_layer,
                         globali=i,
                         self_play_episode_num=self_play_episode_num,
                         result_dir=result_dir,
                         logger=logger,
                         save_ckpt=save_ckpt,
                         ckpt_folder = "./model/",
                         early_stop=early_stop,
                         load_ckpt=load_ckpt,
                         emb_dim=emb_dim,
                         context_len=context_len,
                         rainbow_mode=rain_dict)
        success_count+=success
        logger.log({'success_count':success_count})
    return 






