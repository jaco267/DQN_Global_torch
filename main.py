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
from dataclasses import dataclass
import pyrallis
from pyrallis import field
from typing import List
@dataclass
class TrainConfig:
    """ Training config for Machine Learning """
    algos:str="dqn"
    mode:str="train"  #train, eval
    hid_layer:int=1 
    emb_dim:int=64
    context_len:int = 5   #try other numbers 1~30,  I known 50 is bad and slow
    early_stop:bool=False
    result_dir:str = "solutionsDRL"
    save_ckpt:bool=True
    load_ckpt:bool=True
    self_play_episode_num:int = 150    
    enable_wandb:bool=True 
    wandbName:str=""    
    data_folder:str='train_data_/benchmark_reduced'
    run_benchmark_num:int = -1
    verbose:bool = False
    rainbow_mode:List[str] = field(default=['double','nstep'], is_mutable=True)
    #['double','duel','noisy','per','cat','nstep']
#!! python Router.py --config_path test.yaml
@pyrallis.wrap(config_path='./train.yaml')
def main_fn(cfg:TrainConfig ):  
    '''
    [double] is nice (20/20),  
    In the current setting [duel, noisy] are bad, 
    todo : save replay ckpt when training (this will influence the beta update of PER)
    #[double] at train and [double,per] at eval is still bad 
    '''
    print(">>>>>>>>>>>>>>>\n",locals())
    algos_name = cfg.algos
    rain_dict = {}
    if 'rainbow' in cfg.algos:
        stardard_rainbow_list = ['double','duel','noisy','per','cat','nstep']
        for r_mode in stardard_rainbow_list:
            if r_mode in cfg.rainbow_mode:
                rain_dict[r_mode] = True
                algos_name+= f'_{r_mode}' 
            else:
                rain_dict[r_mode] = False
        print('rainbow_mode--->',rain_dict)
        print('algos_name--->',algos_name)
    if cfg.enable_wandb:
        wandb.login()
        project_name = "Global_route"
        config={
                "algos":algos_name,
                "mode":cfg.mode,
                "layer":cfg.hid_layer,
                "emb_dim":cfg.emb_dim,
                "episode":cfg.self_play_episode_num,
                "context_len":cfg.context_len,
        }
        if cfg.verbose:
            group = cfg.wandbName+"_"+"_".join(
                [f"{key}={val}" for key, val in config.items()]
            ),
        else:
            group = f"{algos_name}_{cfg.mode}_{cfg.wandbName}"
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
    print(cfg.self_play_episode_num,cfg.result_dir)
    os.system(f'rm -r {cfg.result_dir}');    os.makedirs(cfg.result_dir);    
    benchmark_reduced_path = cfg.data_folder
    src_benchmark_file = [li for li in os.listdir(benchmark_reduced_path) if "rewardData" not in li]
    #*  ex. test_benchmark_1.gr,  test_benchmark_2.gr .....
    success_count = 0
    
    env = GridGraphV2.GridGraph 
    algos_fn = importlib.import_module(f'Trainer.algos.agent.{cfg.algos}')
    print(algos_fn.__name__,"...algos module name...")

    if cfg.run_benchmark_num < 0:
        cfg.run_benchmark_num = len(src_benchmark_file)
    cfg.run_benchmark_num = min(cfg.run_benchmark_num,len(src_benchmark_file))
    for i in range(cfg.run_benchmark_num):
        #benchmark_reduced/test_benchmark_1.gr
        read_file_name = f"{benchmark_reduced_path}/test_benchmark_{i+1}.gr"  
        print (f'\n********{i+1}/{cfg.run_benchmark_num}******Working on {read_file_name}****************')
        success = train_one_epoch(read_file_name,
                         env,     
                         algos_fn, 
                         algos_name=cfg.algos,
                         hid_layer=cfg.hid_layer,
                         globali=i,
                         self_play_episode_num=cfg.self_play_episode_num,
                         result_dir=cfg.result_dir,
                         logger=logger,
                         save_ckpt=cfg.save_ckpt,
                         ckpt_folder = "./model/",
                         early_stop=cfg.early_stop,
                         load_ckpt=cfg.load_ckpt,
                         emb_dim=cfg.emb_dim,
                         context_len=cfg.context_len,
                         rainbow_mode=rain_dict)
        success_count+=success
        logger.log({'success_count':success_count})
    return 



if __name__ == '__main__':
   main_fn()


