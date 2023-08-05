from __future__ import print_function
import datetime
import os
import matplotlib;   matplotlib.use('TkAgg');
import numpy as np
import random
import os
import time
import random
import operator

import wandb

from Trainer.algos import DQN
from Trainer.algos import DDQN
from Trainer.algos import Dueling_DDQN
from Trainer.algos import Duel_DDQN_PER

from Trainer.algos import _1_DQN
from Trainer.algos import _5_Duel_DDQN_PER_noisy
from Trainer.algos import _6_PER_noisy_categorical
from Trainer.algos import _7_DQN_rainbow_nocat

from Trainer.algos import _8_DTQN_epsilon
from Trainer.algos import _9_DTQN_PER
from Trainer.algos import _10_DTQN_PER_noisy
from Trainer.algos import _11_DTQN_step_PER_noisy
from Trainer.algos import _12_DTQN_noisy
from Trainer.algos import _13_DTQN_noisy_bf
from Trainer import Initializer as init

from Trainer import Router_utils as utils
from Trainer import Router_save_utils as save_utils
from Trainer import GridGraph
from Trainer import GridGraphV2
from datetime import datetime
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
                    context_len):
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
                                                
    graphcase = env(gridParameters,
                                max_step=100,   #?20
                                twopin_combo=twopinlist_nonet,
                                net_pair=twoPinEachNetClear) 
    # Training DRL
    #!!!  core  DQN_implement.py
    success = 0
    #* print("----start training---")
    if   algos_name == "dqn" or\
         algos_name == "nstep" or\
         algos_name == "dtqn" or\
         algos_name == "dtqn_per_noisy" or\
         algos_name == "dtqn_per" or\
         algos_name == "dtqn_step_per" or\
         algos_name == "dtqn_noisy" or\
         algos_name == "dtqn_noisy_bf":
        print(algos_name,">>>>>>>")
        agent = algos_fn.DQN_Agent( graphcase, hid_layer,emb_dim,
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
    else:
        agent = algos_fn.DQN_Agent( graphcase,self_play_episode_num)  
        results, solutionTwoPin,posTwoPinNum = agent.train(
                                    twoPinEachNetClear,
                                    netSort,
                                    ckpt_folder,
                    )
        print("ffrrrrr",posTwoPinNum,".........")
    # print("======---saving results(Generate output file for DRL solver)---======") 
    assert len(graphcase.twopin_combo)==len(twopinlist_nonet)
    print(f"=====posTwoPinNum  {posTwoPinNum}/{len(graphcase.twopin_combo)}======")
    if posTwoPinNum >= len(twopinlist_nonet): 
        save_utils.save(result_dir,globali,agent.max_episodes,gridParameters,
            results['reward_plot_combo'], results['reward_plot_combo_pure'],
            results['solutionDRL'],sortedHalfWireLength, solutionTwoPin)
    else:
        print("DRL fails with existing max episodes! : (")
    return success

class Print_log:
    def __init__(self) -> None:
        pass
    def log(self,item):
        # print(item)
        pass
def main_fn(
        algos="dtqn_per_noisy",
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
        verbose = False
    ):
    print(">>>>>>>>>>>>>>>\n",locals())
    if enable_wandb:
        wandb.login()
        project_name = "Global_route"
        config={
                "algos":algos,
                "mode":mode,
                "layer":hid_layer,
                "emb_dim":emb_dim,
                "episode":self_play_episode_num,
                "context_len":context_len
        }
        if verbose:
            group = wandbName+"_"+"_".join(
                [f"{key}={val}" for key, val in config.items()]
            ),
        else:
            group = f"{algos}_{mode}"
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
    env = GridGraph.GridGraph
    if algos == "dqn":
        env = GridGraphV2.GridGraph     
        algos_fn =  _1_DQN
        # algos_fn = DQN
    elif algos == "ddqn":
        algos_fn = DDQN
    elif algos =="dddqn":
        algos_fn = Dueling_DDQN
    elif algos =="dddqn_PER":
        algos_fn = Duel_DDQN_PER
    elif algos == "noisy":
        algos_fn = _5_Duel_DDQN_PER_noisy
    elif algos == "categorical":
        algos_fn = _6_PER_noisy_categorical
    elif algos == "nstep":
        #! v2  is a new environ interface, only support in several algos
        env = GridGraphV2.GridGraph     
        algos_fn =  _7_DQN_rainbow_nocat
    elif algos == "dtqn":
        env = GridGraphV2.GridGraph     #! v2
        algos_fn = _8_DTQN_epsilon
    elif algos == "dtqn_per":
        env = GridGraphV2.GridGraph     #! v2
        algos_fn = _9_DTQN_PER
    elif algos == "dtqn_per_noisy":  #** episode per
        env = GridGraphV2.GridGraph     #! v2
        algos_fn =  _10_DTQN_PER_noisy
    elif algos == "dtqn_step_per":
        env = GridGraphV2.GridGraph     #! v2
        algos_fn =  _11_DTQN_step_PER_noisy
    elif algos == "dtqn_noisy":
        env = GridGraphV2.GridGraph     #! v2
        algos_fn =  _12_DTQN_noisy
    elif algos == "dtqn_noisy_bf":
        env = GridGraphV2.GridGraph     #! v2
        algos_fn =  _13_DTQN_noisy_bf
    else:
        raise Exception(f"error...algos {algos} doesn't exist") 

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
                         context_len=context_len)
        success_count+=success
        logger.log({'success_count':success_count})
    return 






