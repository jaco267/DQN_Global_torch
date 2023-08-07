#  DTQN Global routing
In this repo, I used [Deep Transformer Q-Networks ](https://github.com/kevslinger/DTQN) to solve the Global routing problem, and found that dtqn can generalize better than DQN in the Global routing environment.   
The global routing benchmark generator is from the paper ["A Deep Reinforcement Learning Approach for Global Routing"](https://arxiv.org/pdf/1906.08809.pdf).     
Original implementation is in [this repo](https://github.com/haiguanl/DQN_GlobalRouting).

# How to run?
## Step0. Install packages
```sh
pip install -r requirements.txt
```
## Step1. Generate Dataset
To train the agent, first generate the train and eval dataset
```sh
# train data
python gen_data.py --benchmarkNumber 30 --gridSize 8 --netNum 20 --vCap 4  --hCap 4 --maxPinNum 5 --reducedCapNum 3 --prefix ./train_data_/
# test data
python gen_data.py --benchmarkNumber 20 --gridSize 8 --netNum 20 --vCap 4  --hCap 4 --maxPinNum 5 --reducedCapNum 3 --prefix ./test_data_/
```       
## Step2. start training 
default options is in configs.yaml (or Trainer/Router.py main_fn), you can modify it in yaml or overwrite it through command line interface

command line (or yaml) args
- mode:str = 'train' || 'eval'
- algos:str  ( algos names are the filenames in Trainer/algos/agent folder)
  - dqn, dtqn, rainbow_dqn
- wandbName:str
- hid_layer:int = 3
- emb_dim:int = 64
- context_len:int = 5
- early_stop:bool = False
- result_dir:str = solutionsDRL
- load_ckpt:bool
- save_ckpt:bool
- self_play_episode_num:int -> self play budget
- enable_wandb:bool=True -> use wandb for plotting
- data_folder:str -> netlist data to solve
```sh
python run.py --mode "train" --algos dtqn
# python run.py --mode "train" --algos dqn
```
## Step3. start eval
eval run the 20 test benchmark, each with 150 self-play number (in configs.yaml),
eval will take longer time to run (about 1hr on a RTX3060 GPU)
```sh
python run.py --mode "eval" --algos dtqn
# python run.py --mode "eval" --algos dqn
```

## Step4. Go to wandb site to check the result
dtqn (pink) can solve 20/20 pin problem in 150 episode,  
while dqn can only solve 19/20  

<img src="assets/successPin_selfplay150.png" alt= “” width="800px" >
<img src="assets/reward_selfplay150.png" alt= “” width="800px" >

## Step5. plot wire len
```sh
cd eval
python Evaluation.py 
python VisualizeResults.py 
``` 
will generate wirelen images in ./eval/VisualizeResult.
 
<img src="assets/dtqn_wirelength.png" alt= “” width="500px" >

# Differences from the original implementation
The [original implementation](https://github.com/haiguanl/DQN_GlobalRouting) used A* memory burn-in to speed up training.
This implementation didn't use memory burn-in technique for simplicity. 

# Compare DTQN with DQN
DTQN can model the stochastic and partially observable environment better than dqn, making the pretrain more robust.  
### Success count with 50 self-play budget 
dqn needs more self-play budget to connect two pin, making it perform worse than dtqn under lower self-play evaluation budget(50)
When self-play_episode_num is set to 50  
dqn can only complete 14/20 pin benchmarks,   
while dtqn can complete all 20/20 pin benchmarks.  
```sh
#pretrain
python run.py --mode "train" --algos dtqn||dqn --run_benchmark_num 30
#eval
python run.py --mode "eval" --algos dtqn||dqn --self_play_episode_num 50
``` 
<img src="assets/successPin_selfplay50.png" alt= “” width="800px" >

### 150 self-play budget
#### success count
dqn performs better when self-play budget is higher(success count 19/20), but still cannot solve all problems
```sh
#eval
python run.py --mode "eval" --algos dtqn
python run.py --mode "eval" --algos dqn 
```  
<img src="assets/successPin_selfplay150.png" alt= “” width="800px" >

#### success rate
dtqn can solve pin problem in smaller self-play number compare to dqn.  
  
<img src="assets/successRate_seflplay150.png" alt= “” width="800px" >
   
#### espisode reward
dtqn have higher reward than dqn   
<img src="assets/reward_selfplay150.png" alt= “” width="800px" >

#### wire length
compare dqn wire length with A* algorithm (0.52 winning rate) (with 19/20 pins)
<img src="assets/dqn_wirelen.png" alt= “” width="500px" >


compare dtqn wire length with A* algorithm (0.7 winning rate) (with 20/20 pins)
<img src="assets/dtqn_wirelength.png" alt= “” width="500px" >



## Todo
- Reward shaping for Overflow 
- merge rainbow into one file
- translate eval2008.pl to python
- json format for netlist
## run rainbow 
In the current setting, only double dqn can improve the performance.
```sh
python run.py --mode "train" --algos rainbow_dqn --enable_wandb True --rainbow_mode double,nstep 
python run.py --mode "eval" --algos rainbow_dqn --enable_wandb True --rainbow_mode double,nstep   
```

