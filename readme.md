### rainbow DTQN
This repo combine [Deep Transformer Q-Networks ](https://github.com/kevslinger/DTQN) and [rainbow](https://arxiv.org/abs/1710.02298) to solve the Global routing problem. 

The global routing benchmark generator is from the paper ["A Deep Reinforcement Learning Approach for Global Routing"](https://arxiv.org/pdf/1906.08809.pdf).     
In this implementation, we used DTQN and rainbow instead of DQN, and outperform the previous result.   
Rainbow DTQN can solve global routing problem even without A* memory burn-in.

In the experiment, DTQN can generalize better than DQN, making the pretraining possible.

## Generate Dataset
To train the agent, first generate the train and eval dataset
```sh
# train data
python gen_data.py --benchmarkNumber 5 --gridSize 8 --netNum 20 --vCap 4  --hCap 4 --maxPinNum 5 --reducedCapNum 3 --prefix ./train_data_/
# test data
python gen_data.py --benchmarkNumber 20 --gridSize 8 --netNum 20 --vCap 4  --hCap 4 --maxPinNum 5 --reducedCapNum 3 --prefix ./test_data_/
```       
## start training 
```sh
python train.py --algos=dtqn_per_noisy --self_play_episode_num=4 --result_dir=solutionsDRL --load_ckpt=True --save_ckpt=True  --data_folder="train_data_/benchmark_reduced" --wandbName="dtqn_per_noisy_context_len5_train" --hid_layer=3 --emb_dim=64 --context_len=5
```
## start eval
```sh
python train.py --algos=dtqn_per_noisy --self_play_episode_num=150 --result_dir=solutionsDRL --load_ckpt=True --save_ckpt=True  --data_folder="test_data_/benchmark_reduced" --wandbName="dtqn_per_noisy_context_len5" --hid_layer=3 --emb_dim=64 --context_len=5
```

## Go to wandb site to check the result
dtqn (pink) can solve 20/20 pin problem in 150 episode,  
while dqn can only solve 18/20  

<img src="assets/2023-07-28-10-01-43.png" alt= “” width="500px" >
<img src="assets/2023-07-28-10-03-26.png" alt= “” width="500px" >

## wirelength
### dtqn vs A* solution
<img src="assets/2023-07-28-10-05-37.png" alt= “” width="500px" >