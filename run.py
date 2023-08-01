# https://stackabuse.com/executing-shell-commands-with-python/
import os
import fire
import yaml
import json 
import Trainer
def main(algos="dtqn",
         mode="eval",
         wandbName=None):
    algos_list = ['dqn','dtqn','nstep']
    mode_list =  ['eval','train'] 
    if algos not in algos_list:
        raise Exception(f'algo should be in {algos_list}') 
    if mode not in mode_list:
        raise Exception(f'mode should be in {mode_list}') 
    config_dict = {}
    with open("configs.yaml",'r') as f:
        data = yaml.safe_load(f)
    # print(json.dumps(data,indent=2))
    print("=============update default============")
    default_all_dict = data['default']['all']
    print(default_all_dict)
    config_dict.update(default_all_dict)
    if mode == "train":
        print("---pretraining---")
        default_train_eval_dict = data['default']['train']
    elif mode == "eval":
        print("---testing---")
        default_train_eval_dict = data['default']['eval']
    print(default_train_eval_dict)
    config_dict.update(default_train_eval_dict)
    print("============update algos setting============")
    if wandbName:
        wandb_dict = {'algos':algos,'wandbName':wandbName}
    else:
        wandb_dict = {'algos':algos,'wandbName':algos}
    print(wandb_dict)
    config_dict.update(wandb_dict)
    print("============config result============")
    Trainer.Router.main_fn(**config_dict)
    return 
if __name__ == "__main__":
    fire.Fire(main)

