import os
import fire
import yaml
import json 
import Trainer
def main(algos="dtqn_noisy",
         mode="eval",
         wandbName=None,
         enable_wandb=None,
         run_benchmark_num=None):
    mode_list =  ['eval','train'] 
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
    if enable_wandb != None:
        config_dict.update({'enable_wandb':enable_wandb,'run_benchmark_num':run_benchmark_num})
    print("============config result============")
    Trainer.Router.main_fn(**config_dict)
    return 
if __name__ == "__main__":
    fire.Fire(main)

