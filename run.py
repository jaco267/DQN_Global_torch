import Trainer
import fire  
import yaml
if __name__ == '__main__':
    def get_config_dict(mode):
        config_dict = {}
        with open("configs.yaml",'r') as f:
            data = yaml.safe_load(f)
        # print("=============update default============")
        default_all_dict = data['default']['all']
        # print(default_all_dict)
        config_dict.update(default_all_dict)    
        # print(f"-----{mode} update----")
        default_train_eval_dict = data['default'][mode]
        # print(default_train_eval_dict)
        config_dict.update(default_train_eval_dict) 
        return config_dict       
    def main_wrapper(**cmd_configs):
        if 'mode' not in cmd_configs:
            mode = 'train'
        else:
            mode = cmd_configs['mode']
        mode_list =  ['eval','train'] 
        if mode not in mode_list:
            raise Exception(f'mode should be in {mode_list}') 
        config_dict = get_config_dict(mode)
        config_dict.update(cmd_configs)
        # print(config_dict)
        Trainer.Router.main_fn(**config_dict)  
        return
    fire.Fire(main_wrapper)

    # fire.Fire(Trainer.Router.main_fn)   #Trainer/Router.py