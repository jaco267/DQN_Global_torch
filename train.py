import Trainer
import fire  


if __name__ == '__main__':
    # fire.Fire(Trainer.main_fn)  #todo rename it to Trainer.Router
    fire.Fire(Trainer.Router.main_fn) 