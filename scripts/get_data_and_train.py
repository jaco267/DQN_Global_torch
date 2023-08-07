import os 
import fire
def c(cmd_str):
    os.system(cmd_str)
def generate_train_eval_data():
    print("---generating train data---")
    c('python gen_data.py --benchmarkNumber 30 --gridSize 8 --netNum 20 --vCap 4  --hCap 4 --maxPinNum 5 --reducedCapNum 3 --prefix ./train_data_/')

    print("\n---generating test data---")
    c('python gen_data.py --benchmarkNumber 20 --gridSize 8 --netNum 20 --vCap 4  --hCap 4 --maxPinNum 5 --reducedCapNum 3 --prefix ./test_data_/')

def train_eval_plot(algos):
    print(f"\n---start training  {algos}---")
    c(f'python run.py --mode "train" --algos {algos} --run_benchmark_num=30')
    print("\n---start eval dtqn---")
    c(f'python run.py --mode "eval" --algos {algos}')
    
    print("cd to ./eval")
    os.chdir('./eval')
    print(os.getcwd())
    
    c('python Evaluation.py ')
    c(f'python VisualizeResults.py --save_dir VisualizeResult_{algos}')
    
    print("back to root folder")
    os.chdir('..')
    print(os.getcwd())


generate_train_eval_data()

train_eval_plot('dtqn')
train_eval_plot('dqn')

