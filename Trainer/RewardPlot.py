import matplotlib.pyplot as plt
import numpy as np

NUM_POINTS = 300.0

def plot(prefix, rewards):
    x_gap = len(rewards) / NUM_POINTS
    x_vals = np.arange(0, len(rewards), x_gap).astype(int)
    rewards = np.array(rewards)

    for name, axis_label, func in \
        [('sum', 'Reward Sum (to date)', points_sum), \
         ('avg', 'Reward Average (next 100)', points_avg)]:
        y_vals = func(rewards, x_vals)
        for logscale in [True, False]:
            if logscale:
                plt.yscale('log')
            plt.plot(x_vals+1, y_vals)
            plt.xlabel('Loops')
            plt.ylabel(axis_label)
            plt.grid(which='Both')
            plt.tight_layout()
            plt.savefig(prefix + '_' + name + '_' + ('log' if logscale else 'lin') + '4by4simple.png')
            plt.close()

def points_sum(rewards, x_vals):
    return np.array([np.sum(rewards[0:val]) for val in x_vals])

def points_avg(rewards, x_vals):
    return np.array([np.sum(rewards[val:min(len(rewards), val+100)]) \
                     /float(min(len(rewards)-val, 100)) for val in x_vals])



if __name__ == '__main__':
    name = '4by4simple.gr.rewardDataBatch32LR1e-4M32_64_32.npy'
    # file_name = '../{}data/training_log.npz'.format(name)
    data = np.load(name)
    # reward_log = data['train_reward']
    
    plot('Reward plot',data)
#    n = data.shape[0]
#    plt.plot(data,'b-')
#    plt.show()
