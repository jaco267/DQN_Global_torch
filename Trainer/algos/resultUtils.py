import os
import torch as tc
class Result():
    def __init__(self) -> None:
       #** training results   (used in resultUtils.py)
        # episodic
        self.episode_reward = 0.0
        self.route_combo = []
        self.episode_two_pin_rewards = []
        # whole benchmark
        self.best_route = []  #solutions
        self.best_reward = 0.0
        self.PosTwoPinNum = 0#connected pin ex.43/50 32=PosTwoPinNum  50==total_pin=twopin_combo
    def update_episode_reward(self,reward):
        self.episode_reward += reward
    def update_pin_result(self,rewardfortwopin,route):
        self.episode_two_pin_rewards.append(rewardfortwopin)
        self.route_combo.append(route) #self.env.route
    def init_episode(self,agent):
        self.episode_reward = 0.0      #!!! reset results
        self.route_combo = []
        self.episode_two_pin_rewards = []
        agent.env.net_ind = 0
        agent.env.pair_ind = 0
    def end_episode(self,agent,logger,episode,results,twoPinNum):
        agent.env.capacity = agent.env.generate_capacity()
        assert twoPinNum == len(agent.env.twopin_combo)
        if sum([1 for item in self.episode_two_pin_rewards if item > 0]) > self.PosTwoPinNum:
            self.best_reward = self.episode_reward
            self.best_route = self.route_combo
            self.PosTwoPinNum = sum([1 for item in self.episode_two_pin_rewards if item > 0])
        elif sum([1 for item in self.episode_two_pin_rewards if item > 0]) == self.PosTwoPinNum:
            if self.episode_reward>self.best_reward:
                self.best_reward = self.episode_reward
                self.best_route = self.route_combo  
        successPinRate = self.PosTwoPinNum/len(agent.env.twopin_combo)  #todo use twopinNum
        logger.log({"successPinRate":successPinRate,"episodeReward":self.episode_reward})
        print(f'\rNew loop! Episode: {episode+1}/{agent.max_episodes} Reward: {self.episode_reward}  success pin: {self.PosTwoPinNum}/{len(agent.env.twopin_combo)}, success rate {successPinRate:.2f}',end="")

        results['reward_plot_combo'].append(self.episode_reward)
        reward_plot_pure = self.episode_reward-self.PosTwoPinNum*100
        results['reward_plot_combo_pure'].append(reward_plot_pure)


#*** utils
def load_ckpt_or_pass(agent,ckpt_path):
    if os.path.exists(ckpt_path):  #load chpt
        print(f"loading {ckpt_path}")
        statedict = tc.load(ckpt_path)   
        agent.dqn.load_state_dict(statedict)
def save_ckpt_fn(agent,ckpt_path,save_ckpt):
    if save_ckpt:                             #,
      tc.save(agent.dqn.state_dict(),ckpt_path)
      print(f"\nSave model, Model saved in path: {ckpt_path}")
    else:
      print("dont save model")

def make_solutions(self,twoPinNum,netSort,twoPinNumEachNet,results):
    solution = self.result.best_route
    assert len(solution) == twoPinNum
    for i in range(len(netSort)):
      results['solutionDRL'].append([])
    if self.result.PosTwoPinNum  == twoPinNum:   #,
      dumpPointer = 0
      for i in range(len(netSort)):
        netToDump = netSort[i]
        for j in range(twoPinNumEachNet[netToDump]):
          results['solutionDRL'][netToDump].append(solution[dumpPointer])
          dumpPointer = dumpPointer + 1
      success = 1
    else:
      results['solutionDRL'] = solution
      success = 0   
    return success,results,solution