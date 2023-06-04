from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np

from envs.GridworldEnv import GridworldEnv
from controllers.hlevel_controller import HLevelMAC
from controllers.llevel_controller import LLevelMAC

import torch as th

class EpisodeRunner:

    def __init__(self, args, logger): 
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        if 'stag_hunt' in self.args.env:
            self.env = env_REGISTRY[self.args.env](env_args=self.args.env_args, args=args)
        else:
            self.env : GridworldEnv = env_REGISTRY[self.args.env](**self.args.env_args)
        self.episode_limit = self.env.episode_limit
        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000

    def setup(self, scheme, groups, preprocess, mac, action_mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac : HLevelMAC = mac
        self.action_mac : LLevelMAC = action_mac

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch : EpisodeBatch = self.new_batch()
        desired_goal = self.env.reset()
        self.t = 0
        return desired_goal

    def cal_low_reward(self, subgoal, obs, test_mode=False, SR=None):
        ''''''
        t_subgoal = subgoal.clone().detach()
        rewards = []

        if SR:
            rewards = [SR[obs][subgoal]]
        else: # 之间计算goal 和 state的距离
            for i in range(self.env.n_agents):
                grow, gcol = th.argmax(t_subgoal[0][i][:11]), th.argmax(t_subgoal[0][i][11:])
                # grow, gcol = t_subgoal[0][i][0], t_subgoal[0][i][1]
                crow, ccol = np.argmax(obs[0][i][:11]), np.argmax(obs[0][i][11:23])
                if test_mode:
                    print('Agent %d Row and Cols', i, grow, gcol, crow, ccol)
                rewards.append(-np.linalg.norm([grow - crow, gcol - ccol], ord=2) / 10)
                # rewards.append()
                if np.linalg.norm([grow - crow, gcol - ccol], ord=2) < self.args.arrive_g_threshold:
                    rewards[i] += 2
        return rewards

    def arrive_goal(self, subgoal, obs):
        '''计算范围向下取整'''
        t_subgoal = subgoal.clone().detach()
        for i in range(self.env.n_agents):
            grow, gcol = th.argmax(t_subgoal[0][i][:11]), th.argmax(t_subgoal[0][i][11:])
            # grow, gcol = t_subgoal[0][i][0] // 1, t_subgoal[0][i][1] // 1
            goal = t_subgoal[0][i]
            crow, ccol = np.argmax(obs[0][i][:11]), np.argmax(obs[0][i][11:23])

            # if np.linalg.norm(t_subgoal[0][i].numpy() - obs[0][i][:self.args.goal_shape], ord=2) > self.args.arrive_g_threshold:
            if np.linalg.norm([grow - crow, gcol - ccol], ord=2) < self.args.arrive_g_threshold:    
                return True
        return False

    # todo HER
    def run(self, test_mode=False):
        desired_goal = self.reset()

        terminated = False
        episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size)
        self.action_mac.init_hidden(batch_size=self.batch_size)
        subgoal = []
        p = counts = 0

        while not terminated:

            # for i in range(self.args.gener_goal_interval):
            #     pass

            # highlevel 输出 goal
            # if self.t % self.args.gener_goal_interval == 0 or self.arrive_goal(subgoal, [self.env.get_obs()]):
            if subgoal == [] or self.arrive_goal(subgoal, [self.env.get_obs()]) or self.t % self.args.gener_goal_interval == 0:
                subgoal = self.mac.select_subgoal(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
                p += self.arrive_goal(subgoal, [self.env.get_obs()])
                counts += 1
                # print('Goal and subgoal', desired_goal, subgoal)

            # subgoal = th.tensor([[[5., 5.],[5., 6.]]])
            # print('subgoal: ', subgoal.shape) # 1,2,23
            # print('obs: ' , th.tensor([self.env.get_obs()]).shape)
            # print('state: ', th.tensor([self.env.get_state()]).shape)
            # _obs = [self.env.get_obs()]

            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()], # 1, 2, 46
                "Goal": [desired_goal],
                # 所有智能体的目标goal
                "subgoal": subgoal  # 预期 1, 4
            }
            # print(desired_goal)
            # get_obs() --> (2, 23)
            # print('trans data[\'obs\']', th.tensor([self.env.get_obs()]).shape)  # 1, 2, 23
            # print('trans data[\'avail_actions\']', th.tensor([self.env.get_avail_actions()]).shape)  # 1, 2, 5
            # print(self.env.get_obs()) # (2， 23)

            self.batch.update(pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            
            # lowlevel agent 输出 action
            actions = self.action_mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
             
            reward, terminated, env_info = self.env.step(actions[0])

            if reward >= 50:
                print('------->Won and Get Reward: 50')
            # elif reward >= 30:
            #     print('------->Get Reward 30')
            # elif reward > 20:
            #     print('------->Get Reward 20')
            # elif reward == 30:
            #     print('left agent get: 30')
            # elif reward == 20:
            #     print('right agent get 20')
            # next_state = self.env.get_state()
            next_obs = [self.env.get_obs()]
            # low reward底层回报 每个agent获得一个
            # subgoal [1,2,23] obs [1,2,46]
            low_reward = self.cal_low_reward(subgoal, next_obs, test_mode)
            # low_reward = self.env.cal_low_reward(subgoal, next_state)
            
            episode_return += reward

            post_transition_data = {
                "actions": actions,
                "reward": [(reward,)], # high_reward
                "low_reward": [(low_reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }
            
            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1

        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()]
        }
        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        actions = self.action_mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        self.batch.update({"actions": actions}, ts=self.t)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t

        cur_returns.append(episode_return)

        # if test_mode:
        #     mac_out = self.mac.forward(self.batch, 31, test_mode=True)
        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.action_mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        # HER修正
        

        return self.batch, p, counts

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()
