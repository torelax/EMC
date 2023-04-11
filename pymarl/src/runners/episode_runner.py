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
        self.env.reset()
        self.t = 0

    def cal_low_reward(self, goals, obs, SR=None):
        '''
        obs: [1,2,46] np list
        goals: [1,2,23] tensor
        '''
        t_goals = goals.clone().detach()
        rewards = 0

        # print(t_goals.shape)
        # print('obs: ', obs[0].shape)
        # print(obs[0][0][:self.args.goal_shape])
        # print(t_goals[0][0].numpy())
        
        if SR:
            rewards = [SR[obs][goals]]
        else: # 之间计算goal 和 state的距离
            for i in range(self.env.n_agents):
                # rewards.append()
                reward = np.linalg.norm(t_goals[0][i].numpy() - obs[0][i][:self.args.goal_shape], ord=2)
                rewards -= reward

        return rewards

    def arrive_goal(self, goal, obs):
        pass

    # todo HER
    def run(self, test_mode=False):
        self.reset()

        terminated = False
        episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size)
        self.action_mac.init_hidden(batch_size=self.batch_size)
        goals = []

        while not terminated:

            # for i in range(self.args.gener_goal_interval):
            #     pass

            # highlevel 输出 goal
            if self.t % self.args.gener_goal_interval == 0:
                goals = self.mac.select_goals(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)

            # print('goals: ', goals.shape) # 1,2,23
            # print('obs: ' , th.tensor([self.env.get_obs()]).shape)
            # print('state: ', th.tensor([self.env.get_state()]).shape)

            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()], # 1, 2, 46
                # 所有智能体的目标goal
                "goals": goals  # 预期 1, 2, 23
            }

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
            # next_state = self.env.get_state()
            next_obs = [self.env.get_obs()]
            # low reward底层回报 每个agent获得一个
            # goals [1,2,23] obs [1,2,46]
            low_reward = [self.cal_low_reward(goals, next_obs)]
            # low_reward = self.env.cal_low_reward(goals, next_state)
            
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

        # if test_mode and (len(self.test_returns) == self.args.test_nepisode):
        #     self._log(cur_returns, cur_stats, log_prefix)
        # elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
        #     self._log(cur_returns, cur_stats, log_prefix)
        #     if hasattr(self.mac.action_selector, "epsilon"):
        #         self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
        #     self.log_train_stats_t = self.t_env

        return self.batch

    """ def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear() """
