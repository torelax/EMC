import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
from modules.mixers.gmix import GMixer
import torch as th
from torch.optim import RMSprop
import torch.nn.functional as F
from utils.torch_utils import to_cuda

from controllers.hlevel_controller import HLevelMAC

import numpy as np

class HLevelLearner:
    def __init__(self, mac, scheme, logger, args, groups=None):
        self.args = args
        self.mac : HLevelMAC = mac
        self.logger = logger

        self.params = list(mac.parameters())

        self.last_target_update_episode = 0

        self.mixer = None
        # self.mixer_input_shape = self.args.obs_shape
        if args.mixer is not None:
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                self.mixer = QMixer(args)
            elif args.mixer == 'gmix':
                self.mixer = GMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            # self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        # self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        self.g_optimiser = RMSprop(params=self.mac.parameters(), lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        self.m_optimiser = RMSprop(params=self.mixer.parameters(), lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1

    # 目前所有的训练更新相关代码
    # reward TD(n) n 和 间隔有关
    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int, show_demo=False, save_data=None, show_v=False):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])

        # 简单的HER实现
        batch['goals'][:, :-1] = batch['goals'][:, 1:]
        goals = batch['goals'].reshape((batch.batch_size, batch.max_seq_length, -1))

        # 或者直接拿obs 拼成goal
        # Calculate estimated Q-Values
        # 初始化隐变量
        self.mac.init_hidden(batch.batch_size)
        # rnn_fast_agent.forward(batch, batch.max_seq_lenth)
        # hlevel_mac obs --> goals 
        goals_out = self.mac.forward(batch, batch.max_seq_length, batch_inf=True)

        # if show_demo:
            # q_i_data = chosen_action_qvals.detach().cpu().numpy()
            # q_data = (max_action_qvals - chosen_action_qvals).detach().cpu().numpy()

        # Calculate the Q-Values necessary for the target
        self.target_mac.init_hidden(batch.batch_size)
        target_goals_out = self.target_mac.forward(batch, batch.max_seq_length, batch_inf=True)[:, 1:, ...]

        # Max over target Q-Values
        # if self.args.double_q:
        #     # Get actions that maximise live Q (for double q-learning)
        #     mac_out_detach = mac_out.clone().detach()
        #     mac_out_detach[avail_actions == 0] = -9999999
        #     cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
        #     target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        # else:
        #     target_max_qvals = target_mac_out.max(dim=3)[0]

        # Mix

        # Solution 2 拿s,r,a,o来训练
        Vtot_obs = self.mixer(batch['obs'][:, :-1, 0])
        target_Vtot_obs = self.target_mixer(batch['obs'][:, 1:, 0])

        target = rewards + self.args.gamma * (1 - terminated) * target_Vtot_obs
        mixer_loss = th.mean(F.mse_loss(Vtot_obs, target))

        # Solution 1 拿HER后的goal训练

        # if self.mixer is not None:
        #     # chosen_action_qvals = self.mixer(chosen_action_qvals, batch["obs"][:, :-1])
        #     # 输入goals 输出V(goals)
        #     v_goals = self.mixer(goals_out) # V_tot(goal)

        #     target_v_goals = self.target_mixer(target_goals_out, batch["obs"][:, 1:])
        # V(g) = V(g) + \alpha ( r + \gammaV(g') - V(g) )
        # targets = rewards + self.args.gamma * (1 - terminated) * target_v_goals

        """ if show_demo:
            tot_q_data = chosen_action_qvals.detach().cpu().numpy()
            tot_target = targets.detach().cpu().numpy()
            if self.mixer == None:
                tot_q_data = np.mean(tot_q_data, axis=2)
                tot_target = np.mean(tot_target, axis=2)

            print('action_pair_%d_%d' % (save_data[0], save_data[1]), np.squeeze(q_data[:, 0]),
                  np.squeeze(q_i_data[:, 0]), np.squeeze(tot_q_data[:, 0]), np.squeeze(tot_target[:, 0]))
            self.logger.log_stat('action_pair_%d_%d' % (save_data[0], save_data[1]),
                                 np.squeeze(tot_q_data[:, 0]), t_env)
            return """

        # Td-error loss
        # td_error = (v_goals - targets.detach())

        # loss = 1 //2 * (targets - v_goals).sum() // 
        # mixer_loss = th.mean(F.mse_loss(targets - v_goals))

        # mask = mask.expand_as(td_error)

        if show_v:
            mask_elems = mask.sum().item()

            actual_v = rewards.clone().detach()
            for t in reversed(range(rewards.shape[1] - 1)):
                actual_v[:, t] += self.args.gamma * actual_v[:, t + 1]
            self.logger.log_stat("test_actual_return", (actual_v * mask).sum().item() / mask_elems, t_env)

            # self.logger.log_stat("test_q_taken_mean", (chosen_action_qvals * mask).sum().item()/mask_elems, t_env)
            return

        # 0-out the targets that came from padded data
        # masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        # loss = (masked_td_error ** 2).sum() / mask.sum()

        # masked_hit_prob = th.mean(is_max_action, dim=2) * mask
        # hit_prob = masked_hit_prob.sum() / mask.sum()

        # Optimise
        goal_loss : th.Tensor = th.mean(-self.mixer(goals[:, :-1]))
        self.g_optimiser.zero_grad()
        goal_loss.backward()
        self.g_optimiser.step()

        self.m_optimiser.zero_grad()
        mixer_loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.mixer.parameters(), self.args.grad_norm_clip)
        self.m_optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("hlevel:mixer_loss", mixer_loss.item(), t_env)
            self.logger.log_stat("hlevel:goal_loss", goal_loss.item(), t_env)

            self.logger.log_stat("hlevel:grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            # self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            # self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            # self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.log_stats_t = t_env

        return mixer_loss

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        to_cuda(self.mac, self.args.device)
        to_cuda(self.target_mac, self.args.device)
        if self.mixer is not None:
            to_cuda(self.mixer, self.args.device)
            to_cuda(self.target_mixer, self.args.device)

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.g_optimiser.state_dict(), "{}/goal_opt.th".format(path))
        th.save(self.m_optimiser.state_dict(), "{}/mixer_opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.g_optimiser.load_state_dict(th.load("{}/goal_opt.th".format(path), map_location=lambda storage, loc: storage))
        self.m_optimiser.load_state_dict(th.load("{}/mixer_opt.th".format(path), map_location=lambda storage, loc: storage))
