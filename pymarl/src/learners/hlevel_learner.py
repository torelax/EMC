import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
from modules.mixers.gmix import GMixer
from modules.critics.goalcritic import GoalCritic
import torch as th
from torch.optim import RMSprop
import torch.nn.functional as F
from utils.torch_utils import to_cuda
from utils.gumbel_softmax import gumbel_softmax

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
        self.g_optimiser = RMSprop(params=self.mac.parameters(), lr=args.goal_lr, alpha=args.optim_alpha, eps=args.optim_eps)
        self.m_optimiser = RMSprop(params=self.mixer.parameters(), lr=args.mixer_lr, alpha=args.optim_alpha, eps=args.optim_eps)
        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1

    # 目前所有的训练更新相关代码
    # reward TD(n) n 和 间隔有关
    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int, show_demo=False, save_data=None, show_v=False):
        # high HER
        """ batch 结构  
        reward batch-size, episode limit reward  \n
        state, subgoal, action, terminated
        """
        _batch = copy.deepcopy(batch)
        high_rewards = batch["reward"][:, :-1]
        for i in range(batch.batch_size):
            count = self.args.her_gap
            # TODO 终止状态前的变化
            endIndex = batch["terminated"][i].squeeze().tolist().index(1)
            for j in range(endIndex - self.args.her_gap):
                if j + count < batch.max_seq_length and batch['terminated'][i][j + count] == 0:
                    # _batch['subgoal'][i][j][0] = _batch['subgoals'][i][j + count][0][:self.args.subgoal_shape]
                    _batch['subgoals'][i][j][0] = _batch['subgoals'][i][j + count][0]
                    _batch['subgoals'][i][j][1] = _batch['subgoals'][i][j + count][1]
                    high_rewards[i][j] = 0
                    for k in range(count):
                        high_rewards[i][j] += (self.args.gamma ** k) * _batch['reward'][i][j + k]
                    count = (count + self.args.her_gap - 1) % self.args.her_gap
                else:
                    break
            for j in range(endIndex - self.args.her_gap, endIndex + 1):
                # _batch['subgoal'][i][j][0] = _batch['obs'][i][endIndex+1][0][:self.args.subgoal_shape]
                _batch['subgoals'][i][j][0] = _batch['subgoals'][i][endIndex+1][0]
                _batch['subgoals'][i][j][1] = _batch['subgoals'][i][endIndex+1][1]
                high_rewards[i][j] = 0
                for k in range(j, endIndex + 1):
                    high_rewards[i][j] += (self.args.gamma ** k) * _batch['reward'][i][k]
                # high_rewards[i][j] = th.sum(_batch['reward'][i][j : endIndex + 1])
                # pass

        # Get the relevant quantities
        avail_subgoals = batch['avail_subgoals']
        rewards = batch["reward"][:, :-1]
        obs = batch['obs'].reshape(batch.batch_size, batch.max_seq_length, -1)
        state = batch['state'].reshape(batch.batch_size, batch.max_seq_length, -1)
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])

        # batch['subgoal'][:, :-1] = batch['subgoal'][:, 1:]
        # print(batch['subgoals'].shape)
        subgoal = batch['subgoals'].reshape((batch.batch_size, batch.max_seq_length, -1, 1))[:,:-1,:]
        Goal = batch['Goal'].reshape((batch.batch_size, batch.max_seq_length, -1))[:,:-1]
        # states = batch['subgoal'].reshape((batch.batch_size, batch.max_seq_length, -1))

        self.mac.init_hidden(batch.batch_size)
        # subg_r, subg_c = self.mac.forward(_batch, batch.max_seq_length, batch_inf=True)
        # # Gumbel Softmax
        # subg_r = subg_r.reshape(batch.batch_size * batch.max_seq_length * self.args.n_agents, -1)
        # subg_c = subg_c.reshape(batch.batch_size * batch.max_seq_length * self.args.n_agents, -1)
        
        # subg_r = gumbel_softmax(subg_r)
        # subg_c = gumbel_softmax(subg_c)
        # subgoal = th.cat([subg_r, subg_c], dim=-1)
        sg_qvals = self.mac.forward(batch, batch.max_seq_length, batch_inf=True)
        # print(subgoal.shape, sg_qvals.shape)
        chosen_sg_qvals = th.gather(sg_qvals[:, :-1], dim=3, index=subgoal).squeeze(3)

        x_sg_qvals = sg_qvals.clone().detach()
        x_sg_qvals[avail_subgoals == 0] = -9999999
        max_subgoal_qvals, max_subgoal_index = x_sg_qvals[:, :-1].max(dim=3)

        max_subgoal_index = max_subgoal_index.detach().unsqueeze(3)
        is_max_action = (max_subgoal_index == subgoal).int().float()


        self.target_mac.init_hidden(batch.batch_size)
        target_sg_qvals = self.target_mac.forward(batch, batch.max_seq_length, batch_inf=True)[:, 1:, ...]

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            sg_qvals_detach = sg_qvals.clone().detach()
            sg_qvals_detach[avail_subgoals == 0] = -9999999
            cur_max_actions = sg_qvals_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_sg_qvals, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_sg_qvals.max(dim=3)[0]
        # Mixer
        Qtot_sg : th.Tensor = self.mixer(state[:, :-1], Goal, chosen_sg_qvals)
        # print('obs', obs[-1,3])
        # print('subg_', subg_[-1,3])
        # print('vtot_goal', self.mixer(subg_)[-1,3])
        # _target_Vtot_obs = self.target_mixer(obs[:, 1:])

        # @TODO off policy correction
        # max_{g'}Q'(s', g')---> Q_target(s', pi_h(s'))
        # subg_r, subg_c = self.target_mac.forward(_batch, batch.max_seq_length, batch_inf=True)

        # subg_r = subg_r.reshape(batch.batch_size * batch.max_seq_length * self.args.n_agents, -1)
        # subg_c = subg_c.reshape(batch.batch_size * batch.max_seq_length * self.args.n_agents, -1)
        # # Gumbel Softmax
        # # TODO next_sg 不需要gumbel softmax保留梯度
        # subg_r = gumbel_softmax(subg_r)
        # subg_c = gumbel_softmax(subg_c)

        # next_sg = th.cat([subg_r, subg_c], dim=-1)
        # next_sg = next_sg.reshape(batch.batch_size, batch.max_seq_length, -1)[:, 1:, ...]
        # next_sg = self.target_mac.forward(_batch, batch.max_seq_length, batch_inf=True)
        targetQtot_sg : th.Tensor = self.target_mixer(state[:, 1:], Goal, target_max_qvals)

        # print(terminated.shape)
        # print(Qtot_sg.shape, targetQtot_sg.shape, high_rewards.shape, terminated.shape)
        target = high_rewards + self.args.gamma * (1 - terminated) * targetQtot_sg
        td_error = (target.detach() - Qtot_sg)
        # masked_hit_prob = th.mean(is_max_action, dim=2) * mask
        # hit_prob = masked_hit_prob.sum() / mask.sum()
        mask = mask.expand_as(td_error)
        masked_td_error = td_error * mask

        # print('--->masked td error[-1]:\n', masked_td_error[-1])
        mixer_loss = (masked_td_error ** 2).sum() / mask.sum()

        """ if show_v:
            mask_elems = mask.sum().item()

            actual_v = rewards.clone().detach()
            for t in reversed(range(rewards.shape[1] - 1)):
                actual_v[:, t] += self.args.gamma * actual_v[:, t + 1]
            self.logger.log_stat("test_actual_return", (actual_v * mask).sum().item() / mask_elems, t_env)

            # self.logger.log_stat("test_q_taken_mean", (chosen_action_qvals * mask).sum().item()/mask_elems, t_env)
            return """

        # Optimise
        self.m_optimiser.zero_grad()
        mixer_loss.backward()
        m_grad_norm = th.nn.utils.clip_grad_norm_(self.mixer.parameters(), self.args.grad_norm_clip)
        self.m_optimiser.step()

        # lowLoss: -Q(S, g, G)
        # subgoal_loss : th.Tensor = -th.log(self.target_mixer(obs[:, :-1], Goal[:, :-1], subgoal[:, :-1]))
        # masked_sg_loss = (subgoal_loss * mask).sum() / mask.sum()
        # self.g_optimiser.zero_grad()
        # masked_sg_loss.backward()
        # for parms in self.mac.parameters():	
        #         # print('-->name:', name)
        #         print('-->para:', parms)
        #         print('-->grad_requirs:',parms.requires_grad)
        #         print('-->grad_value:',parms.grad)
        #         print("===")
        # g_grad_norm = th.nn.utils.clip_grad_norm_(self.mac.parameters(), self.args.grad_norm_clip)
        self.g_optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            print('-->goal_loss:\n', mixer_loss)
            self.logger.log_stat("hlevel:mixer_loss", mixer_loss.item(), t_env)
            # self.logger.log_stat("hlevel:subgoal_loss", masked_sg_loss.item(), t_env)
            # self.logger.log_stat("hlevel:g_grad_norm", g_grad_norm, t_env)
            self.logger.log_stat("hlevel:m_grad_norm", m_grad_norm, t_env)
            # self.logger.log_stat("Mixer-Vtot(5,5,5,6)", vtot5556, t_env)
            mask_elems = mask.sum().item()
            # self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            # self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            # self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.log_stats_t = t_env


        if self.args.is_prioritized_buffer:
            res = th.sum(masked_td_error ** 2, dim=(1, 2)) / th.sum(mask, dim=(1, 2))
            res = res.cpu().detach().numpy()
            return res

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated Goal and Mixer Target Network")

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
