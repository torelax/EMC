from modules.agents import REGISTRY as agent_REGISTRY
from modules.highlevel import REGISTRY as hlevel_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th


# This multi-agent controller shares parameters between agents
class HLevelMAC:
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        self.input_shape = self._get_input_shape(scheme)
        # self._build_agents(self.input_shape) # self.agent
        # todo 高层策略RNN网络加入上一时刻的g
        self._build_hlevel(self.input_shape) # self.hlevel
        self.agent_output_type = args.agent_output_type

        self.action_selector = action_REGISTRY[args.action_selector](args)

        self.hidden_states = None

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        if hasattr(self.args, 'use_individual_Q') and self.args.use_individual_Q:
            agent_outputs,_ = self.forward(ep_batch, t_ep, test_mode=test_mode)
        else:
            agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)
        return chosen_actions
    
    # avaliable goals
    def select_subgoal(self, ep_batch, t_ep, t_env=None, bs=slice(None), test_mode=False):
        # avail_actions = ep_batch["avail_actions"][:, t_ep]
        if hasattr(self.args, 'use_individual_Q') and self.args.use_individual_Q:
            goal_outputr,goal_outputc,_ = self.forward(ep_batch, t_ep, test_mode=test_mode)
        else:
            goal_outputr,goal_outputc = self.forward(ep_batch, t_ep, test_mode=test_mode)
        return th.cat([goal_outputr, goal_outputc], dim=-1)

    def forward(self, ep_batch, t, test_mode=False, batch_inf=False):
        agent_inputs = self._build_inputs(ep_batch, t, batch_inf)
        epi_len = t if batch_inf else 1

        # goal_outr, goal_outc, self.hidden_states = self.hlevel(agent_inputs, self.hidden_states)
        goal_outr, goal_outc = self.hlevel(agent_inputs, self.hidden_states)
        # goal_outs = th.cat([goal_outr, goal_outc], dim=-1)

        if batch_inf:
            return goal_outr.view(ep_batch.batch_size, self.n_agents, epi_len, -1).transpose(1, 2), \
                    goal_outc.view(ep_batch.batch_size, self.n_agents, epi_len, -1).transpose(1, 2)
        else:
            return goal_outr.view(ep_batch.batch_size, self.n_agents, -1), \
                    goal_outc.view(ep_batch.batch_size, self.n_agents, -1)

    def init_hidden(self, batch_size):
        self.hidden_states = self.hlevel.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav

    def parameters(self):
        return self.hlevel.parameters()

    def load_state(self, other_mac):
        self.hlevel.load_state_dict(other_mac.hlevel.state_dict())

    def cuda(self):
        self.hlevel.cuda()

    def to(self, *args, **kwargs):
        self.hlevel.to(*args, **kwargs)

    def save_models(self, path):
        th.save(self.hlevel.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        self.hlevel.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))

    def _build_agents(self, input_shape):
        self.hlevel = agent_REGISTRY[self.args.agent](input_shape, self.args)

    def _build_hlevel(self, input_shape):
        self.hlevel = hlevel_REGISTRY[self.args.hlevel](input_shape, self.args)


    def _build_inputs(self, batch, t, batch_inf):
        """
        ### @TODO 
            加入上一时刻的goal
        """
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        if batch_inf:
            bs = batch.batch_size
            inputs = []
            inputs.append(batch["obs"][:, :t, :, :23])  # bTav
            inputs.append(batch['Goal'][:, :t])
            # inputs.append(batch['goals'][:, :t:tdn])
            # current False
            if self.args.input_last_goal:
                last_goals = th.zero_like(batch["goals"][:,:t])
                last_goals[:, 1:] = batch["goals"][:, :t-1]
                inputs.append(last_goals)
            if self.args.obs_last_action:
                last_actions = th.zeros_like(batch["actions_onehot"][:, :t])
                last_actions[:, 1:] = batch["actions_onehot"][:, :t-1]
                inputs.append(last_actions)
            if self.args.obs_agent_id:
                inputs.append(th.eye(self.n_agents, device=batch.device).view(1, 1, self.n_agents, self.n_agents).expand(bs, t, -1, -1))

            inputs = th.cat([x.transpose(1, 2).reshape(bs*self.n_agents, t, -1) for x in inputs], dim=2)
            return inputs
        else:
            bs = batch.batch_size
            inputs = []
            inputs.append(batch["obs"][:, t, :, :23])  # b1av
            inputs.append(batch['Goal'][:, t])
            # (b, 1, 2, 46) -- (b, 2, 46)
            # print('input append: ', batch['obs'].shape)
            if self.args.obs_last_action:
                if t == 0:
                    inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
                else:
                    inputs.append(batch["actions_onehot"][:, t-1])
            if self.args.obs_agent_id:
                inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

            inputs = th.cat([x.reshape(bs * self.n_agents, 1, -1) for x in inputs], dim=2)
            return inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"] // 2 # 46
        input_shape += scheme["Goal"]["vshape"] # 2
        if self.args.input_last_goal:
            input_shape += scheme["subgoal"]['vshape'] # 46
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0] # 5
        if self.args.obs_agent_id: # 2
            input_shape += self.n_agents

        return input_shape
    
    def get_goal(obs=None):

        return []
