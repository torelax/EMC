import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GMixer(nn.Module):
    '''
    ### input: 
    state[,Goal], {Q_subgoal}
    ### goal-Mixer: 
    作为G-mix拟合`Q_tot(s,g)`
    '''
    def __init__(self, args):
        super(GMixer, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = args.state_shape + args.Goal_shape * self.n_agents
        self.embed_dim = args.mixing_embed_dim
        
        # self.input_shape = (self.args.subgoal_shape + self.args.Goal_shape) * self.args.n_agents + self.args.state_shape // 2


        if getattr(args, "hypernet_layers", 1) == 1:
            self.hyper_w_1 = nn.Linear(self.state_dim, self.embed_dim * self.n_agents)
            self.hyper_w_final = nn.Linear(self.state_dim, self.embed_dim)
        elif getattr(args, "hypernet_layers", 1) == 2:
            hypernet_embed = self.args.hypernet_embed
            self.hyper_w_1 = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                           nn.ReLU(),
                                           nn.Linear(hypernet_embed, self.embed_dim * self.n_agents))
            self.hyper_w_final = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                           nn.ReLU(),
                                           nn.Linear(hypernet_embed, self.embed_dim))
        elif getattr(args, "hypernet_layers", 1) > 2:
            raise Exception("Sorry >2 hypernet layers is not implemented!")
        else:
            raise Exception("Error setting number of hypernet layers.")

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                               nn.ReLU(),
                               nn.Linear(self.embed_dim, 1))


    def forward(self, state, Goal, q_subgoal):

        states : th.Tensor = th.cat([state, Goal], dim=-1)
        # inputs = th.cat([state, Goal, q_subgoal], dim=-1)

        bs = state.shape[0] # batch size
        epi_len = state.shape[1] # episode size
        # num_feat = inputs.shape[2] # goals * n_agent

        states = states.reshape(bs * epi_len, self.state_dim)
        q_subgoal = q_subgoal.reshape(-1, 1, self.n_agents)

        w1 = th.abs(self.hyper_w_1(states))
        b1 = th.abs(self.hyper_b_1(states))
        w1 = w1.reshape(-1, self.n_agents, self.embed_dim)
        b1 = b1.reshape(-1, 1, self.embed_dim)
        hidden = F.elu(th.bmm(q_subgoal, w1) + b1)
        w_final = th.abs(self.hyper_w_final(states))
        w_final = w_final.reshape(-1, self.embed_dim, 1)

        v = self.V(states).reshape(-1, 1, 1)
        y = th.bmm(hidden, w_final) + v
        # q_tot = y.reshape(bs * epi_len, -1, 1)
        q_tot = y.reshape(bs, epi_len, 1)
        return q_tot