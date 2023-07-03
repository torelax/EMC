import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GoalCritic(nn.Module):
    '''
    ### input: 
    state, tot_subgoal
    ### goal-Mixer: 
    作为Critic拟合`Q_tot(s,g)`
    '''
    def __init__(self, args):
        super(GoalCritic(), self).__init__()

        self.args = args

        # self.input_shape = self.args.goal_shape * self.args.n_agents + self.args.state_shape + self.args.dgoal_shape
        self.input_shape = (self.args.subgoal_shape + self.args.Goal_shape) * self.args.n_agents + self.args.state_shape // 2

        self.state_dim = args.state_shape
        self.embed_dim = args.mixing_embed_dim

        """ if getattr(args, "hypernet_layers", 1) == 1:

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
            raise Exception("Error setting number of hypernet layers.") """

        self.fc1 = nn.Linear(self.input_shape, 128)
        # self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        # self.n_agents = args.n_agents
        

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, state, Goal, subgoal):
        inputs = th.cat([state, Goal, subgoal], dim=-1)

        bs = inputs.shape[0] # batch size
        epi_len = inputs.shape[1] # episode size
        num_feat = inputs.shape[2] # goals * n_agent
        # states = states.reshape(bs * epi_len, self.state_dim)
        inputs = inputs.reshape(bs * epi_len, num_feat)

        x = F.leaky_relu(self.fc1(inputs))
        x = F.leaky_relu(self.fc2(x))
        Q = self.fc3(x)
        # q = self.fc2(x)

        # q = q.reshape(bs, epi_len, 1)
        Q = Q.reshape(bs, epi_len, 1)
        return Q