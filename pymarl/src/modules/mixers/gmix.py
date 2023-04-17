import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GMixer(nn.Module):
    '''
    goal-Mixer: 作为Critic拟合`V_tot(g)`
    '''
    def __init__(self, args):
        super(GMixer, self).__init__()

        self.args = args

        self.input_shape = self.args.goal_shape * self.args.n_agents
        self.fc1 = nn.Linear(self.input_shape, 32)
        # self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(32, 28)
        self.fc3 = nn.Linear(28, 1)
        # self.n_agents = args.n_agents

        self.state_dim = int(np.prod(args.state_shape))

        self.embed_dim = args.mixing_embed_dim


    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs):

        bs = inputs.shape[0] # batch size
        epi_len = inputs.shape[1] # episode size
        num_feat = inputs.shape[2] # goals * n_agent

        inputs = inputs.reshape(bs * epi_len, num_feat)

        x = F.relu(self.fc1(inputs))
        x2 = F.relu(self.fc2(x))
        v= F.relu(self.fc3(x2))
        # q = self.fc2(x)

        # q = q.reshape(bs, epi_len, 1)
        v = v.reshape(bs, epi_len, 1)
        return v