import torch.nn as nn
import torch.nn.functional as F

import torch as th

class HighLevelNoSR(nn.Module):
    """
    Actor--pi_h(s,g)
    不带 SR 的版本 高层直接输出 goal
    目标: 最大E[Q(s, g)] Q(s,g)-->Critic
    每隔t-step输出一个n个智能体每个的子目标sub-goal
    使用RNN
    """
    def __init__(self, input_shape, args) -> None:
        super(HighLevelNoSR, self).__init__()

        self.args = args
        self.fc1 = nn.Linear(input_shape, args.pih_dim)
        # self.rnn = nn.GRUCell(args.pih_dim, args.pih_dim)
        self.rnn = nn.GRU(
            input_size=args.pih_dim,
            num_layers=1,
            hidden_size=args.pih_dim,
            batch_first=True
        )
        """ self.goalrow = nn.Sequential(nn.Linear(args.pih_dim, args.pih_dim // 2),
                        nn.LeakyReLU(),
                        nn.Linear(args.pih_dim // 2, args.env_args['input_rows']))
        self.goalcol = nn.Sequential(nn.Linear(args.pih_dim, args.pih_dim // 2),
                        nn.LeakyReLU(),
                        nn.Linear(args.pih_dim // 2, args.env_args['input_cols'])) """
        
        self.fc2 = nn.Linear(args.pih_dim, args.n_subgoals)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.pih_dim).zero_()

    def forward(self, inputs, hidden_state):
        '''
        input: state/obs, Goal
        Return: [batch, n-agent's subgoal]
        '''
        bs = inputs.shape[0]    # batch size
        epi_len = inputs.shape[1]   # episode lenth?
        num_feat = inputs.shape[2]  # n agent?
        inputs = inputs.reshape(bs * epi_len, num_feat)   
        x = F.leaky_relu(self.fc1(inputs))
        x = x.reshape(bs, epi_len, self.args.pih_dim)
        h_in = hidden_state.reshape(1, bs, self.args.pih_dim).contiguous()
        x, h = self.rnn(x, h_in)
        x = x.reshape(bs * epi_len, self.args.pih_dim)

        # rows = F.softmax(self.goalrow(x), dim=-1)
        # cols = F.softmax(self.goalcol(x), dim=-1)
        # g = F.softmax(self.goalNN(x), dim=1)
        # g = th.cat([rows, cols], dim=-1)
        # g = g.reshape(bs, epi_len, -1)
        q = self.fc2(x)
        q = q.reshape(bs, epi_len, self.args.n_subgoals)

        return q
