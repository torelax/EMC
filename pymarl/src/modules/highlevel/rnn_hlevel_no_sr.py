import torch.nn as nn
import torch.nn.functional as F

import torch as th

class HighLevelNoSR(nn.Module):
    """
    不带 SR 的版本 高层直接输出 goal
    目标: 最大E[R(s)]
    每隔t-step输出一个n个智能体每个的子目标sub-goal
    使用RNN
    """
    def __init__(self, input_shape, args) -> None:
        super(HighLevelNoSR, self).__init__()

        self.args = args
        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        # self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.rnn = nn.GRU(
            input_size=args.rnn_hidden_dim,
            num_layers=1,
            hidden_size=args.rnn_hidden_dim,
            batch_first=True
        )
        # self.goalrow = nn.Linear(args.rnn_hidden_dim, args.env_args['input_rows'])
        # self.goalcol = nn.Linear(args.rnn_hidden_dim, args.env_args['input_cols'])
        # self.goalrow = nn.Linear(args.rnn_hidden_dim, 1)
        # self.goalcol = nn.Linear(args.rnn_hidden_dim, 1)
        self.goalNN = nn.Linear(args.rnn_hidden_dim, 2)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        '''
        Return: [batch, n-agent's goals]
        '''
        bs = inputs.shape[0]    # batch size
        epi_len = inputs.shape[1]   # episode lenth?
        num_feat = inputs.shape[2]  # n agent?
        inputs = inputs.reshape(bs * epi_len, num_feat)   
        x = F.relu(self.fc1(inputs))
        x = x.reshape(bs, epi_len, self.args.rnn_hidden_dim)
        h_in = hidden_state.reshape(1, bs, self.args.rnn_hidden_dim).contiguous()
        x, h = self.rnn(x, h_in)
        x = x.reshape(bs * epi_len, self.args.rnn_hidden_dim)
        # rows = F.softmax(self.goalrow(x))
        # cols = F.softmax(self.goalcol(x))
        g = self.goalNN(x)
        # 对n个状态s/观察obs输出Q(g, a)
        # g = th.cat([rows, cols], dim=-1)
        g = g.reshape(bs, epi_len, 2)
        # rows = rows.reshape(bs, epi_len, 1)
        # cols = cols.reshape(bs, epi_len, 1)
        return g, h
