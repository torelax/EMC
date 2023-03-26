import torch.nn as nn
import torch.nn.functional as F

class HighLevelNoSR(nn.moudle):
    """
    不带 SR 的版本 高层直接输出 goal
    目标: 最大E[R(s)]
    每隔t-step输出一个n个智能体每个的子目标sub-goal
    使用RNN
    """
    def __init__(self, input_shape, args) -> None:
        super(HighLevelNoSR).__init__()

        self.args = args
        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        # self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.rnn = nn.GRU(
            input_size=args.rnn_hidden_dim,
            num_layers=1,
            hidden_size=args.rnn_hidden_dim,
            batch_first=True
        )
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_agents * args.obs_shape)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        '''
        Return: [batch, n-agent's state]
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
        q = self.fc2(x)
        # 对n个状态s/观察obs输出Q(g, a)
        q = q.reshape(bs, epi_len, self.args.n_agents * self.args.obs_shape)
        return q, h
