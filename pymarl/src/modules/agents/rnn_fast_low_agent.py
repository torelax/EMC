import torch.nn as nn
import torch.nn.functional as F



class RNNFastLowAgent(nn.Module):
    '''
    原有基础上添加高层输出的目标作为输入
    回报定义为 目标 和 实际到达状态的"距离"
    '''

    def __init__(self, input_shape, args):
        '''
        '''
        super(RNNFastLowAgent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.lowlevel_nn_dim)
        # self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.rnn = nn.GRU(
            input_size=args.rnn_hidden_dim,
            num_layers=1,
            hidden_size=args.rnn_hidden_dim,
            batch_first=True
        )
        self.fc2 = nn.Linear(args.lowlevel_nn_dim, 64)
        self.fc3 = nn.Linear(64, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        bs = inputs.shape[0]
        epi_len = inputs.shape[1]
        num_feat = inputs.shape[2]
        inputs = inputs.reshape(bs * epi_len, num_feat)
        x = F.leaky_relu(self.fc1(inputs))
        # x = x.reshape(bs, epi_len, self.args.rnn_hidden_dim)
        # h_in = hidden_state.reshape(1, bs, self.args.rnn_hidden_dim).contiguous()
        # x, h = self.rnn(x, h_in)
        q = F.leaky_relu(self.fc2(x))
        q = self.fc3(q)
        # q = F.softmax(self.fc3(q))
        q = q.reshape(bs, epi_len, self.args.n_actions)
        return q
