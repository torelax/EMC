REGISTRY = {}

from .rnn_agent import RNNAgent
from .sco_agent import SCOAgent
from .simple_agent import SimpleAgent
from .rnd_nn_agent import RND_nn_Agent
from .rnn_agent import RNNAgent
from .rnn_fast_agent import RNNFastAgent
from .rnd_history_agent import RNDHistoryAgent
from .rnn_individualQ_agent import RNN_individualQ_Agent
from .rnd_fast_history_agent import RND_Fast_historyAgent
from .rnn_fast_low_agent import RNNFastLowAgent



REGISTRY["rnn"] = RNNAgent
REGISTRY["rnn_fast"] = RNNFastAgent
REGISTRY["sco"] = SCOAgent
REGISTRY["simple"] = SimpleAgent
REGISTRY["rnd_nn"] = RND_nn_Agent
REGISTRY["rnd_history"] = RNDHistoryAgent
REGISTRY["rnn_individualQ"] = RNN_individualQ_Agent
REGISTRY["rnd_fast_history"] = RND_Fast_historyAgent

REGISTRY["rnn_fast_low"] = RNNFastLowAgent