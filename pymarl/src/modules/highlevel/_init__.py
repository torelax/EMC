REGISTRY = {}

from .rnn_highlevel_sr import HighLevelSR
from .rnn_hlevel_no_sr import HighLevelNoSR

REGISTRY['highlevel_sr'] = HighLevelSR
REGISTRY['hlevel_nosr'] = HighLevelNoSR