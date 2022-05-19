from .maml import MAML
from .meta_lstm import MetaLSTM
from .meta_sgd import MetaSGD
algos  = {
    "MAML":MAML,
    "TA_LSTM":MetaLSTM,
    "MetaSGD":MetaSGD
}