
import random 

class Config:
    def __init__(self):
        self.mode             = 'test'                  # The experiment mode: whether to meta-train or meta-test
        self.algo             = 'MAML'                  # The meta-learning algorithm
        self.dataset          = 'omniglot'              # The few-shot learning dataset
        self.n_channels       = 1                       # The no. of channels in the image for the considered dataset
        self.n_filters        = 64                      # The no. of convolutional filters in the backbone
        self.image_size       = 28                      # The size of the images for the considered dataset
        self.n_shot           = 1                       # The no. of shots per class in the few shot learning task
        self.n_class          = 20                      # The no. classes in the few shot learning task
        self.meta_batch_size  = 4                       # The meta-batch size for the meta-learning algorithm
        self.input_size       = 4                       # The size of input to the meta-model (only applicable for learned optimizer)
        self.hidden_size      = 20                      # The hidden size for the meta-model (only applicable for learned optimizer)
        self.meta_lr          = 5e-2                    # The learning rate for the meta-model
        self.base_lr          = 1e-2                    # The learning rate for the base-model (backbone)
        self.num_iterations   = 10000                   # The no. of meta-training iterations
        self.adaptation_steps = 8                       # The no. of adaptation steps that the base model should take for each task
        self.bn_momentum      = 0.95                    # The batch norm momentum for MetaLSTM++
        self.bn_eps           = 1e-3                    # The batch norm epsilon for MetaLSTM++
        self.val_freq         = 250                     # The number of iterations after which to run meta-validation
        self.resume           = True                    # Flag to specify whether to resume from a checkpoint
        self.gpu              = 0                       # The gpu to use
        self.seed             = random.randint(0, 1e3)  # The random seed for the experiment
        self.test_size        = 300                     # The no. of tasks to use for meta testing and validation
        self.ckpt_no          = None                    # The ckpt_no to resume from. If ckpt_no=None and resume=True, the latest checkpoint is loaded.
