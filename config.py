
import random 

class Config:
    def __init__(self):
        self.mode             = 'test'
        self.algo             = 'MAML'
        self.dataset          = 'omniglot'
        self.n_channels       = 1
        self.n_filters        = 64
        self.image_size       = 28
        self.n_shot           = 1
        self.n_class          = 20
        self.meta_batch_size  = 4
        self.input_size       = 4
        self.hidden_size      = 20
        self.meta_lr          = 5e-2
        self.base_lr          = 1e-2
        self.num_iterations   = 10000
        self.adaptation_steps = 8
        self.weigh_task       = False
        self.grad_clip        = 0.25
        self.bn_momentum      = 0.95
        self.lamda            = 0.05
        self.bn_eps           = 1e-3
        self.pin_mem          = True
        self.val_freq         = 250
        self.resume           = True
        self.clear_logs       = True
        self.user             = "Sahil"
        self.gpu              = 0
        self.seed             = random.randint(0, 1e3)
        self.test_size        = 300
        self.ckpt_no          = None
