import numpy as np
from .trainer import Trainer
import learn2learn as l2l 
import torch
from .utils import evaluate_learner, fast_adapt

class MAML(Trainer):
    """The Model Agnostic Meta-Learning algorithm.

    Args:
        Trainer (Trainer): Extends The Base Trainer Class
    """
    def __init__(self, args):
        self.first_order = True
        super(MAML, self).__init__(args)

    def get_init(self):
        return self.meta_learner.clone().get_flat_params()

    def set_init(self,flat_params):
        return self.meta_learner.copy_flat_params(flat_params)

    def adapt(self,batch,learner=None):
        learner = self.meta_learner.clone() if learner is None else learner
        return fast_adapt(batch,learner,self.loss,self.adaptation_steps,self.shots,self.ways,self.dev,self.dataset)

    def evaluate_learner(self,batch,learner=None):
        learner = self.meta_learner.clone() if learner is None else learner
        with torch.no_grad():
            evaluation_error, evaluation_accuracy = evaluate_learner(batch,learner,self.loss,self.shots,self.ways,self.dev,self.dataset)
        return evaluation_error, evaluation_accuracy 
    
    def build_metalearner(self):
        return l2l.algorithms.MAML(self.base_learner, lr=self.base_lr, first_order=self.first_order,allow_nograd=True)


