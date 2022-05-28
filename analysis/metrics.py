import torch 
import numpy as np
from scipy.stats import entropy
from abc import ABC, abstractmethod
from typing import Union, List
from tensorflow_similarity.losses import SoftNearestNeighborLoss as LSN
import tensorflow as tf 

class PathLength(torch.nn.Module):
    def __init__(self, norm_degree=2):
        super().__init__()
        self.norm_degree = norm_degree
        self.distance_metric = torch.nn.PairwiseDistance(self.norm_degree)
    
    def compute_length(self,phis):
        n_steps = len(phis[0])
        length  = torch.zeros(len(phis)).to(phis.device)
        stepwise_dist = [] 
        for step in range(1,n_steps):
            a,b = phis[:,step,:],phis[:,step-1,:]
            step_dist = self.distance_metric(a,b)
            length += step_dist
            stepwise_dist.append(step_dist)
        stepwise_dist = torch.stack(stepwise_dist).mean(dim=-1)
        return length.mean().item(), stepwise_dist.detach().cpu().numpy()

    def forward(self,phis):
        assert len(phis.shape)==3
        return self.compute_length(phis)[0]


class PairwiseDistanceBetweenMinima(torch.nn.Module):
    def __init__(self, norm_degree=2):
        super().__init__()
        self.norm_degree = norm_degree

    def forward(self,phis):
        b = len(phis)
        phis = phis[:,-1,:].view(1,b,-1)
        print(phis.shape)
        return torch.cdist(phis, phis, p=self.norm_degree).mean().item()


class PathLengthPerStep(PathLength):
    def forward(self,phis):
        assert len(phis.shape)==3 and len(phis[0]) > 1
        n_steps = len(phis[0])
        total_length, _ = self.compute_length(phis)
        return total_length//(n_steps-1)

class StepwisePathLength(PathLength):
    def forward(self,phis):
        assert len(phis.shape)==3 and len(phis[0]) > 1
        _, stepwise_dist = self.compute_length(phis)
        return stepwise_dist


class Accuracy(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self,predictions,targets):
        predictions = predictions.argmax(dim=1).view(targets.shape)
        return 100*(predictions == targets).sum().float() / targets.size(0)


class Entropy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self,logits=None,preds=None):
        predictions = self.softmax(logits) if logits is not None else preds
        return np.mean(entropy(predictions.detach().cpu().numpy(),axis=-1))


class SoftNearestNeighbourLoss(torch.nn.Module):
    def __init__(self,distance='euclidean',temperature=1.):
        super().__init__()
        self.temperature = temperature
        self.loss_fn = LSN(distance=distance,temperature=temperature)

    def forward(self,embeddings,labels):
        embeddings, labels = tf.convert_to_tensor(embeddings.detach().cpu().numpy()),tf.convert_to_tensor(labels.detach().cpu().numpy())
        lsn = self.loss_fn(labels,embeddings)
        return lsn

        