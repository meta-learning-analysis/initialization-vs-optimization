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

        # lsn = torch.Tensor([0]).to(embeddings.device)
        # b = labels.shape[0]
        # for i in range(b):
        #     numerator, denominator = 0, 0
        #     for j in range(b):
        #         if i == j:
        #             continue
        #         # q = torch.exp(-1 * (features[i] - features[j]).pow(2).sum(0).sqrt() / T)
        #         cos = torch.nn.CosineSimilarity(dim=0)
        #         q = torch.exp(-1 * cos(embeddings[i], embeddings[j]) / self.temperature)
        #         denominator += q
        #         if(labels[i] == labels[j]):
        #             numerator += q
        #     # print(numerator.shape)
        #     numerator = torch.squeeze(numerator, dim=0)
        #     denominator = torch.squeeze(denominator, dim=0)
        #     lsn += torch.log(numerator / denominator)
        # lsn = -lsn / b
        # print(lsn)
        # return lsn.item()

# class Distance(ABC):
#     """
#     Note: don't forget to add your distance to the DISTANCES list
#     and add alias names in it.
#     """
#     def __init__(self, name: str, aliases: List[str] = []):
#         self.name = name
#         self.aliases = aliases

#     @abstractmethod
#     def call(self, embeddings):
#         """Compute pairwise distances for a given batch.
#         Args:
#             embeddings: Embeddings to compute the pairwise one.
#         Returns:
#             FloatTensor: Pairwise distance tensor.
#         """

#     def __call__(self, embeddings):
#         return self.call(embeddings)

#     def __str__(self) -> str:
#         return self.name

#     def get_config(self):
#         return {}


# class CosineDistance(Distance):
#     """Compute pairwise cosine distances between embeddings.
#     The [Cosine Distance](https://en.wikipedia.org/wiki/Cosine_similarity) is
#     an angular distance that varies from 0 (similar) to 1 (dissimilar).
#     """
#     def __init__(self):
#         "Init Cosine distance"
#         super().__init__('cosine')

#     def call(self, embeddings):
#         """Compute pairwise distances for a given batch of embeddings.
#         Args:
#             embeddings: Embeddings to compute the pairwise one. The embeddings
#             are expected to be normalized.
#         Returns:
#             FloatTensor: Pairwise distance tensor.
#         """
#         distances = 1 - torch.matmul(embeddings, embeddings.t())
#         min_clip_distances = torch.clamp(distances, min=0.0)
#         return min_clip_distances


# class EuclideanDistance(Distance):
#     """Compute pairwise euclidean distances between embeddings.
#     The [Euclidean Distance](https://en.wikipedia.org/wiki/Euclidean_distance)
#     is the standard distance to measure the line segment between two embeddings
#     in the Cartesian point. The larger the distance the more dissimilar
#     the embeddings are.
#     **Alias**: L2 Norm, Pythagorean
#     """
#     def __init__(self):
#         "Init Euclidean distance"
#         super().__init__('euclidean', ['l2', 'pythagorean'])

#     def call(self, embeddings):
#         """Compute pairwise distances for a given batch of embeddings.
#         Args:
#             embeddings: Embeddings to compute the pairwise one.
#         Returns:
#             FloatTensor: Pairwise distance tensor.
#         """
#         squared_norm = torch.square(embeddings)
#         squared_norm = torch.sum(squared_norm, dim=1, keepdim=True)

#         distances = 2.0 * torch.matmul(embeddings, embeddings.t())
#         distances = squared_norm - distances + squared_norm.t()

#         # Avoid NaN and inf gradients when back propagating through the sqrt.
#         # values smaller than 1e-18 produce inf for the gradient, and 0.0
#         # produces NaN. All values smaller than 1e-13 should produce a gradient
#         # of 1.0.
#         dist_mask = torch.ge(distances, 1e-18)
#         distances = torch.clamp(distances, min=1e-18)
#         distances = torch.sqrt(distances) * dist_mask.to(torch.float32)

#         return distances

# class SquaredEuclideanDistance(Distance):
#     """Compute pairwise squared Euclidean distance.
#     The [Squared Euclidean Distance](https://en.wikipedia.org/wiki/Euclidean_distance#Squared_Euclidean_distance) is
#     a distance that varies from 0 (similar) to infinity (dissimilar).
#     """
#     def __init__(self):
#         super().__init__('squared_euclidean', ['sql2', 'sqeuclidean'])

#     def call(self, embeddings):
#         """Compute pairwise distances for a given batch of embeddings.
#         Args:
#             embeddings: Embeddings to compute the pairwise one.
#         Returns:
#             FloatTensor: Pairwise distance tensor.
#         """
#         squared_norm = torch.square(embeddings)
#         squared_norm = torch.sum(squared_norm, dim=1, keepdim=True)

#         distances = 2.0 * torch.matmul(embeddings, embeddings.t())
#         distances = squared_norm - distances + squared_norm.t()
#         distances = torch.clamp(distances, min=0.0)

#         return distances

# class SoftNearestNeighbourLoss(torch.nn.Module):
#     def __init__(self,distance_fn=SquaredEuclideanDistance(),temperature=1.0):
#         super().__init__()
#         self.softmax = torch.nn.Softmax(dim=-1)
#         self.distance = distance_fn
#         self.temperature = temperature


#     def build_masks(self,labels,batch_size):
#         """Build masks that allows to select only the positive or negatives
#         embeddings.
#         Args:
#             labels: 1D int `Tensor` that contains the class ids.
#             batch_size: size of the batch.
#         Returns:
#             Tuple of Tensors containing the positive_mask and negative_mask
#         """
#         if labels.ndim == 1:
#             labels = torch.reshape(labels, (-1, 1))

#         # same class mask
#         positive_mask = torch.eq(labels, labels.t())
#         # not the same class
#         negative_mask = torch.logical_not(positive_mask)

#         # we need to remove the diagonal from positive mask
#         diag = torch.logical_not(torch.diag_embed(torch.ones(batch_size, dtype=torch.bool))).to(labels.device)
#         positive_mask = torch.logical_and(positive_mask, diag)

#         return positive_mask, negative_mask

#     def forward(self,embeddings,labels):
#         """Computes the soft nearest neighbors loss.
#         Args:
#             labels: Labels associated with embeddings.
#             embeddings: Embedded examples.
#             temperature: Controls relative importance given
#                             to the pair of points.
#         Returns:
#             loss: loss value for the current batch.
#         """

#         batch_size = len(labels)
#         eps = 1e-30

#         pairwise_dist = self.distance(embeddings)
#         pairwise_dist = pairwise_dist / self.temperature
#         negexpd = torch.exp(-pairwise_dist)

#         # Mask out diagonal entries
#         diag = torch.diag_embed(torch.ones(batch_size, dtype=torch.bool)).to(embeddings.device)
#         diag_mask = torch.logical_not(diag).to(torch.float32)
#         negexpd = torch.mul(negexpd, diag_mask)

#         # creating mask to sample same class neighboorhood
#         pos_mask, _ = self.build_masks(labels, batch_size)
#         pos_mask = pos_mask.to(torch.float32)

#         # all class neighborhood
#         alcn = torch.sum(negexpd, dim=1)

#         # same class neighborhood
#         sacn = torch.sum(torch.mul(negexpd, pos_mask), dim=1)

#         # exclude examples with unique class from loss calculation
#         excl = torch.ne(torch.sum(pos_mask, dim=1),torch.zeros(batch_size).to(embeddings.device))
#         excl = excl.to(torch.float32)

        
#         loss = torch.div(sacn, alcn+eps)
#         loss = torch.mul(torch.log(eps+loss), excl)
#         loss =-torch.mean(loss)

#         print(self.softnn_util(labels,embeddings,self.temperature),loss.item())
#         return loss

#     # def forward(self,logits):
#     #     predictions = self.softmax(logits)
#     #     return np.mean(entropy(predictions.detach().cpu().numpy(),axis=-1))
        


#     def softnn_util(self,evaluation_labels, features, T = 1):
#         """
#         A simple loop based implementation of soft
#         nearest neighbor loss to test the code.
#         https://arxiv.org/pdf/1902.01889.pdf
#         """

#         lsn = torch.Tensor([0]).to(features.device)
#         b = evaluation_labels.shape[0]
#         for i in range(b):
#             numerator, denominator = 0, 0
#             for j in range(b):
#                 if i == j:
#                     continue
#                 # q = torch.exp(-1 * (features[i] - features[j]).pow(2).sum(0).sqrt() / T)
#                 cos = torch.nn.CosineSimilarity(dim=0)
#                 q = torch.exp(-1 * cos(features[i], features[j]) / T)
#                 denominator += q
#                 if(evaluation_labels[i] == evaluation_labels[j]):
#                     numerator += q
#             # print(numerator.shape)
#             numerator = torch.squeeze(numerator, dim=0)
#             denominator = torch.squeeze(denominator, dim=0)
#             lsn += torch.log(numerator / denominator)
#         lsn = -lsn / b
#         return lsn


