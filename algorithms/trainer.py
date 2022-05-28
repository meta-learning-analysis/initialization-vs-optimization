import torch.nn as nn 
import learn2learn as l2l
import pprint
import torch 
from datasets import get_tasksets
from .utils import *
from models import *
import numpy as np 
from torch import autograd
import os
import sys
from scipy.stats import entropy
from sklearn.metrics.pairwise import cosine_similarity
import tqdm 

class Trainer():
    """An abstract Base Trainer skeleton for meta-learning. Supports task creation, sampling, meta-training,
     meta-validation, meta-testing, checkpointing and performance logging. Concrete classes (meta-learning algorithms)
    that extends this class are required to implement the ``build_learner`` and ``adapt`` functions.

    """
    def __init__(self, args):
        """_summary_

        Args:
            :parameter args (dict): The config file of class Config containing the experiment details

        """
        super(Trainer, self).__init__()
        # Set experiment hyperparameters
        self.args    =   args
        self.ways    =   args.n_class
        self.shots   =   args.n_shot
        self.base_lr =   args.base_lr
        self.meta_lr =   args.meta_lr
        self.device  =   args.dev
        self.dataset =   args.dataset
        self.meta_batch_size =   args.meta_batch_size
        self.test_size       =   args.meta_batch_size if args.test_size is None  else args.test_size
        self.adaptation_steps=   args.adaptation_steps
        self.num_iterations  =   args.num_iterations
        self.meta_lr         =   args.meta_lr
        self.base_lr         =   args.base_lr
        self.data_dir        =   args.DATA_DIR
        self.log_dir         =   args.LOG_DIR
        self.ckpt_dir        =   args.CKPT_DIR
        self.dev             =   args.dev
        self.last_eps        =   0
        self.iteration       =   0
        self.val_freq        =   args.val_freq
        # Literature suggests the use of 15 support samples for complex datasets like mini-imagenet and tiered-imagenet
        self.n_samples       =   2* self.shots if "omniglot" in self.dataset else 15+self.shots
        # Define the N-Way K-Shot classification tasks
        self.tasksets        =   get_tasksets(
                                            self.dataset,
                                            train_ways=self.ways,
                                            train_samples=self.n_samples,
                                            test_ways=self.ways,
                                            test_samples=self.n_samples,
                                            root=self.data_dir
                                            )
        self.base_learner = args.base_learner
        # Currently only intended for fes-shot classification tasks
        self.loss         = nn.CrossEntropyLoss()
        self.meta_learner = self.build_metalearner()
        self.meta_opt     = self.args.meta_opt(self.meta_learner.parameters(),self.meta_lr)
        self.history      = {}
         
        if args.resume:
            self.load()
        
    def log(self,mode,iteration,loss,acc):
        """A function to log the training progress

        Args:
            mode (str): The data-split : train | test | val
            iteration (int): The training iteration
            loss (float): The current loss value
            acc (float): The current accuacy value 
        """
        if("train" not in mode):
            print("=="*50)
        print("|MODE : {:^10s}|ITER : {:^10d}|LOSS : {:^10f}|ACC : {:^10f} |".format(mode,iteration,np.mean(loss),np.mean(acc))) 
        print("--"*50)
    
    def save(self):
        """Saves the training history and model checkpoints
        """
        save_ckpt(self.iteration,self.meta_learner,self.meta_opt,self.ckpt_dir)
        save_history(self.history,self.args)

    def load(self,ckpt_no=None):
        """Loads the training history and latest model checkpoint from the experiment directory, if exists.

        Args:
            ckpt_no (int, optional): The checkpoint no. to load. Defaults to None (will load the latest checkpoint).
        """
        self.history = load_history(self.log_dir)      
        # Load the best performing checkpoint if training history is available, else the latest checkpoint
        if("iter" in self.history and ckpt_no is None):
            idx = np.argmax(self.history["accuracy"]["val"])
            ckpt_no = self.history["iter"]["val"][idx]  
        self.last_eps, self.meta_learner, self.meta_opt = resume_ckpt(metalearner=self.meta_learner, optim=self.meta_opt, resume=self.ckpt_dir,device=self.dev,ckpt_no=ckpt_no)
        self.iteration = self.last_eps

    def meta_train(self):
        """The meta-trainig function
        """
        for m in ["loss","accuracy","iter"]:
            self.history[m]={"train":[],"val":[],"test":[]} if m not in self.history else self.history[m]
        
        val_acc_mu, val_acc_std = self.meta_validate()
        print("-->validation acc:",val_acc_mu,flush=True)
        self.save()
        for self.iteration in tqdm.tqdm(range(self.last_eps+1,self.last_eps+self.num_iterations+1)):
            self.meta_opt.zero_grad()
            loss,acc = 0,0
            for task in range(self.meta_batch_size):
                batch   = self.tasksets.train.sample()
                evaluation_error, evaluation_accuracy, _ = self.adapt(batch)
                loss, acc = loss+evaluation_error.item(), acc+evaluation_accuracy.item()
                evaluation_error.backward()
            loss,acc = loss/self.meta_batch_size , acc/self.meta_batch_size
            
            self.history["loss"]["train"].append(loss)
            self.history["accuracy"]["train"].append(acc)
            self.history["iter"]["train"].append(self.iteration)

            for p in self.meta_learner.parameters():
                p.grad.data.mul_(1.0 / self.meta_batch_size)

            self.meta_opt.step()    
            if(self.iteration%self.val_freq==0):
                val_acc_mu, val_acc_std = self.meta_validate()
                print("-->validation acc:",val_acc_mu,flush=True)
                self.save()
            
        self.save()

    def meta_test(self, load_ckpt=None, return_trajectory=False):
        """The meta-testing function

        Args:
            load_ckpt (int, optional): The checkpoint to load for meta-testing. Defaults to None.
            return_trajectory (bool, optional): If true, returns the trajectory taken by the model for each task along with the final accuracy. Defaults to False.
        
        Returns:
            tuple: The mean and standard deviation of the performance across unseen meta-test tasks
        """
        if(load_ckpt is not None):
            self.load(load_ckpt)
        
        print("\n Meta-testing \n")
        print("Saved Test Accuracy",np.mean(np.array(self.history["accuracy"]["test"])))
        batch_accuracy = []
        batch_loss     = []
        trajectory = {}
        for tid,task in enumerate(tqdm.tqdm(range(self.test_size))):
            batch   = self.tasksets.test.sample()
            data,labels = batch
            evaluation_error, evaluation_accuracy, phi = self.adapt(batch)
            batch_accuracy.append(evaluation_accuracy.item())
            batch_loss.append(evaluation_error.item())
            if(return_trajectory):
                trajectory[tid]={
                    "data":data.detach().cpu().numpy(),
                    "labels":labels.detach().cpu().numpy(),
                    "model":phi,
                    "batch":batch
                }
        batch_loss, batch_accuracy = np.array(batch_loss), np.array(batch_accuracy)
        test_loss, test_accuracy   = np.mean(batch_loss),np.mean(batch_accuracy)       
        mean_test_acc = np.mean(batch_accuracy)
        conf_test_acc = 1.96*np.std(batch_accuracy)/np.sqrt(self.test_size)
        std_test_acc  = np.std(batch_accuracy)
        print(" Test acc : {} +- {}".format(mean_test_acc,conf_test_acc))
        if return_trajectory:
            return trajectory
        return mean_test_acc,std_test_acc

    def meta_validate(self, load_ckpt=None):
        """The meta-validation function

        Returns:
            tuple: The mean and standard deviation of the performance across unseen meta-validation tasks
        """
        print("\n Meta-Validating \n")
        batch_accuracy = []
        batch_loss     = []
        for tid,task in enumerate(range(self.test_size)):
            batch   = self.tasksets.validation.sample()
            evaluation_error, evaluation_accuracy, phi = self.adapt(batch)
            batch_accuracy.append(evaluation_accuracy.item())
            batch_loss.append(evaluation_error.item())
            
        batch_loss, batch_accuracy = np.array(batch_loss), np.array(batch_accuracy)
        self.history["loss"]["val"].append(np.mean(batch_loss))
        self.history["accuracy"]["val"].append(np.mean(batch_accuracy))
        self.history["iter"]["val"].append(self.iteration)
        mean_test_acc = np.mean(batch_accuracy)
        std_test_acc  = np.std(batch_accuracy)
        return mean_test_acc,std_test_acc
    
    def get_test_accuracy(self):
        """Get the saved test performance from the hsitory file 

        Returns:
            tuple: meta-test accuracy mean and standard deviation across test tasks 
        """
        test_acc_mu, test_acc_std = np.mean(np.array(self.history["accuracy"]["test"])), np.std(np.array(self.history["accuracy"]["test"]))
        return test_acc_mu,test_acc_std

        
    def build_metalearner(self):
        """The concrete meta-learner algorithm

        Raises:
            NotImplementedError: This is an abstract class
        """
        raise NotImplementedError
