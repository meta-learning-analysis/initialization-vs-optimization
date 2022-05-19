import argparse, sys
import json
import numpy as np
import torch
import os
import random
import tensorflow as tf
import shutil
from algorithms import *
from config import Config
from models import * 
from algorithms import algos
import os
from datasets import *

has_incorrectly_labelled_lr = []

def parse_arguments():
    config = Config()
    parser      = argparse.ArgumentParser()
    parser.add_argument('--arg', type=json.loads)
    exp_config  = parser.parse_args().arg
    print("\n \n ", "--"*50 ,"\n")
    for param in exp_config:
        if(hasattr(config,param)):
            print(" Overriding Parameter : {} \t Initial Value : {} \t New Value : {}".format(param,config.__dict__[param],exp_config[param]))
            if(param=="algo" and exp_config[param] not in algos):
                print(" Invalid Algorithm : {} \n Should be one of {} ".format(exp_config[param],algos))
                exit(0)
            setattr(config,param,exp_config[param])
        else:
            print(" Unknown Parameter : {} ".format(param))

    os.environ["CUDA_VISIBLE_DEVICES"]="{}".format(config.gpu)
    config.dev  = torch.device("cuda:{}".format(0)) 
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    EXP_DIR  = "./experiments/{}/{}WAY_{}SHOT/{}".format(config.dataset,config.n_class,config.n_shot,config.algo)
    DATA_DIR = "./data/{}".format(config.dataset)
    LOG_DIR  = os.path.join(EXP_DIR,'logs')
    CKPT_DIR = os.path.join(EXP_DIR,'ckpt','ckpt')

    os.makedirs(EXP_DIR,exist_ok=True)
    os.makedirs(DATA_DIR,exist_ok=True)
    os.makedirs(LOG_DIR,exist_ok=True)
    os.makedirs(CKPT_DIR,exist_ok=True)

    config.DATA_DIR  = DATA_DIR
    config.LOG_DIR   = LOG_DIR
    config.CKPT_DIR  = CKPT_DIR
    config.EXP_DIR  = EXP_DIR

    config_file  = os.path.join(LOG_DIR, "config.txt")
    hyperparams  = [('adaptation_steps',int), ('base_lr',float), ('meta_lr',float), ('meta_batch_size',int)]
    if(os.path.exists(config_file)):
        with open(config_file,'r') as file:
            for line in file.readlines():
                for param,dtype in hyperparams:
                    if(param in line and param not in exp_config):
                        print("setting ",param, " to ",dtype(line.split(":")[1].strip(",\n")))
                        setattr(config, param, dtype(line.split(":")[1].strip(",\n")))
    (config.n_filters, config.n_channels, config.image_size) = (64,1,28) if config.dataset == "omniglot" else (32,3,84)
    if(config.algo in has_incorrectly_labelled_lr):
        temp = config.meta_lr
        config.meta_lr = config.base_lr
        config.base_lr = temp
    
    return config