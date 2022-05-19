import os 
import torch 
import random 
import numpy as np 
import pickle 

has_incorrectly_labelled_lr = []#["MAML"]

def load_or_run_exp(model,n_class,n_shot,model_name,load_saved_trajectory):
    os.makedirs("results/files/trajectories",exist_ok=True)

    tpath = f"results/files/trajectories/{n_class}.{n_shot}.{model_name}"

    if(load_saved_trajectory and os.path.exists(tpath)):
        with open(tpath,'rb') as file:
            trajectory = pickle.load(file)
    else:
        trajectory = model.meta_test(return_trajectory=True)   
        with open(tpath,'wb') as file:
            pickle.dump(trajectory,file)

    return trajectory
            
def process_config(config):
    config.dev  = torch.device("cuda") 
    # os.environ["CUDA_VISIBLE_DEVICES"]="{}".format(config.gpu)

    # EXP_DIR  = "./experiments/{}WAY_{}SHOT/{}".format(config.n_class,config.n_shot,config.algo)
    
    if("omniglot" in config.dataset):
        EXP_DIR  = "/home/sahil/drives/shivay/scratch/Hansin/exp/redo/{}WAY_{}SHOT/{}".format(config.n_class,config.n_shot,config.algo)
    else:
        # EXP_DIR  = "./experiments/{}/{}WAY_{}SHOT/{}".format(config.dataset,config.n_class,config.n_shot,config.algo)
        # EXP_DIR  = "/home/sahil/drives/shivay/scratch/Hansin/exp/redo/{}/{}WAY_{}SHOT/{}".format(config.dataset,config.n_class,config.n_shot,config.algo)
        EXP_DIR  = "/home/sahil/drives/shivay/scratch/lsaiml-metalearning/experiments/{}/{}WAY_{}SHOT/{}".format(config.dataset,config.n_class,config.n_shot,config.algo)
        
    dir_path = os.path.dirname(os.path.realpath(__file__))
    print(dir_path)
    DATA_DIR = "{}/../data/{}".format(dir_path,config.dataset)
    LOG_DIR  = os.path.join(EXP_DIR,'logs')
    CKPT_DIR = os.path.join(EXP_DIR,'ckpt','ckpt')
    os.makedirs(DATA_DIR,exist_ok=True)
    
    if(os.path.exists(os.path.join(EXP_DIR,'ckpt','ckpt'))):
        CKPT_DIR = os.path.join(EXP_DIR,'ckpt','ckpt')
        
    elif(os.path.exists(os.path.join(EXP_DIR,'ckpts'))):
        CKPT_DIR = os.path.join(EXP_DIR,'ckpts')

    config.DATA_DIR, config.LOG_DIR, config.CKPT_DIR, config.EXP_DIR  = DATA_DIR, LOG_DIR, CKPT_DIR, EXP_DIR

    config_file  = os.path.join(LOG_DIR, "config.txt")
    hyperparams  = [('seed',int),('adaptation_steps',int), ('base_lr',float), ('meta_lr',float),('val_freq',int),('num_iterations',int)]
    if(os.path.exists(config_file)):
        with open(config_file,'r') as file:
            for line in file.readlines():
                for param,dtype in hyperparams:
                    if(param in line):
                        setattr(config, param, dtype(line.split(":")[1].strip(",\n")))
                        print(param,getattr(config,param))

    if(config.algo in has_incorrectly_labelled_lr):
        temp = config.meta_lr
        config.meta_lr = config.base_lr
        config.base_lr = temp
    
    (config.n_filters, config.n_channels, config.image_size) = (64,1,28) if config.dataset == "omniglot" else (32,3,84)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    
    return config
