from ast import Num
from asyncio import run
import os, sys
from metrics import * 
from ray.tune.suggest.bohb import TuneBOHB

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from models import * 
from algorithms.utils import test_split
import torch
from algorithms import algos
import learn2learn as l2l
from config import Config
import os 
import random 
import numpy as np 
import matplotlib.pyplot as plt 
from helper import process_config, load_or_run_exp
import seaborn as sns 
from matplotlib.pyplot import cm
import pickle

import ray
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler

def run_meta_test(args,init):
    config = Config()
    config.n_class = args["n_class"]
    config.n_shot = args["n_shot"]
    config.algo = args["algo"]
    config.dataset = args["dataset"]
    config = process_config(config)
    config.adaptation_steps = args["adaptation_steps"]
    config.base_lr = args["base_lr"] if "base_lr" in args else config.base_lr
    config.base_learner = Learner(config.image_size, config.bn_eps, config.bn_momentum, config.n_class,n_channels=config.n_channels,n_filters=config.n_filters).to(config.dev)
    config.meta_opt     = torch.optim.Adam
    model            = algos[config.algo](config) 
    model.load()
    model.set_init(torch.from_numpy(init).to(config.dev))
    if("TA_LSTM" in args["algo"]):
        for param in model.meta_learner.parameters():
            param.requires_grad = False
    return model.meta_test(load_ckpt=None)
    
def metalstm_analysis(args):
    
    config = Config()
    config.n_class = args["n_class"]
    config.n_shot = args["n_shot"]
    config.algo = "TA_LSTM"
    config.dataset = args["dataset"]
    config = process_config(config)
    config.adaptation_steps = args["adaptation_steps"]
    config.test_size = 50
    
    config.base_learner = Learner(config.image_size, config.bn_eps, config.bn_momentum, config.n_class,n_channels=config.n_channels,n_filters=config.n_filters).to(config.dev)
    config.meta_opt     = torch.optim.Adam
    metalstm            = algos[config.algo](config) 
    metalstm.load()
    for param in metalstm.meta_learner.parameters():
        param.requires_grad = False

    metalstm.set_init(torch.from_numpy(args["maml_init"]).to(config.dev))
    acc_mu,acc_std = metalstm.meta_validate(load_ckpt=None)
    tune.report(mean_accuracy=acc_mu)
    
def maml_analysis(args):
    
    config = Config()
    config.n_class = args["n_class"]
    config.n_shot = args["n_shot"]
    config.algo = "MAML"
    config.dataset = args["dataset"]
    config = process_config(config)
    config.adaptation_steps = args["adaptation_steps"]
    config.base_lr = args["base_lr"]
    config.test_size = 50
    
    config.base_learner = Learner(config.image_size, config.bn_eps, config.bn_momentum, config.n_class,n_channels=config.n_channels,n_filters=config.n_filters).to(config.dev)
    config.meta_opt    = torch.optim.Adam
    maml               = algos[config.algo](config) 
    
    maml.set_init(torch.from_numpy(args["metalstm_init"]).to(config.dev))
    acc_mu,acc_std = maml.meta_validate(load_ckpt=None)
    tune.report(mean_accuracy=acc_mu)
    
    # if(f"{n_class}.{n_shot}.MetaLSTM++ x MAML" not in plot_data or f"{n_class}.{n_shot}.MetaLSTM++ x MAML" in redo):                
        
    #     maml.set_init(metalstm_init)
    #     acc_mu,acc_std = maml.meta_test(load_ckpt=False)
    #     plot_data[f"{n_class}.{n_shot}.MetaLSTM++ x MAML"] = {"mean":acc_mu,"std":acc_std}
        
    #     metalstm.set_init(maml_init)
    #     acc_mu,acc_std = metalstm.meta_test(load_ckpt=False)
    #     plot_data[f"{n_class}.{n_shot}.MAML x MetaLSTM++"] = {"mean":acc_mu,"std":acc_std}

    # with open(plot_file,'wb') as file:
    #     pickle.dump(plot_data,file)

def run_exp(n_class,n_shot,dataset,maml_init,metalstm_init):
    
    ray.init(num_gpus=2)
    
    # for early stopping
    sched  = AsyncHyperBandScheduler()
    bayesopt = TuneBOHB(metric="mean_accuracy", mode="max")
    config = {
            "base_lr": tune.loguniform(1e-4, 1e1),
            "adaptation_steps":tune.qrandint(1,64),
            "n_class":n_class,
            "n_shot":n_shot,
            "dataset":dataset,
            "maml_init":maml_init,
            "metalstm_init":metalstm_init
    }
    maml_results = tune.run(
        maml_analysis,
        metric="mean_accuracy",
        mode="max",
        name="maml-exp",
        scheduler=sched,
        search_alg=bayesopt,
        resources_per_trial={"cpu": 8, "gpu": 1},  # set this for GPUs
        num_samples = 50,
        config = config,
        raise_on_failed_trial=False
    )
    ray.shutdown()
    
    ray.init(num_gpus=2)
    sched  = AsyncHyperBandScheduler()
    bayesopt = TuneBOHB(metric="mean_accuracy", mode="max")
    config = {
            "adaptation_steps":tune.qrandint(1,256),
            "n_class":n_class,
            "n_shot":n_shot,
            "dataset":dataset,
            "maml_init":maml_init,
            "metalstm_init":metalstm_init
    }

    metalstm_results = tune.run(
        metalstm_analysis,
        metric="mean_accuracy",
        mode="max",
        name="metalstm-exp",
        scheduler=sched,
        search_alg=bayesopt,
        resources_per_trial={"cpu": 8, "gpu": 1},  # set this for GPUs
        num_samples = 50,
        config = config,
        raise_on_failed_trial=False
    )

    maml_results.best_config["algo"] = "MAML"
    metalstm_results.best_config["algo"] = "TA_LSTM"

    maml_acc_mu, maml_acc_std = run_meta_test(maml_results.best_config,metalstm_init)
    metalstm_acc_mu, metalstm_acc_std = run_meta_test(metalstm_results.best_config,maml_init)

    results = {
        "MAML":{
            "acc-mu":maml_acc_mu,
            "acc-std":maml_acc_std,
            "best-config":maml_results.best_config
            },
        "MetaLSTM++":{
            "acc-mu":metalstm_acc_mu,
            "acc-std":metalstm_acc_std,
            "best-config":metalstm_results.best_config
            }
    }
    ray.shutdown()
    return results

if __name__ == '__main__':
    load_saved_trajectory = True
    models  = ["MAML","MetaLSTM++ x MAML"]
    labels  ={
       "MAML":"MAML with MetaLSTM++ Init",
       "MetaLSTM++":"MetaLSTM++ with MAML Init"
    }
    # colors  = ['darkcyan','crimson']
    colors = list(cm.rainbow(np.linspace(0, 1, len(models))))
    n_ways  = [20,40,60,80,100,150,200]
    n_shots = [5]
    dataset = "omniglot"
    plot_data = {}
    plot_file = f"results/{dataset}/files/cross-init"
    redo = []
    
    if(os.path.exists(plot_file)):
        with open(plot_file,'rb') as file:
            plot_data = pickle.load(file)
    
    for n_shot in n_shots:
        for n_class in n_ways:
            print("--"*50)
            print(f" \t {n_class} WAY {n_shot} SHOT - CROSS ")

            if(f"{n_class}.{n_shot}.MAML" not in plot_data or f"{n_class}.{n_shot}.MetaLSTM++" in redo):                
        
                config = Config()
                config.n_class = n_class
                config.n_shot = n_shot
                config.algo = "MAML"
                config.dataset = dataset
                config = process_config(config)
                config.base_learner = Learner(config.image_size, config.bn_eps, config.bn_momentum, config.n_class,n_channels=config.n_channels,n_filters=config.n_filters).to(config.dev)
                config.meta_opt    = torch.optim.Adam
                maml               = algos[config.algo](config) 
                maml.load()
                maml_init = maml.get_init().detach().cpu().numpy()

                config = Config()
                config.n_class = n_class
                config.n_shot = n_shot
                config.algo = "TA_LSTM"
                config = process_config(config)
                config.base_learner = Learner(config.image_size, config.bn_eps, config.bn_momentum, config.n_class,n_channels=config.n_channels,n_filters=config.n_filters).to(config.dev)
                config.meta_opt     = torch.optim.Adam
                metalstm            = algos[config.algo](config) 
                metalstm.load()
                metalstm_init = metalstm.get_init().detach().cpu().numpy()
                
                exp_results = run_exp(n_class,n_shot,dataset,maml_init,metalstm_init)

                plot_data[f"{n_class}.{n_shot}.MAML"] = {"mean":exp_results["MAML"]["acc-mu"],"std":exp_results["MAML"]["acc-std"],"best-config":exp_results["MAML"]["best-config"]}
                plot_data[f"{n_class}.{n_shot}.MetaLSTM++"] = {"mean":exp_results["MetaLSTM++"]["acc-mu"],"std":exp_results["MetaLSTM++"]["acc-std"],"best-config":exp_results["MetaLSTM++"]["best-config"]}
   
                with open(plot_file,'wb') as file:
                    pickle.dump(plot_data,file)
                        

    for n_shot in n_shots:
        sns.set_style("whitegrid")
        plt.rcParams["axes.edgecolor"] = "0.15"
        plt.rcParams["axes.linewidth"]  = 1.50
        plt.figure()
        for model,color in zip(models,colors):
            x_axis = n_ways
            y_axis = np.array([plot_data[f"{x}.{n_shot}.{model}"]["mean"] for x in x_axis])
            y_std  = np.array([plot_data[f"{x}.{n_shot}.{model}"]["std"] for x in x_axis])
            plt.plot(x_axis,y_axis,label=labels[model],lw=2,linestyle="--",alpha=0.75)
            plt.fill_between(x_axis,y_axis+y_std,y_axis-y_std,alpha=0.4)
        plt.xlabel("No. of. Ways",fontsize=14,fontweight="bold")
        plt.ylabel("Accuracy",fontsize=14,fontweight="bold")
        plt.legend(fontsize=12)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        os.makedirs("results/cross-init",exist_ok=True)
        plt.savefig(f"results/cross-init/{n_shot}shot_cross_stress_test.png",bbox_inches="tight")
        plt.close()
                