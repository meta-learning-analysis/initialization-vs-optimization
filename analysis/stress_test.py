import os, sys, pickle
from metrics import * 

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

torch.autograd.set_detect_anomaly(True)
if __name__ == '__main__':
    
    models  = ["MAML","MetaSGD","TA_LSTM"]
    labels  ={
        "MAML":"MAML",
        "MetaSGD":"MetaSGD",
        "TA_LSTM":"MetaLSTM++"
    }
    dataset = "tiered-imagenet"
    redo = []
    colors = list(cm.Set1(np.linspace(0, 1, len(models))))
    n_ways  = [20, 40, 60, 80, 100, 150, 200] #omniglot
    # n_ways  = [5,20,40,90] #tiered-imagenet
    n_shots = [1]
    plot_data = {}
    
    os.makedirs(f"results/{dataset}/files",exist_ok=True)
    plot_file = f"results/{dataset}/files/stress-test"

    if(os.path.exists(plot_file)):
        with open(plot_file,'rb') as file:
            plot_data = pickle.load(file)
            
    for n_shot in n_shots:
        for n_class in n_ways:
            for model_name in models:
                print("--"*50)
                print(f" \t {n_class} WAY {n_shot} SHOT - {model_name} ")
                
                config = Config()
                config.n_class = n_class
                config.n_shot = n_shot
                config.algo = model_name
                config.dataset = dataset
                config = process_config(config)
                ckpt_no = None
                config.base_learner = Learner(config.image_size, config.bn_eps, config.bn_momentum, config.n_class,n_channels=config.n_channels,n_filters=config.n_filters).to(config.dev)
                config.meta_opt     = torch.optim.Adam
                if(f"{n_class}.{n_shot}.{model_name}" not in plot_data or f"{n_class}.{n_shot}.{model_name}" in redo or f"{model_name}" in redo):
                    
                    model               = algos[config.algo](config) 
                    # No need for gradient-accumulation during meta-testing. Saves memory
                    if("TA_LSTM" in model_name):
                        for param in model.meta_learner.parameters():
                            param.requires_grad = False
                    
                    acc_mu,acc_std     = model.meta_test()
                    plot_data[f"{n_class}.{n_shot}.{model_name}"] = {"mean":acc_mu,"std":acc_std}
                
                print(plot_data[f"{n_class}.{n_shot}.{model_name}"])

                with open(plot_file,'wb') as file:
                    pickle.dump(plot_data,file)

    for n_shot in n_shots:
        sns.set_style("whitegrid")
        plt.rcParams["axes.edgecolor"] = "0.15"
        plt.rcParams["axes.linewidth"]  = 1.50
        plt.figure(figsize=(5,5))
        for model,color in zip(models,colors):
            x_axis = n_ways
            y_axis = np.array([plot_data[f"{x}.{n_shot}.{model}"]["mean"] for x in x_axis])
            y_std  = 2.576*np.array([plot_data[f"{x}.{n_shot}.{model}"]["std"] for x in x_axis])/np.sqrt(config.test_size)
            plt.plot(x_axis,y_axis,label=labels[model],lw=2,linestyle="--",alpha=0.75)
            plt.fill_between(x_axis,y_axis+y_std,y_axis-y_std,alpha=0.4)
        plt.xlabel("No. of. Ways",fontsize=14,fontweight="bold")
        plt.ylabel("Accuracy",fontsize=14,fontweight="bold")
        plt.legend(fontsize=12)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        os.makedirs(f"results/{dataset}/stress_test",exist_ok=True)
        plt.savefig(f"results/{dataset}/stress_test/{n_shot}shot_stress_test.png",bbox_inches="tight")
        plt.close()
            
