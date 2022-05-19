import os, sys
from helper import process_config, load_or_run_exp
from metrics import * 

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from models import * 
import torch
from algorithms import algos
import learn2learn as l2l
from config import Config
import os 
import random 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import pandas as pd 
import pickle 
from matplotlib.pyplot import cm

def run_analysis(batch,base_learner,phis): 
    print(phis.shape)
    path_length = PathLength()(phis)
    path_length_per_step = PathLengthPerStep()(phis)
    
    return {
        "path-length":path_length,
        "path_length_per_step":path_length_per_step,
    }

if __name__ == '__main__':
    dataset = "tiered-imagenet"
    models  = ["MAML","MetaSGD","TA_LSTM"]
    fig_labels  ={
        "MAML":"MAML",
        "MetaSGD":"MetaSGD",
        "TA_LSTM":"MetaLSTM++"
    }
    metrics_to_plot = ["path-length", "path_length_per_step"]
    n_ways  = [5,20,40,90]
    n_shots = [1]
    plot_data = {}
    plot_file = f"results/{dataset}/files/pathlen"
    redo = ["all"]

    if(os.path.exists(plot_file)):
        with open(plot_file,'rb') as file:
            plot_data = pickle.load(file)

    for n_shot in n_shots:
        for n_class in n_ways:
            for model_name in models:
                print("--"*50)
                print(f" \t {n_class} WAY {n_shot} SHOT - {model_name} ")
                
                config = Config()
                config.test_size = 100
                config.n_class = n_class
                config.n_shot = n_shot
                config.dataset = dataset
                config.algo = model_name
                config = process_config(config)
                
                config.base_learner = Learner(config.image_size, config.bn_eps, config.bn_momentum, config.n_class,n_channels=config.n_channels,n_filters=config.n_filters).to(config.dev)
                config.meta_opt     = torch.optim.Adam
                
                if(f"{n_class}.{n_shot}.{model_name}" not in plot_data or f"{n_class}.{n_shot}.{model_name}" in redo or "all" in redo):
                    model               = algos[config.algo](config) 
                    if("TA_LSTM" in model_name):
                        for param in model.meta_learner.parameters():
                            param.requires_grad = False
                    
                    trajectory = model.meta_test(return_trajectory=True)
                    batch = np.array([trajectory[tid]["data"] for tid in trajectory])
                    phis  = np.array([trajectory[tid]["model"] for tid in trajectory])
                    phis  = torch.from_numpy(phis).to(config.dev).requires_grad_(False)
                    metric_data = run_analysis(batch,config.base_learner,phis)
                    plot_data[f"{n_class}.{n_shot}.{model_name}"] = metric_data
    
                with open(plot_file,'wb') as file:
                    pickle.dump(plot_data,file)

    sns.set_style("whitegrid")
    plt.rcParams["axes.edgecolor"] = "0.15"
    plt.rcParams["axes.linewidth"]  = 1.50
    width = 0.35

    for metric in metrics_to_plot:
        for n_shot in n_shots:
            plt.figure()
            df_data = [[f"{a}"] for a in n_ways]
            df_labels  = ['Ways']
    
            for i,model in enumerate(models):
                x_label = [f"{a}" for a in n_ways]
                
                # x_axis = np.arange(len(x_label))
                # y_axis = np.array([plot_data[f"{x}.{n_shot}.{model}"][f"{metric}"] for x in x_axis])
                for i,x in enumerate(n_ways):
                    df_data[i].append(plot_data[f"{x}.{n_shot}.{model}"][f"{metric}"])
                df_labels.append(fig_labels[model]) 
        
            df = pd.DataFrame(df_data,columns=df_labels)
            # plt.bar(x_axis-width/len(models),y_axis,label=f"{model}")
            
            df.plot(x='Ways',
            kind='bar',
            stacked=False,
            figsize=(6,5),
            width=0.6
            )

            plt.xlabel("No. of. Ways",fontsize=14,fontweight="bold")
            plt.ylabel(f"{metric}",fontsize=14,fontweight="bold")
            plt.yscale('log')   
            plt.legend(fontsize=12,ncol=len(models),loc=(0,1.04))
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            os.makedirs(f"results/{dataset}/pathlen",exist_ok=True)
            plt.savefig(f"results/{dataset}/pathlen/{n_shot}shot_{metric}.png",bbox_inches="tight")
            plt.close()