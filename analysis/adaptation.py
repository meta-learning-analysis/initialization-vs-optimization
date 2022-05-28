import os, sys
from helper import process_config, load_or_run_exp
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
import seaborn as sns 
from matplotlib.pyplot import cm
import pickle 

def run_analysis(task_batch,base_learner,phis,config): 
    
    accuracy = Accuracy()
    query_loss = type(base_learner.criterion)()
    entropy = Entropy()
    stepwise_path_length = StepwisePathLength()(phis)
    
    analysis_data = {}
    for step in range(phis.shape[1]):
        analysis_data[f"step-{step}"] = {}
        if(step>0):
            analysis_data[f"step-{step-1}"]["stepwise_path_length"]=stepwise_path_length[step-1]
        for m in ["support-accuracy","query-accuracy","query-loss","support-loss","support-entropy","query-entropy"]:
            analysis_data[f"step-{step}"][m] = []
                
    for tid in range(len(phis)):
        batch = task_batch[tid]
        for step in range(phis.shape[1]):
            phi = phis[tid,step,:]
            base_learner.copy_flat_params(phi)
            
            adaptation_data, adaptation_labels, evaluation_data, evaluation_labels = test_split(batch,config.n_shot,config.n_class,config.dev,config.dataset)
            query_predictions = base_learner(evaluation_data)
            support_predictions = base_learner(adaptation_data)
            
            
            analysis_data[f"step-{step}"]["query-accuracy"].append(accuracy(query_predictions,evaluation_labels).item())
            analysis_data[f"step-{step}"]["query-loss"].append(query_loss(query_predictions,evaluation_labels).item())
            analysis_data[f"step-{step}"]["query-entropy"].append(entropy(query_predictions).item())
            
            analysis_data[f"step-{step}"]["support-accuracy"].append(accuracy(support_predictions,adaptation_labels).item())
            analysis_data[f"step-{step}"]["support-loss"].append(query_loss(support_predictions,adaptation_labels).item())
            analysis_data[f"step-{step}"]["support-entropy"].append(entropy(support_predictions).item())
            
    return analysis_data

if __name__ == '__main__':
    dataset = "tiered-imagenet"
    models  = ["MAML","MetaSGD","TA_LSTM"]
    fig_labels  ={
        "MAML":"MAML",
        "MetaSGD":"MetaSGD",
        "TA_LSTM":"MetaLSTM++"
    }
    metrics_to_plot = ["query-accuracy", "query-loss","support-accuracy","support-loss","support-entropy","query-entropy","stepwise_path_length"]
    n_ways  = [20, 40, 90]
    n_shots = [1]
    plot_data = {}
    overfit_data = {}
    redo = []
    plot_file = f"results/{dataset}/files/adaptation"

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
                config.dataset = dataset
                config.n_shot = n_shot
                config.algo = model_name
                config = process_config(config)
                
                overfit_step = config.adaptation_steps
                config.adaptation_steps = 16
                overfit_data[f"{n_class}.{n_shot}.{model_name}"] = overfit_step
                config.base_learner = Learner(config.image_size, config.bn_eps, config.bn_momentum, config.n_class,n_channels=config.n_channels,n_filters=config.n_filters).to(config.dev)
                config.meta_opt     = torch.optim.Adam
                if(f"{n_class}.{n_shot}.{model_name}" not in plot_data or f"{n_class}.{n_shot}.{model_name}" in redo):                
                    model               = algos[config.algo](config) 
                    if("TA_LSTM" in model_name):
                            for param in model.meta_learner.parameters():
                                param.requires_grad = False
                    trajectory = model.meta_test(return_trajectory=True)
                    batch = [trajectory[tid]["batch"] for tid in trajectory]
                    phis  = np.array([trajectory[tid]["model"] for tid in trajectory])
                    phis  = torch.from_numpy(phis).to(config.dev).requires_grad_(False)
                    metric_data = run_analysis(batch,config.base_learner,phis,config)
                    plot_data[f"{n_class}.{n_shot}.{model_name}"] = metric_data
                
                with open(plot_file,'wb') as file:
                    pickle.dump(plot_data,file)


    sns.set_style("whitegrid")
    plt.rcParams["axes.edgecolor"] = "0.15"
    plt.rcParams["axes.linewidth"]  = 1.50

    for metric in metrics_to_plot:
        for n_shot in n_shots:
            for i,model in enumerate(models):
                plt.figure(figsize=(5,5))
                colors = iter(cm.Set1(np.linspace(0, 1, len(n_ways))))
                for n_class in n_ways:
                    color = next(colors)
                    x_axis = []
                    for a in [int(a.split('-')[1]) for a in plot_data[f"{n_class}.{n_shot}.{model}"]]:
                        if(f"{metric}" in plot_data[f"{n_class}.{n_shot}.{model}"][f"step-{a}"]):
                            x_axis.append(a)

                    y_axis = np.array([plot_data[f"{n_class}.{n_shot}.{model}"][f"step-{x}"][f"{metric}"] for x in x_axis])
                    overfit_step = overfit_data[f"{n_class}.{n_shot}.{model}"]
                    label=f"{n_class}.{n_shot}" 
                    if(len(y_axis.shape)>1):
                        y_axis = np.mean(y_axis,axis=-1)
                    plt.plot(x_axis[:overfit_step+1],y_axis[:overfit_step+1],lw=2.0,label=label,linestyle='-',color=color)
                    if(overfit_step<=16):
                        start = 0 if overfit_step-2 < 0 else overfit_step-2
                        plt.plot(x_axis[start:],y_axis[start:],lw=2.0,linestyle='--',color=color)
                    
                    plt.xlabel("Adaptation Steps",fontsize=14,fontweight="bold")
                    plt.ylabel(f"{metric}",fontsize=14,fontweight="bold")
                    
                plt.title(f"{fig_labels[model]}",fontsize=14,fontweight="bold")
                plt.legend(fontsize=12)
                plt.xticks(fontsize=12)
                plt.yticks(fontsize=12)
                os.makedirs(f"results/{dataset}/stepwise/per-model",exist_ok=True)
                plt.savefig(f"results/{dataset}/stepwise/per-model/ALL.{n_shot}_{metric}_{fig_labels[model]}_stepwise.png",bbox_inches="tight")
                plt.close()

    for metric in metrics_to_plot:
        for n_shot in n_shots:
            for n_class in n_ways:
                colors = iter(cm.Set1(np.linspace(0, 1, len(models))))
                plt.figure(figsize=(5,5))
                for i,model in enumerate(models):
                    color = next(colors)
                    x_axis = []
                    for a in [int(a.split('-')[1]) for a in plot_data[f"{n_class}.{n_shot}.{model}"]]:
                        if(f"{metric}" in plot_data[f"{n_class}.{n_shot}.{model}"][f"step-{a}"]):
                            x_axis.append(a)
                    y_axis = np.array([plot_data[f"{n_class}.{n_shot}.{model}"][f"step-{x}"][f"{metric}"] for x in x_axis])
                    overfit_step = overfit_data[f"{n_class}.{n_shot}.{model}"]
                    if(len(y_axis.shape)>1):
                        y_axis = np.mean(y_axis,axis=-1)
                    
                    label=f"{fig_labels[model]}" 
                    plt.plot(x_axis[:overfit_step+1],y_axis[:overfit_step+1],lw=2.0,label=label,linestyle='-',color=color)
                    if(overfit_step<=16):
                        start = 0 if overfit_step-2 < 0 else overfit_step-2
                        plt.plot(x_axis[start:],y_axis[start:],lw=2.0,linestyle='--',color=color)
                    
                    plt.xlabel("Adaptation Steps",fontsize=14,fontweight="bold")
                    plt.ylabel(f"{metric}",fontsize=14,fontweight="bold")
                    
                plt.title(f"{n_class}.{n_shot}",fontsize=14,fontweight="bold")
                plt.legend(fontsize=12)
                plt.xticks(fontsize=12)
                plt.yticks(fontsize=12)
                os.makedirs(f"results/{dataset}/stepwise/per-setting",exist_ok=True)
                plt.savefig(f"results/{dataset}/stepwise/per-setting/{n_class}.{n_shot}_{metric}_stepwise.png",bbox_inches="tight")
                plt.close()