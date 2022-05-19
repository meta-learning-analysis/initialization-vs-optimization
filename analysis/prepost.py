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
    for param in base_learner.parameters():
        param.requires_grad = False
        
    # base_learner.model = torch.nn.DataParallel(base_learner.model).to(torch.device("cuda"))
    entropy = Entropy()
    euclidean_lsn = SoftNearestNeighbourLoss(distance='euclidean')
    cosine_lsn = SoftNearestNeighbourLoss(distance='cosine')
    
    analysis_data = {}
    
    for step in [0,-1]:
        analysis_data[f"step-{step}"] = {
                    "query-entropy":[],
                    "support-entropy":[],
                    "euclidean-lsn-feature":[],
                    "cosine-lsn-feature":[],
                    "euclidean-lsn-output":[],
                    "cosine-lsn-output":[]
        } 
        for tid in range(len(phis)):
            batch = task_batch[tid]
            
            phi = phis[tid,step,:]
            base_learner.copy_flat_params(phi)
            
            adaptation_data, adaptation_labels, evaluation_data, evaluation_labels = test_split(batch,config.n_shot,config.n_class,config.dev,config.dataset)
            query_features = base_learner.extract_features(evaluation_data).reshape(len(evaluation_data),-1)
            query_predictions = torch.nn.Softmax(dim=-1)(base_learner(evaluation_data))
            support_predictions = torch.nn.Softmax(dim=-1)(base_learner(adaptation_data))
            
            analysis_data[f"step-{step}"]["query-entropy"].append(entropy(preds=query_predictions))
            analysis_data[f"step-{step}"]["support-entropy"].append(entropy(preds=support_predictions))

            analysis_data[f"step-{step}"]["euclidean-lsn-feature"].append(euclidean_lsn(query_features,evaluation_labels))
            analysis_data[f"step-{step}"]["cosine-lsn-feature"].append(cosine_lsn(query_features,evaluation_labels))

            analysis_data[f"step-{step}"]["euclidean-lsn-output"].append(euclidean_lsn(query_predictions,evaluation_labels))
            analysis_data[f"step-{step}"]["cosine-lsn-output"].append(cosine_lsn(query_predictions,evaluation_labels))

            # print(f"step-{step} ","euclidean-lsn-feature : ",analysis_data[f"step-{step}"]["euclidean-lsn-feature"][-1]," Feature mean : ",query_features.mean(dim=-1).mean().item()," Feature std : ",query_features.std(dim=-1).mean().item())
    return analysis_data

if __name__ == '__main__':
    dataset = "tiered-imagenet"
    load_saved_trajectory = True
    models  = ["MAML","MetaSGD","TA_LSTM"]
    fig_labels  ={
        "MAML":"MAML",
        "MetaSGD":"MetaSGD",
        "TA_LSTM":"MetaLSTM++"
    }
    metrics_to_plot = ["support-entropy","query-entropy"]
    n_ways  = [5, 20, 40, 90]
    n_shots = [1]
    plot_data = {}
    plot_file = f"results/{dataset}/files/prepost"
    # redo = ["MAML","MetaSGD","TA_LSTM"]
    redo = []
    if(os.path.exists(plot_file)):
            with open(plot_file,'rb') as file:
                plot_data = pickle.load(file)

    for n_shot in n_shots:
        for n_class in n_ways:
            for model_name in models:
                print("--"*50)
                print(f" \t {n_class} WAY {n_shot} SHOT - {model_name} ")
                
                config = Config()
                config.test_size = 10
                config.n_class = n_class
                config.n_shot = n_shot
                config.dataset = dataset
                config.algo = model_name
                config = process_config(config)
                            
                if(f"{n_class}.{n_shot}.{model_name}" not in plot_data or f"{n_class}.{n_shot}.{model_name}" in redo or model_name in redo):                
                        
                    config.base_learner = Learner(config.image_size, config.bn_eps, config.bn_momentum, config.n_class,n_channels=config.n_channels,n_filters=config.n_filters).to(config.dev)
                    config.meta_opt     = torch.optim.Adam
                    
                    model               = algos[config.algo](config) 
                    
                    if("TA_LSTM" in model_name):
                        for param in model.meta_learner.parameters():
                            param.requires_grad = False
                    trajectory = model.meta_test(return_trajectory=True)
                    # for key in trajectory:
                    #     print(trajectory[key])
                    with torch.no_grad():
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
            # fig,axs = plt.subplots(1,len(models),figsize=(5*len(models),5))
            for n_class in n_ways:
                colors = iter(cm.Set1(np.linspace(0, 1, len(models))))
                plt.figure(figsize=(5,5))
                for i,model in enumerate(models):
                    color = next(colors)
                    x_axis = n_ways
                    y_axis_pre  = np.array([np.mean(plot_data[f"{x}.{n_shot}.{model}"][f"step-{0}"][f"{metric}"]) for x in x_axis])
                    y_axis_post = np.array([np.mean(plot_data[f"{x}.{n_shot}.{model}"][f"step-{-1}"][f"{metric}"]) for x in x_axis])
                    
                    label=f"{fig_labels[model]}" 

                    
                    plt.plot(x_axis,y_axis_pre,marker='o',label=f"{label}-PreAdapt",linestyle='--',color=color)
                    plt.plot(x_axis,y_axis_post,marker='D',label=f"{label}-PostAdapt",linestyle='-',color=color)

                    plt.xlabel("No. of Ways",fontsize=14,fontweight="bold")
                    plt.ylabel(f"{metric}",fontsize=14,fontweight="bold")
                    
                # plt.title(f"{n_class}.{n_shot}",fontsize=14,fontweight="bold")
                plt.legend(ncol=len(models),fontsize=8,loc=(-0.09,1.01))
                plt.xticks(fontsize=12)
                plt.yticks(fontsize=12)
                os.makedirs(f"results/{dataset}/prepost",exist_ok=True)
                plt.savefig(f"results/{dataset}/prepost/{n_shot}shot_{metric}.png",bbox_inches="tight")
                plt.close()
