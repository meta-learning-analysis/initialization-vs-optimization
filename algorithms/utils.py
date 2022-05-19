
import numpy as np
import torch
import learn2learn as l2l
from models import Learner
import pickle 
import os 
import glob 
import pprint
import matplotlib.pyplot as plt 
import seaborn as sns 

def accuracy(predictions, targets):
    """Computes accuracy

    Args:
        predictions (torch.Tensor): The tensor consisting of the predicted probability values. Shape: batch x n_classes
        targets (torch.Tensor): The target labels. Shape: batch x 1

    Returns:
        float: The accuracy value
    """
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return 100*(predictions == targets).sum().float() / targets.size(0)

def test_split(batch,shots,ways,device,dataset):
    data, labels = batch
    data, labels = data.to(device), labels.to(device)
    if("omniglot" in dataset):
        adaptation_indices = np.zeros(data.size(0), dtype=bool)
        adaptation_indices[np.arange(shots*ways) * 2] = True
        evaluation_indices = torch.from_numpy(~adaptation_indices)
        adaptation_indices = torch.from_numpy(adaptation_indices)
        adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices]
        evaluation_data, evaluation_labels = data[evaluation_indices], labels[evaluation_indices]
    else:
        adaptation_indices = np.zeros(data.size(0), dtype=bool)
        adaptation_indices[[i*(15+shots)+j for i in range(ways) for j in range(shots)]] = True
        evaluation_indices = torch.from_numpy(~adaptation_indices)
        adaptation_indices = torch.from_numpy(adaptation_indices)
        adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices]
        evaluation_data, evaluation_labels = data[evaluation_indices], labels[evaluation_indices]
    return adaptation_data, adaptation_labels, evaluation_data, evaluation_labels

def preprocess_grad_loss(x):
    p = 10
    indicator = (x.abs() >= np.exp(-p)).to(torch.float32)
    # preproc1
    x_proc1 = indicator * torch.log(x.abs() + 1e-8) / p + (1 - indicator) * -1
    # preproc2
    x_proc2 = indicator * torch.sign(x) + (1 - indicator) * np.exp(p) * x
    return torch.stack((x_proc1, x_proc2), 1)

def get_flat_params(learner):
    return torch.cat([p.view(-1) for p in learner.parameters()], 0)

def fast_adapt(batch, learner, loss, adaptation_steps, shots, ways, device, dataset):
    learner.module = torch.nn.DataParallel(learner.module).to(torch.device("cuda"))
    adaptation_data, adaptation_labels, evaluation_data, evaluation_labels = test_split(batch,shots,ways,device,dataset)
    phi = [get_flat_params(learner).detach().cpu().numpy()]
    for step in range(adaptation_steps):
        train_error = loss(learner(adaptation_data), adaptation_labels)
        learner.adapt(train_error)
        phi.append(get_flat_params(learner).detach().cpu().numpy())
    predictions = learner(evaluation_data)
    valid_error = loss(predictions, evaluation_labels)
    valid_accuracy = accuracy(predictions, evaluation_labels)
    return valid_error, valid_accuracy, np.array(phi)
    
def evaluate_learner(batch, learner, loss, shots, ways, device, dataset):
    adaptation_data, adaptation_labels, evaluation_data, evaluation_labels = test_split(batch,shots,ways,device,dataset)
    predictions =  learner(evaluation_data)
    valid_error =  loss(predictions, evaluation_labels)
    valid_accuracy = accuracy(predictions, evaluation_labels)
    return valid_error, valid_accuracy


def train_learner(learner_w_grad, metalearner, train_input, train_target, args):
    cI = metalearner.metalstm.cI.data
    hs = [None]
    phi = [np.squeeze(cI.detach().cpu().numpy())]
    for adaptation_step in range(args.adaptation_steps):
        learner_w_grad.copy_flat_params(cI)
        output = learner_w_grad(train_input)
        loss = learner_w_grad.criterion(output, train_target)
        acc = accuracy(output, train_target)
        learner_w_grad.zero_grad()
        loss.backward()
        grad = torch.cat([p.grad.data.reshape(-1) for p in learner_w_grad.parameters()], 0)
        grad_prep = preprocess_grad_loss(grad)  # [n_learner_params, 2]
        loss_prep = preprocess_grad_loss(loss.data.unsqueeze(0)) # [1, 2]
        metalearner_input = [loss_prep, grad_prep, grad.unsqueeze(1)]
        cI, h = metalearner(metalearner_input, hs[-1])
        hs.append(h)
        phi.append(np.squeeze(cI.detach().cpu().numpy()))
    return cI, np.array(phi)


def load_history(log_dir, filename="training_history"):
    history = {}
    if(os.path.exists("{}/{}.json".format(log_dir,filename))):
        with open("{}/{}.json".format(log_dir,filename), "rb") as file:
            history = pickle.load(file)    
    return history



def save_ckpt(episode=-1, metalearner=None, optim=None, save="./"):
    print("saving: ", os.path.join(save, 'meta-learner-{}.pth.tar'.format(episode)))
    os.makedirs(save,exist_ok=True)
    torch.save({
        'episode':     episode,
        'metalearner': metalearner.state_dict(),
        'optim':       optim.state_dict()
    }, os.path.join(save, 'meta-learner-{}.pth.tar'.format(episode)))

def resume_ckpt(metalearner, optim, resume, device,ckpt_no=None):
    list_of_files = glob.glob(resume+'/*') # * means all if need specific format then *.csv
    
    if(ckpt_no is None):
        ckpt_no = max([int(a.split("meta-learner-")[1].split(".")[0]) for a in os.listdir(resume)])
    latest_file = resume+"/meta-learner-{}.pth.tar".format(ckpt_no)
    print("Resuming From : ", latest_file)
    ckpt = torch.load(latest_file, map_location=device)
    last_episode          = ckpt['episode']
    pretrained_state_dict = ckpt['metalearner']            
    metalearner.load_state_dict(pretrained_state_dict)
    optim.load_state_dict(ckpt['optim'])
    return last_episode, metalearner, optim



def plot(history,config):
    sns.set_style("whitegrid")
    
    for plot_no,key_1 in enumerate(history):
        plt.figure(figsize=(16,8))
        plt.title("{}".format(config.algo),fontsize=18)
        
        x_axis = history["iter"]["train"] if "iter" in history else range(len(history[key_1]["train"]))
        plt.plot(x_axis,history[key_1]["train"],label = "Training",color='b',linestyle='--',alpha=0.5)
        
        x_axis = history["iter"]["val"] if "iter" in history else range(0,config.val_freq*len(history[key_1]["val"]),config.val_freq)
        plt.plot(x_axis,history[key_1]["val"],label = "Validation",color='coral',linewidth=2)
        plt.axhline(np.mean(history[key_1]["test"]),color='teal',linestyle='--',label='Testing',linewidth=3)

        plt.legend(fontsize=15)
        plt.xlabel('Iterations',fontsize=15)
        plt.ylabel(' {} '.format(key_1) ,fontsize=15)
        plt.savefig("{}/{}.png".format(config.LOG_DIR,key_1))
        plt.close()
        

def save_history(training_history,config):
    
    with open("{}/config.txt".format(config.LOG_DIR), "wt") as file:
        pprint.pprint(vars(config), stream=file)

    data = training_history    
    # if(os.path.exists("{}/training_history.json".format(config.LOG_DIR)) and config.resume):    
    #     with open("{}/training_history.json".format(config.LOG_DIR), "rb") as file:
    #         data = pickle.load(file)
    #         for key_1 in data:
    #             for key_2 in data[key_1]:
    #                 data[key_1][key_2].extend(training_history[key_1][key_2])
            
    with open("{}/training_history.json".format(config.LOG_DIR), "wb") as file:
        pickle.dump(data, file)
    
    plot(data,config)