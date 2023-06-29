#!/bin/bash 
#SBATCH --job-name=LR-MUL-1-5.1            # create a short name for your job 
#SBATCH -N 1                     # node count 
#SBATCH --ntasks-per-node=1      # total number of tasks per nodes 
#SBATCH --gres=gpu:A100-SXM4:1 
#SBATCH --time=36:00:00          # total run time limit (HH:MM:SS)
#SBATCH --output=./console/LR-MUL-1/5.1/console.out    
#SBATCH --error=./console/LR-MUL-1/5.1/console.err      
cd /nlsasfs/home/metalearning/aroof/Bharat/UNICORN-MAML 
source /nlsasfs/home/metalearning/aroof/sahil/einsum-spline-flows/venv/bin/activate 
bash scripts/TI-5.1.sh 
