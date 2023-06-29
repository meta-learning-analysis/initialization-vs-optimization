#!/bin/bash
mkdir -p "console/"$2"/"$1

touch dynamic_script.sh
> dynamic_script.sh
printf "#!/bin/bash \n" >> dynamic_script.sh
printf "#SBATCH --job-name=%s            # create a short name for your job \n" $2-$1               >> dynamic_script.sh
printf "#SBATCH -N 1                     # node count \n"                                           >> dynamic_script.sh
printf "#SBATCH --ntasks-per-node=1      # total number of tasks per nodes \n"                      >> dynamic_script.sh
printf "#SBATCH --gres=gpu:A100-SXM4:1 \n"                                                          >> dynamic_script.sh
printf "#SBATCH --time=36:00:00          # total run time limit (HH:MM:SS)\n"                       >> dynamic_script.sh
printf "#SBATCH --output=./console/%s/%s/console.out    \n" $2 $1                                   >> dynamic_script.sh
printf "#SBATCH --error=./console/%s/%s/console.err      \n" $2 $1                                  >> dynamic_script.sh
printf "cd /nlsasfs/home/metalearning/aroof/Bharat/UNICORN-MAML \n"                                 >> dynamic_script.sh
printf "source /nlsasfs/home/metalearning/aroof/sahil/einsum-spline-flows/venv/bin/activate \n"     >> dynamic_script.sh
printf "bash %s \n" $3                                                                              >> dynamic_script.sh 

sbatch dynamic_script.sh
