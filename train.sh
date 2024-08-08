#!/bin/bash
#
# SLURM script to kick off many jobs on the svante cluster to fit cnn-lstm.
#
# Name of submitted job
#SBATCH -J cnn_edr_pr # m_member_subsets_with_m50_eval_on_all_spcmp_dwp_manyr
# ** THE FOLDER HAS TO BE CREATED BEFOREHAND ** Otherwise unexplained JobLaunchFailure occurs.
#SBATCH -o runs/cnn_lstm/mpi-esm1-2-lr/m_member_subsets_with_m50_eval_on_all_spcmp_dwp_manyr/logs/task-%a.sh.log # outdir directory
#   #SBATCH -o runs/cnn_lstm/mpi-esm1-2-lr/m_member_subsets_with_m50_eval_on_all_spcmp_dwp_manyr/logs/task-6.sh.log # outdir directory
# Create job array. Slurm will execute this same script
# in a 'throughput' parallel style
#SBATCH --array 1-8400 # %n limits to n jobs running at a time
#   #SBATCH --array 1-1# Create job array
#SBATCH --cpus-per-task 4 # number of workers. 
#SBATCH -n 4 # number of workers. 
#SBATCH -p edr # partition. 
#SBATCH -t 4:00:00

echo "hello"
echo 'Job running on node(s): '$SLURM_JOB_NODELIST
echo 'CPUs(x cores) per task: '$SLURM_TASKS_PER_NODE
echo 'Task id: '$SLURM_ARRAY_TASK_ID
echo 'Num tasks '$SLURM_ARRAY_TASK_COUNT
echo 'Path:' 
pwd

source /etc/profile.d/modules.sh # supercloud: source /etc/profile
# eval "$(conda shell.bash hook)"
# conda deactivate
module load anaconda3/2023.07 # supercloud: anaconda/2023a-pytorch; eofe: anaconda3/2020.11
source activate emcli # supercloud: conda activate emcli
export CUBLAS_WORKSPACE_CONFIG=:4096:8 # sets GPU operations like gaussian_kernel to deterministic
# export WANDB_MODE='offline'

# Use these lines if wanting to run job in a specific task folder
#   export SLURM_ARRAY_TASK_ID=6
#   export SLURM_ARRAY_TASK_COUNT=6

# Run the script
# python emcli2/models/pattern_scaling/model.py \
python emcli2/models/cnn_lstm/train.py \
--train_m_member_subsets \
--sweep \
--cfg_path 'runs/cnn_lstm/mpi-esm1-2-lr/m_member_subsets_with_m50_eval_on_all_spcmp_dwp_manyr/config/config.yaml' \
--task_id $SLURM_ARRAY_TASK_ID \
--num_tasks $SLURM_ARRAY_TASK_COUNT \
--data_var 'pr'