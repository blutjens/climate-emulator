#!/bin/bash
#
# SLURM script to kick off many jobs on the svante cluster to fit pattern scaling.
#
# Name of submitted job:
#SBATCH -J ebm_quad # 2K_draw_ornstein_uhlenbeck_process_experiment
# ** THE FOLDER HAS TO BE CREATED BEFOREHAND ** Otherwise unexplained JobLaunchFailure occurs.
#SBATCH -o runs/energy_balance_nonlinear/logs/task-%a.sh.log # outdir directory
#   #SBATCH -o runs/pattern_scaling/mpi-esm1-2-lr/m_member_subsets_with_m50_replace_False_eval_on_all_manyr/logs/task-6.sh.log # outdir directory for testing
# Create job array. Slurm will execute this same script
# in a 'throughput' parallel style
#   #SBATCH --array 1-5 # For pattern scaling with 20 random draws per subset 
#SBATCH --array 1-1# Test job array with single job.
#SBATCH --cpus-per-task 16 # number of workers. 
#SBATCH -p fdr # partition
#SBATCH -t 48:00:00

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

# Use these lines if wanting to test run a single job in a specific task folder
#   export SLURM_ARRAY_TASK_ID=6
#   export SLURM_ARRAY_TASK_COUNT=6

# Initialize the list
#cos_amplitudes=(0.5 0.4 0.3 0.2 0.1S)
#cos_amplitude=${cos_amplitudes[$SLURM_ARRAY_TASK_ID-1]}

# Run the script
python emcli2/utils/stochastic_process.py \
--process_name 'energy_balance_nonlinear' \
--nonlinearity 'quadratic'
# --cos_amplitude $cos_amplitude \