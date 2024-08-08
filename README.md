# climate-emulator

Official repository for the paper 'A cautionary tale about deep learning-based climate emulators and internal variability'. This repository compares linear pattern scaling against a CNN-LSTM for climate emulation. The models are compared on the dataset from ClimateBench and a new data summary from the MPI-ESM1.2-LR model. The repository contains all code to download data and reproduce the paper results.

For a tutorial on how to use pattern scaling for climate emulation, please see: 
[climate_emulator_tutorial.ipynb](https://nbviewer.org/github/blutjens/climate-emulator-tutorial/blob/main/climate_emulator_tutorial.ipynb)

### Installation
```
git clone git@github.com:blutjens/climate-emulator.git
cd climate-emulator
conda create --name emcli
conda activate emcli
conda install pip
pip install -r requirements.txt
pip install -e .
ipython kernel install --user --name=emcli # Link conda environment to jupyter notebook
```

### Download Em-MPI data summary (<10GB)
```
export DATA_DIR=/path/to/data/dir
mkdir -p $DATA_DIR
python download_emmpi.py --data_dir $DATA_DIR
# alternatively, follow instructions at https://huggingface.co/datasets/blutjens/em-mpi
```

### Download input4mips emission inputs and ClimateBench NorESM2-LM targets (<2GB)
```
export PATH_CLIMATEBENCH_DATA=$DATA_DIR/data/raw/climatebench/
mkdir -p $PATH_CLIMATEBENCH_DATA
wget https://zenodo.org/record/7064308/files/train_val.tar.gz -P $PATH_CLIMATEBENCH_DATA
tar -xvf "$PATH_CLIMATEBENCH_DATA/train_val.tar.gz" -C $PATH_CLIMATEBENCH_DATA
rm $PATH_CLIMATEBENCH_DATA/train_val.tar.gz
wget https://zenodo.org/record/7064308/files/test.tar.gz -P $PATH_CLIMATEBENCH_DATA
tar -xvf "$PATH_CLIMATEBENCH_DATA/test.tar.gz" -C $PATH_CLIMATEBENCH_DATA
rm $PATH_CLIMATEBENCH_DATA/test.tar.gz
```

### Reproduce linear pattern scaling (LPS) results on ClimateBench
```
# Calculate LPS entry on ClimateBench scoreboard; plot LPS error map for tas, pr, dtr, pr90; and plot correlation between temperature and global cumulative CO2
jupyter notebook notebooks/calculate_climatebench_metrics.ipynb

# The trained weights of the LPS model are also stored in runs/pattern_scaling/default/models/
```

### Reproduce internal variability experiment
#### First code test: Train and evaluate CNN-LSTM on 50-member ensemble-mean Em-MPI data
```
wandb login # call in a terminal with internet access.
export TF_GPU_ALLOCATOR=cuda_malloc_async
export KERAS_BACKEND=torch
# (optional) export WANDB_MODE='offline' # Use if compute node does have internet access.
vim runs/cnn_lstm/mpi-esm1-2-lr/default/config/config.yaml # Then edit paths to point to /path/to/data/dir
python emcli2/models/cnn_lstm/train.py --cfg_path 'runs/cnn_lstm/mpi-esm1-2-lr/default/config/config.yaml' --data_var 'pr' --verbose
# (optional) set config.yaml -> epochs=100 to train the CNN-LSTM and not just test if the training works.
```

#### Second code test: Train LPS and CNN-LSTM on single draws of subsets with 1,2,...,50 members. Then plot RMSE over number of realizations.
```
# (Need to first edit paths in below configs)

python emcli2/models/cnn_lstm/train.py --train_m_member_subsets --cfg_path runs/cnn_lstm/mpi-esm1-2-lr/m_member_subsets_with_m50_eval_on_all_spcmp_dwp_manyr/config/config.yaml --data_var pr

python emcli2/models/pattern_scaling/model.py --train_m_member_subsets --cfg_path runs/pattern_scaling/mpi-esm1-2-lr/m_member_subsets_with_m50_replace_False_eval_on_all_manyr/config/config.yaml --data_var pr

python emcli2/utils/plotting.py --plot_m_member_subsets_experiment --data_var pr
```

#### Full experiment: Train LPS and CNN-LSTM (with multiple seeds) on multiple draws of subsets with 1,2,...,50 members using SLURM. Plot RMSE over number of realizations including uncertainty bars.
```
# Send CNN-LSTM off to supercomputer
sbatch train.sh

# Send pattern scaling to supercomputer
sbatch train_pattern_scaling.sh

python emcli2/utils/plotting.py --plot_m_member_subsets_experiment --data_var pr

# repeat for --data_var pr and --data_var tas
```

### Reproduce the other figures in JAMES24 paper submission
```
# notebooks/explore_linear_relationships.ipynb -> Plot functional relationships in cumlative CO2 emissions, surface temperature, and precipitation; also plot for multiple regions
# notebooks/explore_local_internal_variability.ipynb -> Plot internal variability in 3-member NorESM2-LM vs 50-member MPI-ESM1.2-LR ensemble-mean; also plot for multiple regions
```

### Reproduce the Em-MPI data summary from raw CMIP6 data.
#### Download raw MPI-ESM1.2-LR from CMIP6 data on ESGF (tested on svante)
```
cd ClimateSet
# Download inputs / forcers:
python -m data_building.builders.downloader --cfg data_building/configs/downloader/core_dataset.yaml
# Download outputs / climate variables:
python -m data_building.builders.downloader --cfg data_building/configs/downloader/mpi-esm1-2-lr.yaml

# Delete years in piControl that are past np.datetime64. Maintain 400yrs.
cd /d0/lutjens/raw/CMIP6/MPI-ESM1-2-LR/
ls r1i1p1f1/piControl/@(tas|pr|uas|vas|psl|huss|tasmax|tasmin)/250_km//@(day|mon)/@(23*|24*|25*|26*|27*|28*|29*) # replace ls w. rm -r
ls r1i1p1f1/piControl/@(tas|pr|uas|vas|psl|huss|tasmax|tasmin)/250_km/@(day|mon)/@(225*|226*|227*|228*|229*) # replace ls w. rm -r
```

#### Reprocess MPI-ESM1.2-LR raw data to get the interim Em-MPI data
```
cd ~/climate-emulator
# Get ensemble_summary.nc that contains statistical summaries across members
# Monthly variables, e.g., tas, pr
python emcli2/dataset/mpi_esm1_2_lr.py --data_var 'huss' --get_ensemble_summaries

# Compute Diurnal Temperature Range; takes ~ 16min per scenario for 30 realizations. 
python emcli2/dataset/mpi_esm1_2_lr.py --data_var 'dtr' --get_ensemble_summaries

# Compute extreme precipitation; takes 5hrs 30min for historical scenario and 50 realizations.
python emcli2/dataset/mpi_esm1_2_lr.py --data_var 'pr90' --get_ensemble_summaries

# optional: Daily variables, e.g., pr, tasmax, tasmin
python emcli2/dataset/mpi_esm1_2_lr.py --data_var 'tasmax' --get_ensemble_summaries --frequency 'day'

# Get ensemble.nc that merges all years and ensemble members 
python emcli2/dataset/mpi_esm1_2_lr.py --data_var 'tas' --get_ensemble_concat
```

### Known issues
```
# In case of import tensorflow error in wandblogger. $ pip install tensorflow
# In case xr.openmfdataset(..., parallel=True) crashes: $ conda install --channel=conda-forge eccodes
# In case Segmentation fault: try config.yaml -> "open_data_parallel: False"
```

### Reference
If this repository is useful for your analysis please consider citing:
```
@article{lutjens24internalvar,
  title={A Cautionary Tale about Deep Learning-based Climate Emulators and Internal Variability},
  year={2024},
}
```