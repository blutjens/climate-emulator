import argparse
import glob
import random 
import math
import yaml
import numpy as np
from netCDF4 import Dataset as ncDataset # Without loading this module before xr, xr.open_dataset might cause OSError: [Errno -101] NetCDF: HDF error
import xarray as xr
import pandas as pd
import gc
from pathlib import Path
from tqdm.notebook import tqdm
from xskillscore import rmse
from dask.diagnostics import ProgressBar

import os
os.environ["KERAS_BACKEND"] = "torch"
import keras
from keras import Sequential
from keras.layers import Dense, Activation, Conv2D, Flatten, Input, Reshape, AveragePooling2D, MaxPooling2D, Conv2DTranspose, TimeDistributed, LSTM, GlobalAveragePooling2D, BatchNormalization
from keras.regularizers import l2
from tensorflow.data import Dataset
import wandb
try:
    from wandb.keras import WandbMetricsLogger
except:
    # todo: write "If wandb version >= 0.17.0" instead of try-except
    from wandb.integration.keras import WandbMetricsLogger
import torch

from emcli2.utils.utils import init_sweep_config
from emcli2.utils.utils import set_all_seeds
from emcli2.dataset.interim_to_processed import calculate_global_weighted_average
from emcli2.dataset.mpi_esm1_2_lr import get_filepaths_mpi_esm1_2_lr
from emcli2.utils.metrics import LatWeightedMeanSquaredError
from emcli2.dataset.mpi_esm1_2_lr import get_meta
from emcli2.dataset.climatebench import load_climatebench_inputs_as_np_time_series
from emcli2.dataset.mpi_esm1_2_lr import load_mpi_data_as_xr
from emcli2.dataset.mpi_esm1_2_lr import create_train_splits
from emcli2.dataset.mpi_esm1_2_lr import convert_xr_to_np_timeseries
from emcli2.utils.metrics import calculate_nrmse
from emcli2.dataset.utils import get_random_m_member_subsets

def init_model(X_train_all, 
               Y_train_all, 
               cfg, 
               lats=None, 
               lons=None,
               device='cpu',
               verbose=True,
               seed=42):
    """
    X_train_all shape: n_samples, n_time, n_lat, n_lon, n_var
    Y_train_all shape: n_samples, n_time, n_lat, n_lon
    """
    random.seed(seed)
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    keras.utils.set_random_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic=True

    in_shape = X_train_all.shape[1:]
    out_shape = Y_train_all.shape[1:]

    keras.backend.clear_session()
    cnn_model = None

    cnn_model = Sequential()
    cnn_model.add(Input(shape=in_shape))
    cnn_model.add(TimeDistributed(Conv2D(20, (3, 3), padding='same', activation='relu'))) # , input_shape=in_shape))
    cnn_model.add(TimeDistributed(AveragePooling2D(2)))
    cnn_model.add(TimeDistributed(GlobalAveragePooling2D()))
    cnn_model.add(LSTM(25, activation='relu'))
    cnn_model.add(Dense(np.prod(out_shape)))
    cnn_model.add(Activation('linear'))
    cnn_model.add(Reshape(out_shape))

    if verbose:
        print(f'input shape: {in_shape}, output shape: {out_shape}')
        cnn_model.summary()

    if cfg['loss_function'] == 'weighted_mse' and lats is not None and lons is not None:
        loss = LatWeightedMeanSquaredError(
            lats = lats,
            lons = lons,
            device = device)
    else:
        loss = 'mse'

    if cfg['optimizer'] == 'adam':
        optimizer = keras.optimizers.Adam(
            learning_rate=cfg['learning_rate'],
            weight_decay=cfg['weight_decay'])
    else:
        optimizer = keras.optimizers.RMSprop(
            learning_rate=cfg['learning_rate'],
            weight_decay=cfg['weight_decay'])

    cnn_model.compile(optimizer=optimizer, loss=loss, metrics=['mse'])

    return cnn_model

def init_callbacks(cfg, no_wandb=False):
    # make directories
    Path(cfg['path_checkpoints']).mkdir(parents=True, exist_ok=True)
    Path(cfg['path_logs']).mkdir(parents=True, exist_ok=True)
                                 
    # Init callbacks    
    path_checkpoints = cfg['path_checkpoints'] + 'model.e{epoch:02d}-{val_loss:.4f}.keras' # runs/cnn-lstm/mpi-esm1-2-lr/
    # path_checkpoints = cfg['path_checkpoints'] + 'model.incumbent.keras' # runs/cnn-lstm/mpi-esm1-2-lr/
    my_callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=cfg['early_stopping_patience'], mode='min'),
        keras.callbacks.ModelCheckpoint(filepath=path_checkpoints,
                                        save_best_only=True,
                                        monitor='val_loss',
                                        mode='min')
        ]

    if no_wandb == True:
        logger = None
    else:
        Path(cfg['path_wandb']).mkdir(parents=True, exist_ok=True)

        # Initialize a new W&B run
        wandb_run = wandb.init(project=cfg['wandb_project_name'], 
                entity='blutjens',
                resume='allow', 
                anonymous='must',
                dir=cfg['path_wandb'],
                config=cfg,
                allow_val_change=True,)

        logger = WandbMetricsLogger()
        my_callbacks.append(logger)
    return my_callbacks

import logging

def get_args():
    parser = argparse.ArgumentParser(description='Train ClimateBench CNN-LSTM model')
    parser.add_argument('--data_var', type=str, default=None, help='Data variable that fit')
    parser.add_argument('--cfg_path', type=str, default='runs/cnn_lstm/mpi-esm1-2-lr/default/config/config.yaml', help='Path to config yaml')
    parser.add_argument('--train_m_member_subsets', action='store_true', default=False, help=
                        'Train the neural network on the mean of randomly picked subsets of realizations from '\
                        'the target climate model.')
    parser.add_argument('--no_wandb', action='store_true', default=False, help='Disable wandb logs')
    parser.add_argument('--debug', action='store_true', default=False, help='debug')
    parser.add_argument('--verbose', action='store_true', default=False, help='verbose print statements')
    parser.add_argument('--task_id', type=int, default=1, help='SLURM task id, when script is called in job array')
    parser.add_argument('--num_tasks', type=int, default=1,
                        help='Total number of SLURM tasks when script is called in job array')
    parser.add_argument('--sweep', action='store_true', default=False,
                        help='If true, indicates that program is running a hyperparameter sweep')
    parser.add_argument('--overwrite_m_member_subsets_csv', action='store_true', default=False, help='If true, will overwrite m_member_subsets.csv file that contains '\
                         'a list of realization IDs for the random member subsets')
    return parser.parse_args()

if __name__ == '__main__':
    # Get command line arguments
    args = get_args()
    if args.data_var is None:
        data_vars = ['pr', 'tas']# , 'vas','dtr','uas',  'pr90', 'huss', 'psl'] # 'pr90', 'dtr'
    else:
        data_vars = [args.data_var]

    # Initialize logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # Import cfg
    cfg = yaml.safe_load(open(args.cfg_path, 'r'))
    cfg['path_cfg'] = args.cfg_path

    # Init cpu or gpu
    if cfg['device'] == 'cpu' or not torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')
        os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
    print(f'Using device {device}')

    # Initialize hyperparameter sweep
    if args.sweep:
        cfg = init_sweep_config(cfg, cfg['path_sweep_cfg'], args.task_id, args.num_tasks)

        # Skip random seeds that have already been run
        rm_vars = []
        for data_var in data_vars:
            cfg['path_checkpoints'] = f'{cfg["path_overflow"]}/{data_var}/memberseed-{cfg["memberseed"]}/member_subset{cfg["idx_member_subset"]}/sweep/task-{args.task_id}/checkpoints/'
            if Path(cfg['path_checkpoints']).exists():
                print(f'WARNING Sweep for {data_var}/memberseed-{cfg["memberseed"]}/member_subset{cfg["idx_member_subset"]}/sweep/task-{args.task_id} already exists. Skipping.')
                rm_vars.append(data_var)
        for var in rm_vars:
            data_vars.remove(var)
        if not data_vars:
            print('All data_vars already swept. Exiting program.')
            exit()

    # Set seeds
    set_all_seeds(cfg['seed'], device=device.type, 
                  use_deterministic_algorithms=cfg['use_deterministic_algorithms'],
                  warn_only=cfg['warn_only'])

    if args.train_m_member_subsets:
        filename_wildcard = 'ensemble.nc'
    else:
        filename_wildcard = 'ensemble_summaries_yr.nc'

    scenarios_aux = [] # ['ssp119'] # auxiliary scenarios that are not used for train or test, e.g., 1pctCO2, picontrol, hist-aer, hist-GHG, abrupt-4xCO2
    scenarios_test = ['ssp245']
    scenarios_train = ['historical', 'ssp126', 'ssp370','ssp585'] # 1pctCO2, picontrol, hist-aer, hist-GHG, abrupt-4xCO2
    if args.debug:
        scenarios_train = ['historical']
        cfg['epochs'] = 3

    meta = get_meta(cfg['data_path'], cfg['data_path_interim'])
    # Add 'tas' to filepath list, because LPS always needs it as global intermediate variable
    data_vars_fp = data_vars if 'tas' in data_vars else data_vars + ['tas'] 
    frequencies = [meta[data_var]['frequency'] for data_var in data_vars_fp]

    scenarios = scenarios_train + scenarios_test + scenarios_aux
    df = get_filepaths_mpi_esm1_2_lr(data_vars=data_vars_fp,
        scenarios=scenarios,
        data_path=cfg['data_path_interim'],
        filename_wildcard=filename_wildcard,
        frequencies=frequencies,
        verbose=args.verbose)

    # Load input data, input4mips, from ClimateBench
    assert 'historical' in scenarios, 'historical scenario must be included in scenarios'
    X_train_all, meanstd_inputs = load_climatebench_inputs_as_np_time_series(
        scenarios=scenarios_train, 
        data_path=cfg['data_path_climatebench'],
        split='train',
        slider=cfg['slider'],
        normalize=True
    )

    X_val_np, _ = load_climatebench_inputs_as_np_time_series(
        scenarios=scenarios_test,
        data_path=cfg['data_path_climatebench'],
        split='test',
        meanstd_inputs=meanstd_inputs,
        slider=cfg['slider'],
        normalize=True,
        verbose=False
    )

    for data_var in data_vars:
        df_data_var = df[df.data_var == data_var]
        cfg['data_var'] = data_var

        if args.train_m_member_subsets:
            cfg['dir_m_members_subsets_csv'] = f'{cfg["path_experiment"]}/{data_var}/memberseed-{cfg["memberseed"]}/'
            m_member_subsets, all_member_ids, idcs_member_subsets = get_random_m_member_subsets(df=df_data_var,
                verify_all_scenarios_have_same_member_coords=True,
                skip_subsets=cfg['skip_subsets'],
                max_number_of_members_per_subset=cfg['max_number_of_members_per_subset'],
                dir_m_members_subsets_csv=cfg['dir_m_members_subsets_csv'],
                filename_m_members_subsets_csv=cfg['filename_m_members_subsets_csv'],
                equal_number_of_members_in_each_subset=cfg['equal_number_of_members_in_each_subset'],
                number_of_subsets=cfg['number_of_subsets'],
                overwrite=args.overwrite_m_member_subsets_csv,
                member_mode=cfg['member_mode'],
                replace=cfg['replace'],
                idx_member_subset=cfg['idx_member_subset'],
                seed=cfg['memberseed']
            )
        else:
            m_member_subsets = [None]
            all_member_ids = [None]
            idcs_member_subsets = [0]
            cfg['memberseed'] = cfg['memberseed'] if 'memberseed' in cfg else 0

        climatology = None
        incumbent = dict()

        for s_idx, m_member_subset in zip(idcs_member_subsets, m_member_subsets):
            print('#################')
            if m_member_subset is not None:
                print(f'Training on subset idx={s_idx} with m={len(m_member_subset)} members on task-{args.task_id}:')
                print(m_member_subset)
            else:
                print(f'Training on ensemble-mean of all members.')
            print('scenarios_test, data_var:', scenarios_test, data_var)
            print('#################')

            cfg['open_data_parallel'] = cfg['open_data_parallel'] if 'open_data_parallel' in cfg else True
            datasets, climatology = load_mpi_data_as_xr(df=df_data_var, meta=meta,
                                climatology=climatology, 
                                m_member_subset=m_member_subset,
                                open_data_parallel=cfg['open_data_parallel'])

            Y_train, Y_test, Y_aux = create_train_splits(datasets, data_var=data_var, 
                                                        scenarios_train=scenarios_train, 
                                                        scenarios_test=scenarios_test, 
                                                        scenarios_aux=scenarios_aux,
                                                        verbose=True)

            coords = Y_test[0].coords

            if 'eval_on_all_member_mean' in cfg:
                if cfg['eval_on_all_member_mean']:
                    print('Eval on all member mean')
                    # Replace the test dataset with the mean of all realizations
                    [yt.close() for yt in Y_test];
                    del Y_test
                    gc.collect()
                    df_test = pd.concat((df_data_var[df_data_var.scenario == scenario_test] for scenario_test in (['historical'] + scenarios_test)))
                    datasets_test, climatology = load_mpi_data_as_xr(df=df_test, meta=meta,
                                climatology=climatology, 
                                m_member_subset=all_member_ids)
                    _, Y_test, _ = create_train_splits(datasets_test, data_var=data_var, 
                                                        scenarios_train=[], 
                                                        scenarios_test=scenarios_test, 
                                                        scenarios_aux=[],
                                                        verbose=(s_idx==0))

            Y_train_all, Y_val_np = convert_xr_to_np_timeseries(
                data_var=data_var, slider=cfg['slider'],
                Y_train=Y_train, scenarios_train=scenarios_train,
                Y_test=Y_test, scenarios_test=scenarios_test)
            
            train_dataset = Dataset.from_tensor_slices((X_train_all, Y_train_all))
            train_dataset = train_dataset.batch(cfg['batch_size']).repeat()

            val_dataset = Dataset.from_tensor_slices((X_val_np, Y_val_np))#, shuffle=False, seed=cfg['seed'])
            val_dataset = val_dataset.batch(cfg['batch_size']).repeat()

            incumbent[s_idx] = {'best_val_loss': 9999999999999}
            for seed in [cfg['seed']]:
                if args.sweep:
                    cfg['path_checkpoints'] = f'{cfg["path_overflow"]}/{data_var}/memberseed-{cfg["memberseed"]}/member_subset{s_idx}/sweep/task-{args.task_id}/checkpoints/'
                    cfg['path_logs'] = f'{cfg["path_experiment"]}/{data_var}/memberseed-{cfg["memberseed"]}/member_subset{s_idx}/sweep/task-{args.task_id}/logs'
                    cfg['path_wandb'] = f'{cfg["path_experiment"]}/{data_var}/memberseed-{cfg["memberseed"]}/member_subset{s_idx}/sweep/task-{args.task_id}'
                    path_incumbent = f'{cfg["path_experiment"]}{data_var}/memberseed-{cfg["memberseed"]}/member_subset{s_idx}/sweep/task-{args.task_id}/incumbents/'
                else:
                    cfg['path_checkpoints'] = f'{cfg["path_experiment"]}/{data_var}/checkpoints/'
                    cfg['path_logs'] = f'{cfg["path_experiment"]}/{data_var}/logs/'
                    cfg['path_wandb'] = f'{cfg["path_experiment"]}/{data_var}'
                    path_incumbent = f'{cfg["path_experiment"]}/{data_var}/incumbents/'
                my_callbacks = init_callbacks(cfg,
                    no_wandb=args.no_wandb)

                print(f'sweep-{args.task_id}: \n data_var: {data_var}, \n optimizer: {cfg["optimizer"]}, \n lr: {cfg["learning_rate"]}, \n weight_decay: {cfg["weight_decay"]}, \n batch_size: {cfg["batch_size"]}')

                cnn_model = init_model(X_train_all, 
                                    Y_train_all, 
                                    cfg, 
                                    lats=coords['latitude'], 
                                    lons=coords['longitude'],
                                    device=device,
                                    verbose=(args.task_id==0),
                                    seed=seed)
                hist = cnn_model.fit(train_dataset,
                                    # Y_train_all,
                                    validation_data=val_dataset,
                                    shuffle=True,
                                    validation_steps=len(X_val_np)//cfg['batch_size']+1,
                                    # batch_size=cfg['batch_size'],
                                    epochs=cfg['epochs'],
                                    steps_per_epoch=math.floor(X_train_all.shape[0]/16),
                                    initial_epoch=0,
                                    verbose=1,
                                    callbacks=my_callbacks)

                ckpts = glob.glob(cfg['path_checkpoints'] + '*.keras')
                ckpts.sort()
                best_checkpoint_path = ckpts[-1]
                best_val_loss = float(Path(best_checkpoint_path).stem.split('-')[-1])
                print(f'best loss {best_val_loss} for {s_idx}th subset\n'\
                    f'opt: {cfg["optimizer"]}, lr: {cfg["learning_rate"]}, weight_decay: {cfg["weight_decay"]}. Stored at:\n'\
                    f'{best_checkpoint_path}')

                ## Evaluate
                # Load best
                best_model = keras.models.load_model(best_checkpoint_path)
                # Apply model on test data
                m_pred = best_model.predict(X_val_np, verbose=0)
                # reshape to xarray
                len_historical = 165
                m_pred = m_pred.reshape(m_pred.shape[0], m_pred.shape[2], m_pred.shape[3])
                m_pred = xr.DataArray(m_pred, dims=['time', 'lat', 'lon'], coords=[coords['time'].data[len_historical:], coords['latitude'].data, coords['longitude'].data])
                m_pred = m_pred.transpose('lat', 'lon', 'time').sel(time=slice('2015', '2101')).to_dataset(name=data_var)
                preds_model_ds = [m_pred]
                preds_model_ds[0] = preds_model_ds[0].rename({'lat': 'latitude', 'lon': 'longitude'})

                # Calculate loss on test set
                Y_rmse_spatial = calculate_nrmse(Y_true=Y_test[0][data_var].compute(), Y_pred=preds_model_ds[0][data_var], normalize=False)
                Y_rmse_global = rmse(calculate_global_weighted_average(Y_test[0][data_var].sel(time=slice('2080',None))), 
                                    calculate_global_weighted_average(preds_model_ds[0][data_var].sel(time=slice('2080',None))))
                Y_rmse_global_np = Y_rmse_global.values
                Y_rmse_global.close(); del Y_rmse_global
                print(f'Spatial RMSE {Y_rmse_spatial}')
                print(f'Global RMSE {Y_rmse_global_np}')
                
                # Update incumbent
                if best_val_loss < incumbent[s_idx]['best_val_loss']:
                    print(f'New incumbent on {data_var} with subset #{s_idx}')
                    incumbent[s_idx] = {
                        'data_var': data_var,
                        'Y_rmse_spatial': Y_rmse_spatial.item(),
                        'Y_rmse_global_np': Y_rmse_global_np.item(),
                        'best_val_loss': best_val_loss,
                        'idx_member_subset': s_idx,
                        'task_id': args.task_id,
                        'm_member_subset': m_member_subset,
                        'best_checkpoint_path': best_checkpoint_path
                    }
                    incumbent[s_idx].update(cfg)
                    if not args.no_wandb:
                        print(f'New incumbent on wandb is {wandb.run.name}')
                        incumbent[s_idx]['wandb_run_id'] = wandb.run.id
                        incumbent[s_idx]['wandb_run_name'] = wandb.run.name
                    Path(path_incumbent).mkdir(parents=True, exist_ok=True)
                    with open(f'{path_incumbent}/incumbents.yaml', 'w') as file:
                        yaml.dump(incumbent, file)
                
                # Log to wandb
                if not args.no_wandb:
                    wandb.log({
                        'idx_member_subset': s_idx,
                        'm_member_subset': m_member_subset,
                        'Best Val Loss': best_val_loss,
                        'Best Checkpoint Path': best_checkpoint_path,
                        'Test spatial RMSE': Y_rmse_spatial,
                        'Test set global RMSE': Y_rmse_global_np,
                        'seed': seed,
                        'memberseed': cfg['memberseed'],
                    })
                    wandb.finish()
                
                gc.collect()
                
                print(f'Finished {data_var} {s_idx}th subset in task-{args.task_id}')
                # Close xarray datasets
                for data in Y_test + Y_train + Y_aux:
                    data.close(); del data
                del Y_test; del Y_train; del Y_aux
                gc.collect()