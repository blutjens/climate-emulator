import os
import argparse
import yaml
import random
import gc
import pickle # saving model
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from netCDF4 import Dataset as ncDataset # Without loading this module before xr, xr.open_dataset might cause OSError: [Errno -101] NetCDF: HDF error
import xarray as xr
import pandas as pd
from xskillscore import rmse
from sklearn.linear_model import LinearRegression
from emcli2.dataset.mpi_esm1_2_lr import get_meta
from emcli2.dataset.climatebench import load_climatebench_data
from emcli2.dataset.mpi_esm1_2_lr import get_filepaths_mpi_esm1_2_lr
from emcli2.dataset.utils import get_random_m_member_subsets
from emcli2.dataset.mpi_esm1_2_lr import load_mpi_data_as_xr
from emcli2.dataset.mpi_esm1_2_lr import create_train_splits
import emcli2.models.pattern_scaling.model as ps
import emcli2.dataset.interim_to_processed as i2p
from emcli2.utils.metrics import calculate_nrmse
from emcli2.dataset.interim_to_processed import calculate_global_weighted_average

class PatternScaling(object):
    """
    Does pattern scaling. Here we fit one linear model per 
    grid point. The linear model maps a global variable, 
    e.g., cum. CO2 emissions or temperature, to 
    the grid point's local value. 
    This model captures temporal patterns in each grid cell.
    The model is local, i.e., it will be independent of 
    neighboring grid points. The model is linear in time, 
    i.e., it assumes no non-linearly amplifying feedbacks 
    between the global and local variable.
    """
    def __init__(self, deg=1):
        """
        Args:
            deg int: degree of polynomial fit. Default is 1 
                for linear fit.
        """
        self.deg = deg
        self.coeffs = None

    def train(self, in_global, out_local):
        """
        Fits polynomial with degree self.deg from in_global to 
        every location in out_local. Choose deg=1 for linear fit.

        Args:
            in_global np.array((n_t,)): The model input is a 
                global variable, e.g., annual global mean surface 
                temperature anomalies of in °C
            out_local np.array((n_t,n_lat,n_lon)): The model 
                output is a locally-resolved variable. E.g., annual 
                mean surface temperature anomalies at every lat,lon
                in °C
        Sets:
            coeffs np.array((deg+1, n_lat, n_lon))
        """
        n_t, n_lat, n_lon = out_local.shape
        
        # Preprocess data, by flattening in space
        out_local = out_local.reshape(n_t,-1) # (n_t, n_lat*n_lon)

        # Fit linear regression coefficients to every grid point
        self.coeffs = np.polyfit(in_global, out_local, deg=self.deg) # (2, n_lat*n_lon)

        # Reshape coefficients onto locally-resolved grid
        self.coeffs = self.coeffs.reshape(-1, n_lat, n_lon) # (2, n_lat, n_lon)

    def predict(self, in_global):
        """
        Args:
            in_global np.array((n_t,))
        Returns:
            preds np.array((n_t, n_lat, n_lon))
        """
        n_lat = self.coeffs.shape[1]
        n_lon = self.coeffs.shape[2]

        # Predict by applying pattern scaling coefficients on locally-resolved grid
        in_global = np.tile(in_global[:,np.newaxis, np.newaxis], reps=(1,n_lat,n_lon)) # repeat onto local grid to get shape (n_t, n_lat, n_lon)
        preds = np.polyval(self.coeffs, in_global) # (n_t, n_lat, n_lon)

        return preds

    def train_multivariate(self, in_global, out_local):
        """
        Fits a linear regression from (multivariate) global 
        variables to local variables
        Args:
            in_global np.array((n_t,)) or np.array((n_t,n_ch)): The 
                model input is a (multivariate) 
                global variable, e.g., global cum. CO2 emissions
                and global methane emissions
            out_local same as self.train()
        """
        if self.deg != 1:
            raise NotImplementedError('Multivariate pattern scaling is only implemented'\
                'for linear fits. Choose deg=1 instead of current deg={self.deg}')
        
        # Add channel dimension if not already in inputs
        if len(in_global.shape) == 1:
            in_global = in_global[:,None]

        # Convert from dask to np
        if not isinstance(in_global, np.ndarray):
            in_global = in_global.compute()
        if not isinstance(out_local, np.ndarray):
            out_local = out_local.compute()

        n_t, n_lat, n_lon = out_local.shape
        _, n_ch = in_global.shape

        # Add intercept
        in_global = np.concatenate([in_global,np.ones((n_t,1))],axis=1) # (n_t, n_ch+1)

        # Flatten data in space
        out_local = out_local.reshape(n_t,-1) # (n_t, n_lat*n_lon)

        # Fit coefficients using least squares
        # (This is the same as polyfit with deg=1 and n_ch=1)
        self.coeffs = np.linalg.lstsq(in_global, out_local, rcond=None)[0]
        
        # Reshape coefficients onto locally-resolved grid
        self.coeffs = self.coeffs.reshape(-1, n_lat, n_lon) # (n_ch+1, n_lat, n_lon)
    
    def predict_multivariate(self, in_global):
        """
        Args:
            in_global: same as self.train_multivariate()
        Returns:
            preds: same as self.predict()
        """
        # Add channel dimension if not already in inputs
        if len(in_global.shape) == 1:
            in_global = in_global[:,None]

        # Convert from dask to np
        if not isinstance(in_global, np.ndarray):
            in_global = in_global.compute()

        # Add intercept
        n_t = in_global.shape[0]
        in_global = np.concatenate([in_global,np.ones((n_t,1))],axis=1) # (n_t, n_ch+1)

        # Calculate the prediction
        _, n_lat, n_lon = self.coeffs.shape
        preds = np.matmul(in_global, self.coeffs.reshape(-1,n_lat*n_lon)) # (n_t, n_lat*n_lon)
        preds = preds.reshape(-1, n_lat, n_lon) # (n_t, n_lat, n_lon)
        
        return preds 

def save(model, 
         dir='./runs/pattern_scaling/default/models/', 
         filename='model.pkl'):
    """
    Saves model at dir
    """
    Path(dir).mkdir(parents=True, exist_ok=True)
    path = dir + filename
    with open(path,'wb') as f:
        pickle.dump(model,f)

def load(dir='./runs/pattern_scaling/default/models/', 
        filename='model.pkl'):
    """
    Loads model from file.
    """
    path = dir + filename
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model

def fit_linear_regression_global_global(
    data_dir='data/interim/global_global/train/', 
    plot=False):
    """
    Fits a linear regression model from globally-averaged GHG
    forcings at t to global average variables at t. E.g.,
    global co2 at t -> global tas at t. Accepts multiple in-/
    and output channels.

    Args:
        data_dir str: Path to directory with input and target data. See 
            emcli.dataset.interim_to_processed.interim_to_global_global for
            data format.
        plot bool: if True, plots training data and fit.
    Returns:
        model sklearn.LinearRegression: linear regression model fit to data.
    """
    # Load processed data
    input = np.load(data_dir + 'input.npy') # (n_samples, in_channels, lat, lon)
    target = np.load(data_dir + 'target.npy')  # (n_samples, out_channels, lat, lon)
    
    # Process data for linear regression
    input_lr = np.reshape(input, input.shape[:2]) # (n_samples, in_channels)
    target_lr = np.reshape(target, target.shape[:2]) # (n_samples, out_channels)

    # Initialize and fit a LinearRegression model
    model = LinearRegression()
    model.fit(input_lr, target_lr)

    if plot:
        fig, axs = plt.subplots(1,1, figsize =(4,4))
        axs.plot(input_lr, target_lr, '.', label='train')
        axs.plot(input_lr, model.predict(input_lr), color='black', label='pred')
        axs.set_xlabel("input")
        axs.set_ylabel("target")
        axs.set_title("Linear Regression on train data")
        axs.legend()

    return model

def predict_linear_regression_global_global(model, 
    data_dir='data/interim/global_global/test/',
    plot=False):
    """
    Predict linear regression model on global-averages variables
    to global average variables.
    Args:
        model sklearn.LinearRegression model
        data_dir str: Path to directory with input and target data. See 
            emcli.dataset.interim_to_processed.interim_to_global_global for
            data format.
        plot bool: if True, plots test data and fit.
    Returns
        preds np.array(n_samples,out_channels): predictions
    """
    # Load processed data
    input_test = np.load(data_dir + 'input.npy') # (n_samples, in_channels, lat, lon)
    target_test = np.load(data_dir + 'target.npy')  # (n_samples, out_channels, lat, lon)

    # Process data for linear regression
    input_test_lr = np.reshape(input_test, input_test.shape[:2]) # (n_samples, in_channels)
    target_test_lr = np.reshape(target_test, target_test.shape[:2]) # (n_samples, out_channels)
    
    # Predict linear regression model on test data
    preds = model.predict(input_test_lr)
    
    if plot:
        fig, axs = plt.subplots(1,1, figsize =(4,4))
        axs.plot(input_test_lr, target_test_lr, '.', label='true')
        axs.plot(input_test_lr, preds, color='black', label='pred')
        axs.set_xlabel("input")
        axs.set_ylabel("target")
        axs.set_title("Linear regression on test data")
        axs.legend()

    return preds

def get_args():
    parser = argparse.ArgumentParser(description='Fit linear pattern scaling model')
    parser.add_argument('--data_var', type=str, default=None, help='Data variable that fit')
    parser.add_argument('--cfg_path', type=str, default='runs/pattern_scaling/mpi-esm1-2-lr/default/config/config.yaml', help='Path to config yaml')
    parser.add_argument('--train_m_member_subsets', action='store_true', default=False, help=
                        'Fit the model on the mean of randomly picked subsets of realizations from '\
                        'the target climate model.')
    parser.add_argument('--debug', action='store_true', default=False, help='debug')
    parser.add_argument('--task_id', type=int, default=1, help='SLURM task id, when script is called in job array')
    parser.add_argument('--num_tasks', type=int, default=1,
                        help='Total number of SLURM tasks when script is called in job array')
    parser.add_argument('--sweep', action='store_true', default=False,
                        help='If true, indicates that program is running a hyperparameter sweep')
    parser.add_argument('--overwrite_m_member_subsets_csv', action='store_true', default=False, help='If true, will overwrite m_member_subsets.csv file that contains '\
                         'a list of realization IDs for the random member subsets')
    return parser.parse_args()

import logging
from emcli2.utils.utils import init_sweep_config
import glob

if __name__ == '__main__':
    # Get command line arguments
    args = get_args()
    if args.data_var is None:
        data_vars = ['pr', 'tas'] # , 'vas','dtr','uas',  'pr90', 'huss', 'psl'] # 'pr90', 'dtr'
    else:
        data_vars = [args.data_var]

    # Initialize logging
    logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')

    # Import cfg
    cfg = yaml.safe_load(open(args.cfg_path, 'r'))
    cfg['path_cfg'] = args.cfg_path

    # Initialize hyperparameter sweep
    if args.sweep:
        cfg = init_sweep_config(cfg, cfg['path_sweep_cfg'], args.task_id, args.num_tasks)
    
        # Skip random seeds that have already been run
        rm_vars = []
        for data_var in data_vars:
            wildcard_checkpoints = f'{cfg["path_experiment"]}/{data_var}/memberseed-{cfg["memberseed"]}/member_subset*/checkpoints/'
            glob_checkpoints = glob.glob(wildcard_checkpoints)
            if len(glob_checkpoints) == cfg['max_number_of_members_per_subset']:
                print(f'WARNING Sweep for {data_var}/memberseed-{cfg["memberseed"]} already completed. Skipping.')
                rm_vars.append(data_var)
        for var in rm_vars:
            data_vars.remove(var)
        if not data_vars:
            print('All data_vars already swept. Exiting program.')
            exit()

    # Set seeds
    random.seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    os.environ['PYTHONHASHSEED'] = str(cfg['seed'])

    if args.train_m_member_subsets:
        filename_wildcard = 'ensemble.nc'
    else:
        filename_wildcard = 'ensemble_summaries_yr.nc'

    scenarios_aux = [] # ['ssp119'] # auxiliary scenarios that are not used for train or test, e.g., 1pctCO2, picontrol, hist-aer, hist-GHG, abrupt-4xCO2
    scenarios_test = ['ssp245']
    scenarios_train = ['historical', 'ssp126', 'ssp370','ssp585'] # 1pctCO2, picontrol, hist-aer, hist-GHG, abrupt-4xCO2
    scenarios = scenarios_train + scenarios_test
    if args.debug:
        scenarios_train = ['historical']

    meta = get_meta(cfg['data_path'], cfg['data_path_interim'])

    # Add 'tas' to filepath list, because LPS model always needs it as global intermediate variable
    data_vars_fp = data_vars if 'tas' in data_vars else data_vars + ['tas'] 
    frequencies = [meta[data_var]['frequency'] for data_var in data_vars_fp]

    # Get ClimateBench input4mips emission variables
    X_train, _ = load_climatebench_data(
        simus=scenarios_train, len_historical=0, 
        data_path=cfg['data_path_climatebench'])

    X_test, _ = load_climatebench_data(
        simus=scenarios_test, len_historical=165,
        data_path=cfg['data_path_climatebench'])
    
    df = get_filepaths_mpi_esm1_2_lr(data_vars=data_vars_fp,
        scenarios=scenarios,
        data_path=cfg['data_path_interim'],
        filename_wildcard=filename_wildcard,
        frequencies=frequencies,
        verbose=(not args.sweep))

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
                replace=(cfg['replace'] if 'replace' in cfg else None),
                seed=cfg['memberseed']
            )
        else:
            m_member_subsets = [None]
            all_member_ids = [None]
            idcs_member_subsets = [None]

        climatology = None
        climatology_tas = None
        incumbent = dict()

        for s_idx, m_member_subset in zip(idcs_member_subsets, m_member_subsets):
            cfg['path_checkpoints'] = f'{cfg["path_experiment"]}/{data_var}/memberseed-{cfg["memberseed"]}/member_subset{s_idx}/checkpoints/'
            path_incumbent = f"{cfg['path_experiment']}{data_var}/memberseed-{cfg['memberseed']}/incumbents/"

            logging.info('#################')
            logging.info(f'Fitting to subset idx={s_idx} with m={len(m_member_subset)} members and memberseed-{cfg["memberseed"]}:')
            logging.info(m_member_subset)
            logging.info(f'scenarios_test, data_var: {scenarios_test}, {data_var}')
            logging.info('#################')

            ###
            # Fit global model
            ###
            # Load global tas data for LPS model
            cfg['open_data_parallel'] = cfg['open_data_parallel'] if 'open_data_parallel' in cfg else True
            datasets_tas, climatology_tas = load_mpi_data_as_xr(df=df[df.data_var == 'tas'], meta=meta,
                                climatology=climatology_tas, 
                                m_member_subset=m_member_subset,
                                open_data_parallel=cfg['open_data_parallel'])

            Y_train_tas, Y_test_tas, Y_aux_tas = create_train_splits(datasets_tas, data_var='tas', 
                                                        scenarios_train=scenarios_train, 
                                                        scenarios_test=scenarios_test, 
                                                        scenarios_aux=scenarios_aux,
                                                        verbose=(s_idx==0))
            logging.debug('convert interim to global')

            # Convert training data from interim to processed
            # data_var = 'tas' # Other variables can be plotted, but our Linear Pattern Scaling should always use global 'tas' as intermediate. 'tas', 'pr', 'pr90', 'diurnal_temperature_range'
            dir_global_global = cfg['data_root'] + 'processed/global_global/'
            input_train, target_train = i2p.interim_to_global_global(
                                        X_global_local=X_train,
                                        Y_global_local=Y_train_tas,
                                        input_keys=['CO2'],
                                        target_keys=['tas'],
                                        save_dir=dir_global_global+'train/',
                                        verbose=(s_idx==0))

            logging.debug('fit_linear_regression_global_global')
            # Fit global linear regression model from global ghg at t -> global data_var as t
            model_lr = ps.fit_linear_regression_global_global(data_dir=dir_global_global+'train/', plot=False)

            # Save global model
            model_name = f'global_co2_to_global_{data_var}.pkl'
            ps.save(model_lr, dir=cfg['path_checkpoints'], filename=model_name)

            # Convert test data from interim to processed
            input_test, target_test = i2p.interim_to_global_global(X_global_local=X_test, # [X_train[2]],# 
                                        Y_global_local=Y_test_tas, # [Y_train[2]],#
                                        input_keys=['CO2'],
                                        target_keys=['tas'],
                                        save_dir=dir_global_global+'test/')
            # target_test is not used unless the target global temperatures should
            # be used as inputs for pattern scaling
            del target_test

            # Apply global linear regression model on test data
            # model_lr = ps.load(dir=cfg['path_checkpoints'], filename=model_name)
            preds_lr = ps.predict_linear_regression_global_global(model_lr, 
                        data_dir=dir_global_global+'test/',
                        plot=False)
            
            for data in Y_test_tas + Y_train_tas + Y_aux_tas:
                data.close(); del data
            del Y_test_tas; del Y_train_tas; del Y_aux_tas

            ###
            # Fit local model
            ###
            datasets, climatology = load_mpi_data_as_xr(df=df_data_var, meta=meta,
                                climatology=climatology, 
                                m_member_subset=m_member_subset,
                                open_data_parallel=cfg['open_data_parallel'])

            Y_train, Y_test, Y_aux = create_train_splits(datasets, data_var=data_var, 
                                                        scenarios_train=scenarios_train, 
                                                        scenarios_test=scenarios_test, 
                                                        scenarios_aux=scenarios_aux,
                                                        verbose=(s_idx==0))
            coords = Y_test[0].coords

            if 'eval_on_all_member_mean' in cfg:
                if cfg['eval_on_all_member_mean']:
                    # Replace the test dataset with the mean of all realizations
                    [yt.close() for yt in Y_test];
                    del Y_test
                    gc.collect()
                    df_test = pd.concat((df_data_var[df_data_var.scenario == scenario_test] for scenario_test in (['historical'] + scenarios_test)))
                    datasets_test, climatology = load_mpi_data_as_xr(df=df_test, meta=meta,
                                climatology=climatology, 
                                m_member_subset=all_member_ids,
                                open_data_parallel=cfg['open_data_parallel'])
                    _, Y_test, _ = create_train_splits(datasets_test, data_var=data_var, 
                                                        scenarios_train=[], 
                                                        scenarios_test=scenarios_test, 
                                                        scenarios_aux=[],
                                                        verbose=(s_idx==0))

            # Retrieve global temperatures for training
            var_global = target_train.flatten() # (n_time,)
            # Retrieve annual local temperature field for training
            var_local = np.concatenate([dataset[data_var].data for dataset in Y_train],axis=0) # (n_time, n_lat, n_lon)

            # Initialize and fit pattern scaling model
            pattern_scaling = ps.PatternScaling(deg=1)
            pattern_scaling.train(var_global, var_local)

            # Save model
            model_name = f'global_tas_to_local_{data_var}.pkl'
            ps.save(pattern_scaling, dir=cfg['path_checkpoints'],filename=model_name)

            # Retrieve test data. Use global tas predictions from previous model as input
            var_global_test = preds_lr.flatten() # (n_time,)
            # var_global_test = target_test.flatten() # (n_time,) # uncomment to use ground-truth global tas as input

            # Load model
            # pattern_scaling = ps.load(dir=cfg['path_checkpoints'])

            # Apply model on test data
            preds_pattern_scaling = pattern_scaling.predict(var_global_test) # (n_time, n_lat, n_lon)

            # Reshape to xarray
            preds_pattern_scaling_xr = xr.DataArray(data=preds_pattern_scaling, 
                coords=Y_test[0][data_var].coords, name=data_var) # convert predictions into axarray
            preds_model_ds = [xr.merge([preds_pattern_scaling_xr])]

            # Calculate loss on test set
            Y_rmse_spatial = calculate_nrmse(Y_true=Y_test[0][data_var].compute(), Y_pred=preds_model_ds[0][data_var], normalize=False)
            Y_rmse_global = rmse(calculate_global_weighted_average(Y_test[0][data_var].sel(time=slice('2080',None))), 
                                calculate_global_weighted_average(preds_model_ds[0][data_var].sel(time=slice('2080',None))))
            Y_rmse_global_np = Y_rmse_global.values
            Y_rmse_global.close(); del Y_rmse_global
            logging.info(f'Spatial RMSE {Y_rmse_spatial}')
            logging.info(f'Global RMSE {Y_rmse_global_np}')

            incumbent[s_idx] = {
                'data_var': data_var,
                'Y_rmse_spatial': Y_rmse_spatial.item(),
                'Y_rmse_global_np': Y_rmse_global_np.item(),
                'best_val_loss': None,
                'm_member_subset': m_member_subset,
                'best_checkpoint_path': cfg['path_checkpoints']
            }
            incumbent[s_idx].update(cfg)
            Path(path_incumbent).mkdir(parents=True, exist_ok=True)
            with open(f'{path_incumbent}/incumbents.yaml', 'w') as file:
                yaml.dump(incumbent, file)

            for data in Y_test + Y_train + Y_aux:
                data.close(); del data
            del Y_test; del Y_train; del Y_aux
            gc.collect()

        logging.info(f'Finished test for {data_var}')
        [logging.info(f'Member subset {key} with len {len(incumbent[key]["m_member_subset"])} has \n\tRMSE_spatial {incumbent[key]["Y_rmse_spatial"]} and \n\tRMSE_global {incumbent[key]["Y_rmse_global_np"]}') for key in incumbent.keys()]

        logging.info('Throwing error to exit and not go into next data_var')
        import sys;sys.exit()