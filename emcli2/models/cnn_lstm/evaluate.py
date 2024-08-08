import argparse
import yaml

import keras
import xarray as xr
import pandas as pd
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from emcli2.utils.metrics import compute_metrics
from emcli2.dataset.mpi_esm1_2_lr import get_meta
from emcli2.dataset.climatebench import load_climatebench_data
from emcli2.dataset.mpi_esm1_2_lr import load_data
from emcli2.utils.plotting import plot_regression_fit

from emcli2.utils.metrics import calculate_end_of_century_err
from xskillscore import rmse
from xskillscore import r2, mape

def get_args():
    parser = argparse.ArgumentParser(description='Evaluate ClimateBench CNN-LSTM model')
    parser.add_argument('--data_var', type=str, default=None, help='Data variable that fit')
    parser.add_argument('--cfg_path', type=str, default='runs/cnn_lstm/mpi-esm1-2-lr/default/config/config.yaml', help='Path to config yaml')
    parser.add_argument('--debug', action='store_true', default=False, help='debug')
    return parser.parse_args()

if __name__ == '__main__':
    # Get command line arguments
    args = get_args()
    if args.data_var is None:
        data_vars = ['uas'] # 'pr', 'tas', 'huss', 'psl', 'uas', 'vas'] # 'pr90', 'dtr'
    else:
        data_vars = [args.data_var]

    # # Load data
    # Import cfg and set seeds
    cfg = yaml.safe_load(open(args.cfg_path, 'r'))
    cfg['path_cfg'] = args.cfg_path
    path_best_checkpoints = {
        'tas': '/home/lutjens/climate-emulator/runs/cnn_lstm/mpi-esm1-2-lr/default/tas/sweep/task-19/checkpoints/model.e55-0.1938.keras',
        'pr':  '/home/lutjens/climate-emulator/runs/cnn_lstm/mpi-esm1-2-lr/default/pr/sweep/task-3/checkpoints/model.e58-0.1720.keras',
        'pr90': '/home/lutjens/climate-emulator/runs/cnn_lstm/mpi-esm1-2-lr/default/pr90/sweep/task-63/checkpoints/model.e57-1.3054.keras',
        'dtr': '/home/lutjens/climate-emulator/runs/cnn_lstm/mpi-esm1-2-lr/default/dtr/sweep/task-44/checkpoints/model.e56-0.0616.keras',
        'vas': '/home/lutjens/climate-emulator/runs/cnn_lstm/mpi-esm1-2-lr/default/vas/sweep/task-26/checkpoints/model.e82-0.0675.keras',
        'uas': '/home/lutjens/climate-emulator/runs/cnn_lstm/mpi-esm1-2-lr/default/uas/sweep/task-56/checkpoints/model.e80-0.1137.keras',
        'psl': '/home/lutjens/climate-emulator/runs/cnn_lstm/mpi-esm1-2-lr/default/psl/sweep/task-87/checkpoints/model.e79-0.4775.keras',
        'huss': '/home/lutjens/climate-emulator/runs/cnn_lstm/mpi-esm1-2-lr/default/huss/sweep/task-77/checkpoints/model.e99-0.0000.keras'
    }

    # Get ClimateBench input4mips emission variables
    scenarios_aux = ['ssp119'] # auxiliary scenarios that are not used for train or test, e.g., 1pctCO2, picontrol, hist-aer, hist-GHG, abrupt-4xCO2
    scenarios_test = ['ssp245']
    scenarios_train = ['historical', 'ssp126','ssp370','ssp585'] # 1pctCO2, picontrol, hist-aer, hist-GHG, abrupt-4xCO2

    meta = get_meta(cfg['data_path'], cfg['data_path_interim'])

    cfg["model_key"] = 'cnn_lstm'

    for data_var in data_vars:
        path_checkpoint = path_best_checkpoints[data_var]

        # The model (that are considered the best) can be loaded as -
        cnn_model = keras.models.load_model(path_checkpoint)

        _, _, X_test_np, Y_test_np, coords, Y_test_local = load_data(scenarios_train=scenarios_train, 
            data_path_climatebench=cfg['data_path_climatebench'],
            scenarios_test=scenarios_test,
            scenarios_aux=scenarios_aux,
            data_var=data_var,
            data_path_interim=cfg['data_path_interim'],
            meta=meta,
            slider=cfg['slider'],
            return_Y_test_xr=True)

        len_historical = 165

        # Make predictions using trained model 
        m_pred = cnn_model.predict(X_test_np)

        # reshape to xarray
        m_pred = m_pred.reshape(m_pred.shape[0], m_pred.shape[2], m_pred.shape[3])
        # m_pred = xr.DataArray(m_pred, dims=['time', 'lat', 'lon'], coords=[coords['time'].data[cfg['slider'] -1:], coords['latitude'].data, coords['longitude'].data])
        m_pred = xr.DataArray(m_pred, dims=['time', 'lat', 'lon'], coords=[coords['time'].data[len_historical:], coords['latitude'].data, coords['longitude'].data])
        m_pred = m_pred.transpose('lat', 'lon', 'time').sel(time=slice('2015', '2101')).to_dataset(name=data_var)
        # if ((data_var == "pr90") | (data_var == "pr")):
        #     m_pred = m_pred.assign({data_var: m_pred[data_var] / 86400})

        preds_model_ds = [m_pred]
        preds_model_ds[0] = preds_model_ds[0].rename({'lat': 'latitude', 'lon': 'longitude'})

                # Save test predictions as .nc 
                #if data_var == 'diurnal_temperature_range':
                #    xr_prediction.to_netcdf(cfg['data_path'] + 'outputs_ssp245_predict_dtr.nc', 'w')
                #else:
                #    xr_prediction.to_netcdf(cfg['data_path'] + 'outputs_ssp245_predict_{}.nc'.format(data_var), 'w')

        plot_regression_fit(y_true=Y_test_local[0][data_var], 
                            y_pred=preds_model_ds[0][data_var],
                            data_var=data_var, meta=meta, 
                            scenario=scenarios_test[0],
                            filepath_to_save=f'{cfg["repo_root"]}/docs/figures/mpi-esm1-2-lr/{data_var}/{cfg["model_key"]}/ssp245_2080_2100/regplot.png')

        # Plot error map
        from emcli2.utils.plotting import plot_tas_annual_local_err_map 
        axs = plot_tas_annual_local_err_map(Y_test_local[0][data_var], preds_model_ds[0][data_var], data_var=data_var, unit=meta[data_var]['unit'],
            filepath_to_save=f'{cfg["repo_root"]}/docs/figures/mpi-esm1-2-lr/{data_var}/{cfg["model_key"]}/ssp245_2080_2100/preds_map.png')
        axs[1].set_title(f'Piecewise Linear Map: global CO2 -> global tas -> local {data_var} \n trained on 1850-2100. Tested on held-out {scenarios_test[0]}')
        # plt.show()

        Y_r2 = calculate_end_of_century_err(metric=r2, Y_true=Y_test_local[0][data_var].compute(), Y_pred=preds_model_ds[0][data_var])

        # Calculate MAPE (skipping all values that are within n decimal digits of zero)
        Y_mape = calculate_end_of_century_err(metric=mape, 
                    Y_true=Y_test_local[0][data_var].where((abs(Y_test_local[0][data_var]) >= 0.0001)).compute(), 
                    Y_pred=preds_model_ds[0][data_var].where((abs(preds_model_ds[0][data_var]) >= 0.0001)),
                    skipna=True)
        print('R2: ', Y_r2)
        print('MAPE: ', Y_mape)

        # Test calculation of NRMSE
        from emcli2.utils.metrics import calculate_nrmse

        Y_nrmse = calculate_nrmse(Y_true=Y_test_local[0][data_var].compute(), Y_pred=preds_model_ds[0][data_var], normalize=True)
        print(Y_nrmse)

        import pandas as pd
        from xskillscore import rmse

        variables = [data_var]
        model_labels = ['CNN LSTM']
        models = [preds_model_ds[0]]
        Y = Y_test_local[0].expand_dims(dim='member', axis=0).compute()
        metrics_df = compute_metrics(variables, model_labels,models,Y=Y)
        Y.close(); del Y

        # Compute spatial RMSE
        Y_rmse_spatial = calculate_nrmse(Y_true=Y_test_local[0][data_var].compute(), Y_pred=preds_model_ds[0][data_var], normalize=False)
        print('Spatial RMSE', Y_rmse_spatial)

        from emcli2.dataset.interim_to_processed import calculate_global_weighted_average
        import numpy as np

        Y_rmse_global = rmse(calculate_global_weighted_average(Y_test_local[0][data_var].sel(time=slice('2080',None))), 
                            calculate_global_weighted_average(preds_model_ds[0][data_var].sel(time=slice('2080',None))))

        print('Global RMSE', Y_rmse_global.values)
        Y_rmse_global.close(); del Y_rmse_global

        # Plot predictions for 2020-2100
        from pathlib import Path
        from matplotlib import colors

        divnorm = colors.TwoSlopeNorm(vmin=m_pred[data_var].min(), vcenter=0., vmax=m_pred[data_var].max())

        projection = ccrs.Robinson(central_longitude=0)
        transform = ccrs.PlateCarree(central_longitude=0)

        f, axes = plt.subplots(1, 3,
                            subplot_kw=dict(projection=projection),
                            figsize=(9,3),
                            dpi=300)

        print(data_var)
        for i, yr in enumerate([2020, 2050, 2100]): 
            ctr = axes[i].pcolormesh(m_pred.lon, m_pred.lat, m_pred[data_var].sel(time=f'{yr}').squeeze(), cmap="coolwarm", norm=divnorm, transform=transform)
            plt.colorbar(ctr, orientation='horizontal', fraction=0.05)
            axes[i].set_title(f"Predicted {yr}")
            axes[i].coastlines()

        plt.suptitle(f'{cfg["model_key"]} predictions for {data_var} on {scenarios_test[0]}')
        filepath_to_save = f'{cfg["repo_root"]}/docs/figures/mpi-esm1-2-lr/{data_var}/{cfg["model_key"]}/ssp245/preds_2020_50_100.png'
        plt.tight_layout()
        if filepath_to_save is not None:
            Path(filepath_to_save).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(filepath_to_save)

        plt.show()
        plt.close()


        # Save test predictions as .nc 
        # output_name = f'outputs_ssp245_predict_{data_var}.nc'
        # m_pred.to_netcdf(cfg['data_path'] + output_name, 'w')

        m_pred.close(); del m_pred
        for ds in preds_model_ds:
            ds.close(); del ds