import numpy as np
import pandas as pd
import xarray as xr
from xskillscore import rmse
from xskillscore import r2

import os
os.environ["KERAS_BACKEND"] = "torch"
import torch
from keras.losses import Loss
from keras.losses import MeanSquaredError

from emcli2.dataset.interim_to_processed import calculate_global_weighted_average

def calculate_end_of_century_err(metric, Y_true, Y_pred, avg_only_over_last_20_yrs=True, skipna=False):
    """
    Args:
        metric function: Metric function from xskillscore, e.g., metric = r2
        Y_true xr.DataArray(lat, lon, time)
        Y_pred xr.DataArray(lat, lon, time)
        avg_only_over_last_20_yrs bool: If True, take temporal average 
            only over last 20 years
    """
    lats = Y_true.latitude
    lons = Y_true.longitude

    # Take temporal average over test period
    if avg_only_over_last_20_yrs:
      try:
          Y_true_mean = Y_true.sel(time=slice("2080",None)).mean('time') 
          Y_pred_mean = Y_pred.sel(time=slice("2080",None)).mean('time')
      except:
          # Using the cftime._cftime.DatetimeNoLeap format
          Y_true_mean = Y_true.sel(time=Y_true.time.dt.year.isin(range(2080,2101))).mean('time') 
          Y_pred_mean = Y_pred.sel(time=Y_pred.time.dt.year.isin(range(2080,2101))).mean('time')
    else:
        Y_true_mean = Y_true.mean('time') 
        Y_pred_mean = Y_pred.mean('time')

    # Take spatial average and metric
    weights = np.cos(np.deg2rad(lats)).expand_dims(longitude=len(lons)).assign_coords(longitude=lons)
    Y_metric = metric(Y_true_mean, Y_pred_mean, weights=weights,skipna=skipna).data

    return Y_metric


def calculate_nrmse(Y_true, Y_pred, normalize=False, avg_only_over_last_20_yrs=True):
    """
    Args:
        Y_true xr.DataArray(lat, lon, time)
        Y_pred xr.DataArray(lat, lon, time)
        avg_only_over_last_20_yrs bool: If True, take temporal average 
            only over last 20 years
    """
    lats = Y_true.latitude
    lons = Y_true.longitude

    # Take temporal average over test period
    if avg_only_over_last_20_yrs:
      try:
          Y_true_mean = Y_true.sel(time=slice("2080",None)).mean('time') 
          Y_pred_mean = Y_pred.sel(time=slice("2080",None)).mean('time')
      except:
          # Using the cftime._cftime.DatetimeNoLeap format
          Y_true_mean = Y_true.sel(time=Y_true.time.dt.year.isin(range(2080,2101))).mean('time') 
          Y_pred_mean = Y_pred.sel(time=Y_pred.time.dt.year.isin(range(2080,2101))).mean('time')
    else:
        Y_true_mean = Y_true.mean('time')
        Y_pred_mean = Y_pred.mean('time')

    # Take spatial average and rmse
    weights = np.cos(np.deg2rad(lats)).expand_dims(longitude=len(lons)).assign_coords(longitude=lons)
    Y_rmse = rmse(Y_true_mean, Y_pred_mean, weights=weights).data

    # Normalize
    if normalize:
        Y_true_abs = np.abs(calculate_global_weighted_average(Y_true_mean)).data
        Y_nrmse = Y_rmse / Y_true_abs
    else:
        Y_nrmse = Y_rmse

    # np.abs(global_mean(Y_true_mean)data)
    return Y_nrmse

class LatWeightedMeanSquaredError(Loss):
    def __init__(self,
                lats=None,
                lons=None,
                reduction="sum_over_batch_size",
                name="lat_weighted_mean_squared_error",
                device='cpu'):
        super().__init__(reduction=reduction, name=name)
        
        self.reduction = reduction
        self.MSE = MeanSquaredError(reduction = 'none')
        if lats is not None and lons is not None:
            weights = np.cos(np.deg2rad(lats)).expand_dims(longitude=len(lons)).assign_coords(longitude=lons).transpose('latitude', 'longitude')
            weights = weights / weights.sum()
            weights_np = weights.to_numpy()
            self.weights_torch = torch.from_numpy(weights_np).to(device=device)
            weights.close(); del weights
        else:
            self.weights_torch = torch.ones(1, device=device)
        self.name = name

    def __call__(self, y_true, y_pred, placeholder=None):
        # I verified that the above code applies weights correctly with the below snippet:
        # weights_np_repeats = np.repeat(weights_np[None,None,...], repeats=Y_val_all.shape[0], axis=0)
        # weighted_mse_repeats = mse * weights_np_repeats
        # assert torch.all(weighted_mse == weighted_mse_repeats)
        mse = self.MSE(y_true[...,None], y_pred[...,None])
        mse = mse * self.weights_torch
        mse = mse.sum(axis=np.arange(len(mse.shape))[1:].tolist()) # reduce over time, lat, lon
        if self.reduction == "sum_over_batch_size":
            mse = mse.sum()
        elif self.reduction == "avg_over_batch_size":
            mse = mse.mean()
        return mse

    def get_config(self):
        return Loss.get_config(self)

def compute_metrics(variables, model_labels, models,
    data_path: str='~/climate-emulator-tutorial/data/raw/climatebench/',
    Y=None
    ):
    """
    Src: ClimateBenchv1.0

    Args:
        models n_models*[xr.DataArray] : List of model predictions
        data_path: str, path to ground-truth target data
        Y: xr.Dataset : Ground-truth target data. If None, data is loaded from data_path
    """
    if Y is None:
        Y = xr.open_dataset(data_path + 'outputs_ssp245.nc')
        Y = Y.rename({'lon':'longitude', 'lat': 'latitude'})
        # Convert the precip values to mm/day
        Y["pr"] *= 86400
        Y["pr90"] *= 86400

    weights = np.cos(np.deg2rad(Y[variables[0]].latitude)).expand_dims(longitude=len(Y.longitude)).assign_coords(lon=Y.longitude)

    def global_mean(ds):
        weights = np.cos(np.deg2rad(ds.latitude))
        return ds.weighted(weights).mean(['latitude', 'longitude'])

    # Spatial NRMSE
    NRMSE = pd.DataFrame({
        label: {variable: rmse(Y.mean('member')[variable].sel(time=slice("2080", None)).mean('time'), 
                                model[variable].sel(time=slice("2080", None)).mean('time'), weights=weights).data/ np.abs(global_mean(Y.mean('member')[variable].sel(time=slice("2080", None)).mean('time')).data) for variable in variables} 
        for label, model in zip(model_labels, models)
    })
    NRMSE.T.round(3).style.highlight_min(subset=slice("Random Forest", None), axis = 0, props='font-weight: bold').format("{:.4f}")
    print(f'spatial NRMSE, {NRMSE.keys()[0]} {NRMSE.index[0]}: {NRMSE[NRMSE.keys()[0]][NRMSE.index[0]]}')

    # Global NRMSE
    R2E = pd.DataFrame({
        label: {variable: rmse( global_mean(Y.mean('member')[variable].sel(time=slice("2080", None))), 
                                    global_mean(model[variable].sel(time=slice("2080", None)))).data/ np.abs(global_mean(Y.mean('member')[variable].sel(time=slice("2080", None)).mean('time')).data) for variable in variables} 
    #                                 global_mean(model[variable].sel(time=slice(2080, None)))).data for variable in variables} 
                            for label, model in zip(model_labels[:], models[:])
    })
    R2E.T.round(3).style.highlight_min(subset=slice("Random Forest", None), axis = 0, props='font-weight: bold').format("{:.4f}")
    print(f'global NRMSE, {R2E.keys()[0]} {R2E.index[0]}: {R2E[R2E.keys()[0]][R2E.index[0]]}')

    # Total NRMSE
    total_nrmse = (NRMSE+5*R2E)
    total_nrmse.T.round(3).style.highlight_min(subset=slice("Random Forest", None), axis = 0, props='font-weight: bold').format("{:.4f}")
    print(f'total NRMSE, {total_nrmse.keys()[0]} {total_nrmse.index[0]}: {total_nrmse[total_nrmse.keys()[0]][total_nrmse.index[0]]}')

    combined_df = pd.concat([NRMSE, R2E, NRMSE+5*R2E], keys=['Spatial', 'Global', 'Total'])[model_labels[:]].T.swaplevel(axis=1)[variables]
    combined_df.style.highlight_min(subset=slice(None, "Random Forest"), axis = 0, props='font-weight: bold').format("{:.3f}")

    return combined_df