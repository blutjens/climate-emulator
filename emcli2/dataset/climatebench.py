import xarray as xr
import numpy as np

DATA_VAR_LABELS = {
    'tas': {
        'ylabel': 'Annual Global Surface \n Temperate Anomaly in °C',
        'title': 'Surface Temperature, \'tas\'',
    },
    'pr': {
        'ylabel': 'Annual Mean Precipitation in mm/day',
        'title': 'Precipitation, \'pr\'',
    }
}

# Functions for reshaping the data 
def input_for_training(X_train_xr, skip_historical=False, len_historical=None, slider=10):
    """
    Src: climatebench github
    """
    X_train_np =  X_train_xr.to_array().transpose('time', 'latitude', 'longitude', 'variable').data

    time_length = X_train_np.shape[0]

    # If we skip historical data, the first sequence created has as last element the first scenario data point
    if skip_historical:
        X_train_to_return = np.array([X_train_np[i:i+slider] for i in range(len_historical-slider+1, time_length-slider+1)])
    # Else we just go through the whole dataset historical + scenario (does not matter in the case of 'hist-GHG' and 'hist_aer')
    else:
        X_train_to_return = np.array([X_train_np[i:i+slider] for i in range(0, time_length-slider+1)])
    
    return X_train_to_return 

def output_for_training(Y_train_xr, var, skip_historical=False, len_historical=None, slider=10): 
    """
    Src: climatebench github
    """
    Y_train_np = Y_train_xr[var].data
    
    time_length = Y_train_np.shape[0]
    
    # If we skip historical data, the first sequence created has as target element the first scenario data point
    if skip_historical:
        Y_train_to_return = np.array([[Y_train_np[i+slider-1]] for i in range(len_historical-slider+1, time_length-slider+1)])
    # Else we just go through the whole dataset historical + scenario (does not matter in the case of 'hist-GHG' and 'hist_aer')
    else:
        Y_train_to_return = np.array([[Y_train_np[i+slider-1]] for i in range(0, time_length-slider+1)])
    
    return Y_train_to_return

# Climatebench utilities for normalizing the input data
# Code adapted from: https://github.com/duncanwp/ClimateBench/
# blob/main/baseline_models/CNN-LTSM_model.ipynb
def normalize(data, var, meanstd):
    """
    # to-do: move to dataloader.py
    #        and use torch
    Args:
        data
        var string: dictionary key to data variable, e.g., 'CO2'
        meanstd: dictionary of means and standard deviations
            of each var. E.g., meanstd['CO2'] is a tuple of (mean, std)
    """
    mean = meanstd[var][0]
    std = meanstd[var][1]
    return (data - mean)/std

def unnormalize(data, var, meanstd):
    mean = meanstd[var][0]
    std = meanstd[var][1]
    return data * std + mean

def normalize_climatebench_inputs(X_train, meanstd_inputs=None, verbose=False):
    """
    Normalizes input data from climatebench
    Args:
        X_train len(scenarios)*xr.Dataset
        meanstd_inputs dict(input_var: (np.float64, np.float64)): Mean 
            and standard deviation of each data variable
    Returns:
        X_train_norm: same as X_train, but normalized
        meanstd_inputs
    """
    # Normalize input data:
    if meanstd_inputs is None:
        # Compute mean/std of each variable for the whole dataset
        meanstd_inputs = {}
        for input_var in X_train[0].data_vars:
            # To not take the historical data into account several time we have to slice the scenario datasets
            # and only keep the historical data once (in the first ssp index 0 in the simus list)
            array = np.concatenate([X_train[i][input_var].data for i in range(len(X_train))])
            meanstd_inputs[input_var] = (array.mean(), array.std())
    # Print data statistics
    if verbose:
        print('Data statistics before normalization:')
        for input_var in meanstd_inputs.keys():
            print(f'{input_var}:\t mean: {meanstd_inputs[input_var][0]} \t std:  {meanstd_inputs[input_var][0]}')
    # normalize input data
    X_train_norm = []
    for i, train_xr in enumerate(X_train):
        for input_var in X_train[0].data_vars:
            var_dims = train_xr[input_var].dims
            train_xr = train_xr.assign({input_var: 
                                        (var_dims, normalize(train_xr[input_var].data, input_var, meanstd_inputs))})
        X_train_norm.append(train_xr)
    # todo: im not sure if this is needed.
    for data in X_train:
        data.close(); del data
    return X_train_norm, meanstd_inputs

def compute_mean_std_X_train(data_arr, var='CO2'):
    """
    Compute mean and standard deviaation across a data_arr that is list
    of xarray.DataArray for [ssp126, ssp370, ssp585, hist-aer and hist-ghg]. The
    magic numbers in the code are indices from [0,...,4] for [ssp126,..., hist-ghg].
    The data_arr is concatenated in a way that the overlapping historical periods
    are removed before calculating mean and standard deviation.
    Returns
        mean np.array(1)
        std np.array(1)
    """
    # To not take the historical data into account several time we have to slice the scenario datasets
    # and only keep the historical data once (in the first ssp index 0 in the simus list)
    len_historical = len(data_arr[3].time)
    data_arr_concat = np.concatenate(
        [data_arr[i][var].data for i in [0, # ssp126 historical + future data 
                                        3, 4]] + # modified scenarios over historical time-period
        [data_arr[i][var].sel(time=slice(len_historical, None)).data for i in 
         range(1, 3)]) # future points from ssp370, ssp585
    mean = data_arr_concat.mean()
    std = data_arr_concat.std()

    return mean, std

def normalize_data_arr(data_arr, meanstd, keys=['CO2', 'CH4', 'SO2', 'BC']):
    """ normalizes all variables in the data_arr. The data_arr is a list
    of xarray.DataArray that each can contain multiple keys. The keys
    can have varying dimensions, e.g., local or global.
    Args:
        meanstd: dictionary of means and standard deviations of each var. E.g., meanstd['CO2'] is a tuple of (mean, std)
    Returns:
        data_arr_norm 
    """
    data_arr_norm = [] 
    for i, data in enumerate(data_arr): 
        for key in keys: 
            # Get dimensions of each variable, e.g., 'time' for global variables and ['time', 'lat', 'lon'] for local variables
            dims = data[key].dims
            # Apply the normalization function across every variable via xarray's assign
            data=data.assign({key: (dims, normalize(data[key].data, key, meanstd))}) 
        data_arr_norm.append(data)
    return data_arr_norm

from emcli2.dataset.climatebench import input_for_training
def load_climatebench_inputs_as_np_time_series(scenarios,
                                  data_path,
                                  meanstd_inputs=None,
                                  split='train',
                                  slider=10,
                                  normalize=True,
                                  verbose=True):
    """
    Load Climatebench data and returns it as np.ndarray time-series 

    Returns:
        X_all np.ndarray with shape (n_samples, time, lats, lons, channels)
    """
    if split == 'train':
        len_historical = 0
    elif split == 'val' or split == 'test':
        len_historical = 165
    X_data, _ = load_climatebench_data(
        simus=scenarios, 
        len_historical=len_historical,
        data_path=data_path)
    if normalize:
        X_norm, meanstd_inputs = normalize_climatebench_inputs(X_data, 
                                                               verbose=verbose, 
                                                               meanstd_inputs=meanstd_inputs)
        for data in X_data:
            data.close(); del data
    else:
        X_norm = X_data
        meanstd_inputs = None
    # ## Reshape data to feed into the model CNN - LSTM architecture
    if split == 'train':
        X_all = np.concatenate([input_for_training(X_norm[i], slider=slider) for i in range(len(scenarios))], axis = 0)
    elif split == 'val' or split == 'test':
        X_all = input_for_training(X_norm[0], skip_historical=True, len_historical=165, slider=slider)
    # Close data
    for data in X_norm:
        data.close(); del data
    return X_all, meanstd_inputs

def load_climatebench_data(
    simus = ['ssp126','ssp370','ssp585','hist-GHG','hist-aer'],
    len_historical = 165,
    data_path = 'data/',
    avg_over_ensemble = True):
    """ Load ClimateBench Training Data
    Loads the scenario passed in <simus> from data_path into memory. 
    The data from scenarios, e.g., ssp126 is concatenated with historical 
    data.
    Args:
        avg_over_ensemble bool: If True, will return average over ensemble
            members in dataset.
    Returns:
        inputs list(xarray.Dataset[len_historical (+ len_future), lon, lat]) 
            with CO2, SO2, CH4, BC in float64
        targets list(xarray.Dataset[len_historical (+ len_future), lon, lat]) 
            with diurnal_temperature_range & tas in float32 and pr & pr90 in float64
    """
    # to-do: load only co2 and temp
    #        make possible to load rest later on
    # do lazy loading to highlight benefit of netCDF
    inputs = []
    targets = []

    for i, simu in enumerate(simus):

        input_name = 'inputs_' + simu + '.nc'
        output_name = 'outputs_' + simu + '.nc'

        if 'hist' in simu:
            # load inputs
            input_xr = xr.open_dataset(data_path + input_name)

            # load outputs
            output_xr = xr.open_dataset(data_path + output_name)

        # Concatenate with historical data in the case of scenario 'ssp126', 'ssp370' and 'ssp585'
        else:
            # load inputs
            input_paths = [data_path + input_name] # future data, e.g., (86,)
            output_paths = [data_path + output_name]
            if len_historical == 165:
                # add ground-truth historical data, e.g., (165,))
                input_paths.insert(0, data_path + 'inputs_historical.nc')
                output_paths.insert(0, data_path + 'outputs_historical.nc')
            elif len_historical != 0:
                raise ValueError("len_historical must be 0 or 165")

            # Load in- and outputs
            input_xr = xr.open_mfdataset(input_paths).compute() 
            output_xr = xr.open_mfdataset(output_paths)

        # Process precipitations
        output_xr = output_xr.assign({"pr": output_xr.pr * 86400,
                                    "pr90": output_xr.pr90 * 86400})
        output_xr = output_xr.drop(['quantile'])
        # Process dimensions
        output_xr = output_xr.rename({'lon':'longitude', 'lat': 'latitude'})
        if avg_over_ensemble:
            # Average over ensemble members
            output_xr = output_xr.mean(dim='member')
            output_xr = output_xr.transpose('time','latitude', 'longitude')
        else:
            output_xr = output_xr.transpose('member', 'time','latitude', 'longitude')
        # output_xr = output_xr.compute() # load into memory

        print(input_xr.dims, simu)

        # Assign attributes
        input_xr.attrs['experiment_id'] = simu

        # Append to list
        inputs.append(input_xr)
        targets.append(output_xr)

    return inputs, targets
