
import re
import argparse
import glob
from pathlib import Path
from tqdm import tqdm
from typing import List
import numpy as np
import xarray as xr
import pandas as pd
from dask.diagnostics import ProgressBar
import logging

from emcli2.dataset.interim_to_processed import calculate_global_weighted_average
from emcli2.dataset.climatebench import load_climatebench_data
from emcli2.dataset.climatebench import input_for_training
from emcli2.dataset.climatebench import output_for_training
from emcli2.dataset.climatebench import normalize
from emcli2.dataset.climatebench import load_climatebench_inputs_as_np_time_series
from emcli2.dataset.climatebench import unnormalize

from emcli2.dataset.climatebench import output_for_training
def convert_xr_to_np_timeseries(data_var, slider=10,
                                Y_train=None, scenarios_train=[], 
                                Y_test=None, scenarios_test=[], 
                                ):
    # Convert from xarray to numpy array in cnn-lstm time-series format
    # Process train data
    logging.info('Loading train data into memory')
    if Y_train is not None:
        with ProgressBar():
            Y_train_np = np.concatenate([output_for_training(Y_train[i], var=data_var, slider=slider) for i in range(len(scenarios_train))], axis=0)
        for data in Y_train:
            data.close(); del data
        del Y_train
    else:
        Y_train_np = None
    # Repeat for test data
    logging.info('Loading test data into memory')
    if Y_test is not None:
        Y_test_np = np.concatenate([output_for_training(Y_test[i], var=data_var, skip_historical=True, len_historical=165) for i in range(len(scenarios_test))], axis=0)
        for data in Y_test:
            data.close(); del data
        del Y_test
    else:
        Y_test_np = None
    logging.info('Converted xr to np timeseries')
    return Y_train_np, Y_test_np

def create_train_splits(datasets, 
                        data_var='tas',
                        scenarios_train=['historical'], 
                        scenarios_test=[], 
                        scenarios_aux=[],
                        verbose=False):
    # Create train, val, test splits for MPI data.
    Y_train = []
    Y_test = []
    Y_aux = []
    for i, dataset in enumerate(datasets):
        if dataset.attrs['experiment_id'] in scenarios_train:
            Y_train.append(dataset)
        elif dataset.attrs['experiment_id'] in scenarios_aux:
            Y_aux.append(dataset)
        if dataset.attrs['experiment_id'] in (['historical'] + scenarios_test):
            Y_test.append(dataset)
    Y_test = [xr.concat(Y_test, 'time', combine_attrs='drop_conflicts')] # historical + ssp245
    Y_test[0].attrs['experiment_id'] = 'historical+ssp245'
    if verbose:
        [logging.info(f'Y_train set shape: {yt[data_var].shape}') for yt in Y_train]
        [logging.info(f'Y_test set shape: {yt[data_var].shape}') for yt in Y_test];
        [logging.info(f'Y_aux set shape: {yt[data_var].shape}') for yt in Y_aux];
    return Y_train, Y_test, Y_aux

def load_mpi_data_as_xr(df, meta, climatology=None, m_member_subset=None, verbose=True,
    open_data_parallel=True):
    """
    Loads all MPI output data as postprocessed anomalies that is passed in df

    Args:
        df pd.DataFrame: dataframe with all filepaths that should be loaded
        meta dict() meta information on data attributes
        climatology xr.DataArray: Climatology that is used to calculate anomalies; 
            if None will load and return climatology
        m_member_subset List(str): List with member IDs for the m-member subset 
            that should be loaded. If not None, returned datasets will contain mean over this subset.
            E.g., ['r8i1f1p1', 'r24i1f1p1', 'r3i1f1p1', 'r5i1f1p1', 'r27i1f1p1', 'r26i1f1p1']
    Returns:
        datasets len(scenarios)*[xr.Dataset]
        climatology xr.DataArray
    """
    # Extract selectors from dataframe
    assert len(df.data_var.unique()) == 1, 'data_var must be non-empty and unique'
    data_var = df.data_var.unique()[0]
    scenarios = df.scenario.unique().tolist()

    # Load climatology to calculate anomalies
    if climatology is None:
        climatology = get_baseline_climatology_mpi_esm1_2_lr(data_var=data_var, 
                                data_path=meta[data_var]['data_path_piControl'],
                                scenario='piControl',
                                frequency=meta[data_var]['frequency'],
                                open_data_parallel=open_data_parallel)
        if verbose:
            logging.info('Calculating climatology')
        with ProgressBar():
            climatology.load();

    # Load data of each scenario; postprocess and store in list
    datasets = []
    for scenario in scenarios:
        df_scenario = df[df.scenario == scenario]
        if len(df_scenario.filepath.values.tolist()) == 1:
            dataset_xr = xr.open_dataset(df_scenario.filepath.values.tolist()[0])
        else:
            dataset_xr = xr.open_mfdataset(df_scenario.filepath.values.tolist())

        # Optionally, compute mean over m-member subset
        if m_member_subset is not None:
            # todo: delete this once ensemble.nc are recomputed with member dimension. Im creating dummy coordinates
            if 'member' not in dataset_xr.coords:
                n_realizations = dataset_xr.sizes['member']
                member_ids = [f'r{r:1d}i1f1p1' for r in np.arange(1, n_realizations+1)]
                dataset_xr.coords['member'] = member_ids
            # Compute mean across ensemble subset
            subset = dataset_xr.sel(member=m_member_subset).mean(dim='member', keep_attrs=True)
        else:
            subset = dataset_xr
        dataset_xr.close(); del dataset_xr

        # todo: this preprocessing should have happened in raw -> interim conversion, because otherwise _std has the wrong scale.
        from emcli2.dataset.mpi_esm1_2_lr import preprocess_data_var_mpi_esm1_2_lr
        if (data_var == 'psl' and subset[data_var].attrs['units'] == 'Pa'): # or \
            subset = preprocess_data_var_mpi_esm1_2_lr(subset, data_var)
        elif (data_var == 'pr' and subset[data_var].attrs['units'] == 'kg m-2 s-1'):
            # pr attribute was incorrectly labeled kg m-2 s-1 instead of mm/day
            subset[data_var].attrs['units'] = 'mm/day'

        # Check for nans and fill via linear interpolation
        if np.any(np.isnan(subset[data_var])):
            invalid_times = np.unique(np.where(np.isnan(subset[data_var]))[0])
            logging.info(f'Warning: Found nan value in {data_var}/{scenario}/{m_member_subset} at '\
                  f't={invalid_times}. Filling via linear interpolation.')
            logging.info(f'That\'s year(s): {subset[data_var].time.isel(time=invalid_times).dt.year.values}')
            subset = subset.interpolate_na(dim='time', method='linear')

        # Calculate the anomalies by subtracting the climatology
        subset[data_var] -= climatology

        # Update attributes in dataset
        if m_member_subset is not None:
            subset.attrs['m_member_subset'] = m_member_subset
            subset.attrs['title'] = f"{meta[data_var]['title']}, {len(m_member_subset)}-member mean, in {subset[data_var].attrs['units']}"
        else:
            subset.attrs['title'] = f"{meta[data_var]['title']}, in {subset[data_var].attrs['units']}"
        subset[data_var].attrs['title'] = subset.attrs['title']
        datasets.append(subset)
        subset.close(); del subset

    return datasets, climatology

def load_data(scenarios_train, 
              data_path_climatebench,
              scenarios_test,
              scenarios_aux,
              data_var,
              data_path_interim,
              meta,
              slider,
              return_Y_test_xr=False,
              open_data_parallel=True
              ):
    """
    Loads ClimateBench input4mips inputs and MPI-ESM1.2-LR targets.
    """
    
    scenarios = scenarios_train + scenarios_test + scenarios_aux
    assert 'historical' in scenarios, 'historical scenario must be included in scenarios'

    # Load precomputed ensemble summaries for runtime increase.
    df = get_filepaths_mpi_esm1_2_lr(data_var=data_var,
            scenarios=scenarios,
            data_path=data_path_interim,
            filename_wildcard='ensemble_summaries_yr.nc',
            frequency=meta[data_var]['frequency'],
            verbose=False)

    datasets, _ = load_mpi_data_as_xr(df=df, meta=meta, 
                    climatology=None, m_member_subset=None,
                    open_data_parallel=open_data_parallel)

    Y_train, Y_test, Y_aux = create_train_splits(datasets, data_var=data_var, 
                                                     scenarios_train=scenarios_train, 
                                                     scenarios_test=scenarios_test, 
                                                     scenarios_aux=scenarios_aux)

    coords = Y_test[0].coords

    Y_train_all, Y_test_np = convert_xr_to_np_timeseries(
        data_var=data_var, slider=slider,
        Y_train=Y_train, scenarios_train=scenarios_train,
        Y_test=Y_test, scenarios_test=scenarios_test)

    # Load inputs for train from climatebench
    X_train_all, meanstd_inputs = load_climatebench_inputs_as_np_time_series(
        scenarios=scenarios_train,
        data_path=data_path_climatebench,
        split='train',
        slider=slider,
        normalize=True
    )

    # Load inputs for val/test from climatebench
    X_test_np, _ = load_climatebench_inputs_as_np_time_series(
        scenarios=scenarios_test,
        data_path=data_path_climatebench,
        split='test',
        meanstd_inputs=meanstd_inputs,
        slider=slider,
        normalize=True
    )
    
    logging.info(f'train: {X_train_all.shape}, {Y_train_all.shape}')
    logging.info(f'train: {X_train_all.shape}, {Y_train_all.shape}')
    logging.info(f'val/test: {X_test_np.shape}, {Y_test_np.shape}')

    return X_train_all, Y_train_all, X_test_np, Y_test_np, coords, Y_test

def get_meta(data_path, data_path_interim):
    meta = {
        'tas': {
            'ylabel': 'Annual Global Surface \n Temperate Anomaly in °C',
            'title': 'Surface Temperature Anom., "tas"',
            'unit': '°C',
            'frequency': 'mon',
            'data_path_piControl': data_path,
        },
        'pr': {
            'ylabel': 'Annual Mean Precipitation Anomaly in mm/day',
            'title': 'Precipitation, "pr"',
            'unit': 'mm/day',
            'frequency': 'mon',
            'data_path_piControl': data_path,
        },
        'dtr': {
            'ylabel': 'Annual Diurnal Temperature Range Anomaly in °C',
            'title': 'Diurnal Temperature Range, "dtr"',
            'unit': '°C',
            'frequency': 'day',
            'data_path_piControl': data_path_interim,
        },
        'pr90': {
            'ylabel': 'Annual Mean Extreme Precipitation Anomaly in mm/day',
            'title': 'Extreme Precipitation, "pr90"',
            'unit': 'mm/day',
            'frequency': 'day',
            'data_path_piControl': data_path_interim,
        },
        'uas': {
            'ylabel': 'Annual Mean Eastward \nNear-Surface Wind Anomaly in m/s',
            'title': 'Eastward Wind, "uas"',
            'unit': 'm/s',
            'frequency': 'mon',
            'data_path_piControl': data_path,
        },
        'vas': {
            'ylabel': 'Annual Mean Northward \nNear-Surface Wind Anomaly in m/s',
            'title': 'Northward Wind, "vas"',
            'unit': 'm/s',
            'frequency': 'mon',
            'data_path_piControl': data_path,
        },
        'psl': {
            'ylabel': 'Annual Mean Sea Level \nPressure Anomaly in hPa',
            'title': 'Sea Level Pressure, "psl"',
            'unit': 'hPa',
            'frequency': 'mon',
            'data_path_piControl': data_path,
        },
        'huss': {
            'ylabel': 'Annual Mean Near-Surface (2m)\nSpecific Humidity in kg/kg',
            'title': 'Specific Humidity, "huss"',
            'unit': 'kg/kg',
            'frequency': 'mon',
            'data_path_piControl': data_path,
        },
    }
    return meta

def preprocess_data_var_mpi_esm1_2_lr(data, data_var):
    """
    Preprocesses a data variable. Processes dimension, unit, and resamples from 
     monthly to annual data.
    """
    # Process dimensions
    try:
        data = data.drop_vars(['lat_bnds', 'lon_bnds', 'time_bnds'])
    except:
        pass
    try:
        data = data.drop_vars(['height'])
    except:
        pass
    try:
        data = data.rename({'lon':'longitude', 'lat': 'latitude'})
    except:
        pass

    # Process each data variable
    if data_var == 'tas' and data['tas'].attrs['units'] == 'K':
        # Near-surface temperature
        data['tas'] -= 273.15 # convert from Kelvin to Celsius
        data['tas'].attrs['units'] = '°C'
    elif data_var == 'pr' and data['pr'].attrs['units'] == 'kg m-2 s-1':
        # Precipitation
        data['pr'] *= 86400 # convert from kg m-2 s-1 to mm/day
        data['pr'].attrs['units'] = 'mm/day'
    elif data_var == 'psl' and data['psl'].attrs['units'] == 'Pa':
        # Sea Level Pressure
        data['psl'] /= 100 # convert from Pa to hPa
        data['psl'].attrs['units'] = 'hPa'
    else:
        logging.warning(f'Warning: no postprocessing defined for data variable {data_var}')
        pass # raise NotImplementedError('So far, only tas is implemented.Need to define how other data_var should be processed')

    # Convert from daily or monthly to annual averages; resample accounts by default for the varying number of days per month
    #  store in new dataset because of different time dimension
    data_annual = data[data_var].resample(time='Y').mean('time').to_dataset(promote_attrs=True)
    
    if 'experiment_id' in data.attrs.keys():
        data_annual.attrs['experiment_id'] = data.attrs['experiment_id']
    # data_annual[data_var].attrs['title'] = f'Global Mean Surface Temperature in {data[data_var].attrs["units"]}'

    return data_annual

def open_mpi_esm1_2_lr_ensemble_summaries(data_path, 
                                          data_var='tas', 
                                          scenario='historical',
                                          frequency='mon'):
    # Load data
    realization_id ='r*i1p1f1'
    years_dir = '*'
    # filename = 'CMIP6_MPI-ESM1-2-LR_r10-r30i1p1f1_ssp126_tas_250_km_mon_2015-2100.nc'
    data_load_wildcard = data_path + f'{realization_id}/{scenario}/{data_var}/250_km/{frequency}/{years_dir}/ensemble_summaries_yr.nc'
    logging.info(f'data_load_wildcard, {data_load_wildcard}')
    data_load_path = glob.glob(data_load_wildcard)[0]

    data = xr.open_mfdataset(data_load_path)
    return data

def get_baseline_climatology_mpi_esm1_2_lr(data_var: str='tas', 
                    data_path:str='../data/interim/CMIP5/MPI-GE/',
                    scenario='historical',
                    frequency='mon',
                    verbose=False,
                    open_data_parallel=True):
    """
    Load and compute baseline climatology that is used to calculate anomalies.
        todo: use picontrol to calculate the baseline.
    Returns:
        climatology xr.DataArray: climatology over baseline period
    """
    # Load baseline data
    if scenario == 'historical':
        realization_id ='r1-r50i1p1f1'
        years_dir = '1850-2014'
        # filename = 'CMIP6_MPI-ESM1-2-LR_r10-r30i1p1f1_ssp126_tas_250_km_mon_2015-2100.nc'
        filename = 'ensemble_summaries_yr.nc'
        baseline_slice = slice("1850-01-01", "1900-12-31")
        decode_times = True
    elif scenario == 'piControl':
        realization_id = 'r1*i1p1f1'
        years_dir = '*'
        filename = '*.nc'
        # Limited baseline lengths to 2250, because np.datetime64 is limited to ~450 years and piControl might exceed that.
        baseline_slice = slice("1850-01-01", "2250-12-31")
        # bash: years_dir = '@('+'|'.join([f'{yr}' for yr in np.arange(1850,2251)])+')'
        if 'raw/' in data_path:
            # The raw piControl data is not np.datetime64. The interim piControl data currently has np.datetime64.
            decode_times = False
        else:
            decode_times = True

    data_load_wildcard = data_path + f'{realization_id}/{scenario}/{data_var}/250_km/{frequency}/{years_dir}/{filename}'

    data_load_path = glob.glob(data_load_wildcard)

    print('If next line throws segmentation fault, try changing config.open_data_parallel to False.')
    data = xr.open_mfdataset(data_load_path, concat_dim="time", combine="nested",
                                        data_vars='minimal', coords='minimal', compat='override',
                                        parallel=open_data_parallel, decode_times=decode_times)

    if decode_times == False:
        if data.attrs['frequency'] == 'yr':
            freq = 'Y'
        else:
            # E.g., for tas raw data this might be the case.
            freq = 'ms'
        # Convert from cftime to np.datetime64
        units, reference_date = data.time.attrs['units'].split('since')
        data['time'] = pd.date_range(start=reference_date, periods=data.sizes['time'], freq=freq)
    elif type(data.time.values[0]) == np.int64:
        # All my piControl data currently np.datetime64 and this section is not needed. However,
        #  it will be needed if we'd like to read in > 423 years of piControl data. 
        logging.warning('Warning: date stored as int64. Assuming years to convert to np.datetime64.')
        # Convert years from int64 1850, 1851, ... to np.datetime64 1850-01-01, 1851-01-01, ...
        data['time'] = pd.to_datetime(data['time'], format='%Y')
        data['time'].attrs['units'] = 'years'

    # Process each data variable
    data_annual = preprocess_data_var_mpi_esm1_2_lr(data, data_var)

    # Define the reference period for the baseline (e.g., 1850-1900)
    baseline_period = data_annual[data_var].time.sel(time=baseline_slice)

    # Calculate the climatology (long-term average) for the reference period
    climatology = data_annual[data_var].sel(time=baseline_period).mean(dim="time")

    # Set attributes
    climatology.attrs = data_annual[data_var].attrs
    climatology.attrs['time_start'] = baseline_slice.start
    climatology.attrs['time_stop'] = baseline_slice.stop
    climatology.attrs['parent_experiment_id'] = scenario

    data_annual.close(); del data_annual

    return climatology

def get_filepaths_mpi_esm1_2_lr(data_vars: List[str]=[],
        scenarios: List[str]=['historical'], # , 'rcp26', 'rcp45', 'rcp85', '1pctCO2'],
        data_path: str='../data/raw/CMIP5/MPI-GE/',
        nominal_resolution: str='250_km',
        frequencies: List[str]=[],
        verbose=False,
        filename_wildcard: str='*.nc',
        max_n_realizations: int=None,
        data_var: str='tas',
        frequency: str='mon',
        ):
    """
    Retrieves the filepaths for all files of the MPI-ESM1-2-LR model for the 
    given scenarios and data_var and returns them as a pandas dataframe.

    Args:
        filename_wildcard: Wildcard to constrain filenames that should be fetched.
            E.g., 'ensemble_summaries_yr.nc' for interim data.
        max_n_realizations : If not None, set the number of maximum realizations that are returned
        data_var, frequency: legacy arguments to support non List type arguments
    Returns:
        df pd.Dataframe(): filepaths with columns that described the filepaths, e.g.,
            scenario, realization_id, year, filepath
    """
    filepaths = []

    # Handle legacy arguments
    if not data_vars:
        logging.warning('Using data_var instead of data_vars. Preferred use is to pass data_vars.')
        data_vars = [data_var]
    if not frequencies:
        frequencies = [frequency]

    for data_var, frequency in zip(data_vars, frequencies):
        # Iterate over every scenario
        for scenario in scenarios:
            # Retrieve the list of realizations that are available for this scenario
            realization_ids = []
            realization_id_patterns = ["r*"]
            for realization_id_pattern in realization_id_patterns:
                realization_dir_pattern = f"{realization_id_pattern}i1p1f1"
                realization_wildcard = data_path + rf'{realization_dir_pattern}/{scenario}/{data_var}/{nominal_resolution}/{frequency}'
                if verbose: logging.info(f'Searching for realizations with wildcard: {realization_wildcard}')
                realization_paths = glob.glob(realization_wildcard)

                # Extract substrings that match the realization_dir_pattern:
                for path in realization_paths:
                    realization_ids.extend([dir for dir in path.split('/') if 'i1p1f1' in dir])

                if verbose:
                    logging.info(f'Found realizations: {realization_ids}')
                
                # sort the realization ids, e.g., r1, r2, ..., r50
                if len(realization_ids) > 1:
                    first_final_realization_id_patterns = ["r[0-9]i1p1f1", "r[0-9][0-9]i1p1f1"]
                    first_realization_ids = sorted([rid for rid in realization_ids if re.match(rf'{first_final_realization_id_patterns[0]}', rid)]) # e.g., ['r1i1p1f1', ..., 'r9i1p1f1']
                    final_realization_ids = sorted([rid for rid in realization_ids if re.match(rf'{first_final_realization_id_patterns[-1]}', rid)]) # e.g., ['r10i1p1f1', ..., 'r50i1p1f1']
                    realization_ids = first_realization_ids + final_realization_ids

            logging.info(f'Found {len(realization_ids)} realizations of {data_var} for {scenario} scenario.')

            for i, realization_id in enumerate(realization_ids):
                filepath_wildcard = data_path + f'{realization_id}/{scenario}/{data_var}/250_km/{frequency}/*/{filename_wildcard}'
                output_paths = sorted(glob.glob(filepath_wildcard)) # sorted by ascending year
                if verbose: logging.info(f'Found {len(output_paths)} years in {realization_id}/{scenario}/{data_var}.')

                # Skip to next scenario if no files are found for this realization.
                if len(output_paths) == 0:
                    continue

                for output_path in output_paths:
                    year = output_path.split('/')[-2]
                    filename = Path(output_path).stem
                    filepaths.append(dict(
                        project_id='CMIP6',
                        model_id='MPI-ESM1-2-LR',
                        realization_id=realization_id,
                        scenario=scenario,
                        data_var=data_var,
                        nominal_resolution='250_km',
                        frequency=frequency,
                        year=year,
                        filename=filename,
                        filepath=output_path
                    ))

                if max_n_realizations is not None:
                    if i == max_n_realizations-1:
                        logging.warning(f'Warning: max_n_realizations is not None. Truncating output to {max_n_realizations} realizations.')
                        break
    df = pd.DataFrame(filepaths)

    return df

def get_output_storage_path_mpi_esm1_2_lr(df_scenario,
                                            realization_id_patterns = ["r[0-9]", "r[0-9][0-9]"],
                                            data_path_interim='./data/interim/CMIP6/MPI-ESM1-2-LR/',
                                            avg_over_ensemble=True,
                                            data_var=None):
    """
    Creates storage path

    Args:
        df_scenario pd.Dataframe(): filepaths to all data with columns that describes the
            filepaths, e.g., scenario, realization_id, year, filepath
        realization_id_patterns [str]: Regex pattern to find first and final 
            realization id, e.g., "r[0-9]" for one digit: r1, r2
        data_path_interim str: Path to the interim data directory
        avg_over_ensemble bool: If true, sets filename for ensemble summary else for
            file with all ensemble members
    """
    if data_var is None:
        data_var = df_scenario.data_var.values[0]

    # Name directory for multiple realizations
    realization_ids = sorted(df_scenario.realization_id.unique())
    first_realization_ids = sorted([rid for rid in realization_ids if re.match(rf'{realization_id_patterns[0]}', rid)]) # e.g., ['r1i1p1f1', ..., 'r9i1p1f1']
    first_realization_id = re.search(rf'{realization_id_patterns[0]}', first_realization_ids[0]).group() # e.g., r1
    final_realization_ids = sorted([rid for rid in realization_ids if re.match(rf'{realization_id_patterns[-1]}', rid)]) # e.g., ['r10i1p1f1', ..., 'r50i1p1f1']
    if len(final_realization_ids) == 0:
        logging.warning(f'Warning. Did not find any realizations with final_realization_id pattern {realization_id_patterns[-1]}. Using first realization_id_pattern {realization_id_patterns[0]} instead.')
        realization_id_patterns[-1] = realization_id_patterns[0]
        final_realization_ids = sorted([rid for rid in realization_ids if re.match(rf'{realization_id_patterns[-1]}', rid)]) # e.g., ['r6i1p1f1', ..., 'r9i1p1f1']
    final_realization_id = re.search(rf'{realization_id_patterns[-1]}', final_realization_ids[-1]).group() # e.g., r50
    ensemble_realization_dir = f'{first_realization_id}-{final_realization_id}i1p1f1' # e.g., 'r10-r50i1p1f1'

    # Name directory for multiple years
    years_dir = f'{df_scenario.year.min()}-{df_scenario.year.max()}' # e.g., '1850-2014'

    # Create output filename following ClimateSet standard, 
    #  e.g., CMIP6_MPI-ESM1-2-LR_r1-r50i1p1f1_historical_tas_250_km_mon_1850-2014.nc
    # filename = f'{df_scenario.project_id.values[0]}_'\
    #    f'{df_scenario.model_id.values[0]}_'\
    #    f'{ensemble_realization_dir}_'\
    #    f'{df_scenario.scenario.values[0]}_'\
    #    f'{df_scenario.data_var.values[0]}_'\
    #    f'{df_scenario.nominal_resolution.values[0]}_'\
    #    f'{df_scenario.frequency.values[0]}_'\
    #    f'{years_dir}.nc'
    if avg_over_ensemble:
        filename = 'ensemble_summaries_yr.nc'
    else:
        filename = 'ensemble.nc'
    
    # Assemble full path following ClimateSet standard
    #  e.g., '/d0/lutjens/interim/CMIP6/MPI-ESM1-2-LR/r10-r50i1p1f1/historical/tas/250_km/mon/1850-2014/ensemble_summary.nc'
    output_storage_path = Path(data_path_interim + f'{ensemble_realization_dir}/'\
                            f'{df_scenario.scenario.values[0]}/'\
                            f'{data_var}/'\
                            f'{df_scenario.nominal_resolution.values[0]}/'\
                            f'{df_scenario.frequency.values[0]}/'\
                            f'{years_dir}/'\
                            f'{filename}')
    return output_storage_path

def add_pr90_to_ensemble(pr90_annual_ens, df_realization, 
                        max_yr=-1, data_var='pr90'):
    # Calculate pr90
    filepaths = df_realization[df_realization.data_var == 'pr'].filepath.values
    ds_pr = xr.open_mfdataset(filepaths[:max_yr], concat_dim="time", combine="nested", data_vars='minimal', coords='minimal', compat='override', parallel=False)

    if ds_pr['pr'].attrs['units'] == 'kg m-2 s-1':
        ds_pr['pr'] *= 86400 # convert from kg m-2 s-1 to mm/day
        ds_pr['pr'].attrs['units'] = 'mm/day'

    ds_pr90 = ds_pr.groupby(ds_pr.time.dt.year).quantile(0.9, dim='time',keep_attrs=True) # .mean(dim='time').plot()

    # Process dimensions
    try:
        ds_pr90 = ds_pr90.drop_vars(['lat_bnds', 'lon_bnds', 'time_bnds'])
    except:
        pass
    try:
        ds_pr90 = ds_pr90.drop_vars(['height'])
    except:
        pass
    ds_pr90 = ds_pr90.rename({'year':'time'})
    ds_pr90 = ds_pr90.rename({'pr':'pr90'})
    ds_pr90 = ds_pr90.rename({'lon':'longitude', 'lat': 'latitude'})

    if type(ds_pr90.time.values[0]) == np.int64:
        logging.info('convert to np.datetime64')
        # Recover the original time format that was changed with the groupby() operation.
        #  Convert years from int64 1850, 1851, ... to np.datetime64 1850-01-01, 1851-01-01, ...
        ds_pr90['time'] = pd.to_datetime(ds_pr90['time'], format='%Y')
        ds_pr90['time'].attrs['units'] = 'yr'

    # Add the current realization to the ensemble of realizations
    ds_pr90_exp = ds_pr90.expand_dims('member', axis=0) # expand by the member dimension
    if pr90_annual_ens is None:
        pr90_annual_ens = ds_pr90_exp
    else:
        pr90_annual_ens = xr.concat([pr90_annual_ens, ds_pr90_exp], dim='member')

    # Assign attributes
    if len(pr90_annual_ens.member) == 1:
        logging.info('assign attr')
        pr90_annual_ens.attrs = ds_pr.attrs
        # Fyi, original resolution is kept in 'table_id', e.g., attrs['table_id'] == 'Aday'
        pr90_annual_ens.attrs['frequency'] = 'yr'
        pr90_annual_ens.attrs['variable_id'] = 'pr90'
        pr90_annual_ens['time'].attrs = ds_pr90.time.attrs
        pr90_annual_ens[data_var].attrs = ds_pr.pr.attrs
        pr90_annual_ens[data_var].attrs['standard_name'] = '90th_percentile_precipitation'
        pr90_annual_ens[data_var].attrs['long_name'] = '90th percentile precipitation'
        pr90_annual_ens[data_var].attrs['comment'] = 'Annual-mean of 90th percentile in every years daily precipitation'

    # Close to avoid memory issues. todo: find out which variables actually need to be closed.
    ds_pr.close(); del ds_pr
    ds_pr90.close(); del ds_pr90
    ds_pr90_exp.close(); del ds_pr90_exp

    return pr90_annual_ens

def add_dtr_to_ensemble(dtr_annual_ens, df_realization, 
                        max_yr=-1, data_var='dtr'):

    # Open tasmin and tasmax, takes about 9s for every realization
    logging.info('Opening datasets')
    filepaths_tasmin = df_realization[df_realization.data_var == 'tasmin'].filepath.values
    ds_tasmin = xr.open_mfdataset(filepaths_tasmin[:max_yr], concat_dim="time", combine="nested", data_vars='minimal', coords='minimal', compat='override', parallel=False)

    filepaths_tasmax = df_realization[df_realization.data_var == 'tasmax'].filepath.values
    ds_tasmax = xr.open_mfdataset(filepaths_tasmax[:max_yr], concat_dim="time", combine="nested", data_vars='minimal', coords='minimal', compat='override', parallel=False)

    # Compute daily diurnal temperature range
    dtr = abs(ds_tasmax.tasmax - ds_tasmin.tasmin)
    # Compute annual mean
    dtr_annual = dtr.resample(time='Y').mean('time').to_dataset(name='dtr')

    # Process dimensions
    try:
        dtr_annual = dtr_annual.drop_vars(['lat_bnds', 'lon_bnds', 'time_bnds'])
        dtr_annual = dtr_annual.drop_vars(['height'])
    except:
        pass
    dtr_annual = dtr_annual.rename({'lon':'longitude', 'lat': 'latitude'})

    # Add the current realization to the ensemble of realizations
    dtr_annual_exp = dtr_annual.expand_dims('member', axis=0) # expand by the member dimension
    if dtr_annual_ens is None:
        dtr_annual_ens = dtr_annual_exp
    else:
        dtr_annual_ens = xr.concat([dtr_annual_ens, dtr_annual_exp], dim='member')

    # Assign attributes
    if len(dtr_annual_ens.member) == 1:
        dtr_annual_ens.attrs = ds_tasmax.attrs
        # Fyi, original resolution is kept in 'table_id', e.g., attrs['table_id'] == 'Aday'
        dtr_annual_ens.attrs['frequency'] = 'yr'
        dtr_annual_ens.attrs['variable_id'] = 'dtr'
        dtr_annual_ens[data_var].attrs = ds_tasmax.tasmax.attrs
        dtr_annual_ens[data_var].attrs['standard_name'] = 'diurnal_temperature_range'
        dtr_annual_ens[data_var].attrs['long_name'] = 'Diurnal temperature range'
        dtr_annual_ens[data_var].attrs['comment'] = 'Difference between daily maximum and minimum near-surface (usually, 2 meter) air temperature'
        dtr_annual_ens[data_var].attrs['units'] = '°C'

    # Close to avoid memory issues. todo: find out which variables actually need to be closed.
    dtr.close(); del dtr
    dtr_annual.close(); del dtr_annual
    dtr_annual_exp.close(); del dtr_annual_exp

    return dtr_annual_ens

def get_ensemble_summaries(df=None,
    data_var: str='tas',
    scenarios: List[str]=['historical', 'rcp26', 'rcp45', 'rcp60', 'rcp85'],
    data_path_interim: str='../data/interim/CMIP5/MPI-GE/',
    debug: bool=True, 
    avg_over_ensemble: bool=True, 
    save_to_disk: bool=False,
    return_outputs: bool=False) -> List[xr.Dataset]:
    """
    Processes one data variable from CMIP6 data. The data is loaded
    from disk at the filenames specified in df, 
    various variable transformations (unit conversion, annual averaging) 
    are applied, ensemble summaries are calculated. The final 
    ensemble summaries are stored in data_path_interim and returned as 
    list of xr datasets. The ensemble summaries contain, e.g., the mean 
    and standard deviation across all model realizations.

    Args:       
        df pd.Dataframe(): filepaths to all data with columns that describes the
            filepaths, e.g., scenario, realization_id, year, filepath
        data_var: Data variable that should be loaded, e.g., tas, pr, ps
        scenarios: List of scenarios from which the data should be loaded.
            The data of each scenario is processed sequentially.
        data_path_interim: Path to output data
        debug: If debug, stops after 2 realizations
        avg_over_ensemble: If true, returns ensemble mean 
            and std deviation. Otherwise returns all ensemble members. 
        save_to_disk: If Ture, saves final xarray to disk. Useful 
            for precomputation of mean and std.
    Returns:
        outputs: If return_outputs, returns list of xarray datasets, one for each scenario.
    """
    if df.empty:
        logging.info(f'df argument is empty')
        return None

    # Check if there's any filenames for the given data_var
    if len(df[df.data_var == data_var]) == 0:
        logging.info(f'Warning, found no entries in df for data_var {data_var}')

    # Initialize list of xr datasets, one for each scenario
    outputs = []

    # Iterate over every scenario
    for scenario in scenarios:
        # Skip to next scenario if no files are found.
        df_scenario = df[df.scenario == scenario]
        if len(df_scenario) == 0: continue

        # Init ensemble of realizations
        output_xr_annual_ens = None

        # Iterate over all realizations in scenarios
        realization_ids = df_scenario.realization_id.unique() # assuming they're already sorted.
        for i,realization_id in tqdm(enumerate(realization_ids)):
            df_realization = df_scenario[df_scenario.realization_id == realization_id]

            # Define a limit for the number of years for faster debugging
            max_yr = len(df_realization.year.unique())
            if debug:
                max_yr = 5

            if data_var == 'dtr':
                output_xr_annual_ens = add_dtr_to_ensemble(output_xr_annual_ens, df_realization, max_yr)
            elif data_var == 'pr90':
                logging.info(f'Adding pr90 for {realization_id}')
                output_xr_annual_ens = add_pr90_to_ensemble(output_xr_annual_ens, df_realization, max_yr)
            else:
                # tested for tas, pr
                # load outputs
                # Src: https://docs.xarray.dev/en/stable/user-guide/io.html#reading-multi-file-datasets
                output_paths = df_realization.filepath.values
                output_xr = xr.open_mfdataset(output_paths[:max_yr], concat_dim="time", combine="nested", data_vars='minimal', coords='minimal', compat='override', parallel=False)

                # Preprocess data
                output_xr_annual = preprocess_data_var_mpi_esm1_2_lr(data=output_xr, data_var=data_var)

                # Add the current realization to the ensemble of realizations
                output_xr_annual_exp = output_xr_annual.expand_dims('member', axis=0) # expand by the member dimension
                if output_xr_annual_ens is None:
                    output_xr_annual_ens = output_xr_annual_exp
                else:
                    output_xr_annual_ens = xr.concat([output_xr_annual_ens, output_xr_annual_exp], dim='member')

                # Assign attributes
                if i == 0:
                    output_xr_annual_ens.attrs = output_xr.attrs
                    output_xr_annual_ens[data_var].attrs = output_xr[data_var].attrs

                # Close output_xr to avoid memory issues. todo: find out which variables actually need to be closed.
                output_xr.close(); del output_xr
                output_xr_annual.close(); del output_xr_annual
                output_xr_annual_exp.close(); del output_xr_annual_exp

            if debug and i == 1:
                logging.info('Debug: stopping after 2 realizations.')
                break

        if avg_over_ensemble:
            # Compute ensemble summaries
            output = output_xr_annual_ens.mean(dim='member')
            output = output.transpose('time','latitude', 'longitude')
            # Calculate standard deviation across members for every location
            output[f'{data_var}_std'] = output_xr_annual_ens[data_var].std(dim='member').transpose('time','latitude', 'longitude')
            # Globally average and then calculate standard deviation across members
            var_global = calculate_global_weighted_average(output_xr_annual_ens[data_var])
            output[f'{data_var}_global_std'] = var_global.std(dim='member')
        else:
            output = output_xr_annual_ens.transpose('member', 'time','latitude', 'longitude')

        # Define the attributes for the output dataset
        output.attrs = output_xr_annual_ens.attrs
        output[data_var].attrs = output_xr_annual_ens[data_var].attrs
        # todo: these realization id's dont follow the standard specification for CMIP5 attributes, but it doesn't seem important rn
        output.attrs['realization'] = f'{realization_ids[0]}-{realization_ids[-1]}'

        # Declare default title for plotting
        output[data_var].attrs['title'] = 'Ensemble ' + output_xr_annual_ens[data_var].attrs['long_name'] + ' in ' + output_xr_annual_ens[data_var].attrs['units']

        if save_to_disk:
            output_storage_path = get_output_storage_path_mpi_esm1_2_lr(
                df_scenario=df_scenario,
                realization_id_patterns = ["r[0-9]", "r[0-9][0-9]"],
                data_path_interim=data_path_interim,
                avg_over_ensemble=avg_over_ensemble,
                data_var=data_var)
            logging.info(f'time dim {output.time}')
            logging.info('Computing ensemble (summaries)')
            with ProgressBar():
                output.load()
            logging.info(f'Saving to disk at: {output_storage_path}')
            Path(output_storage_path).parent.mkdir(parents=True, exist_ok=True)
            output.to_netcdf(output_storage_path, "w")
            logging.info('Saved.')
        if return_outputs:
            outputs.append(output)
        output.close(); del output

    return outputs

def get_args():
    parser = argparse.ArgumentParser(description='Process MPI-ESM1-2-LR data')
    parser.add_argument('--get_ensemble_summaries', action='store_true', default=False, help='Compute ensemble summaries')
    parser.add_argument('--get_ensemble_concat', action='store_true', default=False, help='Compute concatenated ensemble data across realizations and years')
    parser.add_argument('--data_var', type=str, default=None, help='Data variable that shall be processed')
    parser.add_argument('--scenario', type=str, default=None, help='Scenario that shall be processed')
    parser.add_argument('--frequency', type=str, default='mon', help='Frequency that shall be processed')
    parser.add_argument('--debug', action='store_true', default=False, help='debug')
    parser.add_argument('--test_dataloader', action='store_true', default=False, help='Test dataloader')
    return parser.parse_args()

if __name__ == '__main__':
    # Get command line arguments
    args = get_args()
    if args.data_var is None:
        data_vars = ['tas', 'pr', 'hus', 'psl', 'uas', 'vas'] # 'pr90', 'dtr'
    else:
        data_vars = [args.data_var]

    # Calculate ensemble summaries for every variable in the MPI-GE dataset
    if args.get_ensemble_summaries or args.get_ensemble_concat:
        if args.scenario is None:
            scenarios = ['historical', 'ssp119', 'ssp126',  'ssp245', 'ssp370', 'ssp585', 'piControl'] # 
        else:
            scenarios = [args.scenario]
        
        for data_var in data_vars:
            repo_root = ''
            data_root = repo_root + '/d0/lutjens/'
            data_path = data_root + 'raw/CMIP6/MPI-ESM1-2-LR/' # raw data downloaded from CMIPX
            data_path_interim = data_root + 'interim/CMIP6/MPI-ESM1-2-LR/' # interim data, such as, statistical summaries

            if data_var == 'dtr':
                # Retrieve tasmax files
                df_tasmax = get_filepaths_mpi_esm1_2_lr(data_var='tasmax',
                        scenarios=scenarios,
                        data_path=data_path,
                        filename_wildcard='*.nc',
                        frequency='day',
                        verbose=False,
                        max_n_realizations=None)
                # Retrieve tasmin files
                df_tasmin = get_filepaths_mpi_esm1_2_lr(data_var='tasmin',
                        scenarios=scenarios,
                        data_path=data_path,
                        filename_wildcard='*.nc',
                        frequency='day',
                        verbose=False,
                        max_n_realizations=None)
                df = pd.concat([df_tasmin, df_tasmax])
            elif data_var == 'pr90':
                df = get_filepaths_mpi_esm1_2_lr(data_var='pr',
                        scenarios=scenarios,
                        data_path=data_path,
                        filename_wildcard='*.nc',
                        frequency='day',
                        verbose=False,
                        max_n_realizations=None)
            else:
                df = get_filepaths_mpi_esm1_2_lr(data_var=data_var,
                    scenarios=scenarios,
                    frequency=args.frequency,
                    data_path=data_path,
                    verbose=False)

            outputs = get_ensemble_summaries(df=df.copy(),
                data_var=data_var,
                scenarios=scenarios,
                data_path_interim=data_path_interim,
                debug=args.debug,
                avg_over_ensemble=args.get_ensemble_summaries, # if False, we're achieving the functionality of get_ensemble_concat
                save_to_disk=True)

            if outputs is not None:
                for output in outputs:
                    output.close();del output
                del outputs
    
    if args.test_dataloader:
        scenarios = ['historical', 'ssp126',  'ssp245', 'ssp370', 'ssp585']
        data_path_interim = '/d0/lutjens/interim/CMIP6/MPI-ESM1-2-LR/' # interim data, such as, statistical summaries

        data_var = 'tas'
        df = get_filepaths_mpi_esm1_2_lr(data_var=data_var,
                scenarios=scenarios,
                data_path=data_path_interim,
                verbose=True,
                filename_wildcard='ensemble_summaries_yr.nc')
