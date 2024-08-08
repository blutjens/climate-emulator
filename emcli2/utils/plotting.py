from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import cartopy.crs as ccrs
import matplotlib.colors as colors
import cftime # time format in xarray
import argparse
import yaml
import glob
import pandas as pd

from emcli2.dataset.interim_to_processed import calculate_global_weighted_average


def plot_co2_data_var_global_over_time(X_train, 
          X_test, 
          Y_train, 
          Y_test, 
          scenarios_train=['ssp126','ssp370','ssp585','hist-GHG','hist-aer'],
          scenarios_test=['ssp245'],
          preds=None,
          data_var='tas',
          data_var_labels={'tas':{
                'ylabel': 'Annual Global Surface \n Temperate Anomaly in °C',
                'title': 'Surface Temperature, "tas"'}},
          filepath_to_save=None,
          plot_std=False,
          dpi=200,
          ):
    """
    Creates three plots to highlight linear relationship between co2 and 
    temperature. First, plots the globally averaged cumulative CO2 
    emissions over time. Second, plots the globally-averaged surface
    temperature anomaly, tas. Third, creates a scatter plot of tas vs. co2. 
    Args:
        X_train list(n_scenarios_train * xarray.Dataset{
              key: xarray.DataArray(n_time)},
              key: xarray.DataArray(n_time, lat, lon)}, 
              ...)
        X_test similar format to X_train
        Y_train similar format to X_train
        Y_test similar format to X_train
        scenarios_train
        scenarios_test
        plot_std: If true, plot standard deviation
    Returns:
        axs
    """
    fig, axs = plt.subplots(1,3, figsize =(11,4), dpi=dpi)
    
    # colors_train = ['blue', 'green', 'red', 'purple', 'orange', 'black']
    # colors_test = ['blue', 'green', 'red', 'purple', 'orange', 'black']

    def xr_get_years(xr):
        # If xarray is using the cftime format
        if isinstance(X_train[0].time.values[0], cftime.DatetimeNoLeap):
            return xr.time.dt.year
        else:
            return xr.time
        
    # Plot global cumulative CO2 emissions over time
    for idx, scenario in enumerate(scenarios_test):
        axs[0].plot(xr_get_years(X_test[idx]), X_test[idx]['CO2'], color='black', label=scenario)
    for idx, scenario in enumerate(scenarios_train):
        axs[0].plot(xr_get_years(X_train[idx]), X_train[idx]['CO2'], label=scenario)
    axs[0].set_xlabel("Time in years")
    axs[0].set_ylabel("Cumulative anthropogenic CO2 \n emissions since 1850 (GtCO2)")
    axs[0].set_title("Cumulative CO2 emissions")
    axs[0].legend()
    
    # Plot global surface temperature over time
    for idx, scenario in enumerate(scenarios_test):
        time = xr_get_years(Y_test[idx])
        global_avg = calculate_global_weighted_average(Y_test[idx][data_var])
        line, = axs[1].plot(time, global_avg, color='black', label=scenario)
        # Plot standard deviation
        if plot_std and f'{data_var}_global_std' in Y_test[idx]:
            axs[1].fill_between(time, 
                global_avg - Y_test[idx][f'{data_var}_global_std'], 
                global_avg + Y_test[idx][f'{data_var}_global_std'], 
                color=line.get_color(), alpha=0.3)

        if preds is not None:
            axs[1].plot(xr_get_years(Y_test[idx]), preds, color='black', linestyle='--', alpha=0.8, label=scenario+'-pred')
    
    for idx, scenario in enumerate(scenarios_train):
        time = xr_get_years(Y_train[idx])
        global_avg = calculate_global_weighted_average(Y_train[idx][data_var])
        axs[1].plot(time, global_avg, label=scenario)

        # Plot standard deviation
        if plot_std and f'{data_var}_global_std' in Y_train[idx]:
            axs[1].fill_between(time, 
                global_avg - Y_train[idx][f'{data_var}_global_std'], 
                global_avg + Y_train[idx][f'{data_var}_global_std'], alpha=0.3)

    axs[1].set_xlabel("Time in years")
    axs[1].set_ylabel(data_var_labels[data_var]['ylabel'])
    axs[1].set_title(data_var_labels[data_var]['title'])
    axs[1].legend()

    # Plot global surface temperature over cum. co2 emissions
    for idx, scenario in enumerate(scenarios_test):
        axs[2].plot(X_test[idx]['CO2'], calculate_global_weighted_average(Y_test[idx][data_var]), color='black', label=scenario)
        if preds is not None:
            axs[2].plot(X_test[idx]['CO2'], preds, linestyle='--', color='black', label=scenario+'-pred')
    for idx, scenario in enumerate(scenarios_train):
        axs[2].plot(X_train[idx]['CO2'], calculate_global_weighted_average(Y_train[idx][data_var]), label=scenario)
    axs[2].set_xlabel("Cum. CO2 Emissions")
    axs[2].set_ylabel(data_var_labels[data_var]['ylabel'])
    axs[2].set_title(f"CO2 vs. {data_var}")
    
    plt.tight_layout()

    if filepath_to_save is not None:
        Path(filepath_to_save).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(filepath_to_save)

    return axs

def plot_all_scenarios(Y_data, 
                       data_var='tas',
                       scenarios=['ssp126','ssp370','ssp585','hist-GHG','hist-aer'],
                       plot_std=True, meta=None,
                       filepath_to_save=None):
    fig, axs = plt.subplots(1,1, dpi=100)

    def xr_get_years(xr):
        # If xarray is using the cftime format
        if isinstance(xr.time.values[0], cftime.DatetimeNoLeap):
            return xr.time.dt.year
        else:
            return xr.time

    # Plot global surface temperature over time
    for idx, scenario in enumerate(scenarios):
        time = xr_get_years(Y_data[idx])
        global_avg = calculate_global_weighted_average(Y_data[idx][data_var])
        line, = axs.plot(time, global_avg, label=scenario) # 
        # Plot standard deviation
        if plot_std and f'{data_var}_global_std' in Y_data[idx]:
            axs.fill_between(time, 
                global_avg - Y_data[idx][f'{data_var}_global_std'], 
                global_avg + Y_data[idx][f'{data_var}_global_std'], 
                color=line.get_color(), alpha=0.3)

    axs.set_xlabel("Time in years")
    axs.set_ylabel(meta[data_var]['ylabel'])
    axs.set_title(meta[data_var]['title'])
    axs.legend()

    plt.tight_layout()
    if filepath_to_save is not None:
        Path(filepath_to_save).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(filepath_to_save)

    plt.show()
    plt.close()

def plot_all_vars_global_avg(X_train, 
            X_test, 
            Y_train, 
            Y_test, 
            scenarios_train=['ssp126','ssp370','ssp585','hist-GHG','hist-aer'],
            scenarios_test=['ssp245']):
    """
    Plots the global averages over time of all input and output variables. 
    Args:
        X_train list(n_scenarios_train * xarray.Dataset{
              key: xarray.DataArray(n_time)},
              key: xarray.DataArray(n_time, lat, lon)}, 
              ...)
        X_test similar format to X_train
        Y_train similar format to X_train
        Y_test similar format to X_train
        scenarios_train
        scenarios_test
    """
    fig, axs = plt.subplots(4,2, figsize =(12,8))
    
    # Plot global cumulative CO2 emissions over time
    ax = axs[0,0]
    for idx, scenario in enumerate(scenarios_train):
        ax.plot(X_train[idx].time, X_train[idx]['CO2'], label=scenario)
    for idx, scenario in enumerate(scenarios_test):
        ax.plot(X_test[idx].time, X_test[idx]['CO2'], linestyle='--', color='black', label=scenario)
    #ax.set_xlabel("Time in years")
    ax.set_ylabel("Cumulative anthropogenic CO2 \n emissions since 1850 (GtCO2)")
    ax.set_title("Cumulative CO2 emissions")
    ax.legend()
    
    # Plot global CH4 emissions over time
    ax = axs[0,1]
    for idx, scenario in enumerate(scenarios_train):
        ax.plot(X_train[idx].time, X_train[idx]['CH4'], label=scenario)
    for idx, scenario in enumerate(scenarios_test):
        ax.plot(X_test[idx].time, X_test[idx]['CH4'], linestyle='--', color='black', label=scenario)
    #ax.set_xlabel("Time in years")
    ax.set_ylabel("Anthropogenic CH4 \n emissions (GtCH4/year)")
    ax.set_title("CH4 emissions")
    
    # Plot globally-averaged SO2 emissions over time
    ax = axs[1,0]
    for idx, scenario in enumerate(scenarios_train):
        ax.plot(X_train[idx].time, calculate_global_weighted_average(X_train[idx]['SO2']), label=scenario)
    for idx, scenario in enumerate(scenarios_test):
        ax.plot(X_test[idx].time, calculate_global_weighted_average(X_test[idx]['SO2']), linestyle='--', color='black', label=scenario)
    #ax.set_xlabel("Time in years")
    ax.set_ylabel("Anthropogenic SO2 \n emissions (?SO2/yr)")# todo check units (TgSO2/year)")
    ax.set_title("SO2 emissions")
    
    # Plot globally-averaged BC emissions over time
    ax = axs[1,1]
    for idx, scenario in enumerate(scenarios_train):
        ax.plot(X_train[idx].time, calculate_global_weighted_average(X_train[idx]['BC']), label=scenario)
    for idx, scenario in enumerate(scenarios_test):
        ax.plot(X_test[idx].time, calculate_global_weighted_average(X_test[idx]['BC']), linestyle='--', color='black', label=scenario)
    #ax.set_xlabel("Time in years")
    ax.set_ylabel("Anthropogenic BC \n emissions (?BC/yr)")# todo check units (TgBC/year)")
    ax.set_title("BC emissions")
    
    # Plot global surface temperature over time
    ax = axs[2,0]
    for idx, scenario in enumerate(scenarios_train):
        ax.plot(Y_train[idx].time, calculate_global_weighted_average(Y_train[idx]['tas']), label=scenario)
    for idx, scenario in enumerate(scenarios_test):
        ax.plot(Y_test[idx].time, calculate_global_weighted_average(Y_test[idx]['tas']), linestyle='--', color='black', label=scenario)
    #ax.set_xlabel("Time in years")
    ax.set_ylabel("Annual Global Surface \n Temperate Anomaly in °C")
    ax.set_title("Surface temperature")
    
    # Plot global diurnal temperature range over time
    ax = axs[2,1]
    for idx, scenario in enumerate(scenarios_train):
        ax.plot(Y_train[idx].time, calculate_global_weighted_average(Y_train[idx]['diurnal_temperature_range']), label=scenario)
    for idx, scenario in enumerate(scenarios_test):
        ax.plot(Y_test[idx].time, calculate_global_weighted_average(Y_test[idx]['diurnal_temperature_range']), linestyle='--', color='black', label=scenario)
    #ax.set_xlabel("Time in years")
    ax.set_ylabel("Annual Global Diurnal \n Temperature Range in °C")
    ax.set_title("Diurnal Temperature Range")
    
    # Plot global precipitation over time
    ax = axs[3,0]
    for idx, scenario in enumerate(scenarios_train):
        ax.plot(Y_train[idx].time, calculate_global_weighted_average(Y_train[idx]['pr']), label=scenario)
    for idx, scenario in enumerate(scenarios_test):
        ax.plot(Y_test[idx].time, calculate_global_weighted_average(Y_test[idx]['pr']), linestyle='--', color='black', label=scenario)
    ax.set_xlabel("Time in years")
    ax.set_ylabel("Annual Global Precipitation\n (mm/day)")
    ax.set_title("Precipitation")
    
    # Plot global extreme precipitation over time
    ax = axs[3,1]
    for idx, scenario in enumerate(scenarios_train):
        ax.plot(Y_train[idx].time, calculate_global_weighted_average(Y_train[idx]['pr90']), label=scenario)
    for idx, scenario in enumerate(scenarios_test):
        ax.plot(Y_test[idx].time, calculate_global_weighted_average(Y_test[idx]['pr90']), linestyle='--', color='black', label=scenario)
    ax.set_xlabel("Time in years")
    ax.set_ylabel("Annual Global Extreme \n Precipitation (mm/day)")
    ax.set_title("Extreme Precipitation")
    
    plt.tight_layout()

    return axs

def plot_regression_fit(y_true, y_pred, 
                        data_var='tas', 
                        meta={'tas':{'title': 'tas', 'unit': '°C'}},
                        scenario='ssp245',
                        filepath_to_save=None):
    y_true_end_of_century = y_true.sel(time = slice('2080', '2100')).mean(dim='time').values.flatten()
    y_pred_end_of_century = y_pred.sel(time = slice('2080', '2100')).mean(dim='time').values.flatten()

    #y_true_end_of_century = Y_test_local[0][data_var].sel(time = slice('2080', '2100')).values.flatten()
    #y_pred_end_of_century = preds_pattern_scaling_ds[0][data_var].sel(time = slice('2080', '2100')).values.flatten()

    coef = np.polyfit(y_true_end_of_century,y_pred_end_of_century,1)
    poly1d_fn = np.poly1d(coef) 
    # poly1d_fn is now a function which takes in y_true_end_of_century and returns an estimate for y_pred_end_of_century

    plt.plot(y_true_end_of_century,y_pred_end_of_century, 
            '.', y_true_end_of_century, poly1d_fn(y_true_end_of_century), '--k') #'--k'=black dashed line, 'yo' = yellow circle marker

    plt.xlabel(f'Target {meta[data_var]["title"]} \naveraged over {scenario}ys 2080-2100 in {meta[data_var]["unit"]}')
    plt.ylabel(f'Predicted {meta[data_var]["title"]} \naveraged over {scenario} 2080-2100 in {meta[data_var]["unit"]}')
    plt.title(f'Local {meta[data_var]["title"]}')

    plt.tight_layout()
    if filepath_to_save is not None:
        Path(filepath_to_save).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(filepath_to_save)

    plt.show()
    plt.close()

def plot_tas_annual_local_err_map(tas_true, tas_pred, data_var='tas', unit='°C',
        filepath_to_save: str=None):
    """
    Plots three maps of surface temperature anomalies (tas).
    This includes the ground-truth tas, predicted tas,
    and error of ground-truth - predicted. 
    Args:
        tas_true xr.DataArray(n_t, n_lat, n_lon): Ground-truth 
            annual mean surface temperature anomalies over the 
            globe in °C
        tas_pred xr.DataArray(n_t, n_lat, n_lon): Predicted tas
        unit 'str': unit of data_var, used for plot labels
        filepath_to_save: If not None, will save figure at this path overwriting existing files
    Returns:
        axs: matplotlib axes object
    """
    # Get coordinates
    try:
        lon = tas_true.longitude.data
        lat = tas_true.latitude.data
    except:
        lon = tas_true.lon.data
        lat = tas_true.lat.data

    # Compute temporal average of target and prediction over evaluation timeframe
    tas_true_t_avg = tas_true.sel(time=slice("2081", "2100")).mean(dim="time")
    tas_pred_t_avg = tas_pred.sel(time=slice("2081", "2100")).mean(dim="time")

    # Compute error of prediction minus target
    err_pattern_scaling = tas_pred_t_avg - tas_true_t_avg

    # Create figure with PlateCarree projection
    projection = ccrs.Robinson(central_longitude=0)
    transform = ccrs.PlateCarree(central_longitude=0)
    fig, axs = plt.subplots(1, 3, figsize=(12, 3), 
        subplot_kw=dict(projection=projection),
        dpi=300)
    params = {'legend.fontsize': 'x-large',
            'axes.labelsize': 'x-large',
            'axes.titlesize':'x-large',
            'xtick.labelsize':'x-large',
            'ytick.labelsize':'x-large'}
    plt.rcParams.update(params)


    # Plot ground-truth surface temperature anomalies
    bounds_tas = np.hstack((np.linspace(tas_true_t_avg.min().values, -0.1, 5), np.linspace(0.1, tas_true_t_avg.max().values, 5)))
    cnorm = colors.BoundaryNorm(boundaries=bounds_tas, ncolors=256)
    if data_var == 'psl' or data_var == 'vas' or data_var == 'tas' or data_var == 'dtr':
        cnorm = colors.TwoSlopeNorm(vmin=tas_true_t_avg.min().values, vcenter=0, vmax=tas_true_t_avg.max().values) # center colorbar around zero
    elif data_var == 'huss':
        cnorm = colors.TwoSlopeNorm(vmin=tas_true_t_avg.min().values, vcenter=(tas_true_t_avg.max().values-tas_true_t_avg.min().values)/2 + tas_true_t_avg.min().values, vmax=tas_true_t_avg.max().values) # center colorbar around zero
    mesh = axs[0].pcolormesh(lon, lat, tas_true_t_avg.data, cmap='coolwarm',norm=cnorm, transform=transform)
    cbar = plt.colorbar(mesh, ax=axs[0], orientation='horizontal', shrink=0.95, pad=0.05)
    cbar.set_label(f'Target {data_var} anom., \naveraged over 2080-2100 in {unit}')
    cbar.ax.set_xscale('linear')
    axs[0].coastlines()

    # Plot predicted surface temperature anomalies
    mesh = axs[1].pcolormesh(lon, lat, tas_pred_t_avg.data, cmap='coolwarm',norm=cnorm, transform=transform)
    cbar = plt.colorbar(mesh, ax=axs[1], orientation='horizontal', shrink=0.95, pad=0.05)
    cbar.set_label(f'Predicted {data_var} anom., \navg 2080-2100 in {unit}')
    cbar.ax.set_xscale('linear')
    axs[1].coastlines()

    # Plot error of pattern scaling
    if data_var == 'pr90':
        bounds = np.hstack((np.linspace(err_pattern_scaling.min(), -0.10, 5), np.linspace(0.10, err_pattern_scaling.max(), 5)))
    elif data_var == 'pr':
        bounds = np.hstack((np.linspace(err_pattern_scaling.min(), -0.10, 5), np.linspace(0.10, err_pattern_scaling.max(), 5)))
    elif data_var == 'dtr' or data_var == 'diurnal_temperature_range':
        bounds = np.hstack((np.linspace(err_pattern_scaling.min(), -0.05, 5), np.linspace(0.05, err_pattern_scaling.max(), 5))) # /4
    elif data_var == 'uas' or data_var == 'psl':
        bounds = np.hstack((np.linspace(err_pattern_scaling.min(), -0.05, 5), np.linspace(0.05, err_pattern_scaling.max(), 5)))
    elif data_var == 'huss':
        bounds = np.hstack((np.linspace(err_pattern_scaling.min(), -0.00005, 5), np.linspace(0.00005, err_pattern_scaling.max(), 5)))
    elif data_var == 'tas':
        bounds = np.hstack((np.linspace(err_pattern_scaling.min(), -0.25, 5), np.linspace(0.25, err_pattern_scaling.max(), 5)))
    else:
        bounds = None
    if bounds is not None:
        cnorm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
    else:
        cnorm = colors.TwoSlopeNorm(vmin=err_pattern_scaling.min().values, vcenter=0, vmax=err_pattern_scaling.max().values) # center colorbar around zero
    mesh = axs[2].pcolormesh(lon, lat, err_pattern_scaling.data, cmap='coolwarm', norm=cnorm, transform=transform)
    cbar = plt.colorbar(mesh, ax=axs[2], orientation='horizontal', shrink=0.95, pad=0.05)
    cbar.set_label(f'Error (pred-target) in {data_var} anom. \n avg 2080-2100 in {unit}')
    cbar.ax.set_xscale('linear')
    cbar.ax.minorticks_on()
    axs[2].coastlines()
    
    plt.tight_layout()
    if filepath_to_save is not None:
        Path(filepath_to_save).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(filepath_to_save)

    return axs

def plot_histogram(hist, bin_values, 
        xlabel="Annual Mean Local Surface Temp. Anomaly, tas, in K",
        ylabel="Relative frequency",
        title="Histogram of surface temp. in train set",
        ax=None):
    """
    Args:
        hist: occurence frequencies for every section in bin_values
        bin_values: x-values of edges of histogram bins
    """
    # Plot the histogram using matplotlib's bar function
    if ax is None:
        fig, ax = plt.subplots(figsize =(4, 2))
    w = abs(bin_values[1]) - abs(bin_values[0])
    ax.bar(bin_values[:-1], hist, width=w, alpha=0.5, align='center')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return ax

def plot_climatology_map(
    climatology, 
    data_var: str='tas', 
    unit: str='°C',
    scenario: str='piControl',
    filepath_to_save: str=None,
    dpi: int=300):
    """
    Plots climatology
    Args:
        climatology xr.DataArray(n_lat, n_lon): Climatology of data_var
    """

    # Get coordinates
    try:
        lon = climatology.longitude.data
        lat = climatology.latitude.data
    except:
        lon = climatology.lon.data
        lat = climatology.lat.data

    projection = ccrs.Robinson(central_longitude=0)
    transform = ccrs.PlateCarree(central_longitude=0)
    fig, axs = plt.subplots(1, 1, #figsize=(9, 9), 
        subplot_kw=dict(projection=projection),
        dpi=dpi)
    if data_var == 'dtr' or data_var == 'pr' or data_var == 'pr90' or data_var == 'psl' or data_var == 'huss':
        climatology_mid = (climatology.max() - climatology.min())/2 + climatology.min()
        bounds = np.hstack((np.linspace(climatology.min(), climatology_mid, 5), 
                            np.linspace(climatology_mid, climatology.max(), 5)))
    else:
        bounds = np.hstack((np.linspace(climatology.min(), -0.10, 5), np.linspace(0.10, climatology.max(), 5)))
    divnorm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
    mesh = axs.pcolormesh(lon, lat, climatology.data, cmap='coolwarm', norm=divnorm, transform=transform)
    cbar = plt.colorbar(mesh, ax=axs, orientation='horizontal', shrink=0.95, pad=0.05)
    cbar.set_label(f'Climatology of {climatology.attrs["long_name"]}, {data_var},  in {unit}\n '\
        f'Mean over {climatology.attrs["parent_experiment_id"]}: {climatology.attrs["time_start"]} to {climatology.attrs["time_stop"]}')
    cbar.ax.set_xscale('linear')
    axs.coastlines()

    plt.tight_layout()
    if filepath_to_save is not None:
        Path(filepath_to_save).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(filepath_to_save)
    return axs

def plot_m_member_subsets_experiment(data_var='pr', 
        metric='Y_rmse_spatial',
        model_keys = ['cnn_lstm', 'pattern_scaling'],
        experiment_keys = {'cnn_lstm': 'm_member_subsets_with_m50_eval_on_all_spcmp_dwp_manyr',
                            'pattern_scaling': 'm_member_subsets_with_m50_replace_False_eval_on_all_manyr'},
        verbose = False,
        data_var_labels = None,
        reduction_model_seed = 'mean',
        xinset = None,
        xscale = 'log',
        plt_sample = False,
        plot_inset_in_separate_fig = True
        ):
    """
    Creates the plots for the JAMES24 submission internal variability experiment.
    The first plot shows the RMSE of the model predictions over the number of realizations
    in the training set. The second plot shows difference in RMSE of each model.
    
    Args:
        data_var str: The variable to plot, e.g., 'tas', 'pr', 'pr90', 'psl', 'huss', 'uas'
        metric str: The metric to plot, e.g., 'Y_rmse_spatial', 'Y_rmse_global_np'
        model_keys list(str): The models to plot, e.g., ['cnn_lstm', 'pattern_scaling']
        experiment_keys list(str): The experiment keys that match each model. len(model_keys) must be len(experiment_keys)
        verbose bool: If True, will add some print statements to help debugging
        data_var_labels {'data_var1': {'ylabel': '...', 'title': '...},
                        'data_var2: ...}    Labels for plotting each data_var
        reduction_model_seed str: Method to reduce across model seeds; e.g., 'mean','min'
        xinset int: If not None, will add a 2nd column to the plot with an inset of realizations from 1 to xinset.
        plot_inset_in_separate_fig bool: If True, will plot inset into separate figure (will only plot Delta RMSE).
        xscale str: The scale of the x-axis, e.g., 'log', 'linear'
        plt_sample bool: If True, will add curve illustrating a sample realization into the plot. Not used anymore
            because difference plot shows all sample realizations using small green dots.
    """
    print(f'Starting to analyse {metric}({data_var})')
    filepath_to_save = f'docs/figures/mpi-esm1-2-lr/{data_var}/{experiment_keys["cnn_lstm"]}/{metric}.png'
    # Deprecated paths for:
    # paths_experiment = [f'runs/cnn_lstm/mpi-esm1-2-lr/m_member_subsets_with_m50_eval_on_all_spcmp_dwp/{data_var}/member_subset*/sweep/task-*/incumbents/incumbents.yaml', 
    #                    f'runs/pattern_scaling/mpi-esm1-2-lr/m_member_subsets_with_m50_replace_False_eval_on_all/{data_var}/incumbents/incumbents.yaml']
    paths_experiment = []
    for model_key in model_keys:
        assert model_key in experiment_keys.keys(), f'experiment_keys needs to contain experiment ID for {model_key}.'
        if model_key == 'cnn_lstm':
            if args.use_debug_paths:
                paths_experiment.append(f'runs/cnn_lstm/mpi-esm1-2-lr/{experiment_keys["cnn_lstm"]}/{data_var}/incumbents/incumbents.yaml') 
            else:
                paths_experiment.append(f'runs/cnn_lstm/mpi-esm1-2-lr/{experiment_keys["cnn_lstm"]}/{data_var}/memberseed-*/member_subset*/sweep/task-*/incumbents/incumbents.yaml') 
        elif model_key == 'pattern_scaling':
            paths_experiment.append(f'runs/pattern_scaling/mpi-esm1-2-lr/{experiment_keys["pattern_scaling"]}/{data_var}/memberseed-*/incumbents/incumbents.yaml')
        else:
            raise ValueError(f'Unknown model key {model_key}.')

    metric_labels = {'Y_rmse_spatial_tex': r'RMSE$_s$', 
                    'Y_rmse_spatial_text': 'Spatial RMSE', 
                    'Y_rmse_global_np_tex': r'RMSE$_g$',
                    'Y_rmse_global_np_text': 'Global RMSE',
                    }
    
    # Iterate over every model
    lens = dict()
    metric_value = dict()
    idcs_member_id = dict()
    entries = []
    for model_key, path_experiment in zip(model_keys, paths_experiment):
        idcs_member_id[model_key] = dict()
        lens[model_key] = dict()
        metric_value[model_key] = dict()
        idcs_member_id[model_key] = dict()

        paths_incumbents = glob.glob(path_experiment)

        for path_incumbents in paths_incumbents:
            inc = yaml.safe_load(open(path_incumbents, 'r'))
            for idx_member_subset in inc.keys():
                entries.append({'model_key': model_key,
                                'idx_member_subset': idx_member_subset,
                                'len': len(inc[idx_member_subset]["m_member_subset"]),
                                'seed': inc[idx_member_subset]["seed"],
                                'memberseed': inc[idx_member_subset]["memberseed"],
                                metric: inc[idx_member_subset][metric],
                })
                if verbose:
                    # Print values to cmd line
                    for key in inc.keys():
                        print(f'Member subset {key} with length m={len(inc[key]["m_member_subset"])} has '\
                            f'\n\t{metric} {inc[key][metric]}')

    # Convert to pandas dataframe
    df = pd.DataFrame(entries)
    
    # Compute average model performance per member subset realization
    #  i.e., Reduce across multiple model runs on a given subset
    df[f'{metric}_modelmean'] = df.groupby(['model_key','idx_member_subset', 'memberseed'])[metric].transform('mean')
    if reduction_model_seed != 'mean':
        df[f'{metric}_modelmin'] = df.groupby(['model_key','idx_member_subset', 'memberseed'])[metric].transform('min')
        df[f'{metric}_modelstd'] = df.groupby(['model_key','idx_member_subset', 'memberseed'])[metric].transform('std')
        assert NotImplementedError(f'Only reduction_model_seed="mean" is currently implemented.')
    
    # Reduce across memberseed, i.e., across subsets that have the same number of members
    #  Compute an average across all entries that have the same model_key and idx_member_subset; only counting entries with duplicate memberseed once.
    df[f'{metric}_membermean'] = df.drop_duplicates(subset=['model_key', 'idx_member_subset', 'memberseed']).groupby(['model_key', 'idx_member_subset'])[f'{metric}_modelmean'].transform('mean')
    #  Fill the nan values in the duplicate entries with the mean value
    df[f'{metric}_membermean'] = df.groupby(['model_key', 'idx_member_subset'])[f'{metric}_membermean'].transform(lambda x: x.fillna(x.mean()))
    # Repeat for standard deviation across memberseeds
    df[f'{metric}_memberstd'] = df.drop_duplicates(subset=['model_key', 'idx_member_subset', 'memberseed']).groupby(['model_key', 'idx_member_subset'])[f'{metric}_modelmean'].transform('std')
    df[f'{metric}_memberstd'] = df.groupby(['model_key', 'idx_member_subset'])[f'{metric}_memberstd'].transform(lambda x: x.fillna(x.mean()))

    if 'equal' in filepath_to_save:
        xaxis_key = 'idx_member_subset'
        xlabel = 'Member subset index'
    else:
        xaxis_key = 'len'
        xlabel = '#realizations in train set'

    # Initialize figure
    n_cols = 2 if (xinset is not None and not plot_inset_in_separate_fig) else 1
    figsize = (10,5) if (xinset is not None and not plot_inset_in_separate_fig) else None
    fig, axs = plt.subplots(2,n_cols,figsize=figsize,dpi=300)
    st = fig.suptitle(f'Model error over number of realizations in training set \nfor {data_var_labels[data_var]["title"]}', fontsize='large')
    if xinset is None or plot_inset_in_separate_fig:
        axs = axs[:,None]

    colorpallete = ['tab:blue', 'tab:orange']
    model_labels = {'cnn_lstm': 'CNN-LSTM', 'pattern_scaling': 'Pattern Scaling'}

    for m, model_key in enumerate(model_keys):
        df_model = df[df['model_key'] == model_key]
        # Plot a sample RMSE over number of realizations
        if plt_sample:
            if m==0:
                sample_memberseed = df_model['memberseed'].unique()[0]
            df_sample = df_model[(df_model['memberseed'] == sample_memberseed)]
            df_sample = df_sample.drop_duplicates(subset=['model_key', 'idx_member_subset'])
            df_sample = df_sample.sort_values(by=[xaxis_key])
            axs[0,0].plot(df_sample[xaxis_key].astype(int), 
                        df_sample[f'{metric}_modelmean'], label=f'{model_labels[model_key]} sample',
                        linestyle='--',color=colorpallete[m],alpha=0.5)
        # Print memberseed-averaged RMSE at n=50
        df_uniq = df_model.drop_duplicates(subset=['model_key', 'idx_member_subset'])
        df_uniq = df_uniq.sort_values(by=[xaxis_key])

        if np.any(df_uniq.len==50):
            metric_at_n50 = df_uniq[df_uniq.len==50][f'{metric}_membermean'].values.squeeze()
            std_at_n50 = df_uniq[df_uniq.len==50][f'{metric}_memberstd'].values.squeeze()
            print(f'{model_key:<15} {metric:<20} {data_var:<5} at n=50; mean: {metric_at_n50:.10f} std: {std_at_n50:.10f}')
        else:
            print(f'{model_key:<15} {metric:<20} {data_var:<5} at n=50; mean: (did not find len==50 in dataset)')
        # Plot memberseed-averaged RMSE over number of realizations
        axs[0,0].plot(df_uniq[xaxis_key].astype(int), 
                    df_uniq[f'{metric}_membermean'], marker='o', markersize=2.5, label=rf'{model_labels[model_key]} $\pm \sigma$',
                    color=colorpallete[m])
        # Plot std deviation of memberseed-averaged
        axs[0,0].fill_between(df_uniq[xaxis_key].astype(int), 
                    df_uniq[f'{metric}_membermean'] - df_uniq[f'{metric}_memberstd'],
                    df_uniq[f'{metric}_membermean'] + df_uniq[f'{metric}_memberstd'], 
                    alpha=0.2, color=colorpallete[m])# label=f'{model_key}_std',)
        # Plot inset from idx_member_subset = 1-10
        if (xinset is not None and not plot_inset_in_separate_fig):
            # todo: i can probably delete this xinset code, because I prefer to 
            #  plot xinset as a separate plot. 
            if plt_sample:
                df_inset = df_sample[df_sample[xaxis_key]<=xinset]
                axs[0,1].plot(df_inset[xaxis_key].astype(int), 
                            df_inset[f'{metric}_modelmean'], label=f'{model_labels[model_key]} sample',
                            linestyle='--',color=colorpallete[m],alpha=0.5)
            df_uniq_inset = df_uniq[df_uniq[xaxis_key]<=xinset]
            axs[0,1].plot(df_uniq_inset[xaxis_key].astype(int), 
                        df_uniq_inset[f'{metric}_membermean'], marker='o', markersize=2.5, label=rf'{model_labels[model_key]} $\pm \sigma$',
                        color=colorpallete[m])
            axs[0,1].fill_between(df_uniq_inset[xaxis_key].astype(int), 
                        df_uniq_inset[f'{metric}_membermean'] - df_uniq_inset[f'{metric}_memberstd'],
                        df_uniq_inset[f'{metric}_membermean'] + df_uniq_inset[f'{metric}_memberstd'], 
                        alpha=0.2, color=colorpallete[m])# label=f'{model_key}_std',)

    # minimum number of subsets across all model_keys
    df_uniqs = {}
    for model_key in model_keys:
        df_uniqs[model_key] = df[df['model_key'] == model_key].drop_duplicates(subset=['model_key', 'idx_member_subset'])

    min_m_subsets = min([len(df_uniqs[model_key]) for model_key in model_keys])
    max_m_subsets = max([len(df_uniqs[model_key]) for model_key in model_keys])
    xvalues = df_uniqs['pattern_scaling'][xaxis_key][:max_m_subsets].to_numpy(dtype=int).tolist()

    from matplotlib.ticker import ScalarFormatter
    axs[0,0].set_xscale(xscale)
    if xscale == 'log':
        xticks = xvalues[:3][::2] + xvalues[4:20][::5] + xvalues[29:][::10]
        axs[0,0].set_xticks(xticks)
        axs[0,0].get_xaxis().set_major_formatter(ScalarFormatter())

    axs[0,0].set_xlabel(xlabel)
    axs[0,0].set_ylabel(f'{metric_labels[f"{metric}_tex"]}')
    axs[0,0].set_title(f'{metric_labels[f"{metric}_text"]}')
    # Add legend
    if xinset is None or plot_inset_in_separate_fig:
        axs[0,0].legend()
    else:
        axs[0,1].legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')

    if (xinset is not None and not plot_inset_in_separate_fig):
        from matplotlib.ticker import MaxNLocator
        axs[0,1].xaxis.set_major_locator(MaxNLocator(integer=True))
        axs[0,1].set_ylim(axs[0,0].get_ylim())
        axs[0,1].set_xlabel(xlabel)
        axs[0,1].set_title(f'{metric_labels[f"{metric}_text"]} inset for [0, {xinset-1}]')

    # Plot the difference between the two models
    df_ps = df[df['model_key'] == 'pattern_scaling'].loc[df['idx_member_subset'].isin(df[df['model_key'] == 'cnn_lstm'].idx_member_subset)] # Keep only the idx_member_subsets for which the cnn_lstm was run
    df_cnn = df[df['model_key'] == 'cnn_lstm']
    df_cnn = df_cnn.drop_duplicates(subset=['model_key', 'idx_member_subset', 'memberseed'])
    # Merging datasets to handle missing entries
    df_merged = pd.merge(df_cnn, df_ps, on=['memberseed','idx_member_subset','len'], suffixes=('_cnn', '_ps'))
    # Compute difference between CNN modelseed average and PS for each memberseed
    df_merged[f'{metric}_modelmean_diff'] = df_merged[f'{metric}_modelmean_cnn'] - df_merged[f'{metric}_modelmean_ps']
    # Keep the relevant columns
    df_diff = df_merged[['len', 'memberseed', 'idx_member_subset', f'{metric}_modelmean_diff']]
    # Sort by len or idx_member_subset
    df_diff = df_diff.sort_values(by=[xaxis_key])
    # Compute mean and std over memberseed
    df_diff[f'{metric}_modelmean_diff_membermean'] = df_diff.groupby(['len', 'idx_member_subset'])[f'{metric}_modelmean_diff'].transform('mean')
    df_diff[f'{metric}_modelmean_diff_memberstd'] = df_diff.groupby(['len', 'idx_member_subset'])[f'{metric}_modelmean_diff'].transform('std')
    if plt_sample:
        df_diff_sample = df_diff[df_diff['memberseed'] == sample_memberseed]

    df_diff_means = df_diff.drop_duplicates(subset=[xaxis_key])
    xvalues = df_diff_means[xaxis_key].to_numpy(dtype=int).tolist()
    means = df_diff_means[f'{metric}_modelmean_diff_membermean'].to_numpy()
    std = df_diff_means[f'{metric}_modelmean_diff_memberstd'].to_numpy()

    axs[1,0].plot(xvalues, means, color='black', marker='o', markersize=2.5, label=rf'$\Delta${metric_labels[f"{metric}_tex"]}$\pm \sigma_k$')
    axs[1,0].fill_between(xvalues, 
                means - std,
                means + std,
                alpha=0.2,color='black')# label=f'{model_key}_std',)
    axs[1,0].plot(df_diff[xaxis_key], df_diff[f'{metric}_modelmean_diff'], marker='o', markersize=0.5, linestyle="", color='tab:green',label=rf'$\Delta${metric_labels[f"{metric}_tex"]}'+r'$_{,k}$')
    axs[1,0].plot(xvalues, np.zeros(len(xvalues)), linestyle='--', color='black', alpha=0.5)#, label=rf'$\Delta${metric_labels[f"{metric}_tex"]}$=0$')

    axs[1,0].set_xscale(xscale)
    if xscale == 'log':
        axs[1,0].set_xticks(xticks)
        axs[1,0].get_xaxis().set_major_formatter(ScalarFormatter())

    # Set yscale
    if data_var == 'pr':
        vmax = 1.25 * abs(df_diff[f'{metric}_modelmean_diff']).max()
    else:
        vmax = abs(means).max() + abs(std).max()

    axs[1,0].set_ylim([-vmax, vmax])
    axs[1,0].set_xlabel(xlabel)
    axs[1,0].set_ylabel(f'{metric_labels[f"{metric}_tex"]}(CNN-LSTM) \n - {metric_labels[f"{metric}_tex"]}(Pattern Scaling)')
    axs[1,0].set_title(f'Difference in {metric_labels[f"{metric}_text"]}')
    #axs[1,0].legend(bbox_to_anchor = (1.50, 0.6), loc='upper center')# loc='upper right', prop={'size': 9})
    if data_var == 'tas':
        axs[1,0].legend(loc='lower center',ncol=2, fancybox=True, shadow=True) # loc='upper right', bbox_to_anchor=(1.0, 1.05),
    else:
        axs[1,0].legend(loc='upper right',ncol=2, fancybox=True, shadow=True) # loc='upper right', bbox_to_anchor=(1.0, 1.05),

    # Plot inset for difference plot.
    if (xinset is not None and not plot_inset_in_separate_fig):
        xvalues_inset = [x for x in xvalues if x <= xinset]
        df_diff_means_inset = df_diff_means[df_diff_means[xaxis_key]<=xinset]
        means_inset = df_diff_means_inset[f'{metric}_modelmean_diff_membermean'].to_numpy()
        std_inset = df_diff_means_inset[f'{metric}_modelmean_diff_memberstd'].to_numpy()
        axs[1,1].plot(xvalues_inset, means_inset, marker='o', markersize=2.5, label=rf'$\Delta${metric_labels[f"{metric}_tex"]}')
        axs[1,1].plot(xvalues_inset, np.zeros(len(xvalues_inset)), linestyle='--', color='black', alpha=0.5, label=rf'$\Delta${metric_labels[f"{metric}_tex"]}$=0$')
        # Set labels
        axs[1,1].set_title(f'Difference inset for [0, {xinset-1}]')
        axs[1,1].set_ylim(axs[1,0].get_ylim())
        axs[1,1].set_xlabel(xlabel)

    # save plot
    plt.tight_layout()
    # shift subplots down:
    st.set_y(0.98)
    fig.subplots_adjust(top=0.85)

    if filepath_to_save is not None:
        Path(filepath_to_save).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(filepath_to_save)
        print('Saved plot at: ', filepath_to_save)

    # plt.show()
    plt.close()

    ##########
    # Create new figure to show xinset in plot with linear scale
    ##########
    if xinset is not None and plot_inset_in_separate_fig:
        fig, axs = plt.subplots(1,1,dpi=300)
        xvalues_inset = [x for x in xvalues if x <= xinset]
        df_diff_means_inset = df_diff_means[df_diff_means[xaxis_key]<=xinset]
        means_inset = df_diff_means_inset[f'{metric}_modelmean_diff_membermean'].to_numpy()
        std_inset = df_diff_means_inset[f'{metric}_modelmean_diff_memberstd'].to_numpy()
        axs.plot(xvalues_inset, means_inset, color='black', marker='o', markersize=2.5, label=rf'$\Delta${metric_labels[f"{metric}_tex"]}$\pm \sigma_k$')
        axs.fill_between(xvalues_inset, 
                    means_inset - std_inset,
                    means_inset + std_inset,
                    alpha=0.2,color='black')# label=f'{model_key}_std',)
        axs.plot(xvalues_inset, np.zeros(len(xvalues_inset)), linestyle='--', color='black', alpha=0.5, label=rf'$\Delta${metric_labels[f"{metric}_tex"]}$=0$')
        # Add linear fit 
        xstat = max(xvalues_inset) # Maximum xvalue for which we'd expect the ensemble to be representative.
        coef = np.polyfit(xvalues_inset,means_inset,1)
        print(f'{"Difference":<15} {metric:<20} {data_var:<5} from x=0 to {xstat} has a slope of {coef[0]:.8f}')
        poly1d_fn = np.poly1d(coef)
        axs.plot(xvalues_inset, poly1d_fn(xvalues_inset), '--', color='tab:blue', label='Linear fit')
        # axs.xaxis.set_major_locator(MaxNLocator(integer=True))
        # Set labels
        axs.set_ylim([-vmax, vmax])
        axs.set_xlabel(xlabel, fontsize=16)
        axs.set_ylabel(f'{metric_labels[f"{metric}_tex"]}(CNN-LSTM) \n - {metric_labels[f"{metric}_tex"]}(Pattern Scaling)', fontsize=16)
        axs.xaxis.set_ticks(np.arange(0, xstat + 1, 5))
        axs.set_xlim((0, xstat + 1))
        axs.legend(loc='lower right',ncol=2, fancybox=True, shadow=True,fontsize=16) # loc='upper right', bbox_to_anchor=(1.0, 1.05),
        # save plot
        axs.tick_params(axis='both', which='major', labelsize=16)
        plt.tight_layout()
        if filepath_to_save is not None:
            filepath_inset = Path(filepath_to_save).parent / (Path(filepath_to_save).stem + '_inset.png')
            plt.savefig(filepath_inset)
            print('Saved inset plot at: ', filepath_inset)
        # plt.show()
        plt.close()
        plt.rcParams.update(plt.rcParamsDefault)



def get_args():
    parser = argparse.ArgumentParser(description='Plotting routines')
    parser.add_argument('--data_var', type=str, default=None, help='Data variable that fit')
    parser.add_argument('--plot_m_member_subsets_experiment', action='store_true', default=False, help=
                        'Plots the results of the m_member_subsets experiment')
    parser.add_argument('--use_debug_paths', action='store_true', default=False, help=
                        'Use paths for debugging')
    return parser.parse_args()

if __name__ == '__main__':
    # Get command line arguments
    args = get_args()
    if args.data_var is None:
        data_vars = ['pr', 'tas'] # , 'vas','dtr','uas',  'pr90', 'huss', 'psl'] # 'pr90', 'dtr'
    else:
        data_vars = [args.data_var]

    if args.plot_m_member_subsets_experiment:
        for data_var in data_vars:
            for metric in ['Y_rmse_spatial', 'Y_rmse_global_np']:
                from emcli2.dataset.climatebench import DATA_VAR_LABELS
                plot_m_member_subsets_experiment(
                    data_var=data_var, 
                    metric=metric,
                    model_keys = ['cnn_lstm', 'pattern_scaling'],
                    experiment_keys = {
                        'cnn_lstm': 'm_member_subsets_with_m50_eval_on_all_spcmp_dwp_manyr',
                        'pattern_scaling': 'm_member_subsets_with_m50_replace_False_eval_on_all_manyr'
                        },
                    data_var_labels = DATA_VAR_LABELS,
                    xinset=20,
                    plot_inset_in_separate_fig=True
                    )
