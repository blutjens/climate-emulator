"""
Utils for data analysis and processing
"""

from tqdm import tqdm
import numpy as np
import torch
import xarray as xr
import pandas as pd
from pathlib import Path

def get_random_m_member_subsets(df, 
    verify_all_scenarios_have_same_member_coords=True,
    skip_subsets=1,
    max_number_of_members_per_subset=None,
    dir_m_members_subsets_csv=None,
    filename_m_members_subsets_csv=None,
    overwrite=False,
    equal_number_of_members_in_each_subset=50,
    number_of_subsets=5,
    member_mode='ascending',
    replace=True,
    idx_member_subset=None,
    seed=None,
    ):
    """
    Returns a list with member ids of random subsets 
    Args:
        verify_all_scenarios_have_same_member_coords bool: if True, will open every scenario's dataset and verify that the member coordinates are the same across every scenario
        skip_subsets int: Discretization of the number of members per subset; with skip_subsets=3, e.g., [1,4,7,...,49]
        max_number_of_members_per_subset int: Maximum number of members per subset. If None, the default is the number of members in ensemble
        replace bool: If True, members within one subset are sampled with replacement, i.e., they can appear multiple times within one subset.
        seed int: If seed is not None and member_mode==ascending-repeat, will reset the numpy.random.seed to the given seed on each function call.
    Returns:
        m_member_subsets List(List(str)): List with member IDs for each m-member subset
        member_ids List(str): List with all member IDs
        idcs_member_subsets List(int): List with indices to iterate over all m-member subsets. Used, e.g., used during sweep.
    """
    # Load
    if dir_m_members_subsets_csv is not None and filename_m_members_subsets_csv is not None:
        path_m_members_subsets_csv = str(Path(dir_m_members_subsets_csv) / Path(filename_m_members_subsets_csv))
    else:
        path_m_members_subsets_csv = None
    
    member_ids = None
    if Path(path_m_members_subsets_csv).exists() and not overwrite:
        print('M member subsets already exists. Loading from csv file.')
        csv_file = Path(path_m_members_subsets_csv) 
        df_subsets = pd.read_csv(csv_file, header=None)
        # todo: convert df_m_member_subsets into List(List(str))
        number_of_members_per_subset = df_subsets[0].values
        m_member_subsets = df_subsets[1].values.tolist()
        m_member_subsets = [subset.strip('][').split(', ') for subset in m_member_subsets]
        m_member_subsets = [[s.strip('\'') for s in sub] for sub in m_member_subsets]
        if max_number_of_members_per_subset is not None:
            number_of_members_per_subset_desired = np.arange(1, max_number_of_members_per_subset+1)[::skip_subsets]
            assert np.all(number_of_members_per_subset_desired == number_of_members_per_subset), f'Number of members per subset in csv file {number_of_members_per_subset} '\
                f'differs from desired number of members per subset {number_of_members_per_subset_desired}. Activate args.overwrite or specify different max_number_of_members_per_subset'\
                'and skip_subsets'
        if len(m_member_subsets) == 50:
            member_ids = m_member_subsets[-1]
        else:
            print('Last member of m_member_subsets does not contain all 50 members; returning member_ids = None')
    else:
        print('Calculating member subsets.')
        # Get list of member IDs, i.e., realization IDs
        scenarios = df.scenario.values
        for scenario in scenarios:
            # Open dataset
            df_scenario = df[df.scenario == scenario]
            ensemble_xr = xr.open_dataset(df_scenario.filepath.values[0])

            # todo: delete this once ensemble.nc are recomputed with member dimension. Im creating dummy coordinates
            if 'member' not in ensemble_xr.coords:
                n_realizations = ensemble_xr.dims['member']
                member_ids = [f'r{r:1d}i1f1p1' for r in np.arange(1, n_realizations+1)]
                ensemble_xr.coords['member'] = member_ids

            # Extract member IDs from dataset
            if member_ids is None:
                member_ids = ensemble_xr.coords['member']
            elif verify_all_scenarios_have_same_member_coords:
                    # Verify that all scenarios have the same ensemble member coordinates             
                    assert np.all(member_ids == ensemble_xr.coords['member']), f'Number of ensemble members in '\
                        f'scenario {scenario} differs from other scenarios which have {len(member_ids)} members'

            # Close dataset
            ensemble_xr.close(); del ensemble_xr
            if not verify_all_scenarios_have_same_member_coords:
                # End here without opening any other scenarios; assuming that every scenario has the same member coordinates
                break
        
        # Define there should be an increasing number or the same number of members per subset
        if 'ascending' in member_mode:
            if max_number_of_members_per_subset is None:
                max_number_of_members_per_subset = len(member_ids)
            number_of_members_per_subset = np.arange(1, max_number_of_members_per_subset+1)[::skip_subsets]
        elif member_mode=='equal':
            number_of_members_per_subset = equal_number_of_members_in_each_subset * np.ones(number_of_subsets, dtype=int)
        else:
            raise ValueError(f'Unknown member_mode {member_mode}')

        # Reset numpy seed such that the random subsets are the same across models
        if member_mode == 'ascending-repeat' and seed is not None:
            np.random.seed(seed)

        # Get random selection of members to fill each subset
        m_member_subsets = len(number_of_members_per_subset)*[None]
        for m_idx, number_of_members in enumerate(number_of_members_per_subset):
            m_member_subsets[m_idx] = np.random.choice(member_ids, number_of_members, replace=replace).tolist()

        # Save information as csv
        if path_m_members_subsets_csv is not None:
            series = pd.Series(m_member_subsets, [len(me) for me in m_member_subsets])
            Path(path_m_members_subsets_csv).parent.mkdir(parents=True, exist_ok=True) # Create parent directory, if not exist
            series.to_csv(path_m_members_subsets_csv, header=False, index=True)
        print('Successfully calculated member subsets.')

    # Train on specified member subset, used, e.g., during sweep
    if idx_member_subset is not None:
        m_member_subsets = [m_member_subsets[idx_member_subset]]
        idcs_member_subsets = [idx_member_subset]
    else:
        idcs_member_subsets = np.arange(len(m_member_subsets),dtype=int).tolist()

    return m_member_subsets, member_ids, idcs_member_subsets

def calc_statistics(data=None, var_key='tas'):
    """
    Calculates min, max, mean, std dev over all scenarios in dataset
    todo:
        Currently calculates the standard deviation over all cmip scenarios by just taking the average across scenarios
        Currently calculates the spatial mean by equally weighting every grid point
    Args:
        data list(n_scenarios * xarray.DataArray([n_time, n_lon, n_lat]): List of xarray DataArray that contain, e.g., global CO2 emissions over time or local surface temperatures over time
        var_key string: Key to variable in data for which the statistics should be calculated 
    """
    min = 1.e10
    max = -1.e10
    mean = 0.
    std  = 0.
    n_scenarios = len(data)
    for scenario_id in range(n_scenarios):
        min = np.min((min, data[scenario_id][var_key].data.min()))
        max = np.max((max, data[scenario_id][var_key].data.max()))
        mean += data[scenario_id][var_key].data.mean()
        std += data[scenario_id][var_key].data.std()

    mean /= float(n_scenarios)
    std /= float(n_scenarios)
    return min, max, mean, std

def create_histogram(data=None, var_key='tas', n_bins=2, normalize=True):
    """ Creates a histogram of all values in data
    Args:
        data list(xarray.DataArray([n_time, n_lon, n_lat])
        var_key string: Key to variable in data for which the histogram should be calculated 
        n_bins int: number of bins in the histogram
        normalize bool: if True will normalize the histogram to get relative frequency
    """
    hist = torch.zeros(n_bins)
    n_scenarios = len(data)
    
    min,max,_,_ = calc_statistics(data, var_key=var_key)

    # Loop over all scenarios in the data
    for scenario_id in tqdm(range(n_scenarios)):
        # Get tas (n_time, n_lon, n_lat) where time is treated as batch dimension
        tas = torch.from_numpy(data[scenario_id][var_key].data)

        # Calculate the histogram of each image using torch.histogram function [^2^][2]
        h, bin_values = torch.histogram(tas, bins=n_bins, range=(min,max))

        # Add the histogram values to the hist array
        hist += h
    # Normalize the hist array to get the relative frequency of each pixel value
    if normalize:
        hist = hist / hist.sum()

    hist = hist.numpy()
    bin_values = bin_values.numpy()
    
    return hist, bin_values
