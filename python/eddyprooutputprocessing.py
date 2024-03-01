from pathlib import Path
from datetime import datetime

import pandas as pd
from tqdm import tqdm
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

__author__ = "Alexander Fox"
__copyright__ = "Copyright 2024"
__license__ = "GPL3"
__email__ = "afox18@uwyo.edu"

"""
TODO
Distinguish between primary keys that are dates and primary keys that arent. 
E.g. BB-NF_17m_202301010000_wind1 and BB-NF_17m_202301010000_wind2 have differe primary keys, 
but BB-NF_17m_202301010000_wind1 and BB-NF_17m_202201010000_wind1 have the same primary key

Allow the user to select a date range. THis is important for collecting cospectra, since all the cospectra together can quickly exceed a tractable size.

Add stats parsing

Improve documentation

Add Xarray support
"""

def _build_file_list(directory, parallel, globstring) -> list:
    """helper function to build a filelist from a directory."""
    if parallel:
        dirs = list(Path(directory).glob('*'))
        files = []
        for d in dirs:
            # only grab latest file in each environment
            files += [sorted(list(d.glob(globstring)))[-1]]
    else:
        # only grab latest file in this environment
        files = [sorted(list(Path(directory).glob(globstring)))[-1]]
    return files

def read_full_output(directory, parallel:bool=False, na_values=-9999) -> tuple[pd.DataFrame, list]:
    """gets the full_output file from an output directory and returns the data and the units
    Important note: if multiple full_output files are found in a single output directory, this method will only return the most recent one!
    This method will add a new column called "primary key" that is the parent directory of each file read.

    Parameters
    ----------
    directory: the output directory. If parallel=true, then this is the out_parent directory. Otherwise, this is just the run directory.
    parallel: whether this output data was generated from a set of parallel runs. If True, then this function will combine multiple full_output files and concatenate them by time. If True, provide out_parent as d. Default False.
    na_values: na value representation in the output file. Default -9999

    Returns
    -------
    df: the full_output dataframe
    units: list of str representing units for each column
    """

    files = _build_file_list(directory, parallel, '*_full_output_*.csv')

    df = [pd.DataFrame()]*len(files)
    # progress bar
    if len(files) > 10:
        pbar = tqdm(enumerate(files), total=len(files))
    else: pbar = enumerate(files)
    for i, f in pbar:
        idf = pd.read_csv(f, parse_dates=[['date', 'time']], skiprows=[0, 2], na_values=na_values)
        idf['primary_key'] = f.parent.name
        df[i] = idf
    df = pd.concat(df).sort_values(['primary_key', 'date_time']).set_index('date_time')
    
    with open(files[0], 'r') as f:
        f.readline()
        columns = f.readline()[:-1].split(',')
        units = f.readline()[:-1].split(',')
        units = [u.strip('[]') for u in units]
        units = {c:u for c, u in zip(columns, units)}
    return df, units

def read_fluxnet(directory, parallel:bool=False, na_values=-9999) -> pd.DataFrame:
    """gets the fluxnet file from an output directory and returns the data and the units
    Important note: if multiple full_output files are found in a single output directory, this method will only return the most recent one!
    This method will add a new column called "primary key" that is the parent directory of each file read.
    This method will add a new column called 'date_time' that is the TIMESTAMP_START column converted to a datetime object.

    Parameters
    ----------
    directory: the output directory. If parallel=true, then this is the out_parent directory. Otherwise, this is just the run directory.
    parallel: whether this output data was generated from a set of parallel runs. If True, then this function will combine multiple full_output files and concatenate them by time. If True, provide out_parent as d. Default False.
    na_values: na value representation in the output file. Default -9999

    Returns
    -------
    df: the fluxnet file
    """
    files = _build_file_list(directory, parallel, '*_fluxnet_*.csv')

    df = [pd.DataFrame()]*len(files)
    # progress bar
    if len(files) > 10:
        pbar = tqdm(enumerate(files), total=len(files))
    else: pbar = enumerate(files)
    for i, f in pbar:
        idf = pd.read_csv(f, na_values=na_values)
        idf['date_time'] = pd.to_datetime(idf.TIMESTAMP_START, format=r'%Y%m%d%H%M')
        idf['primary_key'] = f.parent.name
        df[i] = idf
    df = pd.concat(df).sort_values(['primary_key', 'date_time']).set_index('date_time')
    return df

def read_biomet(directory, parallel:bool=False, na_values=-9999) -> tuple[pd.DataFrame, list]:
    """gets the biomet file from an output directory and returns the data and the units
    Important note: if multiple full_output files are found in a single output directory, this method will only return the most recent one!
    This method will add a new column called "primary key" that is the parent directory of each file read.

    Parameters
    ----------
    directory: the output directory. If parallel=true, then this is the out_parent directory. Otherwise, this is just the run directory.
    parallel: whether this output data was generated from a set of parallel runs. If True, then this function will combine multiple full_output files and concatenate them by time. If True, provide out_parent as d. Default False.
    na_values: na value representation in the output file. Default -9999

    Returns
    -------
    df: the full_output dataframe
    units: list of str representing units for each column
    """

    files = _build_file_list(directory, parallel, '*_biomet_*.csv')

    df = [pd.DataFrame()]*len(files)
    # progress bar
    if len(files) > 10:
        pbar = tqdm(enumerate(files), total=len(files))
    else: pbar = enumerate(files)
    for i, f in pbar:
        idf = pd.read_csv(f, parse_dates=[['date', 'time']], skiprows=[1], na_values=na_values)
        idf['primary_key'] = f.parent.name
        df[i] = idf
    df = pd.concat(df).sort_values(['primary_key', 'date_time']).set_index('date_time')
    
    with open(files[0], 'r') as f:
        columns = f.readline()[:-1].split(',')
        units = f.readline()[:-1].split(',')
        units = [u.strip('[]') for u in units]
        units = {c:u for c, u in zip(columns, units)}
    return df, units

def read_single_planar_fit(fn) -> tuple[dict, pd.DataFrame, np.array]:
    """read a single planar fit file
    
    Parameters
    ----------
    fn: path to planar fit file
    
    Returns
    -------
    params: dict of fix parameter settings
    pf_coefs: dataframe containing sector information, including span, linear coefficients, and sector numerosity
    matrices: planar fit matrices for each sector, dims (sector, row, column)
    """

    def read_data_line(ln=None, f=None):
        if f is not None:
            ln = f.readline()[-1]
        return [col for col in ln.split(' ') if col != '']


    with open(fn, 'r') as f:
        lines = f.readlines()
    
    i = 0
    # input parameters
    params = dict(
        n_sectors = int(lines[1][:-1].split(' ')[-1]),
        min_num_per_sector = int(lines[2][:-1].split(' ')[-1]),
        max_avg_w = float(lines[3][:-1].split(' ')[-1]),
        min_avg_u = float(lines[4][:-1].split(' ')[-1]),
        start = datetime.strptime(lines[5][:-1].split(' ')[-1], r'%Y-%m-%d'),
        end = datetime.strptime(lines[6][:-1].split(' ')[-1], r'%Y-%m-%d'),
    )
    
    # planar fit coefficients
    header = ['WindSector', 'Start', 'End', 'B0', 'B1', 'B2', 'numerosity']
    sectors = []
    i = 10
    while True:
        ln = lines[i][:-1]
        if len(ln.strip()) > 0:
            sectors.append(read_data_line(ln=ln))
            i += 1
        else: break
    i += 2
    
    
    # rotation matrices
    n_sectors = len(sectors)
    matrices = []
    k = 0
    for j in range(i, i + n_sectors*4, 4):
        # get numerosity
        sectors[k].append(lines[j][:-1].split(' ')[-1])
        # get matrix
        row_1 = [float(x) if x != '-9999' else np.nan for x in read_data_line(ln=lines[j + 1][:-1])]
        row_2 = [float(x) if x != '-9999' else np.nan for x in read_data_line(ln=lines[j + 2][:-1])]
        row_3 = [float(x) if x != '-9999' else np.nan for x in read_data_line(ln=lines[j + 3][:-1])]
        matrix = [row_1, row_2, row_3]
        matrices.append(matrix)
        k += 1
    matrices = np.asarray(matrices)

    # convert coefficients to dataframe
    pf_coefs = pd.DataFrame(columns=header, data=sectors)
    pf_coefs['WindSector'] = pf_coefs['WindSector'].astype(int)
    pf_coefs['Start'] = pf_coefs['Start'].str[:-1].astype(float)
    pf_coefs['End'] = pf_coefs['End'].astype(float)
    pf_coefs['End'] = pf_coefs['End'].astype(float)
    pf_coefs['B0'] = pf_coefs['B0'].astype(float)
    pf_coefs['B1'] = pf_coefs['B1'].astype(float)
    pf_coefs['B2'] = pf_coefs['B2'].astype(float)
    pf_coefs['numerosity'] = pf_coefs['numerosity'].astype(int)
    pf_coefs = pf_coefs.where(pf_coefs != -9999)

    pf_coefs = (
        pd.wide_to_long(
            pf_coefs, 
            stubnames='B', 
            i='WindSector', 
            j='Coefficient'
        )
        .reset_index()
        .rename(columns=dict(B='value'))
    )

    return params, pf_coefs, matrices

def read_single_binned_cospectrum_file(fn) -> pd.DataFrame:
    """read a single cospectrum file
    Parameters
    ----------
    fn: cospectrum file
    """
    # fixed params
    with open(fn, 'r') as f:
        f.readline(); f.readline(); f.readline(); f.readline(); f.readline()
        aq_freq = float(f.readline()[:-1].split('_=_')[-1].strip())
        hgt_above_d = float(f.readline()[:-1].split('_=_')[-1].strip())
        ws = float(f.readline()[:-1].split('_=_')[-1].strip())
        avg_int = float(f.readline()[:-1].split('_=_')[-1].strip())
        bins = int(f.readline()[:-1].split('_=_')[-1].strip())
        win_type = f.readline()[:-1].split('_=_')[-1].strip()
    
    df = pd.read_csv(fn, skiprows=11, na_values=-9999)
    df[[
        'acquisition_frequency_[Hz]', 
        'measuring_height_(z-d)_[m]', 
        'wind_speed_[m+1s-1]', 
        'averaging_interval_[min]', 
        'number_of_bins', 
        'tapering_window'
    ]] = [aq_freq, hgt_above_d, ws, avg_int, bins, win_type]

    df.columns = [c.strip() for c in df.columns]
    df = df.dropna(subset=['natural_frequency', 'normalized_frequency'], how='all')
    
    
    return df

def read_binned_cospectra(directory, parallel=False, return_xarray=False, return_pandas=True, subset=None)->tuple:
    """
    read all cospectrum files from a single output directory
    
    Parameters
    ----------
    directory: the output directory that contains the eddypro_binned_cospectra directory
    subset: how to filter dates/times. Provide a list of datetimes.
    """
    if not parallel:
        files = sorted(Path(directory).glob('eddypro_binned_cospectra/*_binned_cospectra_*.csv'))
        print(files)
    else:
        files = sorted(Path(directory).glob('*/eddypro_binned_cospectra/*_binned_cospectra_*.csv'))
    if len(files) > 10_000:
        print(f'WARNING: {len(files)} detected. Outputs could exceed 100MB unless date filtering is applied.')
    if subset is not None:
        dates = [d.strftime(r'%Y%m%d-%H%M') for d in subset]
        files = [f for f in files if f.name.split('_')[0] in dates]
        print(f'Reading {len(files)} detected after date filtering.')
        print('test')
    df = [pd.DataFrame()]*len(files)
    for i, fn in tqdm(enumerate(files), total=len(files)):
        timestamp = datetime.strptime(fn.name.split('_')[0], r'%Y%m%d-%H%M')
        idf = read_single_binned_cospectrum_file(fn)
        idf['date_time'] = timestamp
        idf['run_id'] = fn.parent.name
        df[i] = idf

    df = pd.concat(df)

    if return_xarray:
        ds = xr.Dataset.from_dataframe(df.set_index(['date_time', 'natural_frequency', 'run_id']))
    if return_pandas and return_xarray: return df, ds
    elif return_pandas: return df, None
    else: return None, ds

def read_binned_ogives(directory, parallel=False, return_xarray=False, return_pandas=True, subset=None)->tuple:
    """
    read all cospectrum files from a single output directory
    
    Parameters
    ----------
    directory: the output directory that contains the eddypro_binned_ogives directory
    subet: list of datetimes to specify exactly which files to pull out"""
    if not parallel:
        files = sorted(Path(directory).glob('eddypro_binned_cospectra/*_binned_ogives_*.csv'))
        print(files)
    else:
        files = sorted(Path(directory).glob('*/eddypro_binned_cospectra/*_binned_ogives_*.csv'))
    if len(files) > 10_000:
        print(f'WARNING: {len(files)} detected. Outputs could exceed 100MB unless date filtering is applied.')
    if subset is not None:
        dates = [d.strftime(r'%Y%m%d-%H%M') for d in subset]
        files = [f for f in files if f.name.split('_')[0] in dates]
        print(f'Reading {len(files)} detected after date filtering.')
        print('test')
    df = [pd.DataFrame()]*len(files)
    for i, fn in tqdm(enumerate(files), total=len(files)):
        timestamp = datetime.strptime(fn.name.split('_')[0], r'%Y%m%d-%H%M')
        idf = read_single_binned_cospectrum_file(fn)
        idf['date_time'] = timestamp
        idf['run_id'] = fn.parent.name
        df[i] = idf

    df = pd.concat(df)

    if return_xarray:
        ds = xr.Dataset.from_dataframe(df.set_index(['date_time', 'natural_frequency', 'run_id']))
    if return_pandas and return_xarray: return df, ds
    elif return_pandas: return df, None
    else: return None, ds


    
if __name__ == '__main__':
    directory = '/Users/alex/Documents/Data/Platinum_EC/ExpectedOutputs/Chimney/17m/dev/BB-NF-17m-202106190000'
    # full_output, units = read_full_output(directory)
    # fluxnet = read_fluxnet(directory)
    # biomet = read_biomet(directory)
    # params, pf_coefs, matrices = read_single_planar_fit('/Users/alex/Documents/Data/Platinum_EC/ExpectedOutputs/Chimney/17m/dev/BB-NF-17m-202106190000/eddypro_BB-NF-17m-202106190000_planar_fit_2024-01-16T145152_adv.txt')
    co_df, co_ds = read_binned_cospectra(directory, return_xarray=True)
    og_df, og_ds = read_binned_ogives(directory, return_xarray=True)
    og_ds['og(w_h2o)'].plot(vmin=0, vmax=1)
    plt.xscale('log')
    plt.show()
    
