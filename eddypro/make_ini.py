import configparser
from pathlib import Path
import multiprocessing
from math import ceil, floor
from datetime import datetime
from os import PathLike

import numpy as np
import pandas as pd


def modify_eddypro(
        ref_fn: str | PathLike[str], 
        out_fn: str | PathLike[str], 
        FluxCorrection_SpectralAnalysis_General: dict = {},
        Project: dict = {},
        RawProcess_BiometMeasurements: dict = {},
        RawProcess_General: dict = {},
        RawProcess_ParameterSettings: dict = {},
        RawProcess_Settings: dict = {},
        RawProcess_Tests: dict = {},
        RawProcess_TiltCorrection_Settings: dict = {},
        RawProcess_TimelagOptimization_Settings: dict = {},
        RawProcess_WindDirectionFilter: dict = {},
) -> None:
    """modifies a .eddypro config file
    ini_fn: reference .eddypro file to modify
    out_fn: path to write modified .eddypro files to
    all other arguments: a dictionary-like object containing option-value pairs to modify within a given section. If an empty dict (default), then that section will not be modified.
    """

    ini = configparser.ConfigParser()
    ini.read(ref_fn)

    for k, v in FluxCorrection_SpectralAnalysis_General.items():
        ini.set(
            section='FluxCorrection_SpectralAnalysis_General',
            option=k,
            value=str(v)
        )
    for k, v in Project.items():
        ini.set(
            section='Project',
            option=k,
            value=str(v)
        )
    for k, v in RawProcess_BiometMeasurements.items():
        ini.set(
            section='RawProcess_BiometMeasurements',
            option=k,
            value=str(v)
        )
    for k, v in RawProcess_General.items():
        ini.set(
            section='RawProcess_General',
            option=k,
            value=str(v)
        )
    for k, v in RawProcess_ParameterSettings.items():
        ini.set(
            section='RawProcess_ParameterSettings',
            option=k,
            value=str(v)
        )
    for k, v in RawProcess_Settings.items():
        ini.set(
            section='RawProcess_Settings',
            option=k,
            value=str(v)
        )
    for k, v in RawProcess_Tests.items():
        ini.set(
            section='RawProcess_Tests',
            option=k,
            value=str(v)
        )
    for k, v in RawProcess_TiltCorrection_Settings.items():
        ini.set(
            section='RawProcess_TiltCorrection_Settings',
            option=k,
            value=str(v)
        )
    for k, v in RawProcess_TimelagOptimization_Settings.items():
        ini.set(
            section='RawProcess_TimelagOptimization_Settings',
            option=k,
            value=str(v)
        )
    for k, v in RawProcess_WindDirectionFilter.items():
        ini.set(
            section='RawProcess_WindDirectionFilter',
            option=k,
            value=str(v)
        )

    with open(out_fn, 'w') as configfile:
        configfile.write(';EDDYPRO_PROCESSING\n')  # header line
        ini.write(fp=configfile, space_around_delimiters=False)

    return

def configure_parallel_workers(
    ref_fn: str | PathLike[str], 
    metadata_fn: str | PathLike[str] | None = None,
    num_workers: int | None = None,
    file_duration: int | None = None,
) -> None:
    """
    given a .eddypro file, split it up into num_workers separate .eddypro files, each handling a separate time chunk.
    all .eddypro files will be identical except in their project IDs, file names, and start/end dates.
    
    Note that some processing methods are not compatible "out-of-the-box" with paralle processing: some methods like the planar fit correction and ensemble spectral corrections will need the results from a previous, longer-term eddypro run to function effectively.

    ref_fn: the base .eddypro file to modify when creating paralle workers.
    metadata_fn: path to a static .metadata file for this project. Must be provided if file_duration is None.
    num_workers: the number of parallel processes to configure. If None (default), then processing is split up according to the number of available processors on the machine.
    file_duration: how many minutes long each file is. If None (Default), then that information will be gleaned from the metadata file.
    """

    # get file duration
    if file_duration is None:
        assert metadata_fn is not None, 'metadata_fn must be provided'
        metadata = configparser.ConfigParser()
        metadata.read(metadata_fn)
        file_duration = int(metadata['Timing']['file_duration'])

    if num_workers is None:
        num_workers = max(multiprocessing.cpu_count() - 1, 1)

    ini = configparser.ConfigParser()
    ini.read(ref_fn)

    # split up file processing dates
    start = str(datetime.strptime(
        f"{ini['Project']['pr_start_date']} {ini['Project']['pr_start_time']}", 
        r'%Y-%m-%d %H:%M'
    ))
    end = str(datetime.strptime(
        f"{ini['Project']['pr_end_date']} {ini['Project']['pr_end_time']}" , 
        r'%Y-%m-%d %H:%M'
    ))

    n_files = len(pd.date_range(start, end, freq=f'{file_duration}min'))
    job_size = ceil(file_duration*n_files/num_workers)
    job_size = f'{int(ceil(job_size/file_duration)*file_duration)}min'

    job_starts = pd.date_range('2020-06-21 00:00', '2020-07-22 00:00', freq=job_size)
    job_ends = job_starts + pd.Timedelta(job_size) - pd.Timedelta(file_duration)  # dates are inclusive, so subtract 30min for file duration
    job_start_dates = job_starts.strftime(date_format=r'%Y-%m-%d')
    job_start_times = job_starts.strftime(date_format=r'%H:%M')
    job_end_dates = job_ends.strftime(date_format=r'%Y-%m-%d')
    job_end_times = job_ends.strftime(date_format=r'%H:%M')

    # give each project a unique id and file name
    project_ids = [f'worker{start}' for start in job_starts.strftime(date_format=r"%Y%m%d%H%M")]
    ini_fns = [ref_fn.parent / f'{project_id}.eddypro' for project_id in project_ids]

    # modify fns
    for i, fn in enumerate(ini_fns):
        ini['Project']['file_name'] = str(fn)
        ini['Project']['pr_start_date'] = job_start_dates[i]
        ini['Project']['pr_end_date'] = job_end_dates[i]
        ini['Project']['pr_start_time'] = job_start_times[i]
        ini['Project']['pr_end_time'] = job_end_times[i]
        ini['Project']['project_id'] = project_ids[i]

        with open(fn, 'w') as configfile:
            configfile.write(';EDDYPRO_PROCESSING\n')  # header line
            ini.write(fp=configfile, space_around_delimiters=False)

    return



if __name__ == '__main__':
    project_dir = Path('/Users/alex/Documents/Work/UWyo/Research/Flux Pipeline Project/Eddypro-ec-testing/eddypro')
    reference_ini_fn = project_dir/'ini/template.eddypro'
    base_eddypro_fn = reference_ini_fn.parent / 'base.eddypro'
    metadata_fn = project_dir / 'ini/lcreek.metadata'
    out_path = project_dir/'output'
    data_dir = Path('/Users/alex/Documents/Data/Platinum_EC/LostCreek')
    
    # modify template
    modify_eddypro(
        ref_fn=reference_ini_fn,
        out_fn=base_eddypro_fn,
        FluxCorrection_SpectralAnalysis_General=dict(ex_file=''),
        Project=dict(last_change_date='', proj_file=metadata_fn, out_path=out_path, biom_dir=data_dir/'biomet'),
        RawProcess_General=dict(data_path=data_dir/'raw_data')
    )

    configure_parallel_workers(
        ref_fn=base_eddypro_fn,
        metadata_fn=metadata_fn,
    )