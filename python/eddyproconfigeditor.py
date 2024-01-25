"""
TODO
Settings to add
** = Priority: settings that need to be automated.
E.g. raw file format probably doesn't need to be automated

Project creation
----------------------------------------
* Project name
* Raw file format
* Metadata file location
* Dynamic metadata file location/presence
** Biomet data

Basic Settings
----------------------------------------
* Raw data directory
** DONE Processing dates
* Raw file name format
** DONE (sort of, in to_eddypro methods) output directory
* DONE missing samples allowance
* DONE flux averaging interval
* DONE north reference
* Items for flux computation:
    * master anemometer/diagnostics/fast-temperature-reading
    * CO2/H2O/CH4/Gas4/CelTemp/CellPress/etc
    ** Flags
    ** Wind filter

Advanced Settings
----------------------------------------
    Processing Options
    ------------------------------------
    * DONE WS offsets
    * Fix w boost bug
    * AoA correction
    ** Axis rotations for tilt correction
      * Need to remove "excess" sectors. Ie. if we provide 5 sectors, but 10 exist, we need to remove the extra ones.
    ** DONE turbulent fluctuation
    ** DONE time lag compensations
    ** DONE WPL Corrections
        ** DONE Compensate density fluctuations
        ** DONE Burba correction for 7500
    * Quality check
    * Foodprint estimation

    Statistical Analysis
    ------------------------------------
    ** DONE VM97
        ** DONE Spike count/removal
        ** DONE Ampl. res
        ** DONE Dropouts
        ** DONE Abs lims
        ** DONE Skew + Kurt
        ** DONE Discont.
        ** DONE Time lags
        ** DONE AoA
        ** DONE Steadiness of Hor. Wind
    ** DONE Random uncertainty estimates

    Spectral Corrections
    ------------------------------------
    ** Spectra and Cospectra Calculation
    ** Removal of HF noise
    ** Spectra/Cospectra QA/QC
    * Quality test filtering
    ** Spectral correction options
        ** Low-freq
        ** High-freq
        ** Assessment
        ** Fratini et al 2012

    Output Files
    ------------------------------------
    * Results files
    * Fluxnet output settings
    * Spectra and cospectra output
    * Processed raw data outputs

TODO: edge cases
    * when setting timespan-sensitive settings, check that the timespan is appropriate.
    e.g. when setting planar fit, check that the overlap between the project timespan and the planar fit window is greater than 2 weeks
    * when writing output, make sure that the program will be able to run,
    * when writing output, check that dates are consistent:
        * if time-dependent settings are selected, such a start='project' for 
    * PF data does NOT have to overlap with project data, but must be a valid range of raw data.
    * Spectral data DOES have to sufficiently overlap with project data.
    * If a time is empty in the .eddypro file, it can be interpreted as 00:00

NOTE: Eddypro's order of operations

Calculations on raw, unprocessed data (not applied yet)
1. Perform Timelag autoopt calculation if requested
2. Perform planar fit calculation if requested (but do not apply yet)

Apply corrections to raw data:
1. Apply calibration events from metadata (but do not apply yet)
2. Apply statistical tests
    a. Absolute limits
    b. Despike
    c. Dropouts
    d. Amplitude resolution
    e. Skewness & Kurtosis
    f. Discontinuities
    g. Time lag
    h. AoA
    i. Non-steady horizontal wind
3. Cross-wind correction
4. AoA Correction
5. APPLY Tilt correction: compute DR/TR if requested, otherwise use planar fit file generated or loaded in (2)
6. APPLY Time lag compensation: compute CovMax if requested, otherwise use the to file generated or loaded in (1)
7. Detrend/average data

Calculate spectral information (do not apply yet)
1. Compute burba corrections
2. Compute low-pass correction factors
3. Compute high-pass corrections factors
4. Compute level 1 fluxes
5. Compute level 2 and 3 fluxes
6. Compute footprint
7. Compute storage
8. Compute foken turbulence tests

Flux computation

Note: If Ibrom, Fratini, or Horst low-pass filtering is requested, that will be performed in eddypro_fcc, in a final step.
"""

import configparser
from pathlib import Path
from typing import Literal
import datetime
from os import PathLike
import os
from collections.abc import Sequence
import multiprocessing
from math import ceil
import warnings
import tempfile
from copy import deepcopy
from time import perf_counter as timer
from collections.abc import Sequence
import shutil

from pandas import date_range, Timedelta, DataFrame

__author__ = "Alexander Fox"
__copyright__ = "Copyright 2023"
__license__ = "GPL3"
__email__ = "afox18@uwyo.edu"


def or_isinstance(object, *types):
    """helper function to chain together multiple isinstance arguments
    e.g. or_isinstance(a, float, int) does the same as (isinstance(a, float) or isinstance(a, int))"""
    for t in types:
        if isinstance(object, t):
            return True
    return False

def in_range(v, interval):
    """helper function to determine if a value is in some interval.
    intervals are specified using a string:
    (a, b) for an open interval
    [a, b] for a closed interval
    [a, b) for a right-open interval
    etc.

    use inf as one of the interval bounds to specify no bound in that direction, but I don't know why you wouldn't just use > at that point"""

    assert interval.strip()[0] in ['(', '['] and interval.strip()[-1] in [')', ']'], 'interval must be a string of the form (a, b), [a, b], (a, b], or [a, b)'
    # remove whitespace
    interval = interval.strip()

    # extract the boundary conditions
    lower_bc = interval[0]
    upper_bc = interval[-1]

    # extract the bounds
    interval = [float(bound.strip())
                for bound in interval[1:-1].strip().split(',')]
    lower, upper = interval
    if lower == float('inf'):
        lower *= -1

    bounds_satisfied = 0
    if lower_bc == '(':
        bounds_satisfied += (lower < v)
    else:
        bounds_satisfied += (lower <= v)

    if upper_bc == ')':
        bounds_satisfied += (v < upper)
    else:
        bounds_satisfied += (v <= upper)

    return bounds_satisfied == 2

def compare_configs(df1: DataFrame, df2: DataFrame) -> DataFrame:
    """compare differences between two configs
    
    Parameters
    ----------
    df1, df2: pandas.DataFrame objects output by EddyproConfigEditor.to_pandas()

    Returns
    -------
    pandas.DataFrame containing lines that differed between df1 and df2
    """
    df1_new = df1.loc[df1['Value'] != df2['Value'],
                      ['Section', 'Option', 'Value']]
    df2_new = df2.loc[df1['Value'] != df2['Value'],
                      ['Section', 'Option', 'Value']]
    name1 = df1['Name'].values[0]
    name2 = df2['Name'].values[0]
    df_compare = (
        df1_new
        .merge(
            df2_new, 
            on=['Section', 'Option'], 
            suffixes=['_' + name1, '_' + name2])
        .sort_values(['Section', 'Option']))
    return df_compare

def compute_date_overlap(
        interval_1: Sequence[str, str] | Sequence[datetime.datetime, datetime.datetime], 
        interval_2: Sequence[str, str] | Sequence[datetime.datetime, datetime.datetime]) -> datetime.timedelta:
    """given two time intervals of strings or datetime objects, compute their overlap and report them as a timdelta object
    Strings must be of the form YYYY-mm-dd HH:MM"""

    # assure inputs conform to standards
    assert isinstance(interval_1, Sequence), 'intervals must be sequences of strings or datetimes'
    assert isinstance(interval_2, Sequence), 'intervals must be sequences of strings or datetimes'
    assert len(interval_1) == 2, 'intervals must be length 2'
    assert len(interval_2) == 2, 'intervals must be length 2'
    interval_1 = list(interval_1)
    interval_2 = list(interval_2)
    for i in range(2):
        assert or_isinstance(interval_1[i], str, datetime.datetime), 'inputs must be strings of format YYYY-mm-dd HH:MM or datetime.datetime objects'
        assert or_isinstance(interval_2[i], str, datetime.datetime), 'inputs must be strings of format YYYY-mm-dd HH:MM or datetime.datetime objects'
        if isinstance(interval_1[i], str):
            assert len(interval_1[i].strip()) == 16, 'inputs must be strings of format YYYY-mm-dd HH:MM'
        if isinstance(interval_2[i], str):
            assert len(interval_2[i].strip()) == 16, 'inputs must be strings of format YYYY-mm-dd HH:MM'
    
    # convert to datetime objects
    for i in range(2):
        if isinstance(interval_1[i], str):
            interval_1[i] = datetime.datetime.strptime(interval_1[i].strip(), r'%Y-%m-%d %H:%M')
        if isinstance(interval_2[i], str):
            interval_2[i] = datetime.datetime.strptime(interval_2[i].strip(), r'%Y-%m-%d %H:%M')

    # compute overlap
    start = max(interval_1[0], interval_2[0])
    end = min(interval_1[1], interval_2[1])
    overlap = end - start

    return overlap



class EddyproConfigEditor(configparser.ConfigParser):
    '''
    Class designed to mimic the functionality of the eddypro 7 GUI, built as a child class of configparser.ConfigParser.

    Parameters
    ----------
    reference_ini: path to a .eddypro file to modify 

    Variables
    ---------
    history: a dictionary to keep track of changes made to the config file. Structured as follow:
        {pane:
            {setting:
                [
                    (_num_changes, setting_kwargs)
                ]
            }
        }
        where pane is one of 'Project', 'Basic', or 'Advanced,' and
        setting is the name of a setting in that pane (project_start_date or wind_speed_measurement_offsets, for example). This makes
        history[pane][setting] contain a list of changes made to that setting. This list is composed of tuples of the form
        (_num_changes, setting_kwargs), where _num_changes records the TOTAL number of changes made to the config file up to that point
        and setting_kwargs records the new settings recorded at that time, as returned by the get_XXXX function for that setting. The
        first entry in this list is always the initial state of that setting before any changes were made, meaning that _num_changes is
        not unique.

        example: the initial state of the wind_speed_measurement_offsets setting was u=0, v=0, w=0. 
        the third change made to the config file modified this to u=5, v=10, w=5.
        >>> ref = EddyproConfigEditor('config.eddypro')
        >>> ref.Basic.set_project_date_range('2021-01-01 00:00', '2023-10-13 14:54')
        >>> ref.Advanced.Processing.set_wind_speed_measurement_offsets(5, 10, 5)
        >>> print(ref.history['Advanced']['wind_speed_measurement_offsets'])
        [(0, {'u': 0.0, 'v': 0.0, 'w': 0.0}), (3, {'u': 5.0, 'v': 10.0, 'w': 5.0})]

        To better view and interpret the config file history, a print_history method is provided.



    Notes
    -----
    This class splits up settings by how they are laid out in the EddyPro 7 GUI,
    and contains 3 nested classes:
        * `Proj` contains settings from the Project Creation pane
        * `Basic` contains settings from the basic settings pane
        * `Adv` contains settings from the advanced settings pane, which is broken up into four more nested classes:
            * `Proc` -- settings from the processing option pane
            * `Stat` -- settings from the statistical analysis pane
            * `Spec` -- settings from the spectral analysis and corrections pane
            * `Out` -- settings from the output options pane

    To imitate how the eddypro GUI changes INI settings, each of these nested classes contains
    `set_XXXX` methods which reproduce the functionality of the respective buttons and panels in the eddypro GUI.
    For example, the `Adv.Proc.set_turbulent_fluctuations` method will reproduce the behavior
    of the "Turbulent fluctuations" options in the Eddypro 7 GUI, which can be found in the Advanced/Processing Options pane.
    Some functions are more complicated, like `Adv.Proce.set_axis_rotations_for_tilt_correction,` which needs to accomodate planar fit options. Read the documentation of 
    each method carefully.

    Additionally, each `set` method is paired with a `get` method, which will retrieve the *current* selected options for that method already stored in the file.
    the object returned by a given `get` method can be passed as **kwargs to the paired `set` method.

    e.g.: 
    >>> # instantiate from file
    >>> ini = EddyProConfigEditor('./config.eddypro')
    >>> # retrieve current turbulent fluctiations settings
    >>> tf_kwargs = ini.Adv.Proc.get_turbulent_fluctuations()
    >>> tf_kwargs
    {'detrend_method': 'block', 'time_constant': 0}
    >>> # re-write turbulent fluctuation settings without changing them
    >>> ini.Adv.Proc.set_turbulent_fluctuation(**tf_kwargs)  

    Finally, since EddyproConfigEditor is a child of configparser.ConfigParser, we can directly
    use ConfigParser methods on it. However, this is not recommended, since it can create changes
    in the config file that may not be tracked by EddyproConfigEditor. For example:
    >>> # instantiate from file
    >>> ini = EddyProConfigEditor('./config.eddypro')
    >>> # print ini file sections
    >>> for s in ini.sections(): print(s)
    FluxCorrection_SpectralAnalysis_General
    Project
    RawProcess_BiometMeasurements
    RawProcess_General
    RawProcess_ParameterSettings
    RawProcess_Settings
    RawProcess_Tests
    RawProcess_TiltCorrection_Settings
    RawProcess_TimelagOptimization_Settings
    RawProcess_WindDirectionFilter
    >>> # retrieve the entry "pr_start_date"
    >>> ini.get(section='Project', option='pr_start_date')
    '2020-06-21'
    >>> # set a new start dat
    >>> ini.set(section='Project', option='pr_start_date', value='2021-04-20')
    >>> # check that it worked
    >>> ini.get(section='Project', option='pr_start_date')
    '2021-04-20'
    >>> # see if the EddyproConfigEditor noticed the change: it didn't!
    >>> ini._project_start_date
    False
    >>> # instead, use the method Basic.set_project_start_date
    >>> ini.set_project_start_date('2021-04-21 00:00')
    >>> ini.get_project_start_date()
    datetime.datetime(2021, 4, 21, 0, 0)
    >>> ini.get(section='Project', option='pr_start_date')
    '2021-04-21'
    >>> ini._project_start_date
    True

    Some settings affect the function of other settings. 
    For example, consider the following set of commands:
    >>> # instantiate from file
    >>> ini = EddyProConfigEditor('./config.eddypro')
    >>> # set the project date range, but mistakenly enter 1901 instead of 2021
    >>> ini.Basic.set_project_date_range('1901-01-01 00:00', '1902-01-01 00:00')
    >>> # set the planar fit method to fit the project time window
    >>> ini.Adv.Proc.set_axis_rotations_for_tilt_correction(
    >>>     method='planar_fit',
            configure_planar_fit_settings_kwargs=dict(
                w_max=2, 
                u_min=0.1, 
                num_per_sector_min=5, 
                sectors=[(False, 360)], 
                start='project',
                end='project'))
    >>> # fix our earlier mistake by setting the correct project date range
    >>> ini.Basic.set_project_date_range('2021-01-01 00:00', '2022-01-01 00:00')
    >>> # read out the planar fit time window
    >>> ini.Adv.Proc.get_axis_rotations_for_tilt_correction()['configure_planar_fit_settings_kwargs']['start']
    '1901-01-01 00:00'

    Note how when we set the project date range AFTER setting the planar fit date range to 'project', the planar fit date range did not update.
    To avoid errors like this, it's highly recommended to modify settings in the following order:
        1. Project settings
        2. Basic Settings
        3. Advanced Settings
            1. Processing settings
            2. Statistical settings
            3. Spectral settings
            4. Output settings.
    So if you make a change in output settings, make sure that it comes after, and not before, any changes in Project, Basic, Advanced/Processing, Advanced/Statistical, and Advanced/Spectral
        
    '''
    def __init__(self, reference_ini: str | PathLike[str]):
        super().__init__(allow_no_value=True)
        self.read(reference_ini)

        # for tracking changes to the ini file
        self.history = dict()
        self._num_changes = 0

        self.Proj = self._Proj(self)
        self.Basic = self._Basic(self)
        self.Adv = self._Adv(self)

    # ---------------utilities-----------------------
    def to_eddypro(
            self,
            ini_file: str | PathLike[str],
            out_path: str | PathLike[str] | Literal['keep'] = 'keep') -> None:
        """
        Write this object to a .eddypro file.

        Parameters
        ----------
        ini_file: the file name to write to
        out_path: the path for eddypro to output results to. If 'keep' (default), use the outpath already in the config file (you can check this using Basic.get_out_path)
        """
        
        if str(ini_file)[-8:] != '.eddypro':
            ini_file = str(ini_file) + '.eddypro'
        self.set('Project', 'file_name', str(ini_file))
        if out_path != 'keep':
            self.set('Project', 'out_path', str(out_path))
        with open(ini_file, 'w') as configfile:
            configfile.write(';EDDYPRO_PROCESSING\n')  # header line
            self.write(fp=configfile, space_around_delimiters=False)

    def to_eddypro_parallel(
        self,
        environment_parent: str | PathLike[str],
        out_parent: str | PathLike[str],
        metadata_fn: str | PathLike[str],
        worker_windows: Sequence[datetime.datetime],
        dynamic_metadata_fn: str | PathLike[str] | None = None,
        ep_bin: None | str | PathLike[str] = None,
        file_duration: int | None = None,
        subset_pf_dates: bool = True,
        subset_sa_dates: bool = True,
        pf_file: str | PathLike[str] | Sequence[str | PathLike[str]] | None = None,
        binned_cosp_dir: str | PathLike[str] | Sequence[str | PathLike[str]] | None = None,
        autoopt_file: str | PathLike[str] | Sequence[str | PathLike[str]] | None = None,

    ) -> None:
        """
        Write this object to a collection of .eddypro files which can be run in parellel. For example, a configuration that tells eddypro to process data from 2019-2024 
        will be split up into multiple smaller sub-files, each covering a smaller timespan, as directed by the function arguments (for example, one config file covering each of
        2019-2020, 2020-2021, 2021-2022, 2022-2023, and 2023-2024). This can be used to run eddypro in a pseudo-parallel configuration, speeding up processing times by orders of magnitude. 
         
        However, this comes with some caveats and traps that can introduce errors into your final fluxes if you aren't careful. Some processes inside eddypro require long, continuous datasets 
        to produce quality results: the planar fit method, some spectral analysis methods, and the automatic time lag optimization methods require at least 2 weeks to 1 month of continuous 
        data to work properly, and ideally more. This function can work around these issues, but needs user guidance on how to properly do that. Blindly telling this function to generate 365 eddypro sub-files 
        each covering only 1 day of data without properly configuring the planar fit or spectral analysis settings will result in incorrect fluxes.

        Additionally, EddyPro does not like to be run in this "parallel-jank" format, so this function tries to work around some of its idiosynchrasies:
            1. Each sub-file gets its own environment directory. In this directory, we place .eddypro file, the eddypro_rp and eddypro_fcc executable files, and any additional files (such as planar fit files)
            2. Each sub-file also gets its own output directory. These directories should be kept separate from the environment directories.
       
        Here is an example file structure after running to_eddypro_parellel with instructions to generate 2 sub-files. Each eddypro instance will write output to its respective output directory in out_parent:
            /environment_parent
            |-- environment_20220101
            |  |-- bin
            |  |  |-- eddypro_rp
            |  |  |-- eddypro_fcc
            |  |-- ini
            |  |  |-- processing.eddypro
            |  |-- tmp
            |-- environment_20230101
            |  |-- bin
            |  |  |-- eddypro_rp
            |  |  |-- eddypro_fcc
            |  |-- ini
            |  |  |-- processing.eddypro
            |  |-- tmp

            /out_parent
            |-- environment_20220101
            |-- environment_20230101

        Parameters
        ----------
        environment_parent: each instance of eddypro needs its own environment directory. This defines the parent directory that will contain all the separate environments.
        out_parent: Similar to the environment parent directoy. Sub-folders will be created within out_parent to contain output files from each worker. Must be different from environment_parent.
        metadata_fn: path to a static .metadata file for this project.
        worker_windows: list of the breakpoints dilineating subfiles, as datetime objects. Each worker will span from worker_windows[i] to worker_windows[i + 1]. So [2022, 2023, 2024] will generate 2 workers: 2022-2023, and 2023-2024.
        dynamic_metadata_fn: same as metadata_fn, except for dynamic metadata. Can be None (default)
        ep_bin: the path to the directory containing both the eddypro_rp and eddypro_fcc executables. The entire contents of this directory will be copied into each environment. If None, then you will have to copy the eddypro executables into the bin directory of each environment yourself.
        file_duration: how many minutes long each file is (NOT the averaging interval). If None (Default), then that information will be gleaned from the metadata file.
        subset_pf_dates: if True (default), each worker will compute a planar fit only on the dataset allocated to it by the worker window. If False, each worker will compute a planar fit using all data in the raw data directory. This parameter is ignored if a planar fit is not required, or if pf_files are provided. Note that setting to False will drastically increase computation time. Be aware that setting this option to True when worker windows are shorter than 1 month. This option should not be set to true when worker windows are shorter than 2 weeks.
        subset_sa_dates: same as for subset_pf_dates, but for spectral computation. At least one month of data is required for a robust spectral assessment, so this option should not be set to True when each worker window covers less than one month of data.
        pf_file: a single planar fit file or a list of planar fit files to use. If multiple planar fit files are provided, they must match 1:1 onto each worker. This argument overrides pf_subset. This setting is highly recommended in all cases where a planar fit must be used with worker windows of <3 month durations.
        binned_cosp_dir: a directory or a list of directories that contain binned (co)spectra files relevant to this run. Similarly to pf_file, if multiple directories are provided, they much match 1:1 onto each worker. This argument overrides sa_subset. This setting is highly recommended in all cases where a spectral must be used with worker windows of <3 month durations.
        autoopt_file: a single automatic timelag optimization file or list of such files. If multiple such files are provided, they must match 1:1 onto each worker. This argument overrides autoopt_subset. This setting is highly recommended in all cases where a time lag optimization must be used with worker windows of <3 month durations.

        Examples
        --------
        >>> from datetime import datetime
        >>> from pandas import date_range
        >>> #
        >>> metadata_fn = '~/static.metadata'
        >>> ep_bin = '~/project/bin'
        >>> # Example 1: assuming you already have a ready-to-go config file configured for planar fit, each worker processes 180 days of data and performs a spectral assessment and planar fit correction on those 180 days
        >>> start, end = '2019-01-01 00:00', '2023-12-31 23:30'  # start and end of this run
        >>> # each sub-file will process 180 days of data
        >>> worker_windows = [datetime.strptime(d, r'%Y-%m-%d %H:%M') for d in date_range(start, end, freq='180d').strftime(r'%Y-%m-%d %H:%M')]  # timespan to allocate to each processor
        >>> template.to_eddypro_parallel(
        >>>     environment_parent='environments',
        >>>     out_parent='outs',
        >>>     metadata_fn=metadata_fn,
        >>>     file_duration=1440,
        >>>     ep_bin=ep_bin,
        >>>     worker_windows=worker_windows
        >>> )
        >>> #
        >>> # Example 2: assuming you already have a ready-to-go config file configured for planar fit, each worker processes only 7 days of data.
        >>> # In this case, each worker window is too short to properly compute a planar fit and spectral assessment, so we must provide that information ourselves.
        >>> worker_windows = [datetime.strptime(d, r'%Y-%m-%d %H:%M') for d in date_range(start, end, freq='7d').strftime(r'%Y-%m-%d %H:%M')]  # timespan to allocate to each processor
        >>> pf_file = '~/previous_run/eddypro...planar_fit....txt'
        >>> sa_dir = '~/previous_run/eddypro_binned_cospectra'
        >>> template.to_eddypro_parallel(
        >>>     environment_parent='environments',
        >>>     out_parent='outs',
        >>>     metadata_fn=metadata_fn,
        >>>     file_duration=1440,
        >>>     ep_bin=ep_bin,
        >>>     worker_windows=worker_windows,
        >>>     pf_file=pf_file,
        >>>     sa_dir=sa_dir
        >>> )
        >>> #
        >>> # Example 3: assuming you already have a ready-to-go config file configured for planar fit, each worker processes only 7 days of data.
        >>> # In this case, each worker window is too short to properly compute a planar fit and spectral assessment, so we must provide that information ourselves.
        >>> # Here, we have 4 planar fit files, one for each of 2019, 2020, 2021, 2022, and 2023. We must match these up with the worker windows. We have 260 worker windows (261 edges - 1)
        >>> # each worker is assigned the planar fit file computed on the worker's year.
        >>> worker_windows = [datetime.strptime(d, r'%Y-%m-%d %H:%M') for d in date_range(start, end, freq='7d').strftime(r'%Y-%m-%d %H:%M')]  # timespan to allocate to each processor
        >>> _pf_file = [
        >>>     ['~/previous_run/planar_fit_2019']*52,
        >>>     ['~/previous_run/planar_fit_2020']*52,
        >>>     ['~/previous_run/planar_fit_2021']*52,
        >>>     ['~/previous_run/planar_fit_2022']*52,
        >>>     ['~/previous_run/planar_fit_2023']*52,
        >>> ]
        >>> pf_files = []
        >>> for lst in _pf_files: pf_files += lst  # create 6 copies of each planar fit file, since we have 6x more workers
        >>> #
        >>> sa_dir = '~/previous_run/eddypro_binned_cospectra'
        >>> template.to_eddypro_parallel(
        >>>     environment_parent='environments',
        >>>     out_parent='outs',
        >>>     metadata_fn=metadata_fn,
        >>>     file_duration=1440,
        >>>     ep_bin=ep_bin,
        >>>     worker_windows=worker_windows,
        >>>     pf_file=pf_file,
        >>>     sa_dir=sa_dir
        >>> )
        """
        
        # get file duration
        if file_duration is None:
            metadata = configparser.ConfigParser()
            metadata.read(metadata_fn)
            file_duration = int(metadata['Timing']['file_duration'])

        #### determine how to allocate jobs to each worker ####
        start, end = self.Basic.get_project_date_range().values()
        assert start != 'all_available', 'when using to_eddypro_parallel, must explicitly set the project date range to something other than all_available with Basic.set_project_date_range.'

        for i in worker_windows: 
            assert isinstance(i, datetime.datetime), 'worker starts must be a list of datetime.datetime'
        job_starts = worker_windows[:-1]
        job_ends = [start - Timedelta(file_duration) for start in worker_windows[1:]]

        # make sure that the number of planar fit files, if provided, matches the number of workers
        if pf_file is not None:
            pf_file = list(pf_file)
            if len(pf_file) == 1: pf_file *= len(worker_windows) - 1
            assert len(pf_file) == len(worker_windows) - 1, 'planar fit files must match 1:1 with workers'
        # make sure that the number of sa files, if provided, matches the number of workers
        if binned_cosp_dir is not None:
            binned_cosp_dir = list(binned_cosp_dir)
            if len(binned_cosp_dir) == 1: binned_cosp_dir *= len(worker_windows) - 1
            assert len(binned_cosp_dir) == len(worker_windows) - 1, 'spectral analysis directories must match 1:1 with workers'
        # make sure that the number of autoopt files, if provided, matches the number of workers
        if autoopt_file is not None:
            autoopt_file = list(autoopt_file)
            if len(autoopt_file) == 1: autoopt_file *= len(worker_windows) - 1
            assert len(autoopt_file) == len(worker_windows) - 1, 'autoopt files must match 1:1 with workers'
        
        #### file organization stuff ####
        # give each project a unique id and file name
        proj_id = self.Basic.get_output_id()['output_id']
        project_ids = [
            f'{proj_id}-{start.strftime(r"%Y%m%d%H%M")}' for start in job_starts
        ]
        ini_fns = [environment_parent / project_id / 'ini' / 'processing.eddypro' for project_id in project_ids]
        for fn in ini_fns: fn.parent.mkdir(exist_ok=True, parents=True)
        # create output directories
        Path(out_parent).mkdir(exist_ok=True)
        out_dirs = [out_parent / project_id for project_id in project_ids]
        for d in out_dirs: d.mkdir(exist_ok=True)

        #### modify the .eddypro file for parallel operation ####
        # save original settings
        old_file_name = self.get('Project', 'file_name')
        old_out_path = self.Basic.get_out_path()
        project_id = self.Basic.get_output_id()
        old_sa_settings = self.Adv.Spec.get_calculation()
        old_timelag_settings = self.Adv.Proc.get_timelag_compensations()
        old_tilt_settings = self.Adv.Proc.get_axis_rotations_for_tilt_correction()
        # write new files
        for i, fn in enumerate(ini_fns):
            # make a collection of new environments
            new_metadata = fn.parent / 'static.metadata'
            shutil.copy(metadata_fn, new_metadata)
            if dynamic_metadata_fn is not None:
                new_dynamic_metadata = fn.parent / 'dynamic.metadata'
                shutil.copy(dynamic_metadata_fn, new_dynamic_metadata)
            else:
                new_dynamic_metadata = False
            self.Proj.set_metadata(static=new_metadata, dynamic=new_dynamic_metadata)
            
            self.Basic.set_out_path(out_dirs[i])
            
            self.set('Project', 'file_name', str(fn))
            self.Basic.set_output_id(project_ids[i])

            self.Basic.set_project_date_range(job_starts[i], job_ends[i])
            
            # modify spectral corrections time window
            new_sa_settings = deepcopy(old_sa_settings)
            if (
                (old_sa_settings['start'] != 'project')
                and ('binned_cosp_dir' not in old_sa_settings
                     or subset_sa_dates)
            ):
                new_sa_settings['start'] = 'project'
                new_sa_settings['end'] = 'project'
                self.Adv.Spec.set_calculation(**new_sa_settings)
            if binned_cosp_dir is not None:
                self.Adv.Spec.set_calculation(binned_cosp_dir=binned_cosp_dir[i])

            # modify timelag settings timewindow
            new_timelag_settings = deepcopy(old_timelag_settings)
            if (
                (old_timelag_settings['method'] == 'automatic_optimization')
                and (old_timelag_settings['autoopt_file'] is None)
                and (old_timelag_settings['configure_TimelagAutoOpt_kwargs']['start'] != 'project')
            ):
                new_timelag_settings['configure_TimelagAutoOpt_kwargs']['start'] = 'project'
                new_timelag_settings['configure_TimelagAutoOpt_kwargs']['end'] = 'project'
                self.Adv.Proc.set_timelag_compensations(**new_timelag_settings)
            if autoopt_file is not None:
                new_autoopt_file = fn.parent.parent / 'auto_opt.txt'
                shutil.copy(autoopt_file[i], new_autoopt_file)
                method = self.Adv.Proc.get_timelag_compensations()['method']
                self.Adv.Proc.set_timelag_compensations(method=method, autoopt_file=new_autoopt_file)

            # modify planar fit settings time window
            new_tilt_settings = deepcopy(old_tilt_settings)
            if (
                ('planar_fit' in old_tilt_settings['method'])
                and (old_tilt_settings['pf_file'] is None)
                and (old_tilt_settings['configure_planar_fit_settings_kwargs']['start'] in ['project', 'all_available']
                     or subset_pf_dates)
            ):
                new_tilt_settings['configure_planar_fit_settings_kwargs']['start'] = 'project'
                new_tilt_settings['configure_planar_fit_settings_kwargs']['end'] = 'project'
                self.Adv.Proc.set_axis_rotations_for_tilt_correction(**new_tilt_settings)
            if pf_file is not None:
                new_pf_file = fn.parent.parent / 'planar_fit.txt'
                shutil.copy(pf_file[i], new_pf_file)
                method = self.Adv.Proc.get_axis_rotations_for_tilt_correction()['method']
                self.Adv.Proc.set_axis_rotations_for_tilt_correction(method=method, pf_file=new_pf_file)
            
            # write to file
            with open(fn, 'w') as configfile:
                configfile.write(';EDDYPRO_PROCESSING\n')  # header line
                self.write(fp=configfile, space_around_delimiters=False)

            # copy bin to environment
            if ep_bin is not None:
                shutil.copytree(ep_bin, fn.parent.parent / 'bin', dirs_exist_ok=True)
            else:
                (fn.parent.parent / 'bin').mkdir(exist_ok=True)

            # create a tmp directory
            (fn.parent.parent / 'tmp').mkdir(exist_ok=True)

        # revert to original
        self.set('Project', 'file_name', old_file_name)
        self.Basic.set_out_path(old_out_path)
        self.Basic.set_project_date_range(start, end)
        self.Adv.Spec.set_calculation(**old_sa_settings)
        self.Adv.Proc.set_timelag_compensations(**old_timelag_settings)
        self.Adv.Proc.set_axis_rotations_for_tilt_correction(**old_tilt_settings)
        self.Basic.set_output_id(**project_id)

        return

    def to_pandas(self) -> DataFrame:
        """convert current ini state to a pandas dataframe"""
        lines = []
        for section in self.sections():
            for option, value, in self[section].items():
                lines.append([section, option, value])
        df = DataFrame(lines, columns=['Section', 'Option', 'Value'])
        df = df.sort_values(['Section', 'Option'])
        df['Name'] = Path(self.get('Project', 'file_name')).stem

        return df

    def check_dates(
            self, 
            interval: Sequence[str | datetime.datetime, str | datetime.datetime], 
            reference: Sequence[str | datetime.datetime, str | datetime.datetime] | Literal['project'] = 'project',
            min_overlap: float = 0) -> bool:
        """
        Checks all current settings to find any conflicts. 
        Can be configured to either warn the users of conflicts, or to raise an error when conflicts are detected

        Parameters
        ----------
        interval: input sequence to check validity for. 
        sequence of length 2 containing datetime.datetime objects or strings of format YYYY-mm-dd HH:MM. 
        reference: t: sequence to reference input sequence against.
        sequence of length 2 containing datetime.datetime objects or strings of format YYYY-mm-dd HH:MM. 
        Alternatively, pass the keyword 'project' to use the project start and end dates as a reference point.
        tolerance: float specifying the minimum allowable overlap tolerance between interval and reference, in days. Default 0

        Returns 
        -------
        True if overlap between interval and reference is greater than or equal to min_overlap number of days. False otherwise

        Points of concern are:
        * invalid project start and end dates (project date range is shorter than averaging window)
        * invalid planar fit start and end dates (requires two weeks of overlap with project start and end dates)
        * invalid time lag optimization start and end dates (requires one month of overlap with project start and end dates)
        * invalid spectra calculation dates (requires one month of overlap)
        """
        if reference == 'project': reference = list(self.Basic.get_project_date_range().values())
        overlap = compute_date_overlap(interval, reference).days

        return overlap >= min_overlap

    def _add_to_history(self, pane, setting, getter, modify_only_if_first=False):
        # tracks setting history.
        # history structure:
        # {pane:
        #   {setting:
        #       [n_changes, setting_kwargs]}}

        # if the current setting hasn't been modified yet, 
        # initialize the history
        if pane not in self.history:
            self.history[pane] = dict()
        if setting not in self.history[pane]:
            self.history[pane][setting] = list()

        # modify_only_if_first will tell us to only add to the history
        # if the current history is empty.
        if modify_only_if_first:
            if len(self.history[pane][setting]) == 0:
                current_setting = getter()
                self.history[pane][setting].append((0, current_setting))
            return
        current_setting = getter()
        self._num_changes += 1
        self.history[pane][setting].append((deepcopy(self._num_changes), current_setting))

        return
    def print_history(self, grouping: Literal['h', 'c'] = 'h', max_ops: float | int = 5e8):
        """print the (tracked) change history of the config
        Parameters
        ----------
        grouping: if 'hierarchical', then group outputs by Pane, then Setting, then change # (default). If 'chronological', then group outputs by change # only.
        max_ops: the maximum number of operations to perform when searching the history before raising RunTimeError. With complex and long histories, search time can become extremely long when printing with chronological grouping. This should never
        be a problem when printing in hierarchical grouping. The default setting of 5e8 operations equates to about a ~30 second timeout on my (Alex Fox) machine, and will be on the order of 10^5 total changes to the file, so it's unlikely that you'll ever need to modify this setting
        unless you're doing something really wild. Set to float('+inf') to disable this setting."""

        assert grouping in ['h', 'c'], 'grouping must be one of "h" or "c".'
        ops = 0
        if grouping == 'h':
            for pane in self.history:
                print(f'--{pane}------------------')
                for setting in self.history[pane]:
                    print(f'  {setting}')
                    for entry in self.history[pane][setting]:
                        i, history_item = entry
                        print(f'    Change #{i}')
                        for k, v in history_item.items():
                            print(f'      {k}: {v}')
                            ops += 1
                            if ops >= max_ops:
                                raise RuntimeError('Maximum operations reached, aborting')
                    print()
                print()
        elif grouping == 'c':
            # print base state before any changes occured
            history_copy = deepcopy(self.history)
            print('--Base State-----------')
            for pane in history_copy:
                for setting in history_copy[pane]:
                    for entry_num, entry in enumerate(history_copy[pane][setting]):
                        i, history_item = entry
                        # if this entry represents a "base state," print it and remove it from the history
                        ops += 1
                        if ops >= max_ops:
                            raise RuntimeError('Maximum operations reached, aborting')
                            
                        if i == 0:
                            ops -= 1
                            print(f'  {pane}/{setting}')
                            for k, v in history_item.items():   
                                ops += 1
                                print(f'    {k}: {v}')
                            _ = history_copy[pane][setting].pop(entry_num)
            print()
                            
            # print subsequent changes
            print('--Modifications--------')
            max_i = self._num_changes
            target_i = 1
            while target_i <= max_i:
                for pane in history_copy:
                    for setting in history_copy[pane]:
                        for entry_num, entry in enumerate(history_copy[pane][setting]):
                            i, history_item = entry
                            ops += 1
                            if ops >= max_ops:
                                raise RuntimeError('Maximum operations reached, aborting')
                            if i == target_i:
                                ops -= 1
                                print(f'  {target_i} {pane}/{setting}')
                                for k, v in history_item.items():   
                                    ops += 1
                                    print(f'    {k}: {v}')
                                target_i += 1
        print(ops)

    def copy(self):
        """copies this config through a temporary file. The resulting copy is independent of this instance"""
        tmp = tempfile.NamedTemporaryFile(mode='w', delete=False)
        try:
            tmp.write(';EDDYPRO_PROCESSING\n')  # header line
            self.write(fp=tmp, space_around_delimiters=False)
            tmp.close()

            tmp = open(tmp.name, mode='r')
            cls = self.__class__
            new = self.__new__(cls)
            new.__init__(tmp.name)
            tmp.close()

            os.remove(tmp.name)
        except BaseException:
            tmp.close()
            os.remove(tmp.name)
            raise
        
        new.history = deepcopy(self.history)

        return new

    def __copy__(self):
        return self.copy()

    def __repr__(self):
        return str(self.to_pandas().drop(columns='Name'))
    
    # -------------------Project Creation Page---------------------
    class _Proj:
        def __init__(self, root):
            self.root = root

        def set_project_name(self, name: str):
            """Enter a name for the flux computation project. This field is optional."""
            history_args = ('Project', 'project_name', self.get_project_name)
            self.root._add_to_history(*history_args, True)
            self.root.set('Project', 'project_title', str(name))
            self.root._add_to_history(*history_args)
            return
        def get_project_name(self):
            return dict(name=self.root.get('Project', 'project_title'))
        
        def set_raw_file_format(self, fmt, n_ascii_header_lines=None, ascii_header_eol=None, num_bytes_per_var=None, endianess=None):
            """Set the raw file format settings
            
            Parameters
            ----------
            fmt: one of 'ghg', 'ascii', 'generic_binary', 'TOB1_auto', 'TOB1_IEEE4', 'TOB1_FP2', 'EddySoft', 'EdiSol'.
            n_ascii_header_lines, ascii_header_eol, num_bytes_per_var, endianess: required if format='generic_binary,' otherwise ignored.
            """
            history_args = ('Project', 'raw_file_format', self.get_raw_file_format)
            self.root._add_to_history(*history_args, True)

            assert fmt in ['ghg', 'ascii', 'generic_binary', 'TOB1_auto', 'TOB1_IEEE4', 'TOB1_FP2', 'EddySoft', 'EdiSol'], "fmt must be one of 'ghg', 'ascii', 'generic_binary', 'TOB1_auto', 'TOB1_IEEE4', 'TOB1_FP2', 'EddySoft', 'EdiSol'"
            if fmt == 'generic_binary':
                assert (
                    n_ascii_header_lines is not None and 
                    ascii_header_eol is not None and
                    num_bytes_per_var is not None and 
                    endianess is not None
                    ), 'n_ascii_header_lines, eascii_header_eol, num_bytes_per_var, and endianess must be provided when using generic_binary format'

            match fmt:
                case 'ghg':
                    self.root.set('Project', 'file_type', '0')
                case 'ascii':
                    self.root.set('Project', 'file_type', '1')
                case 'generic_binary':
                    self.root.set('Project', 'file_type', '5')
                    self.root.set('Project', 'binary_eol', int(ascii_header_eol))
                    self.root.set('Project', 'hnlines', int(n_ascii_header_lines))
                    self.root.set('Project', 'binary_little_end', int(endianess))
                    self.root.set('Project', 'binary_nbytes', int(num_bytes_per_var))
                case 'TOB1_auto':
                    self.root.set('Project', 'file_type', '2')
                    self.root.set('Project', 'tob1_format', '0')
                case 'TOB1_IEEE4': 
                    self.root.set('Project', 'file_type', '2')
                    self.root.set('Project', 'tob1_format', '1')
                case 'TOB1_FP2':
                    self.root.set('Project', 'file_type', '2')
                    self.root.set('Project', 'tob1_format', '2')
                case 'EddySoft':
                    self.root.set('Project', 'file_type', '3')
                case 'EdiSol':
                    self.root.set('Project', 'file_type', '4')
            self.root._add_to_history(*history_args)
        def get_raw_file_format(self) -> dict:
            kwargs = dict()
            formats = ['ghg', 'ascii', 'tob1', 'EddySoft', 'EdiSol', 'generic_binary']
            fmt = formats[int(self.root.get('Project', 'file_type'))]
            match fmt:
                case 'tob1':
                    tob1_types = ['TOB1_auto', 'TOB1_IEEE4', 'TOB1_FP2']
                    kwargs['fmt'] = tob1_types[int(self.root.get('Project', 'tob1_format'))]
                case 'generic_binary':
                    kwargs['fmt'] = 'generic_binary'
                    kwargs['n_ascii_header_lines'] = int(self.root.get('Project', 'hnlines'))
                    kwargs['ascii_header_eol'] = int(self.root.get('Project', 'binary_eol'))
                    kwargs['num_bytes_per_var'] = int(self.root.get('Project', 'binary_nbytes'))
                    kwargs['endianess'] = int(self.root.get('Project', 'binary_little_end'))
                case _:
                    kwargs['fmt'] = fmt
            return kwargs
        
        def set_metadata(self, static:str|PathLike|Literal['embedded'], dynamic:str|PathLike|bool=False):
            """Set how to process project metadata.
            
            Parameters
            ----------
            static: Path to the static metadata file. Can either be a string, path, or 'embedded'. If 'embedded', use embedded metadata (.ghg files only)
            dynamic: Path to an optional dynamic metadata file. Default False"""
            history_args = ('Project', 'metadata', self.get_metadata)
            self.root._add_to_history(*history_args, True)

            if static == 'embedded':
                assert self.root.get('Project', 'file_type') == '0', 'if using embedded metadata, file type must be .ghg'
            assert (dynamic == False) or or_isinstance(dynamic, PathLike, str), 'dynamic should be False or a string or a path'

            if static != 'embedded':
                self.root.set('Project', 'proj_file', str(static))
                self.root.set('Project', 'use_pfile', '1')
            else:
                self.root.set('Project', 'use_pfile', '0')

            if dynamic:
                self.root.set('Project', 'use_dyn_md_file', '1')
                self.root.set('Project', 'dyn_metadata_file', str(dynamic))
            else:
                self.root.set('Project', 'use_dyn_md_file', '0')

            self.root._add_to_history(*history_args)
        def get_metadata(self)->dict:
            if self.root.get('Project', 'use_pfile') == '0':
                static = 'embdedded'
            else:
                static = self.root.get('Project', 'proj_file')
            
            if self.root.get('Project', 'use_dyn_md_file') == '1':
                dynamic = self.root.get('dyn_metadata_file')
            else:
                dynamic = False
            return dict(static=static, dynamic=dynamic)
        
        def set_biomet(self, mode:Literal['embedded', 'file', 'dir', 'none'], path:str|PathLike|None=None, extension:str|None=None, subfolders:bool=False):
            """
            set biomet file parameters
            
            Parameters
            ----------
            mode: one of 'embedded', 'file', 'dir', or 'none'. If 'embedded', use embedded biomet data in .ghg files. If 'file', use a single biomet file, with the path indicated in the path argument. If 'dir', use a directory of biomet files, with the directory path indicated in the 'path' argument, and the file extension indicated. If 'none', do not use any external biomet data.
            path: if mode id 'file' or 'dir', the path to the file or dir. Leave blank for 'embedded'
            extension: if mode='dir', the extension on the biomet files, such as 'csv', 'txt', 'dat', etc. Only required for mode='dir'. Do NOT prepend a '.' to the extension. E.g. '.csv' is INCORRECT, but 'csv' is correct.
            subfolder: if mode='dir', whether to search in subfolders. Only required if mode='dir'"""
            history_args = ('Project', 'biomet', self.get_biomet)
            self.root._add_to_history(*history_args, True)

            assert mode in ['embedded', 'file', 'dir', 'none'], "mode must be one of 'embedded', 'file', 'dir'"
            if mode == 'embedded':
                assert self.root.get('Project', 'file_type') == '0', 'if using embedded biomet data, file type must be .ghg'
            if mode in ['file', 'dir']:
                assert or_isinstance(path, str, PathLike), 'if mode if file or dir, path must be provided'
            if mode == 'dir':
                assert isinstance(extension, str), 'if mode if dir, extension must be provided'
            
            match mode:
                case 'none':
                    self.root.set('Project', 'use_biom', '0')
                case 'embedded':
                    self.root.set('Project', 'use_biom', '1')
                case 'file':
                    self.root.set('Project', 'use_biom', '2')
                    self.root.set('Project', 'biom_file', str(path))
                case 'dir':
                    self.root.set('Project', 'use_biom', '3')
                    self.root.set('Project', 'biom_dir', str(path))
                    self.root.set('Project', 'biom_rec', str(int(bool(subfolders))))
                    self.root.set('Project', 'biom_ext', '.' + extension)
                
            self.root._add_to_history(*history_args)
        
        def get_biomet(self):
            i_mode = self.root.get('Project', 'use_biom')

            match i_mode:
                case '0':
                    mode = 'none'
                    path=None
                    subfolders=None
                    extension=None
                case '1':
                    mode = 'embedded'
                    path=None
                    subfolders=None
                    extension=None
                case '2':
                    mode = 'file'
                    path = self.root.get('Project', 'biom_file')
                    subfolders=None
                    extension=None
                case '3':
                    mode='dir'
                    path = self.root.get('Project', 'biom_dir')
                    subfolders = bool(int(self.root.get('Project', 'biom_rec')))
                    extension = self.root.get('Project', 'biom_ext')[1:]
            return dict(mode=mode, path=path, subfolders=subfolders, extension=extension)
            
        
    # --------------------Basic Settings Page-----------------------
    class _Basic:
        """Basic processing settings"""
        def __init__(self, root):
            self.root = root

        def set_raw_data(
                self, 
                path:str|PathLike, 
                fmt:str,
                subfolders:bool=True, 
                ):
            """how to find the raw data
            
            Parameters
            ----------
            path: str or pathlike, path to data directory
            format: how the timestamp is encoded in the file name. See eddypro documentation for details
            """

            history_args = ('Project', 'biomet', self.get_raw_data)
            self.root._add_to_history(*history_args, True)

            self.root.set('Project', 'file_prototype', fmt)
            self.root.set('RawProcess_General', 'recurse', str(int(bool(subfolders))))
            self.root.set('RawProcess_General', 'data_path', str(path))
            
            self.root._add_to_history(*history_args)
        def get_raw_data(self):
            fmt = self.root.get('Project', 'file_prototype')
            subfolders = bool(int(self.root.get('RawProcess_General', 'recurse')))
            path = self.root.get('RawProcess_General', 'data_path')

            return dict(path=path, fmt=fmt, subfolders=subfolders)

        def set_out_path(self, d):
            """set the eddypro output path to directory d"""
            history_args = ('Basic', 'out_path', self.get_out_path)
            self.root._add_to_history(*history_args, True)
            self.root.set('Project', 'out_path', str(d))
            self.root._add_to_history(*history_args)
        def get_out_path(self) -> Path:
            return Path(self.root.get('Project', 'out_path'))

        def set_project_start_date(
            self,
            start: str | datetime.datetime,
        ) -> None:
            """format yyyy-mm-dd HH:MM for strings"""

            assert or_isinstance(start, str, datetime.datetime)
            if isinstance(start, str):
                assert len(start.strip()) == 16, 'if using a string, must pass timestamps in YYYY-mm-DD HH:MM format'

            if isinstance(start, str):
                pr_start_date, pr_start_time = start.strip().split(' ')
            else:
                pr_start_date = start.strftime(r'%Y-%m-%d')
                pr_start_time = start.strftime(r'%H:%M')

            self.root.set('Project', 'pr_start_date', str(pr_start_date))
            self.root.set('Project', 'pr_start_time', str(pr_start_time))
        def get_project_start_date(self) -> datetime.datetime:
            """retrieve form the config file the project start date."""
            out = dict()
            start_date = self.root.get('Project', 'pr_start_date')
            start_time = self.root.get('Project', 'pr_start_time')
            if start_time is None:
                start_time = '00:00'
            out['start'] = datetime.datetime.strptime(f'{start_date} {start_time}', r'%Y-%m-%d %H:%M')
            
            return out

        def set_project_end_date(
            self,
            end: str | datetime.datetime
        ) -> None:
            """format yyyy-mm-dd HH:MM for strings"""
            
            assert or_isinstance(end, str, datetime.datetime)
            if isinstance(end, str):
                assert len(end.strip()) == 16, 'if using a string, must pass timestamps in YYYY-mm-DD HH:MM format'

            if isinstance(end, str):
                pr_end_date, pr_end_time = end.strip().split(' ')
            else:
                pr_end_date = end.strftime(r'%Y-%m-%d')
                pr_end_time = end.strftime(r'%H:%M')

            self.root.set('Project', 'pr_end_date', str(pr_end_date))
            self.root.set('Project', 'pr_end_time', str(pr_end_time))
            
        def get_project_end_date(self) -> dict:
            """retrieve from the config file the project end date."""
            out = dict()
            end_date = self.root.get('Project', 'pr_end_date')
            end_time = self.root.get('Project', 'pr_end_time')
            if end_time is None:
                end_time = '00:00'
            out['end'] = datetime.datetime.strptime(f'{end_date} {end_time}', r'%Y-%m-%d %H:%M')
            
            return out

        def set_project_date_range(
            self,
            start: str | datetime.datetime | Literal['all_available'] = 'all_available',
            end: str | datetime.datetime | Literal['all_available'] = 'all_available',
        ):
            """format yyyy-mm-dd HH:MM for strings"""
            history_args = ('Basic', 'project_date_range', self.get_project_date_range)
            self.root._add_to_history(*history_args, modify_only_if_first=True)
            
            if end < start:
                warnings.warn(f'Selected processing period is invalid, start comes after end: {str(start)} -> {str(end)}')
            if (start == 'all_available') or (end == 'all_available'):
                assert start == end, 'if using all_available, both start and end must be set to all_available.'
            
            if start == 'all_available':
                self.root.set('Project', 'pr_subset', '0')
            else:
                self.root.set('Project', 'pr_subset', '1')
                self.set_project_start_date(start)
                self.set_project_end_date(end)
            
            self.root._add_to_history(*history_args)

        def get_project_date_range(self) -> dict:
            """retrieve form the config file the project start and end dates. Output can be can be passed to set_project_date_range as kwargs"""
            if not bool(int(self.root.get('Project', 'pr_subset'))):
                start = end = 'all_available'
            else:
                start = self.get_project_start_date()['start']
                end = self.get_project_end_date()['end']
            return dict(start=start, end=end)

        def set_missing_samples_allowance(self, pct: int):
            """
            Set the missing samples allowance

            Parameters
            ----------
            pct: value from 0 to 40%
            """
            assert pct >= 0 and pct <= 40, 'pct must be between 0 and 40'

            history_args = ('Basic', 'missing_samples_allowance', self.get_missing_samples_allowance)
            self.root._add_to_history(*history_args, True)
            self.root.set('RawProcess_Settings', 'max_lack', str(int(pct)))
            self.root._add_to_history(*history_args)
        def get_missing_samples_allowance(self) -> int:
            """retrieve form the config file the maximum allowed missing samples per averaging window in %."""
            return int(self.root.get('RawProcess_Settings', 'max_lack'))

        def set_flux_averaging_interval(self, minutes: int):
            """
            Set the flux averaging interval.
            
            Parameters
            -----------
            minutes: how long to set the averaging interval to. If 0, use the file as-is"""

            assert minutes >= 0 and minutes <= 9999, 'Must have 0 <= minutes <= 9999'
            
            history_args = ('Basic', 'flux_averagin_interval', self.get_flux_averaging_interval)
            self.root._add_to_history(*history_args, True)
            self.root.set('RawProcess_Settings', 'avrg_len', str(int(minutes)))
            self.root._add_to_history(*history_args)
        def get_flux_averaging_interval(self) -> int:
            """retrieve form the config file the flux averaging interval in minutes"""
            return self.root.get('RawProcess_Settings', 'avrg_len')

        def set_north_reference(
            self,
            method: Literal['mag', 'geo'],
            magnetic_declination: float | None = None,
            declination_date: str | datetime.datetime | None = None,
        ):
            """set the north reference to either magnetic north (mag) or geographic north (geo). If geographic north, then you must provide a magnetic delcination and a declination date.

            Parameters
            ----------
            method: one of 'mag' or 'geo'
            magnetic_declination: a valid magnetic declination as a real number between -90 and 90. If 'geo' is selected, magnetic declination must be provided. Otherwise, does nothing.
            declination_date: the reference date for magnetic declination, either as a yyyy-mm-dd string or as a datetime.datetime object. If method = 'geo', then declination date must be provided. Otherwise, does nothing.
            """

            assert method in [
                'mag', 'geo'], "Method must be one of 'mag' (magnetic north) or 'geo' (geographic north)"

            history_args = ('Basic', 'north_reference', self.get_north_reference)
            self.root._add_to_history(*history_args, True)
            self.root.set('RawProcess_General', 'use_geo_north',
                          str(int(method == 'geo')))
            if method == 'geo':
                assert magnetic_declination is not None and declination_date is not None, 'declination and declination date must be provided if method is "geo."'
                assert magnetic_declination >= - \
                    90 and magnetic_declination <= 90, "Magnetic declination must be between -90 and +90 (inclusive)"
                self.root.set(
                    'RawProcess_General',
                    'mag_dec',
                    str(magnetic_declination))
                if isinstance(declination_date, str):
                    declination_date, _ = declination_date.split(' ')
                else:
                    declination_date = declination_date.strftime(r'%Y-%m-%d')
                self.root.set(
                    'RawProcess_General',
                    'dec_date',
                    str(declination_date))
                
                self.root._add_to_history(*history_args)
        def get_north_reference(self) -> dict:
            """retrieve form the config file the north reference data. output can be passed to set_north_reference__ as kwargs."""
            use_geo_north = self.root.get(
                'RawProcess_General', 'use_geo_north')
            if use_geo_north:
                use_geo_north = 'geo'
            else:
                use_geo_north = 'mag'

            mag_dec = float(self.root.get('RawProcess_General', 'mag_dec'))
            if use_geo_north == 'mag':
                mag_dec = None

            dec_date = datetime.datetime.strptime(self.root.get(
                'RawProcess_General', 'dec_date'), r'%Y-%m-%d')
            if use_geo_north == 'mag':
                dec_date = None

            return dict(
                method=use_geo_north,
                magnetic_declination=mag_dec,
                declination_date=dec_date)

        def set_output_id(self, output_id: str):
            """
            Set the output id. This will be appended to each output file, so short ids are recommended. 
            The characters | \ / : ; ? * ' \" < > CR LF TAB SPACE and other non readable characters are prohibited, but we don't check for this.
            Additionally, underscores "_" are prohibited. This is to help with output file parsing. We recommend using hyphens ("-") instead.

            Parameters
            ----------
            output_id: the output id to use. Must not includes underscores.
            """
            assert ' ' not in output_id and '_' not in output_id, 'output id must not contain spaces or underscores.'

            history_args = ('Basic', 'project_id', self.get_output_id)
            self.root._add_to_history(*history_args, True)
            self.root.set('Project', 'project_id', str(output_id))
            self.root._add_to_history(*history_args)
        def get_output_id(self) -> str:
            """retrieve form the config file the project project ID"""
            return dict(output_id=self.root.get('Project', 'project_id'))

        def set_wind_direction_filter(self, enable: bool = False, sectors: Sequence[Sequence[int | float]] | None = None):
            """configure the wind direction filter for raw data. Any high frequency wind data originating from the designated sectors will be removed from the dataset
            e.g. a wind filter of 170-190 will filter out any wind originating from the south. Note that this means that wind direction is the meteorological wind direction not the angle of the wind vector,
            so if your tower sits to the south of your anemometer, you should filter any southerly winds with a wind filter spanning 170°-190°.

            Parameters
            ----------
            enable: if False (default), do not enable wind direction filtering. If True, enable wind direction filtering as specified by sectors.
            sectors: if None, do not enable wind direction filtering. If provided, wind sectors should be given as a list of tuples, specifying the start and end of each wind sector.
            e.g. the following would specify to filter wind from 2 sectors: sectors=[(0, 90), (170, 190)]."""

            assert isinstance(enable, bool), 'enable must be bool'
            if sectors is not None:
                assert isinstance(sectors, Sequence), 'sectors must be a sequence of 2-tuples'
                for sector in sectors:
                    assert isinstance(sector, Sequence), 'individual sector must be a 2-tuple'
                    assert len(sector) == 2, 'individual sector must be a 2-tuple'
                    for d in sector:
                        assert or_isinstance(d, int, float), 'sectors must be given as int or float'
                        assert d >= 0 and d <= 360, 'sector boundaries must be given in degrees, between 0 and 360'
            
            history_args = ('Basic', 'wind_direction_filter', self.get_wind_direction_filter)
            self.root._add_to_history(*history_args, True)

            if not enable:
                self.root.set('RawProcess_WindDirectionFilter', 'wdf_apply', '0')
            else:
                self.root.set('RawProcess_WindDirectionFilter', 'wdf_apply', '1')
                for i, sector in enumerate(sectors):
                    self.root.set('RawProcess_WindDirectionFilter', f'wdf_sect_{i + 1}_start', str(float(sector[0])))
                    self.root.set('RawProcess_WindDirectionFilter', f'wdf_sect_{i + 1}_end', str(float(sector[1])))
                for j in range(i + 1, 16):
                    self.root.remove_option('RawProcess_WindDirectionFilter', f'wdf_sect_{j + 1}_start')
                    self.root.remove_option('RawProcess_WindDirectionFilter', f'wdf_sect_{j + 1}_end')

            self.root._add_to_history(*history_args)

        def get_wind_direction_filter(self) -> dict:
            kwargs = dict()
            kwargs['enable'] = bool(int(self.root.get('RawProcess_WindDirectionFilter', 'wdf_apply')))
            if kwargs['enable']:
                sectors = []
                for i in range(16):
                    try:
                        sector = (
                            float(self.root.get('RawProcess_WindDirectionFilter', f'wdf_sect_{i + 1}_start')),
                            float(self.root.get('RawProcess_WindDirectionFilter', f'wdf_sect_{i + 1}_end')),
                        )
                    except configparser.NoOptionError:
                        break
                    sectors.append(sector)
                kwargs['sectors'] = sectors
            else: kwargs['sectors'] = None
            return kwargs
                    

                

    # --------------------Advanced Settings Page-----------------------
    class _Adv:
        """Advanced processing settings"""
        def __init__(self, root):
            self.root = root
            self.Proc = self._Proc(self)
            self.Stat = self._Stat(self)
            self.Spec = self._Spec(self)
            self.Out = self._Out(self)

        # --------Processing Options---------
        class _Proc:
            """Processing options"""
            def __init__(self, outer):
                self.outer = outer
                self.root = outer.root

            def set_wind_speed_measurement_offsets(
                    self, u: float = 0, v: float = 0, w: float = 0):
                """Wind speed measurements by a sonic anemometer may be biased by systematic deviation, which needs to be eliminated (e.g., for a proper assessment of tilt angles).
                
                Parameters
                ----------
                u, v, w: the u, v, and w offsets, in m/s. Must be no greater than 10m/s in magnitude."""
                assert max(u**2, v**2, w**2) <= 100, 'Windspeed measurement offsets cannot exceed ±10m/s'
                
                history_args = ('Advanced-Processing', 'wind_speed_measurement_offsets', self.get_wind_speed_measurement_offsets)
                self.root._add_to_history(*history_args, modify_only_if_first=True)
                self.root.set('RawProcess_Settings', 'u_offset', str(u))
                self.root.set('RawProcess_Settings', 'v_offset', str(v))
                self.root.set('RawProcess_Settings', 'w_offset', str(w))
                self.root._add_to_history(*history_args)
            def get_wind_speed_measurement_offsets(self) -> dict:
                """retrieve form the config file the wind speed measurement offsets in m/s. Can be passed to set_windspeedmeasurementoffsets as kwargs"""
                return dict(
                    u=float(self.root.get('RawProcess_Settings', 'u_offset')),
                    v=float(self.root.get('RawProcess_Settings', 'v_offset')),
                    w=float(self.root.get('RawProcess_Settings', 'w_offset'))
                )

            def _configure_planar_fit_settings(
                self,
                w_max: float,
                u_min: float,
                num_per_sector_min: int,
                start: str | datetime.datetime | Literal['project', 'all_available'] = 'all_available',
                end: str | datetime.datetime | Literal['project', 'all_available'] = 'all_available',
                fix_method: Literal['CW', 'CCW', 'double_rotations'] | int = 'CW',
                north_offset: int = 0,
                sectors: Sequence[Sequence[bool | int, float]] = [(False, 360)],
                return_inputs: bool = False
            ) -> dict:
                """outputs a dictionary of planarfit settings that can be written directly to the .ini file

                Parameters
                ----------
                w_max: the maximum mean vertical wind component for a time interval to be included in the planar fit estimation
                u_min: the minimum mean horizontal wind component for a time interval to be included in the planar fit estimation
                start, end: start and end date-times for planar fit computation. If a string, must be in yyyy-mm-dd HH:MM format, "all_available," or "project." If "project", sets the start/end of the planar fit computation to the CURRENT project start/end dates. Note that if you change the project start/end dates AFTER applying this setting, the planar fit will still use the OLD dates. If 'all_available' (default), have eddypro use all available raw data, whether in the project date window or not. You cannot mix types: ie you cannot provide start='project' and end='all_available'. This differs from the behavior of Proc._configure_timelag_autoopt and Spec.set_calculation, which are far less sensitive to this setting.
                num_per_sector_min: the minimum number of valid datapoints for a sector to be computed.
                fix_method: one of CW, CCW, or double_rotations or 0, 1, 2. The method to use if a planar fit computation fails for a given sector. Either next valid sector clockwise, next valid sector, counterclockwise, or double rotations. Default is next valid sector clockwise.
                north_offset: the offset for the counter-clockwise-most edge of the first sector in degrees from -180 to 180. Default 0.
                sectors: list of tuples of the form (exclude, width). Where exclude is either a bool (False, True), or an int (0, 1) indicating whether to ingore this sector entirely when estimating planar fit coefficients. Width is a float between 0.1 and 359.9 indicating the width, in degrees of a given sector. Widths must add to one. defaults to a single active sector of 360 degrees, [(False, 360)]
                return_inputs: bool (default False), whether to return a dictionary containing kwargs (excluding this one) provided as input. Useful when used in conjunction with set_axis_rotations_for_tilt_correction
                
                limits on inputs:
                * w_max: 0.5-10
                * u_min: 0.001 - 10
                * num_per_sector_min: 1-10_000
                * north_offset: -180 - +180
                * sectors: 1-12 sectors, sectors must total ≤360 degrees, each sector between 0.1 and 360 degrees

                Returns
                -------
                a dictionary to provide to set_axis_rotations_for_tiltCorrection
                """

                # check that inputs conform to requirements
                assert in_range(w_max, '[0.1, 10.0]'), 'w_max must be between 0.1 and 10.0'
                assert in_range(u_min, '[0.001, 10.0]'), 'u_min must be between 0.001 and 10.0'
                assert fix_method in ['CW', 'CCW', 'double_rotations', 0, 1, 2], 'fix_method must be one of CW (0), CCW (1), double_rotations (2)'
                assert in_range(num_per_sector_min, '[1, 10_000]'), 'num_per_sector_min must be between 1 and 10_000'
                assert in_range(north_offset, '[-180, 180]'), 'north_offset must be between -180 and +180'
                assert isinstance(sectors, Sequence), f'sectors must be a sequence. Received {type(sectors)} instead'
                assert or_isinstance(start, str, datetime.datetime), 'starting timestamp must be string or datetime.datetime'
                assert or_isinstance(end, str, datetime.datetime), 'ending timestamp must be string or datetime.datetime'
                if isinstance(start, str):
                    assert len(start) == 16 or start in ['project', 'all_available'], 'if start is a string, it must be a timestamp of the form YYYY-mm-dd HH:MM, "all_available", or "project"'
                    if start == 'project':
                        assert end == 'project', 'if one of start, end is "project", the other must be as well.'
                    elif start == 'all_available':
                        assert end == 'all_available', 'if one of start, end is "all_available", the other must be as well.'
                if isinstance(end, str):
                    assert len(end) == 16 or end in ['project', 'all_available'], 'if end is a string, it must be a timestamp of the form YYYY-mm-dd HH:MM, "all_available", or "project"'
                    if end == 'project':
                        assert start == 'project', 'if one of start, end is "project", the other must be as well.'
                    if end == 'all_available':
                        assert start == 'all_available', 'if one of start, end is "all_available", the other must be as well.'
                assert isinstance(sectors, Sequence), 'sectors must be a sequence'
                assert len(sectors) >= 1, 'must provide at least one sector'
                assert len(sectors) <= 12, f'was given {len(sectors)} sectors. No more than 12 are permitted'
                total_width = 0
                for i, s in enumerate(sectors):
                    assert isinstance(s, Sequence), f'Each sector must be a seqeuence. Received {type(s)} for sector {i}'
                    assert len(s) == 2, f'Each sector must be of the form (exclude, width). Received {type(s)} of length {len(s)} for sector {i}'
                    assert or_isinstance(s[0], bool, int), f'The first entry in each sector must be a bool or an int. Received {type(s[0])} for sector {i}'
                    assert or_isinstance(s[1], bool, float, int), f'The second entry in each sector must be a float or an int. Received {type(s[1])} for sector {i}'
                    assert s[1] >= 0.1, f'Each sector must be greater or equal to 0.1° wide. Received width={s[1]}° for sector {i}'
                    total_width += s[1]
                assert total_width <= 360., f'Sum of sector widths must not exceed 360 degrees. Given sectors total {total_width}°'

                # process dates
                # if user specifies "project," we choose start and end dates, but they don't end up mattering because we set pf_subset = 0
                settings_dict = dict()
                
                match start, end:
                    case 'all_available', 'all_available':
                        settings_dict['pf_subset'] = 0
                    case 'project', 'project':
                        if self.root.Basic.get_project_date_range()['start'] == 'all_available':
                            settings_dict['pf_subset'] = 0
                        else:
                            settings_dict['pf_subset'] = 1
                            start, end = self.root.Basic.get_project_date_range().values()
                            start_date, start_time = start.strftime(r'%Y-%m-%d %H:%M').split(' ')
                            end_date, end_time = end.strftime(r'%Y-%m-%d %H:%M').split(' ')
                            settings_dict['pf_start_date'] = start_date
                            settings_dict['pf_start_time'] = start_time
                            settings_dict['pf_end_date'] = end_date
                            settings_dict['pf_end_time'] = end_time
                    case _:
                        settings_dict['pf_subset'] = 1
                        if isinstance(start, datetime.datetime):
                            start = start.strftime(r'%Y-%m-%d %H:%M')
                        if isinstance(end, datetime.datetime):
                            end = end.strftime(r'%Y-%m-%d %H:%M')
                        start_date, start_time = start.split(' ')
                        end_date, end_time = end.split(' ')
                        settings_dict['pf_start_date'] = start_date
                        settings_dict['pf_start_time'] = start_time
                        settings_dict['pf_end_date'] = end_date
                        settings_dict['pf_end_time'] = end_time
                
                # fix method
                fix_dict = dict(CW=0, CCW=1, double_rotations=2)
                if isinstance(fix_method, str):
                    fix_method = fix_dict[fix_method]

                
                settings_dict['pf_u_min'] = u_min
                settings_dict['pf_w_max'] = w_max
                settings_dict['pf_min_num_per_sec'] = int(num_per_sector_min)
                settings_dict['pf_fix'] = fix_method
                settings_dict['pf_north_offset'] = north_offset

                # sectors
                for i, sector in enumerate(sectors):
                    exclude, width = sector
                    n = i + 1
                    settings_dict[f'pf_sect_{n}_exclude'] = int(exclude)
                    settings_dict[f'pf_sect_{n}_width'] = str(width)

                if not return_inputs: return settings_dict
                else:
                    inputs = dict(
                        w_max=w_max,
                        u_min=u_min,
                        num_per_sector_min=num_per_sector_min,
                        start=start,
                        end=end,
                        fix_method=fix_method,
                        north_offset=north_offset,
                        sectors=sectors)
                    return inputs
            def set_axis_rotations_for_tilt_correction(
                self,
                method: Literal['none', 'double_rotations', 'triple_rotations', 'planar_fit', 'planar_fit_nvb'] | int = 'double_rotations',
                pf_file: str | PathLike[str] | None = None,
                configure_planar_fit_settings_kwargs: dict | None = None,
            ):
                """
                Specify how rotate coordinate axes when performing tilt corrections. Note that if you are using a planar fit option (Recommended), you 
                must either provide a planar fit file for configure the planar fit settings (see _configure_planar_fit_settings method)

                Parameters
                ----------
                method: one of 0 or "none" (no tilt correction), 1 or "double_rotations" (default), 2 or "triple_rotations", 3 or "planar_fit" (Wilczak 2001), 4 or "planar_fit_nvb" (planar with with no velocity bias (van Dijk 2004)). If a planar fit-type method is selected, then exactly one of pf_file or pf_settings_kwargs must be provided if method is a planar fit type. 
                pf_file: path to an eddypro-compatible planar fit file. If provided, planar_fit_settings_kwargs must be None. Ignored if a non-planar-fit setting is provided.
                pf_settings_kwargs: Arguments to be passed to _configure_planar_fit_settings. If provided, pf_file must be None. Ignored if a non-planar-fit setting is provided. Note: you can get such a dictionary by calling _configure_planar_fit_kwargs with return_inputs=True.
                """
                history_args = ('Advanced-Processing', 'axis_rotations_for_tilt_correction', self.get_axis_rotations_for_tilt_correction)
                self.root._add_to_history(*history_args, True)

                assert method in ['none', 'double_rotations', 'triple_rotations', 'planar_fit', 'planar_fit_nvb', 0, 1, 2, 3, 4], 'method must be one of none (0), double_rotations (1), triple_rotations (2), planar_fit (3), or planar_fit_nvb (4)'
                if method in ['planar_fit', 'planar_fit_nvb', 3, 4]:
                    assert bool(pf_file) != bool(configure_planar_fit_settings_kwargs), 'If method is a planar-fit type, exactly one of pf_file or pf_settings should be specified.'
                elif pf_file is not None or configure_planar_fit_settings_kwargs is not None:
                    warnings.warn(f'planar fit settings arguments will be ignored when method is not a non-planar-fit type. Received method={method}')
                method_dict = {
                    'none': 0,
                    'double_rotations': 1,
                    'triple_rotations': 2,
                    'planar_fit': 3,
                    'planar_fit_nvb': 4}
                if isinstance(method, str):
                    method = method_dict[method]
                
                self.root.set('RawProcess_Settings', 'rot_meth', str(method))

                # planar fit
                if method in [3, 4]:
                    if pf_file is not None:
                        self.root.set(
                            'RawProcess_TiltCorrection_Settings',
                            'pf_file',
                            str(pf_file))
                        self.root.set(
                            'RawProcess_TiltCorrection_Settings', 'pf_mode', str(0))
                    elif configure_planar_fit_settings_kwargs is not None:
                        self.root.set(
                            'RawProcess_TiltCorrection_Settings', 'pf_file', '')
                        self.root.set(
                            'RawProcess_TiltCorrection_Settings', 'pf_mode', str(1))
                        pf_settings = self._configure_planar_fit_settings(
                            **configure_planar_fit_settings_kwargs)
                        # before we set any sectors, we need to remove all the pre-existing sectors. 
                        # this solves the problem of 5 sectors already existing, but the user only provides ones, resulting in
                        # more sectors that the user specified or invalid sectors.
                        for i in range(1, 17):
                            self.root.remove_option('RawProcess_WindDirectionFilter', f'pf_sect_{i}_exclude')
                            self.root.remove_option('RawProcess_WindDirectionFilter', f'pf_sect_{i}_width')
                        # now we add the planar fit settings, including sector information.
                        for option, value in pf_settings.items():
                            self.root.set(
                                'RawProcess_TiltCorrection_Settings', option, str(value))
                
                self.root._add_to_history(*history_args)
            def get_axis_rotations_for_tilt_correction(self) -> dict:
                """
                extracts axis rotation settings from the config file.
                Returns a dictionary that containing a dictionary of kwargs that can be passed to set_axis_rotations_for_tiltCorrection
                """

                methods = [
                    'none',
                    'double_rotations',
                    'triple_rotations',
                    'planar_fit',
                    'planar_fit_nvb']
                method = methods[int(self.root.get(
                    'RawProcess_Settings', 'rot_meth'))]
                # initially set planar fit config to none
                configure_planar_fit_settings_kwargs = None
                pf_file = None

                # if we have planar fit, then returna  dict for pf_config that
                # can be passed to _configure_planar_fit_settings
                # note that if pf_subset == 1, we don't grab the dates
                if method in ['planar_fit', 'planar_fit_nvb']:
                    configure_planar_fit_settings_kwargs = dict()
                    # case that a manual configuration is provided
                    pf_subset = int(self.root.get('RawProcess_TiltCorrection_Settings', 'pf_subset'))
                    if not pf_subset:
                        configure_planar_fit_settings_kwargs['start'] = 'all_available'
                        configure_planar_fit_settings_kwargs['end'] = 'all_available'
                    else:
                        start_date = self.root.get(
                            'RawProcess_TiltCorrection_Settings', 'pf_start_date')
                        start_time = self.root.get(
                            'RawProcess_TiltCorrection_Settings', 'pf_start_time')
                        if not start_date:
                            start_date = self.root.get('Project', 'pr_start_date')
                        if not start_time:
                            start_time = self.root.get('Project', 'pr_start_time')
                        configure_planar_fit_settings_kwargs['start'] = start_date + \
                            ' ' + start_time
                        end_date = self.root.get(
                            'RawProcess_TiltCorrection_Settings', 'pf_end_date')
                        end_time = self.root.get(
                            'RawProcess_TiltCorrection_Settings', 'pf_end_time')
                        if not end_date:
                            end_date = self.root.get('Project', 'pr_end_date')
                        if not end_time:
                            end_time = self.root.get('Project', 'pr_end_time')
                        configure_planar_fit_settings_kwargs['end'] = end_date + \
                            ' ' + end_time

                    configure_planar_fit_settings_kwargs['u_min'] = float(
                        self.root.get('RawProcess_TiltCorrection_Settings', 'pf_u_min'))
                    configure_planar_fit_settings_kwargs['w_max'] = float(
                        self.root.get('RawProcess_TiltCorrection_Settings', 'pf_w_max'))
                    configure_planar_fit_settings_kwargs['num_per_sector_min'] = int(
                        self.root.get('RawProcess_TiltCorrection_Settings', 'pf_min_num_per_sec'))
                    fixes = ['CW', 'CCW', 'double_rotations']
                    configure_planar_fit_settings_kwargs['fix_method'] = fixes[int(
                        self.root.get('RawProcess_TiltCorrection_Settings', 'pf_fix'))]
                    configure_planar_fit_settings_kwargs['north_offset'] = float(
                        self.root.get('RawProcess_TiltCorrection_Settings', 'pf_north_offset'))

                    n = 1
                    sectors = []
                    while True:
                        try:
                            exclude = int(
                                self.root.get(
                                    'RawProcess_TiltCorrection_Settings',
                                    f'pf_sect_{n}_exclude'))
                            width = float(
                                self.root.get(
                                    'RawProcess_TiltCorrection_Settings',
                                    f'pf_sect_{n}_width'))
                            sectors.append((exclude, width))
                        except configparser.NoOptionError:
                            break
                        n += 1
                    configure_planar_fit_settings_kwargs['sectors'] = sectors

                    # case that a file config is provided
                    manual_pf_config = int(
                        self.root.get(
                            'RawProcess_TiltCorrection_Settings',
                            'pf_mode'))
                    if not manual_pf_config:
                        pf_file = self.root.get(
                            'RawProcess_TiltCorrection_Settings', 'pf_file')
                        configure_planar_fit_settings_kwargs = None

                return dict(
                    method=method,
                    pf_file=pf_file,
                    configure_planar_fit_settings_kwargs=configure_planar_fit_settings_kwargs)

            def set_turbulent_fluctuations(
                self,
                detrend_method: Literal['block',
                                'linear',
                                'running_mean',
                                'exponential_running_mean'] | int = 'block',
                time_constant: float | Literal['averaging_interval'] | None = None):
                '''
                Configure how to extract turbulent fluctuating component from wind and concentration data.

                Parameters
                ----------
                detrend_method: one of 'block' (0), 'detrend (1), running_mean (2), or exponential_running_mean (3). Default 'block'
                time_constant: if detrend, running_mean, or exponential_running_mean are selected, provide a time constant in minutes. Default None. If None and linear_detrend is selected, set time_constant to 0 to indicate to eddypro to use the flux averaging interval as the time constant. If a running mean method is selected and time_constant is None, set time_constant to 250s.
                    detrend_method              default time_constant
                    block                       0 (does nothing)
                    linear                      0 (flux averaging interval)
                    running_mean                250 (seconds)
                    exponential_running_mean    250 (seconds)

                limits:
                time_constant must be between 0 and 5000 minutes
                '''
                history_args = ('Advanced-Processing', 'turbulent_fluctuations', self.get_turbulent_fluctuations)
                self.root._add_to_history(*history_args, True)

                assert detrend_method in ['block', 'linear', 'running_mean', 'exponential_running_mean', 0, 1, 2, 3], "detrend_method must be one of 'block' (0), 'linear' (1), running_mean (2), or exponential_running_mean (3)"
                if time_constant is not None:
                    assert or_isinstance(time_constant, int, float) or time_constant == 'averaging_interval', 'time constant must be numeric, "averaging_interval" or None'
                    assert in_range(time_constant, '[0, 5000.]'), 'time constant must be between 0 and 5000'

                # choose method
                method_dict = {
                    'block': 0,
                    'linear': 1,
                    'running_mean': 2,
                    'exponential_running_mean': 3}
                if isinstance(detrend_method, str):
                    detrend_method = method_dict[detrend_method]
                
                # choose time constant
                default_time_constants = [0, 0, 250/60, 250/60]
                if time_constant is None:
                    # default for linear detrend is flux averaging interval
                    time_constant = default_time_constants[detrend_method]
                elif time_constant == 'averaging_interval':
                    time_constant = 0
                self.root.set(
                    'RawProcess_Settings',
                    'detrend_meth',
                    str(detrend_method))
                if detrend_method != 0:
                    self.root.set(
                        'RawProcess_Settings',
                        'timeconst',
                        str(time_constant*60))

                self.root._add_to_history(*history_args)
            def get_turbulent_fluctuations(self) -> dict:
                out = dict()

                methods = ['block', 'linear', 'running_mean', 'exponential_running_mean']
                out['detrend_method'] = methods[int(self.root.get('RawProcess_Settings', 'detrend_meth'))]
                if out['detrend_method'] != 'block':
                    out['time_constant'] = float(self.root.get('RawProcess_Settings', 'timeconst'))

                return out

            def _configure_timelag_auto_opt(
                self,
                start: str | datetime.datetime | Literal['project'] = 'project',
                end: str | datetime.datetime | Literal['project'] = 'project',
                pg_range: float = 1.5,
                n_rh_classes: int = 10,
                le_min_flux: float = 20.0,
                co2_min_flux: float = 2.000,
                ch4_min_flux: float = 0.200,
                gas4_min_flux: float = 0.020,
                co2_lags: Sequence[float, float] | None = None,
                ch4_lags: Sequence[float, float] | None = None,
                h2o_lags: Sequence[float, float] | None = None,
                gas4_lags: Sequence[float, float] | None = None,
                
                
                
            ) -> dict:
                """
                Generate configuration options for automatic time lag optimization method. Returns a dictionary that can be used to directly modify the ini file.

                Parameters
                ----------
                start, end: start and end date-times for time lag optimization computation. If a string, must be in yyyy-mm-dd HH:MM format or "project." If "project"  (default), sets the start/end to the project start/end date. If one of start, end is project, the other must be as well.
                pg_range: the number of median absolute deviations from the mean a time lag can be for a given class to be accepted. Default mean±1.5mad
                n_rh_classes: the number of relative humidity classes to consider when optimizing H2O time lags
                XXX_min_flux: the minimum flux for a given gas, quantity, in µmol/m2/2, except for le_min_flux, which is in units of W/m2
                XXX_lags: a sequence of (min, max) seconds of lag to define the time lag searching windows. If None, tell eddypro to determine automatically.
                
                limits on inputs:
                pg_range: [0.1, 100]
                n_rh_classes: [1, 20]
                le_min_flux: [0, 1000]
                co2_min_flux: 
                ch4_min_flux: [0, 100]
                gas4_min_flux: [0, 100]
                min/max_lag: [-10_000, +10_000]

                """

                # check inputs
                assert or_isinstance(start, str, datetime.datetime), 'starting timestamp must be string or datetime.datetime'
                assert or_isinstance(end, str, datetime.datetime), 'ending timestamp must be string or datetime.datetime'
                if isinstance(start, str):
                    assert len(start) == 16 or start == 'project', 'if start is a string, it must be a timestamp of the form YYYY-mm-dd HH:MM or "project"'
                    if start == 'project':
                        assert end == 'project', 'if one of start, end is "project", the other must be as well.'
                if isinstance(end, str):
                    assert len(end) == 16 or end == 'project', 'if end is a string, it must be a timestamp of the form YYYY-mm-dd HH:MM or "project"'
                    if end == 'project':
                        assert start == 'project', 'if one of start, end is "project", the other must be as well.'
                assert or_isinstance(pg_range, float, int) and in_range(pg_range, '[0.1, 100]'), f'pg_range must be float or int and between 0.1 and 100'
                assert or_isinstance(n_rh_classes, int) and in_range(n_rh_classes, '[1, 20]'), f'h2o_nclass must be int and between 1 and 20'
                assert or_isinstance(le_min_flux, float, int) and in_range(le_min_flux, '[0, 1000]'), f'le_min_flux must be float or int and in range [0, 1000]'
                assert or_isinstance(co2_min_flux, float, int) and in_range(co2_min_flux, '[0, 100]'), f'co2_min_flux must be float or int and in range [0, 100]'
                assert or_isinstance(ch4_min_flux, float, int) and in_range(ch4_min_flux, '[0, 100]'), f'ch4_min_flux must be float or int and in range [0, 100]'
                assert or_isinstance(gas4_min_flux, float, int) and in_range(gas4_min_flux, '[0, 100]'), f'gas4_min_flux must be float or int and in range [0, 100]'
                for k, v in dict(co2_lags=co2_lags, ch4_lags=ch4_lags, h2o_lags=h2o_lags, gas4_lags=gas4_lags).items():
                    if v is not None:
                        assert isinstance(v, Sequence), f'{k} must be None, or a sequence of 2 numbers'
                        assert len(v) == 2, f'{k} must be None, or a sequence of 2 numbers'
                        assert v[0] < v[1], f'time lag search window must have positive width. Received {k}={v}.'
                
                settings_dict = dict()
                # process dates
                settings_dict['to_subset'] = 1
                if start == 'project':
                    settings_dict['to_subset'] = 0
                elif isinstance(start, datetime.datetime):
                    to_start = start
                    settings_dict['to_start_date'], settings_dict['to_start_time'] = to_start.strftime(r'%Y-%m-%d %H:%M').split(' ')
                else:
                    to_start = start
                    settings_dict['to_start_date'], settings_dict['to_start_time'] = to_start.split(' ')

                if end == 'project':
                    pass
                elif isinstance(end, datetime.datetime):
                    to_end = end
                    settings_dict['to_end_date'], settings_dict['to_end_time'] = to_end.strftime(r'%Y-%m-%d %H:%M').split(' ')
                else:
                    to_end = end
                    settings_dict['to_end_date'], settings_dict['to_end_time'] = to_end.split(' ')
                
                # lag settings default to "automatic detection" for the value -1000.1
                settings_with_special_defaults = [
                    h2o_lags,
                    ch4_lags,
                    co2_lags,
                    gas4_lags]
                for i, setting in enumerate(settings_with_special_defaults):
                    if setting is None:
                        settings_with_special_defaults[i] = (-10000.1, -10000.1)
                h2o_min_lag, h2o_max_lag = h2o_lags
                co2_min_lag, co2_max_lag = ch4_lags
                ch4_min_lag, ch4_max_lag = co2_lags
                gas4_min_lag, gas4_max_lag = gas4_lags

                settings_dict.update(dict(
                    to_ch4_min_lag=ch4_min_lag,
                    to_ch4_max_lag=ch4_max_lag,
                    to_ch4_min_flux=ch4_min_flux,
                    to_co2_min_lag=co2_min_lag,
                    to_co2_max_lag=co2_max_lag,
                    to_co2_min_flux=co2_min_flux,
                    to_gas4_min_lag=gas4_min_lag,
                    to_gas4_max_lag=gas4_max_lag,
                    to_gas4_min_flux=gas4_min_flux,
                    to_h2o_min_lag=h2o_min_lag,
                    to_h2o_max_lag=h2o_max_lag,
                    to_le_min_flux=le_min_flux,
                    to_h2o_nclass=int(n_rh_classes),
                    to_pg_range=pg_range,
                    to_subset=to_subset
                ))

                return settings_dict
            def set_timelag_compensations(self,
                                          method: Literal['none',
                                                          'constant',
                                                          'covariance_maximization_with_default',
                                                          'covariance_maximization',
                                                          'automatic_optimization'] | int = 'covariance_maximization_with_default',
                                          autoopt_file: PathLike[str] | str | None = None,
                                          autoopt_settings_kwargs: dict | None = None):
                """
                Configure how to compensate for time lags between instruments. Note that if automatic optimization is selected, an autoopt file must be provided, or configure_TimeLagAutoOpt_kwargs must be provided.

                Parameters
                ----------
                method: one of 0 or "none" (no time lag compensation), 1 or "constant" (constant time lag from instrument metadata), 2 or "covariance_maximization_with_default" (Default), 3 or "covariance_maximization", or 4 or "automatic_optimization." one of autoopt_file or autoopt_settings_kwargs must be provided if method is a planar fit type.
                autoopt_file: Mututally exclusive with autoopt_settings_kwargs. If method is a planar fit type, path to an eddypro-compatible automatic time lag optimization file. This can be build by hand, or taken from the output of a previous eddypro run. Typically labelled as "eddypro_<project id>_timelag_opt_<timestamp>_adv.txt" or similar
                autoopt_settings_kwargs: Mututally exclusive with autoopt_file. Arguments to be passed to configure_TimelagAutoOpt.
                """
                history_args = ('Advanced-Processing', 'timelag_compensations', self.get_timelag_compensations)
                self.root._add_to_history(*history_args, True)

                # check inputs
                if isinstance(method, str):
                    assert method in ['none', 'constant', 'covariance_maximization_with_default', 'covariance_maximization', 'automatic_optimization'], "method must be one of None, 'none', 'constant', 'covariance_maximization_with_default', 'covariance_maximization', 'automatic_optimization', or 0, 1, 2, 3, or 4."
                else: 
                    assert method in range(5), 'method must be one of None, constant, covariance_maximization_with_default, covariance_maximization, automatic_optimization, or 0, 1, 2, 3, or 4.'
                if method == 4 or method == 'automatic_optimization':
                    assert bool(autoopt_file) != bool(autoopt_settings_kwargs), 'If method is automatic_optimization, exactly one of autoopt_file or configure_TimelagAutoOpt_kwargs should be specified.'
                    if autoopt_settings_kwargs is not None:
                        assert isinstance(autoopt_settings_kwargs, dict), 'configure_TimelagAutoOpt_kwargs must be None or dict.'

                method_dict = {
                    'none': 0,
                    'constant': 1,
                    'covariance_maximization_with_default': 2,
                    'covariance_maximization': 3,
                    'automatic_optimization': 4}
                if isinstance(method, str):
                    method = method_dict[method]

                self.root.set('RawProcess_Settings', 'tlag_meth', str(method))

                # planar fit
                if method == 4:
                    if autoopt_file is not None:
                        self.root.set(
                            'RawProcess_TimelagOptimization_Settings',
                            'to_file',
                            str(autoopt_file))
                        self.root.set(
                            'RawProcess_TimelagOptimization_Settings', 'to_mode', str(0))
                    elif autoopt_settings_kwargs is not None:
                        self.root.set(
                            'RawProcess_TimelagOptimization_Settings', 'to_file', '')
                        self.root.set(
                            'RawProcess_TimelagOptimization_Settings', 'to_mode', str(1))
                        to_settings = self._configure_timelag_auto_opt(
                            **autoopt_settings_kwargs)
                        for option, value in to_settings.items():
                            self.root.set(
                                'RawProcess_TimelagOptimization_Settings', option, str(value))
                self.root._add_to_history(*history_args)
            def get_timelag_compensations(self) -> dict:
                """
                extracts time lag compensation settings from the config file.
                Returns a dictionary that containing a dictionary of kwargs that can be passed to set_time_lag_compensations_
                """

                methods = [
                    'none',
                    'constant',
                    'covariance_maximization_with_default',
                    'covariance_maximization',
                    'automatic_optimization']
                method = methods[int(self.root.get(
                    'RawProcess_Settings', 'tlag_meth'))]
                configure_TimelagAutoOpt_kwargs = None
                autoopt_file = None

                if method == 'automatic_optimization':
                    configure_TimelagAutoOpt_kwargs = dict()
                    to_subset = int(self.root.get('RawProcess_TimelagOptimization_Settings', 'to_subset'))
                    if not to_subset:
                        configure_TimelagAutoOpt_kwargs['start'] = 'project'
                        configure_TimelagAutoOpt_kwargs['end'] = 'project'
                    else:
                        # dates for autoopt fitting
                        start_date = self.root.get(
                            'RawProcess_TimelagOptimization_Settings', 'to_start_date')
                        start_time = self.root.get(
                            'RawProcess_TimelagOptimization_Settings', 'to_start_time')
                        if not start_date:
                            start_date = self.root.get('Project', 'pr_start_date')
                        if not start_time:
                            start_time = self.root.get('Project', 'pr_start_time')
                        configure_TimelagAutoOpt_kwargs['start'] = start_date + \
                            ' ' + start_time
                        end_date = self.root.get(
                            'RawProcess_TimelagOptimization_Settings', 'to_end_date')
                        end_time = self.root.get(
                            'RawProcess_TimelagOptimization_Settings', 'to_end_time')
                        if not end_date:
                            end_date = self.root.get('Project', 'pr_end_date')
                        if not end_time:
                            end_time = self.root.get('Project', 'pr_end_time')
                        configure_TimelagAutoOpt_kwargs['end'] = end_date + \
                            ' ' + end_time

                    configure_TimelagAutoOpt_kwargs['ch4_min_lag'] = self.root.get(
                        'RawProcess_TimelagOptimization_Settings', 'to_ch4_min_lag')
                    configure_TimelagAutoOpt_kwargs['ch4_max_lag'] = self.root.get(
                        'RawProcess_TimelagOptimization_Settings', 'to_ch4_max_lag')
                    configure_TimelagAutoOpt_kwargs['ch4_min_flux'] = self.root.get(
                        'RawProcess_TimelagOptimization_Settings', 'to_ch4_min_flux')
                    configure_TimelagAutoOpt_kwargs['co2_min_lag'] = self.root.get(
                        'RawProcess_TimelagOptimization_Settings', 'to_co2_min_lag')
                    configure_TimelagAutoOpt_kwargs['co2_max_lag'] = self.root.get(
                        'RawProcess_TimelagOptimization_Settings', 'to_co2_max_lag')
                    configure_TimelagAutoOpt_kwargs['co2_min_flux'] = self.root.get(
                        'RawProcess_TimelagOptimization_Settings', 'to_co2_min_flux')
                    configure_TimelagAutoOpt_kwargs['gas4_min_lag'] = self.root.get(
                        'RawProcess_TimelagOptimization_Settings', 'to_gas4_min_lag')
                    configure_TimelagAutoOpt_kwargs['gas4_max_lag'] = self.root.get(
                        'RawProcess_TimelagOptimization_Settings', 'to_gas4_max_lag')
                    configure_TimelagAutoOpt_kwargs['gas4_min_flux'] = self.root.get(
                        'RawProcess_TimelagOptimization_Settings', 'to_gas4_min_flux')
                    configure_TimelagAutoOpt_kwargs['h2o_min_lag'] = self.root.get(
                        'RawProcess_TimelagOptimization_Settings', 'to_h2o_min_lag')
                    configure_TimelagAutoOpt_kwargs['h2o_max_lag'] = self.root.get(
                        'RawProcess_TimelagOptimization_Settings', 'to_h2o_max_lag')
                    configure_TimelagAutoOpt_kwargs['le_min_flux'] = self.root.get(
                        'RawProcess_TimelagOptimization_Settings', 'to_le_min_flux')
                    configure_TimelagAutoOpt_kwargs['h2o_nclass'] = self.root.get(
                        'RawProcess_TimelagOptimization_Settings', 'to_h2o_nclass')
                    configure_TimelagAutoOpt_kwargs['pg_range'] = self.root.get(
                        'RawProcess_TimelagOptimization_Settings', 'to_pg_range')

                    manual_mode = int(
                        self.root.get(
                            'RawProcess_TimelagOptimization_Settings',
                            'to_mode'))
                    if not manual_mode:
                        for k in configure_TimelagAutoOpt_kwargs:
                            configure_TimelagAutoOpt_kwargs = None
                            autoopt_file = self.root.get(
                                'RawProcess_TimelagOptimization_Settings', 'to_file')

                return dict(
                    method=method,
                    autoopt_file=autoopt_file,
                    configure_TimelagAutoOpt_kwargs=configure_TimelagAutoOpt_kwargs)

            def _set_burba_coeffs(self, name, estimation_method, coeffs):
                """helper method called by set_compensationOfDensityFluctuations"""
                if estimation_method == 'multiple':
                    options = [f'm_{name}{i}' for i in [1, 2, 3, 4]]
                    assert len(
                        coeffs) == 4, 'Multiple regression coefficients must be a sequence of length four, representing (offset, Ta_gain, Rg_gain, U_gain)'
                    for option, value in zip(options, coeffs):
                        self.root.set(
                            'RawProcess_Settings', option, str(round(value, 4)))
                elif estimation_method == 'simple':
                    options = [f'l_{name}_{i}' for i in ['gain', 'offset']]
                    assert len(
                        coeffs) == 2, 'Simple regression coefficients must be a sequence of length two, representing (gain, offset)'
                    for option, value in zip(options, coeffs):
                        self.root.set(
                            'RawProcess_Settings', option, str(round(value, 4)))
            def set_compensation_of_density_fluctuations(
                    self,
                    enable: bool = True,
                    burba_method: Literal['simple', 'multiple'] | None = None,
                    day_bot: Sequence | Literal['revert'] | None = None,
                    day_top: Sequence | Literal['revert'] | None = None,
                    day_spar: Sequence | Literal['revert'] | None = None,
                    night_bot: Sequence | Literal['revert'] | None = None,
                    night_top: Sequence | Literal['revert'] | None = None,
                    night_spar: Sequence | Literal['revert'] | None = None,
                    set_all: Literal['revert'] | None = None,
            ):
                """Configure how to correct for density fluctuations. Default mode is to only correct for bulk density fluctuations.

                Parameters
                ----------
                enable: If true, correct for density fluctuations with the WPL term (default)
                burba_correction: If true, add instrument sensible heat components. LI-7500 only. Default False.
                estimation_method: one of 'simple' or 'multiple'. Whether to use simple linear regression or Multiple linear regression. if burba_correction is enabled, this argument cannot be None (default)
                day/night_bot/top/spar: Either (a) 'revert' (revert to defaults) (b) None (default, keep current settings), or (c) a sequence of regression coefficients for the burba correction for the temperature of the bottom, top, and spar of the LI7500.
                    If 'simple' estimation was selected, then this is a sequence of length two, representing (gain, offset) for the equation
                        T_instrument = gain*Ta + offset
                    If 'multiple' estimation was selected, then this is a sequence of length 4, repressinting (offset, Ta_coeff, Rg_coeff, U_coeff) for the equation
                        T_instr - Ta = offset + Ta_coeff*Ta + Rg_coeff*Rg + U_coeff*U
                        where Ta is air temperature, T_instr is instrument part temperature, Rg is global incoming SW radiation, and U is mean windspeed

                    If 'revert,' then revert to default eddypro coefficients.
                    If None (selected by default), then do not change regression coefficients in the file
                set_all: as an alternative to specifying day/night_bot/top/spar, you can provide all = 'revert' to revert all burba correction settings to their eddypro defaults. Default None (do nothing).
                """
                history_args = ('Advanced-Processing', 'compensation_of_density_fluctuations', self.get_compensation_of_density_fluctuations)
                self.root._add_to_history(*history_args, True)
                if not enable:
                    self.root.set('Project', 'wpl_meth', '0')
                    if burba_method is not None:
                        warnings.warn(
                            'burba correction has no effect when density fluctuation compensation is disabled')
                else:
                    self.root.set('Project', 'wpl_meth', '1')

                if burba_method is None:
                    self.root.set('RawProcess_Settings', 'bu_corr', '0')
                    if (
                        isinstance(day_bot, Sequence)
                        or isinstance(day_top, Sequence)
                        or isinstance(day_spar, Sequence)
                        or isinstance(night_bot, Sequence)
                        or isinstance(night_top, Sequence)
                        or isinstance(night_spar, Sequence)
                        or not enable
                    ):
                        warnings.warn(
                            'burba regression coefficients have no effect when burba correction is disabled or density corrections are disabled.')
                else:
                    assert burba_method in [
                        'simple', 'multiple'], 'estimation method must be one of "simple", "multiple"'
                    self.root.set('RawProcess_Settings', 'bu_corr', '1')

                if burba_method == 'simple':
                    self.root.set('RawProcess_Settings', 'bu_multi', '0')
                    # daytime
                    if day_bot == 'revert' or set_all == 'revert':
                        self._set_burba_coeffs(
                            'day_bot', 'simple', (0.944, 2.57))
                    elif day_bot is None:
                        pass
                    else:
                        self._set_burba_coeffs('day_bot', 'simple', day_bot)

                    if day_top == 'revert' or set_all == 'revert':
                        self._set_burba_coeffs(
                            'day_top', 'simple', (1.005, 0.24))
                    elif day_top is None:
                        pass
                    else:
                        self._set_burba_coeffs('day_top', 'simple', day_top)

                    if day_spar == 'revert' or set_all == 'revert':
                        self._set_burba_coeffs(
                            'day_spar', 'simple', (1.010, 0.36))
                    elif day_spar is None:
                        pass
                    else:
                        self._set_burba_coeffs('day_spar', 'simple', day_spar)

                    # nighttime
                    if night_bot == 'revert' or set_all == 'revert':
                        self._set_burba_coeffs(
                            'night_bot', 'simple', (0.883, 2.17))
                    elif night_bot is None:
                        pass
                    else:
                        self._set_burba_coeffs(
                            'night_bot', 'simple', night_bot)

                    if night_top == 'revert' or set_all == 'revert':
                        self._set_burba_coeffs(
                            'night_top', 'simple', (1.008, -0.41))
                    elif night_top is None:
                        pass
                    else:
                        self._set_burba_coeffs(
                            'night_top', 'simple', night_top)

                    if night_spar == 'revert' or set_all == 'revert':
                        self._set_burba_coeffs(
                            'night_spar', 'simple', (1.010, -0.17))
                    elif night_spar is None:
                        pass
                    else:
                        self._set_burba_coeffs(
                            'night_spar', 'simple', night_spar)

                elif burba_method == 'multiple':
                    self.root.set('RawProcess_Settings', 'bu_multi', '1')
                    # daytime
                    if day_bot == 'revert' or set_all == 'revert':
                        self._set_burba_coeffs(
                            'day_bot', 'multiple', (2.8, -0.0681, 0.0021, -0.334))
                    elif day_bot is None:
                        pass
                    else:
                        self._set_burba_coeffs('day_bot', 'multiple', day_bot)

                    if day_top == 'revert' or set_all == 'revert':
                        self._set_burba_coeffs(
                            'day_top', 'multiple', (-0.1, -0.0044, 0.011, -0.022))
                    elif day_top is None:
                        pass
                    else:
                        self._set_burba_coeffs('day_top', 'multiple', day_top)

                    if day_spar == 'revert' or set_all == 'revert':
                        self._set_burba_coeffs(
                            'day_spar', 'multiple', (0.3, -0.0007, 0.0006, -0.044))
                    elif day_spar is None:
                        pass
                    else:
                        self._set_burba_coeffs(
                            'day_spar', 'multiple', day_spar)

                    # nighttime
                    if night_bot == 'revert' or set_all == 'revert':
                        self._set_burba_coeffs(
                            'night_bot', 'multiple', (0.5, -0.1160, 0.0087, -0.206))
                    elif night_bot is None:
                        pass
                    else:
                        self._set_burba_coeffs(
                            'night_bot', 'multiple', night_bot)

                    if night_top == 'revert' or set_all == 'revert':
                        self._set_burba_coeffs(
                            'night_top', 'multiple', (-1.7, -0.0160, 0.0051, -0.029))
                    elif night_top is None:
                        pass
                    else:
                        self._set_burba_coeffs(
                            'night_top', 'multiple', night_top)

                    if night_spar == 'revert' or set_all == 'revert':
                        self._set_burba_coeffs(
                            'night_spar', 'multiple', (-2.1, -0.0200, 0.0070, 0.026))
                    elif night_spar is None:
                        pass
                    else:
                        self._set_burba_coeffs(
                            'night_spar', 'multiple', night_spar)
                self.root._add_to_history(*history_args)
            def get_compensation_of_density_fluctuations(self) -> dict:

                out = dict()
                out['enable'] = bool(int(self.root.get('Project', 'wpl_meth')))
                if out['enable']:
                    out['burba_correction'] = bool(
                        int(self.root.get('RawProcess_Settings', 'bu_corr')))

                    if out['burba_correction']:
                        kwargs = [
                            'day_bot',
                            'day_top',
                            'day_spar',
                            'night_bot',
                            'night_top',
                            'night_spar']
                        use_multiple = int(
                            self.root.get(
                                'RawProcess_Settings',
                                'bu_multi'))
                        if use_multiple:
                            for k in kwargs:
                                out[k] = tuple(
                                    float(
                                        self.root.get('RawProcess_Settings', f'm_{k}{i}')) for i in range(1, 5))
                        else:
                            for k in kwargs:
                                out[k] = tuple(
                                    float(
                                        self.root.get('RawProcess_Settings',f'l_{k}_{i}')) for i in ['gain', 'offset'])

                return out
         # --------Statistical Analysis---------

        class _Stat:
            """Statistical test options"""
            def __init__(self, outer):
                self.root = outer.root
                self.outer = outer

            def set_spike_count_removal(
                    self,
                    enable: bool | int = True,
                    method: Literal['VM97', 'M13'] | int = 'VM97',
                    accepted: float = 1.0,
                    linterp: bool | int = True,
                    max_consec_outliers: int = 3,
                    w: float = 5.0,
                    co2: float = 3.5,
                    h2o: float = 3.5,
                    ch4: float = 8.0,
                    gas4: float = 8.0,
                    others: float = 3.5
            ):
                """Settings for spike count and removaal.

                Parameters
                ----------
                enable: whether to enable despiking. Default True
                method: one of 'VM97' or 'M13' for Vickers & Mart 1997 or Mauder et al 2013. Default 'VM97'. If M13 is selected, only the accepted and linterp options are used.
                accepted: If, for each variable in the flux averaging period, the number of spikes is larger than accepted% of the number of data samples, the variable is hard-flagged for too many spikes. Default 1%
                linterp: whether to linearly interpolate removed spikes (True, default) or to leave them as nan (False)
                max_consec_outliers: for each variable, a spike is detected as up to max_consec_outliers outliers. If more consecutive values are found to exceed the plausibility threshold, they are not flagged as spikes. Default 3.
                w/co2/h2o/ch4/gas4/others: z-score cutoffs for flagging outliers. Defaults are 5.0, 3.5, 3.5, 8.0, 8.0, 3.5, respectively.

                limits on inputs:
                accepted: 0-50%
                consecutive outliers: 3-1000
                z-scores: 1-20
                """
                history_args = ('Advanced-Statistical', 'spike_count_removal', self.get_spike_count_removal)
                self.root._add_to_history(*history_args, True)
                assert or_isinstance(
                    enable, int, bool), 'enable should be int or bool'
                assert method in [
                    'VM97', 'M13', 0, 1], 'method should be one of VM97 (0) or M13 (1)'
                assert in_range(
                    accepted, '[0, 50]'), 'accepted spikes should be be between 0 and 50%'
                assert or_isinstance(
                    linterp, bool, int), 'linterp should be int or bool'
                assert isinstance(max_consec_outliers, int) and in_range(
                    max_consec_outliers, '[3, 1000]'), 'max_consec_outliers should be int from 3 to 1000'
                for v in [w, co2, h2o, ch4, gas4, others]:
                    assert in_range(
                        v, '[1, 20]'), 'variable limits should be between 1 and 20'

                # enable
                if not enable:
                    self.root.set('RawProcess_Tests', 'test_sr', '0')
                    return
                self.root.set('RawProcess_Tests', 'test_sr', '1')

                # enable vm97?
                methods = {'VM97': 0, 'M13': 1}
                if method in methods:
                    method = methods[method]
                use_m13 = method
                self.root.set(
                    'RawProcess_ParameterSettings',
                    'despike_vm',
                    str(use_m13))

                # accepted spikes and linterp
                self.root.set(
                    'RawProcess_ParameterSettings',
                    'sr_lim_hf',
                    str(accepted))
                if linterp:
                    self.root.set(
                        'RawProcess_Settings', 'filter_sr', '1')
                else:
                    self.root.set(
                        'RawProcess_Settings', 'filter_sr', '0')
                if use_m13:
                    return  # m13 takes no futher parameters

                # outliers
                self.root.set(
                    'RawProcess_ParameterSettings',
                    'sr_num_spk',
                    str(max_consec_outliers))

                # limits
                for name, v in zip(['w', 'co2', 'h2o', 'ch4', 'n2o', 'u'], [
                                   w, co2, h2o, ch4, gas4, others]):
                    self.root.set(
                        'RawProcess_ParameterSettings',
                        f'sr_lim_{name}',
                        str(v))
                
                self.root._add_to_history(*history_args)
                return
            def get_spike_count_removal(self) -> dict:
                out_dict = dict()
                out_dict['enable'] = bool(
                    int(self.root.get('RawProcess_Tests', 'test_sr')))
                if not out_dict['enable']:
                    return out_dict

                methods = ['VM97', 'M13']
                out_dict['method'] = methods[int(self.root.get(
                    'RawProcess_ParameterSettings', 'despike_vm'))]
                out_dict['accepted'] = float(
                    self.root.get(
                        'RawProcess_ParameterSettings',
                        'sr_lim_hf'))
                out_dict['linterp'] = bool(
                    int(self.root.get('RawProcess_Settings', 'filter_sr')))
                if out_dict['method'] == 'M13':
                    return out_dict

                for name, k in zip(['w', 'co2', 'h2o', 'ch4', 'n2o', 'u'], [
                                   'w', 'co2', 'h2o', 'ch4', 'gas4', 'others']):
                    out_dict[k] = float(
                        self.root.get(
                            'RawProcess_ParameterSettings',
                            f'sr_lim_{name}'))
                return out_dict

            def set_amplitude_resolution(
                self,
                enable: bool | int = True,
                variation_range: float = 7.0,
                bins: int = 100,
                max_empty_bins: float = 70,
            ):
                """
                Settings for detecting amplitude resolution errors

                Parameters
                ----------
                enable: whether to enable amplitude resolution flagging. Default True
                variation_range: the expected maximum z-score range for the data. Default ±7σ
                bins: int, the number of bins for the histogram. Default 100
                max_empty_bins: float, if more than max_empty_bins% of bins in the histogram are empty, flag for amplitude resolution problems

                limits on inputs:
                variation_range: 1-20
                bins: 50-150
                max_empty_bins: 1-100%
                """
                history_args = ('Advanced-Statistical', 'amplitude_resolution', self.get_amplitude_resolution)
                self.root._add_to_history(*history_args, True)
                assert or_isinstance(
                    enable, int, bool), 'enable should be int or bool'
                assert or_isinstance(variation_range, int, float) and in_range(
                    variation_range, '[1, 20]'), 'variation_range should be numeric and in interval [1, 20]'
                assert isinstance(
                    bins, int) and in_range(
                    bins, '[50, 150]'), 'bins must be within [50, 150]'
                assert or_isinstance(max_empty_bins, int, float) and in_range(
                    max_empty_bins, '[1, 100]'), 'max_empty_bins must be within 1-100%'

                # enable
                if not enable:
                    self.root.set('RawProcess_Tests', 'test_ar', '0')
                    return
                self.root.set('RawProcess_Tests', 'test_ar', '1')

                self.root.set(
                    'RawProcess_ParameterSettings',
                    'ar_lim',
                    str(variation_range))
                self.root.set(
                    'RawProcess_ParameterSettings',
                    'ar_bins',
                    str(bins))
                self.root.set(
                    'RawProcess_ParameterSettings',
                    'ar_hf_lim',
                    str(max_empty_bins))
                
                self.root._add_to_history(*history_args)
                return
            def get_amplitude_resolution(self) -> dict:
                out = dict()
                out['enable'] = bool(
                    int(self.root.get('RawProcess_Tests', 'test_ar')))
                if not out['enable']:
                    return out

                out['variation_range'] = float(self.root.get(
                    'RawProcess_ParameterSettings', 'ar_lim'))
                out['bins'] = int(
                    self.root.get(
                        'RawProcess_ParameterSettings',
                        'ar_bins'))
                out['max_empty_bins'] = float(self.root.get(
                    'RawProcess_ParameterSettings', 'ar_hf_lim'))

                return out

            def set_dropouts(
                self,
                enable: bool | int = True,
                extreme_percentile: int = 10,
                accepted_central_dropouts: float = 10.0,
                accepted_extreme_dropouts: float = 6.0,
            ):
                """
                Settings for detecting instrument dropouts

                Parameters
                ----------
                enable: whether to enable dropout flagging. Default True
                extreme_percentile: int, bins lower than this percentile in the histogram will be considered extreme. Default 10
                accepted_central_dropouts: If consecutive values fall within a non-extreme histogram bin, flag the instrument for a dropout. If more than accepted_central_dropouts% of the averaging interval are flagged as dropouts, flag the whole averagine interval. Default 10%
                accepted_extreme_dropouts: same as for accepted_central_dropouts, except for values in the extreme histogram range. Default 6%

                limits on inputs:
                extreme_percentile: 1-100
                accepted_central/extreme_droupouts: 1-100%
                """
                history_args = ('Advanced-Statistical', 'dropouts', self.get_dropouts)
                self.root._add_to_history(*history_args, True)
                assert or_isinstance(
                    enable, int, bool), 'enable should be int or bool'
                assert isinstance(extreme_percentile, int) and in_range(
                    extreme_percentile, '[1, 100]'), 'extreme_percentile must be between 1 and 100 and numeric'
                assert or_isinstance(accepted_central_dropouts, float, int) and in_range(
                    accepted_central_dropouts, '[1, 100]'), 'accepted_central_dropouts must be between 1 and 100% and numeric'
                assert or_isinstance(accepted_extreme_dropouts, float, int) and in_range(
                    accepted_extreme_dropouts, '[1, 100]'), 'accepted_extreme_dropouts must be between 1 and 100% and numeric'

                # enable
                if not enable:
                    self.root.set('RawProcess_Tests', 'test_do', '0')
                    return
                self.root.set('RawProcess_Tests', 'test_do', '1')

                self.root.set(
                    'RawProcess_ParameterSettings',
                    'do_extlim_dw',
                    str(extreme_percentile))
                self.root.set(
                    'RawProcess_ParameterSettings',
                    'do_hf1_lim',
                    str(accepted_central_dropouts))
                self.root.set(
                    'RawProcess_ParameterSettings',
                    'do_hf2_lim',
                    str(accepted_extreme_dropouts))

                return
            def get_dropouts(self):
                # enable
                out = dict()
                out['enable'] = bool(
                    int(self.root.get('RawProcess_Tests', 'test_do')))
                if not out['enable']:
                    return out

                out['extreme_percentile'] = int(
                    self.root.get(
                        'RawProcess_ParameterSettings',
                        'do_extlim_dw'))
                out['accepted_central_dropouts'] = float(
                    self.root.get('RawProcess_ParameterSettings', 'do_hf1_lim'))
                out['accepted_extreme_dropouts'] = float(
                    self.root.get('RawProcess_ParameterSettings', 'do_hf2_lim'))

                return out

            def set_absolute_limits(
                self,
                enable: bool | int = True,
                u: float = 30.0,
                w: float = 5.0,
                ts: Sequence[float, float] = (-40.0, 50.0),
                co2: Sequence[float, float] = (200.0, 900.0),
                h2o: Sequence[float, float] = (0.0, 40.0),
                ch4: Sequence[float, float] = (0.17, 1000.0),
                gas4: Sequence[float, float] = (0.032, 1000.0),
                filter_outliers: bool | int = True,
            ):
                """
                Settings for flagging unphysically large or small values

                Parameters
                ----------
                enable: whether to enable dropout flagging. Default True
                u, w: absolute limit for |u| and |w| in m/s. Default 30.0, 5.0 respectively.
                ts: sequence of length 2, absolute limits in degrees C for sonic temperature. Default (-40.0, 50.0)
                co2: sequence of length 2, absolute limits in µmol/mol for co2 mixing ratio. Default (200.0, 900.0)
                h2o: sequence of length 2, absolute limits in mmol/mol for water vapor mixing ratio. Default (0.0, 40.0)
                ch4/gas4: sequence of length 2, absolute limits in µmol/mol for methane and gas4 mixing ratio. Default (0.17, 1000.0) and (0.032, 1000.0), respectively
                filter: whether to remove values outside the plausible range. Default True

                bounds on u, w, ts, co2, h2o, ch4, and gas4: upper bound must be >= lower bound.
                u: 1-50
                w: 0.5-10
                ts: -100 - 100
                co2: 100 - 10000
                h2o, ch4, gas4: 0 - 1000
                """
                history_args = ('Advanced-Statistical', 'absolute_limits', self.get_absolute_limits)
                self.root._add_to_history(*history_args, True)

                assert or_isinstance(
                    enable, int, bool), 'enable should be int or bool'
                assert or_isinstance(
                    filter_outliers, int, bool), 'filter_outliers should be int or bool'
                assert or_isinstance(
                    u, int, float) and in_range(
                    u, '[1, 50]'), 'u must be int or float between 1 and 50m/s'
                assert or_isinstance(
                    w, int, float) and in_range(
                    w, '[0.5, 10]'), 'w must be int or float between 0.5 and 10m/s'
                for name, v, lims in zip(
                    ['ts', 'co2', 'h2o', 'ch4', 'gas4'],
                    [ts, co2, h2o, ch4, gas4],
                    ['[-100, 100]', '[100, 10_000]', '[0, 1000]', '[0, 1000]', '[0, 1000]']
                ):
                    if not (
                        or_isinstance(v[0], int, float) and
                        or_isinstance(v[1], int, float) and
                        isinstance(v, Sequence) and
                        len(v) == 2
                    ):
                        raise AssertionError(
                            f'{name} must be a sequence of float or int of length 2')
                    if not (
                        in_range(v[0], lims) and
                        in_range(v[1], lims) and
                        v[1] >= v[0]
                    ):
                        raise AssertionError(
                            f'elements of {name} must be within the interval {lims}')

                # enable
                if not enable:
                    self.root.set('RawProcess_Tests', 'test_al', '0')
                    return
                self.root.set('RawProcess_Tests', 'test_al', '1')

                # limits
                self.root.set(
                    'RawProcess_ParameterSettings',
                    'al_u_max',
                    str(u))
                self.root.set(
                    'RawProcess_ParameterSettings',
                    'al_w_max',
                    str(w))
                for name, v in zip(
                    # eddypro calls gas4 n2o and ts tson
                    ['tson', 'co2', 'h2o', 'ch4', 'n2o'],
                    [ts, co2, h2o, ch4, gas4]
                ):
                    vmin, vmax = v
                    self.root.set(
                        'RawProcess_ParameterSettings',
                        f'al_{name}_min',
                        str(vmin))
                    self.root.set(
                        'RawProcess_ParameterSettings',
                        f'al_{name}_max',
                        str(vmax))

                # filter
                if filter_outliers:
                    self.root.set(
                        'RawProcess_Settings', 'filter_al', '1')
                    return
                self.root.set('RawProcess_Settings', 'filter_al', '0')

                self.root._add_to_history(*history_args)
                return
            def get_absolute_limits(self):
                out = dict()
                out['enable'] = bool(
                    int(self.root.get('RawProcess_Tests', 'test_al')))
                if not out['enable']:
                    return out

                out['u'] = float(
                    self.root.get(
                        'RawProcess_ParameterSettings',
                        'al_u_max'))
                out['w'] = float(
                    self.root.get(
                        'RawProcess_ParameterSettings',
                        'al_w_max'))

                for name, k in zip(
                    # eddypro calls gas4 n2o and ts tson
                    ['tson', 'co2', 'h2o', 'ch4', 'n2o'],
                    ['ts', 'co2', 'h2o', 'ch4', 'gas4']
                ):
                    vmin = float(
                        self.root.get(
                            'RawProcess_ParameterSettings',
                            f'al_{name}_min'))
                    vmax = float(
                        self.root.get(
                            'RawProcess_ParameterSettings',
                            f'al_{name}_max'))
                    out[k] = (vmin, vmax)

                out['filter_outliers'] = bool(
                    int(self.root.get('RawProcess_Settings', 'filter_al')))

                return out

            def set_skewness_and_kurtosis(
                self,
                enable: bool | int = True,
                skew_lower: tuple[float, float] = (-2.0, -1.0),
                skew_upper: tuple[float, float] = (2.0, 1.0),
                kurt_lower: tuple[float, float] = (1.0, 2.0),
                kurt_upper: tuple[float, float] = (8.0, 5.0)
            ):
                """
                Settings for flagging time windows for extreme skewness and kurtosis values

                Parameters
                ----------
                enable: whether to enable skewness and kurtosis flagging. Default True
                skew_lower: a tuple of (hard, soft) defining the upper limit for skewness, where hard defines the hard-flagging threshold and soft defines the soft-flagging threshold. Default is (-2.0, -1.0).
                all following arguments obey similar logic. Defaults are (2.0, 1.0), (1.0, 2.0), (8.0, 5.0) respectively.

                limits are as follows:
                |soft flag| <= |hard flag|
                skew lower in [-3, -0.1]
                skew upper in [0.1, 3]
                kurt lower in [0.1, 3]
                kurt upper in [3, 10]
                """
                
                arg4 = True  # this is just done to make copying and pasting easier
                history_args = ('Advanced-Statistical', 'skewness_and_kurtosis', self.get_skewness_and_kurtosis)
                self.root._add_to_history(*history_args, arg4)
                arg4 = False
                assert or_isinstance(
                    enable, int, bool), 'enable should be int or bool'
                for v, name, bounds in zip(
                    [skew_lower, skew_upper, kurt_lower, kurt_upper],
                    ['skew_lower', 'skew_upper', 'kurt_lower', 'kurt_upper'],
                    ['[-3, -0.1]', '[0.1, 3]', '[0.1, 3]', '[3, 10]']
                ):
                    if not (
                        or_isinstance(v[0], int, float) and
                        or_isinstance(v[1], int, float) and
                        or_isinstance(v, Sequence) and
                        len(v) == 2 and
                        in_range(v[0], bounds) and
                        in_range(v[1], bounds)
                    ):
                        raise AssertionError(
                            f'{name} must be a sequence (hard, soft) of int or float of length 2 with each element within the bounds {bounds} and with hard being more extreme than soft')
                assert skew_lower[0] <= skew_lower[1], 'hard for skew_lower must be <= soft'
                assert skew_upper[0] >= skew_upper[1], 'hard for skew_upper must be >= soft'
                assert kurt_lower[0] <= kurt_lower[1], 'hard for kurt_lower must be <= soft'
                assert kurt_upper[0] >= kurt_upper[1], 'hard for kurt_upper must be <= soft'
                if not enable:
                    self.root.set('RawProcess_Tests', 'test_sk', '0')
                    return
                self.root.set('RawProcess_Tests', 'test_sk', '1')

                for name, v in zip(
                    ['skmin', 'skmax', 'kumin', 'kumax'],
                    [skew_lower, skew_upper, kurt_lower, kurt_upper]
                ):
                    soft, hard = v
                    self.root.set(
                        'RawProcess_ParameterSettings',
                        f'sk_sf_{name}',
                        str(soft))
                    self.root.set(
                        'RawProcess_ParameterSettings',
                        f'sk_hf_{name}',
                        str(hard))
                    
                history_args = ('Advanced-Statistical', 'skewness_and_kurtosis', self.get_skewness_and_kurtosis)
                self.root._add_to_history(*history_args, arg4)
                return
            def get_skewness_and_kurtosis(self):
                out = dict()
                out['enable'] = bool(
                    int(self.root.get('RawProcess_Tests', 'test_sk')))
                if not out['enable']:
                    return out

                for name, k in zip(
                    ['skmin', 'skmax', 'kumin', 'kumax'],
                    ['skew_lower', 'skew_upper', 'kurt_lower', 'kurt_upper']
                ):
                    soft = float(
                        self.root.get(
                            'RawProcess_ParameterSettings',
                            f'sk_sf_{name}'))
                    hard = float(
                        self.root.get(
                            'RawProcess_ParameterSettings',
                            f'sk_hf_{name}'))
                    out[k] = (soft, hard)
                return out

            def set_discontinuities(
                self,
                enable: bool | int = False,
                u: Sequence[float, float] = (4.0, 2.7),
                w: Sequence[float, float] = (2.0, 1.3),
                ts: Sequence[float, float] = (4.0, 2.7),
                co2: Sequence[float, float] = (40.0, 27.0),
                h2o: Sequence[float, float] = (3.26, 2.2),
                ch4: Sequence[float, float] = (40.0, 30.0),
                gas4: Sequence[float, float] = (40.0, 30.0),
                variances: Sequence[float, float] = (3.0, 2.0)
            ):
                """
                settings for detecting semi-permanent distontinuities in timeseries data

                Parameters
                ----------
                enable: whether to enable discontinuity flagging. Default False
                u, w, ts, co2, h2o, ch4, gas4: a sequence of (hard, soft) specifying the hard and soft-flag thresholds for the haar transform on raw data. See eddypro documentation or Vickers and Mahrt 1997 for an explanation of thresholds.
                variances: same as above, but for variances rather than raw data.

                must have hard >= soft, and all values must be within the interval [0, 50]
                """
                history_args = ('Advanced-Statistical', 'discontinuities', self.get_discontinuities)
                self.root._add_to_history(*history_args, True)

                assert or_isinstance(
                    enable, int, bool), 'enable should be int or bool'
                for v, name in zip(
                    [u, w, ts, co2, h2o, ch4, gas4, variances],
                    ['u', 'w', 'ts', 'co2', 'h2o', 'ch4', 'gas4', 'variances'],
                ):
                    if not (
                        or_isinstance(v[0], int, float) and
                        or_isinstance(v[1], int, float) and
                        or_isinstance(v, Sequence) and
                        len(v) == 2 and
                        v[0] >= v[1] and
                        in_range(v[0], '[0, 50]') and
                        in_range(v[1], '[0, 50]')
                    ):
                        raise AssertionError(
                            f'{name} must be a non-increasing sequence of int or float of length 2 with each element within the bounds [0, 50]')

                if not enable:
                    self.root.set('RawProcess_Tests', 'test_ds', '0')
                    return
                self.root.set('RawProcess_Tests', 'test_ds', '1')

                for name, v in zip(
                    ['uv', 'w', 't', 'co2', 'h2o', 'ch4', 'n2o',
                        'var'],  # gas4 called n2o by eddypro
                    [u, w, ts, co2, h2o, ch4, gas4, variances]
                ):
                    soft, hard = v
                    self.root.set(
                        'RawProcess_ParameterSettings',
                        f'ds_sf_{name}',
                        str(soft))
                    self.root.set(
                        'RawProcess_ParameterSettings',
                        f'ds_hf_{name}',
                        str(hard))
                
                self.root._add_to_history(*history_args)
                return
            def get_discontinuities(self):
                out = dict()
                out['enable'] = bool(
                    int(self.root.get('RawProcess_Tests', 'test_ds')))
                if not out['enable']:
                    return out

                for name, k in zip(
                    ['uv', 'w', 't', 'co2', 'h2o', 'ch4', 'n2o',
                        'var'],  # gas4 called n2o by eddypro
                    ['u', 'w', 'ts', 'co2', 'h2o', 'ch4', 'gas4', 'variances']
                ):
                    soft = float(
                        self.root.get(
                            'RawProcess_ParameterSettings',
                            f'ds_sf_{name}'))
                    hard = float(
                        self.root.get(
                            'RawProcess_ParameterSettings',
                            f'ds_hf_{name}'))
                    out[k] = (soft, hard)
                return out

            def set_timelags(
                self,
                enable: bool | int = False,
                covariance_difference: Sequence[float, float] = (20.0, 10.0),
                co2: float = 0.0,
                h2o: float = 0.0,
                ch4: float = 0.0,
                gas4: float = 0.0,
            ):
                """
                Settings for flagging time lags: if, when correcting Cov(w, X) for time lags in X, Cov(w, X) differs significantly from the non-time-lag-corrected covariance, throw a flag.

                Parameters
                ----------
                enable: whether to enable flagging for excessive changes in covariance due to time lags (default False)
                covariance_difference: a tuple of (hard, soft) for covariance differences as a % between uncorrected and time-lag-corrected covariances, where hard defines the hard-flagging threshold and soft defines the soft-flagging threshold.
                co2/h2o/ch4/gas4: the expected time lags for each gas in seconds.
                limits on inputs:
                covariance_difference: 0-100%, with soft <= hard
                all other values: 0-100s
                """
                history_args = ('Advanced-Statistical', 'timelags', self.get_timelags)
                self.root._add_to_history(*history_args, True)

                assert or_isinstance(
                    enable, int, bool), 'enable should be int or bool'
                if not (
                    or_isinstance(covariance_difference[0], int, float) and
                    or_isinstance(covariance_difference[1], int, float) and
                    or_isinstance(covariance_difference, Sequence) and
                    len(covariance_difference) == 2 and
                    covariance_difference[0] >= covariance_difference[1] and
                    in_range(covariance_difference[0], '[0, 100]') and
                    in_range(covariance_difference[1], '[0, 100]')
                ):
                    raise AssertionError(
                        'covariance_difference must be a non-increasing sequence of length 2 of ints or floats between 0 and 100%')
                assert or_isinstance(
                    co2, float, int) and in_range(
                    co2, '[0, 100]'), 'co2 must be numeric and in the range of 0-100 seconds'
                assert or_isinstance(
                    h2o, float, int) and in_range(
                    h2o, '[0, 100]'), 'h2o must be numeric and in the range of 0-100 seconds'
                assert or_isinstance(
                    ch4, float, int) and in_range(
                    ch4, '[0, 100]'), 'ch4 must be numeric and in the range of 0-100 seconds'
                assert or_isinstance(
    gas4, float, int) and in_range(
         gas4, '[0, 100]'), 'gas4 must be numeric and in the range of 0-100 seconds'

                if not enable:
                    self.root.set('RawProcess_Tests', 'test_tl', '0')
                    return
                self.root.set('RawProcess_Tests', 'test_tl', '1')

                soft, hard = covariance_difference
                self.root.set(
                    'RawProcess_ParameterSettings',
                    'tl_sf_lim',
                    str(soft))
                self.root.set(
                    'RawProcess_ParameterSettings',
                    'tl_hf_lim',
                    str(hard))

                self.root.set(
                    'RawProcess_ParameterSettings',
                    'tl_def_co2',
                    str(co2))
                self.root.set(
                    'RawProcess_ParameterSettings',
                    'tl_def_h2o',
                    str(h2o))
                self.root.set(
                    'RawProcess_ParameterSettings',
                    'tl_def_ch4',
                    str(ch4))
                self.root.set(
                    'RawProcess_ParameterSettings',
                    'tl_def_n2o',
                    str(gas4))

                self.root._add_to_history(*history_args)
                return
            def get_timelags(self):
                out = dict()
                out['enable'] = bool(
                    int(self.root.get('RawProcess_Tests', 'test_tl')))
                if not out['enable']:
                    return out

                soft = self.root.get(
                    'RawProcess_ParameterSettings', 'tl_sf_lim')
                hard = self.root.get(
                    'RawProcess_ParameterSettings', 'tl_hf_lim')
                out['covariance_difference'] = (soft, hard)

                out['co2'] = self.root.get(
                    'RawProcess_ParameterSettings', 'tl_def_co2')
                out['h2o'] = self.root.get(
                    'RawProcess_ParameterSettings', 'tl_def_h2o')
                out['ch4'] = self.root.get(
                    'RawProcess_ParameterSettings', 'tl_def_ch4')
                out['gas4'] = self.root.get(
                    'RawProcess_ParameterSettings', 'tl_def_n2o')

                return out

            def set_angle_of_attack(
                self,
                enable: bool | int = False,
                aoa_min: float = -30.0,
                aoa_max: float = 30.0,
                accepted_outliers: float = 10.0,
            ):
                """
                Settings for flagging extreme angles of attack

                Parameters
                ----------
                enable: whether to enable angle-of-attack flagging. Default False
                aoa_min: the minimum acceptable angle of attack in degrees. Default -30.
                aoa_max: the maximum acceptable angle of attack in degrees. Default 30.
                accepted_outliers: if more than accepted_outliers% of values lie outside the specified bounds, flag the averaging window. Default 10%.

                limits on inputs:
                aoa_min: -90 - 0°
                aoa_max: 0 - 90°
                accepted_outliers: 0-100%
                """

                history_args = ('Advanced-Statistical', 'angle_of_attack', self.get_angle_of_attack)
                self.root._add_to_history(*history_args, True)

                assert or_isinstance(
                    enable, int, bool), 'enable should be int or bool'
                assert or_isinstance(aoa_min, int, float) and in_range(
                    aoa_min, '[-90, 0]'), 'aoa_min should be numeric and within the interval [-90, 0]'
                assert or_isinstance(aoa_max, int, float) and in_range(
                    aoa_max, '[0, 90]'), 'aoa_max should be numeric and within the interval [0, 90]'
                assert or_isinstance(accepted_outliers, int, float) and in_range(
                    aoa_max, '[0, 100]'), 'accepted_outliers should be numeric and within the interval [0, 100]'

                if not enable:
                    self.root.set('RawProcess_Tests', 'test_aa', '0')
                    return
                self.root.set('RawProcess_Tests', 'test_aa', '1')

                self.root.set(
                    'RawProcess_ParameterSettings',
                    'aa_min',
                    str(aoa_min))
                self.root.set(
                    'RawProcess_ParameterSettings',
                    'aa_max',
                    str(aoa_max))
                self.root.set(
                    'RawProcess_ParameterSettings',
                    'aa_lim',
                    str(accepted_outliers))
                
                self.root._add_to_history(*history_args)
                return
            def get_angle_of_attack(self):
                out = dict()
                out['enable'] = bool(
                    int(self.root.get('RawProcess_Tests', 'test_aa')))
                if not out['enable']:
                    return out

                out['aoa_min'] = float(
                    self.root.get(
                        'RawProcess_ParameterSettings',
                        'aa_min'))
                out['aoa_max'] = float(
                    self.root.get(
                        'RawProcess_ParameterSettings',
                        'aa_max'))
                out['accepted_outliers'] = float(self.root.get(
                    'RawProcess_ParameterSettings', 'aa_lim'))

                return out

            def set_steadiness_of_horizontal_wind(
                self,
                enable: bool | int = False,
                max_rel_inst: float = 0.5,
            ):
                """
                Settings for flagging horizontal wind steadiness.

                Parameters
                ----------
                enable: whether to enable flagging of horizontal wind steadiness. Default False.
                max_rel_inst: if the change in windspeed over the averaging window normalized by the mean windspeed exceeds this threshold, hard-flag the record. Default 0.5 for 50% relative instationarity.

                max_rel_inst should be within the interval [0, 50]
                """

                history_args = ('Advanced-Statistical', 'steadiness_of_horizontal_wind', self.steadiness_of_horizontal_wind)
                self.root._add_to_history(*history_args, True)

                assert or_isinstance(
                    enable, int, bool), 'enable should be int or bool'
                assert or_isinstance(max_rel_inst, int, float) and in_range(
                    max_rel_inst, '[0, 50]'), 'max_rel_inst should be within [0, 50]'

                if not enable:
                    self.root.set('RawProcess_Tests', 'test_ns', '0')
                    return
                self.root.set('RawProcess_Tests', 'test_ns', '1')

                self.root.set(
                    'RawProcess_ParameterSettings',
                    'ns_hf_lim',
                    str(max_rel_inst))
                
                self.root._add_to_history(*history_args)
                return
            def get_steadiness_of_horizontal_wind(self):
                out = dict()
                out['enable'] = bool(
                    int(self.root.get('RawProcess_Tests', 'test_ns')))
                if not out['enable']:
                    return out

                out['max_rel_inst'] = float(
                    self.root.get(
                        'RawProcess_ParameterSettings',
                        'ns_hf_lim'))

                return out

            def set_estimate_random_uncertainty(
                self,
                method: Literal['disable', 'FS01', 'ML94', 'M98'] | int = 'disable',
                its_definition: Literal['at_1/e', 'at_0', 'whole_period'] | int = 'at_1/e',
                maximum_correlation_period: float = 10.0
            ):
                """
                Settings for estimating random uncertainty due to sampling error

                Parameters
                ----------
                method: one of disable, FS01 (Finkelstein and Sims 2001), ML94 (Mann & Lenschow 1994), or M98 (Mahrt 1998), or 0, 1, or 2, respectively
                its_definition: definition of the integral turbulence scale. Options are 'at_1/e', 'at_0', 'whole_record', or 0, 1, or 2, respecitvely. See EddyPro documentation for more details.
                maximum_correlation_period: maximum time to integrate over when determining the integral turbulence scale. Default is 10.0s. Must be within [0, 10000] seconds
                """

                history_args = ('Advanced-Statistical', 'estimate_random_uncertainty', self.estimate_random_uncertainty)
                self.root._add_to_history(*history_args, True)

                assert isinstance(method, str) or method in range(
                    4), 'method must be one of disable (0), FS01 (1), ML94 (2), or M98 (3)'
                assert isinstance(its_definition, str) or its_definition in range(
                    3), 'its_definition must be one of at_1/e (0), at_0 (1), or whole_period (2)'
                assert or_isinstance(maximum_correlation_period, float, int) and in_range(
                    maximum_correlation_period, '[0, 10_000]'), 'maximum_correlation_period must be numeric and in the range of [0, 10_000] seconds'

                methods = {k: v for k, v in zip(
                    ['disable', 'FS01', 'ML94', 'M98'], range(4))}
                if method in methods:
                    method = methods[method]
                if not method:
                    self.root.set('Project', 'ru_meth', '0')
                    return
                self.root.set('Project', 'ru_meth', str(method))

                its_defs = {k: v for k, v in zip(
                    ['at_1/e', 'at_0', 'whole_period'], range(3))}
                if its_definition in its_defs:
                    its_definition = its_defs[its_definition]
                self.root.set('Project', 'ru_tlag_meth', str(its_definition))

                self.root.set(
                    'Project',
                    'ru_tlag_max',
                    str(maximum_correlation_period))
                
                self.root._add_to_history(*history_args)
                return
            def get_estimate_random_uncertainty(self):
                out = dict()
                methods = ['disable', 'FS01', 'ML94', 'M98']
                out['method'] = methods[(
                    int(self.root.get('Project', 'ru_meth')))]
                if out['method'] == 'disable':
                    return out

                its_defs = ['at_1/e', 'at_0', 'whole_period']
                out['its_definition'] = its_defs[int(
                    self.root.get('Project', 'ru_tlag_meth'))]
                out['maximum_correlation_period'] = float(
                    self.root.get('Project', 'ru_tlag_max'))

                return out

        class _Spec:
            def __init__(self, outer):
                self.root = outer.root
                self.outer = outer

            def set_calculation(
                self,
                binned_cosp_dir: str | PathLike[str] | None = None,
                start: str | datetime.datetime | Literal['project'] = 'project',
                end: str | datetime.datetime | Literal['project'] = 'project',
                window: Literal['squared', 'bartlett', 'welch', 'hamming', 'hann'] | int = 'hamming',
                bins: int = 50,
                power_2: bool | int = True
            ):
                """
                Settings for how to compute (co)spectra for further analysis

                Parameters
                ----------
                binned_cosp_dir: either a str or pathlike object pointing to a directory of eddypro-readable binned cospectra files for this dataset. If None (default) to indicate to eddypro to compute co-spectra on-the-fly. If None, it is HIGHLY recommended to set eddypro to output binned spectra in the output section.
                start, end: start and end date-times for planar fit computation. If a string, must be in yyyy-mm-dd HH:MM format or "project." If "project"  (default), sets the start/end to the project start/end date. If one of start, end is project, the other must be as well. 
                window: the tapering window to use. One of squared (0), bartlett (1), welch (2), hamming (3, default), or hann (4). Ignored if binned_cosp_dir is provided.
                bins: the number of bins to use for cospectra reduction. Default 50. Ignored if binned_cosp_dir is provided.
                power_2: whether to use the nearest power-of-two number of samples when computing the FFT to speed up analysis. Default True. Ignored if binned_cosp_dir is provided.

                limits on inputs
                bins: 10-3000
                """
                history_args = ('Advanced-Spectral', 'calculation', self.get_calculation)
                self.root._add_to_history(*history_args, True)
                
                if binned_cosp_dir is not None:
                    assert or_isinstance(binned_cosp_dir, str, Path, bool), 'binned_files must be a file path or None'
                assert or_isinstance(start, str, datetime.datetime), 'starting timestamp must be string or datetime.datetime'
                assert or_isinstance(end, str, datetime.datetime), 'ending timestamp must be string or datetime.datetime'
                if isinstance(start, str):
                    assert len(start) == 16 or start == 'project', 'if start is a string, it must be a timestamp of the form YYYY-mm-dd HH:MM or "project"'
                    if start == 'project':
                        assert end == 'project', 'if one of start, end is "project", the other must be as well.'
                if isinstance(end, str):
                    assert len(end) == 16 or end == 'project', 'if end is a string, it must be a timestamp of the form YYYY-mm-dd HH:MM or "project"'
                    if end == 'project':
                        assert start == 'project', 'if one of start, end is "project", the other must be as well.'
                assert window in ['squared', 'bartlett', 'welch', 'hamming', 'hann', 0, 1, 2, 3, 4], 'window must be one of squared (0), bartlett (1), welch (2), hamming (3, default), or hann (4).'
                assert bins % 1 == 0 and in_range(bins, '[10, 3000]'), 'bins must have no decimal part and be between 10 and 3000'
                assert or_isinstance(power_2, bool, int), 'power_2 must be bool or int.'

                # binned cospectra
                self.root.set('Project', 'bin_sp_avail', '0')
                if binned_cosp_dir is not None:
                    self.root.set('Project', 'bin_sp_avail', '1')
                    self.root.set('FluxCorrection_SpectralAnalysis_General', 'sa_bin_spectra', str(binned_cosp_dir))
                
                # processing dates
                # process dates
                sa_subset = 1
                if start == 'project':
                    sa_subset = 0
                elif isinstance(start, datetime.datetime):
                    sa_start = start
                    sa_start_date, sa_start_time = sa_start.strftime(r'%Y-%m-%d %H:%M').split(' ')
                else:
                    sa_start = start
                    sa_start_date, sa_start_time = sa_start.split(' ')
                    
                if end == 'project':
                    pass
                elif isinstance(end, datetime.datetime):
                    sa_end = end
                    sa_end_date, sa_end_time = sa_end.strftime(r'%Y-%m-%d %H:%M').split(' ')
                else:
                    sa_end = end
                    sa_end_date, sa_end_time = sa_end.split(' ')
                
                if sa_subset:
                    self.root.set('FluxCorrection_SpectralAnalysis_General', 'sa_start_date', sa_start_date)
                    self.root.set('FluxCorrection_SpectralAnalysis_General', 'sa_start_time', sa_start_time)
                    self.root.set('FluxCorrection_SpectralAnalysis_General', 'sa_end_date', sa_end_date)
                    self.root.set('FluxCorrection_SpectralAnalysis_General', 'sa_end_time', sa_end_time)
                self.root.set('FluxCorrection_SpectralAnalysis_General', 'sa_subset', str(sa_subset))
                
                if binned_cosp_dir is None:
                    # window
                    win_dict = {k:v for k, v in zip(['squared', 'bartlett', 'welch', 'hamming', 'hann'], range(500))}
                    if window in win_dict:
                        window = win_dict[window]
                    self.root.set('RawProcess_Settings', 'tap_win', str(window))

                    # bins
                    self.root.set('RawProcess_Settings', 'nbins', str(int(bins)))

                    # power-of-2
                    self.root.set('RawProcess_Settings', 'power_of_two', str(int(bool(power_2))))

                self.root._add_to_history(*history_args)
                return
            def get_calculation(self):
                out = dict()

                # processing files
                if int(self.root.get('Project', 'bin_sp_avail')):
                    out['binned_cosp_dir'] = self.root.get('FluxCorrection_SpectralAnalysis_General', 'sa_bin_spectra')

                # dates
                if not int(self.root.get('FluxCorrection_SpectralAnalysis_General', 'sa_subset')):
                    out['start'] = 'project'
                    out['end'] = 'project'
                else:
                    out['start'] = (
                        self.root.get('FluxCorrection_SpectralAnalysis_General', 'sa_start_date') 
                        + ' '
                        + self.root.get('FluxCorrection_SpectralAnalysis_General', 'sa_start_time'))
                    out['end'] = (
                        self.root.get('FluxCorrection_SpectralAnalysis_General', 'sa_end_date') 
                        + ' '
                        + self.root.get('FluxCorrection_SpectralAnalysis_General', 'sa_end_time'))
                
                if 'binned_cosp_dir' not in out:
                    out['window'] = ['squared', 'bartlett', 'welch', 'hamming', 'hann'][int(self.root.get('RawProcess_Settings', 'tap_win'))]
                    out['bins'] = int(self.root.get('RawProcess_Settings', 'nbins'))
                    out['power_2'] = bool(int(self.root.get('RawProcess_Settings', 'power_of_two')))
                return out
            
            def set_removal_of_high_frequency_noise(
                self,
                co2: float = 1.0,
                h2o: float = 1.0,
                ch4: float = 1.0,
                gas4: float = 1.0,
            ):
                """
                Settings for removing high frequency noise

                Parameters
                ----------
                for each gas (co2, h2o, ch4, gas4), provide the lowest frequency representing blue noise. If 0, do not remove noise. Default 1.0 for all gasses.

                limits on inputs
                0 - 50Hz
                """
                history_args = ('Advanced-Spectral', 'removal_of_high_frequency_noise', self.get_removal_of_high_frequency_noise)
                self.root._add_to_history(*history_args, True)
                
                assert in_range(co2, '[0, 50]'), 'co2 must be between 0 and 50'
                assert in_range(h2o, '[0, 50]'), 'h2o must be between 0 and 50'
                assert in_range(ch4, '[0, 50]'), 'ch4 must be between 0 and 50'
                assert in_range(gas4, '[0, 50]'), 'gas4 must be between 0 and 50'


                self.root.set('FluxCorrection_SpectralAnalysis_General', 'sa_hfn_co2_fmin', str(co2))
                self.root.set('FluxCorrection_SpectralAnalysis_General', 'sa_hfn_h2o_fmin', str(h2o))
                self.root.set('FluxCorrection_SpectralAnalysis_General', 'sa_hfn_ch4_fmin', str(ch4))
                self.root.set('FluxCorrection_SpectralAnalysis_General', 'sa_hfn_gas4_fmin', str(gas4))
                
                self.root._add_to_history(*history_args)
                return
            def get_removal_of_high_frequency_noise(self):
                out = dict()
                out['co2'] = self.root.get('FluxCorrection_SpectralAnalysis_General', 'sa_hfn_co2_fmin')
                out['h2o'] = self.root.get('FluxCorrection_SpectralAnalysis_General', 'sa_hfn_h2o_fmin')
                out['ch4'] = self.root.get('FluxCorrection_SpectralAnalysis_General', 'sa_hfn_ch4_fmin')
                out['gas4'] = self.root.get('FluxCorrection_SpectralAnalysis_General', 'sa_hfn_gas4_fmin')
                return out
            
            def set_qaqc(
                self,
                ustar: Sequence[float, float, float] = (0.2, 0.05, 5.0),
                h: Sequence[float, float, float] = (20., 5., 1000.),
                le: Sequence[float, float, float] = (20., 3., 1000.),
                fco2: Sequence[float, float, float] = (2., 0.5, 100.),
                fch4: Sequence[float, float, float] = (0.01, 0.005, 20.),
                fgas4: Sequence[float, float, float] = (0.01, 0.005, 20.),
                n_min: int = 10,
                filter_vm97: int | bool  = True,
                filter_mf04: Literal['low', 'moderate', 'none'] = 'low'
            ):
                """
                Settings for Qa/Qc of (co)spectra. See EddyPro manual on QA/QC of spectra and cospectra for explanation of settings.

                Parameters
                ----------
                ustar: sequence of (min_unstable, min_stable, max) providing the minimum reasonable values of ustar in unstable and stable conditions, respectively, and the maximum reasonable ustar in any conditions. Default (0.2, 0.05, 5.0) m/s
                h: same as ustar, but for magnitude of sensible heat flux. Default (20., 5., 1000.) W/m2
                le: same as ustar but for magnitude of latent heat flux. Default (20., 3., 1000.) W/m2.
                fco2: same as ustar, but for magnitude of co2 flux. Default (2., 0.5, 100.) µmol/m2/s
                fch4: same as ustar, but for magnitude of ch4 flux. Default (0.01, 0.005, 20.) µmol/m2/s
                fgas4: same as ustar, but for magnitude of gas4 flux. Default (0.01, 0.005, 20.) µmol/m2/s
                n_min: int for the minimum number of cospectra for valid averages. Default 10.
                filter_vm97: int or bool representing whether to omit from spectral caluclations time periods flagged by the vickers and mahrt quality tests in the Advanced/Statistical section, specifically the tests (1) number of spikes; (2) drop-outs; (3) skewness & kurtosis; and (4) discontinuities. Default True.
                filter_mf04: 'low' (default), 'moderate', or 'none' indicating how to filter spectral data before aggregating using Mauder and Foken 2004. If 'low' (default), only filter out low-quality time periods. If 'moderate', filter out both low- and moderate-quality time periods. If 'none,' do not filter.

                limits on inputs
                ustar: 0-5
                h: 0-10_000
                le: 0-10_000
                fco2: 0-5_000
                fch4: 0-5_000
                fgas4: 0-5_000
                n_min: 1-1_000
                """
                history_args = ('Advanced-Spectra', 'qaqc', self.get_qaqc)
                self.root._add_to_history(*history_args, True)
                # check inputs
                for v in [ustar, h, le, fco2, fch4, fgas4]:
                    assert isinstance(v, Sequence), 'ustar, h, le, fco2, fch4, fgas4 must be numeric sequences of length 3'
                    assert len(v) == 3, 'ustar, h, le, fco2, fch4, fgas4 must be numeric sequences of length 3'
                    for iv in v: 
                        assert or_isinstance(iv, float, int), 'ustar, h, le, fco2, fch4, fgas4 must be numeric sequences of length 3'
                for i in range(3):
                    assert in_range(ustar[i], '[0, 5]'), 'ustar must be between 0 and 5m/s'
                    assert in_range(h[i], '[0, 10_000]'), 'h must be between 0 and 10_000W/m2'
                    assert in_range(le[i], '[0, 10_000]'), 'le must be between 0 and 10_000W/m2'
                    assert in_range(fco2[i], '[0, 5_000]'), 'fco2 must be between 0 and 10_000µmol/m2/s'
                    assert in_range(fch4[i], '[0, 5_000]'), 'fch4 must be between 0 and 10_000µmol/m2/s'
                    assert in_range(fgas4[i], '[0, 5_000]'), 'fgas4 must be between 0 and 10_000µmol/m2/s'
                assert n_min % 1 == 0, 'n_min must have no decimal part'
                assert in_range(n_min, '[1, 1000]'), 'n_min must be between 1 and 1000'
                assert filter_mf04 in ['low', 'moderate', 'none'], 'filter_mf04 must be one of low, moderate or none.'

                # set min/max
                section = 'FluxCorrection_SpectralAnalysis_General'
                def set_minmax(name, v):
                    un, st, mx = v
                    self.root.set(section, f'sa_min_un_{name}', str(un))
                    self.root.set(section, f'sa_min_st_{name}', str(st))
                    self.root.set(section, f'sa_max_{name}', str(mx))
                set_minmax('ustar', ustar)
                set_minmax('h', h)
                set_minmax('le', le)
                set_minmax('co2', fco2)
                set_minmax('ch4', fch4)
                set_minmax('gas4', fgas4)
                
                # n_min
                self.root.set(section, 'sa_min_smpl', str(int(n_min)))

                # quality flags
                filter_vm97 = int(bool(filter_vm97))
                self.root.set(section, 'sa_use_vm_flags', str(filter_vm97))

                match filter_mf04:
                    case 'low':
                        self.root.set(section, 'sa_use_foken_low', '1')
                        self.root.set(section, 'sa_use_foken_mid', '0')
                    case 'moderate':
                        self.root.set(section, 'sa_use_foken_low', '1')
                        self.root.set(section, 'sa_use_foken_mid', '1')
                    case 'none':
                        self.root.set(section, 'sa_use_foken_low', '0')
                        self.root.set(section, 'sa_use_foken_mid', '0')
                
                self.root._add_to_history(*history_args)
                return
            def get_qaqc(self):
                out = dict()

                section = 'FluxCorrection_SpectralAnalysis_General'
                out['ustar'] = (
                    float(self.root.get(section, 'sa_min_un_ustar')), 
                    float(self.root.get(section, 'sa_min_st_ustar')), 
                    float(self.root.get(section, 'sa_max_ustar')))
                out['h'] = (
                    float(self.root.get(section, 'sa_min_un_h')), 
                    float(self.root.get(section, 'sa_min_st_h')), 
                    float(self.root.get(section, 'sa_max_h')))
                out['le'] = (
                    float(self.root.get(section, 'sa_min_un_le')), 
                    float(self.root.get(section, 'sa_min_st_le')), 
                    float(self.root.get(section, 'sa_max_le')))
                out['fco2'] = (
                    float(self.root.get(section, 'sa_min_un_co2')), 
                    float(self.root.get(section, 'sa_min_st_co2')), 
                    float(self.root.get(section, 'sa_max_co2')))
                out['fch4'] = (
                    float(self.root.get(section, 'sa_min_un_ch4')), 
                    float(self.root.get(section, 'sa_min_st_ch4')), 
                    float(self.root.get(section, 'sa_max_ch4')))
                out['fgas4'] = (
                    float(self.root.get(section, 'sa_min_un_gas4')), 
                    float(self.root.get(section, 'sa_min_st_gas4')), 
                    float(self.root.get(section, 'sa_max_gas4')))
                
                out['n_min'] = int(self.root.get(section, 'sa_min_smpl'))

                out['filter_vm97'] = bool(int(self.root.get(section, 'sa_use_vm_flags')))

                mf04_low = int(self.root.get(section, 'sa_use_foken_low'))
                mf04_mod = int(self.root.get(section, 'sa_use_foken_mid'))
                
                match mf04_low, mf04_mod:
                    case 0, 0: out['filter_mf04'] = 'none'
                    case 1, 0: out['filter_mf04'] = 'low'
                    case 1, 1: out['filter_mf04'] = 'moderate'
                    case _:
                        warnings.warn('found filter_mf04 to have an invalid configuration. Setting to moderate.')
                        out['filter_mf04'] = 'moderate'
                return out

            def set_lf_correction(self, enable: bool = True):
                """
                whether to apply analytic correction of high-pass filtering affects.
                
                Parameters
                ---------
                enable: bool. If true (default), apply analytic correction of high-pass filtering effects from Moncrieff et al, 2004"""
                history_args = ('Advanced-Spectra', 'lf_correction', self.get_lf_correction)
                self.root._add_to_history(*history_args, True)
                
                assert isinstance(enable, bool), 'low must be bool'
                self.root.set('Project', 'lf_meth', str(int(enable)))

                self.root._add_to_history(*history_args)
                return
            def get_lf_correction(self): return dict(enable=bool(int(self.root.get('Project', 'lf_meth'))))

            def _configure_horst(
                self,
                assessment_file: PathLike | str | None = None,
                co2: Sequence[float, float] = (0.005, 2.),
                h2o: Sequence[float, float] = (0.005, 2.),
                ch4: Sequence[float, float] = (0.005, 2.),
                gas4: Sequence[float, float] = (0.005, 2.),
            ) -> dict:
                """
                configure the settings needed to use Horst 1997 spectral corrections

                Parameters
                ----------
                assessment_file: path to a spectral assessment file for this dataset. If False (none), compute spectral assessment on-the-fly
                co2/h2o/ch4/gas4: sequence of the form (low, high) indicating the the frequency range for fitting in-situe transfer functions, based on temperature and concentration spectra. Default 0-005-2Hz.

                limits on inputs
                all inputs 0-50Hz, with high > low

                Returns
                dicts of ini parameter settings
                """
                # check inputs
                if assessment_file is not None:
                    assert or_isinstance(assessment_file, Path, str), 'assessment_file must be a path, str, or None'
                for v in [co2, h2o, ch4, gas4]:
                    assert isinstance(v, Sequence), 'co2, h2o, ch4, and gas4 must be sequences of the form (low, high) with 0Hz < low < high <= 50Hz'
                    assert len(v) == 2,'co2, h2o, ch4, and gas4 must be sequences of the form (low, high) with 0Hz < low < high <= 50Hz'
                    assert v[1] > v[0], 'co2, h2o, ch4, and gas4 must be sequences of the form (low, high) with 0Hz < low < high <= 50Hz'
                    assert v[0] >= 0,'co2, h2o, ch4, and gas4 must be sequences of the form (low, high) with 0Hz < low < high <= 50Hz'
                    assert v[1] <= 50,'co2, h2o, ch4, and gas4 must be sequences of the form (low, high) with 0Hz < low < high <= 50Hz'

                FluxCorrection_SpectralAnalysis_General = dict()
                RawProcess_Settings = dict()

                
                # assessment file?
                FluxCorrection_SpectralAnalysis_General['sa_mode'] = 1
                if assessment_file is not None:
                    FluxCorrection_SpectralAnalysis_General['sa_mode'] = 0
                    FluxCorrection_SpectralAnalysis_General['sa_file'] = assessment_file
                    return dict(FluxCorrection_SpectralAnalysis_General=FluxCorrection_SpectralAnalysis_General, RawProcess_Settings=RawProcess_Settings)
                
                FluxCorrection_SpectralAnalysis_General['sa_fmin_co2'] = co2[0]
                FluxCorrection_SpectralAnalysis_General['sa_fmax_co2'] = co2[1]
                
                FluxCorrection_SpectralAnalysis_General['sa_fmin_h20'] = h2o[0]
                FluxCorrection_SpectralAnalysis_General['sa_fmax_h20'] = h2o[1]
                
                FluxCorrection_SpectralAnalysis_General['sa_fmin_ch4'] = ch4[0]
                FluxCorrection_SpectralAnalysis_General['sa_fmax_ch4'] = ch4[1]
                
                FluxCorrection_SpectralAnalysis_General['sa_fmin_gas4'] = gas4[0]
                FluxCorrection_SpectralAnalysis_General['sa_fmax_gas4'] = gas4[1]

                # when spectral assessment file is not available, all binnned cospectra must be output
                RawProcess_Settings['out_bin_sp'] = 1
                
                return dict(FluxCorrection_SpectralAnalysis_General=FluxCorrection_SpectralAnalysis_General, RawProcess_Settings=RawProcess_Settings)           
            def _configure_ibrom(
                self,
                assessment_file: PathLike | str | None = None,
                co2: Sequence[float, float] = (0.005, 2.),
                h2o: Sequence[float, float] = (0.005, 2.),
                ch4: Sequence[float, float] = (0.005, 2.),
                gas4: Sequence[float, float] = (0.005, 2.),
                separation: Literal['none', 'uvw', 'vw'] | int = 'none'
            ) -> dict:
                """
                configure the settings needed to use Ibrom 2007 spectral corrections

                Parameters
                ----------
                assessment_file: path to a spectral assessment file for this dataset. If False (none), compute spectral assessment on-the-fly
                co2/h2o/ch4/gas4: sequence of the form (low, high) indicating the the frequency range for fitting in-situe transfer functions, based on temperature and concentration spectra. Default 0-005-2Hz.
                separation: how to correct for instrument separation. If none or 0 (default), do not correct for instrument separation. If uvw or 1, correct for separation in the u, v, and w directions. If vw, only correct for separation in the v and w directions. 


                limits on inputs
                all inputs 0-50Hz, with high > low

                Returns
                dicts of ini parameter settings
                """

                assert separation in ['none', 'uvw', 'vw', 0, 1, 2], 'separation must be one of none (0), uvw (1), vw (2)'

                # all horst settings also apply to ibrom
                FluxCorrection_SpectralAnalysis_General, RawProcess_Settings = self._configure_horst(assessment_file, co2, h2o, ch4, gas4).values()

                methods = {k:v for k, v in zip(['none', 'uvw', 'vw'], range(1000))}
                if separation in methods:
                    separation = methods[separation]
                FluxCorrection_SpectralAnalysis_General['horst_lens'] = separation
                return dict(FluxCorrection_SpectralAnalysis_General=FluxCorrection_SpectralAnalysis_General, RawProcess_Settings=RawProcess_Settings)
            def _configure_fratini(
                self,
                assessment_file: PathLike | str | None = None,
                co2: Sequence[float, float] = (0.005, 2.),
                h2o: Sequence[float, float] = (0.005, 2.),
                ch4: Sequence[float, float] = (0.005, 2.),
                gas4: Sequence[float, float] = (0.005, 2.),
                separation: Literal['none', 'uvw', 'vw'] | int = 'none',
                full_wts_dir: str | PathLike[str] | None = None,
                include_anemometer_losses: bool = True
            ) -> dict:
                """
                configure the settings needed to use Fratini 2012 spectral corrections

                Parameters
                ----------
                assessment_file: path to a spectral assessment file for this dataset. If False (none), compute spectral assessment on-the-fly
                co2/h2o/ch4/gas4: sequence of the form (low, high) indicating the the frequency range for fitting in-situe transfer functions, based on temperature and concentration spectra. Default 0-005-2Hz.
                separation: how to correct for instrument separation. If none or 0 (default), do not correct for instrument separation. If uvw or 1, correct for separation in the u, v, and w directions. If vw, only correct for separation in the v and w directions. 
                full_wts_dir: if full-length w/Ts cospectra are available for this dataset, provide a path to that directory. If None (default), compute full w/Ts cospectra on the fly.
                include_anemometer_losses: if True (default), correct w/Ts cospectra for anemometer path averaging and time response losses before using them as a model. 

                limits on inputs
                all inputs 0-50Hz, with high > low

                Returns
                dicts of ini parameter settings
                """
                
                # check inputs
                if full_wts_dir is not None:
                    assert or_isinstance(full_wts_dir, str, Path), 'full_wts_dir must be a str or path to a directory'
                assert isinstance(include_anemometer_losses, bool), 'include_anemometer_losses must be bool'

                # horst and ibrom settings also apply to fratini
                FluxCorrection_SpectralAnalysis_General, RawProcess_Settings = self._configure_ibrom(assessment_file, co2, h2o, ch4, gas4, separation).values()
                

                Project = dict()

                # if full spectra are available, we don't need full spectra outputs.
                Project['full_sp_avail'] = 0
                RawProcess_Settings['out_full_cosp_w_ts'] = 1
                if full_wts_dir is not None:
                    FluxCorrection_SpectralAnalysis_General['sa_full_spectra'] = full_wts_dir
                    Project['full_sp_avail'] = 1
                    RawProcess_Settings['out_full_cosp_w_ts'] = 0
                
                FluxCorrection_SpectralAnalysis_General['add_sonic_lptf'] = int(include_anemometer_losses)

                return dict(FluxCorrection_SpectralAnalysis_General=FluxCorrection_SpectralAnalysis_General, RawProcess_Settings=RawProcess_Settings, Project=Project)          
            def set_hf_correction(
                self,
                low_pass_method: Literal['none', 'moncrieff', 'horst', 'ibrom', 'fratini', 'massman'] | int = 'moncrieff',
                horst_kwargs: dict | None = None,
                ibrom_kwargs: dict | None = None,
                fratini_kwargs: dict | None = None,
            ):
                """how to apply low-pass filtering effects

                Parameters
                ---------
                low_pass_method: one of 'none', 'moncrieff', 'horst', 'ibrom', 'fratini', 'massman' or int 0-5 to indicate whih low-pass filtering correction method to apply. If 'horst', 'ibrom', or 'fratini' is selected, then the corresponding kwargs dict must be provided too.
                horst/ibrom/fratini_kwargs: kwargs passed to one of _configure_horst, _configure_ibrom, or _configure_fratini. Provide the one matching the low_pass_method provided. Required for horst, ibrom, and fratini.
                """
                # history
                history_args = ('Advanced-Spectra', 'hf_correction', self.get_hf_correction)
                self.root._add_to_history(*history_args, True)
                
                # check inputs
                methods = ['none', 'moncrieff', 'horst', 'ibrom', 'fratini', 'massman']
                assert or_isinstance(low_pass_method, str, int), "low_pass_method must be one of 'none' (0), 'moncrieff' (1), 'horst' (2), 'ibrom' (3), 'fratini' (4), or 'massman' (5)"
                if isinstance(low_pass_method, str):
                    assert low_pass_method in methods, "low_pass_method must be one of 'none' (0), 'moncrieff' (1), 'horst' (2), 'ibrom' (3), 'fratini' (4), or 'massman' (5)"
                if isinstance(low_pass_method, int):
                    assert in_range(low_pass_method, '[0, 5]'), "low_pass_method must be one of 'none' (0), 'moncrieff' (1), 'horst' (2), 'ibrom' (3), 'fratini' (4), or 'massman' (5)"
                    low_pass_method = methods[low_pass_method]
                
                # none, moncrieff, and massman require no settings
                # horst, ibrom, and fratini require additoinal settings
                settings = None
                match low_pass_method:
                    case 'none':
                        if horst_kwargs is not None or ibrom_kwargs is not None or fratini_kwargs is not None:
                            warnings.warn('no correction method provided. Ignoring all method-specific kwargs')
                        self.root.set('Project', 'hf_meth', '0')
                    case 'moncrieff':
                        if horst_kwargs is not None or ibrom_kwargs is not None or fratini_kwargs is not None:
                            warnings.warn('moncrieff method provided. Ignoring all method-specific kwargs')
                        self.root.set('Project', 'hf_meth', '1')
                    case 'horst':
                        assert isinstance(horst_kwargs, dict), 'horst_kwargs must be a dict of kwargs to pass to _configure_horst'
                        if ibrom_kwargs is not None or fratini_kwargs is not None:
                            warnings.warn('horst method provided. Ignoring ibrom and fratini kwargs')
                        self.root.set('Project', 'hf_meth', '2')
                        settings = self._configure_horst(**horst_kwargs)
                    case 'ibrom':
                        assert isinstance(ibrom_kwargs, dict), 'ibrom_kwargs must be a dict of kwargs to pass to _configure_ibrom'
                        if ibrom_kwargs is not None or fratini_kwargs is not None:
                            warnings.warn('ibrom method provided. Ignoring horst and fratini kwargs')
                        self.root.set('Project', 'hf_meth', '3')
                        settings = self._configure_ibrom(**ibrom_kwargs)
                    case 'fratini':
                        assert isinstance(fratini_kwargs, dict), 'fratini_kwargs must be a dict of kwargs to pass to _configure_fratini'
                        if ibrom_kwargs is not None or horst_kwargs is not None:
                            warnings.warn('fratini method provided. Ignoring horst and ibrom kwargs')
                        self.root.set('Project', 'hf_meth', '4')
                        settings = self._configure_fratini(**fratini_kwargs)
                    case 'massman':
                        if horst_kwargs is not None or ibrom_kwargs is not None or fratini_kwargs is not None:
                            warnings.warn('massman method provided. Ignoring all method-specific kwargs')
                        self.root.set('Project', 'hf_meth', '5')
                    
                if settings is not None:
                    for section in settings:
                        for option, value in settings[section].items():
                            self.root.set(section, option, str(value))

                self.root._add_to_history(*history_args)
                return      
            def get_hf_correction(self):
                out = dict()

                methods = ['none', 'moncrieff', 'horst', 'ibrom', 'fratini', 'massman']
                method = methods[int(self.root.get('Project', 'hf_meth'))]

                out['low_pass_method'] = method

                method_kwargs = dict()
                if method in ['horst', 'ibrom', 'fratini']:
                    method_kwargs['method'] = dict()
                    if not int(self.root.get('FluxCorrection_SpectralAnalysis_General', 'sa_mode')):
                        method_kwargs['assessment_file'] = self.root.get('FluxCorrection_SpectralAnalysis_General', 'sa_file')
                    else:
                        method_kwargs['co2'] = (
                            float(self.root.get('FluxCorrection_SpectralAnalysis_General', 'sa_fmin_co2')), 
                            float(self.root.get('FluxCorrection_SpectralAnalysis_General', 'sa_fmax_co2')))
                        method_kwargs['h2o'] = (
                            float(self.root.get('FluxCorrection_SpectralAnalysis_General', 'sa_fmin_h20')), 
                            float(self.root.get('FluxCorrection_SpectralAnalysis_General', 'sa_fmax_h20')))
                        method_kwargs['ch4'] = (
                            float(self.root.get('FluxCorrection_SpectralAnalysis_General', 'sa_fmin_ch4')), 
                            float(self.root.get('FluxCorrection_SpectralAnalysis_General', 'sa_fmax_ch4')))
                        method_kwargs['gas4'] = (
                            float(self.root.get('FluxCorrection_SpectralAnalysis_General', 'sa_fmin_gas4')), 
                            float(self.root.get('FluxCorrection_SpectralAnalysis_General', 'sa_fmax_gas4')))
                if method in ['ibrom', 'fratini']:
                    separation_methods = ['none', 'uvw', 'vw']
                    method_kwargs['separation'] = separation_methods[int(self.root.get('FluxCorrection_SpectralAnalysis_General', 'horst_lens'))]
                if method == 'fratini':
                    if int(self.root.get('Project', 'full_sp_avail')):
                        method_kwargs['full_wts_dir'] = self.root.get('RawProcess_Settings', 'out_full_cosp_w_ts')
                    method_kwargs['include_anemometer_losses'] = bool(int(self.root.get('FluxCorrection_SpectralAnalysis_General', 'add_sonic_lptf')))
                
                match method:
                    case 'horst': out['horst_kwargs'] = method_kwargs
                    case 'ibrom': out['ibrom_kwargs'] = method_kwargs
                    case 'fratini': out['fratini_kwargs'] = method_kwargs
                
                return out
                
        class _Out:
            def __init__(self, outer):
                self.root = outer.root
                self.outer = outer

            def set_results(
                self,
                full_output: bool = True,
                output_only_available: bool = True,
                fluxnet_labels_units: bool = True,
                err_label: Literal['fluxnet', '-9999.0', '-6999.0', 'NaN', 'Error', 'N/A', 'NOOP'] = 'fluxnet',
                continuous: bool = True,
                biomet: bool = True,
                details_f04: bool = False,
                metadata: bool = True,
            ):
                """
                General settings for output files 

                Parameters
                ----------
                full_output: bool, whether to generate the main eddypro output file, the fulloutput file. Default True
                output_only_available: if True (default), include in the fulloutput file only the results which are actually available, eliminating “error code” columns that are created when results are unavailable. If False output all results, whether they are available or not.
                fluxnet_labels_units: bool, default True. If True, use the fluxnet standard for variable labels and units
                err_label: str, one of 'fluxnet', '-9999.0', '-6999.0', 'NaN', 'Error', 'N/A', 'NOOP', default 'fluxnet'. If 'fluxnet', use the fluxnet standard fro error labelling. If any other input, use that string as the error label.
                continuous: bool, default True, whether to output a continuous time series, with empty time slots filled with err_label
                biomet: bool, default True, whether to output biomet data in a unique file.
                details_f04: bool, default False, whether to report the details of the Foken et al 2004 steady state and developed turbulence tests
                metadata: bool, default True, whether to output metadata as a single timeseries file, compatible with 'dynamic metadata' for eddypro
                """
                history_args = ('Advanced-Output', 'results', self.get_results)
                self.root._add_to_history(*history_args, True)

                assert isinstance(full_output, bool), 'full_output must be bool'
                assert isinstance(output_only_available, bool), 'output_format must be bool'
                assert isinstance(fluxnet_labels_units, bool), 'fluxnet_labels_units must be bool'
                assert err_label in ['fluxnet', '-9999.0', '-6999.0', 'NaN', 'Error', 'N/A', 'NOOP'], "err_label must be one of 'fluxnet', '-9999.0', '-6999.0', 'NaN', 'Error', 'N/A', 'NOOP'"
                assert isinstance(continuous, bool), 'continuous must be bool'
                assert isinstance(biomet, bool), 'biomet must be bool'
                assert isinstance(details_f04, bool), 'details_f04 must be bool'
                assert isinstance(metadata, bool), 'metadata must be bool'

                self.root.set('Project', 'out_rich', str(int(full_output)))
                self.root.set('Project', 'fix_out_format', str(1 - int(output_only_available)))
                self.root.set('Project', 'fluxnet_standardize_biomet', str(int(fluxnet_labels_units)))

                if err_label == 'fluxnet':
                    self.root.set('Project', 'fluxnet_err_label', '1')
                else:
                    self.root.set('Project', 'fluxnet_err_label', '0')
                    self.root.set('Project', 'err_label', err_label)
                
                self.root.set('Project', 'make_dataset', str(int(continuous)))
                self.root.set('Project', 'out_biomet', str(int(biomet)))
                self.root.set('Project', 'out_metadata', str(int(metadata)))
                self.root.set('RawProcess_Settings', 'out_qc_details', str(int(details_f04)))

                self.root._add_to_history(*history_args, False)
                return
            def get_results(self):
                out = dict()
                out['full_output'] = bool(int(self.root.get('Project', 'out_rich')))
                out['output_only_available'] = bool(1 - int(self.root.get('Project', 'fix_out_format')))
                out['fluxnet_labels_units'] = bool(int(self.root.get('Project', 'fluxnet_standardize_biomet')))
                
                if int(self.root.get('Project', 'fluxnet_err_label')):
                    out['err_label'] = 'fluxnet'
                else:
                    out['err_label'] = self.root.get('Project', 'err_label')
                
                out['continuous'] = bool(int(self.root.get('Project', 'make_dataset')))
                out['biomet'] = bool(int(self.root.get('Project', 'out_biomet')))
                out['metadata'] = bool(int(self.root.get('Project', 'out_metadata')))
                out['details_f04'] = bool(int(self.root.get('RawProcess_Settings', 'out_qc_details')))

                return out

            def set_spectral_output(
                self,    
                binned_spectra: bool = True,  # NEED FOR MANY SPECTRAL METHODS
                binned_ogives: bool = True,
                ensemble_spectra: bool = True,
                ensemble_cospectra: bool = True,
                full_spectra: list[Literal['u', 'v', 'w', 'ts', 'co2', 'h2o', 'ch4', 'gas4']] | Literal['u', 'v', 'w', 'ts', 'co2', 'h2o', 'ch4', 'gas4', 'all', 'none'] = 'none',
                full_cospectra: list[Literal['w/u', 'w/v', 'w/ts', 'w/co2', 'w/h2o', 'w/ch4', 'w/gas4']] | Literal['w/u', 'w/v', 'w/ts', 'w/co2', 'w/h2o', 'w/ch4', 'w/gas4', 'all', 'none'] = 'w/ts',  # NEED FOR FRATINI
            ):
                """
                Settings for spectral output files

                Parameters
                ----------
                binned_spectra: bool, default True, whether to output binned spectra. This argument MUST be set to true when binned cospectra files are not available from a previous run (when binned_cosp_dir argument is None in Spectral.set_calculation)
                binned_ogives: bool, default True, whether to output binned ogives
                ensemble_spectra: bool, default True, whether to output ensemble averaged spectra
                ensemble_cospectra: bool, default True, whether to output ensemble averaged cospectra
                full_spectra: variables to output full-length spectra for. Sequence or string containing one or several of 'u', 'v', 'w', 'ts', 'co2', 'h2o', 'ch4', 'gas4'. Alternatively, provide 'all' to output all such variables, or 'none' to output none. Default 'none'.
                full_cospectra: variables to output full-length cospectra for. Sequence or string containing one or several of 'w/u', 'w/v', 'w/ts', 'w/co2', 'w/h2o', 'w/ch4', 'w/gas4'. Alternatively, provide 'all' to output all such variables, or 'none' to output none. Default 'w/ts'. If using the Fratini method for high frequency corrections without full w/ts cospectra files available, this argument must include 'w/ts'
                """
                history_args = ('Advanced-Output', 'spectral_output', self.get_spectral_output)
                self.root._add_to_history(*history_args, True)
                assert isinstance(binned_spectra, bool), 'binned_spectra must be bool'
                if not binned_spectra:
                    if 'binned_cosp_dir' not in self.root.Spectral.get_calculation():
                        warnings.warn('you should not set binned_spectra to False when binned cospectra are not available for this dataset. Either set binned_spectra to True or point eddypro to a previous set of spectral data by running Spectral.set_calculation the the binned_cosp_dir argument')
                assert isinstance(binned_ogives, bool), 'binned_ogives must be bool'
                assert isinstance(ensemble_spectra, bool), 'ensemble_spectra must be bool'
                assert isinstance(ensemble_cospectra, bool), 'ensemble_cospectra must be bool'
                assert or_isinstance(full_spectra, str, Sequence), 'full_spectra must be sequence of str or str'
                if isinstance(full_spectra, list):
                    for v in full_spectra:
                        assert v in ['u', 'v', 'w', 'ts', 'co2', 'h2o', 'ch4', 'gas4'], "if full_spectra is a sequence, it must be a sequence of 'u', 'v', 'w', 'ts', 'co2', 'h2o', 'ch4', or 'gas4'"
                if isinstance(full_spectra, str):
                    assert full_spectra in ['u', 'v', 'w', 'ts', 'co2', 'h2o', 'ch4', 'gas4', 'all', 'none'], "if full_spectra is str, it must be one of 'u', 'v', 'w', 'ts', 'co2', 'h2o', 'ch4', 'gas4', 'all', 'none'"
                    full_spectra = [full_spectra]
                assert or_isinstance(full_spectra, str, Sequence), 'full_spectra must be sequence of str or str'
                if isinstance(full_cospectra, list):
                    for v in full_cospectra:
                        assert v in ['w/u', 'w/v', 'w/ts', 'w/co2', 'w/h2o', 'w/ch4', 'w/gas4'], "if full_cospectra is a sequence, it must be a sequence of 'w/u', 'w/v', 'w/ts', 'w/co2', 'w/h2o', 'w/ch4', or 'w/gas4'"
                if isinstance(full_cospectra, str):
                    assert full_cospectra in ['w/u', 'w/v', 'w/ts', 'w/co2', 'w/h2o', 'w/ch4', 'w/gas4', 'all', 'none'], "if full_cospectra is str, it must be one of 'w/u', 'w/v', 'w/ts', 'w/co2', 'w/h2o', 'w/ch4', 'w/gas4', 'all', 'none'"
                    full_cospectra = [full_cospectra]
                if 'w/ts' not in full_cospectra:
                    if 'fratini_kwargs' in self.root.Adv.Spec.get_hf_correction():
                        if 'full_wts_dir' not in self.root.Spectral.get_hf_correction()['fratini_kwargs']:
                            warnings.warn("when using the Fratini method for high-frequency spectral corrections without available w/Ts spectra, you should include 'w/ts' in the list of full-length cospectral outputs. Either add 'w/ts' to the full_cospectra argument, or run Spectral.set_hf_correction() with full_wts_dir in fratini_kwargs")
                
                self.root.set('RawProcess_Settings', 'out_bin_sp', str(int(binned_spectra)))
                self.root.set('RawProcess_Settings', 'out_bin_og', str(int(binned_ogives)))
                self.root.set('Project', 'out_mean_spec', str(int(ensemble_spectra)))
                self.root.set('Project', 'out_mean_cosp', str(int(ensemble_cospectra)))

                all_spectra = ['u', 'v', 'w', 'ts', 'co2', 'h2o', 'ch4', 'gas4']
                if 'none' in full_spectra:
                    zero_sp = all_spectra
                    one_sp = []
                elif 'all' in full_spectra:
                    zero_sp = []
                    one_sp = all_spectra
                else:
                    zero_sp = list(set(all_spectra).difference(full_spectra))
                    one_sp = full_spectra
                for s in zero_sp:
                    if s == 'gas4': s = 'n2o'
                    self.root.set('RawProcess_Settings', f'out_full_sp_{s}', '0')
                for s in one_sp:
                    if s == 'gas4': s = 'n2o'
                    self.root.set('RawProcess_Settings', f'out_full_sp_{s}', '1')
                
                all_cospectra = ['w/u', 'w/v', 'w/ts', 'w/co2', 'w/h2o', 'w/ch4', 'w/gas4']
                if 'none' in full_cospectra:
                    zero_sp = all_cospectra
                    one_sp = []
                elif 'all' in full_cospectra:
                    zero_sp = []
                    one_sp = all_cospectra
                else:
                    zero_sp = list(set(all_cospectra).difference(full_cospectra))
                    one_sp = full_cospectra
                for s in zero_sp:
                    if s == 'w/gas4': s = 'w/n2o'
                    self.root.set('RawProcess_Settings', f'out_full_cosp_w_{s[2:]}', '0')
                for s in one_sp:
                    if s == 'w/gas4': s = 'w/n2o'
                    self.root.set('RawProcess_Settings', f'out_full_cosp_w_{s[2:]}', '1')
                
                self.root._add_to_history(*history_args)
                return
            def get_spectral_output(self):
                out = dict()
                out['binned_spectra'] = bool(int(self.root.get('RawProcess_Settings', 'out_bin_sp')))
                out['binned_ogives'] = bool(int(self.root.get('RawProcess_Settings', 'out_bin_og')))
                out['ensemble_spectra'] = bool(int(self.root.get('Project', 'out_mean_spec')))
                out['ensemble_cospectra'] = bool(int(self.root.get('Project', 'out_mean_cosp')))

                full_spectra = []
                for s in ['u', 'v', 'w', 'ts', 'co2', 'h2o', 'ch4', 'n2o']:
                    if int(self.root.get('RawProcess_Settings', f'out_full_sp_{s}')):
                        if s == 'n2o': s = 'gas4'
                        full_spectra.append(s)
                if len(full_spectra) == 0:
                    full_spectra = 'none'
                elif len(full_spectra) == 8:
                    full_spectra = 'all'
                out['full_spectra'] = full_spectra

                full_cospectra = []
                for s in ['w/u', 'w/v', 'w/ts', 'w/co2', 'w/h2o', 'w/ch4', 'w/n2o']:
                    if int(self.root.get('RawProcess_Settings', f'out_full_cosp_w_{s[2:]}')):
                        if s == 'w/n2o': s = 'w/gas4'
                        full_cospectra.append(s)
                if len(full_cospectra) == 0:
                    full_cospectra = 'none'
                elif len(full_cospectra) == 8:
                    full_cospectra = 'all'
                out['full_cospectra'] = full_cospectra

                return out
            
            def set_intermediate_results(
                self,
                unprocessed: Literal['stats', 'timeseries', 'both', 'none'] = 'stats',
                despiked: Literal['stats', 'timeseries', 'both', 'none'] = 'none',
                crosswind_corrected: Literal['stats', 'timeseries', 'both', 'none'] = 'none',
                aoa_corrected: Literal['stats', 'timeseries', 'both', 'none'] = 'none',
                tilt_corrected: Literal['stats', 'timeseries', 'both', 'none'] = 'none',
                timelag_corrected: Literal['stats', 'timeseries', 'both', 'none'] = 'none',
                detrended: Literal['stats', 'timeseries', 'both', 'none'] = 'none',
                variables: list[Literal['u', 'v', 'w', 'ts', 'co2', 'h2o', 'ch4', 'gas4', 'ta', 'pa', 'all', 'none']] = 'none'
            ):
                """
                Settings for intermediate result output files/chain of custody tracking

                Parameters
                ----------
                for each parameter other than variables: one of 'stats', 'timeseries', 'both', or 'none' specifying whether to output just the statistics, or the full timeseries for that level of processing.
                variables: sequence of strings indicating which timeseries to output data for when timeseries is selected"""
                history_args = ('Advanced-Output', 'intermediate_results', self.get_intermediate_results)
                self.root._add_to_history(*history_args, True)

                for v in [unprocessed, despiked, crosswind_corrected, aoa_corrected, tilt_corrected, timelag_corrected, detrended]:
                    assert v in ['stats', 'timeseries', 'both', 'none'], "unprocessed, despiked, crosswind_corrected, aoa_corrected, tilt_corrected, timelag_corrected, and detrended must be one of 'stats', 'timeseries', 'both', 'none'"
                assert isinstance(variables, Sequence)
                if isinstance(variables, list):
                    for v in variables:
                        assert v in ['u', 'v', 'w', 'ts', 'co2', 'h2o', 'ch4', 'gas4', 'ta', 'pa'], "If variables is a sequence, it can only contain 'u', 'v', 'w', 'ts', 'co2', 'h2o', 'ch4', 'gas4', 'ta', 'pa'"
                if isinstance(variables, str):
                    assert variables in ['u', 'v', 'w', 'ts', 'co2', 'h2o', 'ch4', 'gas4', 'ta', 'pa', 'all', 'none'], "If variables is a string, it must be one of 'u', 'v', 'w', 'ts', 'co2', 'h2o', 'ch4', 'gas4', 'ta', 'pa', 'all', 'none'"
                    variables = [variables]

                for i, v in enumerate([unprocessed, despiked, crosswind_corrected, aoa_corrected, tilt_corrected, timelag_corrected, detrended]):
                    level = i + 1
                    match v:
                        case 'stats':
                            self.root.set('RawProcess_Settings', f'out_st_{level}', '1')
                            self.root.set('RawProcess_Settings', f'out_raw_{level}', '0')
                        case 'timeseries':
                            self.root.set('RawProcess_Settings', f'out_st_{level}', '0')
                            self.root.set('RawProcess_Settings', f'out_raw_{level}', '1')
                        case 'both':
                            self.root.set('RawProcess_Settings', f'out_st_{level}', '1')
                            self.root.set('RawProcess_Settings', f'out_raw_{level}', '1')
                        case 'none':
                            self.root.set('RawProcess_Settings', f'out_st_{level}', '0')
                            self.root.set('RawProcess_Settings', f'out_raw_{level}', '0')
                
                all_vars = set(['u', 'v', 'w', 'ts', 'co2', 'h2o', 'ch4', 'gas4', 'ta', 'pa'])
                if 'all' in variables:
                    zero_vars = set()
                    one_vars = all_vars
                elif 'none' in variables:
                    zero_vars = all_vars
                    one_vars = set()
                else:
                    one_vars = set(variables)
                    zero_vars = all_vars.difference(one_vars)
                for v in zero_vars:
                    if v == 'pa': v = 'p_air'
                    elif v == 'ta': v = 't_air'
                    self.root.set('RawProcess_Settings', f'out_raw_{v}', '0')
                for v in one_vars:
                    if v == 'pa': v = 'p_air'
                    elif v == 'ta': v = 't_air'
                    self.root.set('RawProcess_Settings', f'out_raw_{v}', '1')    

                self.root._add_to_history(*history_args)
                return
            def get_intermediate_results(self):
                out = dict()

                for i, k in enumerate(['unprocessed', 'despiked', 'crosswind_corrected', 'aoa_corrected', 'tilt_corrected', 'timelag_corrected', 'detrended']):
                    level = i + 1
                    match (self.root.get('RawProcess_Settings', f'out_st_{level}'), self.root.get('RawProcess_Settings', f'out_raw_{level}')):
                        case '1', '0': out[k] = 'stats'
                        case '0', '1': out[k] = 'timeseries'
                        case '0', '0': out[k] = 'none'
                        case '1', '1': out[k] = 'both'
                
                variables = []
                for k in ['u', 'v', 'w', 'ts', 'co2', 'h2o', 'ch4', 'gas4', 't_air', 'p_air']:
                    if int(self.root.get('RawProcess_Settings', f'out_raw_{k}')):
                        if k == 't_air': k = 'ta'
                        elif k == 'p_air': k = 'pa'
                        variables.append(k)
                if len(variables) == 0:
                    variables = 'none'
                elif len(variables) == 10:
                    variables = 'all'
                out['variables'] = variables
                return out

            
if __name__ == '__main__':
    pass
    import argparse
    from pathlib import Path
    import sys
    from datetime import datetime

    wd = Path('/Users/alex/Documents/Work/UWyo/Research/Flux Pipeline Project/Eddypro-ec-testing/workflows/BB-NF_17m')
    sys.path.append('/Users/alex/Documents/Work/UWyo/Research/Flux Pipeline Project/Eddypro-ec-testing/python')
    from eddyproconfigeditor import EddyproConfigEditor

    arcc_wd = Path('/project/eddycovworkflow/afox18/Platinum-EddyPro7/workflows/BB-NF_17m')
    data_dir = Path('/gscratch/afox18/eddycovworkflow/InputData/Chimney')
    out_dir = Path('/gscratch/afox18/eddycovworkflow/ExpectedOutputs/Chimney/BB-NF/17m')

    template = EddyproConfigEditor(wd / 'ini/BB-NF_17m_template.eddypro')

    template.Proj.set_metadata(arcc_wd / 'ini/BB-NF_17m.metadata')
    template.Proj.set_biomet(mode='dir', path=data_dir / 'biomet/BB-NF/EC/17m/EddyPro_Biomet', extension='dat', subfolders=False)
    template.Proj.set_project_name('BB-NF_17m')

    template.Basic.set_raw_data(path=data_dir / 'raw/BB-NF/EC/17m/Calibrated', fmt='yyyy_mm_dd_HHMM.dat', subfolders=False)
    template.Basic.set_out_path(out_dir / 'Template')
    template.Basic.set_project_date_range(start='2019-01-01 00:00', end='2023-12-31 23:30')
    template.Basic.set_missing_samples_allowance(pct=10)
    template.Basic.set_flux_averaging_interval(minutes=30)
    template.Basic.set_north_reference(method='mag')
    template.Basic.set_output_id(output_id='template')

    template.Adv.Proc.set_wind_speed_measurement_offsets(0, 0, 0)
    template.Adv.Proc.set_axis_rotations_for_tilt_correction(
        method='planar_fit',
        configure_planar_fit_settings_kwargs=dict(
            w_max=0.5,
            u_min=0.5,
            num_per_sector_min=30,
            start='project',
            end='project',
            fix_method='CW',
            north_offset=0,
            sectors=[(False, 90)]*4
        )
    )
    template.Adv.Proc.set_turbulent_fluctuations(detrend_method='block')
    template.Adv.Proc.set_timelag_compensations(method='covariance_maximization_with_default')
    template.Adv.Proc.set_compensation_of_density_fluctuations(
        burba_method='multiple',
        day_bot=[3.0935, -0.0819, 0.0018, -0.3601],
        day_top=[0.5773, -0.0107, 0.0012, -0.0914],
        day_spar=[0.7714, -0.0154, 0.0011, -0.1164],
        night_bot=[2.2022, -0.122, 0, -0.3001],
        night_top=[-0.2505, -0.0303, 0, 0.0556],
        night_spar=[0.0219, -0.0361, 0, 0.0145]
    )

    template.Adv.Out.set_spectral_output(
        binned_spectra=True,
        binned_ogives=True,
        ensemble_spectra=True,
        full_spectra='none',
        full_cospectra='none'
    )
    template.Adv.Out.set_intermediate_results(
        unprocessed='stats',
        despiked='stats',
        timelag_corrected='timeseries',
        variables=['u', 'v', 'w', 'ts', 'co2', 'h2o', ]
    )

    template.to_eddypro(ini_file=wd / 'ini/BB-NF_17m_template.eddypro')
    template.to_eddypro_parallel(
        environment_parent=wd / 'ini/BB-NF_17m_template_parallel',
        out_parent=out_dir / 'Template_Parallel',
        file_duration=1440,
        worker_windows=[datetime(y, 1, 1, 0, 0, 0) for y in range(2019, 2025)]
    )