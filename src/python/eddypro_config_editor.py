import configparser
from pathlib import Path
from typing import Literal, Self
import datetime
from os import PathLike
import os
from collections.abc import Sequence
import multiprocessing
from math import ceil
import warnings
import tempfile
from time import sleep
import sys
from copy import copy

from pandas import date_range, Timedelta, DataFrame

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
    ** DONE Axis rotations for tilt correction
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
    * when writing output, make sure that the program will be able to run.
"""


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
    df_compare = (df1_new .merge(df2_new, on=['Section', 'Option'], suffixes=[
        '_' + name1, '_' + name2]) .sort_values(['Section', 'Option']))
    return df_compare


class EddyproConfigEditor(configparser.ConfigParser):
    '''
    a child class of configparser.ConfigParser.
    Adds specific methods to work with .eddypro INI files

    Parameters
    ----------
    reference_ini: path to a .eddypro file to modify
    '''

    def __init__(self, reference_ini: str | PathLike[str]):
        super().__init__(allow_no_value=True)
        self.read(reference_ini)

        self._start_set = False
        self._end_set = False

        self.Basic = self._Basic(self)
        self.Advanced = self._Advanced(self)

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
        out_path: the path for eddypro to output results to. If 'keep' (default), use the outpath already in the config file
        """
        self.set('Project', 'file_name', str(ini_file))
        if out_path != 'keep':
            self.set('Project', 'out_path', str(out_path))
        with open(ini_file, 'w') as configfile:
            configfile.write(';EDDYPRO_PROCESSING\n')  # header line
            self.write(fp=configfile, space_around_delimiters=False)

    def to_eddypro_parallel(
        self,
        ini_dir: str | PathLike[str],
        out_path: str | PathLike[str],
        metadata_fn: str | PathLike[str] | None = None,
        num_workers: int | None = None,
        file_duration: int | None = None,
        min_worker_timespan: int | None = None,
    ) -> None:
        """
        split this config file up into a set of .eddypro files, each handling a smaller time chunk that the main config.
        all .eddypro files will be identical except in their project IDs, file names, and start/end dates.

        Note that some processing methods are not compatible "out-of-the-box" with paralle processing in this format,
        specifically those methods which rely on aggregate data from longer time series. This includes:
            * planar fit methods require at least 2 weeks worth of data, ideally more.
            * automatic time-lag optimization
            * most spectral assessments require at least one month of data, ideally more.
        To solve this issue, you can do one of the following:
            * set the minimum worker timspan to be at least one month
            * provide sufficient and representative spectral assesment, binned cospectra, planar fit, and/or automatic time lag optimization files from a previous run by modifying the axis_rotations_for_tilt_correction settings, the timelag_compenstaions settings, the spectra_calculations settings, and/or the spectra_qaqc settings.
            * use methods that do not require long time spans of data.

        
        Parameters
        ----------
        ini_dir: the directory to output configured .eddypro files to. Does not have to exist.
        out_path: the path to direct eddypro to write results to.
        metadata_fn: path to a static .metadata file for this project. Must be provided if file_duration is None.
        num_workers: the number of parallel processes to configure. If None (default), then processing is split up according to the number of available processors on the machine minus 1.
        file_duration: how many minutes long each file is (NOT the averaging interval). If None (Default), then that information will be gleaned from the metadata file.
        min_worker_timespan: the minimum amount of data each worker can process, in days. If None (default), then set no minimum. Recommended if using methods that require aggregate data (see above)
        """

        # get file duration
        if file_duration is None:
            assert metadata_fn is not None, 'metadata_fn must be provided'
            metadata = configparser.ConfigParser()
            metadata.read(metadata_fn)
            file_duration = int(metadata['Timing']['file_duration'])

        if num_workers is None:
            num_workers = max(multiprocessing.cpu_count() - 1, 1)

        # split up file processing dates
        start = str(
            datetime.datetime.strptime(
                f"{self.get('Project', 'pr_start_date')} {self.get('Project', 'pr_start_time')}",
                r'%Y-%m-%d %H:%M'))
        end = str(
            datetime.datetime.strptime(
                f"{self.get('Project', 'pr_end_date')} {self.get('Project', 'pr_end_time')}",
                r'%Y-%m-%d %H:%M'))

        n_files = len(date_range(start, end, freq=f'{file_duration}min'))
        job_size = ceil(file_duration * n_files / num_workers)
        job_size = f'{int(ceil(job_size/file_duration)*file_duration)}min'

        if min_worker_timespan is not None:
            job_size = f'{int(min_worker_timespan)}d'
        job_starts = date_range(
            start,
            end - Timedelta(job_size),
            freq=job_size)
        if len(job_starts) <= 1: warnings.warn(f"WARNING: job size too long. Submitting {len(job_starts)} jobs")
        # dates are inclusive, so subtract 30min for file duration
        job_ends = job_starts + Timedelta(job_size) - Timedelta(file_duration)
        job_start_dates = job_starts.strftime(date_format=r'%Y-%m-%d')
        job_start_times = job_starts.strftime(date_format=r'%H:%M')
        job_end_dates = job_ends.strftime(date_format=r'%Y-%m-%d')
        job_end_times = job_ends.strftime(date_format=r'%H:%M')

        # give each project a unique id and file name
        project_ids = [
            f'worker{start}' for start in job_starts.strftime(
                date_format=r"%Y%m%d%H%M")]
        ini_fns = [
            ini_dir /
            f'{project_id}.eddypro' for project_id in project_ids]

        # save original settings
        old_file_name = self.get('Project', 'file_name')
        old_out_path = self.get('Project', 'out_path')
        pr_start_date = self.get('Project', 'pr_start_date')
        pr_end_date = self.get('Project', 'pr_end_date')
        pr_start_time = self.get('Project', 'pr_start_time')
        pr_end_time = self.get('Project', 'pr_end_time')
        project_id = self.get('Project', 'project_id')

        # write new files
        if not os.path.isdir(Path(ini_dir)):
            Path.mkdir(Path(ini_dir))
        for i, fn in enumerate(ini_fns):
            self.set('Project', 'out_path', str(out_path))
            self.set('Project', 'file_name', str(fn))
            self.set('Project', 'pr_start_date', str(job_start_dates[i]))
            self.set('Project', 'pr_end_date', str(job_end_dates[i]))
            self.set('Project', 'pr_start_time', str(job_start_times[i]))
            self.set('Project', 'pr_end_time', str(job_end_times[i]))
            self.set('Project', 'project_id', str(project_ids[i]))

            with open(fn, 'w') as configfile:
                configfile.write(';EDDYPRO_PROCESSING\n')  # header line
                self.write(fp=configfile, space_around_delimiters=False)

        # revert to original
        self.set('Project', 'file_name', old_file_name)
        self.set('Project', 'out_path', old_out_path)
        self.set('Project', 'pr_start_date', pr_start_date)
        self.set('Project', 'pr_end_date', pr_end_date)
        self.set('Project', 'pr_start_time', pr_start_time)
        self.set('Project', 'pr_end_time', pr_end_time)
        self.set('Project', 'project_id', project_id)

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

    def copy(self) -> Self:
        """copies this config through a temporary file"""
        tmp = tempfile.NamedTemporaryFile(mode='w', delete=False)
        try:
            tmp.write(';EDDYPRO_PROCESSING\n')  # header line
            self.write(fp=tmp, space_around_delimiters=False)
            tmp.close()

            tmp = open(tmp.name, mode='r')
            cls = self.__class__
            new = self.__new(cls)
            new.__init__(tmp.name)
            tmp.close()

            os.remove(tmp.name)
        except BaseException:
            tmp.close()
            os.remove(tmp.name)
            raise

        new._start_set = self._start_set
        new._end_set = self._end_set

        return new

    def __copy__(self):
        return self.copy()

    # --------------------Basic Settings Page-----------------------
    class _Basic:
        def __init__(self, root):
            self.root = root

        def set_start_date(
            self,
            start: str | datetime.datetime | None = None,
        ) -> None:
            """format yyyy-mm-dd HH:MM for strings"""
            if isinstance(start, str):
                assert len(
                    start) == 16, 'if using a string, must pass timestamps in YYYY-mm-DD HH:MM format'
                pr_start_date, pr_start_time = start.split(' ')
            else:
                pr_start_date = start.strftime(r'%Y-%m-%d')
                pr_start_time = start.strftime(r'%H:%M')

            if start is not None:

                self.root.set('Project', 'pr_start_date', str(pr_start_date))
                self.root.set('Project', 'pr_start_time', str(pr_start_time))

            self.root._start_set = True

        def get_start_date(self) -> datetime.datetime:
            """retrieve form the config file the project start date."""
            start_date = self.root.get('Project', 'pr_start_date')
            start_time = self.root.get('Project', 'pr_start_time')
            return datetime.datetime.strptime(
                f'{start_date} {start_time}', r'%Y-%m-%d %H:%M')

        def set_end_date(
            self,
            end: str | datetime.datetime | None = None
        ) -> None:
            """format yyyy-mm-dd HH:MM for strings"""
            if isinstance(end, str):
                assert len(
                    end) == 16, 'if using a string, must pass timestamps in YYYY-mm-DD HH:MM format'
                pr_end_date, pr_end_time = end.split(' ')
            else:
                pr_end_date = end.strftime(r'%Y-%m-%d')
                pr_end_time = end.strftime(r'%H:%M')
            if end is not None:
                self.root.set('Project', 'pr_end_date', str(pr_end_date))
                self.root.set('Project', 'pr_end_time', str(pr_end_time))

            self.root._end_set = True

        def get_end_date(self) -> datetime.datetime:
            """retrieve form the config file the project end date."""
            end_date = self.root.get('Project', 'pr_end_date')
            end_time = self.root.get('Project', 'pr_end_time')
            return datetime.datetime.strptime(
                f'{end_date} {end_time}', r'%Y-%m-%d %H:%M')

        def set_date_range(
            self,
            start: str | datetime.datetime | None = None,
            end: str | datetime.datetime | None = None
        ):
            """format yyyy-mm-dd HH:MM for strings"""
            self.set_start_date(start)
            self.set_end_date(end)

        def get_date_range(self) -> dict:
            """retrieve form the config file the project start and end dates. Output can be can be passed to set_date_range__ as kwargs"""
            start = self.get_start_date()
            end = self.get_end_date()
            return dict(start=start, end=end)

        def set_missing_samples_allowance(self, pct: int):
            # pct: value from 0 to 40%
            assert pct >= 0 and pct <= 40
            self.root.set('RawProcess_Settings', 'max_lack', str(int(pct)))

        def get_missing_samples_allowance(self) -> int:
            """retrieve form the config file the maximum allowed missing samples per averaging window in %."""
            return int(self.root.get('RawProcess_Settings', 'max_lack'))

        def set_flux_averaging_interval(self, minutes: int):
            """minutes: how long to set the averaging interval to. If 0, use the file as-is"""
            assert minutes >= 0 and minutes <= 9999, 'Must have 0 <= minutes <= 9999'
            self.root.set('RawProcess_Settings', 'avrg_len', str(int(minutes)))

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

            method: one of 'mag' or 'geo'
            magnetic_declination: a valid magnetic declination as a real number between -90 and 90. If 'geo' is selected, magnetic declination must be provided. Otherwise, does nothing.
            declination_date: the reference date for magnetic declination, either as a yyyy-mm-dd string or as a datetime.datetime object. If method = 'geo', then declination date must be provided. Otherwise, does nothing.
            """

            assert method in [
                'mag', 'geo'], "Method must be one of 'mag' (magnetic north) or 'geo' (geographic north)"

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

        def set_project_id(self, project_id: str):
            assert ' ' not in project_id and '_' not in project_id, 'project id must not contain spaces or underscores.'
            self.root.set('Project', 'project_id', str(project_id))

        def get_project_id(self) -> str:
            """retrieve form the config file the project project ID"""
            return self.root.get('Project', 'project_id')

    # --------------------Advanced Settings Page-----------------------
    class _Advanced:
        def __init__(self, root):
            self.root = root
            self.Processing = self._Processing(self)
            self.Statistical = self._Statistical(self)
            self.Spectral = self._Spectral(self)
            self.Output = self._Output(self)

        # --------Processing Options---------
        class _Processing:
            def __init__(self, outer):
                self.outer = outer
                self.root = outer.root

            def set_wind_speed_measurement_offsets(
                    self, u: float = 0, v: float = 0, w: float = 0):
                assert max(
                    u**2, v**2, w**2) <= 100, 'Windspeed measurement offsets cannot exceed ±10m/s'
                self.root.set('RawProcess_Settings', 'u_offset', str(u))
                self.root.set('RawProcess_Settings', 'v_offset', str(v))
                self.root.set('RawProcess_Settings', 'w_offset', str(w))

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
                u_min: float = 0,
                start: str | datetime.datetime | None = None,
                end: str | datetime.datetime | None = None,
                num_per_sector_min: int = 0,
                fix_method: Literal['CW', 'CCW', 'double_rotations'] | int = 'CW',
                north_offset: int = 0,
                sectors: Sequence[Sequence[bool | int, float]] | None = None,
            ) -> dict:
                """outputs a dictionary of planarfit settings
                w_max: the maximum mean vertical wind component for a time interval to be included in the planar fit estimation
                u_min: the minimum mean horizontal wind component for a time interval to be included in the planar fit estimation
                start, end: start and end date-times for planar fit computation. If a string, must be in yyyy-mm-dd HH:MM format. If None (default), set to the date range of the processing file.
                num_per_sector_min: the minimum number of valid datapoints for a sector to be computed. Default 0.
                fix_method: one of CW, CCW, or double_rotations or 0, 1, 2. The method to use if a planar fit computation fails for a given sector. Either next valid sector clockwise, next valid sector, counterclockwise, or double rotations. Default is next valid sector clockwise.
                north_offset: the offset for the counter-clockwise-most edge of the first sector in degrees from -180 to 180. Default 0.
                sectors: list of tuples of the form (exclude, width). Where exclude is either a bool (False, True), or an int (0, 1) indicating whether to ingore this sector entirely when estimating planar fit coefficients. Width is a float between 0.1 and 359.9 indicating the width, in degrees of a given sector. Widths must add to one. If None (default), provide no sector information.

                Returns: a dictionary to provide to set_axis_rotations_for_tiltCorrection
                """

                # start/end date/time
                if start is not None:
                    if isinstance(start, str):
                        assert len(
                            start) == 16, 'datetime strings must be in yyyy-mm-dd HH:MM format'
                        pf_start_date, pf_start_time = start.split(' ')
                    else:
                        pf_start_date = start.strftime(r'%Y-%m-%d')
                        pf_start_time = start.strftime(r'%H:%M')
                else:
                    pf_start_date = self.root.get('Project', 'pr_start_date')
                    pf_start_time = self.root.get('Project', 'pr_start_time')
                    if not self.root._start_set:
                        warnings.warn(
                            f"Warning: Using the start date and time provided by the original reference file: {pf_start_date} {pf_start_time}")
                if end is not None:
                    if isinstance(end, str):
                        assert len(
                            end) == 16, 'datetime strings must be in yyyy-mm-dd HH:MM format'
                        pf_end_date, pf_end_time = end.split(' ')
                    else:
                        pf_end_date = end.strftime(r'%Y-%m-%d')
                        pf_end_time = end.strftime(r'%H:%M')
                else:
                    pf_end_date = self.root.get('Project', 'pr_end_date')
                    pf_end_time = self.root.get('Project', 'pr_end_time')
                    if not self.root._start_set:
                        warnings.warn(
                            f"Warning: Using the end date and time provided by the original reference file: {pf_end_date} {pf_end_time}")

                # simple settings
                assert u_min >= 0 and u_min <= 10, 'must have 0 <= u_min <= 10'
                assert w_max > 0 and w_max <= 10, 'must have 0 < w_max <= 10'
                assert isinstance(
                    num_per_sector_min, int) and num_per_sector_min >= 0 and num_per_sector_min <= 9999, 'must have 0 <= num_sectors_min <= 9999'
                assert fix_method in ['CW', 'CCW', 'double_rotations', 0, 1,
                                      2], 'fix method must be one of CW, CCW, double_rotations, 0, 1, 2'
                fix_dict = dict(CW=0, CCW=1, double_rotations=2)
                if isinstance(fix_method, str):
                    fix_method = fix_dict[fix_method]

                assert north_offset >= - \
                    179.9 and north_offset <= 180, 'must have -179.9 <= north_offset <= 180'

                settings_dict = dict(
                    pf_start_date=pf_start_date,
                    pf_start_time=pf_start_time,
                    pf_end_date=pf_end_date,
                    pf_end_time=pf_end_time,
                    pf_u_min=u_min,
                    pf_w_max=w_max,
                    pf_min_num_per_sec=int(num_per_sector_min),
                    pf_fix=int(fix_method),
                    pf_north_offset=north_offset,
                )

                # sectors
                if sectors is not None:
                    assert len(
                        sectors) <= 10, "Can't have more than 10 sectors"
                    total_width = 0
                    for _, width in sectors:
                        total_width += width
                    assert total_width <= 360, 'Sector widths cannot add up to more than 360.'
                    for i, sector in enumerate(sectors):
                        exclude, width = sector
                        n = i + 1
                        settings_dict[f'pf_sector_{n}_exclude'] = int(exclude)
                        settings_dict[f'pf_sector_{n}_width'] = str(width)

                return settings_dict

            def set_axis_rotations_for_tilt_correction(
                self,
                method: Literal['none', 'double_rotations', 'triple_rotations', 'planar_fit', 'planar_fit_nvb'] | int,
                pf_file: str | PathLike[str] | None = None,
                configure_planar_fit_settings_kwargs: dict | None = None,
            ):
                """
                method: one of 0 or "none" (no tilt correction), 1 or "double_rotations" (double rotations), 2 or "triple_rotations" (triple rotations), 3 or "planar_fit" (planar fit, Wilczak 2001), 4 or "planar_fit_nvb" (planar with with no velocity bias (van Dijk 2004)). one of pf_file or pf_settings_kwargs must be provided if method is a planar fit type.
                pf_file: Mututally exclusive with pf_settings_kwargs. If method is a planar fit type, path to an eddypro-compatible planar fit file. This can be build by hand, or taken from the output of a previous eddypro run. Typically labelled as "eddypro_<project id>_planar_fit_<timestamp>_adv.txt"
                pf_settings_kwargs: Mututally exclusive with pf_file. Arguments to be passed to configure_planar_fit_settings.
                """
                method_dict = {
                    'none': 0,
                    'double_rotations': 1,
                    'triple_rotations': 2,
                    'planar_fit': 3,
                    'planar_fit_nvb': 4}
                if isinstance(method, str):
                    assert method in ['none', 'double_rotations', 'triple_rotations', 'planar_fit',
                                      'planar_fit_nvb'], 'method must be one of None, double_rotations, triple_rotations, planar_fit, planar_fit_nvb, or 0, 1, 2, 3, or 4.'
                    method = method_dict[method]
                assert method in range(
                    5), 'method must be one of None, double_rotations, triple_rotations, planar_fit, planar_fit_nvb, or 0, 1, 2, 3, or 4.'

                self.root.set('RawProcess_Settings', 'rot_meth', str(method))

                # planar fit
                if method in [3, 4]:
                    assert bool(pf_file) != bool(
                        configure_planar_fit_settings_kwargs), 'If method is a planar-fit type, exactly one of pf_file or pf_settings should be specified.'
                    if pf_file is not None:
                        self.root.set(
                            'RawProcess_TiltCorrection_Settings',
                            'pf_file',
                            str(pf_file))
                        self.root.set(
                            'RawProcess_TiltCorrection_Settings', 'pf_mode', str(0))
                        self.root.set(
                            'RawProcess_TiltCorrection_Settings', 'pf_subset', str(1))
                    elif configure_planar_fit_settings_kwargs is not None:
                        self.root.set(
                            'RawProcess_TiltCorrection_Settings', 'pf_file', '')
                        self.root.set(
                            'RawProcess_TiltCorrection_Settings', 'pf_mode', str(1))
                        self.root.set(
                            'RawProcess_TiltCorrection_Settings', 'pf_subset', str(1))
                        pf_settings = self._configure_planar_fit_settings(
                            **configure_planar_fit_settings_kwargs)
                        for option, value in pf_settings.items():
                            self.root.set(
                                'RawProcess_TiltCorrection_Settings', option, str(value))

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
                if method in ['planar_fit', 'planar_fit_nvb']:
                    configure_planar_fit_settings_kwargs = dict()
                    # case that a manual configuration is provided
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
                                    f'pf_sector_{n}_exclude'))
                            width = float(
                                self.root.get(
                                    'RawProcess_TiltCorrection_Settings',
                                    f'pf_sector_{n}_width'))
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

            def set_turbulent_fluctuations(self,
                                           method: Literal['block',
                                                           'detrend',
                                                           'running_mean',
                                                           'exponential_running_mean'] | int = 0,
                                           time_const: float | None = None):
                '''time constant in seconds not required for block averaging (0) (default)'''
                method_dict = {
                    'block': 0,
                    'detrend': 1,
                    'running_mean': 2,
                    'exponential_running_mean': 3}
                if isinstance(method, str):
                    assert method in method_dict, 'method must be one of block, detrend, running_mean, exponential_running_mean'
                    method = method_dict[method]
                if time_const is None:
                    # default for linear detrend is flux averaging interval
                    if method == 1:
                        time_const = 0.
                    # default for linear detrend is 250s
                    elif method in [2, 3]:
                        time_const = 250.
                self.root.set(
                    'RawProcess_Settings',
                    'detrend_meth',
                    str(method))
                self.root.set(
                    'RawProcess_Settings',
                    'timeconst',
                    str(time_const))

            def _configure_timelag_auto_opt(
                self,
                start: str | datetime.datetime | None = None,
                end: str | datetime.datetime | None = None,
                ch4_min_lag: float | None = None,
                ch4_max_lag: float | None = None,
                ch4_min_flux: float = 0.200,
                co2_min_lag: float | None = None,
                co2_max_lag: float | None = None,
                co2_min_flux: float = 2.000,
                gas4_min_lag: float | None = None,
                gas4_max_lag: float | None = None,
                gas4_min_flux: float = 0.020,
                h2o_min_lag: float | None = None,  # -1000.1 is default
                h2o_max_lag: float | None = None,
                le_min_flux: float = 20.0,
                h2o_nclass: int = 10,
                pg_range: float = 1.5,
            ) -> dict:
                """
                configure settings for automatic time lag optimization.
                start, end: the time period to consider when performing automatic timelag optimization. Default (None) is to use the whole timespan of the data.
                CO2, CH4, and 4th gas:
                    x_min/max_lag: the minimum and maximum allowed time lags in seconds. Must be between -1000 and +1000, and x_max_lag > x_min_lag. If None (default), then detect automatically.
                    x_min_flux: the minimum allowed flux to perform time lag adjustments on, in µmol/m2/s.
                H2O:
                    h2o_min/max_lag: identical to co2/ch4/gas4_min/max_lag.
                    le_min_flux: the minimum allowed flux to perform time lag adjustments on, in W/m2
                    h2o_nclass: the number of RH classes to consider when performing time lag optimization.
                pg_range: the number of median absolute deviations from the mean a time lag can be for a given class to be accepted. Default mean±1.5mad
                """

                # start/end date/time
                if start is not None:
                    if isinstance(start, str):
                        assert len(
                            start) == 16, 'datetime strings must be in yyyy-mm-dd HH:MM format'
                        to_start_date, to_start_time = start.split(' ')
                    else:
                        to_start_date = start.strftime(r'%Y-%m-%d')
                        to_start_time = start.strftime(r'%H:%M')
                else:
                    to_start_date = self.root.get('Project', 'pr_start_date')
                    to_start_time = self.root.get('Project', 'pr_start_time')
                    if not self.root._start_set:
                        warnings.warn(
                            f"Warning: Using the start date and time provided by the original reference file: {to_start_date} {to_start_time}")
                if end is not None:
                    if isinstance(end, str):
                        assert len(
                            end) == 16, 'datetime strings must be in yyyy-mm-dd HH:MM format'
                        to_end_date, to_end_time = end.split(' ')
                    else:
                        to_end_date = end.strftime(r'%Y-%m-%d')
                        to_end_time = end.strftime(r'%H:%M')
                else:
                    to_end_date = self.root.get('Project', 'pr_end_date')
                    to_end_time = self.root.get('Project', 'pr_end_time')
                    if not self.root._start_set:
                        warnings.warn(
                            f"Warning: Using the end date and time provided by the original reference file: {to_end_date} {to_end_time}")

                # lag settings default to "automatic detection" for the value
                # -1000.1
                settings_with_special_defaults = [
                    ch4_min_lag,
                    ch4_max_lag,
                    co2_min_lag,
                    co2_max_lag,
                    gas4_min_lag,
                    gas4_max_lag,
                    h2o_min_lag,
                    h2o_max_lag]
                for i, setting in enumerate(settings_with_special_defaults):
                    if setting is None:
                        settings_with_special_defaults[i] = str(-1000.1)
                ch4_min_lag, ch4_max_lag, co2_min_lag, co2_max_lag, gas4_min_lag, gas4_max_lag, h2o_min_lag, h2o_max_lag = settings_with_special_defaults

                settings_dict = dict(
                    to_start_date=to_start_date,
                    to_start_time=to_start_time,
                    to_end_date=to_end_date,
                    to_end_time=to_end_time,
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
                    to_h2o_nclass=int(h2o_nclass),
                    to_pg_range=pg_range,
                )

                return settings_dict

            def set_timelag_compensations(self,
                                          method: Literal['none',
                                                          'constant',
                                                          'covariance_maximization_with_default',
                                                          'covariance_maximization',
                                                          'automatic_optimization'] | int = 2,
                                          autoopt_file: PathLike[str] | str | None = None,
                                          configure_TimelagAutoOpt_kwargs: dict | None = None):
                """
                method: one of 0 or "none" (no time lag compensation), 1 or "constant" (constant time lag from instrument metadata), 2 or "covariance_maximization_with_default" (Default), 3 or "covariance_maximization", or 4 or "automatic_optimization." one of autoopt_file or autoopt_settings_kwargs must be provided if method is a planar fit type.
                autoopt_file: Mututally exclusive with autoopt_settings_kwargs. If method is a planar fit type, path to an eddypro-compatible automatic time lag optimization file. This can be build by hand, or taken from the output of a previous eddypro run. Typically labelled as "eddypro_<project id>_timelag_opt_<timestamp>_adv.txt" or similar
                autoopt_settings_kwargs: Mututally exclusive with autoopt_file. Arguments to be passed to configure_TimelagAutoOpt.
                """
                method_dict = {
                    'none': 0,
                    'constant': 1,
                    'covariance_maximization_with_default': 2,
                    'covariance_maximization': 3,
                    'automatic_optimization': 4}
                if isinstance(method, str):
                    assert method in ['none', 'constant', 'covariance_maximization_with_default', 'covariance_maximization',
                                      'automatic_optimization'], 'method must be one of None, double_rotations, triple_rotations, planar_fit, planar_fit_nvb, or 0, 1, 2, 3, or 4.'
                    method = method_dict[method]
                assert method in range(
                    5), 'method must be one of None, constant, covariance_maximization_with_default, covariance_maximization, automatic_optimization, or 0, 1, 2, 3, or 4.'

                self.root.set('RawProcess_Settings', 'tlag_meth', str(method))

                # planar fit
                if method == 4:
                    assert bool(autoopt_file) != bool(
                        configure_TimelagAutoOpt_kwargs), 'If method is a planar-fit type, exactly one of pf_file or pf_settings should be specified.'
                    if autoopt_file is not None:
                        self.root.set(
                            'RawProcess_TimelagOptimization_Settings',
                            'to_file',
                            str(autoopt_file))
                        self.root.set(
                            'RawProcess_TimelagOptimization_Settings', 'to_mode', str(0))
                        self.root.set(
                            'RawProcess_TimelagOptimization_Settings', 'to_subset', str(1))
                    elif configure_TimelagAutoOpt_kwargs is not None:
                        self.root.set(
                            'RawProcess_TimelagOptimization_Settings', 'to_file', '')
                        self.root.set(
                            'RawProcess_TimelagOptimization_Settings', 'to_mode', str(1))
                        self.root.set(
                            'RawProcess_TimelagOptimization_Settings', 'to_subset', str(1))
                        to_settings = self._configure_timelag_auto_opt(
                            **configure_TimelagAutoOpt_kwargs)
                        for option, value in to_settings.items():
                            self.root.set(
                                'RawProcess_TimelagOptimization_Settings', option, str(value))

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
                    options = [f'm_{name}_{i}' for i in [1, 2, 3, 4]]
                    assert len(
                        coeffs) == 2, 'Multiple regression coefficients must be a sequence of length four, representing (offset, Ta_gain, Rg_gain, U_gain)'
                    for option, value in zip(options, coeffs):
                        self.root.set(
                            'RawProcess_Settings', option, str(value))
                elif estimation_method == 'simple':
                    options = [f'l_{name}_{i}' for i in ['gain', 'offset']]
                    assert len(
                        coeffs) == 2, 'Simple regression coefficients must be a sequence of length two, representing (gain, offset)'
                    for option, value in zip(options, coeffs):
                        self.root.set(
                            'RawProcess_Settings', option, str(value))

            def set_compensation_of_density_fluctuations(
                    self,
                    enable: bool = True,
                    burba_correction: bool = False,
                    estimation_method: Literal['simple', 'multiple'] | None = None,
                    day_bot: Sequence | Literal['revert'] | None = None,
                    day_top: Sequence | Literal['revert'] | None = None,
                    day_spar: Sequence | Literal['revert'] | None = None,
                    night_bot: Sequence | Literal['revert'] | None = None,
                    night_top: Sequence | Literal['revert'] | None = None,
                    night_spar: Sequence | Literal['revert'] | None = None,
                    set_all: Literal['revert'] | None = None,
            ):
                """how to correct for density fluctuations. Default mode is to only correct for bulk density fluctuations.

                enable: If true, correct for density fluctuations with the WPL term (default)
                burba_correction: If true, add instrument sensible heat components. LI-7500 only. Default False.
                estimation_method: one of 'simple' or 'multiple'. Whether to use simple linear regression or Multiple linear regression. if burba_correction is enabled, this argument cannot be None (default)
                day/night_bot/top/spar: Either (a) 'default' (default) (b) 'keep', or (c) a sequence of regression coefficients for the burba correction for the temperature of the bottom, top, and spar of the LI7500.
                    If 'simple' estimation was selected, then this is a sequence of length two, representing (gain, offset) for the equation
                        T_instrument = gain*Ta + offset
                    If 'multiple' estimation was selected, then this is a sequence of length 4, repressinting (offset, Ta_coeff, Rg_coeff, U_coeff) for the equation
                        T_instr - Ta = offset + Ta_coeff*Ta + Rg_coeff*Rg + U_coeff*U
                        where Ta is air temperature, T_instr is instrument part temperature, Rg is global incoming SW radiation, and U is mean windspeed

                    If 'revert,' then revert to default eddypro coefficients.
                    If None (selected by default), then do not change regression coefficients in the file
                set_all: as an alternative to specifying day/night_bot/top/spar, you can provide all = 'revert' to revert all burba correction settings to their eddypro defaults. Default None (do nothing).
                """

                if not enable:
                    self.root.set('Project', 'wpl_meth', '0')
                    if burba_correction:
                        warnings.warn(
                            'WARNING: burba_correction has no effect when density fluctuation compensation is disabled')
                else:
                    self.root.set('Project', 'wpl_meth', '1')

                if not burba_correction:
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
                            'WARNING: burba regression coefficients have no effect when burba correction is disabled or density corrections are disabled.')
                else:
                    assert estimation_method in [
                        'simple', 'multiple'], 'estimation method must be one of "simple", "multiple"'
                    self.root.set('RawProcess_Settings', 'bu_corr', '1')

                if estimation_method == 'simple':
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

                elif estimation_method == 'multiple':
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

            def get_compensation_of_density_fluctuations(self) -> dict:

                out = dict()
                out['enable'] = bool(int(self.root.get('Project', 'wpl_meth')))
                if out['enable']:
                    out['burba_correction'] = bool(
                        int(self.get('RawProcess_Settings', 'bu_corr')))

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
                                        self.root.get(
                                            'RawProcess_Settings',
                                            f'm_{k}_{i}')) for i in range(
                                        1,
                                        5))
                        else:
                            for k in kwargs:
                                out[k] = tuple(
                                    float(
                                        self.root.get(
                                            'RawProcess_Settings',
                                            f'l_{k}_{i}')) for i in [
                                        'gain',
                                        'offset'])

                return out

         # --------Statistical Analysis---------

        class _Statistical:
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
                        'RawProcess_ParameterSettings', 'filter_sr', '1')
                else:
                    self.root.set(
                        'RawProcess_ParameterSettings', 'filter_sr', '0')
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

                return

            def get_spike_count_remova(self) -> dict:
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
                    int(self.root.get('RawProcess_ParameterSettings', 'filter_sr')))
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
                enable: whether to enable amplitude resolution flagging. Default True
                variation_range: the expected maximum z-score range for the data. Default ±7σ
                bins: int, the number of bins for the histogram. Default 100
                max_empty_bins: float, if more than max_empty_bins% of bins in the histogram are empty, flag for amplitude resolution problems

                limits on inputs:
                variation_range: 1-20
                bins: 50-150
                max_empty_bins: 1-100%
                """
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
                enable: whether to enable dropout flagging. Default True
                extreme_percentile: int, bins lower than this percentile in the histogram will be considered extreme. Default 10
                accepted_central_dropouts: If consecutive values fall within a non-extreme histogram bin, flag the instrument for a dropout. If more than accepted_central_dropouts% of the averaging interval are flagged as dropouts, flag the whole averagine interval. Default 10%
                accepted_extreme_dropouts: same as for accepted_central_dropouts, except for values in the extreme histogram range. Default 6%

                limits on inputs:
                extreme_percentile: 1-100
                accepted_central/extreme_droupouts: 1-100%
                """
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
                    # eddypro calls gas4 n2o
                    ['ts', 'co2', 'h2o', 'ch4', 'n2o'],
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
                        'RawProcess_ParameterSettings', 'filter_al', '1')
                    return
                self.root.set('RawProcess_ParameterSettings', 'filter_al', '0')
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
                    # eddypro calls gas4 n2o
                    ['ts', 'co2', 'h2o', 'ch4', 'n2o'],
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
                    int(self.root.get('RawProcess_ParameterSettings', 'filter_al')))

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
                enable: whether to enable discontinuity flagging. Default False
                u, w, ts, co2, h2o, ch4, gas4: a sequence of (hard, soft) specifying the hard and soft-flag thresholds for the haar transform on raw data. See eddypro documentation or Vickers and Mahrt 1997 for an explanation of thresholds.
                variances: same as above, but for variances rather than raw data.

                must have hard >= soft, and all values must be within the interval [0, 50]
                """
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
                enable: whether to enable flagging for excessive changes in covariance due to time lags (default False)
                covariance_difference: a tuple of (hard, soft) for covariance differences as a % between uncorrected and time-lag-corrected covariances, where hard defines the hard-flagging threshold and soft defines the soft-flagging threshold.
                co2/h2o/ch4/gas4: the expected time lags for each gas in seconds.

                limits on inputs:
                covariance_difference: 0-100%, with soft <= hard
                all other values: 0-100s
                """
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
                enable: whether to enable angle-of-attack flagging. Default False
                aoa_min: the minimum acceptable angle of attack in degrees. Default -30.
                aoa_max: the maximum acceptable angle of attack in degrees. Default 30.
                accepted_outliers: if more than accepted_outliers% of values lie outside the specified bounds, flag the averaging window. Default 10%.

                limits on inputs:
                aoa_min: -90 - 0°
                aoa_max: 0 - 90°
                accepted_outliers: 0-100%
                """
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

                enable: whether to enable flagging of horizontal wind steadiness. Default False.
                max_rel_inst: if the change in windspeed over the averaging window normalized by the mean windspeed exceeds this threshold, hard-flag the record. Default 0.5 for 50% relative instationarity.

                max_rel_inst should be within the interval [0, 50]
                """
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
                method: one of disable, FS01 (Finkelstein and Sims 2001), ML94 (Mann & Lenschow 1994), or M98 (Mahrt 1998), or 0, 1, or 2, respectively
                its_definition: definition of the integral turbulence scale. Options are 'at_1/e', 'at_0', 'whole_record', or 0, 1, or 2, respecitvely. See EddyPro documentation for more details.
                maximum_correlation_period: maximum time to integrate over when determining the integral turbulence scale. Default is 10.0s. Must be within [0, 10000] seconds
                """

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

        class _Spectral:
            def __init__(self, outer):
                self.root = outer.root
                self.outer = outer

        class _Output:
            def __init__(self, outer):
                self.root = outer.root
                self.outer = outer


if __name__ == '__main__':
    ref = EddyproConfigEditor(
        '/Users/alex/Documents/Work/UWyo/Research/Flux Pipeline Project/Eddypro-ec-testing/investigate_eddypro/ini/base.eddypro')
