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
    ** VM97
        ** Spike count/removal
        ** Ampl. res
        ** Dropouts
        ** Abs lims
        ** Skew + Kurt
        ** Discont.
        ** Time lags
        ** AoA
        ** Steadiness of Hor. Wind
    ** Random uncertainty estimates

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
"""

def compare_configs(df1: DataFrame, df2: DataFrame):
    """compare differences between two configs"""
    df1_new = df1.loc[df1['Value'] != df2['Value'], ['Section', 'Option', 'Value']]
    df2_new = df2.loc[df1['Value'] != df2['Value'], ['Section', 'Option', 'Value']]
    name1 = df1['Name'].values[0]
    name2 = df2['Name'].values[0]
    df_compare = (
        df1_new
        .merge(df2_new, on=['Section', 'Option'], suffixes=['_'+name1, '_'+name2])
        .sort_values(['Section', 'Option'])
    )
    return df_compare

class eddypro_ConfigParser(configparser.ConfigParser):
    '''a child class of configparser.ConfigParser added methods to modify eddypro-specific settings'''
    def __init__(self, reference_ini: str | PathLike[str]):
        '''reference_ini: a .eddypro file to modify'''
        super().__init__(allow_no_value=True)
        self.read(reference_ini)

        self._start_set = False
        self._end_set = False

    # --------------------Basic Settings Page-----------------------
    def set_StartDate(
        self,
        start: str | datetime.datetime | None = None, 
    ):
        if isinstance(start, str):
            pr_start_date, pr_start_time = start.split(' ')
        else:
            pr_start_date = start.strftime(r'%Y-%m-%d')
            pr_start_time = start.strftime(r'%H:%M')
        
        if start is not None:
            self.set(section='Project', option='pr_start_date', value=str(pr_start_date))
            self.set(section='Project', option='pr_start_time', value=str(pr_start_time))

        self._start_set = True 
    def get_StartDate(self) ->datetime.datetime:
        start_date = self.get(section='Project', option='pr_start_date')
        start_time = self.get(section='Project', option='pr_start_time')
        return datetime.datetime.strptime(f'{start_date} {start_time}', r'%Y-%m-%d %H:%M')
    
    def set_EndDate(
        self,
        end: str | datetime.datetime | None = None
    ):
        """format yyyy-mm-dd HH:MM for strings"""
        if isinstance(end, str):
            pr_end_date, pr_end_time = end.split(' ')
        else:
            pr_end_date = end.strftime(r'%Y-%m-%d')
            pr_end_time = end.strftime(r'%H:%M')
        if end is not None:
            self.set(section='Project', option='pr_end_date', value=str(pr_end_date))
            self.set(section='Project', option='pr_end_time', value=str(pr_end_time))

        self._end_set = True
    def get_EndDate(self) -> datetime.datetime:
        end_date = self.get(section='Project', option='pr_end_date')
        end_time = self.get(section='Project', option='pr_end_time')
        return datetime.datetime.strptime(f'{end_date} {end_time}', r'%Y-%m-%d %H:%M')
    
    def set_DateRange(
        self,
        start: str | datetime.datetime | None = None, 
        end: str | datetime.datetime | None = None
    ):
        """format yyyy-mm-dd HH:MM for strings"""
        self.set_start_date(start)
        self.set_end_date(end)
    def get_DateRange(self) -> dict:
        start = self.get_StartDate()
        end = self.get_EndDate()
        return dict(start=start, end=end)
        
    def set_MissingSamplesAllowance(self, pct: int):
        # pct: value from 0 to 40%
        assert pct >= 0 and pct <= 40
        self.set(section='RawProcess_Settings', option='max_lack', value=str(int(pct)))
    def get_MissingSamplesAllowance(self) -> int:
        return int(self.get(section='RawProcess_Settings', option='max_lack'))
    
    def set_FluxAveragingInterval(self, minutes: int):
        """minutes: how long to set the averaging interval to. If 0, use the file as-is"""
        assert minutes >= 0 and minutes <= 9999, 'Must have 0 <= minutes <= 9999'
        self.set(section='RawProcess_Settings', option='avrg_len', value=str(int(minutes)))
    def get_FluxAveragingInterval(self) -> int:
        return self.get(section='RawProcess_Settings', option='avrg_len')
    
    def set_NorthReference(
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

        assert method in ['mag', 'geo'], "Method must be one of 'mag' (magnetic north) or 'geo' (geographic north)"

        self.set(section='RawProcess_General', option='use_geo_north', value=str(int(method == 'geo')))
        if method == 'geo':
            assert magnetic_declination is not None and declination_date is not None, 'declination and declination date must be provided if method is "geo."'
            assert magnetic_declination >= -90 and magnetic_declination <= 90, "Magnetic declination must be between -90 and +90 (inclusive)"
            self.set(section='RawProcess_General', option='mag_dec', value=str(magnetic_declination))
            if isinstance(declination_date, str):
                declination_date, _ = declination_date.split(' ')
            else:
                declination_date = declination_date.strftime(r'%Y-%m-%d')
            self.set(section='RawProcess_General', option='dec_date', value=str(declination_date))
    def get_NorthReference(self) -> dict:
        use_geo_north = self.get(section='RawProcess_General', option='use_geo_north')
        if use_geo_north: use_geo_north = 'geo'
        else: use_geo_north = 'mag'

        mag_dec = float(self.get(section='RawProcess_General', option='mag_dec'))
        if use_geo_north == 'mag': mag_dec = None

        dec_date = datetime.datetime.strptime(self.get(section='RawProcess_General', option='dec_date'), r'%Y-%m-%d')
        if use_geo_north == 'mag': dec_date = None

        return dict(method=use_geo_north, magnetic_declination=mag_dec, declination_date=dec_date)
    
    def set_ProjectId(self, project_id: str):
        assert ' ' not in project_id and '_' not in project_id, 'project id must not contain spaces or underscores.'
        self.set(section='Project', option='project_id', value=str(project_id))
    def get_ProjectId(self) -> str:
        return self.get(section='Project', option='project_id')
    
    # --------------------Advanced Settings Page-----------------------
    # --------Processing Options---------
    def set_WindSpeedMeasurementOffsets(self, u: float = 0, v: float = 0, w: float = 0):
        assert max(u**2, v**2, w**2) <= 100, 'Windspeed measurement offsets cannot exceed ±10m/s'
        self.set(section='RawProcess_Settings', option='u_offset', value=str(u))
        self.set(section='RawProcess_Settings', option='v_offset', value=str(v))
        self.set(section='RawProcess_Settings', option='w_offset', value=str(w))
    def get_WindSpeedMeasurementOffsets(self) -> dict:
        return dict(
            u=float(self.get(section='RawProcess_Settings', option='u_offset')),
            v=float(self.get(section='RawProcess_Settings', option='v_offset')),
            w=float(self.get(section='RawProcess_Settings', option='w_offset'))
        )
    
    def _configure_PlanarFitSettings(
        self,
        w_max: float,
        u_min: float = 0,
        start: str | datetime.datetime | None = None,
        end: str | datetime.datetime | None = None,
        num_per_sector_min: int = 0,
        fix_method: Literal['CW', 'CCW', 'double_rotations'] | int = 'CW',
        north_offset: int = 0,
        sectors: Sequence[Sequence[bool | int, float]] | None  = None,
    ) -> dict:
        """outputs a dictionary of planarfit settings
        w_max: the maximum mean vertical wind component for a time interval to be included in the planar fit estimation
        u_min: the minimum mean horizontal wind component for a time interval to be included in the planar fit estimation
        start, end: start and end date-times for planar fit computation. If a string, must be in yyyy-mm-dd HH:MM format. If None (default), set to the date range of the processing file.
        num_per_sector_min: the minimum number of valid datapoints for a sector to be computed. Default 0.
        fix_method: one of CW, CCW, or double_rotations or 0, 1, 2. The method to use if a planar fit computation fails for a given sector. Either next valid sector clockwise, next valid sector, counterclockwise, or double rotations. Default is next valid sector clockwise.
        north_offset: the offset for the counter-clockwise-most edge of the first sector in degrees from -180 to 180. Default 0.
        sectors: list of tuples of the form (exclude, width). Where exclude is either a bool (False, True), or an int (0, 1) indicating whether to ingore this sector entirely when estimating planar fit coefficients. Width is a float between 0.1 and 359.9 indicating the width, in degrees of a given sector. Widths must add to one. If None (default), provide no sector information.

        Returns: a dictionary to provide to set_AxisRotationsForTiltCorrection
        """

        # start/end date/time
        if start is not None:
            if isinstance(start, str):
                pf_start_date, pf_start_time = start.split(' ')
            else:
                pf_start_date = start.strftime(r'%Y-%m-%d')
                pf_start_time = start.strftime(r'%H:%M')
        else:
            pf_start_date = self.get(section='Project', option='pr_start_date')
            pf_start_time = self.get(section='Project', option='pr_start_time')
            if not self._start_set:
                warnings.warn(f"Warning: Using the start date and time provided by the original reference file: {pf_start_date} {pf_start_time}")
        if end is not None:
            if isinstance(end, str):
                    pf_end_date, pf_end_time = end.split(' ')
            else:
                pf_end_date = end.strftime(r'%Y-%m-%d')
                pf_end_time = end.strftime(r'%H:%M')
        else:
            pf_end_date = self.get(section='Project', option='pr_end_date')
            pf_end_time = self.get(section='Project', option='pr_end_time')
            if not self._start_set:
                warnings.warn(f"Warning: Using the end date and time provided by the original reference file: {pf_end_date} {pf_end_time}")

        # simple settings
        assert u_min >= 0 and u_min <= 10, 'must have 0 <= u_min <= 10'
        assert w_max > 0 and w_max <= 10, 'must have 0 < w_max <= 10'
        assert isinstance(num_per_sector_min, int) and num_per_sector_min >= 0 and num_per_sector_min <= 9999, 'must have 0 <= num_sectors_min <= 9999'
        assert fix_method in ['CW', 'CCW', 'double_rotations', 0, 1, 2], 'fix method must be one of CW, CCW, double_rotations, 0, 1, 2'
        fix_dict = dict(CW = 0, CCW=1, double_rotations=2)
        if isinstance(fix_method, str):
            fix_method = fix_dict[fix_method]

        assert north_offset >= -179.9 and north_offset <= 180, 'must have -179.9 <= north_offset <= 180'

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
            assert len(sectors) <= 10, "Can't have more than 10 sectors"
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
    def set_AxisRotationsForTiltCorrection(
            self, 
            method: Literal['none', 'double_rotations', 'triple_rotations', 'planar_fit', 'planar_fit_nvb'] | int,
            pf_file: str | PathLike[str] | None = None,
            configure_PlanarFitSettings_kwargs: dict | None = None,
        ):
        """
        method: one of 0 or "none" (no tilt correction), 1 or "double_rotations" (double rotations), 2 or "triple_rotations" (triple rotations), 3 or "planar_fit" (planar fit, Wilczak 2001), 4 or "planar_fit_nvb" (planar with with no velocity bias (van Dijk 2004)). one of pf_file or pf_settings_kwargs must be provided if method is a planar fit type.
        pf_file: Mututally exclusive with pf_settings_kwargs. If method is a planar fit type, path to an eddypro-compatible planar fit file. This can be build by hand, or taken from the output of a previous eddypro run. Typically labelled as "eddypro_<project id>_planar_fit_<timestamp>_adv.txt"
        pf_settings_kwargs: Mututally exclusive with pf_file. Arguments to be passed to configure_PlanarFitSettings.
        """
        method_dict = {'none':0, 'double_rotations':1, 'triple_rotations':2, 'planar_fit':3, 'planar_fit_nvb':4}
        if isinstance(method, str):
            assert method in ['none', 'double_rotations', 'triple_rotations', 'planar_fit', 'planar_fit_nvb'], 'method must be one of None, double_rotations, triple_rotations, planar_fit, planar_fit_nvb, or 0, 1, 2, 3, or 4.'
            method = method_dict[method]
        assert method in range(5), 'method must be one of None, double_rotations, triple_rotations, planar_fit, planar_fit_nvb, or 0, 1, 2, 3, or 4.'

        self.set(section='RawProcess_Settings', option='rot_meth', value=str(method))

        # planar fit
        if method in [3, 4]:
            assert bool(pf_file) != bool(configure_PlanarFitSettings_kwargs), 'If method is a planar-fit type, exactly one of pf_file or pf_settings should be specified.'
            if pf_file is not None:
                self.set(section='RawProcess_TiltCorrection_Settings', option='pf_file', value=str(pf_file))
                self.set(section='RawProcess_TiltCorrection_Settings', option='pf_mode', value=str(0))
                self.set(section='RawProcess_TiltCorrection_Settings', option='pf_subset', value=str(1))
            elif configure_PlanarFitSettings_kwargs is not None:
                self.set(section='RawProcess_TiltCorrection_Settings', option='pf_file', value='')
                self.set(section='RawProcess_TiltCorrection_Settings', option='pf_mode', value=str(1))
                self.set(section='RawProcess_TiltCorrection_Settings', option='pf_subset', value=str(1))
                pf_settings = self._configure_PlanarFitSettings(**configure_PlanarFitSettings_kwargs)
                for option, value in pf_settings.items():
                    self.set(section='RawProcess_TiltCorrection_Settings', option=option, value=str(value))
    def get_AxisRotationsForTiltCorrection(self) -> tuple[str, dict]:
        methods = ['none', 'double_rotations', 'triple_rotations', 'planar_fit', 'planar_fit_nvb']
        method = methods[int(self.get(section='RawProcess_Settings', option='rot_meth'))]

        pf_config = dict()
        # case that a manual configuration is provided
        pf_config['pf_file'] = None
        start_date = self.get(section='RawProcess_TiltCorrection_Settings', option='pf_start_date')
        start_time = self.get(section='RawProcess_TiltCorrection_Settings', option='pf_start_time')
        if not start_date: start_date = self.get(section='Project', option='pr_start_date')
        if not start_time: start_time = self.get(section='Project', option='pr_start_time')
        pf_config['start'] = start_date + ' ' + start_time
        end_date = self.get(section='RawProcess_TiltCorrection_Settings', option='pf_end_date')
        end_time = self.get(section='RawProcess_TiltCorrection_Settings', option='pf_end_time')
        if not end_date: end_date = self.get(section='Project', option='pr_end_date')
        if not end_time: end_time = self.get(section='Project', option='pr_end_time')
        pf_config['end'] = end_date + ' ' + end_time

        pf_config['u_min'] = float(self.get(section='RawProcess_TiltCorrection_Settings', option='pf_u_min'))
        pf_config['w_max'] = float(self.get(section='RawProcess_TiltCorrection_Settings', option='pf_w_max'))
        pf_config['num_per_sector_min'] = int(self.get(section='RawProcess_TiltCorrection_Settings', option='pf_min_num_per_sec'))          
        fixes = ['CW', 'CCW', 'double_rotations']
        pf_config['fix_method'] = fixes[int(self.get(section='RawProcess_TiltCorrection_Settings', option='pf_fix'))]
        pf_config['north_offset'] = float(self.get(section='RawProcess_TiltCorrection_Settings', option='pf_north_offset'))
        
        n = 1
        sectors = []
        while True:
            try:
                exclude = int(self.get(section='RawProcess_TiltCorrection_Settings', option=f'pf_sector_{n}_exclude'))
                width = float(self.get(section='RawProcess_TiltCorrection_Settings', option=f'pf_sector_{n}_width'))
                sectors.append((exclude, width))
            except configparser.NoOptionError:
                break
            n += 1
        pf_config['sectors'] = sectors
        
        # case that a file config is provided
        manual_pf_config = int(self.get(section='RawProcess_TiltCorrection_Settings', option='pf_mode'))
        if not manual_pf_config:
            for k in pf_config:
                pf_config[k] = None
            pf_config['pf_file'] = self.get(section='RawProcess_TiltCorrection_Settings', option='pf_file')
        pf_config['pf_subset'] = int(self.get(section='RawProcess_TiltCorrection_Settings', option='pf_subset'))

        if method in ['none', 'double_rotations', 'triple_rotations']:
            pf_config = None
        
        return method, pf_config

    def set_TurbulentFluctuations(self, method: Literal['block', 'detrend', 'running_mean', 'exponential_running_mean'] | int = 0, time_const: float | None = None):
        '''time constant in seconds not required for block averaging (0) (default)'''
        method_dict = {'block':0, 'detrend':1, 'running_mean':2, 'exponential_running_mean':3}
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
        self.set(section='RawProcess_Settings', option='detrend_meth', value=str(method))
        self.set(section='RawProcess_Settings', option='timeconst', value=str(time_const))

    def _configure_TimeLagAutoOpt(
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
            h2o_min_lag: float | None = None,  #-1000.1 is default
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
                to_start_date, to_start_time = start.split(' ')
            else:
                to_start_date = start.strftime(r'%Y-%m-%d')
                to_start_time = start.strftime(r'%H:%M')
        else:
            to_start_date = self.get(section='Project', option='pr_start_date')
            to_start_time = self.get(section='Project', option='pr_start_time')
            if not self._start_set:
                warnings.warn(f"Warning: Using the start date and time provided by the original reference file: {to_start_date} {to_start_time}")
        if end is not None:
            if isinstance(end, str):
                    to_end_date, to_end_time = end.split(' ')
            else:
                to_end_date = end.strftime(r'%Y-%m-%d')
                to_end_time = end.strftime(r'%H:%M')
        else:
            to_end_date = self.get(section='Project', option='pr_end_date')
            to_end_time = self.get(section='Project', option='pr_end_time')
            if not self._start_set:
                warnings.warn(f"Warning: Using the end date and time provided by the original reference file: {to_end_date} {to_end_time}")

        # lag settings default to "automatic detection" for the value -1000.1
        settings_with_special_defaults = [ch4_min_lag ,ch4_max_lag ,co2_min_lag ,co2_max_lag ,gas4_min_lag ,gas4_max_lag ,h2o_min_lag ,h2o_max_lag]
        for i, setting in enumerate(settings_with_special_defaults):
            if setting is None:
                settings_with_special_defaults[i] = str(-1000.1)
        ch4_min_lag ,ch4_max_lag ,co2_min_lag ,co2_max_lag ,gas4_min_lag ,gas4_max_lag ,h2o_min_lag ,h2o_max_lag = settings_with_special_defaults

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

    def set_TimeLagCompensations(
            self, 
            method: Literal['none', 'constant', 'covariance_maximization_with_default', 'covariance_maximization', 'automatic_optimization'] | int = 2, 
            autoopt_file: PathLike[str] | str | None = None, 
            configure_TimeLagAutoOpt_kwargs:dict | None = None
        ):
        """
        method: one of 0 or "none" (no time lag compensation), 1 or "constant" (constant time lag from instrument metadata), 2 or "covariance_maximization_with_default" (Default), 3 or "covariance_maximization", or 4 or "automatic_optimization." one of autoopt_file or autoopt_settings_kwargs must be provided if method is a planar fit type.
        autoopt_file: Mututally exclusive with autoopt_settings_kwargs. If method is a planar fit type, path to an eddypro-compatible automatic time lag optimization file. This can be build by hand, or taken from the output of a previous eddypro run. Typically labelled as "eddypro_<project id>_timelag_opt_<timestamp>_adv.txt" or similar
        autoopt_settings_kwargs: Mututally exclusive with autoopt_file. Arguments to be passed to configure_TimeLagAutoOpt.
        """
        method_dict = {'none':0, 'constant':1, 'covariance_maximization_with_default':2, 'covariance_maximization':3, 'automatic_optimization':4}
        if isinstance(method, str):
            assert method in ['none', 'constant', 'covariance_maximization_with_default', 'covariance_maximization', 'automatic_optimization'], 'method must be one of None, double_rotations, triple_rotations, planar_fit, planar_fit_nvb, or 0, 1, 2, 3, or 4.'
            method = method_dict[method]
        assert method in range(5), 'method must be one of None, constant, covariance_maximization_with_default, covariance_maximization, automatic_optimization, or 0, 1, 2, 3, or 4.'

        self.set(section='RawProcess_Settings', option='tlag_meth', value=str(method))

        # planar fit
        if method == 4:
            assert bool(autoopt_file) != bool(configure_TimeLagAutoOpt_kwargs), 'If method is a planar-fit type, exactly one of pf_file or pf_settings should be specified.'
            if autoopt_file is not None:
                self.set(section='RawProcess_TimelagOptimization_Settings', option='to_file', value=str(autoopt_file))
                self.set(section='RawProcess_TimelagOptimization_Settings', option='to_mode', value=str(0))
                self.set(section='RawProcess_TimelagOptimization_Settings', option='to_subset', value=str(1))
            elif configure_TimeLagAutoOpt_kwargs is not None:
                self.set(section='RawProcess_TimelagOptimization_Settings', option='to_file', value='')
                self.set(section='RawProcess_TimelagOptimization_Settings', option='to_mode', value=str(1))
                self.set(section='RawProcess_TimelagOptimization_Settings', option='to_subset', value=str(1))
                to_settings = self._configure_TimeLagAutoOpt(**configure_TimeLagAutoOpt_kwargs)
                for option, value in to_settings.items():
                    self.set(section='RawProcess_TimelagOptimization_Settings', option=option, value=str(value))
    
    def _set_burba_coeffs(self, name, estimation_method, coeffs):
        """helper method called by set_compensationOfDensityFluctuations"""
        if estimation_method == 'multiple':
            options=[f'm_{name}_{i}' for i in [1, 2, 3, 4]]
            assert len(coeffs) == 2, 'Multiple regression coefficients must be a sequence of length four, representing (offset, Ta_gain, Rg_gain, U_gain)'
            for option, value in zip(options, coeffs):
                self.set(section='RawProcess_Settings', option=option, value=str(value))
        elif estimation_method == 'simple':
            options=[f'l_{name}_{i}' for i in ['gain', 'offset']]
            assert len(coeffs) == 2, 'Simple regression coefficients must be a sequence of length two, representing (gain, offset)'
            for option, value in zip(options, coeffs):
                self.set(section='RawProcess_Settings', option=option, value=str(value))

    def set_compensationOfDensityFluctuations(
            self,
            enable: bool = True,
            burba_correction: bool = False,
            estimation_method: Literal['simple', 'multiple'] = 'simple',
            day_bot: Sequence | Literal['keep', 'default'] = 'default',
            day_top: Sequence | Literal['keep', 'default'] = 'default',
            day_spar: Sequence | Literal['keep', 'default'] = 'default',
            night_bot: Sequence | Literal['keep', 'default'] = 'default',
            night_top: Sequence | Literal['keep', 'default'] = 'default',
            night_spar: Sequence | Literal['keep', 'default'] = 'default',
    ):
        """how o correct for density fluctuations. Default mode is to only correct for bulk density fluctuations.
        
        enable: If true, correct for density fluctuations with the WPL term (default)
        burba_correction: If true, add instrument sensible heat components. LI-7500 only. Default False.
        estimation_method: one of 'simple' or 'multiple'. Whether to use simple linear regression or Multiple linear regression.
        day/night_bot/top/spar: Either (a) 'default' (default) (b) 'keep', or (c) a sequence of regression coefficients for the burba correction for the temperature of the bottom, top, and spar of the LI7500. 
            If 'simple' estimation was selected, then this is a sequence of length two, representing (gain, offset) for the equation
                T_instrument = gain*Ta + offset
            If 'multiple' estimation was selected, then this is a sequence of length 4, repressinting (offset, Ta_coeff, Rg_coeff, U_coeff) for the equation
                T_instr - Ta = offset + Ta_coeff*Ta + Rg_coeff*Rg + U_coeff*U
                where Ta is air temperature, T_instr is instrument part temperature, Rg is global incoming SW radiation, and U is mean windspeed
            
            If 'default' (default), then revert to default eddypro coefficients.
            If 'keep', then do not change regression coefficients in the file
        """

        if not enable:
            self.set(section='Project', option='wpl_meth', value='0')
            if burba_correction:
                warnings.warn('WARNING: burba_correction has no effect when density fluctuation compensation is disabled')
        else:
            self.set(section='Project', option='wpl_meth', value='1')
        
        if not burba_correction:
            self.set(section='RawProcess_Settings', option='bu_corr', value='0')
            if (
                isinstance(day_bot, Sequence) 
                or isinstance(day_top, Sequence) 
                or isinstance(day_spar, Sequence) 
                or isinstance(night_bot, Sequence) 
                or isinstance(night_top, Sequence) 
                or isinstance(night_spar, Sequence)
                or not enable
            ):
                warnings.warn('WARNING: burba regression coefficients have no effect when burba correction is disabled or density corrections are disabled.')
        else:
            assert estimation_method in ['simple', 'multiple'], 'estimation method must be one of "simple (0)", "multiple (1)"'
            self.set(section='RawProcess_Settings', option='bu_corr', value='1')
        
        if estimation_method == 'simple':
            # daytime
            if day_bot == 'default': self._set_burba_coeffs('day_bot', 'simple', (0.944, 2.57))
            elif day_bot == 'keep': pass
            else: self._set_burba_coeffs('day_bot', 'simple', day_bot)

            if day_top == 'default': self._set_burba_coeffs('day_top', 'simple', (1.005, 0.24))
            elif day_top == 'keep': pass
            else: self._set_burba_coeffs('day_top', 'simple', day_top)
            
            if day_spar == 'default': self._set_burba_coeffs('day_spar', 'simple', (1.010, 0.36))
            elif day_spar == 'keep': pass
            else: self._set_burba_coeffs('day_spar', 'simple', day_spar)

            # nighttime
            if night_bot == 'default': self._set_burba_coeffs('night_bot', 'simple', (0.883, 2.17))
            elif night_bot == 'keep': pass
            else: self._set_burba_coeffs('night_bot', 'simple', night_bot)

            if night_top == 'default': self._set_burba_coeffs('night_top', 'simple', (1.008, -0.41))
            elif night_top == 'keep': pass
            else: self._set_burba_coeffs('night_top', 'simple', night_top)
            
            if night_spar == 'default': self._set_burba_coeffs('night_spar', 'simple', (1.010, -0.17))
            elif night_spar == 'keep': pass
            else: self._set_burba_coeffs('night_spar', 'simple', night_spar)

        elif estimation_method == 'multiple':
            # daytime
            if day_bot == 'default': self._set_burba_coeffs('day_bot', 'multiple', (2.8, -0.0681, 0.0021, -0.334))
            elif day_bot == 'keep': pass
            else: self._set_burba_coeffs('day_bot', 'multiple', day_bot)

            if day_top == 'default': self._set_burba_coeffs('day_top', 'multiple', (-0.1, -0.0044, 0.011, -0.022))
            elif day_top == 'keep': pass
            else: self._set_burba_coeffs('day_top', 'multiple', day_top)
            
            if day_spar == 'default': self._set_burba_coeffs('day_spar', 'multiple', (0.3, -0.0007, 0.0006, -0.044))
            elif day_spar == 'keep': pass
            else: self._set_burba_coeffs('day_spar', 'multiple', day_spar)

            # nighttime
            if night_bot == 'default': self._set_burba_coeffs('night_bot', 'multiple', (0.5, -0.1160, 0.0087, -0.206))
            elif night_bot == 'keep': pass
            else: self._set_burba_coeffs('night_bot', 'multiple', night_bot)

            if night_top == 'default': self._set_burba_coeffs('night_top', 'multiple', (-1.7, -0.0160, 0.0051, -0.029))
            elif night_top == 'keep': pass
            else: self._set_burba_coeffs('night_top', 'multiple', night_top)
            
            if night_spar == 'default': self._set_burba_coeffs('night_spar', 'multiple', (-2.1, -0.0200, 0.0070, 0.026))
            elif night_spar == 'keep': pass
            else: self._set_burba_coeffs('night_spar', 'multiple', night_spar)

    def to_eddypro(self, ini_file: str | PathLike[str], out_path: str | PathLike[str] | Literal['keep'] = 'keep'):
        """write to a .eddypro file.
        ini_file: the file name to write to
        out_path: the path for eddypro to output results to. If 'keep' (default), use the outpath already in the config file
        """
        self.set(section='Project', option='file_name', value=str(ini_file))
        if out_path != 'keep':
            self.set(section='Project', option='out_path', value=str(out_path))
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
    ) -> None:
        """
        writes to a set of .eddypro files split up into bite-size time chunks.
         .eddypro files, each handling a separate time chunk.
        all .eddypro files will be identical except in their project IDs, file names, and start/end dates.
        
        Note that some processing methods are not compatible "out-of-the-box" with paralle processing: some methods like the planar fit correction and ensemble spectral corrections will need the results from a previous, longer-term eddypro run to function effectively.

        ini_dir: the directory to output configured .eddypro files to. Does not have to exist.
        out_path: the path to direct eddypro to write results to.
        metadata_fn: path to a static .metadata file for this project. Must be provided if file_duration is None.
        num_workers: the number of parallel processes to configure. If None (default), then processing is split up according to the number of available processors on the machine minus 1.
        file_duration: how many minutes long each file is (NOT the averaging interval). If None (Default), then that information will be gleaned from the metadata file.
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
        start = str(datetime.datetime.strptime(
            f"{self.get(section='Project', option='pr_start_date')} {self.get(section='Project', option='pr_start_time')}", 
            r'%Y-%m-%d %H:%M'
        ))
        end = str(datetime.datetime.strptime(
            f"{self.get(section='Project', option='pr_end_date')} {self.get(section='Project', option='pr_end_time')}" , 
            r'%Y-%m-%d %H:%M'
        ))

        n_files = len(date_range(start, end, freq=f'{file_duration}min'))
        job_size = ceil(file_duration*n_files/num_workers)
        job_size = f'{int(ceil(job_size/file_duration)*file_duration)}min'

        job_starts = date_range('2020-06-21 00:00', '2020-07-22 00:00', freq=job_size)
        job_ends = job_starts + Timedelta(job_size) - Timedelta(file_duration)  # dates are inclusive, so subtract 30min for file duration
        job_start_dates = job_starts.strftime(date_format=r'%Y-%m-%d')
        job_start_times = job_starts.strftime(date_format=r'%H:%M')
        job_end_dates = job_ends.strftime(date_format=r'%Y-%m-%d')
        job_end_times = job_ends.strftime(date_format=r'%H:%M')

        # give each project a unique id and file name
        project_ids = [f'worker{start}' for start in job_starts.strftime(date_format=r"%Y%m%d%H%M")]
        ini_fns = [ini_dir / f'{project_id}.eddypro' for project_id in project_ids]

        # save original settings
        old_file_name = self.get(section='Project', option='file_name')
        old_out_path = self.get(section='Project', option='out_path')
        pr_start_date = self.get(section='Project', option='pr_start_date')
        pr_end_date = self.get(section='Project', option='pr_end_date')
        pr_start_time = self.get(section='Project', option='pr_start_time')
        pr_end_time = self.get(section='Project', option='pr_end_time')
        project_id = self.get(section='Project', option='project_id')


        # write new files
        if not os.path.isdir(Path(ini_dir)):
            Path.mkdir(Path(ini_dir))
        for i, fn in enumerate(ini_fns):
            self.set(section='Project', option='out_path', value=str(out_path))
            self.set(section='Project', option='file_name', value=str(fn))
            self.set(section='Project', option='pr_start_date', value=str(job_start_dates[i]))
            self.set(section='Project', option='pr_end_date', value=str(job_end_dates[i]))
            self.set(section='Project', option='pr_start_time', value=str(job_start_times[i]))
            self.set(section='Project', option='pr_end_time', value=str(job_end_times[i]))
            self.set(section='Project', option='project_id', value=str(project_ids[i]))

            with open(fn, 'w') as configfile:
                configfile.write(';EDDYPRO_PROCESSING\n')  # header line
                self.write(fp=configfile, space_around_delimiters=False)
        
        # revert to original
        self.set(section='Project', option='file_name', value=old_file_name)
        self.set(section='Project', option='out_path', value=old_out_path)
        self.set(section='Project', option='pr_start_date', value=pr_start_date)
        self.set(section='Project', option='pr_end_date', value=pr_end_date)
        self.set(section='Project', option='pr_start_time', value=pr_start_time)
        self.set(section='Project', option='pr_end_time', value=pr_end_time)
        self.set(section='Project', option='project_id', value=project_id)

        return
    
    def to_pandas(self) -> DataFrame:
        """convert current ini state to a pandas dataframe"""
        lines = []
        for section in self.sections():
            for option, value, in self[section].items():
                lines.append([section, option, value])
        df = DataFrame(lines, columns=['Section', 'Option', 'Value'])
        df = df.sort_values(['Section', 'Option'])
        df['Name'] = Path(self.get(section='Project', option='file_name')).stem

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
            new = self.__new__(cls)
            new.__init__(tmp.name)
            tmp.close()

            os.remove(tmp.name)
        except:
            tmp.close()
            os.remove(tmp.name)
            raise

        new._start_set = self._start_set
        new._end_set = self._end_set

        return new
    
    def __copy__(self):
        return self.copy()

if __name__ == '__main__':
    base = eddypro_ConfigParser('/Users/alex/Documents/Work/UWyo/Research/Flux Pipeline Project/Eddypro-ec-testing/investigate_eddypro/ini/base.eddypro')
    copy(base)

