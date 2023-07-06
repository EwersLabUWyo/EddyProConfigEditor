import configparser
from pathlib import Path

import numpy as np
import pandas as pd

eddypro_dir = Path('.')
# read in ini file
ini_fn = eddypro_dir/'ini/template.eddypro'
ini = configparser.ConfigParser()
ini.read(ini_fn)

# break up processing into multiple separate jobs
# from the template file, here are the start/end dates:
# pr_end_date=2020-06-27
# pr_end_time=23:30
# pr_start_date=2020-06-26
# pr_start_time=00:00
job_size = '6H'
starts = pd.date_range('2020-06-26 00:00', '2020-06-27 23:30', freq=job_size)
ends = starts + pd.Timedelta(job_size) - pd.Timedelta('30m')  # dates are inclusive, so subtract 30min for file duration
start_dates = starts.strftime(date_format=r'%Y-%m-%d')
start_times = starts.strftime(date_format=r'%H:%M')
end_dates = ends.strftime(date_format=r'%Y-%m-%d')
end_times = ends.strftime(date_format=r'%H:%M')

# give each project a unique id
project_ids = [f'worker_{start}' for start in starts.strftime(date_format=r"%Y%m%d%H%M")]
# where to save the file to
ini_fns = [(eddypro_dir/f'ini/{project_id}.eddypro') for project_id in project_ids]
# where to save outputs to
out_path = eddypro_dir/'output'

# modify the ini files
for i, fn in enumerate(ini_fns):
    # empty ex_file I think?
    ini.set(section='FluxCorrection_SpectralAnalysis_General', option='ex_file', value='')
    ini.set(section='Project', option='file_name', value=fn.__str__())
    ini.set(section='Project', option='last_change_date', value='')
    ini.set(section='Project', option='pr_start_date', value=start_dates[i])
    ini.set(section='Project', option='pr_end_date', value=end_dates[i])
    ini.set(section='Project', option='pr_start_time', value=start_times[i])
    ini.set(section='Project', option='pr_end_time', value=end_times[i])
    ini.set(section='Project', option='project_id', value=project_ids[i])
    ini.set(section='Project', option='proj_file', value=(eddypro_dir/'ini/lcreek.metadata').__str__())
    # I guess set magnetic declination date to the processing date?
    ini.set(section='RawProcess_General', option='dec_date', value=start_dates[i]) 
    ini.set(section='Project', option='out_path', value=out_path.__str__())
    ini.set(section='RawProcess_General', option='data_path', value=(eddypro_dir/'raw_data').__str__())
    ini.set(section='Project', option='biom_dir', value=(eddypro_dir/'biomet').__str__())

    # write the .ini file
    with open(fn, 'w') as configfile:
        configfile.write(';EDDYPRO_PROCESSING\n')  # header line
        ini.write(fp=configfile, space_around_delimiters=False)