# this generates a series of eddypro config files for the basic workflow
import argparse
from pathlib import Path
import sys

wd = Path(__file__).parent
sys.path.append(str(wd/'../..'))
from python.eddyproconfigeditor import EddyproConfigEditor

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pf_file', action='store_true')
    
    args = parser.parse_args()

    ini_dir = wd/'ini'
    base_fn = ini_dir / 'base.eddypro'
    out_path = wd/'output'

    # read in default .eddypro config and change some system-specific settings for beartooth
    base = EddyproConfigEditor(base_fn)
    base.Proj.set_metadata(static=ini_dir / 'lcreek.metadata')
    base.Proj.set_biomet(mode='dir', path='/home/afox18/eddycovworkflow/InputData/LostCreek/biomet', extension='data', subfolders=False)
    base.Basic.set_raw_data(path='/home/afox18/eddycovworkflow/InputData/LostCreek/raw_data', fmt='LostCreek_ts_data_yyyy_mm_dd_HHMM.dat', subfolders=True)
    
    # set EP to perform a planar fit correction in "manual mode"
    # running in manual mode will prompt EP to output a planar fit config file
    # that we can use in subsequent runs on this dataset
    _id = 'manual'
    pf_base = base.copy()
    base.Adv.Out.set_spectral_output(full_cospectra='none')
    base.Adv.Out.set_chain_of_custody(
        unprocessed='stats',
        despiked='stats',
        tilt_corrected='stats',
        timelag_corrected='stats',)
    pf_base.Basic.set_project_date_range('2020-06-21 00:00', '2020-06-28 00:00')
    pf_base.Adv.Proc.set_axis_rotations_for_tilt_correction(
        method='planar_fit',
        configure_planar_fit_settings_kwargs=pf_base.Adv.Proc._configure_planar_fit_settings(
            w_max=2, 
            u_min=0.1, 
            num_per_sector_min=5, 
            sectors=[(False, 360)], 
            start='project',
            end='project',
            return_inputs=True))
    pf_base.Proj.set_project_name('basic workflow')
    pf_base.Basic.set_output_id(_id)
    pf_base.to_eddypro(ini_file=ini_dir / 'pf_base.eddypro', out_path=out_path)

    # now set EP to instead load the planar fit configuration from the file generated in the previous run. 
    # using planar fit coefficients loaded from a ~halves the runtime of eddypro, and allows us to 
    # perform a sensitivity analysis much faster than we could otherwise.
    # so here, we'll perform a sensitivity analysis.
    if args.pf_file:
        pf_new = pf_base.copy()

        pf_file = sorted(out_path.glob(f'eddypro_{_id}_planar_fit*.txt'))[-1]
        pf_new.Adv.Proc.set_axis_rotations_for_tilt_correction(
            method='planar_fit', pf_file=pf_file
        )

        # set up a variety of new runs using the planar fit config file
        for tl_method in ['none', 'covariance_maximization_with_default', 'covariance_maximization']:
            pf_new.Adv.Proc.set_timelag_compensations(method=tl_method)
            pf_new.Basic.set_output_id(output_id=tl_method.replace('_', '-'))
            pf_new.to_eddypro(ini_file=ini_dir / (f'pf_{tl_method}'), out_path=out_path)











