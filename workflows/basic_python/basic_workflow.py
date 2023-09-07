from pathlib import Path
import sys

wd = Path(__file__).parent
sys.path.append(str(wd/'../..'))
from python.eddyproconfigeditor import EddyproConfigEditor

print('running')


if __name__ == '__main__':
    print('running main')
    ini_dir = wd/'ini'
    base_fn = ini_dir / 'base.eddypro'
    out_path = wd/'output'

    base = EddyproConfigEditor(base_fn)
    
    # run 1: planar fit, basic settings.
    # change some output settings to reduce size of outputs
    pf_base = base.copy()
    pf_settings = dict(w_max=2, u_min=0.1, num_per_sector_min=5)
    pf_base.Adv.Proc.set_axis_rotations_for_tilt_correction(
        method='planar_fit',
        configure_planar_fit_settings_kwargs=pf_settings)
    # speed up processing/reduce disk space needed
    base.Adv.Out.set_spectral_output(full_cospectra='none')
    base.Adv.Out.set_chain_of_custody(
        unprocessed='stats',
        despiked='stats',
        tilt_corrected='stats',
        timelag_corrected='stats',)
    pf_base.to_eddypro(ini_file=ini_dir / 'pf_base.eddypro', out_path=out_path)









