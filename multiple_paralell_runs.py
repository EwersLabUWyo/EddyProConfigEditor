## python script to create multiple separate eddypro configurations using different settings, and to then run all of those in parallel
from pathlib import Path

from eddypro_config_editor import eddypro_ConfigParser

if __name__ == '__main__':
    environment = Path('/Users/alex/Documents/Work/UWyo/Research/Flux Pipeline Project/Eddypro-ec-testing/eddypro_compare_parallel')
    ini_dir = environment / 'ini'
    reference_ini = Path('/Users/alex/Documents/Work/UWyo/Research/Flux Pipeline Project/Eddypro-ec-testing/eddypro_compare_parallel/bin/ini/base.eddypro')
    base = eddypro_ConfigParser(reference_ini=reference_ini)
    print(base.to_pandas())