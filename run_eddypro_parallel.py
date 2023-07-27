"""
script to run a set of parallel eddypro runs
Author: Alex Fox
Created: July 2023
"""

import multiprocessing
import subprocess
from pathlib import Path
import platform

from collections.abc import Sequence, Generator
from os import PathLike
from os.path import isfile, isdir
from typing import Literal

def rm_tree(pth: str | PathLike[str], warn: bool = True) -> None:
    """recursively remove the provided path.
    set warn=True when calling this function to enable interactive warnings."""   
    if warn:
        answer = input(f'\nDeleting {str(pth)}.\n"Y" to confirm... ')
        assert answer.upper() == 'Y', f'Answered {answer}. Aborting!'
    pth = Path(pth)
    for child in pth.glob('*'):
        if child.is_file():
            child.unlink()
        else:
            rm_tree(child, warn=False)
    return 

def call(args: tuple[str, str, str | PathLike[str]]) -> None:
    '''call eddypro_rp and eddypro_fcc once, and write output to .stdout file'''
    rp_call, fcc_call, stdout_file = args
    with open(stdout_file, 'wb') as f:
        try:
            stdout = subprocess.check_output(rp_call, shell=True)
            print(f'{stdout_file.stem} completed eddypro_rp gracefully')
        except subprocess.CalledProcessError as e:
            print(f'{stdout_file.stem} completed eddypro_rp with errors')
            stdout = e.output
        f.write(stdout)

        try:
            stdout = subprocess.check_output(fcc_call, shell=True)
            print(f'{stdout_file.stem} completed eddypro_fcc gracefully')
        except subprocess.CalledProcessError as e:
            stdout = e.output
            print(f'{stdout_file.stem} completed eddypro_fcc with errors')
        f.write(stdout)
    
    return

def make_eddypro_calls(
        environment: str | PathLike[str], 
        clean_children: bool = False,
        PROJ_FILES: None | str | PathLike[str] | Sequence[str | PathLike[str]] | Generator[str | PathLike[str]] = None, 
        eddypro_rp: None | str | PathLike[str] = None, 
        eddypro_fcc: None | str | PathLike[str] = None,
        system: None | Literal['win', 'linux', 'mac'] = None,
    ) -> tuple[list[str], list[str], list[Path]]:
    '''construct construct a set of shell commands to call eddypro_rp and eddypro_fcc
    
    environment: the --environment argument for eddypro, ie. the working directory.
    clean_outputs_dirs: If False (default), do nothing. If True, then wipe the output, stdout, and tmp directories located in the environment directory. Note that if output, stdout, and tmp directories are not found, an error will be thrown.
    PROJ_FILES: If None (default), then assume that project files (.eddypro files)) are located in the ini directory of the environment directory. If provided, use the .eddypro files provided.
    eddypro_rp: If None (default), then assume that the eddypro_rp executable is located in the bin directory of the environment directory. Otherwise, provide the path to the eddypro_rp executable.
    eddypro_fcc: If None (default), then assume that the eddypro_fcc executable is located in the bin directory of the environment directory. Otherwise, provide the path to the eddypro_fcc executable.
    system: If None, auto-detect the operating system. Noe that this hasn't been tested on Cygwin. Othewrise, use the provided operating system.
    '''

    environment = Path(environment)

    if clean_children:
        rm_tree(environment / 'output', warn=False)
        rm_tree(environment / 'stdout', warn=False)
        rm_tree(environment / 'tmp', warn=False)
    if not isdir(environment / 'output'): (environment / 'output').mkdir()
    if not isdir(environment / 'stdout'): (environment / 'stdout').mkdir()
    if not isdir(environment / 'tmp'): (environment / 'tmp').mkdir()
    

    # assume project files are located in the ini directory if not provided
    if PROJ_FILES is None:
        PROJ_FILES = list(environment.glob('ini/*.eddypro'))
        assert len(PROJ_FILES), f'No .eddypro files found in the {environment / "ini"} directory. Did you forget to provide a file or set of .eddypro files to use?'
    # coerce whatever .eddypro files were found into a list
    if isinstance(PROJ_FILES, str) or isinstance(PROJ_FILES, PathLike):
        PROJ_FILES = list(Path(str))
    if isinstance(PROJ_FILES, Generator):
        PROJ_FILES = [Path(PROJ_FILE) for PROJ_FILE in PROJ_FILES]
    for PROJ_FILE in PROJ_FILES:
        assert isfile(PROJ_FILE), f'.eddypro file {PROJ_FILE} not found.'

    # find the eddypro executable files
    if eddypro_rp is None:
        eddypro_rp = Path(environment / 'bin/eddypro_rp')
    assert isfile(eddypro_rp), f'Executable {eddypro_rp} not found. Did you Did you specify the location properly?'
    if eddypro_fcc is None:
        eddypro_fcc = Path(environment / 'bin/eddypro_fcc')
    assert isfile(eddypro_fcc), f'Executable {eddypro_fcc} not found. Did you Did you specify the location properly?'

    # find the right operating system
    if system is None:
        system_key = platform.system()
        system_dict = dict(Linux='linux', Darwin='mac', Windows='win')
        assert system_key in system_dict, f'System was found to be {system_key}, but must be one of {", ".join(system_dict.keys())}'
        system = system_dict[system_key]
    assert system in ['linux', 'mac', 'win'], f'System was given as {system}, but must be one of mac, linux, or win.'

    mode="desktop"
    
    # construct calls
    eddypro_rp_calls = [
        f'"{str(eddypro_rp)}" -s "{system}" -m "{mode}" -e "{str(environment)}" "{str(PROJ_FILE)}"'
        for PROJ_FILE in PROJ_FILES
    ]
    eddypro_fcc_calls = [
        f'"{str(eddypro_fcc)}" -s "{system}" -m "{mode}" -e "{str(environment)}" "{str(PROJ_FILE)}"'
        for PROJ_FILE in PROJ_FILES
    ]

    return eddypro_rp_calls, eddypro_fcc_calls, PROJ_FILES

if __name__ == '__main__':

    environment = Path('/Users/alex/Documents/Work/UWyo/Research/Flux Pipeline Project/Eddypro-ec-testing/eddypro')
    PROJ_FILES = environment.glob('ini/worker*.eddypro')
    eddypro_rp = environment / 'bin/eddypro_rp'
    eddypro_fcc = environment / 'bin/eddypro_fcc'

    eddypro_rp_calls, eddypro_fcc_calls, project_files = make_eddypro_calls(
        environment=environment,
        clean_children=True,
        PROJ_FILES=PROJ_FILES,
        eddypro_rp=eddypro_rp,
        eddypro_fcc=eddypro_fcc,
    )
    
    args = [(rp_call, fcc_call, f) for rp_call, fcc_call, f in zip(eddypro_rp_calls, eddypro_fcc_calls, stdout_files)]

    # multiprocessing
    processes = max(multiprocessing.cpu_count() - 2, 1)
    print(f'\nBeginning EddyPro Runs: {len(args)} runs on {processes} cores\n')
    with multiprocessing.Pool(processes) as p:
        p.map(call, args)