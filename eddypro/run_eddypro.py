import multiprocessing
import subprocess
from pathlib import Path

def rm_tree(pth, root=True):
            if root:
                answer = input(f'\nDeleting {str(pth)}.\n"Y" to confirm... ')
                assert answer.upper() == 'Y', f'Answered {answer}. Aborting!'
            pth = Path(pth)
            for child in pth.glob('*'):
                if child.is_file():
                    child.unlink()
                else:
                    rm_tree(child, root=False)

def call(args):
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

def make_eddypro_calls(eddypro_dir):
    # location of eddypro rp and fcc
    eddypro_rp = eddypro_dir/'bin/eddypro_rp'.__str__()
    eddypro_fcc = eddypro_dir/'bin/eddypro_fcc'.__str__()
    # inputs to rp and fcc
    system="mac"
    mode="desktop"
    environment = eddypro_dir.__str__()
    ini_dir = eddypro_dir/'ini'
    project_files = [fn.__str__() for fn in ini_dir.glob("worker*.eddypro")]
    # construct call
    eddypro_rp_calls = [
        f'"{eddypro_rp}" -s "{system}" -m "{mode}" -e "{environment}" "{PROJ_FILE}"'
        for PROJ_FILE in project_files
    ]
    eddypro_fcc_calls = [
        f'"{eddypro_fcc}" -s "{system}" -m "{mode}" -e "{environment}" "{PROJ_FILE}"'
        for PROJ_FILE in project_files
    ]

    return eddypro_rp_calls, eddypro_fcc_calls, project_files
if __name__ == '__main__':
    eddypro_dir = Path('/Users/alex/Documents/Work/UWyo/Research/Flux Pipeline Project/Eddypro-ec-testing/eddypro')
    stdout_dir = Path('/Users/alex/Documents/Work/UWyo/Research/Flux Pipeline Project/Eddypro-ec-testing/eddypro/stdout')
    eddypro_rp_calls, eddypro_fcc_calls, project_files = make_eddypro_calls(eddypro_dir)
    
    clear_output = True
    if clear_output:
        rm_tree(eddypro_dir / 'output')
        Path.mkdir(eddypro_dir / 'output', exist_ok=True)
        rm_tree(eddypro_dir / 'stdout')
        Path.mkdir(eddypro_dir / 'stdout', exist_ok=True)
        rm_tree(eddypro_dir / 'tmp')
        Path.mkdir(eddypro_dir / 'tmp', exist_ok=True)

    stdout_files = [stdout_dir / (Path(f).stem + '.stdout') for f in project_files]
    args = [(rp_call, fcc_call, f) for rp_call, fcc_call, f in zip(eddypro_rp_calls, eddypro_fcc_calls, stdout_files)]

    processes = max(multiprocessing.cpu_count() - 2, 1)
    print(f'\nBeginning EddyPro Runs: {len(args)} runs on {processes} cores\n')
    with multiprocessing.Pool(processes) as p:
        p.map(call, args)