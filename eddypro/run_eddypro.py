from tempfile import TemporaryFile
from multiprocessing import Pool
import subprocess
from pathlib import Path

def call_rp_fcc(call):
    print(call)
    output = subprocess.check_output(call, shell=True)
    return output

def make_eddypro_calls():
    # location of eddypro rp and fcc
    eddypro_rp = Path('./bin/eddypro_rp').__str__()
    eddypro_fcc = Path('./bin/eddypro_fcc').__str__()
    # inputs to rp and fcc
    system="mac"
    mode="desktop"
    environment = Path('./').__str__()
    ini_dir = Path("./ini")
    project_files = [fn.__str__() for fn in ini_dir.glob("worker*.eddypro")]
    # construct call
    eddypro_rp_calls = [f'"{eddypro_rp}" -s "{system}" -m "{mode}" -e "{environment}" "{PROJ_FILE}"' for PROJ_FILE in project_files]
    eddypro_fcc_calls = [f'"{eddypro_fcc}" -s "{system}" -m "{mode}" -e "{environment}" "{PROJ_FILE}"' for PROJ_FILE in project_files]
    full_call_lst = [f'{rp_call} && {fcc_call}' for rp_call, fcc_call in zip(eddypro_rp_calls, eddypro_fcc_calls)]

    return full_call_lst
if __name__ == '__main__':
    full_call_lst = make_eddypro_calls()
    with Pool(4) as p: 
        outputs = p.map(call_rp_fcc, full_call_lst)
        for i, out in enumerate(outputs):
            with open(Path(f'./{i}.log'), 'w') as logfile:
                logfile.write(out)