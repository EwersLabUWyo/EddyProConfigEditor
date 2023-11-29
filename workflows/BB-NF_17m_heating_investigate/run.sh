#!/bin/bash

#SBATCH --account=bbtrees
#SBATCH --time=24:00:00
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=40
#SBATCH --cpus-per-task=1
#SBATCH --array=0-183
#SBATCH --mem=64G
#SBATCH -o stdout/%a.out # STDOUT
# if working on beartooth:
module load arcc/1.0 gcc/12.2.0 eddyproengine/7.0.9

system=linux  # change before running on mac or win
# set up directory struct
environment="/project/eddycovworkflow/afox18/Platinum-EddyPro7/workflows/BB-NF_17m_heating_investigate"
ini_dir="${environment}/ini/BB-NF_17m_Parallel"
echo $environment
mkdir "${environment}/output" -p
mkdir "${environment}/tmp" -p
filelist=($(ls "${ini_dir}"))

echo "${ini_dir}/${filelist[$SLURM_ARRAY_TASK_ID]}"
eddypro_rp \
    -s $system \
    -e "${environment}" \
    "${ini_dir}/${filelist[$SLURM_ARRAY_TASK_ID]}"
eddypro_fcc \
    -s $system \
    -e "${environment}" \
    "${ini_dir}/${filelist[$SLURM_ARRAY_TASK_ID]}"