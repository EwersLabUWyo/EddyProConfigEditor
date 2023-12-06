#!/bin/bash

#SBATCH --account=bbtrees
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --array=0-3539
#SBATCH --mem=500G
#SBATCH -o stdout/%a.out # STDOUT
# if working on beartooth:

module load arcc/1.0 gcc/12.2.0 eddyproengine/7.0.9

system=linux  # change before running on mac or win
# set up directory struct
environment="/project/eddycovworkflow/afox18/Platinum-EddyPro7/workflows/BB-NF_17m_Cal"
ini_dir="${environment}/ini/BB-NF_17m_Parallel"
echo $environment
mkdir "${environment}/output" -p
mkdir "${environment}/tmp" -p
filelist=($(ls "${ini_dir}"))

# find the right file
echo "${ini_dir}/${filelist[$SLURM_ARRAY_TASK_ID]}"

max_attempts=5
attempts=0

while [ $attempts -lt $max_attempts ]; do
    # Run the program and collect the output into the "res" variable
    rp_res=$(eddypro_rp \
        -s $system \
        -e "${environment}" \
        "${ini_dir}/${filelist[$SLURM_ARRAY_TASK_ID]}" 2>&1)
    echo "$rp_res"

    fcc_res=$(eddypro_fcc \
        -s $system \
        -e "${environment}" \
        "${ini_dir}/${filelist[$SLURM_ARRAY_TASK_ID]}" 2>&1)
    echo "$fcc_res"

    # Check if the output contains the specified error message
    # Check for Fortran runtime error
    if [[ "$res" =~ "Fortran runtime error: Index" ]]; then
        # Increment the run count
        echo; echo
        echo "Detected Fortran runtime error. Retrying"
        echo; echo
        ((n_runs++))
    elif [[ "$res" =~ "Fatal error" ]]; then
        # Exit with status 1 if Fatal Error is found
        exit 1
    else
        # No error, exit the loop
        break
    fi
done

# Check if the maximum number of attempts has been reached
if [ $attempts -ge $max_attempts ]; then
    echo; echo;
    echo "Error: Maximum attempts reached. Exiting with status 1."
    exit 1
fi