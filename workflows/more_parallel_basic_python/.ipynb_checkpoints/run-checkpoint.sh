# if working on beartooth:
# module load arcc/1.0 gcc/12.2.0 eddyproengine/7.0.9
system=linux  # change before running on mac or win

# set up directory struct
environment="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
echo $environment
mkdir "${environment}/output" -p
mkdir "${environment}/tmp" -p

# 1. generate a re-useable planar fit config file by running eddypro in manual planar fit mode
# generating the reusable file takes about 30 seconds per day (0.6s per HH or 3hr/year) of data
# processing the remaining fluxes takes around 40 seconds per day (0.8s per HH or 4.1hr/year) of data
# so not having to perform the planar fit calculation multiple times can save 3 hr of CPU time per year of data processed
# from my BOTE calculation, this script should run in ~22 minutes.
# if we were to perform the planar fit calculation for every eddypro run, it would take ~25 minutes instead.
# as the number of "duplicate" runs increases, this workflow quickly asymptotes to run in ~60% of the time.
python "${environment}/basic_workflow.py"
PROJ_FILE="${environment}/ini/pf_base.eddypro"
eddypro_rp \
     -s $system \
     -e "${environment}" \
     "${PROJ_FILE}"
 eddypro_fcc \
     -s $system \
     -e "${environment}" \
     "${PROJ_FILE}"

# 2. run eddypro using the planar fit config file.
python "${environment}/basic_workflow.py" --pf_file
for NEW_PROJ_FILE in \
    "${environment}/ini/pf_covariance_maximization.eddypro" \
    "${environment}/ini/pf_covariance_maximization_with_default.eddypro" \
    "${environment}/ini/pf_none.eddypro" 
do
    eddypro_rp \
        -s $system \
        -e "${environment}" \
        "${NEW_PROJ_FILE}"
    eddypro_fcc \
        -s $system \
        -e "${environment}" \
        "${NEW_PROJ_FILE}"
done
