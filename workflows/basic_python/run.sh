
eddypro_rp="/Users/alex/Documents/Work/UWyo/Research/Flux Pipeline Project/Eddypro-ec-testing/eddypro/eddypro_rp"
eddypro_fcc="/Users/alex/Documents/Work/UWyo/Research/Flux Pipeline Project/Eddypro-ec-testing/eddypro/eddypro_fcc"
environment="/Users/alex/Documents/Work/UWyo/Research/Flux Pipeline Project/Eddypro-ec-testing/workflows/basic_python"
script="/Users/alex/Documents/Work/UWyo/Research/Flux Pipeline Project/Eddypro-ec-testing/workflows/basic_python/basic_workflow.py"

# 1. generate a re-useable planar fit config file by running eddypro in manual planar fit mode
# generating the reusable file takes about 30 seconds per day (0.6s per HH or 3hr/year) of data
# processing the remaining fluxes takes around 40 seconds per day (0.8s per HH or 4.1hr/year) of data
# so not having to perform the planar fit calculation multiple times can save 3 hr of CPU time per year of data processed
# python "${script}"
# first eddypro call: generate planar fit data
PROJ_FILE="/Users/alex/Documents/Work/UWyo/Research/Flux Pipeline Project/Eddypro-ec-testing/workflows/basic_python/ini/pf_base.eddypro"
"${eddypro_rp}" \
    -s mac \
    -e "${environment}" \
    "${PROJ_FILE}"
"${eddypro_fcc}" \
    -s mac \
    -e "${environment}" \
    "${PROJ_FILE}"

# 2. run eddypro using the planar fit config file.
# python "${script}" --pf_file
for NEW_PROJ_FILE in \
    "/Users/alex/Documents/Work/UWyo/Research/Flux Pipeline Project/Eddypro-ec-testing/workflows/basic_python/ini/pf_covariance_maximization.eddypro" \
    "/Users/alex/Documents/Work/UWyo/Research/Flux Pipeline Project/Eddypro-ec-testing/workflows/basic_python/ini/pf_covariance_maximization_with_default.eddypro" \
    "/Users/alex/Documents/Work/UWyo/Research/Flux Pipeline Project/Eddypro-ec-testing/workflows/basic_python/ini/pf_none.eddypro" 
do
    "${eddypro_rp}" \
        -s mac \
        -e "${environment}" \
        "${NEW_PROJ_FILE}"
    "${eddypro_fcc}" \
        -s mac \
        -e "${environment}" \
        "${NEW_PROJ_FILE}"
done
