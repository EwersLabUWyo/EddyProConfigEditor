environment="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
for D in \
    "${environment}/ini/covariance_maximization" \
    "${environment}/ini/covariance_maximization_with_default" \
    "${environment}/ini/none"
do
    for F in "$(ls "${D}" | grep .eddypro)"
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
done