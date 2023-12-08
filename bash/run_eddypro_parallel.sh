#!/bin/bash

# Usage: ./run_eddypro.sh $system $environment $PROJ_FILE $max_retries

system=$1
environment=$2
PROJ_FILE=$3
max_retries=$4

sleep $((RANDOM % 10 + 1))
n_runs=0
found_rte=false
found_fe=false
kill=false
while true; do
    # run eddypro, parse output
    # stop parsing if we get a runtime error
    # stop parsing if we get a fatal error
    # do this as a process substitution to avoid making another shell, so that we can
    # use found_rte and found_fe elsewhere.
    while IFS= read -r line; do
        echo "$line"
        if [[ "$line" =~ "runtime error" ]]; then
            found_rte=true
            break
        elif [[ "$line" =~ "Fatal error" ]]; then
            found_fe=true
            break
        fi   
    done < <(eddypro_rp -s "$system" -m desktop -e "$environment" $PROJ_FILE 2>&1)
    # fatal error

    # runtime error means we need to retry.
    if [ "$found_rte" = true ]; then
        ((n_runs++))
        sleep $((RANDOM % 10 + 1))
        echo
        echo ">>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
        echo ">>>>>Encountered Fortran runtime error. Retrying<<<<<"
        echo ">>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
    elif [ "$found_fe" = true ]; then
        ((n_runs++))
        sleep $((RANDOM % 10 + 1))
        echo
        echo ">>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
        echo ">>>>>Encountered Fatal Error. Retrying<<<<<"
        echo ">>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
    # no runtime or fatal error: move on to fcc
    else
        break
    fi

    # too many retries
    if [ "$n_runs" -ge "$max_retries" ]; then
        echo
        echo ">>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<"
        echo ">>>>>Too many retries! exiting....<<<<<"
        echo ">>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<"
        kill=true
        break
    fi
done

# exit if fatal error
if [ "$kill" = true ]; then
    exit 1
fi

sleep $((RANDOM % 10 + 1))
n_runs=0
found_rte=false
found_fe=false
kill=false
while true; do
    # run eddypro, parse output
    # stop parsing if we get a runtime error
    # stop parsing if we get a fatal error
    # do this as a process substitution to avoid making another shell, so that we can
    # use found_rte and found_fe elsewhere.
    while IFS= read -r line; do
        echo "$line"
        if [[ "$line" =~ "runtime error" ]]; then
            found_rte=true
            break
        elif [[ "$line" =~ "Fatal error" ]]; then
            found_fe=true
            break
        fi   
    done < <(eddypro_fcc -s "$system" -m desktop -e "$environment" $PROJ_FILE 2>&1)
    # fatal error

    # runtime error means we need to retry.
    if [ "$found_rte" = true ]; then
        ((n_runs++))
        sleep $((RANDOM % 10 + 1))
        echo
        echo ">>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
        echo ">>>>>Encountered Fortran runtime error. Retrying<<<<<"
        echo ">>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
    elif [ "$found_fe" = true ]; then
        ((n_runs++))
        sleep $((RANDOM % 10 + 1))
        echo
        echo ">>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
        echo ">>>>>Encountered Fatal Error. Retrying<<<<<"
        echo ">>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
    # no runtime or fatal error: move on to fcc
    else
        break
    fi

    # too many retries
    if [ "$n_runs" -ge "$max_retries" ]; then
        echo
        echo ">>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<"
        echo ">>>>>Too many retries! exiting....<<<<<"
        echo ">>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<"
        kill=true
        break
    fi
done

# exit if fatal error
if [ "$kill" = true ]; then
    exit 1
fi

exit 0