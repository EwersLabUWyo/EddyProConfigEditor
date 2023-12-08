#!/bin/bash

# Usage: ./run_eddypro.sh $system $environment $PROJ_FILE $max_retries
# this bash script gets around the random Fortran runtime errors and eddypro fatal errors
# that are randomly thrown when running eddypro in parallel on the same dataset.
# it's a bit of a kludge fix, and works by running eddypro, then if it fails, waiting a random
# amount of time before trying again. Once a maximum number of retries is reached, the script gives
# up and accepts its fate.
# Author: Alexander Fox
# Copyright 2023
# License: GPL3
# Email: afox18@uwyo.edu

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
    # do this as a process substitution to avoid making another shell
    while IFS= read -r line; do
        echo "$line"
        if [[ "$line" =~ "runtime error" ]]; then
            found_rte=true
            break
        elif [[ "$line" =~ "Fatal error" ]]; then
            found_fe=true
            break
        fi   
    # 2>&1 captures both stderr and stdout
    done < <(eddypro_rp -s "$system" -m desktop -e "$environment" $PROJ_FILE 2>&1)
    # fatal error

    # retry if we get an error
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
    if [ "$n_runs" -gt "$max_retries" ]; then
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

# redo for eddypro_fcc
sleep $((RANDOM % 10 + 1))
n_runs=0
found_rte=false
found_fe=false
kill=false
while true; do
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
    else
        break
    fi

    if [ "$n_runs" -gt "$max_retries" ]; then
        echo
        echo ">>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<"
        echo ">>>>>Too many retries! exiting....<<<<<"
        echo ">>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<"
        kill=true
        break
    fi
done

if [ "$kill" = true ]; then
    exit 1
fi

exit 0