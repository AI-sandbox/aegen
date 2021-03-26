#!/bin/bash
source ini.sh
for ARGUMENT in "$@" 
do
    KEY=$(echo $ARGUMENT | cut -f1 -d=)
    VALUE=$(echo $ARGUMENT | cut -f2 -d=)   
    case "$KEY" in
            cluster)     cluster=${VALUE} ;;
            experiment) experiment=${VALUE} ;;     
            *)   
    esac    
done

if [[ -z $cluster ]]; then cluster=$CLUSTER; fi
if [[ -z $experiment ]]; then echo "Missing experiment number."; exit 1; fi

echo "[$CLUSTER] Watch request for experiment #$experiment"
STATE=$(squeue | grep VAEgen$experiment | awk '{print $5}')
if [[ -z $STATE ]] || [ "$STATE" != "R" ]; then
    echo "[$CLUSTER] Experiment #$experiment is not running."
else less +F $OUT_PATH/experiments/exp$experiment/exp$experiment.log; fi
