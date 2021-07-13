#!/bin/bash
source ini.sh
for ARGUMENT in "$@" 
do
    KEY=$(echo $ARGUMENT | cut -f1 -d=)
    VALUE=$(echo $ARGUMENT | cut -f2 -d=)   
    case "$KEY" in
            cluster)     cluster=${VALUE} ;;
            experiment)  experiment=${VALUE} ;; 
            channel)     channel=${VALUE} ;;
            *)   
    esac    
done

if [[ -z $cluster ]]; then cluster=$CLUSTER; fi
if [[ -z $experiment ]]; then echo "[$CLUSTER] Missing experiment number."; exit 1; fi
if [[ -z $channel ]]; then 
    echo "[$CLUSTER] Watching default channel 2."; 
    channel="err"
fi

echo "[$CLUSTER] Watch request for experiment #$experiment"
STATE=$(squeue | grep AEgen$experiment | awk '{print $5}')
if [[ -z $STATE ]] || [ "$STATE" != "R" ]; then
    echo "[$CLUSTER] Experiment #$experiment is not running."
else
    if [ "$channel" == "1" ] || [ "$channel" == "log" ];  then
        less +F $OUT_PATH/experiments/exp$experiment/exp$experiment.log
    elif [ "$channel" == "2" ] || [ "$channel" == "err" ]; then
        less +F $OUT_PATH/experiments/exp$experiment/exp$experiment.err
    else echo "[$CLUSTER] Wrong channel."; fi
fi
