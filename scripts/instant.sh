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

echo "[$CLUSTER] Executing experiment #$experiment"
rm -rf $OUT_PATH/experiments/exp$experiment
mkdir -p $OUT_PATH/experiments/exp$experiment
cp $USER_PATH/params.yaml $OUT_PATH/experiments/exp$experiment/
touch $OUT_PATH/experiments/exp$experiment/exp$experiment.log
chmod +rwx $OUT_PATH/experiments/exp$experiment/exp$experiment.log

if python3 $USER_PATH/src/trainer.py \
--species human \
--chr 22 \
--params $OUT_PATH/experiments/exp$experiment/params.yaml \
--num $experiment \
--verbose True \
--evolution False;
then echo "[$CLUSTER] Success!"
else echo "[$CLUSTER] Fail!"; fi