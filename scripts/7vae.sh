#!/bin/bash
source ini.sh
for ARGUMENT in "$@" 
do
    KEY=$(echo $ARGUMENT | cut -f1 -d=)
    VALUE=$(echo $ARGUMENT | cut -f2 -d=)   
    case "$KEY" in
            cluster)           cluster=${VALUE} ;;
            start_experiment)  experiment=${VALUE} ;;
            *)   
    esac    
done

if [[ -z $cluster ]]; then cluster=$CLUSTER; fi
if [[ -z $experiment ]]; then echo "Missing start experiment number."; exit 1; fi

## Default settings:
python3 $USER_PATH/scripts/hypertuning.py \
    --ipath $USER_PATH/params.yaml    \
    --opath $USER_PATH/optparams.yaml

counter=0
for i in $(seq $experiment $(($experiment+7)) ); do

    python3 $USER_PATH/scripts/hypertuning.py \
        --ipath $USER_PATH/optparams.yaml    \
        --opath $USER_PATH/optparams.yaml \
        --only $counter
                
    rm -rf $OUT_PATH/experiments/exp$i
    mkdir -p $OUT_PATH/experiments/exp$i
    cp $USER_PATH/optparams.yaml $OUT_PATH/experiments/exp$i/
    mv $OUT_PATH/experiments/exp$i/optparams.yaml $OUT_PATH/experiments/exp$i/params.yaml
    touch $OUT_PATH/experiments/exp$i/exp$i.log
    chmod +rwx $OUT_PATH/experiments/exp$i/exp$i.log
    
    echo "Experiment: $i, Counter: $counter, CUDA DEVICE: $((7+1-$counter))"
    CUDA_VISIBLE_DEVICES=$((7+1-$counter)) python3 $USER_PATH/src/trainer.py \
    --params $OUT_PATH/experiments/exp$i/params.yaml \
    --num $i \
    --verbose True & 
   let counter++
done
