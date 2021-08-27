#!/bin/bash
## Never spend 6 minutes doing something by hand (change yaml and execute a run) when 
## you can spend 6 hours failing to automate it (change yaml and execute runs in a row).
source ini.sh
for ARGUMENT in "$@" 
do
    KEY=$(echo $ARGUMENT | cut -f1 -d=)
    VALUE=$(echo $ARGUMENT | cut -f2 -d=)   
    case "$KEY" in
            cluster)           cluster=${VALUE} ;;
            start_experiment)  experiment=${VALUE} ;;
            parameter)         parameter=${VALUE} ;;
            *)   
    esac    
done

if [[ -z $cluster ]]; then cluster=$CLUSTER; fi
if [[ -z $experiment ]]; then echo "Missing start experiment number."; exit 1; fi
if [[ -z $parameter ]]; then echo "Choose a hyperparameter to test."; exit 1; fi

## Default settings:
python3 $USER_PATH/scripts/hypertuning.py \
        --ipath $USER_PATH/params.yaml    \
        --opath $USER_PATH/optparams.yaml \
        --lr 0.025 \
        --optimizer Adam \
        --weight_decay 0.01 \
        --heads 1 \
        --vqbeta 2 \
        --bottleneck 512

if [ "$parameter" == "lr" ]; then
    device=3
    for lr in 0.1 0.05 0.01 0.001
    do 
        python3 $USER_PATH/scripts/hypertuning.py \
                --ipath $USER_PATH/optparams.yaml    \
                --opath $USER_PATH/optparams.yaml \
                --lr $lr 

        echo "[$CLUSTER] Executing experiment #$experiment"
        rm -rf $OUT_PATH/experiments/exp$experiment
        mkdir -p $OUT_PATH/experiments/exp$experiment
        cp $USER_PATH/optparams.yaml $OUT_PATH/experiments/exp$experiment/
        mv $OUT_PATH/experiments/exp$experiment/optparams.yaml $OUT_PATH/experiments/exp$experiment/params.yaml
        touch $OUT_PATH/experiments/exp$experiment/exp$experiment.log
        chmod +rwx $OUT_PATH/experiments/exp$experiment/exp$experiment.log

        CUDA_VISIBLE_DEVICES=$device python3 $USER_PATH/src/trainer.py \
        --params $OUT_PATH/experiments/exp$experiment/params.yaml \
        --num $experiment \
        --verbose True &
        
        let experiment++
        let device++
    done
elif [ "$parameter" == "optimizer" ]; then
    device=4
    for opt in RAdam QHAdam Yogi DiffGrad
    do 
        python3 $USER_PATH/scripts/hypertuning.py \
                --ipath $USER_PATH/optparams.yaml    \
                --opath $USER_PATH/optparams.yaml \
                --lr 0.1 \
                --weight_decay 0.00001 \
                --optimizer $opt 

        echo "[$CLUSTER] Executing experiment #$experiment"
        rm -rf $OUT_PATH/experiments/exp$experiment
        mkdir -p $OUT_PATH/experiments/exp$experiment
        cp $USER_PATH/optparams.yaml $OUT_PATH/experiments/exp$experiment/
        mv $OUT_PATH/experiments/exp$experiment/optparams.yaml $OUT_PATH/experiments/exp$experiment/params.yaml
        touch $OUT_PATH/experiments/exp$experiment/exp$experiment.log
        chmod +rwx $OUT_PATH/experiments/exp$experiment/exp$experiment.log

        CUDA_VISIBLE_DEVICES=$device python3 $USER_PATH/src/trainer.py \
        --params $OUT_PATH/experiments/exp$experiment/params.yaml \
        --num $experiment \
        --verbose True &
        
        let experiment++
        let device++
    done
fi


