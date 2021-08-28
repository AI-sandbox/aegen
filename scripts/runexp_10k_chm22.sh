#!/bin/bash
####################################################################################
################################## DO NOT CHANGE ###################################
####################################################################################
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

## Default settings for all runs:
python3 $USER_PATH/scripts/hypertuning.py \
        --ipath $USER_PATH/params.yaml    \
        --opath $USER_PATH/optparams.yaml \
        --latent Uniform \
        --lr 0.1 \
        --optimizer Adam \
        --weight_decay 0.001 \
        --heads 1 \
        --vqbeta 2 \
        --bottleneck 512 \
        --chm 22 \
        --isize 10000
####################################################################################
################################## DO NOT CHANGE ###################################
####################################################################################

## Define below all the experiments to try out:
## Max experiments per device: 3

## ON DEVICE 2:
device=2
#1####################################################################################
echo "[$CLUSTER] Executing experiment #$experiment"
python3 $USER_PATH/scripts/hypertuning.py \
    --ipath $USER_PATH/optparams.yaml    \
    --opath $USER_PATH/optparams.yaml \
    --optimizer AdamW \
    --lr 0.1 \
    --weight_decay 0.001 \
    --vqbeta 2

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
#2####################################################################################
echo "[$CLUSTER] Executing experiment #$experiment"
python3 $USER_PATH/scripts/hypertuning.py \
    --ipath $USER_PATH/optparams.yaml    \
    --opath $USER_PATH/optparams.yaml \
    --optimizer QHAdam \
    --lr 0.1 \
    --weight_decay 0.001 \
    --vqbeta 2

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
#3####################################################################################
echo "[$CLUSTER] Executing experiment #$experiment"
python3 $USER_PATH/scripts/hypertuning.py \
    --ipath $USER_PATH/optparams.yaml    \
    --opath $USER_PATH/optparams.yaml \
    --optimizer QHAdam \
    --lr 0.25 \
    --weight_decay 0.001 \
    --vqbeta 2

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

## ON DEVICE 3:
let device++
#4####################################################################################
echo "[$CLUSTER] Executing experiment #$experiment"
python3 $USER_PATH/scripts/hypertuning.py \
    --ipath $USER_PATH/optparams.yaml    \
    --opath $USER_PATH/optparams.yaml \
    --optimizer QHAdam \
    --lr 0.25 \
    --weight_decay 0.001 \
    --vqbeta 10

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
#5####################################################################################
echo "[$CLUSTER] Executing experiment #$experiment"
python3 $USER_PATH/scripts/hypertuning.py \
    --ipath $USER_PATH/optparams.yaml    \
    --opath $USER_PATH/optparams.yaml \
    --optimizer QHAdam \
    --lr 0.075 \
    --weight_decay 0.001 \
    --vqbeta 10

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
#6####################################################################################
echo "[$CLUSTER] Executing experiment #$experiment"
python3 $USER_PATH/scripts/hypertuning.py \
    --ipath $USER_PATH/optparams.yaml    \
    --opath $USER_PATH/optparams.yaml \
    --optimizer AdamW \
    --lr 0.25 \
    --weight_decay 0.001 \
    --vqbeta 2 \
    --heads 2

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

## ON DEVICE 4:
let device++
#7####################################################################################
echo "[$CLUSTER] Executing experiment #$experiment"
python3 $USER_PATH/scripts/hypertuning.py \
    --ipath $USER_PATH/optparams.yaml    \
    --opath $USER_PATH/optparams.yaml \
    --optimizer QHAdam \
    --lr 0.25 \
    --weight_decay 0.001 \
    --vqbeta 2 \
    --heads 2

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
#8####################################################################################
echo "[$CLUSTER] Executing experiment #$experiment"
python3 $USER_PATH/scripts/hypertuning.py \
    --ipath $USER_PATH/optparams.yaml    \
    --opath $USER_PATH/optparams.yaml \
    --optimizer QHAdam \
    --lr 0.25 \
    --weight_decay 0.001 \
    --vqbeta 2 \
    --heads 3

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
#9####################################################################################
echo "[$CLUSTER] Executing experiment #$experiment"
python3 $USER_PATH/scripts/hypertuning.py \
    --ipath $USER_PATH/optparams.yaml    \
    --opath $USER_PATH/optparams.yaml \
    --optimizer AdamW \
    --lr 0.25 \
    --weight_decay 0.001 \
    --vqbeta 2 \
    --winshare true

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

## ON DEVICE 5:
let device++
#10####################################################################################
echo "[$CLUSTER] Executing experiment #$experiment"
python3 $USER_PATH/scripts/hypertuning.py \
    --ipath $USER_PATH/optparams.yaml    \
    --opath $USER_PATH/optparams.yaml \
    --optimizer QHAdam \
    --lr 0.25 \
    --weight_decay 0.001 \
    --vqbeta 2 \
    --winshare true

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
#11####################################################################################
echo "[$CLUSTER] Executing experiment #$experiment"
python3 $USER_PATH/scripts/hypertuning.py \
    --ipath $USER_PATH/optparams.yaml    \
    --opath $USER_PATH/optparams.yaml \
    --optimizer QHAdam \
    --lr 0.075 \
    --weight_decay 0.001 \
    --vqbeta 5 \
    --winshare true

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

# END.