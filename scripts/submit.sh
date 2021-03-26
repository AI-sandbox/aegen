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

if [ "$cluster" == "SHERLOCK" ]; then
sbatch <<EOT
#!/bin/sh
#SBATCH --job-name=VAEgen$experiment
#SBATCH -p gpu
#SBATCH -c 10
#SBATCH -G 1
#SBATCH --mem=64G
#SBATCH --time=23:59:59
#SBATCH -o $OUT_PATH/experiments/exp$experiment/exp$experiment.log
#SBATCH -e $OUT_PATH/experiments/exp$experiment/exp$experiment.err

ml load py-pytorch/1.4.0_py36
ml load py-numpy/1.18.1_py36
ml load py-matplotlib/3.2.1_py36
ml load py-pandas/1.0.3_py36
ml load py-h5py/2.10.0_py36
ml load py-scikit-learn/0.19.1_py36

if python3 $USER_PATH/src/trainer.py \
--params $OUT_PATH/experiments/exp$experiment/params.yaml \
--experiment "[S] Run $experiment: ReLU 50K 512-256" \
--verbose True \
--num $experiment \
--evolution True
then echo "[$CLUSTER] Success!"
else echo "[$CLUSTER] Fail!"; fi
EOT

    sleep .5
    cat $OUT_PATH/experiments/exp$experiment/params.yaml | grep size
    cat $OUT_PATH/experiments/exp$experiment/params.yaml | grep activation
    echo "LOG DIR: $OUT_PATH/experiments/exp$experiment/exp$experiment.log"

elif [ "$cluster" == "CALCULA" ]; then
sbatch <<EOT
#!/bin/sh
#SBATCH --job-name=VAEgen$experiment
#SBATCH -p gpi.compute
#SBATCH -c 4
#SBATCH --gres=gpu:2,gpumem:10G
#SBATCH --mem=64G
#SBATCH --time=23:59:59
#SBATCH -o $OUT_PATH/experiments/exp$experiment/exp$experiment.log
#SBATCH -e $OUT_PATH/experiments/exp$experiment/exp$experiment.err

if python3 $USER_PATH/src/trainer.py \
--params $OUT_PATH/experiments/exp$experiment/params.yaml \
--experiment "[C] Run $experiment: ReLU 10K 512-64 WAS" \
--verbose True \
--num $experiment \
--evolution True \
--only $experiment
then echo "[$CLUSTER] Success!"
else echo "[$CLUSTER] Fail!"; fi
EOT

    sleep .5
    cat $OUT_PATH/experiments/exp$experiment/params.yaml | grep size
    cat $OUT_PATH/experiments/exp$experiment/params.yaml | grep activation
    echo "LOG DIR: $OUT_PATH/experiments/exp$experiment/exp$experiment.log"

else echo "Undefined cluster."; fi