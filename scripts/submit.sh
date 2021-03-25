#!/bin/bash
echo "Executing experiment $1"
rm -rf $OUT_PATH/experiments/exp$1
mkdir -p $OUT_PATH/experiments/exp$1
cp $USER_PATH/params.yaml $OUT_PATH/experiments/exp$1/
touch $OUT_PATH/experiments/exp$1/exp$1.log
chmod +rwx $OUT_PATH/experiments/exp$1/exp$1.log

sbatch <<EOT
#!/bin/sh
#SBATCH --job-name=VAEgen$1
#SBATCH -p gpi.compute
#SBATCH -c 4
#SBATCH --gres=gpu:2,gpumem:11G
#SBATCH --mem=64G
#SBATCH --time=23:59:59
#SBATCH -o $OUT_PATH/experiments/exp$1/exp$1.log
#SBATCH -e $OUT_PATH/experiments/exp$1/exp$1.err

python3 $USER_PATH/src/trainer.py \
--params $OUT_PATH/experiments/exp$1/params.yaml \
--experiment "Run $1 (new): ReLU 1K 512-256" \
--verbose True \
--num $1 \
--evolution True

echo "Success!"
EOT

sleep .5
cat $OUT_PATH/experiments/exp$1/params.yaml | grep size
cat $OUT_PATH/experiments/exp$1/params.yaml | grep activation
echo "LOG DIR: $OUT_PATH/experiments/exp$1/exp$1.log"