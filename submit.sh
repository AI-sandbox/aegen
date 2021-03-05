#!/bin/bash
echo "Executing experiment $1"
rm -rf $OUT_PATH/exp_$1
mkdir -p $OUT_PATH/exp_$1
cp $USER_PATH/params.yaml $OUT_PATH/exp_$1/

sbatch <<EOT
#!/bin/sh
#SBATCH --job-name=VAEgen$1
#SBATCH -p gpi.compute
#SBATCH -c 4
#SBATCH --gres=gpu:1,gpumem:11G
#SBATCH --mem=32G
#SBATCH --time=23:59:59
#SBATCH -o logs/exp_$1.log
#SBATCH -e logs/exp_$1.err

python3 src/trainer.py \
--params $OUT_PATH/exp_$1/params.yaml \
--experiment "Run $1: GELU 25K 256-64" \
--verbose True \
--num $1 \
--evolution True

echo "Success!"
EOT

sleep .5
cat $OUT_PATH/exp_$1/params.yaml | grep size
cat $OUT_PATH/exp_$1/params.yaml | grep activation