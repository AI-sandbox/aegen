#!/bin/bash
echo "Executing instant experiment $1"
mkdir -p $OUT_PATH/exp_$1
cp params.json $OUT_PATH/exp_$1/
if srun -u --gres=gpu:1,gpumem:16G -p gpi.compute -c 4 --time 23:59:59 --mem 80GB python3 src/trainer.py \
    --params $OUT_PATH/exp_$1/params.json; then
    echo "Success!"
else
    echo 'Fail!'
fi
