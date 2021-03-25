# !/bin/bash
# Sample command: ./instant.sh -e 1 -s "1K 256-50 ReLU" --verbose
while [[ $# -gt 0 ]]; do 
    case $1 in
        -v | --verbose ) shift; verbose="True" ;;
        -e | --exp ) shift; exp=$1 ;;
        -s | --summary ) shift; summary=$1 ;;
        * ) echo "Incorrect argument"; exit 1 ;;
    esac; shift
done

if [[ -z $verbose ]]; then verbose="False"; fi
if [[ -z $exp ]]; then echo "Missing experiment number."; exit 1; fi
if [[ -z $summary ]]; then echo "Missing experiment summary."; exit 1; fi

echo "Executing instant experiment ($summary)(#$exp)"
mkdir -p $OUT_PATH/experiments/exp$exp
cp $USER_PATH/params.yaml $OUT_PATH/experiments/exp$exp/
if srun -u --gres=gpu:2,gpumem:11G -p gpi.develop -c 4 --time 01:59:59 --mem 32GB python3 $USER_PATH/src/trainer.py \
    --params $OUT_PATH/experiments/exp$exp/params.yaml \
    --experiment "$summary" \
    --verbose $verbose \
    --num $exp \
    --evolution True; then
    echo "Success!"
else echo 'Fail!'; fi
