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
mkdir -p $OUT_PATH/exp_$exp
cp params.json $OUT_PATH/exp_$exp/
if srun -u --gres=gpu:1,gpumem:16G -p gpi.compute -c 4 --time 23:59:59 --mem 80GB python3 src/trainer.py \
    --params $OUT_PATH/exp_$1/params.json \
    --experiment "$summary" \
    --verbose $verbose; then
    echo "Success!"
else echo 'Fail!'; fi
