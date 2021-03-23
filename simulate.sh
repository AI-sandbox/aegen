#!/bin/bash
dataset=("train" "valid" "test")
population=("EUR" "EAS" "AMR" "SAS" "AFR" "OCE" "WAS")
for set in "${dataset[@]}"
do mkdir -p $OUT_PATH/data/chr22/prepared/$set
    for pop in "${population[@]}"
    do mkdir -p $OUT_PATH/data/chr22/prepared/$set/$pop; 
       mkdir -p $OUT_PATH/data/chr22/prepared/$set/$pop/generations
    done
done
python3 $USER_PATH/src/utils/mapper.py
echo "Generating $1 generations with $2 individuals in each population"
if srun -u -p gpi.develop -c 4 --time 02:00:00 --mem 80GB \
python3 $USER_PATH/src/pyadmix/admix.py \
    $IN_PATH/data/chr22/ref_final_beagle_phased_1kg_hgdp_sgdp_chr22_hg19.vcf.gz \
    $OUT_PATH/data/chr22/prepared/ $1 $2; then
    echo "Success!"
else echo 'Fail!'; fi