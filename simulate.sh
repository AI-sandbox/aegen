#!/bin/bash
echo "Generating $1 generations with $2 individuals in each"
if srun -u -p gpi.develop -c 4 --time 02:00:00 --mem 80GB python3 src/pyadmix/admix.py \
    $IN_PATH/data/chr22/ref_final_beagle_phased_1kg_hgdp_sgdp_chr22_hg19.vcf.gz \
    $IN_PATH/data/chr22/sample.map \
    $IN_PATH/data/allchrs.b37.gmap \
    $IN_PATH/data/chm22/prepared/ $1 $2; then
    echo "Success!"
else
    echo 'Fail!'
fi