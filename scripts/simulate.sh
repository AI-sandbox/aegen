#!/bin/bash
source ini.sh
dataset=("train" "valid" "test")
population=("EUR" "EAS" "AMR" "SAS" "AFR" "OCE" "WAS")
for set in "${dataset[@]}"
do mkdir -p $OUT_PATH/data/chr22/prepared/$set
    for pop in "${population[@]}"
    do mkdir -p $OUT_PATH/data/human/chr22/prepared/$set/$pop; 
       mkdir -p $OUT_PATH/data/human/chr22/prepared/$set/$pop/generations
    done
done
for ARGUMENT in "$@" 
do
    KEY=$(echo $ARGUMENT | cut -f1 -d=)
    VALUE=$(echo $ARGUMENT | cut -f2 -d=)   
    case "$KEY" in
            cluster)     cluster=${VALUE} ;;
            generations) generations=${VALUE} ;;  
            individuals) individuals=${VALUE} ;;     
            *)   
    esac    
done
if [[ -z $cluster ]]; then cluster=$CLUSTER; fi
if [[ -z $generations ]]; then echo "Missing generations parameters."; exit 1; fi
if [[ -z $individuals ]]; then echo "Missing individuals parameters."; exit 1; fi
echo "[$CLUSTER] Generating $generations generations with $individuals individuals in each population"
cd $USER_PATH/src
if [ "$cluster" == "SHERLOCK" ]; then
sbatch <<EOT
#!/bin/sh
#SBATCH --job-name=sVAEgen$1
#SBATCH -p gpu
#SBATCH -c 10
#SBATCH -G 1
#SBATCH --mem=80G
#SBATCH -t 00:30:00
#SBATCH -o $OUT_PATH/logs/sim.log
#SBATCH -e $OUT_PATH/logs/sim.err

ml load py-pytorch/1.4.0_py36
ml load py-numpy/1.18.1_py36
ml load py-pandas/1.0.3_py36
ml load py-h5py/2.10.0_py36

python3 $USER_PATH/src/utils/mapper.py
if python3 $USER_PATH/src/pyadmix/admix.py \
    $IN_PATH/data/human/chr22/ref_final_beagle_phased_1kg_hgdp_sgdp_chr22_hg19.vcf \
    $OUT_PATH/data/human/chr22/prepared/ $generations $individuals
then echo "[$CLUSTER] Success!"
else echo "[$CLUSTER] Fail!"; fi
EOT
elif [ "$cluster" == "CALCULA" ]; then
    python3 $USER_PATH/src/utils/mapper.py
    if srun -u -p gpi.develop -c 4 --time 02:00:00 --mem 80GB \
    python3 $USER_PATH/src/pyadmix/admix.py \
        $IN_PATH/data/chr22/ref_final_beagle_phased_1kg_hgdp_sgdp_chr22_hg19.vcf.gz \
        $OUT_PATH/data/chr22/prepared/ $generations $individuals 
    then echo "[$CLUSTER] Success!"
    else echo "[$CLUSTER] Fail!"; fi
else echo "Undefined cluster."; fi
