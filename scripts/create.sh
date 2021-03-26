#!/bin/bash
for ARGUMENT in "$@" 
do
    KEY=$(echo $ARGUMENT | cut -f1 -d=)
    VALUE=$(echo $ARGUMENT | cut -f2 -d=)   
    case "$KEY" in
            cluster)     cluster=${VALUE} ;;
            individuals) individuals=${VALUE} ;;     
            *)   
    esac    
done
if [[ -z $cluster ]]; then cluster=$CLUSTER; fi
if [[ -z $individuals ]]; then echo "Missing individuals parameters."; exit 1; fi
echo "[$CLUSTER] Generating datasets with $individuals individuals"
if [ "$cluster" == "SHERLOCK" ]; then
source ini.sh
cd $USER_PATH/src
sbatch <<EOT
#!/bin/sh
#SBATCH --job-name=cVAEgen$1
#SBATCH -p gpu
#SBATCH -c 10
#SBATCH -G 1
#SBATCH --mem=32G
#SBATCH -t 00:30:00
#SBATCH -o $OUT_PATH/logs/sim.log
#SBATCH -e $OUT_PATH/logs/sim.err

ml load py-pytorch/1.4.0_py36
ml load py-numpy/1.18.1_py36
ml load py-pandas/1.0.3_py36
ml load py-h5py/2.10.0_py36

python3 $USER_PATH/src/utils/mapper.py
if python3 -c """
from utils.assemblers import create_dataset; import logging;
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
create_dataset(max_size=$individuals)
""" 
then echo "[$CLUSTER] Success!"
else echo "[$CLUSTER] Fail!"; fi
EOT
elif [ "$cluster" == "CALCULA" ]; then
    if srun -u -p gpi.develop -c 4 --time 02:00:00 --mem 32GB \
python3 -c """
from utils.assemblers import create_dataset; import logging;
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
create_dataset(max_size=$individuals)
"""
    then echo "[$CLUSTER] Success!"
    else echo "[$CLUSTER] Fail!"; fi
else echo "Undefined cluster."; fi