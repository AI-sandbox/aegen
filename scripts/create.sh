#!/bin/bash
source ini.sh
cd $USER_PATH/src
for ARGUMENT in "$@" 
do
    KEY=$(echo $ARGUMENT | cut -f1 -d=)
    VALUE=$(echo $ARGUMENT | cut -f2 -d=)   
    case "$KEY" in
        cluster)  cluster=${VALUE} ;;
	    ini)      ini=${VALUE} ;;
    	end)      end=${VALUE} ;;
	    species)  species=${VALUE};;
	    chm)      chm=${VALUE};;
	    split)    split=${VALUE};;
    esac    
done
if [[ -z $cluster ]]; then cluster=$CLUSTER; fi
if [[ -z $ini ]]; then ini=0; fi
if [[ -z $end ]]; then end=-1; fi
if [[ -z $species ]]; then echo "Undefined species."; exit 1; fi
if [[ -z $chm ]]; then echo "Undefined chromosome."; exit 1; fi
if [[ -z $split ]]; then echo "Undefined split."; exit 1; fi
if [ "$cluster" == "NERO" ]; then
if [ "$chm" == "all" ]; then
echo "[$CLUSTER] Generating $split dataset with subset ($ini,$end) of $species SNPs from $chm chromosomes"
if python3 -c """
from utils.assemblers import create_dataset; import logging;
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
create_dataset('$species','$chm',split='$split',arange=($ini,$end))
""" 
then echo "[$CLUSTER] Success!"
else echo "[$CLUSTER] Fail!"; fi
else
echo "[$CLUSTER] Generating $split dataset with subset ($ini,$end) of $species SNPs from chromosome $chm"
fi
if python3 -c """
from utils.assemblers import create_dataset; import logging;
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
create_dataset('$species',$chm,split='$split',arange=($ini,$end))
""" 
then echo "[$CLUSTER] Success!"
else echo "[$CLUSTER] Fail!"; fi
elif [ "$cluster" == "SHERLOCK" ]; then
sbatch <<EOT
#!/bin/sh
#SBATCH --job-name=cVAEgen$1
#SBATCH -p gpu
#SBATCH -c 10
#SBATCH -G 1
#SBATCH --mem=240G
#SBATCH -t 23:59:00
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
create_dataset(arange=($ini,$end))
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
create_dataset(max_size=$snps)
"""
    then echo "[$CLUSTER] Success!"
    else echo "[$CLUSTER] Fail!"; fi
else echo "Undefined cluster."; fi
