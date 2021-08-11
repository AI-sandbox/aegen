#!/bin/bash
source /home/geleta/aegen/scripts/ini.sh
for ARGUMENT in "$@"
do
    KEY=$(echo $ARGUMENT | cut -f1 -d=)
    VALUE=$(echo $ARGUMENT | cut -f2 -d=)
    case "$KEY" in
            cluster)     cluster=${VALUE} ;;
            species)     species=${VALUE} ;;
	    chr)         chr=${VALUE} ;;
            *)
    esac
done
if [[ -z $cluster ]]; then cluster=$CLUSTER; fi
if [[ -z $species ]]; then echo "Missing species. Exiting..."; exit 1; fi
if [[ -z $chr ]]; then echo "Missing chromosome. Exiting..."; exit 1; fi
dataset=("train" "valid" "test")
population=("EUR" "EAS" "AMR" "SAS" "AFR" "OCE" "WAS")
for set in "${dataset[@]}"
do mkdir -p $OUT_PATH/data/$species/chr$chr/prepared/$set
    for pop in "${population[@]}"
    do mkdir -p $OUT_PATH/data/$species/chr$chr/prepared/$set/$pop;
       mkdir -p $OUT_PATH/data/$species/chr$chr/prepared/$set/$pop/generations
    done
done
