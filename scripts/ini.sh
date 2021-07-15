export CLUSTER="SHERLOCK"

if [ "$CLUSTER" == "SHERLOCK" ]; then

    export USER_PATH="$HOME/aegen"
    export IN_PATH="$GROUP_SCRATCH/rita"
    export OUT_PATH="$GROUP_SCRATCH/rita"

elif [ "$CLUSTER" == "CALCULA" ]; then

    export USER_PATH="/home/usuaris/imatge/margarita.geleta/aegen"
    export IN_PATH="/mnt/gpid07/users/margarita.geleta"
    export OUT_PATH="/mnt/gpid07/users/margarita.geleta"
    source $USER_PATH/env/bin/activate

else echo "Define cluster."; fi
