
cd $USER_PATH/src

if srun -u -p gpi.develop -c 4 --time 02:00:00 --mem 32GB \
python3 -c """
from utils.assemblers import create_dataset; import logging;
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
create_dataset(max_size=$1)
"""
then echo "Success!"
else echo 'Fail!'; fi