#!/bin/bash
sbatch <<EOT
#!/bin/sh
#SBATCH --job-name=VAEgen-$1
#SBATCH -p gpi.compute
#SBATCH --gres=gpu:1,gpumem:10G
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH -o logs/exp_$1.log
#SBATCH -e logs/exp_$1.err

echo "Executing experiment $1"
python3 src/trainer.py
echo "Success!"
EOT