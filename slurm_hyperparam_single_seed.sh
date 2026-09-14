#!/usr/bin/env bash
#SBATCH --job-name=HOQBM
#SBATCH --comment="Running 1 seeds with hyperparameter optimization"
#SBATCH --partition=Krater
#SBATCH --nodes=1-1
#SBATCH -n 1

export TMPDIR=$HOME/tmp
mkdir -p $TMPDIR

mkdir -p out/

source ~/miniconda3/etc/profile.d/conda.sh
conda activate qbm-env
echo "conda env loaded"

SEED=$1
BATCH_SIZE=$2
LEARNING_RATE=$3
#CHEBYSHEV_DEGREE=$4
BETA=$4
#GV_1=$6
#GV_2=$7
WAY=$5


# Dispatch each seed in parallel

echo "Dispatching job: seed $SEED with hyperparameters"
srun --exclusive --ntasks=1 --nodes=1 -c 1 python3 slurm_single_seed.py \
    --seed $SEED \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --beta $BETA \
    --path $WAY &
    #--chebyshev_degree $CHEBYSHEV_DEGREE \
    #--gv_2 $GV_2 \
    #--gv_1 $GV_1 \


echo -e "\tWaiting for Job completion."
wait

echo -e "\nAll jobs finished"

echo "Slurm execution done."
