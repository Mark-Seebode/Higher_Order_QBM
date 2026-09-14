#!/usr/bin/env bash
#SBATCH --job-name=hyperHOQBM
#SBATCH --comment="GQEVT"
#SBATCH --mail-type=ALL
#SBATCH --partition=Krater
#SBATCH --nodes=1
#SBATCH -n 1

export TMPDIR=$HOME/tmp
mkdir -p $TMPDIR

mkdir -p out/

source ~/miniconda3/etc/profile.d/conda.sh
conda activate qbm-env
echo "conda env loaded"




echo "Dispatching job"
srun --exclusive --ntasks=1 --nodes=1 -c 1 python3 hyper.py --path "out/"

echo -e "\tWaiting for Job completion."
wait

echo -e "\nAll jobs finished"

#mail -s "QCV-test-run-finished"
echo "Slurm execution done."
