#!/bin/bash
#SBATCH --job-name=distribution_analysis
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --mem=8GB
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=./logs/%j.out

module purge

singularity exec --nv \
--overlay $SCRATCH/overlay/conda.ext3:ro \
/scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif /bin/bash -c "

source /ext3/env.sh;
python experiments/distribution_analysis/get_results.py --size 10 --datasetid 7dd9f0ea --modelid 811110cc
"
