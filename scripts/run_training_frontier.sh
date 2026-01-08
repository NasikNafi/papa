#!/bin/bash
##SBATCH -A ######
#SBATCH -J diff-tuning
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH -t 00:20:00
#SBATCH -p batch
#SBATCH -o ./logs/training-log-%j.out
#SBATCH -e ./logs/training-log-%j.out

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ddpm

path='/lustre/orion/stf006/proj-shared/nafi/projects/papa/papa/'
echo $path
cd $path

module load PrgEnv-gnu
module load gcc/12.2.0
module load rocm/6.0.0 libtool

# export OMP_NUM_THREADS=1
export MIOPEN_DISABLE_CACHE=1
export NCCL_DEBUG=INFO
export NCCL_PROTO=Simple

export MIOPEN_USER_DB_PATH=/tmp/$JOBID
mkdir -p $MIOPEN_USER_DB_PATH
export PYTHONPATH=$PWD:$PYTHONPATH

time srun -n $((SLURM_JOB_NUM_NODES*8)) python -m tools.train_papa --config config/default_papa.yaml