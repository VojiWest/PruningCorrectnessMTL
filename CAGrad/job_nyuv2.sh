#!/bin/bash
#SBATCH --job-name=attempt1
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=4GB

# remove all previously loaded modules
module purge

# load python 3.8.16
module load Python/3.9.6-GCCcore-11.2.0   
 
# activate virtual environment
source $HOME/venvs/first_env/bin/activate

# make a directory in the TMPDIR for the pre-trained model
mkdir $TMPDIR/pt

# Copy code to $TMPDIR
cp -r /scratch/$USER/nyuv2 $TMPDIR

# Navigate to TMPDIR
cd $TMPDIR/nyuv2

# Run training
# mkdir -p logs
mkdir $TMPDIR/logs

dataroot=/scratch/$USER/nyuv2
weight=equal
seed=0

# python -u model_segnet_cross.py  --apply_augmentation --dataroot $dataroot --seed $seed --weight $weight                             > logs/cross-$weight-sd$seed.log
python -u model_segnet_mtan.py   --apply_augmentation --dataroot $dataroot --seed $seed --weight $weight                             > $TMPDIR/logs/mtan-$weight-sd$seed.log
#python -u model_segnet_mt.py     --apply_augmentation --dataroot $dataroot --seed $seed --weight $weight --method cagrad --alpha 0.4 > logs/cagrad-$weight-4e-1-sd$seed.log
#python -u model_segnet_mt.py     --apply_augmentation --dataroot $dataroot --seed $seed --weight $weight --method mgd                > logs/mgd-$weight-sd$seed.log
#python -u model_segnet_mt.py     --apply_augmentation --dataroot $dataroot --seed $seed --weight $weight --method pcgrad             > logs/pcgrad-$weight-sd$seed.log
#python -u model_segnet_mt.py     --apply_augmentation --dataroot $dataroot --seed $seed --weight $weight --method graddrop           > logs/graddrop-$weight-sd$seed.log


# Save models by compressing and copying from TMPDIR
mkdir -p /scratch/$USER/basemodel/job_${job-name}
mv $TMPDIR/logs/* /scratch/$USER/basemodel/job_${job-name}
