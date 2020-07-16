#!/bin/bash

#SBATCH --ntasks 1
#SBATCH --partition allgroups
#SBATCH --time 48:00:00
#SBATCH --mem 128G

#SBATCH --gres=gpu:1

#SBATCH --job-name colotti_lorenzi_HDA
#SBATCH --output output_%j.txt
#SBATCH --error errors_%j.txt
#SBATCH --mail-user giovanni.colotti@studenti.unipd.it

# setup my personal/shared project folder
MY_DIR=/nfsd/hda/colottigio

# make my folder the current directory
cd $MY_DIR

# setup python3 environment for machine learning computations 
source /nfsd/opt/anaconda3/anaconda3.sh 
conda activate /nfsd/signet/hda_env

# run code
srun python3 /nfsd/hda/colottigio/humandata_project/features_CNN_raw.py

# deactivate environment at the end of the job
conda deactivate
