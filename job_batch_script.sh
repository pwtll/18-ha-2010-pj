#!/bin/bash

#SBATCH -A kurs00048
#SBATCH -p kurs00048
#SBATCH --reservation=kurs00048

#SBATCH -J 2D_CNN_Densenet121_Training_2021-11-30
#SBATCH --mail-type=END

#SBATCH -n 1
#SBATCH --mem-per-cpu=3500
#SBATCH -t 01:00:00

##SBATCH --gres=gpu

#SBATCH -e /home/kurse/kurs00048/pw77bene/logs/%x.err.%j
#SBATCH -o /home/kurse/kurs00048/pw77bene/logs/%x.out.%j
#SBATCH -D /home/kurse/kurs00048/pw77bene/18-ha-2010-pj


echo "2D CNN Densenet121 Training 2021-11-30"

module purge
module load gcc
module load python/3.8.0

srun pip install -r requirements.txt

srun python all_csv_to_three_peaks_images.py
srun python train.py
srun python predict_pretrained.py
srun python score.py