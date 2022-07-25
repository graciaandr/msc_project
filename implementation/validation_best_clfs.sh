#!/bin/bash
#$ -cwd
#$ -j y
#$ -pe smp 4
#$ -l h_rt=24:0:0
#$ -l h_vmem=4G
#$ -N best_clfs_validation
#$ -m bea

# load Python
module load python
source msc_pyenv/bin/activate

# run SVM python script
python python_code/validation_of_best_clfs.py
