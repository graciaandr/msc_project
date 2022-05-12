#!/bin/bash
#$ -cwd
#$ -j y
#$ -pe smp 4
#$ -l h_rt=240:0:0
#$ -l h_vmem=16G
#$ -N adaboost_job
#$ -m bea

# load Python
module load python
source msc_pyenv/bin/activate

# run SVM python script
python python_code/adaboost.py
