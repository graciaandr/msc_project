#!/bin/bash
#$ -cwd
#$ -j y
#$ -pe smp 4
#$ -l h_rt=240:0:0
#$ -l h_vmem=8G
#$ -N CLL_DMA_job
#$ -m bea

# load Python
module load python

# run SVM python script
python python_code/svm.py