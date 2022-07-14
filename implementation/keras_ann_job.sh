#!/bin/bash
#$ -cwd
#$ -j y
#$ -pe smp 2
#$ -l h_rt=2:0:0
#$ -l h_vmem=4G
#$ -N ANN_keras_job
#$ -m bea

# load Python
module load python
source msc_pyenv/bin/activate

# run SVM python script
python python_code/ANN_keras.py
