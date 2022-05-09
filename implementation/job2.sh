#!/bin/bash
#$ -cwd
#$ -j y
#$ -pe smp 4
#$ -l h_rt=240:0:0
#$ -l h_vmem=8G
#$ -N CD4-T-cell_job
#$ -m bea

# load R
module load R/3.6.1

# run R script that did DMA for CD4 t cell study 
Rscript DMA_of_CD4_study.R