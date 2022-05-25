#!/bin/bash
#$ -cwd
#$ -j y
#$ -pe smp 4
#$ -l h_rt=240:0:0
#$ -l h_vmem=32G
#$ -l highmem
#$ -N CLL_DMA_job
#$ -m bea

# load R
module load R/3.6.1

# run R script that did DMA for CLL study with 451 samples 
Rscript DMA_of_CLL_study.R