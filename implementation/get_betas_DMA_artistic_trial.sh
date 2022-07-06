#!/bin/bash
#$ -cwd
#$ -j y
#$ -pe smp 8
#$ -l h_rt=240:0:0
#$ -l h_vmem=20G
#$ -l highmem
#$ -N betas_artistic
#$ -m bea

# load R
module load R/3.6.1

# run R script that did DMA for artistic study 
Rscript DMA_get_beta_values.R