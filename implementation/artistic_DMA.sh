#!/bin/bash
#$ -cwd
#$ -j y
#$ -pe smp 4
#$ -l h_rt=240:0:0
#$ -l h_vmem=50G
#$ -l highmem
#$ -N job_artistic_DMA
#$ -m bea

# load R
module load R/3.6.1

# run R script that did DMA for artistic study 
Rscript DMA_of_artistic_study.R