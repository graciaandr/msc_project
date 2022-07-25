#!/bin/bash
#$ -cwd
#$ -j y
#$ -pe smp 4
#$ -l h_rt=240:0:0
#$ -l h_vmem=128G
#$ -l highmem
#$ -N artistic_DMA_part1
#$ -m bea

# load R
module load R/3.6.1

# run R script that did DMA for artistic study 
Rscript negative_CTRL_subgroups_DMA_of_artistic_study.R