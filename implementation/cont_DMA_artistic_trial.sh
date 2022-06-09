#!/bin/bash
#$ -cwd
#$ -j y
#$ -pe smp 4
#$ -l h_rt=240:0:0
#$ -l h_vmem=16G
#$ -l highmem
#$ -N job2_artistic
#$ -m bea

# load R
module load R/3.6.1

# run R script that did DMA for artistic study 
Rscript continued_analysis_of_artistic_study.R