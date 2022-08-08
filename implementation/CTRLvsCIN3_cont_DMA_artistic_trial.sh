#!/bin/bash
#$ -cwd
#$ -j y
#$ -pe smp 8 
#$ -l h_rt=24:0:0
#$ -l h_vmem=32G
#$ -l highmem
#$ -N part2_CTRLvsCIN3
#$ -m bea

# load R
module load R/3.6.1

# run R script that did DMA for artistic study 
Rscript CTRLvsCIN3_continued_analysis_of_artistic_study.R