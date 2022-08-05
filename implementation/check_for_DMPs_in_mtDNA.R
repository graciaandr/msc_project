library(mixtools)
library(data.table)
library(magrittr)
library(dplyr)
library(annotatr)

setwd("C:/Users/andri/Documents/Uni London/QMUL/SemesterB/Masters_project/msc_project/data/classifying_data/")


CTRLvsCIN2plus_artistic_study_filt = read.table("artistic_study_filt-beta-values_0722_50threshold.txt",  sep = ";")
which(grepl(pattern = "M", rownames(CTRLvsCIN2plus_artistic_study_filt) ))

CTRLvsCIN2_artistic_study_filt = read.table("CTRLvsCIN2_artistic_study_filt-beta-values_0722_50threshold.txt",  sep = ";")
which(grepl(pattern = "M", rownames(CTRLvsCIN2_artistic_study_filt) ))

CTRLvsCIN3_artistic_study_filt = read.table("CTRLvsCIN3_artistic_study_filt-beta-values_0722_50threshold.txt",  sep = ";")
which(grepl(pattern = "M", rownames(CTRLvsCIN3_artistic_study_filt) ))

negCTRLvsCIN2plus_artistic_study_filt = read.table("negCTRL_CIN2+_artistic_study_filt-beta-values_0722_50threshold.txt",  sep = ";")
which(grepl(pattern = "M", rownames(negCTRLvsCIN2plus_artistic_study_filt) ))

negCTRLvsCIN2_artistic_study_filt = read.table("negCTRLvsCIN2_artistic_study_filt-beta-values_0722_50threshold.txt",  sep = ";")
which(grepl(pattern = "M", rownames(negCTRLvsCIN2_artistic_study_filt) ))

negCTRLvsCIN3_artistic_study_filt = read.table("negCTRLvsCIN3_artistic_study_filt-beta-values_0722_50threshold.txt",  sep = ";")
which(grepl(pattern = "M", rownames(negCTRLvsCIN3_artistic_study_filt) ))


