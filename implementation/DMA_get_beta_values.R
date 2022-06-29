library(methylKit)
library(GenomicRanges)
library(IRanges)
library(devtools)
library(edmr)
library(mixtools)
library(data.table)
library(magrittr)
library(dplyr)
# library(annotatr)
# library(lumi)

### info about study & data: 
### RRBS bisulfite-converted hg19 reference genome using Bismark v0.15.0
### Genome_build: hg19 (GRCh37)
### chr, start, strand, number of cytosines (methylated bases) , number of thymines (unmethylated bases),context and trinucletide context format

setwd("/data/home/bt211038/makisoeo/MSc-project-cov-files/")


path = "/data/home/bt211038/makisoeo/MSc-project-cov-files/"
metadata = read.csv("/data/home/bt211038/msc_project/artistic_trial/Masterfile_groups-MSc-project.csv", sep = ";")
colnames(metadata)[c(1,3)] = c("lab_no", "CIN.type")
sampleids = as.list(as.character(metadata$lab_no))
treatments = as.vector(as.numeric(metadata$CIN.type))
list_of_files = as.list(paste0(path, metadata$coverage.file))
covariates = data.frame(hpv = as.factor(metadata$HPV.type), age = as.numeric(metadata$age))
print(list_of_files[1:10])


meth = readrDS("/data/home/bt211038/msc_project/artistic_trial/df_meth.rds")
df_meth = methylKit::getData(meth)
myDiff = readrDS("/data/home/bt211038/msc_project/artistic_trial/calculateDiffMeth_object.rds")

## Actual methylation values for each samples:
mat = percMethylation(meth, rowids = TRUE )
beta_values = mat/100
head(beta_values)

## add postions as own column to beta and m value data frames ==> for fitering & eventually classifier training
df_beta_vals = data.frame(beta_values) %>% dplyr::mutate(pos = df_meth$start, chrom = df_meth$chr)
# df_m_vals = data.frame(m_values) %>% dplyr::mutate(pos = df_meth$start, chrom = df_meth$chr)

## add chr in front of all chromosome names to be able to compare to seqnmaes in DMR dataframe later on when filtering
df_beta_vals['chr'] = paste0('chr', df_beta_vals$chrom)
# df_m_vals['chr'] = paste0('chr', df_m_vals$chrom)

write.table(df_beta_vals,
            file = "/data/home/bt211038/msc_project//classifying_data/artistic_study_initial_beta_values.txt",
            col.names = TRUE, sep = ";", row.names = TRUE)