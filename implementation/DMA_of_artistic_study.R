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


## Differential Methylation Analysis
start.time1 <- Sys.time()
print(start.time1)
## read files with methRead
myobj=methylKit::methRead(location = list_of_files,
               sample.id = sampleids,
               assembly ="hg19", # used GrCh37 - hg19 for mapping
               treatment = treatments,
               context="CpG",
               header = TRUE, 
               pipeline = 'bismarkCytosineReport',
               resolution = "base",
               sep = '\t',
               dbdir = "/data/home/bt211038/msc_project/artistic_trial/"
)

end.time1 <- Sys.time()
time.taken1 <- end.time1 - start.time1
print(time.taken1)


### filter out low coverage / low quality CpGs

myobj_filtered = filterByCoverage(myobj, lo.count = 10, lo.perc = NULL,
                 hi.count = NULL, hi.perc = 99.5)

## Unite / merge samples, only keep CpGs that are methylated in at least 2 samples
start.time2 <- Sys.time()
print(start.time2)
meth = unite(myobj_filtered, destrand=FALSE, min.per.group = 5L) 
end.time2 <- Sys.time()
time.taken2 <- end.time2 - start.time2
print(time.taken2)

print("df meth has this many rows:")
print(nrow(meth))

saveRDS(meth, file = "/data/home/bt211038/msc_project/artistic_trial/df_meth.rds")



start.time3 <- Sys.time()
## Finding differentially methylated bases
myDiff <- calculateDiffMeth(meth,
                            overdispersion = "MN",
                            effect         = "wmean",
                            test           = "F",
                            adjust         = 'BH',
                            slim           = F,
                            weighted.mean  = T,
                            covariates = covariates, 
                            mc.cores = 4)

saveRDS(myDiff, file = "/data/home/bt211038/msc_project/artistic_trial/calculateDiffMeth_object.rds")
# myDiff

end.time3 <- Sys.time()
time.taken3 <- end.time3 - start.time3
print(time.taken3)

## Actual methylation values for each samples:
mat = percMethylation(meth, rowids = TRUE )
beta_values = mat/100
head(beta_values)


df_meth = methylKit::getData(meth)
print(nrow(df_meth))
  
## add postions as own column to beta and m value data frames ==> for fitering & eventually classifier training
df_beta_vals = data.frame(beta_values) %>% dplyr::mutate(pos = df_meth$start, chrom = df_meth$chr)
# df_m_vals = data.frame(m_values) %>% dplyr::mutate(pos = df_meth$start, chrom = df_meth$chr)

## add chr in front of all chromosome names to be able to compare to seqnmaes in DMR dataframe later on when filtering
df_beta_vals['chr'] = paste0('chr', df_beta_vals$chrom)
# df_m_vals['chr'] = paste0('chr', df_m_vals$chrom)

print(nrow(df_beta_vals))

write.table(df_beta_vals, file = "/data/home/bt211038/msc_project/classifying_data/artistic_study_initial_beta_values.txt", col.names = TRUE, sep = ";", row.names = TRUE)   
