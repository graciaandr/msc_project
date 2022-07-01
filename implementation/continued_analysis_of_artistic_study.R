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

setwd("/data/home/bt211038/msc_project/")
# setwd("C:/Users/andri/Documents/Uni London/QMUL/SemesterB/Masters_project/msc_project/")


metadata = read.csv("artistic_trial/Masterfile_groups-MSc-project.csv", sep = ";")
colnames(metadata)[c(1,3)] = c("lab_no", "CIN.type")
sampleids = as.list(as.character(metadata$lab_no))
treatments = as.vector(as.numeric(as.factor(metadata$CIN.type)))
covariates = data.frame(hpv = as.factor(metadata$HPV.type), age = as.numeric(metadata$age))
print(metadata %>% head(5))

meth <- readRDS("artistic_trial/df_meth.rds")
print("df meth has this many rows:")
print(nrow(meth))

df_meth <- methylKit::getData(meth)

myDiff <- readRDS("artistic_trial/calculateDiffMeth_object.rds")
print("myDiff has this many rows:")
print(nrow(myDiff))

df_beta_vals <- read.table("classifying_data/artistic_study_initial_beta_values.txt", header=T, sep=";")
# only look at mtDNA --> remove other chromosomes
# df_beta_vals = df_beta_vals %>% filter(chrom == "chrM")
print(head(df_beta_vals))
print(nrow(df_beta_vals))

## try thresholds of 10%, 25% and 50% for cpgs to keep per conditions
## Removing NAs: function to keep rows with n many NAs in that row -- default n is 0.25 
keep_rows <- function(df, metadata,  perc = 0.25) {
  samp_per_cpg = round((perc) * 240)
  print(samp_per_cpg)
  col_indeces_Ctrl = which(grepl( "Control" , metadata$Phenotype.RRBS ) )
  col_indeces_Case = which(grepl( "Case" , metadata$Phenotype.RRBS ) )
  df_ctrl <- df[ , col_indeces_Ctrl ]
  df_cases <- df[ , col_indeces_Case ]
  print(dim(df_ctrl))
  print(dim(df_cases))
  row_indeces_NAs <- c()
  keep_in_ctrl <- which(rowSums(!is.na(df_ctrl)) >= samp_per_cpg)
  keep_in_cases <- which(rowSums(!is.na(df_cases)) >= samp_per_cpg)
  tmp <- c(keep_in_ctrl, keep_in_cases)
  indeces = unique(sort(tmp))
  if (length(indeces) == 0) {
    indeces = NULL
    return(indeces)
  }
  return(indeces)
}


### remove droplist CpGs
df_bed_file <- as.data.frame(read.table("bed_file/hg19-blacklist.v2.bed",header = FALSE, sep="\t",stringsAsFactors=FALSE, quote=""))
colnames(df_bed_file) <- c("chromosome", "start", "end", "info")
print("bed file:")
head(df_bed_file)

df_dmrs_false_cpgs = NULL
print("filtering of droplist CpGs from CpGs:")

### filter droplist CpGs from CpGs and calcDiff object (myDiff)
for (i in (1:length(df_bed_file$start))) {
  df_tmp = df_beta_vals %>%
    dplyr::filter(pos >= df_bed_file$start[[i]] & pos <= df_bed_file$end[[i]] & chrom != df_bed_file$chromosome[[i]])
  df_dmrs_false_cpgs = rbind(df_dmrs_false_cpgs, df_tmp)
}

## retrieve the valid CpGs
df_valid_cpgs = setdiff(df_beta_vals,df_dmrs_false_cpgs)
print( nrow(df_valid_cpgs))


df_dmrs_false_DMCs = NULL
df_methDiff = getData(myDiff)
print("filtering of droplist CpGs from calcDiff object (myDiff):")

### filter droplist CpGs from calcDiff object (myDiff)
for (i in (1:length(df_bed_file$start))) {
# for (i in (1:10)) { # to test if for loop works 
  df_tmp = df_methDiff %>%
    dplyr::filter(start >= df_bed_file$start[[i]] & end <= df_bed_file$end[[i]] & chr != df_bed_file$chromosome[[i]])
  df_dmrs_false_DMCs = rbind(df_dmrs_false_DMCs, df_tmp)
}

## retrieve the valid DMCs
df_valid_DMCs = setdiff(df_methDiff,df_dmrs_false_DMCs)
print( nrow(df_valid_DMCs))


### save valid CpGs and DMCs
write.table(df_valid_cpgs,
            file = "classifying_data/df_valid_cpgs.txt",
            col.names = TRUE, sep = ";", row.names = TRUE)


write.table(df_valid_DMCs,
            file = "classifying_data/df_valid_DMCs.txt",
            col.names = TRUE, sep = ";", row.names = TRUE)


# remove all unnecessary rows
print("removal of rows with too many NAs (10% threshold) ")
indeces_to_keep = keep_rows(df = df_valid_cpgs, metadata, perc = 0.1)
df_beta_vals_filt = df_valid_cpgs[indeces_to_keep,]
df_meth_new =  df_meth[indeces_to_keep,]
myDiff2 = df_valid_DMCs[indeces_to_keep,]

print("#Rows of df beta vals after NA handeling: ")
print(nrow(df_beta_vals_filt))
write.table(df_beta_vals_filt,
             file = "classifying_data/artistic_study_betas_b4_EDMR_10threshold_mtDNA.txt",
             col.names = TRUE, sep = ";", row.names = TRUE)

## Q Value adjustment of filtered calc Meth Diff Object
adj_q_vals = p.adjust(myDiff2$pvalue, method = "BH")
myDiff2$qvalue = adj_q_vals

print("adjust qvalues in ")
# filter myDiff for qvalue of 0.05 and methylationDifference of 10 
# df_adjusted_diff_meth = methylKit::getMethylDiff(qvalue = 0.05, difference = 10)
df_adjusted_diff_meth = myDiff2 %>% filter(qvalue < 0.05 & abs(meth.diff) >= 10)
print(dim(df_adjusted_diff_meth))

write.table(df_adjusted_diff_meth,
            file = "classifying_data/df_adjusted_diff_meth.txt",
            col.names = TRUE, sep = ";", row.names = TRUE)


## EDMR: calculate all DMRs candidate from complete myDiff dataframe
print("DMR Analysis:")
dm_regions=edmr(myDiff = df_adjusted_diff_meth, mode=2, ACF=TRUE, DMC.qvalue = 0.05, plot = FALSE)
df_dmrs = data.frame(dm_regions)
print(nrow(df_dmrs))
write.table(df_dmrs,
            file = "classifying_data/all_DMRs.txt",
            col.names = TRUE, sep = ";", row.names = TRUE)


## for loop that goes through the start pos, end pos, and seqnames per row in beta/m value dataframe and DMR data
## to retrieve sig. diff. meth. CpG sites in DMRs
df_tmp1 = data.frame(matrix(NA, nrow = 1, ncol = ncol(df_beta_vals_filt)))
# df_tmp2 = data.frame(matrix(NA, nrow = 1, ncol = ncol(df_m_vals)))
colnames(df_tmp1) <- colnames((df_beta_vals_filt))
# colnames(df_tmp2) <- colnames((df_m_vals))
df_beta_vals_filtered = NULL
# df_m_vals_filtered = NULL
for (i in (1:length(df_valid_DMCs$start))) {
  df_tmp1 = df_beta_vals_filt %>%
    dplyr::filter(pos >= df_valid_DMCs$start[[i]] & pos <= df_valid_DMCs$end[[i]] & chrom == df_valid_DMCs$seqnames[[i]])
  #   df_tmp2 = df_m_vals %>%
  # dplyr::filter(pos >= df_dmrs$start[[i]] & pos <= df_dmrs$end[[i]] & chrom == df_dmrs$seqnames[[i]])
  
  df_beta_vals_filtered = rbind(df_beta_vals_filtered, df_tmp1)
  # df_m_vals_filtered = rbind(df_m_vals_filtered, df_tmp2)
}

# print(df_m_vals_filtered)
print(head(df_beta_vals_filtered))
print(nrow(df_beta_vals_filtered))


# store filtered beta and m values as TXT ==> will be used to classify data
write.table(df_beta_vals_filtered,
            file = "classifying_data/artistic_study_filt-beta-values_0622_10threshold_mtDNA.txt",
            col.names = TRUE, sep = ";", row.names = TRUE)
 