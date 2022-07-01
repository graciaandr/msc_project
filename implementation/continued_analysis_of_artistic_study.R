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


metadata = read.csv("artistic_trial/Masterfile_groups-MSc-project.csv", sep = ";")
colnames(metadata)[c(1,3)] = c("lab_no", "CIN.type")
sampleids = as.list(as.character(metadata$lab_no))
treatments = as.vector(as.numeric(as.factor(metadata$CIN.type)))
covariates = data.frame(hpv = as.factor(metadata$HPV.type), age = as.numeric(metadata$age))
print(metadata %>% head(5))


calcdiffmeth <- readRDS("artistic_trial/calculateDiffMeth_object.rds")
df_methDiff = getData(calcdiffmeth)
print("calcdiffmeth has this many rows:")
print(nrow(calcdiffmeth))
print(head(calcdiffmeth))

df_beta_vals <- read.table("classifying_data/artistic_study_initial_beta_values.txt", header=T, sep=";")
# only look at mtDNA --> remove other chromosomes
# df_beta_vals = df_beta_vals %>% filter(chrom == "chrM")
print(head(df_beta_vals))
print(nrow(df_beta_vals))

### remove droplist CpGs in beta values
df_bed_file <- as.data.frame(read.table("bed_file/hg19-blacklist.v2.bed",header = FALSE, sep="\t",stringsAsFactors=FALSE, quote=""))
colnames(df_bed_file) <- c("chromosome", "start", "end", "info")
print("bed file:")
head(df_bed_file)

print("filtering of droplist CpGs from CpGs:")
df_dmrs_false_cpgs = NULL

### filter droplist CpGs from CpGs and calcDiff object (calcdiffmeth)
for (i in (1:length(df_bed_file$start))) {
  df_tmp = df_beta_vals %>%
    dplyr::filter(pos >= df_bed_file$start[[i]] & pos <= df_bed_file$end[[i]] & chrom != df_bed_file$chromosome[[i]])
  df_dmrs_false_cpgs = rbind(df_dmrs_false_cpgs, df_tmp)
}

## retrieve the valid CpGs
df_valid_cpgs = setdiff(df_beta_vals,df_dmrs_false_cpgs)
print( nrow(df_valid_cpgs))

### save valid CpGs and DMCs
write.table(df_valid_cpgs,
            file = "classifying_data/df_valid_cpgs.txt",
            col.names = TRUE, sep = ";", row.names = TRUE)


# remove all unnecessary rows
print("removal of rows with too many NAs (10% threshold) ")
indeces_to_keep = keep_rows(df = df_valid_cpgs, metadata, perc = 0.1)
df_beta_vals_filt = df_valid_cpgs[indeces_to_keep,] # clean beta values

print("#Rows of df beta vals after NA handeling: ")
print(nrow(df_beta_vals_filt))
write.table(df_beta_vals_filt,
             file = "classifying_data/artistic_study_betas_b4_EDMR_10threshold_mtDNA.txt",
             col.names = TRUE, sep = ";", row.names = TRUE)

### filter calcdiffmeth so that only keeps rows from cleaned beta values
calcdiffmeth2 = NULL
# for (i in (1:length(df_beta_vals$start))) {
for (i in (1:500)) { # to check if it works
  df_tmp = df_methDiff %>%
    dplyr::filter(start <= df_beta_vals$pos[[i]] & end >= df_beta_vals$pos[[i]] & chr == df_beta_vals$chrom[[i]])
  calcdiffmeth2 = rbind(calcdiffmeth2, df_tmp)
}

calcdiffmeth2 = calcdiffmeth2 %>% distinct()
nrow(calcdiffmeth2)

## Q Value adjustment of filtered calc Meth Diff Object
calcdiffmeth2$qvalue = p.adjust(calcdiffmeth2$pvalue, method = "BH")

# filter calcdiffmeth for qvalue of 0.05 and methylationDifference of 10 
# df_adjusted_diff_meth = methylKit::getMethylDiff(qvalue = 0.05, difference = 10)
df_DMPs = calcdiffmeth2 %>% filter(qvalue < 0.05 & abs(meth.diff) >= 10)

print("df_DMPs has these dimensions:")
print(dim(df_DMPs))
write.table(df_DMPs,
            file = "classifying_data/df_adjusted_diff_meth.txt",
            col.names = TRUE, sep = ";", row.names = TRUE)


## for loop that goes through the start pos, end pos, and seqnames per row in beta/m value dataframe and DMR data
## to retrieve CpG sites that are DMPs
df_tmp1 = data.frame(matrix(NA, nrow = 1, ncol = ncol(df_beta_vals_filt)))
colnames(df_tmp1) <- colnames((df_beta_vals_filt))
df_beta_vals_filtered = NULL
for (i in (1:length(df_DMPs$start))) {
  df_tmp1 = df_beta_vals_filt %>%
    dplyr::filter(pos >= df_DMPs$start[[i]] & pos <= df_DMPs$end[[i]] & chrom == df_DMPs$chr[[i]])
  df_beta_vals_filtered = rbind(df_beta_vals_filtered, df_tmp1)
}

print(head(df_beta_vals_filtered))
print(nrow(df_beta_vals_filtered))
# store filtered beta and m values as TXT ==> will be used to classify data
write.table(df_beta_vals_filtered,
            file = "classifying_data/artistic_study_filt-beta-values_0622_10threshold_mtDNA.txt",
            col.names = TRUE, sep = ";", row.names = TRUE)

## EDMR: calculate all DMRs candidate from complete calcdiffmeth dataframe
print("DMR Analysis:")
dm_regions=edmr(myDiff = calcdiffmeth2, mode=2, ACF=TRUE, DMC.qvalue = 0.05, plot = FALSE)
df_dmrs = data.frame(dm_regions)
print(nrow(df_dmrs))
write.table(df_dmrs,
            file = "classifying_data/DMRs.txt",
            col.names = TRUE, sep = ";", row.names = TRUE)

 