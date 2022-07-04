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
  samp_per_cpg = round((perc) * 120)
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
df_methDiff = methylKit::getData(calcdiffmeth)
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
df_valid_cpgs = setdiff(df_beta_vals, df_dmrs_false_cpgs)
print(nrow(df_valid_cpgs))

### save valid CpGs
write.table(df_valid_cpgs,
            file = "classifying_data/df_valid_cpgs.txt",
            col.names = TRUE, sep = ";", row.names = TRUE)
saveRDS(df_valid_cpgs, file = "df_valid_cpgs.rds")


# remove all unnecessary rows
print("removal of rows with too many NAs")
indeces_to_keep_10 = keep_rows(df = df_valid_cpgs, metadata, perc = 0.1)
indeces_to_keep_25 = keep_rows(df = df_valid_cpgs, metadata, perc = 0.25)
indeces_to_keep_50 = keep_rows(df = df_valid_cpgs, metadata, perc = 0.5)

df_beta_vals_filt_10 = df_valid_cpgs[indeces_to_keep_10,] # clean beta values for 10% thrreshold
df_beta_vals_filt_25 = df_valid_cpgs[indeces_to_keep_25,] # clean beta values for 25% thrreshold
df_beta_vals_filt_50 = df_valid_cpgs[indeces_to_keep_50,] # clean beta values for 50% thrreshold

print("#Rows of df beta vals after NA handeling: ")
print(nrow(df_beta_vals_filt_10))
print(nrow(df_beta_vals_filt_25))
print(nrow(df_beta_vals_filt_50))

write.table(df_beta_vals_filt_10,
             file = "classifying_data/artistic_study_betas_b4_EDMR_10threshold.txt",
             col.names = TRUE, sep = ";", row.names = TRUE)
saveRDS(df_beta_vals_filt_10, file = "classifying_data/artistic_study_betas_b4_EDMR_10threshold.rds")

write.table(df_beta_vals_filt_25,
            file = "classifying_data/artistic_study_betas_b4_EDMR_25threshold.txt",
            col.names = TRUE, sep = ";", row.names = TRUE)
saveRDS(df_beta_vals_filt_25, file = "classifying_data/artistic_study_betas_b4_EDMR_25threshold.rds")

write.table(df_beta_vals_filt_50,
            file = "classifying_data/artistic_study_betas_b4_EDMR_50threshold.txt",
            col.names = TRUE, sep = ";", row.names = TRUE)
saveRDS(df_beta_vals_filt_50, file = "classifying_data/artistic_study_betas_b4_EDMR_50threshold.rds")


### filter calcdiffmeth so that only keeps rows from cleaned beta values
calcdiffmeth2_10 = NULL
calcdiffmeth2_25 = NULL
calcdiffmeth2_50 = NULL
df_tmp_10 = NULL
df_tmp_25 = NULL
df_tmp_50 = NULL

for (i in (1:nrow(df_beta_vals_filt_10))) {
  df_tmp_10 = df_methDiff %>%
    dplyr::filter(start <= df_beta_vals_filt_10$pos[[i]] & end >= df_beta_vals_filt_10$pos[[i]] & chr == df_beta_vals_filt_10$chrom[[i]])
  calcdiffmeth2_10 = rbind(calcdiffmeth2_10, df_tmp_10)
}

for (i in (1:nrow(df_beta_vals_filt_25))) {
  df_tmp_25 = df_methDiff %>%
    dplyr::filter(start <= df_beta_vals_filt_25$pos[[i]] & end >= df_beta_vals_filt_25$pos[[i]] & chr == df_beta_vals_filt_25$chrom[[i]])
  calcdiffmeth2_25 = rbind(calcdiffmeth2_25, df_tmp_25)
}

for (i in (1:nrow(df_beta_vals_filt_50))) {
  df_tmp_50 = df_methDiff %>%
    dplyr::filter(start <= df_beta_vals_filt_50$pos[[i]] & end >= df_beta_vals_filt_50$pos[[i]] & chr == df_beta_vals_filt_50$chrom[[i]])
  calcdiffmeth2_50 = rbind(calcdiffmeth2_50, df_tmp_50)
}

calcdiffmeth2_10 = calcdiffmeth2_10 %>% distinct()
print("calcdiffmeth2 has these dimensions (10%):")
print(nrow(calcdiffmeth2_10))

## Q Value adjustment of filtered calc Meth Diff Object
calcdiffmeth2_10$qvalue = p.adjust(calcdiffmeth2_10$pvalue, method = "BH")
saveRDS(calcdiffmeth2_10, file = "classifying_data/calcdiffmeth2_10.rds")

calcdiffmeth2_25$qvalue = p.adjust(calcdiffmeth2_25$pvalue, method = "BH")
saveRDS(calcdiffmeth2_25, file = "classifying_data/calcdiffmeth2_25.rds")

calcdiffmeth2_50$qvalue = p.adjust(calcdiffmeth2_50$pvalue, method = "BH")
saveRDS(calcdiffmeth2_50, file = "classifying_data/calcdiffmeth2_50.rds")

# filter calcdiffmeth for qvalue of 0.05 and methylationDifference of 10 
# df_adjusted_diff_meth = methylKit::getMethylDiff(qvalue = 0.05, difference = 10)
df_DMPs_10 = calcdiffmeth2_10 %>% filter(qvalue < 0.05 & abs(meth.diff) >= 10)
df_DMPs_25 = calcdiffmeth2_25 %>% filter(qvalue < 0.05 & abs(meth.diff) >= 10)
df_DMPs_50 = calcdiffmeth2_50 %>% filter(qvalue < 0.05 & abs(meth.diff) >= 10)

print("df_DMPs has these dimensions (10%):")
print(dim(df_DMPs_10))

write.table(df_DMPs_10,
            file = "classifying_data/df_DMPs_10.txt",
            col.names = TRUE, sep = ";", row.names = TRUE)
saveRDS(df_DMPs_10, file = "classifying_data/df_DMPs_10.rds")

write.table(df_DMPs_25,
            file = "classifying_data/df_DMPs_25",
            col.names = TRUE, sep = ";", row.names = TRUE)
saveRDS(df_DMPs_25, file = "classifying_data/df_DMPs_25.rds")

write.table(df_DMPs_50,
            file = "classifying_data/df_DMPs_50.txt",
            col.names = TRUE, sep = ";", row.names = TRUE)
saveRDS(df_DMPs_50, file = "classifying_data/df_DMPs_50.rds")


## for loop that goes through the start pos, end pos, and seqnames per row in beta/m value dataframe and DMR data
## to retrieve CpG sites that are DMPs
df_tmp10 = data.frame(matrix(NA, nrow = 1, ncol = ncol(df_beta_vals_filt_10)))
df_tmp25 = data.frame(matrix(NA, nrow = 1, ncol = ncol(df_beta_vals_filt_25)))
df_tmp50 = data.frame(matrix(NA, nrow = 1, ncol = ncol(df_beta_vals_filt_50)))

colnames(df_tmp10) <- colnames((df_beta_vals_filt_10))
colnames(df_tmp25) <- colnames((df_beta_vals_filt_25))
colnames(df_tmp50) <- colnames((df_beta_vals_filt_50))

df_beta_vals_filtered_10 = NULL
df_beta_vals_filtered_25 = NULL
df_beta_vals_filtered_50 = NULL

for (i in (1:length(df_DMPs_10$start))) {
  df_tmp10 = df_beta_vals_filt_10 %>%
    dplyr::filter(pos >= df_DMPs_10$start[[i]] & pos <= df_DMPs_10$end[[i]] & chrom == df_DMPs_10$chr[[i]])
  df_beta_vals_filtered_10 = rbind(df_beta_vals_filtered_10, df_tmp10)
}
print(head(df_beta_vals_filtered_10))
print(nrow(df_beta_vals_filtered_10))

for (i in (1:length(df_DMPs_25$start))) {
  df_tmp25 = df_beta_vals_filt_25 %>%
    dplyr::filter(pos >= df_DMPs_25$start[[i]] & pos <= df_DMPs_25$end[[i]] & chrom == df_DMPs_25$chr[[i]])
  df_beta_vals_filtered_25 = rbind(df_beta_vals_filtered_25, df_tmp25)
}
print(head(df_beta_vals_filtered_25))
print(nrow(df_beta_vals_filtered_25))

for (i in (1:length(df_DMPs_50$start))) {
  df_tmp50 = df_beta_vals_filt_50 %>%
    dplyr::filter(pos >= df_DMPs_50$start[[i]] & pos <= df_DMPs_50$end[[i]] & chrom == df_DMPs_50$chr[[i]])
  df_beta_vals_filtered_50 = rbind(df_beta_vals_filtered_50, df_tmp50)
}
print(head(df_beta_vals_filtered_50))
print(nrow(df_beta_vals_filtered_50))

col_indeces_Ctrl = which(grepl( "Control" , metadata$Phenotype.RRBS ) )
col_indeces_Case = which(grepl( "Case" , metadata$Phenotype.RRBS ) )

df_ctrl_10 <- df_beta_vals_filtered_10[ , col_indeces_Ctrl ]
df_ctrl_10[nrow(df_ctrl_10) + 1,] <- paste0("Control", (1:120))
df_cases_10 <- df_beta_vals_filtered_10[ , col_indeces_Case ]
df_cases_10[nrow(df_cases_10) + 1,] <- paste0("Case", (1:120))

df_ctrl_25 <- df_beta_vals_filtered_25[ , col_indeces_Ctrl ]
df_ctrl_25[nrow(df_ctrl_25) + 1,] <- paste0("Control", (1:120))
df_cases_25 <- df_beta_vals_filtered_25[ , col_indeces_Case ]
df_cases_25[nrow(df_cases_25) + 1,] <- paste0("Case", (1:120))

df_ctrl_50 <- df_beta_vals_filtered_50[ , col_indeces_Ctrl ]
df_ctrl_50[nrow(df_ctrl_50) + 1,] <- paste0("Control", (1:120))
df_cases_50 <- df_beta_vals_filtered_50[ , col_indeces_Case ]
df_cases_50[nrow(df_cases_50) + 1,] <- paste0("Case", (1:120))

df_beta_vals_filtered_10 <- cbind(df_ctrl_10, df_cases_10)
rownames(df_beta_vals_filtered_10)[nrow(df_beta_vals_filtered_10)] <- "Phenotype"

df_beta_vals_filtered_50 <- cbind(df_ctrl_25, df_cases_25)
rownames(df_beta_vals_filtered_25)[nrow(df_beta_vals_filtered_50)] <- "Phenotype"

df_beta_vals_filtered_50 <- cbind(df_ctrl_50, df_cases_50)
rownames(df_beta_vals_filtered_50)[nrow(df_beta_vals_filtered_50)] <- "Phenotype"

# store filtered beta and m values as TXT ==> will be used to classify data
write.table(df_beta_vals_filtered_10,
            file = "classifying_data/artistic_study_filt-beta-values_0722_10threshold.txt",
            col.names = TRUE, sep = ";", row.names = TRUE)
write.table(df_beta_vals_filtered_25,
            file = "classifying_data/artistic_study_filt-beta-values_0722_25threshold.txt",
            col.names = TRUE, sep = ";", row.names = TRUE)
write.table(df_beta_vals_filtered_50,
            file = "classifying_data/artistic_study_filt-beta-values_0722_50threshold.txt",
            col.names = TRUE, sep = ";", row.names = TRUE)

saveRDS(df_beta_vals_filtered_10, file = "classifying_data/df_beta_vals_filtered_10.rds")
saveRDS(df_beta_vals_filtered_25, file = "classifying_data/df_beta_vals_filtered_25.rds")
saveRDS(df_beta_vals_filtered_50, file = "classifying_data/df_beta_vals_filtered_50.rds")

# ## EDMR: calculate all DMRs candidate from complete calcdiffmeth dataframe
# print("DMR Analysis:")
# dm_regions=edmr(myDiff = calcdiffmeth2, mode=2, ACF=TRUE, DMC.qvalue = 0.05, plot = FALSE)
# df_dmrs = data.frame(dm_regions)
# print(nrow(df_dmrs))
# write.table(df_dmrs,
#             file = "classifying_data/DMRs.txt",
#             col.names = TRUE, sep = ";", row.names = TRUE)
# saveRDS(df_dmrs, file = "classifying_data/DMRs.rds")

 