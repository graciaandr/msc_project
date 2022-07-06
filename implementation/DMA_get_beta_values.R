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

setwd("/data/home/bt211038/msc_project/")

metadata = read.csv("artistic_trial/Masterfile_groups-MSc-project.csv", sep = ";")
colnames(metadata)[c(1,3)] = c("lab_no", "CIN.type")
sampleids = as.list(as.character(metadata$lab_no))
print(metadata %>% head(5))

## calcDiff object
calcdiffmeth <- readRDS("artistic_trial/calculateDiffMeth_object.rds")
df_methDiff = methylKit::getData(calcdiffmeth)

### load df beta values before filtering for DMPs

df_beta_vals_filt_10 = readRDS("classifying_data/artistic_study_betas_b4_EDMR_10threshold.rds")
df_beta_vals_filt_25 = readRDS("classifying_data/artistic_study_betas_b4_EDMR_25threshold.rds")
df_beta_vals_filt_50 = readRDS("classifying_data/artistic_study_betas_b4_EDMR_50threshold.rds")


### filter calcdiffmeth so that only keeps rows from cleaned beta values
calcdiffmeth2_10 = NULL
calcdiffmeth2_25 = NULL
calcdiffmeth2_50 = NULL
df_tmp_10 = NULL
df_tmp_25 = NULL
df_tmp_50 = NULL

#for (i in (1:nrow(df_beta_vals_filt_10))) {
for (i in (1:10)) {
  df_tmp_10 = df_methDiff %>%
    dplyr::filter(start <= df_beta_vals_filt_10$pos[[i]] & end >= df_beta_vals_filt_10$pos[[i]] & chr == df_beta_vals_filt_10$chrom[[i]])
  calcdiffmeth2_10 = rbind(calcdiffmeth2_10, df_tmp_10)
}

print(calcdiffmeth2_10)
stoppppp

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
        