library(methylKit)
library(GenomicRanges)
library(IRanges)
library(devtools)
library(edmr)
library(mixtools)
library(data.table)
library(magrittr)
library(dplyr)
library(annotatr)
# library(lumi)

setwd("/data/home/bt211038/msc_project/")
# setwd("C:/Users/andri/Documents/Uni London/QMUL/SemesterB/Masters_project/msc_project/data/")

## try thresholds of 10%, 25% and 50% for cpgs to keep per conditions
## Removing NAs: function to keep rows with n many NAs in that row -- default n is 0.5 
keep_rows <- function(df, metadata,  perc = 0.5) {
  col_indeces_Ctrl = which(grepl( "Control" , metadata$Phenotype.RRBS ) )
  col_indeces_Case = which(grepl( "Case" , metadata$Phenotype.RRBS ) )
  df_ctrl <- df[ , col_indeces_Ctrl ]
  df_cases <- df[ , col_indeces_Case ]
  samp_per_cpg_ctrl = round((perc) * length(col_indeces_Ctrl))
  samp_per_cpg_cases = round((perc) * length(col_indeces_Case))
  print(samp_per_cpg_ctrl)
  print(samp_per_cpg_cases)
  # print(dim(df_ctrl))
  # print(dim(df_cases))
  row_indeces_NAs <- c()
  keep_in_ctrl <- which(rowSums(!is.na(df_ctrl)) >= samp_per_cpg_ctrl)
  keep_in_cases <- which(rowSums(!is.na(df_cases)) >= samp_per_cpg_cases)
  tmp <- c(keep_in_ctrl, keep_in_cases)
  indeces = unique(sort(tmp))
  if (length(indeces) == 0) {
    indeces = NULL
    return(indeces)
  }
  return(indeces)
}

# meta data about study, sample id, disease type etc
metadata = read.csv("artistic_trial/global_masterfile_Artistic_Trial.csv", sep = ";")
colnames(metadata)[c(7)] = c("CIN.type")
metadata = metadata %>% dplyr::filter(
  (metadata$cytology == "Negative" & metadata$Phenotype.RRBS == "Control") |
    (metadata$Histology == "CIN 3" | metadata$Histology == "CIN 3/Ca in situ") )
print(nrow(metadata))
print(metadata %>% tail(5))

# load calculated methylation differences object
calcdiffmeth <- readRDS("classifying_data/calculateDiffMeth_object_negCTRLvsCIN3.rds")
# calcdiffmeth <- readRDS("artistic_trial//calculateDiffMeth_object.rds")
df_methDiff = methylKit::getData(calcdiffmeth)
df_methDiff$id <- paste(df_methDiff$chr, df_methDiff$start, df_methDiff$start, sep=".")

print("calcdiffmeth has this many rows:")
print(nrow(calcdiffmeth))
print(head(calcdiffmeth, 5))

# df_beta_vals <- read.table("artistic_trial//artistic_study_initial_beta_values.txt.gz", header=T, sep=";", nrows = 10000)
meth = readRDS("artistic_trial/df_meth_negCTRLvsCIN3.rds")
print(head(meth, 5))
mat = methylKit::percMethylation(meth, rowids = TRUE )
print(head(mat, 5))
beta_values = mat/100
df_beta_vals = data.frame(beta_values) %>% dplyr::mutate(pos = meth$start, chrom = meth$chr)
print(nrow(df_beta_vals))

### remove droplist CpGs in beta values
# df_bed_file <- as.data.frame(read.table("hg19-blacklist.v2.bed",header = FALSE, sep="\t",stringsAsFactors=FALSE, quote=""))
df_bed_file <- as.data.frame(read.table("bed_file/hg19-blacklist.v2.bed", header = FALSE, sep="\t",stringsAsFactors=FALSE, quote=""))
colnames(df_bed_file) <- c("chromosome", "start", "end", "info")
print(head(df_bed_file, 5))

### filter droplist CpGs from CpGs
print("filter droplist CpGs from CpGs")
df_methDiff_grObj <- makeGRangesFromDataFrame(df_methDiff[,1:4]) # only take columns with chr, start, end & strand, so columns 1 to 4
droplist_Regions <- data.frame(annotatr::annotate_regions(regions = df_methDiff_grObj, annotations = makeGRangesFromDataFrame(df_bed_file)))
droplist_Regions$id <- paste(droplist_Regions$seqnames, droplist_Regions$start, droplist_Regions$start, sep = ".")
keep <- which(rownames(df_beta_vals) %in% droplist_Regions$id)
df_valid_cpgs <- df_beta_vals[-keep,] ## retrieve the valid CpGs
dim(df_valid_cpgs)


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

saveRDS(df_beta_vals_filt_10, file = "classifying_data/negCTRLvsCIN3_artistic_study_betas_b4_EDMR_10threshold.rds")
saveRDS(df_beta_vals_filt_25, file = "classifying_data/negCTRLvsCIN3_artistic_study_betas_b4_EDMR_25threshold.rds")
saveRDS(df_beta_vals_filt_50, file = "classifying_data/negCTRLvsCIN3_artistic_study_betas_b4_EDMR_50threshold.rds")


### filter calcdiffmeth so that only keeps rows from cleaned beta values for different thresholds
keep <- which(df_methDiff$id %in%   rownames(df_beta_vals_filt_10))
calcdiffmeth2_10 <- df_methDiff[keep,]

keep <- which(df_methDiff$id %in%   rownames(df_beta_vals_filt_25))
calcdiffmeth2_25 <- df_methDiff[keep,]

keep <- which(df_methDiff$id %in%   rownames(df_beta_vals_filt_50))
calcdiffmeth2_50 <- df_methDiff[keep,]

print(nrow(calcdiffmeth2_10))
print(nrow(calcdiffmeth2_25))
print(nrow(calcdiffmeth2_50))


## Q Value adjustment of filtered calc Meth Diff Object
calcdiffmeth2_10$qvalue = p.adjust(calcdiffmeth2_10$pvalue, method = "BH")
saveRDS(calcdiffmeth2_10, file = "classifying_data/negCTRLvsCIN3_calcdiffmeth2_10.rds")

calcdiffmeth2_25$qvalue = p.adjust(calcdiffmeth2_25$pvalue, method = "BH")
saveRDS(calcdiffmeth2_25, file = "classifying_data/negCTRLvsCIN3_calcdiffmeth2_25.rds")

calcdiffmeth2_50$qvalue = p.adjust(calcdiffmeth2_50$pvalue, method = "BH")
saveRDS(calcdiffmeth2_50, file = "classifying_data/negCTRLvsCIN3_calcdiffmeth2_50.rds")

# filter calcdiffmeth for qvalue of 0.05 and methylationDifference of 10 
df_DMPs_10 = calcdiffmeth2_10 %>% filter(qvalue < 0.05 & abs(meth.diff) > 10)
df_DMPs_25 = calcdiffmeth2_25 %>% filter(qvalue < 0.05 & abs(meth.diff) > 10)
df_DMPs_50 = calcdiffmeth2_50 %>% filter(qvalue < 0.05 & abs(meth.diff) > 10)

print("df_DMPs has these dimensions (10%):")
print(dim(df_DMPs_10))

print("df_DMPs has these dimensions (25%):")
print(dim(df_DMPs_25))

print("df_DMPs has these dimensions (50%):")
print(dim(df_DMPs_50))
print(head(df_DMPs_50, 5))

write.table(df_DMPs_10,
            file = "classifying_data/df_DMPs_10.txt",
            col.names = TRUE, sep = ";", row.names = TRUE)
saveRDS(df_DMPs_10, file = "classifying_data/negCTRLvsCIN3_df_DMPs_10.rds")

write.table(df_DMPs_25,
            file = "classifying_data/df_DMPs_25",
            col.names = TRUE, sep = ";", row.names = TRUE)
saveRDS(df_DMPs_25, file = "classifying_data/negCTRLvsCIN3_df_DMPs_25.rds")

write.table(df_DMPs_50,
            file = "classifying_data/df_DMPs_50.txt",
            col.names = TRUE, sep = ";", row.names = TRUE)
saveRDS(df_DMPs_50, file = "classifying_data/negCTRLvsCIN3_df_DMPs_50.rds")


## retrieve beta values for CpG sites that are DMPs

df_DMPs_10$id <- paste(df_DMPs_10$chr,df_DMPs_10$start,df_DMPs_10$start, sep=".")
keep <- which(rownames(df_beta_vals_filt_10) %in% df_DMPs_10$id)
df_beta_vals_filtered_10 <- df_beta_vals_filt_10[keep,]
print(nrow(df_beta_vals_filtered_10))


df_DMPs_25$id <- paste(df_DMPs_25$chr, df_DMPs_25$start, df_DMPs_25$start, sep=".")
keep <- which(rownames(df_beta_vals_filt_25) %in% df_DMPs_25$id)
df_beta_vals_filtered_25 <- df_beta_vals_filt_25[keep,]
print(nrow(df_beta_vals_filtered_25))

df_DMPs_50$id <- paste(df_DMPs_50$chr, df_DMPs_50$start, df_DMPs_50$start, sep=".")
keep <- which(rownames(df_beta_vals_filt_50) %in% df_DMPs_50$id)
df_beta_vals_filtered_50 <- df_beta_vals_filt_50[keep,]
print(nrow(df_beta_vals_filtered_50))

# Assign disease state / group label for later filtering in classification & prediction
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
print(tail(df_beta_vals_filtered_10, 5))

df_beta_vals_filtered_25 <- cbind(df_ctrl_25, df_cases_25)
rownames(df_beta_vals_filtered_25)[nrow(df_beta_vals_filtered_25)] <- "Phenotype"
print(tail(df_beta_vals_filtered_25, 5))

df_beta_vals_filtered_50 <- cbind(df_ctrl_50, df_cases_50)
rownames(df_beta_vals_filtered_50)[nrow(df_beta_vals_filtered_50)] <- "Phenotype"
print(tail(df_beta_vals_filtered_50, 5))

# store filtered beta and m values as TXT ==> will be used to classify data
write.table(df_beta_vals_filtered_10,
            file = "classifying_data/negCTRLvsCIN3_artistic_study_filt-beta-values_0722_10threshold.txt",
            col.names = TRUE, sep = ";", row.names = TRUE)
write.table(df_beta_vals_filtered_25,
            file = "classifying_data/negCTRLvsCIN3_artistic_study_filt-beta-values_0722_25threshold.txt",
            col.names = TRUE, sep = ";", row.names = TRUE)
write.table(df_beta_vals_filtered_50,
            file = "classifying_data/negCTRLvsCIN3_artistic_study_filt-beta-values_0722_50threshold.txt",
            col.names = TRUE, sep = ";", row.names = TRUE)

saveRDS(df_beta_vals_filtered_10, file = "classifying_data/negCTRLvsCIN3_df_beta_vals_filtered_10.rds")
saveRDS(df_beta_vals_filtered_25, file = "classifying_data/negCTRLvsCIN3_df_beta_vals_filtered_25.rds")
saveRDS(df_beta_vals_filtered_50, file = "classifying_data/negCTRLvsCIN3_df_beta_vals_filtered_50.rds")

# ## EDMR: calculate all DMRs candidate from complete calcdiffmeth dataframe
# print("DMR Analysis:")
# dm_regions=edmr(myDiff = calcdiffmeth2, mode=2, ACF=TRUE, DMC.qvalue = 0.05, plot = FALSE)
# df_dmrs = data.frame(dm_regions)
# print(nrow(df_dmrs))
# write.table(df_dmrs,
#             file = "classifying_data/DMRs.txt",
#             col.names = TRUE, sep = ";", row.names = TRUE)
# saveRDS(df_dmrs, file = "classifying_data/DMRs.rds")

 