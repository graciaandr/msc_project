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


metadata = read.csv("artistic_trial/Masterfile_groups-MSc-project.csv", sep = ";")
colnames(metadata)[c(1,3)] = c("lab_no", "CIN.type")
sampleids = as.list(as.character(metadata$lab_no))
treatments = as.vector(as.numeric(as.factor(metadata$CIN.type)))
covariates = data.frame(hpv = as.factor(metadata$HPV.type), age = as.numeric(metadata$age))
print(metadata %>% head(5))

df_meth <- read.table("classifying_data/artistic_study_df_meth.txt.gz", header=T, quote="\"", sep=";")
print("df meth has this many rows:")
print(nrow(df_meth))

myDiff <- readRDS("artistic_trial/calculateDiffMeth_object.txt")
print(nrow(myDiff))

df_beta_vals <- read.table("classifying_data/artistic_study_initial_beta_values.txt", header=T, sep=";")
print(head(df_beta_vals))

# ## Actual methylation values for each samples:
# mat = methylKit::percMethylation(df_meth, rowids = TRUE )
# beta_values = mat/100
# head(beta_values)

## try thresholds of 10%, 25% and 50% for cpgs to keep per conditions
## Removing NAs: function to remove rows with n many NAs in that row -- default n is 1 
rows_to_delete_NAs <- function(df, metadata,  p = 0.25) {
  n = round(p * nrow(df))
  col_indeces_Ctrl = which(grepl( "Control" , metadata$Phenotype.RRBS ) )
  col_indeces_Case = which(grepl( "Case" , metadata$Phenotype.RRBS ) )
  df_ctrl <- df[ , col_indeces_Ctrl ]
  df_cases <- df[ , col_indeces_Case ]
  print(dim(df_ctrl))
  print(dim(df_cases))
  row_indeces_NAs <- c()
  for (i in (1:nrow(df))) {
    no_of_NAs_in_ctrls = sum(is.na(df_ctrl[i,]))  
    no_of_NAs_in_cases = sum(is.na(df_cases[i,]))  
    if ( no_of_NAs_in_ctrls == no_of_NAs_in_cases & no_of_NAs_in_cases > n){
      row_indeces_NAs <- c(row_indeces_NAs, i)
    }
  }
  if (length(row_indeces_NAs) == 0) {
    row_indeces_NAs = NULL
    return(row_indeces_NAs)
  }
  return(row_indeces_NAs)
}



# remove all unnecessary rows
row_indeces_NAs = rows_to_delete_NAs(df_beta_vals, metadata, p = 0.001)
print(row_indeces_NAs)
df_beta_vals_filt = df_beta_vals[-row_indeces_NAs,]
df_meth_new =  df_meth[-row_indeces_NAs,]
myDiff2 = myDiff[-row_indeces_NAs,]

print("#Rows of df beta vals after NA handeling: ")
print(nrow(df_beta_vals_filt))
# write.table(df_beta_vals_filt,
#             file = "/data/home/bt211038/msc_project//classifying_data/artistic_study_betas_b4_filtering.txt",
#             col.names = TRUE, sep = ";", row.names = TRUE)


## Q Value adjustment
adj_q_vals = p.adjust(myDiff2$pvalue, method = "BH")
myDiff2$qvalue = adj_q_vals
df_adjusted_diff_meth = myDiff2


## EDMR: calculate all DMRs candidate from complete myDiff dataframe
dm_regions=edmr(myDiff = df_adjusted_diff_meth, mode=2, ACF=TRUE, DMC.qvalue = 0.5, plot = TRUE)
print("EDMR:")
df_dmrs = data.frame(dm_regions)
nrow(df_dmrs)
write.table(df_dmrs,
            file = "/data/home/bt211038/msc_project/classifying_data/all_DMRs.txt",
            col.names = TRUE, sep = ";", row.names = TRUE)


### remove droplist CpGs
df_bed_file <- as.data.frame(read.table("/data/home/bt211038/msc_project/bed_file/hg19-blacklist.v2.bed",header = FALSE, sep="\t",stringsAsFactors=FALSE, quote=""))
colnames(df_bed_file) <- c("chromosome", "start", "end", "info")
head(df_bed_file)
df_dmrs_false_cpgs = NULL

for (i in (1:length(df_bed_file$start))) {
  df_tmp = df_dmrs %>%
    dplyr::filter(start >= df_bed_file$start[[i]] & end <= df_bed_file$end[[i]] & seqnames != df_bed_file$chromosome[[i]])
  df_dmrs_false_cpgs = rbind(df_dmrs_false_cpgs, df_tmp)
}

print(nrow(df_dmrs_false_cpgs))

## retrieve the valid CpG regions
df_valid_cpg_regions = setdiff(df_dmrs,df_dmrs_false_cpgs)
print( nrow(df_valid_cpg_regions))

write.table(df_valid_cpg_regions,
            file = "/data/home/bt211038/msc_project/classifying_data/validated_DMRs.txt",
            col.names = TRUE, sep = ";", row.names = TRUE)


## for loop that goes through the start pos, end pos, and seqnames per row in beta/m value dataframe and DMR data
## to retrieve sig. diff. meth. CpG sites in DMRs
df_tmp1 = data.frame(matrix(NA, nrow = 1, ncol = ncol(df_beta_vals_filt)))
# df_tmp2 = data.frame(matrix(NA, nrow = 1, ncol = ncol(df_m_vals)))
colnames(df_tmp1) <- colnames((df_beta_vals_filt))
# colnames(df_tmp2) <- colnames((df_m_vals))
df_beta_vals_filtered = NULL
# df_m_vals_filtered = NULL
for (i in (1:length(df_valid_cpg_regions$start))) {
  df_tmp1 = df_beta_vals_filt %>%
    filter(pos >= df_valid_cpg_regions$start[[i]] & pos <= df_valid_cpg_regions$end[[i]] & chr == df_valid_cpg_regions$seqnames[[i]])
  #   df_tmp2 = df_m_vals %>%
  # filter(pos >= df_dmrs$start[[i]] & pos <= df_dmrs$end[[i]] & chr == df_dmrs$seqnames[[i]])
  
  df_beta_vals_filtered = rbind(df_beta_vals_filtered, df_tmp1)
  #   df_m_vals_filtered = rbind(df_m_vals_filtered, df_tmp2)
}

# print(df_m_vals_filtered)
print(head(df_beta_vals_filtered))
print(nrow(df_beta_vals_filtered))


# # ## Gene Annotation with annotatr 
# # ### use Bioconductor package *annotatr*: https://bioconductor.org/packages/release/bioc/html/annotatr.html
# # ### https://bioconductor.org/packages/release/bioc/vignettes/annotatr/inst/doc/annotatr-vignette.html
# # 
# # annots = c('hg19_cpgs', 'hg19_basicgenes', 'hg19_genes_intergenic',
# #            'hg19_genes_intronexonboundaries')
# # 
# # # Build the annotations (a single GRanges object)
# # annotations = build_annotations(genome = 'hg19', annotations = annots)
# # 
# # # Intersect the regions we read in with the annotations
# # dm_annotated = annotate_regions(
# #   regions = dm_regions,
# #   annotations = annotations,
# #   ignore.strand = TRUE,
# #   quiet = FALSE)
# # A GRanges object is returned
# # print(dm_annotated)


# store filtered beta and m values as TXT ==> will be used to classify data
write.table(df_beta_vals_filtered,
            file = "/data/home/bt211038/msc_project/classifying_data/artistic_study_filt-beta-values_08062022.txt",
            col.names = TRUE, sep = ";", row.names = TRUE)
 
# # write.table(df_m_vals_filtered,
# #             file = "C:/Users/andri/Documents/Uni London/QMUL/SemesterB/Masters_project/msc_project/data/classifying_data/CLL_study_filt-m-values.txt",
# #             col.names = TRUE, sep = ";", row.names = TRUE)
# # 
# # 
# # # t(df_beta_vals_filtered) %>% as.data.frame() %>% rownames()