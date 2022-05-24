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
### Trimmed reads were then mapped to the in silico bisulfite converted human reference genome (GRCh38) 
### by using Bismark v.0.22.1 with and non directional parameter applied
### Genome_build: GRCh38
### The genome-wide cytosine methylation output file is tab-delimited in the following format: 
### <chromosome> <position> <strand> <count methylated> <count non-methylated> <C-context> <trinucleotide context>

setwd("C:/Users/andri/Documents/Uni London/QMUL/SemesterB/Masters_project/msc_project/data/CD4_Tcell_study/")
# setwd("/data/scratch/bt211038/msc_project/CD4_Tcell_study/")

## create file list
file.list = list.files(path = "C:/Users/andri/Documents/Uni London/QMUL/SemesterB/Masters_project/msc_project/data/CD4_Tcell_study", pattern= '*.txt$')
# file.list = list.files(path = "/data/scratch/bt211038/msc_project/CD4_Tcell_study/", pattern= '*.txt$')
list_of_files = as.list(file.list)
print(list_of_files)

start.time1 <- Sys.time()

## read files with methRead
myobj=methRead(location = list_of_files,
               sample.id =list("ctrl1","ctrl2", "ctrl3","ctrl4","ctrl5","ctrl6","ctrl7",
                               "case1","case2","case3","case4","case5", "case6","case7"),
               assembly ="hg38", # study used GrCh38 - hg38
               treatment = c(0,0,0,0,0,0,0,1,1,1,1,1,1, 1),
               context="CpG",
               header = TRUE, 
               pipeline = 'bismarkCytosineReport',
               resolution = "base",
               sep = '\t',
               dbdir = "CD4_Tcell_study/"
)

end.time1 <- Sys.time()
time.taken1 <- end.time1 - start.time1
print(time.taken1)

## Unite / merge samples, only keep CpGs that are methylated in at least 2 samples
start.time2 <- Sys.time()
meth=unite(myobj, destrand=FALSE, min.per.group = 2L) ## adjust min per group to see if i get more cpgs eventually
  # check parameter 'min.per.group' (want cpg in ALL samples incld. case/ctrl) -- no missing values since small pilot study
  # By default only regions/bases that are covered in all samples are united as methylBase object -- according to https://www.rdocumentation.org/packages/methylKit/versions/0.99.2/topics/unite
end.time2 <- Sys.time()
time.taken2 <- end.time2 - start.time2
print(time.taken2)

df_meth = data.frame(meth)
nrow(df_meth)

## cluster samples
#clusterSamples(meth, dist="correlation", method="ward.D2", plot=TRUE) 

## PCA plot
# PCASamples(meth, screeplot=TRUE)
# png("PCA_CD4_study.png")
# PCASamples(meth) 
# dev.off()

start.time3 <- Sys.time()
## Finding differentially methylated bases
# myDiff <- calculateDiffMeth(meth,
#                             overdispersion = "MN",
#                             effect         = "wmean",
#                             test           = "F",
#                             adjust         = 'BH',
#                             slim           = F,
#                             weighted.mean  = T)
# saveRDS(myDiff, file = "CD4_Tcell_study/calculateDiffMeth_object.txt")
myDiff <- readRDS(file = "C:/Users/andri/Documents/Uni London/QMUL/SemesterB/Masters_project/msc_project/data/calculateDiffMeth_object_060522.txt")


end.time3 <- Sys.time()
time.taken3 <- end.time3 - start.time3
print(time.taken3)

## show different methlyation patterns per Chromosome - Plot 
# png("diffMethPerChr.png")
# diffMethPerChr(myDiff_filtered, plot=TRUE,qvalue.cutoff=0.05, meth.cutoff=10)
# dev.off()

## Actual methylation values for each samples:
mat = percMethylation(meth, rowids = TRUE )
beta_values = mat/100
head(beta_values)

## Removing NAs: function to remove rows with n many NAs in that row -- default n is 1 
rows_to_delete_NAs <- function(df, n=1) {
  df_ctrl <- df[ , grepl( "ctrl" , colnames( df ) ) ]
  df_cases <- df[ , grepl( "case" , colnames( df ) ) ]
  row_indeces_NAs <- c()
  for (i in (1:nrow(df))) {
    no_of_NAs_in_ctrls = sum(is.na(df_ctrl[i,]))  
    no_of_NAs_in_cases = sum(is.na(df_cases[i,]))  
    # cat("NAs in ctrl: ", no_of_NAs_in_ctrls, "\n")
    # cat("NAs in cases: ",no_of_NAs_in_cases, "\n")
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

## add postions as own column to beta and m value data frames ==> for fitering & eventually classifier training
df_beta_vals = data.frame(beta_values) %>% dplyr::mutate(pos = df_meth$start, chrom = df_meth$chr) ###
df_beta_vals[order(df_beta_vals$pos),]

# df_m_vals = data.frame(m_values) %>% dplyr::mutate(pos = df_meth$start, chrom = df_meth$chr)
# df_m_vals[order(df_m_vals$pos),]

## add chr in front of all chromosome names to be able to compare to seqnmaes in DMR dataframe later on when filtering
df_beta_vals['chr'] = paste0('chr', df_beta_vals$chrom)
# df_m_vals['chr'] = paste0('chr', df_m_vals$chrom)


# remove all unnecessary rows
row_indeces_NAs = rows_to_delete_NAs(df_beta_vals, 2)
df_beta_vals_filt = df_beta_vals[-row_indeces_NAs,]
meth_new =  meth[-row_indeces_NAs,]
myDiff2 = myDiff[-row_indeces_NAs,]

## Q Value adjustment

## adjust p values --> will become new Q VALUES !!!!
adj_q_vals = p.adjust(myDiff2$pvalue, method = "BH")
myDiff2$qvalue = adj_q_vals
###
df_adjusted_diff_meth = myDiff2
###


# write.table(df_adjusted_diff_meth,
#             file = "/classifying_data/adjusted_myDiff_df.txt",
#             col.names = TRUE, sep = ";", row.names = TRUE)
# 
# write.table(df_beta_vals_filt,
#             file = "/classifying_data/df_beta_vals_filt.txt",
#             col.names = TRUE, sep = ";", row.names = TRUE)
# 
# write.table(meth_new,
#             file = "/classifying_data/adjusted_methylation_df.txt",
#             col.names = TRUE, sep = ";", row.names = TRUE)


## EDMR: calculate all DMRs candidate from complete myDiff dataframe
dm_regions=edmr(myDiff = df_adjusted_diff_meth, mode=2, ACF=TRUE, DMC.qvalue = 0.75, plot = TRUE)
# dm_regions
df_dmrs = data.frame(dm_regions)
nrow(df_dmrs)

### remove droplist CpGs
df_bed_file <- as.data.frame(read.table("../hg19-blacklist.v2.bed",header = FALSE, sep="\t",stringsAsFactors=FALSE, quote=""))
colnames(df_bed_file) <- c("chromosome", "start", "end", "info")
head(df_bed_file)
df_dmrs_upd = NULL
head(df_dmrs)

for (i in (1:length(df_dmrs$start))) {
  df_tmp1 = df_dmrs_upd %>%
    filter(pos >= df_bed_file$end[[i]] & pos <= df_bed_file$start[[i]] & chr != df_bed_file$chromosome[[i]])
  df_dmrs_upd = rbind(df_dmrs_upd, df_tmp1)
}

head(df_dmrs_upd)
#### HIER #####



## for loop that goes through the start pos, end pos, and seqnames per row in beta/m value dataframe and DMR data
## to retrieve sig. diff. meth. CpG sites in DMRs
df_tmp1 = data.frame(matrix(NA, nrow = 1, ncol = ncol(df_beta_vals_filt)))
# df_tmp2 = data.frame(matrix(NA, nrow = 1, ncol = ncol(df_m_vals)))
colnames(df_tmp1) <- colnames((df_beta_vals_filt))
# colnames(df_tmp2) <- colnames((df_m_vals))
df_beta_vals_filtered = NULL
# df_m_vals_filtered = NULL
for (i in (1:length(df_dmrs$start))) {
  df_tmp1 = df_beta_vals_filt %>%
    filter(pos >= df_dmrs$start[[i]] & pos <= df_dmrs$end[[i]] & chr == df_dmrs$seqnames[[i]])
  #   df_tmp2 = df_m_vals %>%
  # filter(pos >= df_dmrs$start[[i]] & pos <= df_dmrs$end[[i]] & chr == df_dmrs$seqnames[[i]])
  
  df_beta_vals_filtered = rbind(df_beta_vals_filtered, df_tmp1)
  #   df_m_vals_filtered = rbind(df_m_vals_filtered, df_tmp2)
}

# print(df_m_vals_filtered)
print(head(df_beta_vals_filtered))
print(nrow(df_beta_vals_filtered))


# store filtered beta and m values as TXT ==> will be used to classify data
# write.table(df_beta_vals_filtered,
#             file = "../classifying_data/filt-beta-values.txt",
#             col.names = TRUE, sep = ";", row.names = TRUE)
