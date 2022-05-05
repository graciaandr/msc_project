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
### chrBase	chr	base	strand	coverage	freqC	freqT
# setwd("C:/Users/andri/Documents/Uni London/QMUL/SemesterB/Masters_project/msc_project/data/CLL_study/")
setwd("/data/scratch/bt211038/msc_project/CLL_study/")


## create file list
# file.list = list.files(path = "C:/Users/andri/Documents/Uni London/QMUL/SemesterB/Masters_project/msc_project/data/CLL_study", pattern= '*.txt$')
file.list = list.files(path = "CLL_study/", pattern= '*.txt$')
list_of_files = as.list(file.list)
print(list_of_files)

vec_ctrl = rep(0, 355)
vec_treatment = rep(1, 96)
vec_label = c(vec_ctrl, vec_treatment)

id_ctrl = rep("ctrl", 355)
id_no_ctrl = (1:355)

id_treatment = rep("case", 96)
id_no_trt = (1:96)

sampleids = c(paste0(id_ctrl, id_no_ctrl), paste0(id_treatment, id_no_trt) )

print(sampleids)

print(vec_label)

## Differential Methylation Analysis
start.time1 <- Sys.time()

## read files with methRead
myobj=methRead(location = list_of_files,
               sample.id = sampleids,
               assembly ="hg19", # study used GrCh37 - hg19
               treatment = vec_label,
               context="CpG",
               header = TRUE, 
               pipeline = 'bismark',
               resolution = "base",
               sep = '\t',
               dbdir = "CLL_study/"
)

end.time1 <- Sys.time()
time.taken1 <- end.time1 - start.time1
print(time.taken1)

## Unite / merge samples, only keep CpGs that are methylated in at least 2 samples
start.time2 <- Sys.time()
meth=unite(myobj, destrand=FALSE, min.per.group = 2L) 
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
myDiff <- calculateDiffMeth(meth,
                            overdispersion = "MN",
                            effect         = "wmean",
                            test           = "F",
                            adjust         = 'BH',
                            # mc.cores       = 4, # does not work on (my?) windows acc. to R
                            slim           = F,
                            weighted.mean  = T)
saveRDS(myDiff, file = "CLL_study/calculateDiffMeth_object.txt")
myDiff

end.time3 <- Sys.time()
time.taken3 <- end.time3 - start.time3
print(time.taken3)

# readRDS("calculateDiffMeth_object.txt", refhook = NULL)

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
row_indeces_NAs = rows_to_delete_NAs(df_beta_vals, 3)
df_beta_vals_filt = df_beta_vals[-row_indeces_NAs,]
meth_new =  meth[-row_indeces_NAs,]
myDiff2 = myDiff[-row_indeces_NAs,]

# ## convert methylation Beta-value to M-value
# m_values = lumi::beta2m(beta_values)
# !!!! use fixlimits for Na/Inf values in m values



## Q Value adjustment

## adjust p values --> will become new Q VALUES !!!!
adj_q_vals = p.adjust(myDiff2$pvalue, method = "BH")
myDiff2$qvalue = adj_q_vals
###
df_adjusted_diff_meth = myDiff2
###



## EDMR: calculate all DMRs candidate from complete myDiff dataframe
dm_regions=edmr(myDiff = df_adjusted_diff_meth, mode=2, ACF=TRUE, DMC.qvalue = 0.30, plot = TRUE)
dm_regions
df_dmrs = data.frame(dm_regions)
nrow(df_dmrs)

 
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
print(df_beta_vals_filtered)



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
            file = "/classifying_data/filt-beta-values.txt",
            col.names = TRUE, sep = ";", row.names = TRUE)
 
# # write.table(df_m_vals_filtered,
# #             file = "C:/Users/andri/Documents/Uni London/QMUL/SemesterB/Masters_project/msc_project/data/classifying_data/filt-m-values.txt",
# #             col.names = TRUE, sep = ";", row.names = TRUE)
# # 
# # 
# # # t(df_beta_vals_filtered) %>% as.data.frame() %>% rownames()
# 
