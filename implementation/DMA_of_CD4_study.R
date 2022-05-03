library(methylKit)
library(GenomicRanges)
library(IRanges)
library(devtools)
library(edmr)
library(genomation)
library(mixtools)
library(data.table)
library(magrittr)
library(dplyr)
library(annotatr)
library(lumi)

# info about study & data: 
### Trimmed reads were then mapped to the in silico bisulfite converted human reference genome (GRCh38) 
### by using Bismark v.0.22.1 with and non directional parameter applied
### Genome_build: GRCh38
### The genome-wide cytosine methylation output file is tab-delimited in the following format: 
### <chromosome> <position> <strand> <count methylated> <count non-methylated> <C-context> <trinucleotide context>

setwd("C:/Users/andri/Documents/Uni London/QMUL/SemesterB/Masters_project/msc_project/data/CD4_Tcell_study/")

## create file list
file.list = list.files(path = "C:/Users/andri/Documents/Uni London/QMUL/SemesterB/Masters_project/msc_project/data/CD4_Tcell_study", pattern= '*.txt$')
list_of_files = as.list(file.list)
# list_of_files[c(2,5,6,11,13,14)] = NULL
  # remove ctrl1, ctrl4 and ctrl7 as cluster indicated bas results for those, 
  # accordingly also removed one case samples (after clustering and PCA, decided to remove case2 & 4 - case 7 is too much)
print(list_of_files)

start.time1 <- Sys.time()

# read files with methRead
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
               dbdir = "C:/Users/andri/Documents/Uni London/QMUL/SemesterB/Masters_project/msc_project/data/CD4_Tcell_study/"
)

end.time1 <- Sys.time()
time.taken1 <- end.time1 - start.time1
print(time.taken1)

# merge samples
start.time2 <- Sys.time()
meth=unite(myobj, destrand=FALSE, min.per.group = 2L) ## adjust min per group to see if i get more cpgs eventually
  # check parameter 'min.per.group' (want cpg in ALL samples incld. case/ctrl) -- no missing values since small pilot study
  # By default only regions/bases that are covered in all samples are united as methylBase object -- according to https://www.rdocumentation.org/packages/methylKit/versions/0.99.2/topics/unite
end.time2 <- Sys.time()
time.taken2 <- end.time2 - start.time2
print(time.taken2)

head(meth)
df_meth = data.frame(meth)
nrow(df_meth)

# cluster samples
clusterSamples(meth, dist="correlation", method="ward.D2", plot=TRUE) 

# pca plots
# PCASamples(meth, screeplot=TRUE)

png("PCA_CD4_study.png")
PCASamples(meth) 
dev.off()

# clustering and pca show contradicting results regarding which sample(s) to throw out to get equal amount of samples per condition and 
# carry on with DM sites analysis

start.time3 <- Sys.time()

# Finding differentially methylated bases or regions
myDiff <- calculateDiffMeth(meth,
                            overdispersion = "MN",
                            effect         = "wmean",
                            test           = "F",
                            adjust         = 'BH',
                            # mc.cores       = 4, # does not work on (my?) windows acc. to R
                            slim           = F,
                            weighted.mean  = T)
myDiff

end.time3 <- Sys.time()
time.taken3 <- end.time3 - start.time3
print(time.taken3)

df_all_diffmethylation = methylKit::getData(myDiff)

# filter methlyation differences
myDiff_filtered = getMethylDiff(myDiff, difference=10,qvalue=0.05) ### adjust q-value e.g. to 0.05
myDiff_filtered


# png("diffMethPerChr.png")
diffMethPerChr(myDiff_filtered, plot=TRUE,qvalue.cutoff=0.05, meth.cutoff=10)
# dev.off()

## start removing NAs in here before EDMR
## filter myDiff data frame, so that less NAs
## adjust p values --> will become new Q VALUES !!!!
## p.adjust(myDiff2$pvalue, method = "BH")

# EDMR: calculate all DMRs candidate from complete myDiff dataframe
dm_regions=edmr(myDiff = df_all_diffmethylation, mode=2, ACF=TRUE, DMC.qvalue = 0.30, plot = TRUE) 
dm_regions
df_dmrs = data.frame(dm_regions)
nrow(df_dmrs)

## actual methylation values for each samples:
mat = percMethylation(meth, rowids = TRUE )
beta_values = mat/100 

# convert methylation Beta-value to M-value
m_values = lumi::beta2m(beta_values)


### !!! need to figure out if i need to change and if how to set the inf values (NA or 10000 or idk)

# add postions as own column to beta and m value data frames ==> for fitering & eventually classifier training
df_beta_vals = data.frame(beta_values) %>% dplyr::mutate(pos = df_meth$start, chrom = df_meth$chr) ###
df_beta_vals[order(df_beta_vals$pos),]

df_m_vals = data.frame(m_values) %>% dplyr::mutate(pos = df_meth$start, chrom = df_meth$chr)
df_m_vals[order(df_m_vals$pos),]

## add chr in front of all chromosome names to be able to compare to seqnmaes in DMR dataframe later on when filtering
df_beta_vals['chr'] = paste0('chr', df_beta_vals$chrom)
df_m_vals['chr'] = paste0('chr', df_m_vals$chrom)


## for loop that goes through the start pos, end pos, and seqnames per row
df_tmp1 = data.frame(matrix(NA, nrow = 1, ncol = ncol(df_beta_vals)))
df_tmp2 = data.frame(matrix(NA, nrow = 1, ncol = ncol(df_m_vals)))
colnames(df_tmp1) <- colnames((df_beta_vals))
colnames(df_tmp2) <- colnames((df_m_vals))
df_beta_vals_filtered = NULL
df_m_vals_filtered = NULL
for (i in (1:length(df_dmrs$start))) {
  df_tmp1 = df_beta_vals %>%
            filter(pos >= df_dmrs$start[[i]] & pos <= df_dmrs$end[[i]] & chr == df_dmrs$seqnames[[i]])
  df_tmp2 = df_m_vals %>%
            filter(pos >= df_dmrs$start[[i]] & pos <= df_dmrs$end[[i]] & chr == df_dmrs$seqnames[[i]])
  
  df_beta_vals_filtered = rbind(df_beta_vals_filtered, df_tmp1)
  df_m_vals_filtered = rbind(df_m_vals_filtered, df_tmp2)
}

print(df_m_vals_filtered)
print(df_beta_vals_filtered)



# ## Gene Annotation with annotatr 
# ### use Bioconductor package *annotatr*: https://bioconductor.org/packages/release/bioc/html/annotatr.html
# ### https://bioconductor.org/packages/release/bioc/vignettes/annotatr/inst/doc/annotatr-vignette.html
# 
# annots = c('hg19_cpgs', 'hg19_basicgenes', 'hg19_genes_intergenic',
#            'hg19_genes_intronexonboundaries')
# 
# # Build the annotations (a single GRanges object)
# annotations = build_annotations(genome = 'hg19', annotations = annots)
# 
# # Intersect the regions we read in with the annotations
# dm_annotated = annotate_regions(
#   regions = dm_regions,
#   annotations = annotations,
#   ignore.strand = TRUE,
#   quiet = FALSE)
# A GRanges object is returned
# print(dm_annotated)


# store filtered beta and m values as TXT ==> will be used to classify data
# write.table(df_beta_vals_filtered,
#             file = "C:/Users/andri/Documents/Uni London/QMUL/SemesterB/Masters_project/msc_project/data/classifying_data/filt-beta-values.txt",
#             col.names = TRUE, sep = ";", row.names = TRUE)
# 
# write.table(df_m_vals_filtered,
#             file = "C:/Users/andri/Documents/Uni London/QMUL/SemesterB/Masters_project/msc_project/data/classifying_data/filt-m-values.txt",
#             col.names = TRUE, sep = ";", row.names = TRUE)


# t(df_beta_vals_filtered) %>% as.data.frame() %>% rownames()
