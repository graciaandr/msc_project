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

BiocManager::install("lumi")
library(lumi)

setwd("C:/Users/andri/Documents/Uni London/QMUL/SemesterB/Masters_project/msc_project/data/YB5_CRC_study/")

## create file list
file.list = list.files(path = "C:/Users/andri/Documents/Uni London/QMUL/SemesterB/Masters_project/msc_project/data/YB5_CRC_study/", pattern= 'GSM.*.txt$')
list_of_files = as.list(file.list)
list_of_files[c(3,4,5)] = NULL # remove 3rd-5th element, as case 3,4,5 turned out to be of lower quality for DMA

print(list_of_files)
# read files with methRead
myobj=methRead(location = list_of_files,
               sample.id =list("case1","case2","case6", "ctrl1","ctrl2"),
               assembly ="hg19", # CRC study used hg19 and bismark v0.18.1
               treatment = c(1,1,1,0,0),
               context="CpG",
               header = TRUE, 
               pipeline = 'bismark', ### check user guide --> coverage files (do my cov files similar to those of the example)
               resolution = "base",
               sep = '\t',
               dbdir = "C:/Users/andri/Documents/Uni London/QMUL/SemesterB/Masters_project/msc_project/data/YB5_CRC_study/"
)
 
# verify if myobj overlaps with info in coverage files!!!

# calculate methylation & coverage statistics and save plots
for (i in (1:length(list_of_files))) {
  methstats_plot = getMethylationStats(myobj[[i]],plot=TRUE,both.strands=FALSE)
  png(paste0("methyl-stats", i, ".png"))
  dev.off()
  covstats_plot = getCoverageStats(myobj[[i]],plot=TRUE,both.strands=FALSE)
  png(paste0("cov-stats", i, ".png"))
  dev.off()
  
}

# filtered.myobj=filterByCoverage(myobj,lo.count=10,lo.perc=NULL,
#                                 hi.count=NULL,hi.perc=99.9)

# merge samples
meth=unite(myobj, destrand=FALSE, min.per.group = integer(4)) # check parameter 'min.per.group' (want cpg in ALL samples incld. case/ctrl) -- no missing values since small pilot study
head(meth)

# cluster samples
clusterSamples(meth, dist="correlation", method="ward.D2", plot=TRUE) # the plot shows that case 3 can be removed for the analysis 

# pca plots
PCASamples(meth, screeplot=TRUE)
PCASamples(meth) # tells us to loose case4, case 5 (case3 from earlier)

# Finding differentially methylated bases or regions
myDiff=calculateDiffMeth(meth)
myDiff

df_all_diffmethylation = methylKit::getData(myDiff)

# filter methlyation differences
myDiff_filtered = getMethylDiff(myDiff,difference=10,qvalue=0.05) ### adjust q-value e.g. to 0.05
myDiff_filtered

df_filtered_diffmethylation = methylKit::getData(myDiff_filtered) %>% dplyr::filter(pvalue < 0.05)
nrow(df_filtered_diffmethylation)

diffMethPerChr(myDiff, plot=FALSE,qvalue.cutoff=0.5, meth.cutoff=10)
diffMethPerChr(myDiff_filtered, plot=TRUE,qvalue.cutoff=0.5, meth.cutoff=10)

# Gene Annotation 

### use Bioconductor package *annotatr*: https://bioconductor.org/packages/release/bioc/html/annotatr.html
### https://bioconductor.org/packages/release/bioc/vignettes/annotatr/inst/doc/annotatr-vignette.html


# EDMR 
myMixmdl=edmr::myDiff.to.mixmdl(df_all_diffmethylation, plot=T, main="example")
plotCost(myMixmdl, main="cost function")

# calculate all DMRs candidate

mydmr=edmr(methylKit::getData(myDiff), mode=2, ACF=TRUE, DMC.qvalue = 0.5) # mode = 2: return all regions that are either hyper- or hypo-methylated (unidirectional CPGs)
mydmr

## actual methylation values for each samples:
mat = percMethylation(meth, rowids = TRUE )
beta_values = mat/100 
beta_values

lumi::beta2m(beta_values)


### check if beta or m values are better for the classifiers:
### read: https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-11-587
### maybe statistical tests require almost normal distribution (m values >>> beta values then)
### run the classifiers with both values 
### lumi (version 2.24.0) --> beta2m: Convert methylation Beta-value to M-value

## Gene Annotation with annotatr 

annots = c('hg19_cpgs', 'hg19_basicgenes', 'hg19_genes_intergenic',
           'hg19_genes_intronexonboundaries',
           'hg19_custom_ezh2', 'hg19_H3K4me3_Gm12878')

# Build the annotations (a single GRanges object)
annotations = build_annotations(genome = 'hg19', annotations = annots)

# Intersect the regions we read in with the annotations
dm_annotated = annotate_regions(
  regions = dm_regions,
  annotations = annotations,
  ignore.strand = TRUE,
  quiet = FALSE)
# A GRanges object is returned
print(dm_annotated)

### Workflow: get the diff methylated cpgs/cpg regions --> annotation to link them to genes / promoters / gene bodies --> run ML model
## https://www.rdocumentation.org/packages/methylKit/versions/0.99.2/topics/percMethylation
## actual methylation values for each samples: mat = percMethylation(meth, rowids = TRUE )
## beta-matrix: beta-values = mat/100 
## use these values as features for predicting case vs control for cpgs / cpg regions that are diff. methylated

# further filtering the DMRs
mysigdmr=filter.dmr(mydmr)
mysigdmr

