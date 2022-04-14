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

setwd("C:/Users/andri/Documents/Uni London/QMUL/SemesterB/Masters_project/msc_project/data/YB5_CRC_study/")

## create file list
file.list = list.files(path = "C:/Users/andri/Documents/Uni London/QMUL/SemesterB/Masters_project/msc_project/data/YB5_CRC_study/", pattern= 'GSM.*.txt$')
list_of_files = as.list(file.list)
list_of_files

# read files with methRead
myobj=methRead(location = list_of_files,
               sample.id =list("case1","case2", "case3","case4","case5","case6", "ctrl1","ctrl2"),
               assembly ="hg18", ### old version of gnome - either *hg19 or hg38* - check what the CRC study used!
               treatment = c(1,1,1,1,1,1,0,0),
               context="CpG",
               header = TRUE, 
               pipeline = 'bismark', ### check user guide --> coverage files (do my cov files similar to those of the example)
               resolution = "base",
               sep = '\t',
               # dbtype = "tabix",
               dbdir = "C:/Users/andri/Documents/Uni London/QMUL/SemesterB/Masters_project/msc_project/data/YB5_CRC_study/"
)

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
meth=unite(myobj, destrand=FALSE, min.per.group = ) # check parameter 'min.per.group' (want cpg in ALL samples incld. case/ctrl) -- no missing values since small pilot study
head(meth)

# get correlation between samples
# getCorrelation(meth,plot=TRUE)

# cluster samples
clusterSamples(meth, dist="correlation", method="ward.D2", plot=TRUE) # the plot shows that case 3 can be removed for the analysis 

# pca plots
PCASamples(meth, screeplot=TRUE)
PCASamples(meth) # tells us to loose case4, case 5 (case3 from earlier)

# tiles=tileMethylCounts(myobj,win.size=1000,step.size=1000)
# head(tiles[[1]],3)


# Finding differentially methylated bases or regions
myDiff=calculateDiffMeth(meth)
myDiff

df_all_diffmethylation = methylKit::getData(myDiff)

# filter methlyation differences
myDiff_filtered = getMethylDiff(myDiff,difference=10,qvalue=1) ### adjust q-value e.g. to 0.05
myDiff_filtered

df_filtered_diffmethylation = methylKit::getData(myDiff_filtered) %>% dplyr::filter(pvalue < 0.05)
nrow(df_filtered_diffmethylation)

diffMethPerChr(myDiff, plot=FALSE,qvalue.cutoff=0.5, meth.cutoff=10)
diffMethPerChr(myDiff_filtered, plot=TRUE,qvalue.cutoff=0.5, meth.cutoff=10)

# Gene Annotation 

### use Bioconductor package *annotatr*: https://bioconductor.org/packages/release/bioc/html/annotatr.html
### https://bioconductor.org/packages/release/bioc/vignettes/annotatr/inst/doc/annotatr-vignette.html

gene.obj = genomation::readTranscriptFeatures("C:/Users/andri/Documents/Uni London/QMUL/SemesterB/Masters_project/msc_project/data/YB5_CRC_study/refseq.hg18.bed.txt")
annotateWithGeneParts(as(myDiff_filtered,"GRanges"), gene.obj)

cpg.obj=genomation::readFeatureFlank("C:/Users/andri/Documents/Uni London/QMUL/SemesterB/Masters_project/msc_project/data/YB5_CRC_study/cpgi.hg18.bed.txt",
                         feature.flank.name=c("CpGi","shores"))

# convert methylDiff object to GRanges and annotate
diffCpGann=annotateWithFeatureFlank(as(myDiff_filtered,"GRanges"),
                                    cpg.obj$CpGi,cpg.obj$shores,
                                    feature.name="CpGi",flank.name="shores")
promoters=regionCounts(myobj,gene.obj$promoters)
promoters


diffAnn=annotateWithGeneParts(as(myDiff_filtered,"GRanges"), gene.obj)

# target.row is the row number in myDiff25p
head(getAssociationWithTSS(diffAnn))

getTargetAnnotationStats(diffAnn,percentage=TRUE,precedence=TRUE)
plotTargetAnnotation(diffAnn,precedence=TRUE, main="differential methylation annotation")
plotTargetAnnotation(diffCpGann,col=c("green","gray","white"),
                     main="differential methylation annotation")
getFeatsWithTargetsStats(diffAnn,percentage=TRUE)


# EDMR 
myMixmdl=edmr::myDiff.to.mixmdl(df_filtered_diffmethylation, plot=T, main="example")

plotCost(myMixmdl, main="cost function")

# calculate all DMRs candidate

mydmr=edmr(df_filtered_diffmethylation, mode=2, ACF=TRUE) # mode = 2: return all regions that are either hyper- or hypo-methylated (unidirectional CPGs)
mydmr

# for the myDiff object - play around with parameters 


### Workflow: get the diff methylated cpgs/cpg regions --> annotation to link them to genes / promoters / gene bodies --> run ML model
## https://www.rdocumentation.org/packages/methylKit/versions/0.99.2/topics/percMethylation
## actual methylation values for each samples: mat = percMethylation(meth, rowids = TRUE )
## beta-matrix: beta-values = mat/100 
## use these values as features for predicting case vs control for cpgs / cpg regions that are diff. methylated

# further filtering the DMRs
mysigdmr=filter.dmr(mydmr)
mysigdmr

