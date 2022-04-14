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
               assembly ="hg18",
               treatment = c(1,1,1,1,1,1,0,0),
               context="CpG",
               header = TRUE, 
               pipeline = 'bismark',
               resolution = "base",
               sep = '\t',
               dbtype = "tabix",
               dbdir = "C:/Users/andri/Documents/Uni London/QMUL/SemesterB/Masters_project/msc_project/data/YB5_CRC_study/"
)

# calculate methylation & coverage statistics and save plots

for (i in (1:length(list_of_files))) {
  methstats_plot = getMethylationStats(myobj[[i]],plot=TRUE,both.strands=FALSE)
  png(paste0("methyl-stats", i, ".png"))
  covstats_plot = getCoverageStats(myobj[[i]],plot=TRUE,both.strands=FALSE)
  png(paste0("cov-stats", i, ".png"))
  
}

# filtered.myobj=filterByCoverage(myobj,lo.count=10,lo.perc=NULL,
#                                 hi.count=NULL,hi.perc=99.9)

# merge samples
meth=unite(myobj, destrand=FALSE)
head(meth)

# get correlation between samples
# getCorrelation(meth,plot=TRUE)

# cluster samples
clusterSamples(meth, dist="correlation", method="ward.D2", plot=TRUE) 

# pca plots
PCASamples(meth, screeplot=TRUE)
PCASamples(meth)

# tiles=tileMethylCounts(myobj,win.size=1000,step.size=1000)
# head(tiles[[1]],3)


# Finding differentially methylated bases or regions
myDiff=calculateDiffMeth(meth)
myDiff

df_all_diffmethylation = methylKit::getData(myDiff)

# filter methlyation differences
myDiff_filtered = getMethylDiff(myDiff,difference=10,qvalue=1)
myDiff_filtered

df_filtered_diffmethylation = methylKit::getData(myDiff_filtered)
nrow(df_filtered_diffmethylation %>% dplyr::filter(pvalue < 0.05))

# # get hyper methylated bases
# myDiff25p.hyper=getMethylDiff(myDiff,difference=25,qvalue=0.01,type="hyper")
# 
# # get hypo methylated bases
# myDiff25p.hypo=getMethylDiff(myDiff,difference=25,qvalue=0.01,type="hypo")
# 
# # get all differentially methylated bases
# myDiff25p=getMethylDiff(myDiff,difference=25,qvalue=0.01)

diffMethPerChr(myDiff, plot=FALSE,qvalue.cutoff=0.5, meth.cutoff=10)
diffMethPerChr(myDiff_filtered, plot=TRUE,qvalue.cutoff=0.5, meth.cutoff=10)

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

mydmr=edmr(df_filtered_diffmethylation, mode=1, ACF=TRUE)

# calculate all DMRs candidate
mydmr=edmr(df_filtered_diffmethylation, mode=1, ACF=TRUE)

# further filtering the DMRs
mysigdmr=filter.dmr(mydmr)

# ## annotation
# # get genebody annotation GRangesList object
# #genebody=genebody.anno(file="http://edmr.googlecode.com/files/hg19_refseq_all_types.bed")
# genebody.file=system.file("extdata", "chr22.hg19_refseq_all_types.bed.gz", package = "edmr")
# genebody=genebody.anno(file=genebody.file)
# 
# # plot the eDMR genebody annotation
# plotdmrdistr(mysigdmr, genebody)
# 
# # get CpG islands and shores annotation
# #cpgi=cpgi.anno(file="http://edmr.googlecode.com/files/hg19_cpgisland_all.bed")
# cpgi.file=system.file("extdata", "chr22.hg19_cpgisland_all.bed.gz", package = "edmr")
# cpgi=cpgi.anno(file=cpgi.file)
# 
# # plot the eDMR CpG islands and shores annotation
# plotdmrdistr(mysigdmr, cpgi)
# 
# # prepare genes for pathway analysis with significant DMRs at its promoter regions 
# dmr.genes=get.dmr.genes(myDMR=mysigdmr, subject=genebody$promoter, id.type="gene.symbol")
# dmr.genes
# 
