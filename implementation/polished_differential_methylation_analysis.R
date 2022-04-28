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
               pipeline = 'bismark',
               resolution = "base",
               sep = '\t',
               dbdir = "C:/Users/andri/Documents/Uni London/QMUL/SemesterB/Masters_project/msc_project/data/YB5_CRC_study/"
)
 
### verify if myobj overlaps with info in coverage files!!!

# calculate methylation & coverage statistics and save plots
for (i in (1:length(list_of_files))) {
  methstats_plot = getMethylationStats(myobj[[i]],plot=TRUE,both.strands=FALSE)
  png(paste0("methyl-stats", i, ".png"))
  dev.off()
  covstats_plot = getCoverageStats(myobj[[i]],plot=TRUE,both.strands=FALSE)
  png(paste0("cov-stats", i, ".png"))
  dev.off()
  
}


# merge samples
meth=unite(myobj, destrand=FALSE) ## adjust min per group to see if i get more cpgs eventually 
  # check parameter 'min.per.group' (want cpg in ALL samples incld. case/ctrl) -- no missing values since small pilot study
  # By default only regions/bases that are covered in all samples are united as methylBase object -- according to https://www.rdocumentation.org/packages/methylKit/versions/0.99.2/topics/unite
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

# EDMR 
myMixmdl=edmr::myDiff.to.mixmdl(df_all_diffmethylation, plot=T, main="example")
plotCost(myMixmdl, main="cost function")

# calculate all DMRs candidate from complete myDiff dataframe

# dm_regions=edmr(myDiff = df_all_diffmethylation, mode=1, ACF=TRUE, DMC.qvalue = 0.05, plot = TRUE) # mode = 2: return all regions that are either hyper- or hypo-methylated (unidirectional CPGs)
dm_regions=edmr(myDiff = df_all_diffmethylation, mode=2, ACF=TRUE, DMC.qvalue = 1, plot = TRUE) ## set DMC q value to 1 bc otherwise empty dmrs
dm_regions
as.data.frame(dm_regions) %>% dplyr::filter(DMR.pvalue < 0.5)
df_dmrs = data.frame(dm_regions)

## actual methylation values for each samples:
mat = percMethylation(meth, rowids = TRUE )
beta_values = mat/100 
beta_values

# convert methylation Beta-value to M-value
m_values = lumi::beta2m(beta_values)
m_values 

### !!! need to figure out if i need to change and if how to set the inf values (NA or 10000 or idk)

# connect dm regions and beta/m values
split_rownames = (stringr::str_split(rownames(beta_values), pattern = "\\.", n = 3, simplify = FALSE))

## extract and add positions and chromosome info as extra columns
positions = c()
chrs = c()
for (i in (1:length(split_rownames))) {
# for (i in (1:10)) {
  chrs = c(chrs, split_rownames[[i]][1])
  positions = c(positions, split_rownames[[i]][2])
}

# add postions as own column to beta and m value data frames ==> for fitering & eventually classifier training
df_beta_vals = data.frame(beta_values) %>% dplyr::mutate(pos = df_meth$start, chrom = df_meth$chr) ###
df_beta_vals[order(df_beta_vals$pos),]
rownames(df_beta_vals) = NULL

df_meth = data.frame(meth)
# try using regular expression for adding pos and chr names 
df_m_vals = data.frame(m_values) %>% dplyr::mutate(pos = df_meth$start, chrom = df_meth$chr)
df_m_vals[order(df_m_vals$pos),]
rownames(df_m_vals) = NULL


df_beta_vals %>%
  filter(pos >= df_dmrs$start & pos <= df_dmrs$end & chrom == df_dmrs$seqnames)

df_m_vals %>%
  filter(pos >= df_dmrs$start & pos <= df_dmrs$end & chrom == df_dmrs$seqnames)


## Gene Annotation with annotatr 
### use Bioconductor package *annotatr*: https://bioconductor.org/packages/release/bioc/html/annotatr.html
### https://bioconductor.org/packages/release/bioc/vignettes/annotatr/inst/doc/annotatr-vignette.html

annots = c('hg19_cpgs', 'hg19_basicgenes', 'hg19_genes_intergenic',
           'hg19_genes_intronexonboundaries')

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


## store filtered beta and m values as TXT ==> will be used to classify data
write.table(df_beta_vals, 
            file = "C:/Users/andri/Documents/Uni London/QMUL/SemesterB/Masters_project/msc_project/data/classifying_data/beta-values.txt", 
            col.names = TRUE, sep = ";", row.names = TRUE)

write.table(df_m_vals, 
            file = "C:/Users/andri/Documents/Uni London/QMUL/SemesterB/Masters_project/msc_project/data/classifying_data/m-values.txt", 
            col.names = TRUE, sep = ";", row.names = TRUE)


