# if (!require("BiocManager", quietly = TRUE))
#   install.packages("BiocManager")

setwd('C:/Users/andri/Documents/Uni London/QMUL/SemesterB/Masters_project/msc_project/data/YB5_CRC_study/')

library(methylKit)
library(GenomicRanges)
library(IRanges)
library(devtools)
# install_github("ShengLi/edmr")
library(edmr)

# file.list=list( system.file("extdata", 
#                             "GSM2813719_YB5_DMSO4d1.txt", package = "methylKit"),
#                 system.file("extdata",
#                             "GSM2813720_YB5_DMSO4d2.txt", package = "methylKit"),
#                 system.file("extdata", 
#                             "GSM2813725_YB5_con1.txt", package = "methylKit"),
#                 system.file("extdata", 
#                             "GSM2813726_YB5_con2.txt", package = "methylKit") )

file.list = list.files(path = "C:/Users/andri/Documents/Uni London/QMUL/SemesterB/Masters_project/msc_project/data/YB5_CRC_study/", pattern= 'GSM')

file.list = system.file(file.list, package = "methylKit")
myobj=methRead(file.list,
               sample.id=list("test1","test2", "test3","test4","test5","test6","ctrl1","ctrl2"),
               assembly="hg18",
               treatment=c(1,1,0,0),
               context="CpG"
)

getMethylationStats(myobj[[2]],plot=FALSE,both.strands=FALSE)
