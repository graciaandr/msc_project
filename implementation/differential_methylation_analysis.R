if (!require("BiocManager", quietly = TRUE))
  install.packages("BiocManager")

BiocManager::install("methylKit")

install.packages( c("data.table", "mixtools", "devtools"))
source("http://bioconductor.org/biocLite.R")
biocLite(c("GenomicRanges","IRanges"))
# install from github
library(devtools)
install_github("ShengLi/edmr")