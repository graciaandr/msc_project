library(dplyr)
library(magrittr)
install.packages('randomForest')
setwd("C:/Users/andri/Documents/Uni London/QMUL/SemesterB/Masters_project/msc_project/")
df <- read.csv('data/classifying_data/filt-beta-values.txt', sep = ';')
head(df)

df_t = t(df) %>% as.data.frame()
df2 = df_t %>% dplyr::mutate(names = rownames(df_t))

View(df2)
