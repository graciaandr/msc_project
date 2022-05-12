library(mixtools)
library(data.table)
library(dplyr)
library(magrittr)

install.packages("randomForest")
library(randomForest)

setwd("C:/Users/andri/Documents/Uni London/QMUL/SemesterB/Masters_project/msc_project/")
df_test <- read.csv('data/classifying_data/beta_vals_labelled_data.txt', sep = ';')
sum(is.na(df_test))

# df <- read.csv('data/classifying_data/CLL_study_filt-beta-values.txt', sep = ';')
# head(df)
# 
# 
# df_t = t(df) %>% as.data.frame()
# df2 = df_t %>% dplyr::mutate(names = rownames(df_t))
# 
# df_ctrl <- df2[grep("ctrl", df2$names), ]
# df_ctrl <- df_ctrl %>% mutate(Condition = 0)
# 
# df_trt <- df2[grep("case", df2$names), ]
# df_trt <- df_trt %>% mutate(Condition = 1)
# 
# df_labelled <- rbind(df_ctrl, df_trt)
# View(df_labelled)
df_labelled <- df_test

train <- sample(nrow(df_labelled), 0.7*nrow(df_labelled), replace = FALSE)
TrainSet <- df_labelled[train,]
ValidSet <- df_labelled[-train,]
# summary(TrainSet)
# summary(ValidSet)

model1 <- randomForest(Condition ~ ., data = TrainSet, importance = TRUE)


# Fine tuning parameters of Random Forest model
model2 <- randomForest(Condition ~ ., data = TrainSet, ntree = 500, mtry = 6, importance = TRUE)
model2

