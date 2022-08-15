library(ggplot2)
library(dplyr)
library(data.table)

setwd("C:/Users/andri/Documents/Uni London/QMUL/SemesterB/Masters_project/msc_project/")

df = read.table("data/artistic_trial/testdaten.txt", header = T, sep = "\t")
df[,2:13] = df[,2:13]*100
colnames(df)

ColorsDT <-  data.table(Group=df$ALGO,
                        Color=c('#000004', '#8019b3', '#911a34', '#f98e09', '#2679a3'), key="Group")
df = df %>% mutate(color = ColorsDT$Color)

p = ggplot2::ggplot(data = df, aes(x = ALGO,
                                   y = CTRLvsCIN2Plus)) +
  geom_bar(stat="identity", color = df$color, fill = df$color) +
  geom_text(aes(label=CTRLvsCIN2Plus), vjust=1.6, color="white", size=10) +
  theme_minimal() + theme(axis.title.x = element_text(size = 20),
                          axis.text.x = element_text(size = 18),
                          axis.title.y = element_text(size = 20),
                          axis.text.y = element_text(size = 18),
                          legend.position="none") + xlab("ML Model") +
  ylab("%") + ylim(0, 100)
p
ggsave("figures/sensitivity_barplot_CTRLvsCIN2Plus.png",
       bg = "white", width = 10, height = 5)

p = ggplot2::ggplot(data = df, aes(x = ALGO, y = FS_CTRLvsCIN2Plus)) +
  geom_bar(stat="identity", color = df$color, fill = df$color) +
  geom_text(aes(label=FS_CTRLvsCIN2Plus), vjust=1.6, color="white", size=10) +
  theme_minimal() + theme(axis.title.x = element_text(size = 20),
                          axis.text.x = element_text(size = 18),
                          axis.title.y = element_text(size = 20),
                          axis.text.y = element_text(size = 18), legend.position="none") + xlab("ML Model") +
  ylab("%") + ylim(0, 100)
p
ggsave("figures/sensitivity_barplot_FS_CTRLvsCIN2Plus.png",
       bg = "white", width = 10, height = 5)

p = ggplot2::ggplot(data = df, aes(x = ALGO,
                                   y = CTRLvsCIN2)) +
  geom_bar(stat="identity", color = df$color, fill = df$color) +
  geom_text(aes(label=CTRLvsCIN2), vjust=1.6, color="white", size=10) +
  theme_minimal() + theme(axis.title.x = element_text(size = 20),
                          axis.text.x = element_text(size = 18),
                          axis.title.y = element_text(size = 20),
                          axis.text.y = element_text(size = 18),
                          legend.position="none") + xlab("ML Model") +
  ylab("%") + ylim(0, 100)
p
ggsave("figures/sensitivity_barplot_CTRLvsCIN2.png",
       bg = "white", width = 10, height = 5)

p = ggplot2::ggplot(data = df, aes(x = ALGO, y = FS_CTRLvsCIN2)) +
  geom_bar(stat="identity", color = df$color, fill = df$color) +
  geom_text(aes(label=FS_CTRLvsCIN2), vjust=1.6, color="white", size=10) +
  theme_minimal() + theme(axis.title.x = element_text(size = 20),
                          axis.text.x = element_text(size = 18),
                          axis.title.y = element_text(size = 20),
                          axis.text.y = element_text(size = 18), legend.position="none") + xlab("ML Model") +
  ylab("%") + ylim(0, 100)
p
ggsave("figures/sensitivity_barplot_FS_CTRLvsCIN2.png",
       bg = "white", width = 10, height = 5)

p = ggplot2::ggplot(data = df, aes(x = ALGO, y = CTRLvsCIN3Plus)) +
  geom_bar(stat="identity", color = df$color, fill = df$color) +
  geom_text(aes(label=CTRLvsCIN3Plus), vjust=1.6, color="white", size=10) +
  theme_minimal() + theme(axis.title.x = element_text(size = 20),
                          axis.text.x = element_text(size = 18),
                          axis.title.y = element_text(size = 20),
                          axis.text.y = element_text(size = 18), legend.position="none") + xlab("ML Model") + 
  ylab("%") + ylim(0, 100)
p
ggsave("figures/sensitivity_barplot_CTRLvsCIN3Plus.png",
       bg = "white", width = 10, height = 5)


p = ggplot2::ggplot(data = df, aes(x = ALGO, y = FS_CTRLvsCIN3Plus)) +
  geom_bar(stat="identity", color = df$color, fill = df$color) +
  geom_text(aes(label=FS_CTRLvsCIN3Plus), vjust=1.6, color="white", size=10) +
  theme_minimal() + theme(axis.title.x = element_text(size = 20),
                          axis.text.x = element_text(size = 18),
                          axis.title.y = element_text(size = 20),
                          axis.text.y = element_text(size = 18), legend.position="none") + xlab("ML Model") + 
  ylab("%") + ylim(0, 100)
p
ggsave("figures/sensitivity_barplot_FS_CTRLvsCIN3Plus.png",
       bg = "white", width = 10, height = 5)


p = ggplot2::ggplot(data = df, aes(x = ALGO, y = negCTRLvsCIN2Plus)) +
  geom_bar(stat="identity", color = df$color, fill = df$color) +
  geom_text(aes(label=negCTRLvsCIN2Plus), vjust=1.6, color="white", size=10) +
  theme_minimal() + theme(axis.title.x = element_text(size = 20),
                          axis.text.x = element_text(size = 18),
                          axis.title.y = element_text(size = 20),
                          axis.text.y = element_text(size = 18), legend.position="none") + xlab("ML Model") + 
  ylab("%") + ylim(0, 100)
p
ggsave("figures/sensitivity_barplot_negCTRLvsCIN2Plus.png",
       bg = "white", width = 10, height = 5)


p = ggplot2::ggplot(data = df, aes(x = ALGO, y = FS_negCTRLvsCIN2Plus)) +
  geom_bar(stat="identity", color = df$color, fill = df$color) +
  geom_text(aes(label=FS_negCTRLvsCIN2Plus), vjust=1.6, color="white", size=10) +
  theme_minimal() + theme(axis.title.x = element_text(size = 20),
                          axis.text.x = element_text(size = 18),
                          axis.title.y = element_text(size = 20),
                          axis.text.y = element_text(size = 18), legend.position="none") + xlab("ML Model") + 
  ylab("%") + ylim(0, 100)
p
ggsave("figures/sensitivity_barplot_FS_negCTRLvsCIN2Plus.png",
       bg = "white", width = 10, height = 5)


p = ggplot2::ggplot(data = df, aes(x = ALGO, y = negCTRLvsCIN2)) + 
  geom_bar(stat="identity", color = df$color, fill = df$color) +
  geom_text(aes(label=negCTRLvsCIN2), vjust=1.6, color="white", size=10) +
  theme_minimal() + theme(axis.title.x = element_text(size = 20),
                          axis.text.x = element_text(size = 18),
                          axis.title.y = element_text(size = 20),
                          axis.text.y = element_text(size = 18), legend.position="none") + xlab("ML Model") +
  ylab("%") + ylim(0, 100)
p
ggsave("figures/sensitivity_barplot_negCTRLvsCIN2.png", 
       bg = "white", width = 10, height = 5)


p = ggplot2::ggplot(data = df, aes(x = ALGO, y = FS_negCTRLvsCIN2)) + 
  geom_bar(stat="identity", color = df$color, fill = df$color) +
  geom_text(aes(label=FS_negCTRLvsCIN2), vjust=1.6, color="white", size=10) +
  theme_minimal() + theme(axis.title.x = element_text(size = 20),
                          axis.text.x = element_text(size = 18),
                          axis.title.y = element_text(size = 20),
                          axis.text.y = element_text(size = 18), legend.position="none") + xlab("ML Model") + 
  ylab("%") + ylim(0, 100)
p
ggsave("figures/sensitivity_barplot_FS_negCTRLvsCIN2.png", 
       bg = "white", width = 10, height = 5)


p = ggplot2::ggplot(data = df, aes(x = ALGO, y = negCTRLvsCIN3Plus)) + 
  geom_bar(stat="identity", color = df$color, fill = df$color) +
  geom_text(aes(label=negCTRLvsCIN3Plus), vjust=1.6, color="white", size=10) +
  theme_minimal() + theme(axis.title.x = element_text(size = 20),
                          axis.text.x = element_text(size = 18),
                          axis.title.y = element_text(size = 20),
                          axis.text.y = element_text(size = 18), legend.position="none") + xlab("ML Model") + 
  ylab("%") + ylim(0, 100)
p
ggsave("figures/sensitivity_barplot_negCTRLvsCIN3Plus.png", 
       bg = "white", width = 10, height = 5)


p = ggplot2::ggplot(data = df, aes(x = ALGO, y = FS_negCTRLvsCIN3Plus)) + 
  geom_bar(stat="identity", color = df$color, fill = df$color) +
  geom_text(aes(label=FS_negCTRLvsCIN3Plus), vjust=1.6, color="white", size=10) +
  theme_minimal() + theme(axis.title.x = element_text(size = 20),
                          axis.text.x = element_text(size = 18),
                          axis.title.y = element_text(size = 20),
                          axis.text.y = element_text(size = 18), legend.position="none") + xlab("ML Model") + 
  ylab("%") + ylim(0, 100)
p
ggsave("figures/sensitivity_barplot_FS_negCTRLvsCIN3Plus.png", 
       bg = "white", width = 10, height = 5)


