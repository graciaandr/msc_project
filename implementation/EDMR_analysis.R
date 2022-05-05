## EDMR: calculate all DMRs candidate from complete myDiff dataframe
## do local EDMR Analysis, as EDMR did not work on apocrita

df_adjusted_diff_meth = read.table(file = "/classifying_data/adjusted_myDiff_df.txt",
                                   col.names = TRUE, sep = ";", row.names = TRUE)

df_beta_vals_filt = read.table(file = "/classifying_data/df_beta_vals_filt.txt",
                               col.names = TRUE, sep = ";", row.names = TRUE)

meth_new = read.table(file = "/classifying_data/adjusted_methylation_df.txt",
            col.names = TRUE, sep = ";", row.names = TRUE)



## EDMR: calculate all DMRs candidate from complete myDiff dataframe
dm_regions=edmr(myDiff = df_adjusted_diff_meth, mode=2, ACF=TRUE, DMC.qvalue = 0.30, plot = TRUE)
dm_regions
df_dmrs = data.frame(dm_regions)
nrow(df_dmrs)


## for loop that goes through the start pos, end pos, and seqnames per row in beta/m value dataframe and DMR data
## to retrieve sig. diff. meth. CpG sites in DMRs
df_tmp1 = data.frame(matrix(NA, nrow = 1, ncol = ncol(df_beta_vals_filt)))
# df_tmp2 = data.frame(matrix(NA, nrow = 1, ncol = ncol(df_m_vals)))
colnames(df_tmp1) <- colnames((df_beta_vals_filt))
# colnames(df_tmp2) <- colnames((df_m_vals))
df_beta_vals_filtered = NULL
# df_m_vals_filtered = NULL
for (i in (1:length(df_dmrs$start))) {
  df_tmp1 = df_beta_vals_filt %>%
    filter(pos >= df_dmrs$start[[i]] & pos <= df_dmrs$end[[i]] & chr == df_dmrs$seqnames[[i]])
  #   df_tmp2 = df_m_vals %>%
  # filter(pos >= df_dmrs$start[[i]] & pos <= df_dmrs$end[[i]] & chr == df_dmrs$seqnames[[i]])
  
  df_beta_vals_filtered = rbind(df_beta_vals_filtered, df_tmp1)
  #   df_m_vals_filtered = rbind(df_m_vals_filtered, df_tmp2)
}

# print(df_m_vals_filtered)
print(df_beta_vals_filtered)



# # ## Gene Annotation with annotatr 
# # ### use Bioconductor package *annotatr*: https://bioconductor.org/packages/release/bioc/html/annotatr.html
# # ### https://bioconductor.org/packages/release/bioc/vignettes/annotatr/inst/doc/annotatr-vignette.html
# # 
# # annots = c('hg19_cpgs', 'hg19_basicgenes', 'hg19_genes_intergenic',
# #            'hg19_genes_intronexonboundaries')
# # 
# # # Build the annotations (a single GRanges object)
# # annotations = build_annotations(genome = 'hg19', annotations = annots)
# # 
# # # Intersect the regions we read in with the annotations
# # dm_annotated = annotate_regions(
# #   regions = dm_regions,
# #   annotations = annotations,
# #   ignore.strand = TRUE,
# #   quiet = FALSE)
# # A GRanges object is returned
# # print(dm_annotated)


# store filtered beta and m values as TXT ==> will be used to classify data
write.table(df_beta_vals_filtered,
            file = "/classifying_data/filt-beta-values.txt",
            col.names = TRUE, sep = ";", row.names = TRUE)

# # write.table(df_m_vals_filtered,
# #             file = "C:/Users/andri/Documents/Uni London/QMUL/SemesterB/Masters_project/msc_project/data/classifying_data/filt-m-values.txt",
# #             col.names = TRUE, sep = ";", row.names = TRUE)
# # 
# # 
# # # t(df_beta_vals_filtered) %>% as.data.frame() %>% rownames()
# 
