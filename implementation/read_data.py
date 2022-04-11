import pandas as pd
import csv
import numpy as np

# took example rrbs data from https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE104998
df = pd.read_csv('../data/YB5_CRC_study/GSM2813719_YB5_DMSO4d1.txt', sep = '\t')

# # columns according to https://cgmaptools.github.io/cgmaptools_documentation/file-formats.html
# df.columns = ['Chr', 'Nuc', 'Pos', 'Context', 'DiNuc', 'MethyLevel', 'MC', 'NC']
print(df.head(20))
