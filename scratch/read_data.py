import pandas as pd
import csv
import numpy as np

# took example rrbs data from https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM4830470
df = pd.read_csv('3032_S29_R1_001.CGmap', sep = '\t', header=None)

# columns according to https://cgmaptools.github.io/cgmaptools_documentation/file-formats.html
df.columns = ['Chr', 'Nuc', 'Pos', 'Context', 'DiNuc', 'MethyLevel', 'MC', 'NC']
print(df.head(20))
