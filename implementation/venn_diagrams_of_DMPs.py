import matplotlib.pyplot as plt
from matplotlib_venn import venn2
import pandas as pd
import numpy as np


# load data sets

# all controls vs CIN2+ 
CTRLvsCIN2plus_artistic_study_filt = pd.read_csv('./data/classifying_data/artistic_study_filt-beta-values_0722_50threshold.txt', sep = ';')
# CTRLvsCIN2plus_artistic_study_filt = pd.read_csv('./classifying_data/artistic_study_filt-beta-values_0722_50threshold.txt', sep = ';')

# all controls vs CIN2
CTRLvsCIN2_artistic_study_filt = pd.read_csv('./data/classifying_data/CTRLvsCIN2_artistic_study_filt-beta-values_0722_50threshold.txt', sep = ';')
# CTRLvsCIN2_artistic_study_filt = pd.read_csv('./classifying_data/CTRLvsCIN2_artistic_study_filt-beta-values_0722_50threshold.txt', sep = ';')

# all controls vs CIN3
CTRLvsCIN3_artistic_study_filt = pd.read_csv('./data/classifying_data/CTRLvsCIN3_artistic_study_filt-beta-values_0722_50threshold.txt', sep = ';')
# CTRLvsCIN3_artistic_study_filt = pd.read_csv('./classifying_data/CTRLvsCIN3_artistic_study_filt-beta-values_0722_50threshold.txt', sep = ';')


# negative controls vs CIN2+
negCTRL_CIN2plus_artistic_study_filt = pd.read_csv('./data/classifying_data/negCTRL_CIN2+_artistic_study_filt-beta-values_0722_50threshold.txt', sep = ';')
# negCTRL_CIN2plus_artistic_study_filt = pd.read_csv('./classifying_data/negCTRL_CIN2+_artistic_study_filt-beta-values_0722_50threshold.txt', sep = ';')

# negative controls vs CIN2
negCTRLvsCIN2_artistic_study_filt = pd.read_csv('./data/classifying_data/negCTRLvsCIN2_artistic_study_filt-beta-values_0722_50threshold.txt', sep = ';')
# negCTRLvsCIN2_artistic_study_filt = pd.read_csv('./classifying_data/negCTRLvsCIN2_artistic_study_filt-beta-values_0722_50threshold.txt', sep = ';')

# negative controls vs CIN3
negCTRLvsCIN3_artistic_study_filt = pd.read_csv('./data/classifying_data/negCTRLvsCIN3_artistic_study_filt-beta-values_0722_50threshold.txt', sep = ';')
# negCTRLvsCIN3_artistic_study_filt = pd.read_csv('./classifying_data/negCTRLvsCIN3_artistic_study_filt-beta-values_0722_50threshold.txt', sep = ';')

