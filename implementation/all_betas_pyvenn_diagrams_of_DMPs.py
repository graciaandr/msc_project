import matplotlib.pyplot as plt
from venn import venn
from venn import pseudovenn
import pandas as pd
import numpy as np

# CTRL vs CIN2+/CIN2/CIN3
CTRLvsCIN2plus = pd.read_csv('./data/classifying_data/artistic_study_filt-beta-values_0722_50threshold.txt', sep = ';')
CTRLvsCIN2 = pd.read_csv('./data/classifying_data/CTRLvsCIN2_artistic_study_filt-beta-values_0722_50threshold.txt', sep = ';')
CTRLvsCIN3 = pd.read_csv('./data/classifying_data/CTRLvsCIN3_artistic_study_filt-beta-values_0722_50threshold.txt', sep = ';')
print(CTRLvsCIN3.head(5))

# neg CTRL vs CIN2+/CIN2/CIN3
negCTRLvsCIN2plus = pd.read_csv('./data/classifying_data/negCTRL_CIN2+_artistic_study_filt-beta-values_0722_50threshold.txt', sep = ';')
negCTRLvsCIN2 = pd.read_csv('./data/classifying_data/negCTRLvsCIN2_artistic_study_filt-beta-values_0722_50threshold.txt', sep = ';')
negCTRLvsCIN3 = pd.read_csv('./data/classifying_data/negCTRLvsCIN3_artistic_study_filt-beta-values_0722_50threshold.txt', sep = ';')

CTRLvsCIN2plus = list(CTRLvsCIN2plus.index)
CTRLvsCIN2 = list(CTRLvsCIN2.index)
CTRLvsCIN3 = list(CTRLvsCIN3.index)

negCTRLvsCIN2plus = list(negCTRLvsCIN2plus.index)
negCTRLvsCIN2 = list(negCTRLvsCIN2.index)
negCTRLvsCIN3 = list(negCTRLvsCIN3.index)

Cin2plusComparisons = {
    "CTRL vs CIN2+": set(CTRLvsCIN2plus),
    "neg CTRL vs CIN2+": set(negCTRLvsCIN2plus),
}
venn(Cin2plusComparisons, cmap="plasma")
plt.show()

negCtrls_comparisons_dict = {
    "CTRL vs CIN2": set(CTRLvsCIN2),
    "CTRL vs CIN3": set(CTRLvsCIN3),
    "neg CTRL vs CIN2": set(negCTRLvsCIN2),
    "neg CTRL vs CIN3": set(negCTRLvsCIN3), 
}

venn(negCtrls_comparisons_dict, cmap="plasma")
plt.show()


ctrlCin2ctrlCin2plus = {
    "CTRL vs CIN2+": set(CTRLvsCIN2plus),
    "CTRL vs CIN2": set(CTRLvsCIN2),
}

# venn(ctrlCin2ctrlCin2plus, cmap="plasma")
# plt.show()

ctrlCin3ctrlCin2plus = {
    "CTRL vs CIN2+": set(CTRLvsCIN2plus),
    "CTRL vs CIN3": set(CTRLvsCIN3),
}

# venn(ctrlCin3ctrlCin2plus, cmap="plasma")
# plt.show()


ctrlCin3ctrlCin2 = {
    "CTRL vs CIN2": set(CTRLvsCIN2),
    "CTRL vs CIN3": set(CTRLvsCIN3),
}

venn(ctrlCin3ctrlCin2, cmap="plasma")
plt.show()

allctrl_comparisons_dict = {
    "CTRL vs CIN2+": set(CTRLvsCIN2plus),
    "CTRL vs CIN2": set(CTRLvsCIN2),
    "CTRL vs CIN3": set(CTRLvsCIN3),
}

venn(allctrl_comparisons_dict, cmap="viridis", fontsize=20, alpha=0.425)
plt.savefig('./figures/VennDiagram_allCTRLsvsCIN2pCIN2CIN3_AllFeatures.png')
plt.show()

negCtrls_comparisons_dict = {
    "neg CTRL vs CIN2+": set(negCTRLvsCIN2plus),
    "neg CTRL vs CIN2": set(negCTRLvsCIN2),
}

venn(negCtrls_comparisons_dict, cmap="plasma")
plt.show()

negCtrls_comparisons_dict = {
    "neg CTRL vs CIN2+": set(negCTRLvsCIN2plus),
    "neg CTRL vs CIN3": set(negCTRLvsCIN3), 
}

venn(negCtrls_comparisons_dict, cmap="plasma")
plt.show()

negCtrls_comparisons_dict = {
    "neg CTRL vs CIN2": set(negCTRLvsCIN2),
    "neg CTRL vs CIN3": set(negCTRLvsCIN3), 
}

venn(negCtrls_comparisons_dict, cmap="plasma")
plt.show()

negCtrls_comparisons_dict = {
    "neg CTRL vs CIN2+": set(negCTRLvsCIN2plus),
    "neg CTRL vs CIN2": set(negCTRLvsCIN2),
    "neg CTRL vs CIN3": set(negCTRLvsCIN3), 
}

venn(negCtrls_comparisons_dict, cmap="plasma",  fontsize=20, alpha=0.425)
plt.savefig('./figures/VennDiagram_allnegCTRLsvsCIN2pCIN2CIN3_AllFeatures.png')
plt.show()

# all6_comparisons_dict = {
#     "CTRL vs CIN2+": set(CTRLvsCIN2plus),
#     "CTRL vs CIN2": set(CTRLvsCIN2),
#     "CTRL vs CIN3": set(CTRLvsCIN3),
#     "neg CTRL vs CIN2+": set(negCTRLvsCIN2plus),
#     "neg CTRL vs CIN2": set(negCTRLvsCIN2),
#     "neg CTRL vs CIN3": set(negCTRLvsCIN3), 
# }
# pseudovenn(all6_comparisons_dict, cmap="plasma")
# plt.savefig('./figures/VennDiagram_all6comparisons_AllFeatures.png')
# plt.show()

