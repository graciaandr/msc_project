import matplotlib.pyplot as plt
from venn import venn
from venn import pseudovenn
import pandas as pd
import numpy as np

# CTRL vs CIN2+/CIN2/CIN3
RF_features_CTRLvsCIN2plus = pd.read_csv('./data/classifying_data/RF_features_CTRLvsCIN2+.csv', sep = ';')
SVM_features_CTRLvsCIN2 = pd.read_csv('./data/classifying_data/SVM_features_CTRLvsCIN2.csv', sep = ';', index_col=False)
SVM_features_CTRLvsCIN3 = pd.read_csv('./data/classifying_data/SVM_features_CTRLvsCIN3.csv', sep = ';')

# neg CTRL vs CIN2+/CIN2/CIN3
RF_features_negCTRLvsCIN2plus = pd.read_csv('./data/classifying_data/RF_features_negCTRLvsCIN2+.csv', sep = ';')
SVM_features_negCTRLvsCIN2 = pd.read_csv('./data/classifying_data/SVM_features_negCTRLvsCIN2.csv', sep = ';')
GradBoost_features_negCTRLvsCIN3 = pd.read_csv('./data/classifying_data/GradBoost_features_negCTRLvsCIN3.csv', sep = ';', index_col=False)

RF_features_CTRLvsCIN2plus = RF_features_CTRLvsCIN2plus['RF_features'].to_list()
SVM_features_CTRLvsCIN2 = SVM_features_CTRLvsCIN2['SVM_features'].to_list()
SVM_features_CTRLvsCIN3 = SVM_features_CTRLvsCIN3['SVM_features'].to_list()

RF_features_negCTRLvsCIN2plus = RF_features_negCTRLvsCIN2plus['RF_features'].to_list()
SVM_features_negCTRLvsCIN2 = SVM_features_negCTRLvsCIN2['SVM_features'].to_list()
GradBoost_features_negCTRLvsCIN3 = GradBoost_features_negCTRLvsCIN3['GradBoost_features'].to_list()

Cin2plusComparisons = {
    "CTRL vs CIN2+": set(RF_features_CTRLvsCIN2plus),
    "neg CTRL vs CIN2+": set(RF_features_negCTRLvsCIN2plus),
}
venn(Cin2plusComparisons, cmap="plasma")
plt.show()

negCtrls_comparisons_dict = {
    "CTRL vs CIN2": set(SVM_features_CTRLvsCIN2),
    "CTRL vs CIN3": set(SVM_features_CTRLvsCIN3),
    "neg CTRL vs CIN2": set(SVM_features_negCTRLvsCIN2),
    "neg CTRL vs CIN3": set(GradBoost_features_negCTRLvsCIN3), 
}

venn(negCtrls_comparisons_dict, cmap="plasma")
plt.show()


ctrlCin2ctrlCin2plus = {
    "CTRL vs CIN2+": set(RF_features_CTRLvsCIN2plus),
    "CTRL vs CIN2": set(SVM_features_CTRLvsCIN2),
}

# venn(ctrlCin2ctrlCin2plus, cmap="plasma")
# plt.show()

ctrlCin3ctrlCin2plus = {
    "CTRL vs CIN2+": set(RF_features_CTRLvsCIN2plus),
    "CTRL vs CIN3": set(SVM_features_CTRLvsCIN3),
}

# venn(ctrlCin3ctrlCin2plus, cmap="plasma")
# plt.show()


ctrlCin3ctrlCin2 = {
    "CTRL vs CIN2": set(SVM_features_CTRLvsCIN2),
    "CTRL vs CIN3": set(SVM_features_CTRLvsCIN3),
}

# venn(ctrlCin3ctrlCin2, cmap="plasma")
# plt.show()

allctrl_comparisons_dict = {
    "CTRL vs CIN2+": set(RF_features_CTRLvsCIN2plus),
    "CTRL vs CIN2": set(SVM_features_CTRLvsCIN2),
    "CTRL vs CIN3+": set(SVM_features_CTRLvsCIN3),
}

venn(allctrl_comparisons_dict, cmap="viridis", fontsize=20, alpha=0.425)
plt.savefig('./figures/VennDiagram_allCTRLsvsCIN2pCIN2CIN3.png')
plt.show()

negCtrls_comparisons_dict = {
    "neg CTRL vs CIN2+": set(RF_features_negCTRLvsCIN2plus),
    "neg CTRL vs CIN2": set(SVM_features_negCTRLvsCIN2),
}

# venn(negCtrls_comparisons_dict, cmap="plasma")
# plt.show()

negCtrls_comparisons_dict = {
    "neg CTRL vs CIN2+": set(RF_features_negCTRLvsCIN2plus),
    "neg CTRL vs CIN3": set(GradBoost_features_negCTRLvsCIN3), 
}

# venn(negCtrls_comparisons_dict, cmap="plasma")
# plt.show()

negCtrls_comparisons_dict = {
    "neg CTRL vs CIN2": set(SVM_features_negCTRLvsCIN2),
    "neg CTRL vs CIN3": set(GradBoost_features_negCTRLvsCIN3), 
}

# venn(negCtrls_comparisons_dict, cmap="plasma")
# plt.show()

negCtrls_comparisons_dict = {
    "neg CTRL vs CIN2+": set(RF_features_negCTRLvsCIN2plus),
    "neg CTRL vs CIN2": set(SVM_features_negCTRLvsCIN2),
    "neg CTRL vs CIN3+": set(GradBoost_features_negCTRLvsCIN3), 
}

venn(negCtrls_comparisons_dict, cmap="plasma",  fontsize=20, alpha=0.425)
plt.savefig('./figures/VennDiagram_allnegCTRLsvsCIN2pCIN2CIN3.png')
plt.show()

all6_comparisons_dict = {
    "CTRL vs CIN2+": set(RF_features_CTRLvsCIN2plus),
    "CTRL vs CIN2": set(SVM_features_CTRLvsCIN2),
    "CTRL vs CIN3": set(SVM_features_CTRLvsCIN3),
    "neg CTRL vs CIN2+": set(RF_features_negCTRLvsCIN2plus),
    "neg CTRL vs CIN2": set(SVM_features_negCTRLvsCIN2),
    "neg CTRL vs CIN3": set(GradBoost_features_negCTRLvsCIN3), 
}


pseudovenn(all6_comparisons_dict, cmap="plasma")
plt.savefig('./figures/VennDiagram_all6comparisons.png')
plt.show()

