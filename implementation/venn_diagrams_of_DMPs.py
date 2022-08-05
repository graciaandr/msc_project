import matplotlib.pyplot as plt
from matplotlib_venn import venn2, venn3, venn3_circles
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

col_one_list = RF_features_CTRLvsCIN2plus['RF_features'].to_list()
col_two_list = SVM_features_CTRLvsCIN2['SVM_features'].to_list()
col_three_list = SVM_features_CTRLvsCIN3['SVM_features'].to_list()

col_four_list = RF_features_negCTRLvsCIN2plus['RF_features'].to_list()
col_five_list = SVM_features_negCTRLvsCIN2['SVM_features'].to_list()
col_six_list = GradBoost_features_negCTRLvsCIN3['GradBoost_features'].to_list()

set1 = set(col_one_list)
set2 = set(col_two_list)
set3 = set(col_three_list)
set4 = set(col_four_list)
set5 = set(col_five_list)
set6 = set(col_six_list)

# venn2([set1, set2], ("CTRL vs CIN2+", "CTRL vs CIN2"))
# plt.show()

# venn2([set1, set3], ("CTRL vs CIN2+", "CTRL vs CIN3"))
# plt.show()

# venn2([set2, set3], ("CTRL vs CIN2", "CTRL vs CIN3"))
# plt.show()

# venn3([set1, set2, set3], ("CTRL vs CIN2+", "CTRL vs CIN2", "CTRL vs CIN3"))
# plt.savefig('./figures/venn_3comp_CTRL.png')
# plt.show()

# print(set4)
# print(set5)
vd = venn3([set4, set5, set6], ("neg CTRL vs CIN2+", "neg CTRL vs CIN2", "neg CTRL vs CIN3"))

lbl = vd.get_label_by_id("A")
x, y = lbl.get_position()
lbl.set_position((x, y+0.82))  

lbl = vd.get_label_by_id("C")
x, y = lbl.get_position()
lbl.set_position((x, y-0.1))  

lbl = vd.get_label_by_id("B")
x, y = lbl.get_position()
lbl.set_position((x, y-0.1))  
plt.savefig('./figures/venn_3comp_negCTRL.png')
plt.show()