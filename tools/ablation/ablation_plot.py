from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score

import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
params = {'legend.fontsize': 13,
         'axes.labelsize': 15,
         'axes.titlesize':15,
         'xtick.labelsize':15,
         'ytick.labelsize':15} 
pylab.rcParams.update(params)
import os
import torch
from os.path import join
from collections import OrderedDict
import numpy as np

result_list = {"d_base":'./experiments/db1_new/result.npy',
               "d_up1":'./experiments/d_up1/result.npy',
               "d_total":'./experiments/d_total/result.npy'}
save_path = './experiments/Drive_ablation'
if not os.path.exists(save_path): os.makedirs(save_path)

# ===============AUC ROC===============
plt.figure()                
for name,path in result_list.items():
    data = np.load(path)
    target, output = data
    AUC_ROC= roc_auc_score(target, output)
    print(name + ": AUC of ROC curve: " + str(AUC_ROC))

    fpr, tpr, thresholds = roc_curve(target, output)
    plt.plot(fpr, tpr, '-', label=name + ' (AUC = %0.4f)' % AUC_ROC)
    plt.legend(loc="lower right")

plt.title('ROC curve')
plt.xlabel("FPR (False Positive Rate)")
plt.ylabel("TPR (True Positive Rate)")

plt.savefig("{}/ROC.png".format(save_path))

# ===============AUC_PR==================
plt.figure()                
for name,path in result_list.items():
    data = np.load(path)
    target, output = data

    precision, recall, thresholds = precision_recall_curve(target, output)
    precision = np.fliplr([precision])[0]  # so the array is increasing (you won't get negative AUC)
    recall = np.fliplr([recall])[0]  # so the array is increasing (you won't get negative AUC)
    AUC_pr = np.trapz(precision, recall)
    print(name + ": AUC of P-R curve: " + str(AUC_pr))

    plt.plot(recall, precision, '-', label=name+' (AUC = %0.4f)' % AUC_pr)
    plt.legend(loc="lower right")

plt.title('Precision-Recall curve')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.savefig("{}/PRC.png".format(save_path))
