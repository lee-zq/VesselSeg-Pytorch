from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.pylab as pylab
import os
from os.path import join
from collections import OrderedDict
import numpy as np

params = {'legend.fontsize': 12,
         'axes.labelsize': 15,
         'axes.titlesize':15,
         'xtick.labelsize':13,
         'ytick.labelsize':13} # define pyplot parameters
pylab.rcParams.update(params)
#Area under the ROC curve

result_list = {"d_base":'./experiments/db1_new/result.npy',
               "d_up1":'./experiments/d_up1/result.npy',
               "d_total":'./experiments/d_total/result.npy'}
save_path = './experiments/Drive_ablation'

if not os.path.exists(save_path): os.makedirs(save_path)

# ===============AUC ROC===============
fig, ax = plt.subplots(1, 1)
axins = ax.inset_axes((0.45, 0.45, 0.45, 0.45)) # 左下角坐标（x0, y0）,窗口大小（ width, height）   
axins.set_xlim(0.07, 0.11)
axins.set_ylim(0.93, 0.96)

mark_inset(ax, axins, loc1=3, loc2=1, fc="none", ec='k', lw=1) # 建立连线 loc1和loc2取值为1，2，3，4，对应右上，左上，左下，右下
for name,path in result_list.items():
    data = np.load(path[0])
    target, output = data
    AUC_ROC= roc_auc_score(target, output)
    print(name + ": AUC of ROC curve: " + str(AUC_ROC))

    fpr, tpr, thresholds = roc_curve(target, output)
    ax.plot(fpr, tpr, linestyle=path[1], label=name + ' (AUC = %0.4f)' % AUC_ROC)
    axins.plot(fpr, tpr, linestyle=path[1], label=name + ' (AUC = %0.4f)' % AUC_ROC)
    ax.legend(loc="lower right")

plt.title('ROC curve')
plt.xlabel("FPR (False Positive Rate)")
plt.ylabel("TPR (True Positive Rate)")
plt.savefig("{}/ROC1.png".format(save_path))

# ===============AUC_PR==================
fig, ax = plt.subplots(1, 1)
axins = ax.inset_axes((0.15, 0.45, 0.4, 0.4)) # 左下角坐标（x0, y0）,窗口大小（ width, height）   
axins.set_xlim(0.83,0.89)
axins.set_ylim(0.75,0.82)

mark_inset(ax, axins, loc1=1, loc2=4, fc="none", ec='k', lw=1) # 建立连线 loc1和loc2取值为1，2，3，4，对应右上，左上，左下，右下
for name,path in result_list.items():
    data = np.load(path[0])
    target, output = data

    precision, recall, thresholds = precision_recall_curve(target, output)
    precision = np.fliplr([precision])[0]  # so the array is increasing (you won't get negative AUC)
    recall = np.fliplr([recall])[0]  # so the array is increasing (you won't get negative AUC)
    AUC_pr = np.trapz(precision, recall)
    print(name + ": AUC of P-R curve: " + str(AUC_pr))

    ax.plot(recall, precision, linestyle=path[1], label=name+' (AUC = %0.4f)' % AUC_pr)
    axins.plot(recall, precision, linestyle=path[1], label=name+' (AUC = %0.4f)' % AUC_pr)
    ax.legend(loc="lower left")

plt.title('Precision-Recall curve')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.savefig("{}/PRC1.png".format(save_path))
