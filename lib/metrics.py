"""
This part contains functions related to the calculation of performance indicators
"""
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
import os
import torch
from os.path import join
import numpy as np
from collections import OrderedDict
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
params = {'legend.fontsize': 13,
         'axes.labelsize': 15,
         'axes.titlesize':15,
         'xtick.labelsize':15,
         'ytick.labelsize':15} # define pyplot parameters
pylab.rcParams.update(params)
#Area under the ROC curve

class Evaluate():
    def __init__(self,save_path=None):
        self.target = None
        self.output = None
        self.save_path = save_path
        if self.save_path is not None:
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)
        self.threshold_confusion = 0.5

    # Add data pair (target and predicted value)
    def add_batch(self,batch_tar,batch_out):
        batch_tar = batch_tar.flatten()
        batch_out = batch_out.flatten()

        self.target = batch_tar if self.target is None else np.concatenate((self.target,batch_tar))
        self.output = batch_out if self.output is None else np.concatenate((self.output,batch_out))
    
    # Plot ROC and calculate AUC of ROC
    def auc_roc(self,plot=False):
        AUC_ROC = roc_auc_score(self.target, self.output)
        # print("\nAUC of ROC curve: " + str(AUC_ROC))
        if plot and self.save_path is not None:
            fpr, tpr, thresholds = roc_curve(self.target, self.output)
            # print("\nArea under the ROC curve: " + str(AUC_ROC))
            plt.figure()
            plt.plot(fpr, tpr, '-', label='Area Under the Curve (AUC = %0.4f)' % AUC_ROC)
            plt.title('ROC curve')
            plt.xlabel("FPR (False Positive Rate)")
            plt.ylabel("TPR (True Positive Rate)")
            plt.legend(loc="lower right")
            plt.savefig(join(self.save_path , "ROC.png"))
        return AUC_ROC

    # Plot PR curve and calculate AUC of PR curve
    def auc_pr(self,plot=False):
        precision, recall, thresholds = precision_recall_curve(self.target, self.output)
        precision = np.fliplr([precision])[0]
        recall = np.fliplr([recall])[0]
        AUC_pr = np.trapz(precision, recall)
        # print("\nAUC of P-R curve: " + str(AUC_pr))
        if plot and self.save_path is not None:

            plt.figure()
            plt.plot(recall, precision, '-', label='Area Under the Curve (AUC = %0.4f)' % AUC_pr)
            plt.title('Precision - Recall curve')
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.legend(loc="lower right")
            plt.savefig(join(self.save_path ,"Precision_recall.png"))
        return AUC_pr

    # Accuracy, specificity, sensitivity, precision can be obtained by calculating the confusion matrix
    def confusion_matrix(self):
        #Confusion matrix
        y_pred = self.output>=self.threshold_confusion
        confusion = confusion_matrix(self.target, y_pred)
        # print(confusion)
        accuracy = 0
        if float(np.sum(confusion))!=0:
            accuracy = float(confusion[0,0]+confusion[1,1])/float(np.sum(confusion))
        # print("Global Accuracy: " +str(accuracy))
        specificity = 0
        if float(confusion[0,0]+confusion[0,1])!=0:
            specificity = float(confusion[0,0])/float(confusion[0,0]+confusion[0,1])
        # print("Specificity: " +str(specificity))
        sensitivity = 0
        if float(confusion[1,1]+confusion[1,0])!=0:
            sensitivity = float(confusion[1,1])/float(confusion[1,1]+confusion[1,0])
        # print("Sensitivity: " +str(sensitivity))
        precision = 0
        if float(confusion[1,1]+confusion[0,1])!=0:
            precision = float(confusion[1,1])/float(confusion[1,1]+confusion[0,1])
        # print("Precision: " +str(precision))
        return confusion,accuracy,specificity,sensitivity,precision

    # Jaccard similarity index
    def jaccard_index(self):
        pass
        # jaccard_index = jaccard_similarity_score(y_true, y_pred, normalize=True)
        # print("\nJaccard similarity score: " +str(jaccard_index))

    # calculating f1_score
    def f1_score(self):
        pred = self.output>=self.threshold_confusion
        F1_score = f1_score(self.target, pred, labels=None, average='binary', sample_weight=None)
        # print("F1 score (F-measure): " +str(F1_score))
        return F1_score

    # Save performance results to specified file
    def save_all_result(self,plot_curve=True,save_name=None):
        #Save the results
        AUC_ROC = self.auc_roc(plot=plot_curve)
        AUC_pr  = self.auc_pr(plot=plot_curve)
        F1_score = self.f1_score()
        confusion,accuracy, specificity, sensitivity, precision = self.confusion_matrix()
        if save_name is not None:
            file_perf = open(join(self.save_path, save_name), 'w')
            file_perf.write("AUC ROC curve: "+str(AUC_ROC)
                            + "\nAUC PR curve: " +str(AUC_pr)
                            # + "\nJaccard similarity score: " +str(jaccard_index)
                            + "\nF1 score: " +str(F1_score)
                            +"\nAccuracy: " +str(accuracy)
                            +"\nSensitivity(SE): " +str(sensitivity)
                            +"\nSpecificity(SP): " +str(specificity)
                            +"\nPrecision: " +str(precision)
                            + "\n\nConfusion matrix:"
                            + str(confusion)
                            )
            file_perf.close()
        return OrderedDict([("AUC_ROC",AUC_ROC),("AUC_PR",AUC_pr),
                            ("f1-score",F1_score),("Acc",accuracy),
                            ("SE",sensitivity),("SP",specificity),
                            ("precision",precision)
                            ])