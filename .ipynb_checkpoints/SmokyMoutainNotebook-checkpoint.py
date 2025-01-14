import sys
sys.path.append('./Code')
import loadData 
import FS
import RunML
import pandas as pd
import numpy as np
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.model_selection import StratifiedKFold, cross_val_score



data,burn_label,un_label,duration_label,ASVs,df=loadData.loadSoilData("./Data/merge_proportion_asv_smoky_moutain.csv")

data=RunML.normalizingMatrixToRanks(data,cutOff=0.01)
print(np.shape(data))


yList=[list(burn_label),list(un_label),list(duration_label)]

weights=FS.multiLabelFeatureWeighting(data,yList)
scores=(sorted(weights,reverse=True))
eps=FS.elbowPoint(scores)

FS.plotWeightedIndex(weights,xKnee=eps)

topFeature_selected=eps
print(eps)

X_FS,selectedOTU_index=FS.feature_select(data,yList,topFeature=topFeature_selected)

#X_FS,selectedOTU_index,weights, eps=FS_24.feature_select(data,yList)
print (np.shape(X_FS))
print(len(weights))





clfs = {
    "SVM": svm.SVC(kernel='linear', probability=True),
    "Random Forest": RandomForestClassifier(random_state=0)
    #, "Logistic Regression": LogisticRegression()
}

targetLabel=un_label 
targetLabel=burn_label
targetLabel=duration_label


# Initialize a dictionary to hold AUC scores
auc_scores = {}
for name, clf in clfs.items():
    auc = RunML.CFValidation_AUCstatistic(data,targetLabel,classifier= clf )
    auc_scores[name] = auc
print("AllFeatures",auc_scores)


auc_scores = {}
for name, clf in clfs.items():
    auc = RunML.CFValidation_AUCstatistic(X_FS,targetLabel,classifier= clf )
    auc_scores[name] = auc
print("SelectMicro",auc_scores)


auc_list  = []
for ii in range(3):
    auc_scores = {}
    randomFeatures=random.sample(list(range(np.shape(data)[1])), topFeature_selected)
    X_randomFeatures=np.array(data)[:,randomFeatures]
    for name, clf in clfs.items():
        auc = RunML.CFValidation_AUCstatistic(X_randomFeatures,targetLabel,classifier= clf )
        auc_scores[name] = auc
    auc_list.append(auc_scores)
auc_pd = pd.DataFrame(auc_list)

print("Random Selection",auc_pd.mean())


auc_list  = []
for ii in range(3):
    auc_scores = {}
    X_Lasso=RunML.LassoFeatureSelection(data,targetLabel)
    for name, clf in clfs.items():
        auc = RunML.CFValidation_AUCstatistic(X_Lasso,targetLabel,classifier= clf )
        auc_scores[name] = auc
    auc_list.append(auc_scores)
auc_pd = pd.DataFrame(auc_list)


print("LASSO Selection",auc_pd.mean())





















########check function


# confusion matrix
from sklearn.metrics import confusion_matrix
cv = StratifiedKFold(n_splits=5, shuffle=True,random_state = 777)
cf_matrix_all=[]
for i, (train, test) in enumerate(cv.split(X_FS, targetLabel)):
    clf.fit(X_FS[train], targetLabel[train])
    y_pred = clf.predict(X_FS[test])
    cf_matrix = confusion_matrix(targetLabel[test], y_pred)
    cf_matrix_all.append(cf_matrix)


cf_matrix_sum=cf_matrix_all[0]

for i in range(1,5):
    cf_matrix_sum=np.add(cf_matrix_sum,cf_matrix_all[i])

labels = ["True Neg","False Pos","False Neg","True Pos"]
make_confusion_matrix(cf_matrix_sum,
                        group_names=labels,
                        categories=data_classes,
                        cmap="Blues", title=title,cbar=False,figsize=(4,3))
















# first 15 features
entries=15
selectedOTU_index_15=selectedOTU_index[:entries]
X_FS_15=data[:,selectedOTU_index_15]
X_FS_15=X_FS_15.T
presenceCntPos = []
presenceCntNeg = []


label = targetLabel
posLabel = "Natural"



abundanceCutoff=0.01
for k in range(len(X_FS_15)):# for each OTU
    OTUs = X_FS_15[k]# the samples for this OTU
    pos = 0
    neg = 0
    for i in range(len(OTUs)):
        if label[i] == posLabel:
            if OTUs[i] > abundanceCutoff:# if the value of OTU exceed the abundanceCutoff
                pos += 1
        else:
            if OTUs[i] > abundanceCutoff:
                neg += 1
    presenceCntPos.append(pos)# len= # of OTU; each value is the number of samples that exceed the abundanceCutoff for Pos/Neg
    presenceCntNeg.append(neg)


all_pos_label_cnt=list(label).count(posLabel)
all_neg_label_cnt=len(label)-all_pos_label_cnt
print(all_pos_label_cnt,all_neg_label_cnt)# these 3  lines can use  value_count
presenceRatioPos=[float(x)/all_pos_label_cnt for x in presenceCntPos]
presenceRatioNeg=[float(x)/all_neg_label_cnt for x in presenceCntNeg]

import matplotlib.pyplot as plt
y = range(entries)
fig, axes = plt.subplots(ncols=2, sharey=True)
axes[0].barh(y, presenceRatioPos, align='center', color='#ff7f00')
axes[1].barh(y, presenceRatioNeg, align='center', color='#377eb8')
axes[0].set_xlabel("Presence Ratio in "+posText)
axes[1].set_xlabel("Presences Ratio "+negText)
plt.show()

axes[0].set_xlim(0,0.4)
axes[1].set_xlim(0,0.4)
axes[0].invert_xaxis()

axes[0].set(yticks=y, yticklabels=[])
for yloc, selectedASVs in zip(y, featurenames):
    axes[0].annotate(selectedASVs, (0.5, yloc), xycoords=('figure fraction', 'data'),
                        ha='center', va='center', fontsize=9)
fig.tight_layout(pad=2.0)
plt.show()



