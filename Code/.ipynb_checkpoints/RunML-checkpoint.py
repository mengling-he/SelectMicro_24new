
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import ADASYN
from imblearn.combine import SMOTEENN
from sklearn import preprocessing, __all__, svm
from imblearn.over_sampling import SMOTE
import matplotlib
from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.metrics import auc, roc_auc_score, roc_curve
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import StratifiedKFold, cross_val_score


def CFValidation_AUCstatistic(X,y,classifier = svm.SVC(kernel='linear', probability=True),k=5):# test this
    cv = StratifiedKFold(n_splits=k, shuffle=True,random_state = 777)
    aucs = []
   # tprs = []
   # mean_fpr = np.linspace(0, 1, 100)
   # fig, ax = plt.subplots()
    for i, (train, test) in enumerate(cv.split(X, y)):
        classifier.fit(X[train], y[train])
        # Predict probabilities
        y_prob = classifier.predict_proba(X[test])[:, 1]
        # Calculate AUC
        auc = roc_auc_score(y[test], y_prob)
        aucs.append(auc)
    #return aucs.mean(), aucs.std()
    mean_auc = np.mean(aucs)
    return mean_auc

# calculate the stratified-cross-validation score mean and std
# score : accuracy,f1,roc_auc...
# this function not be tested
def CFValidation_score(X, y,k=5,classifier = svm.SVC(kernel='linear', probability=True),score='accuracy'):# test this
    cv = StratifiedKFold(n_splits=k, shuffle=True,random_state = 777)
    acs = []
    results = cross_val_score(classifier, X_train=X, y_train=y, cv = cv,scoring=score)#scoring, only a single metric is permitted
    return results.mean(),results.std()



def CrossFoldValidation_AUC(X,y,classifier = svm.SVC(kernel='linear', probability=True),title="AUC"):
    cv = StratifiedKFold(n_splits=5)
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    fig, ax = plt.subplots()
    for i, (train, test) in enumerate(cv.split(X, y)):
        classifier.fit(X[train], y[train])
        viz = RocCurveDisplay.from_estimator(classifier, X[test], y[test],
                             name='ROC fold {}'.format(i),
                             alpha=0, lw=0,ax=ax)
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
            label='Chance', alpha=.8)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color='b',
            label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                    label=r'$\pm$ 1 std. dev.')
    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
           title=title)
    ax.legend(loc="lower right")
    plt.gcf()
    handles, labels = plt.gca().get_legend_handles_labels()
    handles = handles[6:]
    labels = labels[6:]
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    plt.show()
    
from sklearn.metrics import matthews_corrcoef
def make_confusion_matrix(cf,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=None,
                          cmap='Blues',
                          title=None):

    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names) == cf.size:
        print("CF matrix has group names")
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.2%}".format(value) for value in cf.flatten() / np.sum(cf)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels, group_counts, group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0], cf.shape[1])

    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        # Accuracy is sum of diagonal divided by total observations
        accuracy = np.trace(cf) / float(np.sum(cf))

        # if it is a binary confusion matrix, show some more stats
        if len(cf) == 2:
            # Metrics for Binary Confusion Matrices
            precision = cf[1, 1] / sum(cf[:, 1])
            recall = cf[1, 1] / sum(cf[1, :])
            f1_score = 2 * precision * recall / (precision + recall)
            tn, fp, fn, tp=cf.ravel()
            MCC_Score=((tp*tn)-(fp*fn))/np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nMCC Coefficient={:0.3f}".format(
                accuracy, precision, recall, MCC_Score)
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""

    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize == None:
        # Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if xyticks == False:
        # Do not show categories if xyticks is False
        categories = False

    # MAKE THE HEATMAP VISUALIZATION
    plt.figure(figsize=figsize,dpi=150)
    sns.heatmap(cf, annot=box_labels, fmt="", cmap=cmap, cbar=cbar, xticklabels=categories, yticklabels=categories)

    if xyplotlabels:
        plt.ylabel('True label')
        plt.xlabel('Predicted label' + stats_text)
    else:
        plt.xlabel(stats_text)

    if title:
        plt.title(title)
    plt.show()
from sklearn.ensemble import RandomForestClassifier

def classificationHeatMap(X,y,data_classes,classifier=RandomForestClassifier(random_state=0),title="title"):
    from sklearn.metrics import confusion_matrix
    iterations=5
    cv = StratifiedKFold(n_splits=iterations, shuffle=True,random_state = 777)
    cf_matrix_all=[]
    for i, (train, test) in enumerate(cv.split(X, y)):
        classifier.fit(X[train], y[train])
        y_pred = classifier.predict(X[test])
        cf_matrix = confusion_matrix(y[test], y_pred)
        cf_matrix_all.append(cf_matrix)
    #cf_matrix_sum = sum(cf_matrix_all) # this works!
    cf_matrix_sum=cf_matrix_all[0]
    for i in range(1,iterations):
        cf_matrix_sum=np.add(cf_matrix_sum,cf_matrix_all[i])
    labels = ["True Neg","False Pos","False Neg","True Pos"]
    make_confusion_matrix(cf_matrix_sum,
                          group_names=labels,
                          categories=data_classes,
                          cmap="Blues", title=title,cbar=False,figsize=(4,3))


from sklearn.feature_selection import chi2,f_classif
from sklearn.feature_selection import SelectKBest
def LeaveOneOut(matrix, labels, classfierName, vis=True):
    if len(set(labels)) < 2:
        print("Label only contains extremely unbalanced label or less!")
        return -1
    predictionLabels = []
    for i in range(len(matrix)):
        trainMatrix = np.delete(matrix, i, 0)
        trainLabels = np.delete(labels, i, 0)
        global classifier
        clf=classifier
        clf.fit(trainMatrix, trainLabels)
        testMatrix = []
        testMatrix.append(matrix[i])
        predictionLabels.append(clf.predict(testMatrix)[0])
    return predictionLabels

def getAccuracies(labels,predictions):
    T=0
    F=0
    for i in range(len(labels)):
        if labels[i]==predictions[i]:
            T+=1
        else:
            F+=1
    return(T/float(len(labels)))

def FS(X,y,topFeature=60,CLS="SVM"):
    global scoringFunction
    FS = SelectKBest(f_classif, k=topFeature)
    X_FS = FS.fit_transform(X, y)
    index_mask=FS.get_support()
    return X_FS,index_mask

from sklearn.metrics import f1_score
def plotFeatureTrend(matrix, labels, minFeaure=50, maxFeature=500, title="", increment=1,classifier = svm.SVC(kernel='linear', probability=True)):
    kList = []
    cur = minFeaure
    while cur <= maxFeature:
        kList.append(cur)
        cur += increment
    accuList = []
    precisionList = []
    global scoringFunction
    for myK in kList:
        print(myK)
        FS = SelectKBest(f_classif, k=myK)
        selectedMatrix = FS.fit_transform(matrix, labels)
        predictions = classifier.fit(selectedMatrix,labels).predict(selectedMatrix)
        accu = getAccuracies(labels, predictions)
        f1score = f1_score(labels, predictions)
        accuList.append(accu)
        precisionList.append(f1score)
    fig = plt.figure(dpi=150)
    plt.ylim([0, 1])
    plt.title(title)
    plt1 = plt.plot(kList, accuList)
    plt.xlabel("Number of Features Selected")
    plt.ylabel("F1 Score")
    ax2 = plt.twinx()
    plt2 = plt.plot(kList, precisionList, c="red")
    ax2.set_ylabel("F1 Score")
    ax2.set_ylim([0, 1])
    plt.legend((plt1[0], plt2[0]), ("Accurary", "F1 Score"))
    plt.show()
    return kList, accuList


#load Data using loadData.py


def runSmokyPipeLine(X,y,ADSP=False,vis=True,topFeature=100):
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)
    data_classes = list(le.classes_)
    le.fit(y)
    y = le.transform(y)

    # adaptive sampling, smote
    n_samples = np.shape(X)[0]
    n_classes = len(set(y))
    if ADSP:
        ada = SMOTE(random_state=0)
        X,y = ada.fit_resample(X, y)
    # linear FS using chi2

    X_train, X_test, y_train, y_test = train_test_split(X, y,train_size=0.7,test_size=0.3)
    X_FS,index_mask = FS(X, y, topFeature=topFeature)
    classifier = svm.SVC(kernel='linear', probability=True)
    # Train Test Split
    if vis:
        # Instaniate the classification model and visualizer
        CrossFoldValidation_AUC(X, y, classifier=classifier)
        # plotFeatureTrend(X,y,maxFeature=100000,increment=10000)
        CrossFoldValidation_AUC(X_FS, y, classifier=classifier)
        classificationHeatMap(X, y,data_classes)
        classificationHeatMap(X_FS, y,data_classes)
    return index_mask


import re

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def plotVenn3(set1,set2,set3,groupname1,groupname2,groupname3):
    import matplotlib.pyplot as plt
    from matplotlib_venn import venn3
    v=venn3([set1, set2, set3], (groupname1, groupname2, groupname3))
    plt.show()
    setDict=dict()
    sets=[set1,set2,set3]
    groupnames=[groupname1,groupname2,groupname3]
    for s1 in [0,1,2]:
        for s2 in [0,1,2]:
            if s1<=s2:
                continue
            key=groupnames[s1]+" & " +groupnames[s2]
            overlappedList=(list((sets[s1]&sets[s2])))
            overlappedList.sort(key=natural_keys)
            setDict[key]=overlappedList
    with open('overlapsets', 'w') as data:
        data.write(str(setDict))

    centerList=(set1&set2&set3)
    return centerList


def writeXyToCsv(X,y,ASVs,name="Label",fname="ASV.csv"):
    df=pd.DataFrame(X)
    df.columns=ASVs
    df[name]=y
    df.to_csv(fname)


import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from itertools import chain
def plotBinaryHeapmap(df,ASVs):
    df = df.groupby("Burn")
    fig, ax = plt.subplots()
    # define the colors
    cmap = mpl.colors.ListedColormap(['w', 'k'])
    # create a normalize object the describes the limits of
    # each color
    bounds = [0., 0.5, 1.]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    # plot it
    ax.imshow(df.T, interpolation='none', cmap=cmap, norm=norm)


def plotPresenseRatio(X,label,featurenames,posLabel,posText="",negText="",thresholdPercent=0.90,abundanceCutoff=0.01,entries=15):
    import matplotlib as mpl
    mpl.rcParams['figure.dpi'] = 300

    presenceCntPos = []
    presenceCntNeg = []
    X=X.T
    if abundanceCutoff==0:
        flatten_list = list(chain.from_iterable(X))
        flatten_list_sorted=sorted(flatten_list)
        abundanceCutoff=flatten_list[int(len(flatten_list_sorted)*float(threshold))]

    if posText=="" or negText=="":
        posText=posLabel
        negText="Not "+posLabel

    for k in range(len(X)):## for each OTU
        OTUs = X[k]## the samples for this OTU
        pos = 0
        neg = 0
        for i in range(len(OTUs)):
            if label[i] == posLabel:
                if OTUs[i] > abundanceCutoff:# if the value of OTU exceed the abundanceCutoff
                    pos += 1
            else:
                if OTUs[i] > abundanceCutoff:
                    neg += 1
        presenceCntPos.append(pos)# len= # of samples; each value is the number of OTUs that exceed the abundanceCutoff for Pos/Neg
        presenceCntNeg.append(neg)
    all_pos_label_cnt=list(label).count(posLabel)
    all_neg_label_cnt=len(label)-all_pos_label_cnt
    print(all_pos_label_cnt,all_neg_label_cnt)# these 3  lines can use  value_count
    presenceRatioPos=[float(x)/all_pos_label_cnt for x in presenceCntPos]# each element is for each OTU; shows the ratio of abundanced pos samples over all pos sample 
    presenceRatioNeg=[float(x)/all_neg_label_cnt for x in presenceCntNeg]

    import matplotlib.pyplot as plt
    y = range(entries)
    fig, axes = plt.subplots(ncols=2, sharey=True)
    axes[0].barh(y, presenceRatioPos, align='center', color='#ff7f00')
    axes[1].barh(y, presenceRatioNeg, align='center', color='#377eb8')
    axes[0].set_xlabel("Presence Ratio in "+posText)
    axes[1].set_xlabel("Presences Ratio "+negText)

    axes[0].set_xlim(0,0.4)
    axes[1].set_xlim(0,0.4)
    axes[0].invert_xaxis()

    axes[0].set(yticks=y, yticklabels=[])
    for yloc, selectedASVs in zip(y, featurenames):
        axes[0].annotate(selectedASVs, (0.5, yloc), xycoords=('figure fraction', 'data'),
                         ha='center', va='center', fontsize=9)
    fig.tight_layout(pad=2.0)
    plt.show()

def normalizingMatrix(data,cutOff=0.01):
    data=np.array(data)
    for i in range(np.shape(data)[0]):
        for j in range(np.shape(data)[1]):
            if data[i][j]>=cutOff:
                data[i][j]=1
            else:
                data[i][j]=0
    return data

def listToRanks(lst):
    return [sorted(lst).index(x) for x in lst]

def normalizingMatrixToRanks(data,cutOff=0.01):
    data=np.array(data)
    for i in range(np.shape(data)[0]):
        for j in range(np.shape(data)[1]):
            if data[i][j]>=cutOff:
                continue
            else:
                data[i][j]=0
    for i in range(np.shape(data)[0]):
        data[i]=listToRanks(data[i])
    return data

def LassoFeatureSelection(X,y):
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y= le.fit_transform(y)
    clf = linear_model.Lasso(alpha=0.1, tol=0.01)
    clf.fit(X, y)
    coefficients = (clf.coef_)
    importance = np.abs(coefficients)
    selectedOTU_index_boolean = (importance > 0)
    selectedOTU_index=list(np.where(selectedOTU_index_boolean))
    X=X[:,selectedOTU_index]
    return np.squeeze(X, axis=1)




from sklearn.metrics import confusion_matrix
def binaryLabelVis(y, y_pred, categories, title="title"):
    cf_matrix = confusion_matrix(y, y_pred)
    labels = ["True Neg", "False Pos", "False Neg", "True Pos"]
    make_confusion_matrix(cf_matrix,
                          group_names=labels,
                          categories=categories,
                          cmap="binary", title=title)
    

