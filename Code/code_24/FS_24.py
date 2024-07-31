from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2,f_classif
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats



def binarizeSampleDataByLabel(OTU_acrossSample,y):
    if not (len(set(y))==2):
        print("ERROR! Getting Non Binary Label Class!!",len(set(y)),set(y))
        return
    elif not (len(OTU_acrossSample)==len(y)):
        print("ERROR! Length of OTU and label have difference length")
        return
    else:
        posList = []
        negList = []
        negLabel=""
        if "no" in str(list(set(y))[0]).lower():
            negLabel=list(set(y))[0]
        else:
            negLabel = list(set(y))[1]
        for i in range(len(y)):
            if y[i]==negLabel:
                negList.append(OTU_acrossSample[i])
            else:
                posList.append(OTU_acrossSample[i])
        #print("negative label is ",negLabel)
        return posList,negList


def multiLabelFeatureWeighting(X,yList):#X is the ASV data array, ylist is the label; return the weights, what is weights?
    y_info_index_List=[]
    X_transpose=np.transpose(X)
    weighted_OTU_h_index_List_sum=[0]*(np.shape(X)[1])
    for y in yList:
        OTU_h_score_List=[]
        for OTU_across_sample in X_transpose:
            posList,negList=binarizeSampleDataByLabel(OTU_across_sample,y)
            try:
                OTU_h_score=stats.kruskal(posList,negList)[0]
            except:
                OTU_h_score=0
            OTU_h_score_List.append(OTU_h_score)
        y_info_index=sum(OTU_h_score_List)
        y_info_index_List.append(y_info_index)
        # print(y_info_index_List
        weighted_OTU_h_index_List=[ i*y_info_index for i in  OTU_h_score_List]
        weighted_OTU_h_index_List_sum=[x+y for (x,y) in zip(weighted_OTU_h_index_List,weighted_OTU_h_index_List_sum)]
    return (weighted_OTU_h_index_List_sum)

def elbowPoint(scores,curveType="elbow"):
    curveName="convex"
    if curveType!="elbow":
        curveName="concave"

    from kneed import KneeLocator
    kn = KneeLocator(range(1, len(scores)+1), scores, curve='convex', direction='decreasing',interp_method="polynomial")
    return kn.knee

def feature_select(X,y,topFeature=150): # given the ASV data array and label, return the weights, selected array and selected OTU names 
    weights=multiLabelFeatureWeighting(X,y)
    n_weights = len(weights)
    scores=(sorted(weights,reverse=True))
    eps=elbowPoint(scores)

    weightRanks=np.flip(np.argsort(weights))
    selectedOTU_index=weightRanks[:eps]
    X_FS=(X[:,selectedOTU_index])
    return X_FS,selectedOTU_index, weights, eps





def plotWeightedIndex(weights,xKnee=0):
    plt.figure(dpi=300)
    plt.plot(sorted(weights,reverse=True),linestyle=":")
    plt.ylabel("Weighted Info Index")
    plt.xlabel("Ranked OTUs")
    #plt.axis("square")

    if xKnee!=0:
        plt.axvline(x=xKnee, color='green', linestyle='dashed', label="Elbow Point", lw=1)
    plt.legend()
    plt.show()