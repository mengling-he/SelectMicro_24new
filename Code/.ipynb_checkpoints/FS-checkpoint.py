
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2,f_classif
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('./Code')
import loadData
import pandas as pd
from scipy import stats
'''
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

def multiLabelFeatureWeighting(X,yList):
    # print(np.shape(X))
    from scipy import stats
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

def singleLabelFeatureWeighting(X,y):
    from scipy import stats
    X_transpose=np.transpose(X)
    weighted_OTU_h_index_List_sum=[0]*(np.shape(X)[1])
    OTU_h_score_List=[]
    for OTU_across_sample in X_transpose:
        posList,negList=binarizeSampleDataByLabel(OTU_across_sample,y)
        try:
            OTU_h_score=stats.kruskal(posList,negList)[0]
        except:
            OTU_h_score=0
        OTU_h_score_List.append(OTU_h_score)
    return (OTU_h_score_List)

def feature_select(X,y,topFeature=150):
    weights=multiLabelFeatureWeighting(X,y)
    weightRanks=np.flip(np.argsort(weights))
    selectedOTU_index=weightRanks[:topFeature]
    X_FS=(X[:,selectedOTU_index])
    return X_FS,selectedOTU_index

def feature_select_singleLabel(X,yList,topFeature=150):
    weights=singleLabelFeatureWeighting(X,yList)
    weightRanks=np.flip(np.argsort(weights))
    selectedOTU_index=weightRanks[:topFeature]
    X_FS=(X[:,selectedOTU_index])


    return X_FS,selectedOTU_index


def elbowPoint(scores,curveType="elbow"):
    curveName="convex"
    if curveType!="elbow":
        curveName="concave"

    from kneed import KneeLocator
    kn = KneeLocator(range(1, len(scores)+1), scores, curve='convex', direction='decreasing',interp_method="polynomial")

    kn.plot_knee_normalized()
    kn.plot_knee()

    return kn.knee





def test_main():
    data, burn_label, un_label, duration_label, ASVs, df = loadData.loadSoilData("../Data/merge_soil.csv")
    yList=[list(burn_label),list(un_label),list(duration_label)]
    weights=multiLabelFeatureWeighting(data,yList)
    feature_select(data,yList)
'''








# new FS pipeline-------------

# 1. Relative abundance matrix
def relative_abundance(data):# if the input is the original abundance matrix, will convert it into relative abundance matrix; for missing values will return to 0
    # Convert input to a numpy array
    data = np.array(data) 
    # Iterate over each row (sample) in the array
    total_per_sample = np.nansum(data, axis=1, keepdims=True)
    if np.all(total_per_sample == 0):
        print("All rows have zero total.")  # Check for zero totals

    data_new = np.divide(data, total_per_sample, where=(total_per_sample != 0))  # Normalize
    return np.nan_to_num(data_new)  # Replace NaNs with 0




# 2. rank the samples within each feature and do the H test on the features
def _square_of_sums(a, axis=0):
    s = np.sum(a, axis)
    if not np.isscalar(s):
        return s.astype(float) * s
    else:
        return float(s) * s
'''
def OTU_H_Score0(x,y,cutOff):#This function's output is the same with that from the library
    if not (len(x)==len(y)):
        raise ValueError('ERROR! Length of OTU and label have difference length')
        
    unique_groups = np.unique(y)
    samples = [x[np.array(y) == group] for group in unique_groups]
    num_groups = len(unique_groups)
    if num_groups < 2:
        raise ValueError("Need at least two groups in stats.kruskal()")
    
    n = np.asarray(list(map(len, samples)))# an array, each element is the number of samples in each group
    alldata = np.concatenate(samples)
    ranked = stats.rankdata(alldata)# choose alldata or x, the result are different
    #ranked[alldata < cutOff] = 1
    ties = stats.tiecorrect(ranked)# leave ties problem for now
    #if ties == 0:
    #    raise ValueError('All numbers are identical in kruskal')
        
    # Compute sum^2/n for each group and sum
    j = np.insert(np.cumsum(n), 0, 0)
    ssbn = 0
    for i in range(num_groups):
        ssbn += _square_of_sums(ranked[j[i]:j[i+1]]) / n[i]

    totaln = np.sum(n, dtype=float)
    h = 12.0 / (totaln * (totaln + 1)) * ssbn - 3 * (totaln + 1)
    #df = num_groups - 1
    h /= ties

    #chi2 = _SimpleChi2(df)
    #pvalue = _get_pvalue(h, chi2, alternative='greater', symmetric=False, xp=np)
    return h
'''
def OTU_H_Score(x,y,cutOff=0.01):#x is the relative OTU abundance for each sampele , y is the group for each sample
    #output is the Hstatistics
    x = np.array(x)  # Ensure x is a NumPy array
    x = x.copy()  # Create a copy of x to prevent modification
    if not (len(x)==len(y)):
        raise ValueError('ERROR! Length of OTU and label have difference length')
    x[x<cutOff] =0 # this will put  it at the lowest ranking
    unique_groups = np.unique(y)
    samples = [x[np.array(y) == group] for group in unique_groups]
    num_groups = len(unique_groups)# df = num_groups-1
    if num_groups < 2:
        raise ValueError("Need at least two groups in stats.kruskal()")
    
    n = np.asarray(list(map(len, samples)))# an array, each element is the number of samples in each group
    alldata = np.concatenate(samples)
    ranked = stats.rankdata(alldata)
   
    ties = stats.tiecorrect(ranked)#  a correction factor for ties in statistical tests
    if ties == 0:
        return 0  # Return 0 if all numbers are identical
        
    # Compute sum^2/n for each group and sum
    j = np.insert(np.cumsum(n), 0, 0)
    ssbn = 0
    for i in range(num_groups):
        ssbn += _square_of_sums(ranked[j[i]:j[i+1]]) / n[i]

    totaln = np.sum(n, dtype=float)
    h = 12.0 / (totaln * (totaln + 1)) * ssbn - 3 * (totaln + 1)
    #df = num_groups - 1
    h /= ties

    #p_cutoff = 0.1
    #h_cutoff = stats.chi2.ppf(1 - p_cutoff, num_groups-1)

    #chi2 = _SimpleChi2(df)
    #pvalue = _get_pvalue(h, chi2, alternative='greater', symmetric=False, xp=np)
    return h




def OTU_H_Score_arr(X,y,cutOff=0.01): #X is the relative abundance matrix(np.array), each row is a sample; y is the classification
    X_transpose=np.transpose(X)
    score_List=[]
    for OTU_across_sample in X_transpose:
        h_score=OTU_H_Score(OTU_across_sample,y,cutOff)
        score_List.append(h_score)
    return np.array(score_List)


# this is the function to use
def OTU_H_Score_fun(X,Y,cutOff=0.01):#X is the relative abundance matrix(np.array), each row is a sample; y is the classification results
    # if there is only one response, then y is a 1D array
    # if there is multiple response variable, then y is a 2D array, each column is one variable
    #X_transpose=np.transpose(X)
    Y = np.asarray(Y)
    if Y.ndim == 1:
        result = OTU_H_Score_arr(X,Y,cutOff)# a 1D array showing the H statistics for all the features
        return result
    elif Y.ndim ==2:# if
        Y_transpose=np.transpose(Y)
        H_list = []
        for yi in Y_transpose:
            h_one_class = OTU_H_Score_arr(X,yi,cutOff)
            H_list.append(h_one_class)
        H_combine = np.vstack(H_list)
        return H_combine
    else:
        return "Error: The input Y must be a 1D or 2D array."

   



# 3. get the number of features to keep based on significance
def indice_H_unisig(scorelist,y):# scorelist is the H statistics for featues, num_groups is the number of classes in y
    #return to the indices of significant features and the numer of featurs kept
    # selectedOTU_index, eps=FS.indice_H_sig(weights,y)
    unique_groups = len(np.unique(y))
    df = unique_groups-1
    p_cutoff = 0.1
    h_cutoff = stats.chi2.ppf(1 - p_cutoff, df)
    indices_above_cutoff = [index for index, value in enumerate(scorelist) if value > h_cutoff]
    # Sort the indices based on the corresponding values in descending order
    selected_indices = sorted(indices_above_cutoff, key=lambda x: scorelist[x], reverse=True)

    return selected_indices,len(selected_indices)



def indice_H_multisig(scorelist,Y,p_cutoff = 0.1):# scorelist is the H statistics for featues, num_groups is the number of classes in y
    #return to the indices of significant features and the numer of featurs kept
    # selectedOTU_index, eps=FS.indice_H_sig(weights,y)
    Y = np.asarray(Y)
    if scorelist.shape[0] != Y.shape[1]:
        raise ValueError("The number of rows in the score must match the columns of the response array.")
    
    unique_groups = [np.unique(Y[:, col]).size for col in range(Y.shape[1])]#a list
    df = [g - 1 for g in unique_groups]
    
    h_cutoff = np.array([stats.chi2.ppf(1 - p_cutoff, df_i) for df_i in df])# an array
    indices_above_cutoff = np.where(np.any(scorelist > h_cutoff[:, None], axis=0))[0]
    
    selected_indices = sorted(indices_above_cutoff, key=lambda col: np.sum(scorelist[:, col]), reverse=True)
    
    return selected_indices,len(selected_indices)



 
# this function will plot the H score
def plot_Hscore(scorelist,Y,p_cutoff = 0.1):
    
    Y = np.asarray(Y)
    if scorelist.shape[0] != Y.shape[1]:
        raise ValueError("The number of rows in the score must match the columns of the response array.")
    
    unique_groups = [np.unique(Y[:, col]).size for col in range(Y.shape[1])]#a list
    df = [g - 1 for g in unique_groups]
    
    h_cutoff = np.array([stats.chi2.ppf(1 - p_cutoff, df_i) for df_i in df])# an array
    indices_above_cutoff = np.where(np.any(scorelist > h_cutoff[:, None], axis=0))[0]
    
    selected_indices = sorted(indices_above_cutoff, key=lambda col: np.sum(scorelist[:, col]), reverse=True)
    
    return selected_indices,len(selected_indices)


"""
# 3. get the number of features to keep based on elbow point
# refer to function elbowPoint,plotWeightedIndex
def elbow_fun(scorelist,curveType="elbow"):# the function to get the elbow point and get the indices of the features whose score is higher than the elbow point score
    sorted_score_with_indices = sorted(enumerate(scorelist), key=lambda x: x[1],reverse=True)
    # Filtering to keep only the values greater than 0
    #filtered_sorted_score_with_indices = [(index, value) for index, value in sorted_score_with_indices if value > 0]

    sorted_scores = [item[1] for item in sorted_score_with_indices]
    sorted_indices = [item[0] for item in sorted_score_with_indices]#the smallest possible number is 0
    
    curveName="convex"
    if curveType!="elbow":
        curveName="concave"

    from kneed import KneeLocator
    kn = KneeLocator(range(1, len(sorted_scores)+1), sorted_scores, curve='convex', direction='decreasing',interp_method="polynomial")

    kn.plot_knee_normalized()
    kn.plot_knee()

    elbowpoint = kn.knee
    if elbowpoint is None:
        raise ValueError("No elbow point detected")
        
    elbow_indices = sorted_indices[:elbowpoint]

    return elbowpoint,elbow_indices
"""
def plotWeightedIndex(weights,threshold=0):
    plt.figure(dpi=300)
    plt.plot(sorted(weights,reverse=True),linestyle=":")
    plt.ylabel("Weighted Info Index")
    plt.xlabel("Ranked OTUs")
    #plt.axis("square")

    if threshold!=0:
        plt.axvline(x=threshold, color='green', linestyle='dashed', label=f'Cutoff Point x= {threshold}', lw=1)
    plt.legend()
    plt.show()






"""
# 4. get the features value as an array based on the result of 3.------------------This part is not finished
def H_elbow_indice(h_scorelist,Num):
    score_array = np.array(h_scorelist)
    sorted_indices = np.argsort(score_array)[::-1]
    # Select the top x indices
    top_indices = sorted_indices[:Num]
    return top_indices

def Xarray_indice(X,h_scorelist):
    X = ap.array(X)
    if not (np.shape(X)[1]==len(h_scorelist)):
        raise ValueError('ERROR! Length of matrix and score list have difference length')
    selectedOTU_index = H_elbow_indice(h_scorelist)
    return X[:,selectedOTU_index]
""" 

    



    


def plotPresenseRatio(X,label,featurenames,posLabel,posText="",negText="",thresholdPercent=0.90,abundanceCutoff=0.01,entries=15):
    import matplotlib as mpl
    mpl.rcParams['figure.dpi'] = 300

    presenceCntPos = []
    presenceCntNeg = []
    
    X_relative = relative_abundance(X)
    
    X_relative = X_relative.T
    if abundanceCutoff==0:
        flatten_list = list(chain.from_iterable(X_relative))
        flatten_list_sorted=sorted(flatten_list)
        abundanceCutoff=flatten_list[int(len(flatten_list_sorted)*float(threshold))]

    if posText=="" or negText=="":
        posText=posLabel
        negText="Not "+posLabel

    for k in range(len(X_relative)):## for each OTU
        OTUs = X_relative[k]## the samples for this OTU
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

    axes[0].set_xlim(0,1.2)
    axes[1].set_xlim(0,1.2)
    axes[0].invert_xaxis()# Invert the x-axis of the first subplot

    axes[0].set(yticks=y, yticklabels=[])
    for yloc, selectedASVs in zip(y, featurenames):
        axes[0].annotate(selectedASVs, (0.5, yloc), xycoords=('figure fraction', 'data'),
                         ha='center', va='center', fontsize=9)
    fig.tight_layout(pad=2.0)
    plt.show()



def plotAvarageAbundance(X,label,featurenames,posLabel,posText="",negText="",thresholdPercent=0.90,abundanceCutoff=0.01,entries=15):
    import matplotlib as mpl
    mpl.rcParams['figure.dpi'] = 300

    presenceCntPos = []
    presenceCntNeg = []
    
    X_relative = FS.relative_abundance(X)
    
    X_relative = X_relative.T
    if abundanceCutoff==0:
        flatten_list = list(chain.from_iterable(X_relative))
        flatten_list_sorted=sorted(flatten_list)
        abundanceCutoff=flatten_list[int(len(flatten_list_sorted)*float(threshold))]

    if posText=="" or negText=="":
        posText=posLabel
        negText="Not "+posLabel

    for k in range(len(X_relative)):## for each OTU
        OTUs = X_relative[k]## the samples for this OTU
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
    
    AvarageAbundanceDiffPos=[float(x)/all_pos_label_cnt for x in presenceCntPos]# each element is for each OTU; shows the ratio of abundanced pos samples over all pos sample 
    AvarageAbundanceDiffNeg=[float(x)/all_neg_label_cnt for x in presenceCntNeg]

    import matplotlib.pyplot as plt
    y = range(entries)
    fig, axes = plt.subplots(ncols=2, sharey=True)
    axes[0].barh(y, presenceRatioPos, align='center', color='#ff7f00')
    axes[1].barh(y, presenceRatioNeg, align='center', color='#377eb8')
    axes[0].set_xlabel("Presence Ratio in "+posText)
    axes[1].set_xlabel("Presences Ratio "+negText)

    axes[0].set_xlim(0,1.2)
    axes[1].set_xlim(0,1.2)
    axes[0].invert_xaxis()# Invert the x-axis of the first subplot

    axes[0].set(yticks=y, yticklabels=[])
    for yloc, selectedASVs in zip(y, featurenames):
        axes[0].annotate(selectedASVs, (0.5, yloc), xycoords=('figure fraction', 'data'),
                         ha='center', va='center', fontsize=9)
    fig.tight_layout(pad=2.0)
    plt.show()





# outlier
def Outlier_array(arr_input):
# Calculate Q1, Q3, and IQR
    Q1 = np.percentile(arr_input, 25)
    Q3 = np.percentile(arr_input, 75)
    IQR = Q3 - Q1
    
    # Define outlier bounds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Get the indices of outliers
    outlier_indices = np.where((arr_input < lower_bound) | (arr_input > upper_bound))[0]
    
    # Print the result
    return arr_input[outlier_indices],outlier_indices

    
