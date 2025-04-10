
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2,f_classif
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('./Code')
import loadData
import pandas as pd
from scipy import stats
import seaborn as sns
import scikit_posthocs as sp
import itertools

# new FS pipeline-------------

# 1. Relative abundance matrix
def relative_abundance(data,cutOff=0.01):# if the input is the original abundance matrix, will convert it into relative abundance matrix; for missing values will return to 0
    # Convert input to a numpy array
    data = np.array(data) 
    # Iterate over each row (sample) in the array
    total_per_sample = np.nansum(data, axis=1, keepdims=True)
    if np.all(total_per_sample == 0):
        print("All rows have zero total.")  # Check for zero totals

    data_new = np.divide(data, total_per_sample, where=(total_per_sample != 0))  # Normalize
    data_new[data_new<cutOff] =0
    return np.nan_to_num(data_new)  



# 2. rank the samples within each feature and do the H test on the features
def _square_of_sums(a, axis=0):
    s = np.sum(a, axis)
    if not np.isscalar(s):
        return s.astype(float) * s
    else:
        return float(s) * s


def OTU_H_Score(x,y):#x is the relative OTU abundance for each sampele , y is the group for each sample
    #output is the Hstatistics
    x = np.array(x)  # Ensure x is a NumPy array
    x = x.copy()  # Create a copy of x to prevent modification
    if not (len(x)==len(y)):
        raise ValueError('ERROR! Length of OTU and label have difference length')
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


def OTU_H_Score_arr(X,y): #X is the relative abundance matrix(np.array), each row is a sample; y is the classification
    X_transpose=np.transpose(X)
    score_List=[]
    for OTU_across_sample in X_transpose:
        h_score=OTU_H_Score(OTU_across_sample,y)
        score_List.append(h_score)
    return np.array(score_List)


# this is the function to use for multiple lables
def OTU_H_Score_fun(X,Y):#X is the relative abundance matrix(np.array), each row is a sample; y is the classification results
    # if there is only one response, then y is a 1D array
    # if there is multiple response variable, then y is a 2D array, each column is one variable
    #X_transpose=np.transpose(X)
    Y = np.asarray(Y)
    if Y.ndim == 1:
        result = OTU_H_Score_arr(X,Y)# a 1D array showing the H statistics for all the features
        return result
    elif Y.ndim ==2:# if
        Y_transpose=np.transpose(Y)
        H_list = []
        for yi in Y_transpose:
            h_one_class = OTU_H_Score_arr(X,yi)
            H_list.append(h_one_class)
        H_combine = np.vstack(H_list)
        return H_combine
    else:
        return "Error: The input Y must be a 1D or 2D array."





# calculete dunn_tests
def calculate_dunn_tests_pair(
    X_df,y, p_adjust_method='bonferroni',p_threshold=0.05
):
    results_dunn = {}
    y_series = pd.Series(y, name='Group')
    summary_rows = []

    for column in X_df.columns:
        df_dunntest = X_df[[column]].copy()
        df_dunntest['Group'] = y_series.values

        dunn_results = sp.posthoc_dunn(
            df_dunntest,
            val_col=column,
            group_col='Group',
            p_adjust=p_adjust_method
        )

        #print(f"\n==== {column} ====")
        groups = dunn_results.columns.tolist()
        for g1, g2 in itertools.combinations(groups, 2):
            p_val = dunn_results.loc[g2, g1] if g1 in dunn_results.columns and g2 in dunn_results.index else None
            if pd.notna(p_val):
                #print(f"{g1} vs {g2}: p = {p_val:.4f}")
                if p_val < p_threshold:
                    medians = df_dunntest.groupby('Group')[column].median()
                    m1 = medians.loc[g1]
                    m2 = medians.loc[g2]
                    if m1 > m2:
                        direction = f"↑ {g1}"
                    elif m2 > m1:
                        direction = f"↑ {g2}"
                    else:
                        direction = "=" # this is based on median so can be equal
                    
                    summary_rows.append({
                        'Feature': column,
                        'Group1': g1,
                        'Group2': g2,
                        'P-value': p_val,
                        'EffectDirection': direction
                    })

        results_dunn[column] = dunn_results
        
    summary_df = pd.DataFrame(summary_rows)
    return results_dunn, summary_df



# 3. get the number of features to keep based on significance
def indice_H_unisig(scorelist,y):# scorelist is the H statistics for featues, num_groups is the number of classes in y
    #return to the indices of significant features and the numer of featurs kept
    # selectedOTU_index, eps=FS.indice_H_sig(weights,y)
    unique_groups = len(np.unique(y))
    df = unique_groups-1
    p_cutoff = 0.05
    h_cutoff = stats.chi2.ppf(1 - p_cutoff, df)
    indices_above_cutoff = [index for index, value in enumerate(scorelist) if value > h_cutoff]
    # Sort the indices based on the corresponding values in descending order
    selected_indices = sorted(indices_above_cutoff, key=lambda x: scorelist[x], reverse=True)

    return selected_indices,len(selected_indices)

def SelectOTU_fun(scorelist,y,p_cutoff=0.05,plot=True):# same with the function above, but add plot output
    unique_groups = len(np.unique(y))
    df = unique_groups-1
    p_cutoff = p_cutoff
    h_cutoff = stats.chi2.ppf(1 - p_cutoff, df)
    indices_above_cutoff = [index for index, value in enumerate(scorelist) if value > h_cutoff]
    # Sort the indices based on the corresponding values in descending order
    selected_indices = sorted(indices_above_cutoff, key=lambda x: scorelist[x], reverse=True)

    if plot:
        # Sorting the array in descending order and saving the indices
        sorted_indices = np.argsort(scorelist)[::-1]  # Get indices for sorted order (descending)
        sorted_array = scorelist[sorted_indices]      # Sort the array using the indices
        
        # Plotting the sorted array
        plt.figure(figsize=(8, 5))
        plt.scatter(np.arange(len(sorted_array)), sorted_array, color='blue', s=5)
        #plt.axhline(y=stats.chi2.ppf(1 - 0.1, 1), color='green', linestyle='--', linewidth=1, label='significance cutoff')
        plt.axvline(x=len(selected_indices), color='green', linestyle='-', linewidth=1, label=f"number of selected OTUs ={len(selected_indices)}")
        # Adding labels and title
        plt.xlabel("Number of OTUs", fontsize=12)
        plt.ylabel("H statistics", fontsize=12)
        #plt.title("Dot Plot of Sorted Array (Descending)", fontsize=14)
        #plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()

    return np.array(selected_indices)


################# This is a combined function ######################################
def SelectMicro_fun(df,y,p_cutoff=0.05,plot=True):
    """
    combine calculating H and feature selection in one function
    return selected array (data), selected column names, selected indices and all the H statistics with the original indices, plot H if needed
    input: relative abundance dataframe (to get the column names), target variable
    """
    if y.ndim != 1:
        raise ValueError("Response variable must be 1D array")
    colnames = df.columns
    x = df.to_numpy()
    scorelist = OTU_H_Score_arr(x,y)

    selected_indices = SelectOTU_fun(scorelist,y,p_cutoff=p_cutoff,plot=plot)

    selected_data = x[:,selected_indices]
    selected_columnames = colnames[selected_indices]
    
    result = {"selected_df": pd.DataFrame(selected_data, columns=selected_columnames),
              #"selected_columnames": selected_columnames,
              "selected_indices": selected_indices,
              #"relative_abundance_data": x,
              "H_score": scorelist}
    
    return result







































#################
def indice_H_multisig(scorelist,Y,p_cutoff = 0.05):# scorelist is the H statistics for featues, num_groups is the number of classes in y
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


################# This is a combined function ######################################
def SelectMicro_multi_fun(df,Y, p_cutoff=0.05,plot=True):
    """
    combine calculating H and feature selection in one function
    return selected array (data), selected column names, selected indices and all the H statistics with the original indices, plot H if needed
    input: dataframe (to get the column names), target variable
    """
    Y = np.asarray(Y)

    unique_groups = [np.unique(Y[:, col]).size for col in range(Y.shape[1])]#a list
    dof = [g - 1 for g in unique_groups]
    
    h_cutoff = np.array([stats.chi2.ppf(1 - p_cutoff, df_i) for df_i in dof])# an array
    
    if Y.ndim != 2:
        raise ValueError("Response variable must be 2D array")

    colnames = df.columns
    weights_list = []
    for i in range(Y.shape[1]):
        selectedresult=SelectMicro_fun(df,Y[:,i],plot=False)
        weights = selectedresult['H_score']
        weights_list.append(weights)
        
    H_combine = np.vstack(weights_list)
    x = selectedresult['relative_abundance_data']
    indices_above_cutoff = np.where(np.any(H_combine > h_cutoff[:, None], axis=0))[0]
    
    selected_indices = sorted(indices_above_cutoff, key=lambda col: np.sum(H_combine[:, col]), reverse=True)
    selected_data = x[:,selected_indices]
    selected_columnames = colnames[selected_indices]
    
    result = {"selected_data": selected_data,
              "selected_columnames": selected_columnames,
              "selected_indices": selected_indices,
              "relative_abundance_data": x,
              "H_score": H_combine}
    
    return result




"""
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


def plotPresenseRatio(X,label,featurenames,posLabel,posText="",negText="",thresholdPercent=0.90,abundanceCutoff=0.01):
     # input X is the abundance array, label is the index, featurenames is the column names of X
    if X.shape[0] != len(label):
        raise ValueError('ERROR! The number of samples in X must match the length of label')
    if X.shape[1] != len(featurenames):
        raise ValueError('ERROR! The number of features in X must match the length of featurenames.')
    
    # Continue with the rest of your function...
    print("Validation passed: Dimensions are correct.")

    if posText=="" or negText=="":
        posText=posLabel
        negText="Not "+posLabel
    
    all_pos_label_cnt=list(label).count(posLabel)
    all_neg_label_cnt=len(label)-all_pos_label_cnt
    print(f"{posText}= {all_pos_label_cnt}, {negText} = {all_neg_label_cnt}")# these 3  lines can use  value_count
   
    presenceCntPos = []
    presenceCntNeg = []
    
    X = X.T
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
       
    presenceRatioPos=[float(x)/all_pos_label_cnt for x in presenceCntPos]# each element is for each OTU; shows the ratio of abundanced pos samples over all pos sample 
    presenceRatioNeg=[float(x)/all_neg_label_cnt for x in presenceCntNeg]

    data_barh = {'OTU': featurenames, 'presence_pos': presenceRatioPos,'presence_neg': presenceRatioNeg}
    df_barh = pd.DataFrame(data_barh)
    print(df_barh)

    # Create the figure and axes
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    # Plot the first horizontal bar plot
    df_barh.plot.barh(x='OTU', y='presence_pos', color='skyblue', ax=axes[0])
    # Plot the second horizontal bar plot
    df_barh.plot.barh(x='OTU', y='presence_neg', color='salmon', ax=axes[1])

    left_1 =0.1
    width =0.3
    bottom_ =0.1
    height_ =0.8
    middle_ = 0.05
    axes[0].set_position([left_1,bottom_, width,height_])  # (left, bottom, width, height)
    axes[1].set_position([left_1+width+middle_+0.05, bottom_, width, height_])  # Adjust position of second plot

    axes[0].set_xlabel("Presence Ratio in "+posText)
    axes[1].set_xlabel("Presences Ratio in "+negText)   
    for ax in axes:
        ax.set_ylabel("")
        ax.invert_yaxis()
        ax.legend().remove()
        ax.set_xlim(0,1)

    # Remove y-ticks from the first plot
    axes[0].set_yticks([])
    axes[0].invert_xaxis()# Invert the x-axis of the first subplot
    #axes[0].set_yticklabels([""] * len(featurenames))
    # Move the labels towards the left by adjusting labelpad (negative value moves to the left)
    # Move the labels left (negative) or right (positive) using labelpad for x-axis
    """
    for tick in axes[1].get_yticklabels():
        tick.set_horizontalalignment('right')  # Align labels to the right
        tick.set_position((tick.get_position()[0], tick.get_position()[1]))  # Move labels left by 0.05
    """
    plt.tight_layout()
    plt.show()




def plotAvarageAbundance(X,label,featurenames,posLabel,posText="",negText="",thresholdPercent=0.90,abundanceCutoff=0.01,entries=15):
    import matplotlib as mpl
    mpl.rcParams['figure.dpi'] = 300

    presenceCntPos = []
    presenceCntNeg = []
    
    X = X.T
    if abundanceCutoff==0:
        flatten_list = list(chain.from_iterable(X_relative))
        flatten_list_sorted=sorted(flatten_list)
        abundanceCutoff=flatten_list[int(len(flatten_list_sorted)*float(threshold))]

    if posText=="" or negText=="":
        posText=posLabel
        negText="Not "+posLabel

    unique_label = np.unique(label)

    for k in range(len(X)):## for each OTU
        label_mean = {cat:X[k][label == cat].mean() for cat in unique_label}

        presenceCntPos.append(pos)# len= # of samples; each value is the number of OTUs that exceed the abundanceCutoff for Pos/Neg
        presenceCntNeg.append(neg)
           
    AvarageAbundanceDiffPos=[float(x)/all_pos_label_cnt for x in presenceCntPos]# each element is for each OTU; shows the ratio of abundanced pos samples over all pos sample 
    AvarageAbundanceDiffNeg=[float(x)/all_neg_label_cnt for x in presenceCntNeg]

    import matplotlib.pyplot as plt
    y = range(entries)
    fig, axes = plt.subplots(ncols=2, sharey=True)
    axes[0].barh(y, presenceRatioPos, align='center', color='#ff7f00')
    axes[1].barh(y, presenceRatioNeg, align='center', color='#377eb8')
    axes[0].set_xlabel("Presence Ratio in "+posText)
    axes[1].set_xlabel("Presences Ratio in "+negText)
    # Annotate each bar in the first subplot
    for i, bar in enumerate(bars_pos):
        axes[0].text(presenceRatioPos[i], bar.get_y() + bar.get_height() / 2, f'{presenceRatioPos[i]:.2f}', va='center', ha='right',fontsize=6)

    # Annotate each bar in the second subplot
    for i, bar in enumerate(bars_neg):
        axes[1].text(presenceRatioNeg[i], bar.get_y() + bar.get_height() / 2, f'{presenceRatioNeg[i]:.2f}', va='center', ha='left',fontsize=6)
        
    axes[0].set_xlim(0,1.2)
    axes[1].set_xlim(0,1.2)
    axes[0].invert_xaxis()# Invert the x-axis of the first subplot

    axes[0].set(yticks=y, yticklabels=[])
    for yloc, selectedASVs in zip(y, featurenames):
        axes[0].annotate(selectedASVs, (0.5, yloc), xycoords=('figure fraction', 'data'),
                         ha='center', va='center', fontsize=6)
    fig.tight_layout(pad=2.0)
    plt.show()



def OTUviolin(X,label,featurenames,y_max=None,single=True,title = 'Violin Plot for OTUs',plot_per_row=5):
    """
    input X is the abundance array, 
    label is the index, 
    featurenames is the column names of X
    singel: if what to show each feature's violin plot separately, change it to False
    """
    df = pd.DataFrame(X, columns=featurenames)
    df['Y'] = label

    if single:
        # Melt the DataFrame for seaborn's violinplot
        df_melted = df.melt(id_vars='Y', value_vars=featurenames,
                            var_name='Feature', value_name='RelativeAbundance')
        # Plot the violin plot for all features in one plot
        plt.figure(figsize=(16, 8))
        # KDE helps in visualizing the distribution of data by smoothing the data points
        sns.violinplot(x='Feature', y='RelativeAbundance', hue='Y', data=df_melted, split=True,inner="quart", legend=False,
                       density_norm='width',  # Make the violins use the full width
                       width=0.8,      # Adjust the width of the violins for clarity
                       dodge=True)     # Separate violins for each group
        # Rotate the x-axis labels for better visibility
        if y_max is not None:
            y_lim = y_max
        else:
            y_lim = df_melted['RelativeAbundance'].max()
        #plt.ylim(0, ylim].max())  # Adjust the y-axis limits to avoid negative values
        plt.ylim(0, y_lim)  # Adjust the y-axis limits to avoid negative values
        plt.xlabel('')
        plt.xticks(rotation=45)  # You can change 45 to any other angle if you want a different rotation
        plt.title(title)
        plt.tight_layout()
        plt.show()
    else:
        n_features = X.shape[1]
        # Calculate the number of rows needed (with up to 4 features per row)
        n_cols = plot_per_row  # You want 4 plots per row
        n_rows = math.ceil(n_features / n_cols)  # Calculate required rows
        
        # Plot a violin plot for each feature
        plt.figure(figsize=(15, 5 * n_rows))  # Adjust the height of the plot for multiple rows
        
        for i in range(n_features):
            plt.subplot(n_rows, n_cols, i+1)  # Subplot for each feature
            sns.violinplot(x='Y', y=df.iloc[:, i], data=df, inner="quart", hue='Y', legend=False)
            plt.title(f'{featurenames[i]}')
            plt.ylabel('')
    
        plt.tight_layout()
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

    
