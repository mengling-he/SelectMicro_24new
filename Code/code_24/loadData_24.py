
#This file loads different files and convert them into trainning/test matrix and labels (X, y), each function loads a different intput format
import numpy as np
import pandas as pd
def loadDisease_marker_Cirrhosis(inputF="../DiseasePrediction/marker_Cirrhosis.txt"):
    df=pd.read_csv(inputF,delimiter="\t", index_col=0, header=None).T
    lastLabelIndex=df.columns.get_loc("group")
    df_data=df.iloc[:,lastLabelIndex+1:].copy()
    label=list(df["disease"])
    label_binary=[1 if x=="cirrhosis" else 0 for x in label]
    return np.asarray(df_data),label_binary


def loadDisease_marker_Colorectal(inputF="../DiseasePrediction/marker_Colorectal.txt"):
    df=pd.read_csv(inputF,delimiter="\t", index_col=0, header=None).T
    lastLabelIndex=df.columns.get_loc("group")
    df_data=df.iloc[:,lastLabelIndex+1:].copy()
    label=list(df["disease"])
    label_binary=[0 if x=="n" else 1 for x in label]
    return np.asarray(df_data),label_binary

def loadDisease_marker_IBD(inputF="../DiseasePrediction/marker_IBD.txt"):
    df=pd.read_csv(inputF,delimiter="\t", index_col=0, header=None).T
    lastLabelIndex=df.columns.get_loc("group")
    df_data=df.iloc[:,lastLabelIndex+1:].copy()
    label=list(df["disease"])
    label_binary=[0 if x=="n" else 1 for x in label]
    return np.asarray(df_data),label_binary

def loadDisease_marker_Obesity(inputF="../DiseasePrediction/marker_Obesity.txt"):
    df=pd.read_csv(inputF,delimiter="\t", index_col=0, header=None).T
    lastLabelIndex=df.columns.get_loc("group")
    df_data=df.iloc[:,lastLabelIndex+1:].copy()
    label=list(df["disease"])
    label_binary=[1 if x=="obesity" else 0 for x in label]
    return np.asarray(df_data),label_binary

def loadDisease_marker_T2D(inputF="../DiseasePrediction/marker_T2D.txt"):
    df=pd.read_csv(inputF,delimiter="\t", index_col=0, header=None).T
    lastLabelIndex=df.columns.get_loc("group")
    df_data=df.iloc[:,lastLabelIndex+1:].copy()
    label=list(df["disease"])
    label_binary=[0 if x=="n" else 1 for x in label]
    return np.asarray(df_data),label_binary


def loadDisease_marker_WT2D(inputF="../DiseasePrediction/marker_WT2D.txt"):
    df=pd.read_csv(inputF,delimiter="\t", index_col=0, header=None).T
    lastLabelIndex=df.columns.get_loc("group")
    df_data=df.iloc[:,lastLabelIndex+1:].copy()
    label=list(df["disease"])
    label_binary=[0 if x=="n" else 1 for x in label]
    return np.asarray(df_data),label_binary

def loadSoilData(inputF="../Data/merge_soil.csv"):
    df=pd.read_csv(inputF,delimiter=",")
    lastLabelIndex = df.columns.get_loc("Duration")
    df_data = df.iloc[:, lastLabelIndex + 1:].copy()
    Burn_label=df["Burn"]
    UN_label=df["UN"]
    Duration_label=df["Duration"]
    return np.asarray(df_data),Burn_label,UN_label,Duration_label,list(df.columns[lastLabelIndex + 1:]),df

def loadColoradoGradient(inputF="../Data/ColoradoData_Gradient.csv"):
    df = pd.read_csv(inputF, delimiter=",")
    lastLabelIndex = df.columns.get_loc("Gradient")
    df_data = df.iloc[:, lastLabelIndex + 1:].copy()
    factorSet=set(df["Gradient"])
    df_dummies=(pd.get_dummies(df["Gradient"]))
    df_concat = pd.concat([df_dummies,df], axis=1).reindex(df.index)
    df_labels = df_dummies.iloc[:,0:lastLabelIndex + 6].copy()
    labelList = []
    for col in df_labels:
        labelList.append(list(df_labels[col]))

    return np.asarray(df_data), labelList, df_labels.columns


def loadEarthMicrobiome(inputF="../Data/biome_feature.csv"):
    df = pd.read_csv(inputF, delimiter=",")
    lastLabelIndex = df.columns.get_loc("biome")
    df_data = df.iloc[:, 2:lastLabelIndex].copy()
    label1=np.asarray(df["biome"])
    label2=np.asarray(df["feature"])
    sampleNames=df["OTU_Sample"]
    return np.asarray(df_data),np.asarray(label1),np.asarray(label2),np.asarray(sampleNames)




