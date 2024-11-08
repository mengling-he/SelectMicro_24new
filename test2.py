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

