import numpy as np

def listToRanks(lst):
    return [sorted(lst).index(x) for x in lst]

def normalizingMatrixToRanks(data,cutOff=0.01):
    data = np.array(data)
    for i in range(np.shape(data)[0]):
        for j in range(np.shape(data)[1]):
            if data[i][j]>=cutOff:
                continue
            else:
                data[i][j]=0
    for i in range(np.shape(data)[0]):
        data[i]=listToRanks(data[i])
    return data

