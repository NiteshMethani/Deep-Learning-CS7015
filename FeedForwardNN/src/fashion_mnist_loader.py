from sklearn.decomposition import PCA
import numpy as np
import pandas as pd

def hotEncoding(y, n_classes):
    arr = np.zeros((y.shape[0],n_classes))
    for i in range(y.shape[0]):
        e = [0.0]*n_classes
        e[y[i]] = 1.0
        arr[i] = e
    return arr


def load_data(file, data, normalise = True, mean=0, std=1):
    print('Loading Data...')
    n_classes = 10
    
    df = pd.read_csv(file)
    
    df = df.iloc[np.random.permutation(len(df))]
    
    img_id = df['id']
    
    X = df[df.columns[1:-1]]
    
    #print(X.describe())
    
    if data == 'train':
        mean = X.mean(axis=0)
        std  = X.std(axis = 0)
    
    if normalise == True:
        X = normaliseData(X, mean, std)
    
    X = X.values
    
    #n_components = 550
    #pca = PCA(n_components)
    #X = pca.fit_transform(X)
    
    Y = df['label'].values
    
    
    print('Data Loaded Successfully...')
    
    return (X, hotEncoding(Y, n_classes), mean, std)

def normaliseData(X, mean, std):
    X = X - mean
    X = X/std
    return X


def load_test_data(file, normalise = True, mean=0, std=1):
    print('Loading Data...')
    
    df = pd.read_csv(file)
    
    df = df.iloc[np.random.permutation(len(df))]
    
    img_id = df['id']
    
    X = df[df.columns[1:]]
    
    if normalise == True:
        X = normaliseData(X, mean, std)
    
    X = X.values
    
    print('Data Loaded Successfully...')
    
    return (img_id, X)