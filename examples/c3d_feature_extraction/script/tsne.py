import matlab.engine as mateng
from time import time
import IPython
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn import (manifold, datasets, decomposition, ensemble, lda, random_projection, preprocessing)
from scipy import linalg

PATH_TO_DATA = 'data/'


# Reading fc8-1 features
eng = mateng.start_matlab()
[s, X] = eng.read_binary_blob(PATH_TO_DATA + '1.fc8-1', nargout = 2)
for i in range(2, 100):
	[s, data] = eng.read_binary_blob(PATH_TO_DATA + str(i) + '.fc8-1', nargout = 2)
	data = np.array(data)
	X = np.concatenate((X, data), axis = 0)

#----------------------------------------------------------------------
# t-SNE embedding of the digits dataset
print("Computing t-SNE embedding")
tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
t0 = time()
X_tsne = tsne.fit_transform(X)

#----------------------------------------------------------------------
# Pre-processing
X_scaled = preprocessing.scale(X) #zero mean, unit variance
X_tsne_scaled = tsne.fit_transform(X_scaled)

#normalize the data (scaling individual samples to have unit norm)
X_normalized = preprocessing.normalize(X, norm='l2')
X_tsne_norm = tsne.fit_transform(X_normalized)

#whiten the data 
scaler = preprocessing.StandardScaler( with_std=False)
X_centered = scaler.fit(X)
IPython.embed()
U, s, Vh = linalg.svd(X_centered)
eps = 1
invS = np.diag (np.reciprocal(np.sqrt(s+eps)))

#PCA_whiten
X_pca = np.dot(invS, np.dot(U.T, X_centered))
X_tsne_pca = tsne.fit_transform(X_pca)

#whiten the data (ZCA)
X_zca = np.dot(U, X_pca)
X_tsne_zca = tsne.fit_transform(X_zca)


IPython.embed()

# for x in plots