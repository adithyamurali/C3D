import matlab.engine as mateng
from time import time
import IPython
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn import (manifold, datasets, decomposition, ensemble, lda, random_projection, preprocessing)
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
X_scaled = preprocessing.scale(X)
X_tsne = tsne.fit_transform(X)

IPython.embed()