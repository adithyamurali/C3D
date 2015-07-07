import matlab.engine as mateng
from time import time
import IPython
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn import (manifold, datasets, decomposition, ensemble, lda, random_projection, preprocessing)
from scipy import linalg
import argparse
PATH_TO_DATA = 'data/8fps_nooverlap_conv5b/'


def parse():
	# Reading fc8-1 features
	eng = mateng.start_matlab()
	# [s, X] = eng.read_binary_blob(PATH_TO_DATA + '1.fc8-1', nargout = 2)
	# for i in range(2, 175):
	# 	[s, data] = eng.read_binary_blob(PATH_TO_DATA + str(i) + '.fc8-1', nargout = 2)
	# 	data = np.array(data)
	# 	X = np.concatenate((X, data), axis = 0)

	[s, X] = eng.read_binary_blob(PATH_TO_DATA + '1.conv5b', nargout = 2)
	i = 17;
	while i <= 1489:
		[s, data] = eng.read_binary_blob(PATH_TO_DATA + str(i) + '.conv5b', nargout = 2)
		data = np.array(data)
		X = np.concatenate((X, data), axis = 0)
		i += 16
	return X
#----------------------------------------------------------------------
# Scale and visualize the embedding vectors
def plot_embedding(X, figure_name, title=None):
	x_min, x_max = np.min(X, 0), np.max(X, 0)
	X = (X - x_min) / (x_max - x_min)
	plt.figure()
	for i in range(X.shape[0]):
		plt.text(X[i, 0], X[i, 1], str(i*16 + 1), color=plt.cm.Set1((i*16 + 1)/ 100), fontdict={'weight': 'bold', 'size': 9})
	plt.xticks([]), plt.yticks([])
	if title is not None:
		plt.title(title)
	plt.savefig('plots/'+figure_name)

def pca(figure_name, X):
	print("Computing PCA embedding")
	X_preprocessed = X.copy()
	X_preprocessed = preprocessing.scale(X_preprocessed)
	X_preprocessed = preprocessing.normalize(X_preprocessed, norm = 'l2')
	X_pca = decomposition.TruncatedSVD(n_components=2).fit_transform(X_preprocessed)
	plot_embedding(X_pca, figure_name,
	               "Principal Components projection of C3D conv5b visual features")

def tsne(figure_name, X):
	#----------------------------------------------------------------------
	# t-SNE embedding of the digits dataset
	print("Computing t-SNE embedding")
	tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
	X_tsne = tsne.fit_transform(X)

	plot_embedding(X_tsne, figure_name,
	               "t-SNE embedding of C3D conv5b visual features")

def plot_all():
	tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
	#----------------------------------------------------------------------
	# Pre-processing
	X_scaled = preprocessing.scale(X) #zero mean, unit variance
	X_tsne_scaled = tsne.fit_transform(X_scaled)

	#normalize the data (scaling individual samples to have unit norm)
	X_normalized = preprocessing.normalize(X, norm='l2')
	X_tsne_norm = tsne.fit_transform(X_normalized)


	#whiten the data 
	scaler = preprocessing.StandardScaler(with_std=False).fit(X)
	X_centered = scaler.transform(X)

	# IPython.embed()

	# U, s, Vh = linalg.svd(X_centered)
	U, s, Vh = linalg.svd(X_centered, full_matrices=False)
	eps = 1e-5
	invS = np.diag (np.reciprocal(np.sqrt(s+eps)))

	#PCA_whiten
	X_pca = np.dot(invS, np.dot(U.T, X_centered))
	X_tsne_pca = tsne.fit_transform(X_pca)

	#whiten the data (ZCA)
	X_zca = np.dot(U, X_pca)
	X_tsne_zca = tsne.fit_transform(X_zca)

	return X_tsne_scaled, X_tsne_norm, X_tsne_pca, X_tsne_zca

if __name__ == "__main__":
	X = parse()
	# parser = argparse.ArgumentParser()
	# parser.add_argument("plot", help = "Choose from pca, tsne, etc.")
	# parser.add_argument("figure_name", help = "Figure name to be saved")
	# args = parser.parse_args()
	# plotter_func = locals()[args.plot]
	# plotter_func(args.figure_name, X)

	X_tsne_scaled, X_tsne_norm, X_tsne_pca, X_tsne_zca  = plot_all()
	plotNames = ["X_tsne_scaled", "X_tsne_norm", "X_tsne_pca", "X_tsne_zca"]

	for plotName in plotNames:
		plot_embedding (X=eval(plotName), figure_name='conv5b_'+plotName, title=plotName+'_conv5b')
